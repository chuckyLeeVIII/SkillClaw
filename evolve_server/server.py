"""
Core orchestrator for the current session-level evolve_server pipeline.

Active flow:
1. Drain pending sessions from shared storage.
2. Summarize sessions and extract metadata.
3. Optionally backfill a session-level score with session_judge.
4. Aggregate sessions by referenced skill.
5. Evolve existing-skill groups or create new skills from no-skill groups.
6. Upload skills, persist registry state, and ack processed sessions.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Optional

from skillclaw.object_store import build_object_store

from .aggregation import aggregate_sessions_by_skill
from .config import EvolveServerConfig
from .constants import DecisionAction, NO_SKILL_KEY, SLUG_RE
from .execution import (
    create_skill_from_sessions,
    evolve_skill_from_sessions,
    execute_merge,
    set_evolve_debug_dir,
)
from .llm_client import AsyncLLMClient
from .oss_helpers import (
    delete_session_keys,
    fetch_skill_content,
    list_session_keys,
    load_manifest,
    read_json_object,
    save_manifest,
)
from .session_judge import judge_sessions_parallel
from .skill_registry import SkillIDRegistry
from .summarizer import set_summarizer_debug_dir, summarize_sessions_parallel
from .utils import build_skill_md, parse_skill_content

logger = logging.getLogger(__name__)


class EvolveServer:
    """Session-level evolve server backed by shared object storage."""

    def __init__(
        self,
        config: EvolveServerConfig,
        *,
        mock: bool = False,
        mock_root: str | None = None,
    ) -> None:
        self.config = config
        self._mock = mock
        self._bucket = self._build_bucket(mock=mock, mock_root=mock_root)
        self._prefix = f"{config.group_id}/"
        self._llm = AsyncLLMClient(
            api_key=config.llm_api_key,
            base_url=config.llm_base_url,
            model=config.llm_model,
            max_tokens=config.llm_max_tokens,
            temperature=config.llm_temperature,
        )
        self._id_registry = SkillIDRegistry()
        self._running = False

        set_evolve_debug_dir(config.debug_dump_dir)
        set_summarizer_debug_dir(config.debug_dump_dir)
        self._id_registry.load_from_oss(self._bucket, self._prefix)

    def _build_bucket(self, *, mock: bool, mock_root: str | None):
        if mock:
            from .mock_bucket import LocalBucket

            return LocalBucket(root=mock_root)
        return build_object_store(
            backend=self.config.storage_backend,
            endpoint=self.config.storage_endpoint,
            bucket=self.config.storage_bucket,
            access_key_id=self.config.storage_access_key_id,
            secret_access_key=self.config.storage_secret_access_key,
            region=self.config.storage_region,
            session_token=self.config.storage_session_token,
            local_root=self.config.local_root,
        )

    def _uses_local_storage(self) -> bool:
        backend = str(self.config.storage_backend or "").strip().lower()
        if backend == "local" or self._mock:
            return True
        bucket_type = type(self._bucket).__name__.lower()
        return "local" in bucket_type and bool(self.config.local_root)

    async def _call_storage(self, func, *args):
        if self._uses_local_storage():
            return func(*args)
        return await asyncio.to_thread(func, *args)

    def _append_history(self, record: dict) -> None:
        path = self.config.history_path
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning("[EvolveServer] history write failed: %s", exc)

    async def _drain_sessions(self) -> tuple[list[dict], list[str]]:
        keys = await self._call_storage(list_session_keys, self._bucket, self._prefix)
        sessions: list[dict] = []
        consumed_keys: list[str] = []
        for key in keys:
            session = await self._call_storage(read_json_object, self._bucket, key)
            if session:
                sessions.append(session)
                consumed_keys.append(key)
        logger.info(
            "[EvolveServer] drained %d session(s) from queue (%d keys found)",
            len(sessions),
            len(keys),
        )
        return sessions, consumed_keys

    def _load_remote_skills(self) -> dict[str, dict[str, Any]]:
        return load_manifest(self._bucket, self._prefix)

    def _fetch_skill(self, name: str) -> Optional[str]:
        return fetch_skill_content(self._bucket, self._prefix, name)

    def _upload_skill(self, skill: dict, action: str) -> None:
        name = skill.get("name", "")
        if not name:
            return

        skill_id = self._id_registry.get_or_create(name)
        md_content = build_skill_md(skill)
        object_key = f"{self._prefix}skills/{name}/SKILL.md"
        self._bucket.put_object(object_key, md_content.encode("utf-8"))

        content_sha = hashlib.sha256(md_content.encode("utf-8")).hexdigest()
        version = self._id_registry.record_update(name, content_sha, action=action)

        manifest = self._load_remote_skills()
        manifest[name] = {
            "name": name,
            "skill_id": skill_id,
            "version": version,
            "sha256": content_sha,
            "uploaded_by": "evolve_server",
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "description": skill.get("description", ""),
            "category": skill.get("category", "general"),
        }
        save_manifest(self._bucket, self._prefix, manifest)
        logger.info(
            "[EvolveServer] uploaded skill %s (id=%s, v%d) to %s",
            name,
            skill_id,
            version,
            object_key,
        )

    def _detect_conflict(self, name: str, incoming_skill: dict) -> bool:
        existing_sha = self._id_registry.get_content_sha(name)
        if not existing_sha:
            return False
        incoming_md = build_skill_md(incoming_skill)
        incoming_sha = hashlib.sha256(incoming_md.encode("utf-8")).hexdigest()
        return existing_sha != incoming_sha

    async def _resolve_and_upload(self, skill: dict, action_type: str) -> str:
        name = skill.get("name", "")
        has_conflict = await self._call_storage(self._detect_conflict, name, skill)
        if not has_conflict:
            await self._call_storage(self._upload_skill, skill, action_type)
            return action_type

        logger.info("[EvolveServer] conflict detected for '%s' - merging", name)
        existing_md = await self._call_storage(self._fetch_skill, name)
        if not existing_md:
            await self._call_storage(self._upload_skill, skill, action_type)
            return action_type

        existing_skill = parse_skill_content(name, existing_md)
        existing_skill["_version"] = self._id_registry.get_version(name)
        merged = await execute_merge(self._llm, existing_skill, skill)
        if merged and merged.get("name"):
            merged["name"] = name
            await self._call_storage(self._upload_skill, merged, "merge")
            return "merge"

        logger.warning("[EvolveServer] merge failed for '%s' - keeping incoming version", name)
        await self._call_storage(self._upload_skill, skill, action_type)
        return action_type

    def _empty_judge_summary(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.config.use_session_judge),
            "judged_sessions": 0,
            "scored_sessions": 0,
            "mean_score": None,
            "min_score": None,
            "max_score": None,
        }

    async def _run_session_judge(self, sessions: list[dict]) -> dict[str, Any]:
        summary = self._empty_judge_summary()
        if not self.config.use_session_judge or not sessions:
            return summary

        judged = await judge_sessions_parallel(self._llm, sessions)
        scores = [
            float(judge_scores["overall_score"])
            for session in sessions
            for judge_scores in [session.get("_judge_scores")]
            if isinstance(judge_scores, dict) and isinstance(judge_scores.get("overall_score"), (int, float))
        ]
        summary["judged_sessions"] = judged
        summary["scored_sessions"] = len(scores)
        if scores:
            summary["mean_score"] = round(sum(scores) / len(scores), 3)
            summary["min_score"] = round(min(scores), 3)
            summary["max_score"] = round(max(scores), 3)
        logger.info("[EvolveServer] judged %d sessions without benchmark scores", judged)
        return summary

    def _inherit_current_skill(
        self,
        evolved_skill: Optional[dict[str, Any]],
        current_skill: Optional[dict[str, Any]],
        *,
        overwrite_body: bool = False,
    ) -> None:
        if not evolved_skill or not current_skill:
            return
        if overwrite_body:
            evolved_skill["content"] = current_skill.get("content", "")
            evolved_skill["category"] = current_skill.get("category", "general")
        else:
            evolved_skill.setdefault("content", current_skill.get("content", ""))
            evolved_skill.setdefault("category", current_skill.get("category", "general"))
        evolved_skill.setdefault("extra_frontmatter", current_skill.get("extra_frontmatter") or {})

    async def _materialize_skill(
        self,
        evolved_skill: Optional[dict],
        action_type: str,
        session_ids: list[str],
        rationale: str,
        source: str,
        *,
        current_skill: Optional[dict[str, Any]] = None,
    ) -> Optional[dict]:
        if not evolved_skill or not evolved_skill.get("name"):
            return None

        if action_type == DecisionAction.IMPROVE and current_skill and current_skill.get("name"):
            name = current_skill["name"]
        else:
            name = self._sanitise_name(evolved_skill["name"])
        evolved_skill["name"] = name

        skill_id = self._id_registry.get_or_create(name)
        evolved_skill["skill_id"] = skill_id
        actual_action = await self._resolve_and_upload(evolved_skill, action_type)
        logger.info(
            "[EvolveServer] %s skill '%s' (id=%s, v%d)",
            actual_action,
            name,
            skill_id,
            self._id_registry.get_version(name),
        )
        return {
            "action": actual_action,
            "skill_name": name,
            "skill_id": skill_id,
            "version": self._id_registry.get_version(name),
            "session_ids": session_ids,
            "rationale": rationale,
            "source": source,
            "edit_summary": evolved_skill.get("edit_summary"),
        }

    async def _evolve_skill_group(
        self,
        skill_name: str,
        sessions: list[dict],
        existing_skill_names: list[str],
    ) -> Optional[dict]:
        current_md = await self._call_storage(self._fetch_skill, skill_name)
        current_skill = parse_skill_content(skill_name, current_md) if current_md else None

        result = await evolve_skill_from_sessions(
            self._llm,
            skill_name,
            sessions,
            current_skill,
            existing_skill_names,
        )
        if not result or result.get("action") == DecisionAction.SKIP:
            logger.info("[EvolveServer] skill '%s': LLM decided to skip", skill_name)
            return None

        action_type = result.get("action", DecisionAction.IMPROVE)
        evolved_skill = result.get("skill")
        if not evolved_skill:
            return None

        if action_type == DecisionAction.OPTIMIZE_DESC and current_skill:
            self._inherit_current_skill(evolved_skill, current_skill, overwrite_body=True)
        elif current_skill:
            self._inherit_current_skill(evolved_skill, current_skill)

        return await self._materialize_skill(
            evolved_skill,
            action_type,
            [session.get("session_id", "") for session in sessions],
            result.get("rationale", ""),
            "skill_group",
            current_skill=current_skill,
        )

    async def _handle_no_skill_sessions(
        self,
        sessions: list[dict],
        existing_skill_names: list[str],
    ) -> list[dict]:
        result = await create_skill_from_sessions(self._llm, sessions, existing_skill_names)
        if not result or result.get("action") == DecisionAction.SKIP:
            logger.info("[EvolveServer] no-skill sessions: LLM decided to skip")
            return []

        evolved_skill = result.get("skill")
        if not evolved_skill:
            return []

        record = await self._materialize_skill(
            evolved_skill,
            DecisionAction.CREATE,
            [session.get("session_id", "") for session in sessions],
            result.get("rationale", ""),
            "no_skill",
        )
        return [record] if record else []

    async def run_once(self) -> dict:
        logger.info("[EvolveServer] === starting evolution cycle ===")
        started_at = time.monotonic()

        sessions, session_keys = await self._drain_sessions()
        if not sessions:
            logger.info("[EvolveServer] queue empty - nothing to process")
            return {
                "sessions": 0,
                "skill_groups": 0,
                "no_skill_sessions": 0,
                "actions": 0,
                "skills_evolved": 0,
                "session_judge": self._empty_judge_summary(),
            }

        logger.info("[EvolveServer] summarizing %d sessions", len(sessions))
        await summarize_sessions_parallel(self._llm, sessions)
        judge_summary = await self._run_session_judge(sessions)

        grouped_sessions = aggregate_sessions_by_skill(sessions)
        no_skill_sessions = grouped_sessions.pop(NO_SKILL_KEY, [])
        skill_group_count = len(grouped_sessions)

        manifest = await self._call_storage(self._load_remote_skills)
        existing_skill_names = [item.get("name", "") for item in manifest.values()]

        evolution_records: list[dict] = []

        if grouped_sessions:
            logger.info("[EvolveServer] evolving %d skill group(s)", skill_group_count)
        for skill_name, skill_sessions in grouped_sessions.items():
            try:
                record = await self._evolve_skill_group(skill_name, skill_sessions, existing_skill_names)
            except Exception as exc:
                logger.error("[EvolveServer] skill '%s' evolve failed: %s", skill_name, exc)
                continue
            if record:
                evolution_records.append(record)

        if no_skill_sessions:
            logger.info("[EvolveServer] processing %d no-skill sessions", len(no_skill_sessions))
            evolution_records.extend(
                await self._handle_no_skill_sessions(no_skill_sessions, existing_skill_names)
            )

        await self._call_storage(self._id_registry.save_to_oss, self._bucket, self._prefix)
        await self._call_storage(delete_session_keys, self._bucket, session_keys)

        elapsed = round(time.monotonic() - started_at, 1)
        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": elapsed,
            "sessions": len(sessions),
            "skill_groups": skill_group_count,
            "no_skill_sessions": len(no_skill_sessions),
            "actions": len(evolution_records),
            "skills_evolved": len(evolution_records),
            "evolutions": evolution_records,
            "session_judge": judge_summary,
        }
        self._append_history(summary)
        logger.info(
            "[EvolveServer] === cycle done: %d sessions, %d skill groups, %d skills evolved in %.1fs ===",
            len(sessions),
            skill_group_count,
            len(evolution_records),
            elapsed,
        )
        return summary

    async def run_periodic(self) -> None:
        self._running = True
        logger.info("[EvolveServer] periodic mode: interval=%ds", self.config.interval_seconds)
        while self._running:
            try:
                await self.run_once()
            except Exception as exc:
                logger.error("[EvolveServer] cycle error: %s", exc, exc_info=True)
            await asyncio.sleep(self.config.interval_seconds)

    def stop(self) -> None:
        self._running = False

    def create_http_app(self):
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse

        app = FastAPI(title="SkillClaw Evolve Server")

        @app.post("/trigger")
        async def trigger_evolve():
            return JSONResponse(content=await self.run_once())

        @app.get("/status")
        async def status():
            entries = self._id_registry.all_entries()
            pending_keys = await self._call_storage(list_session_keys, self._bucket, self._prefix)
            return JSONResponse(
                content={
                    "running": self._running,
                    "pending_sessions": len(pending_keys),
                    "registered_skills": len(entries),
                    "skills": {
                        name: {
                            "skill_id": item["skill_id"],
                            "version": item.get("version", 0),
                        }
                        for name, item in entries.items()
                    },
                }
            )

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        return app

    @staticmethod
    def _sanitise_name(raw_name: str) -> str:
        name = raw_name.strip().lower()
        if SLUG_RE.match(name):
            return name
        name = re.sub(r"[^a-z0-9_-]", "-", name).strip("-")
        return name or "unnamed-skill"
