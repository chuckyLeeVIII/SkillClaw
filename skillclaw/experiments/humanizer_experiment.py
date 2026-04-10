"""
Real OpenClaw humanizer experiment runner.

This runner validates a minimal support -> evolve -> query loop using:

- real OpenClaw nodes orchestrated by OpenClawClusterManager
- a local filesystem backend shared by LocalSkillHub and evolve_server
- an experiment-specific judge prompt for "humanized" rewrites

The experiment intentionally lives under skillclaw.experiments so it does not
change the main runtime path.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from evolve_server.config import EvolveServerConfig
from evolve_server.server import EvolveServer

from .local_skill_hub import LocalSkillHub
from .openclaw_cluster import OpenClawClusterManager, OpenClawClusterSettings

logger = logging.getLogger(__name__)

_SCORE_RE = re.compile(r"Score:\s*([-+]?\d)", re.IGNORECASE)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _collect_payload_text(payload: dict[str, Any]) -> str:
    result = payload.get("result")
    if not isinstance(result, dict):
        return ""
    texts: list[str] = []
    for item in result.get("payloads", []) or []:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str) and text:
            texts.append(text)
    return "\n".join(texts).strip()


def _build_runtime_instruction(task: "HumanizerTask") -> str:
    return (
        f"{task.instruction}\n\n"
        "Text to rewrite:\n"
        f"{task.source_text}\n"
    )


def _parse_judge_score(text: str) -> int | None:
    matches = _SCORE_RE.findall(text or "")
    if not matches:
        return None
    try:
        value = int(matches[-1])
    except ValueError:
        return None
    if value not in (-1, 0, 1):
        return None
    return value


def _majority_vote(scores: list[int | None]) -> float:
    valid = [s for s in scores if s is not None]
    if not valid:
        return 0.0
    counts: dict[int, int] = {}
    for value in valid:
        counts[value] = counts.get(value, 0) + 1
    top_score, top_count = max(counts.items(), key=lambda item: item[1])
    if list(counts.values()).count(top_count) > 1:
        return 0.0
    return float(top_score)


@dataclass
class HumanizerTask:
    task_id: str
    instruction: str
    source_text: str
    device: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "HumanizerTask":
        task_id = str(payload.get("task_id") or payload.get("id") or "").strip()
        instruction = str(payload.get("instruction") or "").strip()
        source_text = str(payload.get("source_text") or "").strip()
        if not task_id:
            raise ValueError("humanizer task is missing task_id")
        if not instruction:
            raise ValueError(f"humanizer task {task_id} is missing instruction")
        if not source_text:
            raise ValueError(f"humanizer task {task_id} is missing source_text")
        return cls(
            task_id=task_id,
            instruction=instruction,
            source_text=source_text,
            device=str(payload.get("device") or "").strip(),
            raw=dict(payload),
        )


@dataclass
class HumanizerExperimentConfig:
    name: str = "humanizer-real-openclaw-3node"
    workspace_dir: str = "records/humanizer_experiment"
    support_tasks_path: str = "examples/humanizer_support.jsonl"
    query_tasks_path: str = "examples/humanizer_query.jsonl"
    group_id: str = "demo"
    initial_skills_dir: str = ""
    devices: int = 3
    max_parallel_devices: int = 3
    cluster_configured_nodes: int = 8
    cluster_skillclaw_base_port: int = 39000
    cluster_gateway_base_port: int = 41000
    cluster_openclaw_bin: str = "openclaw"
    cluster_skillclaw_bin: str = "skillclaw"
    cluster_task_timeout_seconds: int = 180
    cluster_start_timeout_seconds: float = 20.0
    cluster_openclaw_mode: str = "gateway"
    cluster_seed_openclaw_dir: str = ""
    cluster_llm_provider: str = "custom"
    cluster_llm_api_base: str = ""
    cluster_llm_api_key: str = ""
    cluster_llm_model_id: str = ""
    cluster_proxy_api_key: str = ""
    cluster_retrieval_mode: str = "template"
    cluster_prm_enabled: bool = False
    cluster_prm_provider: str = "openai"
    cluster_prm_url: str = "https://api.openai.com/v1"
    cluster_prm_model: str = "gpt-5.2"
    cluster_prm_api_key: str = ""
    judge_api_base: str = ""
    judge_api_key: str = ""
    judge_model: str = ""
    judge_m: int = 3
    judge_temperature: float = 0.2
    judge_max_tokens: int = 768

    @classmethod
    def from_file(cls, path: str) -> "HumanizerExperimentConfig":
        cfg_path = Path(path).expanduser().resolve()
        raw = cfg_path.read_text(encoding="utf-8")
        if cfg_path.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(raw) or {}
        else:
            data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("humanizer experiment config must deserialize to a dict")

        base_dir = cfg_path.parent
        for key in (
            "workspace_dir",
            "support_tasks_path",
            "query_tasks_path",
            "initial_skills_dir",
            "cluster_seed_openclaw_dir",
        ):
            value = data.get(key)
            if isinstance(value, str) and value and not Path(value).expanduser().is_absolute():
                data[key] = str((base_dir / value).resolve())
        return cls(**data)


class HumanizerJudge:
    """Experiment-specific judge for humanized rewrites."""

    def __init__(
        self,
        *,
        api_base: str,
        api_key: str,
        model: str,
        m: int = 3,
        temperature: float = 0.2,
        max_tokens: int = 768,
    ):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.m = m
        self.temperature = temperature
        self.max_tokens = max_tokens

    def evaluate(self, *, source_text: str, instruction: str, rewrite_text: str) -> dict[str, Any]:
        messages = self._build_messages(
            source_text=source_text,
            instruction=instruction,
            rewrite_text=rewrite_text,
        )

        results = [self._query_once(messages, vote_id=i) for i in range(self.m)]
        scores = [score for score, _ in results]
        final = _majority_vote(scores)
        representative = ""
        for score, text in results:
            if score is not None and score == int(final):
                representative = text
                break
        return {
            "score": final,
            "votes": [score if score is not None else "fail" for score in scores],
            "eval_text": representative,
        }

    def _query_once(self, messages: list[dict[str, str]], vote_id: int) -> tuple[int | None, str]:
        try:
            content = asyncio.run(self._query_once_async(messages))
            return _parse_judge_score(content), content
        except Exception as exc:
            logger.warning("[HumanizerJudge] vote %d failed: %s", vote_id, exc)
            return None, ""

    async def _query_once_async(self, messages: list[dict[str, str]]) -> str:
        import httpx

        from ..api_server import (
            _assemble_streaming_chat_completion,
            _collect_sse_chat_events,
        )

        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_tokens,
            "stream": False,
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self.api_base}/chat/completions",
                json=body,
                headers=headers,
            )
            if resp.status_code < 400:
                payload = resp.json()
                return str(payload.get("choices", [{}])[0].get("message", {}).get("content") or "")

            if resp.status_code == 400 and "Stream must be set to true" in resp.text:
                stream_body = dict(body)
                stream_body["stream"] = True
                async with client.stream(
                    "POST",
                    f"{self.api_base}/chat/completions",
                    json=stream_body,
                    headers=headers,
                ) as stream_resp:
                    stream_resp.raise_for_status()
                    events = await _collect_sse_chat_events(stream_resp)
                payload = _assemble_streaming_chat_completion(
                    events,
                    fallback_model=self.model,
                )
                return str(payload.get("choices", [{}])[0].get("message", {}).get("content") or "")

            resp.raise_for_status()
            return ""

    @staticmethod
    def _build_messages(
        *,
        source_text: str,
        instruction: str,
        rewrite_text: str,
    ) -> list[dict[str, str]]:
        system = (
            "You are a strict judge for rewrite quality.\n"
            "The task is to rewrite text so it sounds naturally human-written rather than AI-generated.\n"
            "Judge the rewrite on four things only:\n"
            "1. It keeps the original meaning.\n"
            "2. It follows the rewrite instruction.\n"
            "3. It sounds like a real person wrote it, not generic AI prose.\n"
            "4. It avoids obvious AI tells: inflated importance, promotional tone, tidy rule-of-three phrasing, em-dash abuse, stiff structure, vague filler, and obvious chatbot voice.\n"
            "Use Score: 1 if it clearly passes.\n"
            "Use Score: -1 if it still sounds AI-generated, violates the instruction, or changes meaning in a material way.\n"
            "Use Score: 0 if it is mixed or unclear.\n"
            "Give a brief reason, then end with exactly one of: Score: 1 / Score: 0 / Score: -1"
        )
        user = (
            f"Instruction:\n{instruction}\n\n"
            f"Original text:\n{source_text}\n\n"
            f"Rewrite:\n{rewrite_text}\n\n"
            "Evaluate the rewrite."
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]


class HumanizerExperimentRunner:
    def __init__(self, config: HumanizerExperimentConfig):
        self.config = config
        self.workspace_dir = Path(config.workspace_dir).expanduser().resolve()
        self.support_tasks_path = Path(config.support_tasks_path).expanduser().resolve()
        self.query_tasks_path = Path(config.query_tasks_path).expanduser().resolve()

        judge_api_base = config.judge_api_base or config.cluster_llm_api_base
        judge_api_key = config.judge_api_key or config.cluster_llm_api_key
        judge_model = config.judge_model or config.cluster_llm_model_id
        if not judge_api_base or not judge_model:
            raise ValueError("humanizer experiment requires judge_api_base/judge_model or cluster_llm_* fallback")
        self._judge_kwargs = {
            "api_base": judge_api_base,
            "api_key": judge_api_key,
            "model": judge_model,
            "m": config.judge_m,
            "temperature": config.judge_temperature,
            "max_tokens": config.judge_max_tokens,
        }

    def run(self) -> dict[str, Any]:
        run_id = time.strftime("%Y%m%d-%H%M%S")
        run_root = self.workspace_dir / run_id
        run_root.mkdir(parents=True, exist_ok=True)

        initial_skills_dir = self._ensure_initial_skills_dir(run_root)
        support_tasks = self._load_tasks(self.support_tasks_path)
        query_tasks = self._load_tasks(self.query_tasks_path)

        baseline = self._run_query_phase(
            run_root=run_root,
            phase_name="baseline_query",
            tasks=query_tasks,
            backend_root=run_root / "baseline_backend",
            initial_skills_dir=initial_skills_dir,
            preload_remote_skills=False,
        )
        support = self._run_support_phase(
            run_root=run_root,
            tasks=support_tasks,
            backend_root=run_root / "support_backend",
            initial_skills_dir=initial_skills_dir,
        )
        evolve = self._run_evolve_phase(
            run_root=run_root,
            backend_root=run_root / "support_backend",
        )
        post = self._run_query_phase(
            run_root=run_root,
            phase_name="post_evolve_query",
            tasks=query_tasks,
            backend_root=run_root / "support_backend",
            initial_skills_dir=initial_skills_dir,
            preload_remote_skills=True,
        )

        report = self._build_report(
            run_root=run_root,
            baseline=baseline,
            support=support,
            evolve=evolve,
            post=post,
        )
        report_path = run_root / "report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (run_root / "report.md").write_text(self._build_report_markdown(report), encoding="utf-8")
        return report

    def _ensure_initial_skills_dir(self, run_root: Path) -> Path:
        if self.config.initial_skills_dir:
            path = Path(self.config.initial_skills_dir).expanduser().resolve()
            path.mkdir(parents=True, exist_ok=True)
            return path
        path = run_root / "initial_skills"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _load_tasks(self, path: Path) -> list[HumanizerTask]:
        if not path.exists():
            raise FileNotFoundError(f"task file not found: {path}")
        tasks: list[HumanizerTask] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tasks.append(HumanizerTask.from_dict(json.loads(line)))
        return tasks

    def _cluster_settings(self) -> OpenClawClusterSettings:
        cfg = self.config
        return OpenClawClusterSettings(
            configured_nodes=max(cfg.cluster_configured_nodes, cfg.devices),
            active_nodes=cfg.devices,
            skillclaw_base_port=cfg.cluster_skillclaw_base_port,
            gateway_base_port=cfg.cluster_gateway_base_port,
            openclaw_bin=cfg.cluster_openclaw_bin,
            skillclaw_bin=cfg.cluster_skillclaw_bin,
            node_command_timeout_s=cfg.cluster_task_timeout_seconds,
            start_timeout_s=cfg.cluster_start_timeout_seconds,
            openclaw_mode=cfg.cluster_openclaw_mode,
            llm_provider=cfg.cluster_llm_provider,
            llm_api_base=cfg.cluster_llm_api_base,
            llm_api_key=cfg.cluster_llm_api_key,
            llm_model_id=cfg.cluster_llm_model_id,
            proxy_api_key=cfg.cluster_proxy_api_key,
            retrieval_mode=cfg.cluster_retrieval_mode,
            prm_enabled=cfg.cluster_prm_enabled,
            prm_provider=cfg.cluster_prm_provider,
            prm_url=cfg.cluster_prm_url,
            prm_model=cfg.cluster_prm_model,
            prm_api_key=cfg.cluster_prm_api_key,
            seed_openclaw_dir=cfg.cluster_seed_openclaw_dir,
        )

    def _prepare_cluster(
        self,
        *,
        workspace_dir: Path,
        backend_root: Path,
        initial_skills_dir: Path,
        preload_remote_skills: bool,
    ) -> OpenClawClusterManager:
        cluster = OpenClawClusterManager(
            workspace_dir=workspace_dir,
            cloud_dir=backend_root,
            group_id=self.config.group_id,
            settings=self._cluster_settings(),
        )
        cluster.prepare(initial_skills_dir=initial_skills_dir)
        if preload_remote_skills:
            hub = LocalSkillHub(root_dir=str(backend_root), group_id=self.config.group_id, user_alias="humanizer")
            for node_id in cluster._active_ids:
                node = cluster.nodes[node_id]
                hub.pull_skills(str(node.skills_dir))
        cluster.start_active_nodes()
        return cluster

    def _run_support_phase(
        self,
        *,
        run_root: Path,
        tasks: list[HumanizerTask],
        backend_root: Path,
        initial_skills_dir: Path,
    ) -> dict[str, Any]:
        workspace_dir = run_root / "support_workspace"
        cluster = self._prepare_cluster(
            workspace_dir=workspace_dir,
            backend_root=backend_root,
            initial_skills_dir=initial_skills_dir,
            preload_remote_skills=False,
        )
        support_log = run_root / "support_runs.jsonl"
        try:
            buckets: dict[str, list[HumanizerTask]] = {f"device-{idx}": [] for idx in range(self.config.devices)}
            for idx, task in enumerate(tasks):
                if task.device:
                    buckets.setdefault(task.device, []).append(task)
                else:
                    buckets[f"device-{idx % self.config.devices}"].append(task)

            max_workers = max(1, min(self.config.max_parallel_devices, self.config.devices))
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [
                    pool.submit(
                        self._run_support_device,
                        cluster,
                        node_id,
                        device_tasks,
                        backend_root,
                        support_log,
                    )
                    for node_id, device_tasks in buckets.items()
                    if device_tasks
                ]
                results = [record for future in futures for record in future.result()]
        finally:
            cluster.stop_active_nodes()

        success_count = sum(1 for rec in results if rec["success"])
        failure_count = len(results) - success_count
        scores = [float(rec["judge"]["score"]) for rec in results]
        return {
            "tasks": len(results),
            "successes": success_count,
            "failures": failure_count,
            "success_rate": _safe_rate(success_count, len(results)),
            "mean_score": mean(scores) if scores else 0.0,
            "records_path": str(support_log),
        }

    def _run_support_device(
        self,
        cluster: OpenClawClusterManager,
        node_id: str,
        tasks: list[HumanizerTask],
        backend_root: Path,
        log_path: Path,
    ) -> list[dict[str, Any]]:
        judge = HumanizerJudge(**self._judge_kwargs)
        node = cluster.nodes[node_id]
        records: list[dict[str, Any]] = []
        for task in tasks:
            record = self._run_one_task(cluster, node_id, node.skills_dir, task, judge, phase="support")
            self._write_support_session(backend_root=backend_root, device_id=node_id, record=record)
            _append_jsonl(log_path, record)
            records.append(record)
        return records

    def _run_query_phase(
        self,
        *,
        run_root: Path,
        phase_name: str,
        tasks: list[HumanizerTask],
        backend_root: Path,
        initial_skills_dir: Path,
        preload_remote_skills: bool,
    ) -> dict[str, Any]:
        workspace_dir = run_root / f"{phase_name}_workspace"
        cluster = self._prepare_cluster(
            workspace_dir=workspace_dir,
            backend_root=backend_root,
            initial_skills_dir=initial_skills_dir,
            preload_remote_skills=preload_remote_skills,
        )
        log_path = run_root / f"{phase_name}.jsonl"
        try:
            max_workers = max(1, min(self.config.max_parallel_devices, self.config.devices))
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [
                    pool.submit(
                        self._run_query_device,
                        cluster,
                        node_id,
                        tasks,
                        log_path,
                        phase_name,
                    )
                    for node_id in cluster._active_ids
                ]
                results = [record for future in futures for record in future.result()]
        finally:
            cluster.stop_active_nodes()

        success_count = sum(1 for rec in results if rec["success"])
        scores = [float(rec["judge"]["score"]) for rec in results]
        by_device: dict[str, dict[str, Any]] = {}
        for rec in results:
            info = by_device.setdefault(rec["device_id"], {"tasks": 0, "successes": 0, "scores": []})
            info["tasks"] += 1
            info["successes"] += 1 if rec["success"] else 0
            info["scores"].append(float(rec["judge"]["score"]))
        per_device = {
            device_id: {
                "tasks": info["tasks"],
                "successes": info["successes"],
                "success_rate": _safe_rate(info["successes"], info["tasks"]),
                "mean_score": mean(info["scores"]) if info["scores"] else 0.0,
            }
            for device_id, info in by_device.items()
        }
        return {
            "tasks": len(results),
            "successes": success_count,
            "success_rate": _safe_rate(success_count, len(results)),
            "mean_score": mean(scores) if scores else 0.0,
            "records_path": str(log_path),
            "per_device": per_device,
        }

    def _run_query_device(
        self,
        cluster: OpenClawClusterManager,
        node_id: str,
        tasks: list[HumanizerTask],
        log_path: Path,
        phase_name: str,
    ) -> list[dict[str, Any]]:
        judge = HumanizerJudge(**self._judge_kwargs)
        node = cluster.nodes[node_id]
        records: list[dict[str, Any]] = []
        for task in tasks:
            record = self._run_one_task(cluster, node_id, node.skills_dir, task, judge, phase=phase_name)
            _append_jsonl(log_path, record)
            records.append(record)
        return records

    def _run_one_task(
        self,
        cluster: OpenClawClusterManager,
        node_id: str,
        skills_dir: Path,
        task: HumanizerTask,
        judge: HumanizerJudge,
        *,
        phase: str,
    ) -> dict[str, Any]:
        runtime_instruction = _build_runtime_instruction(task)
        invocation = cluster.invoke_task(
            node_id=node_id,
            instruction=runtime_instruction,
            round_idx=1,
            phase=phase,
            task_id=task.task_id,
        )
        payload = invocation.get("payload", {})
        response_text = (
            _collect_payload_text(payload)
            or str(payload.get("summary") or "")
            or str(invocation.get("stdout") or "")
        ).strip()
        judge_result = judge.evaluate(
            source_text=task.source_text,
            instruction=task.instruction,
            rewrite_text=response_text,
        )
        local_skills = sorted(path.parent.name for path in skills_dir.glob("*/SKILL.md"))
        success = float(judge_result["score"]) > 0.0
        return {
            "task_id": task.task_id,
            "device_id": node_id,
            "phase": phase,
            "instruction": task.instruction,
            "runtime_instruction": runtime_instruction,
            "source_text": task.source_text,
            "response_text": response_text,
            "success": success,
            "judge": judge_result,
            "available_skills": local_skills,
            "raw_session_id": invocation.get("session_id", ""),
            "returncode": invocation.get("returncode", 0),
        }

    def _write_support_session(self, *, backend_root: Path, device_id: str, record: dict[str, Any]) -> str:
        session_id = f"support-{record['task_id']}-{device_id}-{uuid.uuid4().hex[:8]}"
        session_dir = backend_root / self.config.group_id / "sessions"
        session_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "session_id": session_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "user_alias": device_id,
            "num_turns": 1,
            "task_id": record["task_id"],
            "phase": "support",
            "turns": [
                {
                    "turn_num": 1,
                    "prompt_text": record["runtime_instruction"],
                    "response_text": record["response_text"],
                    "read_skills": [],
                    "tool_errors": [],
                    "injected_skills": list(record.get("available_skills", [])),
                    "prm_score": float(record["judge"]["score"]),
                }
            ],
            "judge": record["judge"],
            "source_text": record["source_text"],
            "raw_session_id": record.get("raw_session_id", ""),
        }
        session_path = session_dir / f"{session_id}.json"
        session_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        record["session_path"] = str(session_path)
        record["evolve_session_id"] = session_id
        return str(session_path)

    def _run_evolve_phase(self, *, run_root: Path, backend_root: Path) -> dict[str, Any]:
        evolve_root = run_root / "evolve"
        evolve_root.mkdir(parents=True, exist_ok=True)
        config = EvolveServerConfig(
            group_id=self.config.group_id,
            local_root=str(backend_root),
            llm_api_key=self._judge_kwargs["api_key"],
            llm_base_url=self._judge_kwargs["api_base"],
            llm_model=self._judge_kwargs["model"],
            processed_log_path=str(evolve_root / "evolve_processed.json"),
            history_path=str(evolve_root / "evolve_history.jsonl"),
        )
        server = EvolveServer(config)
        summary = asyncio.run(server.run_once())

        manifest_path = backend_root / self.config.group_id / "manifest.jsonl"
        registry_path = backend_root / self.config.group_id / "evolve_skill_registry.json"
        manifest = self._load_manifest(manifest_path)
        registry = self._load_json(registry_path)
        return {
            "summary": summary,
            "manifest_path": str(manifest_path),
            "registry_path": str(registry_path),
            "skills": manifest,
            "registry": registry,
            "processed_log_path": str(evolve_root / "evolve_processed.json"),
            "history_path": str(evolve_root / "evolve_history.jsonl"),
        }

    def _load_manifest(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _load_json(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _build_report(
        self,
        *,
        run_root: Path,
        baseline: dict[str, Any],
        support: dict[str, Any],
        evolve: dict[str, Any],
        post: dict[str, Any],
    ) -> dict[str, Any]:
        baseline_rate = float(baseline.get("success_rate", 0.0))
        post_rate = float(post.get("success_rate", 0.0))
        baseline_score = float(baseline.get("mean_score", 0.0))
        post_score = float(post.get("mean_score", 0.0))
        return {
            "name": self.config.name,
            "run_root": str(run_root),
            "group_id": self.config.group_id,
            "devices": self.config.devices,
            "support_task_count": support.get("tasks", 0),
            "query_task_count": len(self._load_tasks(self.query_tasks_path)),
            "baseline_query": baseline,
            "support": support,
            "evolve": evolve,
            "post_evolve_query": post,
            "summary": {
                "baseline_success_rate": baseline_rate,
                "post_success_rate": post_rate,
                "absolute_gain": post_rate - baseline_rate,
                "baseline_mean_score": baseline_score,
                "post_mean_score": post_score,
                "mean_score_gain": post_score - baseline_score,
                "skills_evolved": int(evolve.get("summary", {}).get("skills_evolved", 0)),
                "failed_turns": int(evolve.get("summary", {}).get("failed_turns", 0)),
            },
        }

    def _build_report_markdown(self, report: dict[str, Any]) -> str:
        summary = report["summary"]
        evolve = report["evolve"]["summary"]
        skills = report["evolve"]["skills"]
        lines = [
            f"# {report['name']}",
            "",
            "## Summary",
            "",
            f"- Baseline query success: {summary['baseline_success_rate']:.3f}",
            f"- Post-evolve query success: {summary['post_success_rate']:.3f}",
            f"- Absolute gain: {summary['absolute_gain']:+.3f}",
            f"- Baseline mean judge score: {summary['baseline_mean_score']:.3f}",
            f"- Post-evolve mean judge score: {summary['post_mean_score']:.3f}",
            f"- Mean score gain: {summary['mean_score_gain']:+.3f}",
            f"- Support failed turns sent to evolve: {evolve.get('failed_turns', 0)}",
            f"- Skills evolved: {evolve.get('skills_evolved', 0)}",
            "",
            "## Evolved skills",
            "",
        ]
        if skills:
            for rec in skills:
                lines.append(
                    f"- {rec.get('name', '?')} (v{rec.get('version', '?')}, sha={str(rec.get('sha256', ''))[:12]})"
                )
        else:
            lines.append("- No skills were evolved.")
        lines.extend(
            [
                "",
                "## Artifacts",
                "",
                f"- JSON report: `{report['run_root']}/report.json`",
                f"- Support runs: `{report['support']['records_path']}`",
                f"- Baseline query: `{report['baseline_query']['records_path']}`",
                f"- Post-evolve query: `{report['post_evolve_query']['records_path']}`",
                f"- Evolve manifest: `{report['evolve']['manifest_path']}`",
                f"- Evolve registry: `{report['evolve']['registry_path']}`",
            ]
        )
        return "\n".join(lines) + "\n"


def run_humanizer_experiment_from_config(config_path: str) -> dict[str, Any]:
    config = HumanizerExperimentConfig.from_file(config_path)
    runner = HumanizerExperimentRunner(config)
    return runner.run()
