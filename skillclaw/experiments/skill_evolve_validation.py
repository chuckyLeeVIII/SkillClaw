"""
Targeted skill-evolve validation experiments.

This module implements two experiment families:

1. WildClawBench case study: one benchmark task with a copied default skill,
   run once before evolve and once after evolve.
2. Mini-task skill repair: four Slack-triage-style local tasks driven by a
   deliberately defective skill, evaluated before and after a single evolve pass.

Both experiments:

- run real OpenClaw + SkillClaw task execution locally
- use remote object storage (typically OSS) as the evolve backend
- trigger ``EvolveServer.run_once()`` explicitly instead of relying on a
  periodic service
"""

from __future__ import annotations

import asyncio
import difflib
import json
import os
import shutil
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from evolve_server.config import EvolveServerConfig
from evolve_server.server import EvolveServer

from ..skill_hub import SkillHub
from ..object_store import build_object_store
from .group_benchmark import BenchmarkTask, GroupBenchmarkConfig
from .openclaw_cluster import (
    OpenClawClusterManager,
    OpenClawClusterNode,
    OpenClawClusterSettings,
)
from .wildclawbench_executor import WildClawBenchClusterExecutor, parse_wildclaw_task_md


def _load_repo_env() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    env_path = root / ".env"
    values: dict[str, str] = {}
    if not env_path.exists():
        return values
    for line in env_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        values[key] = value
    return values


def _expand_string_env(value: str, env_map: dict[str, str]) -> str:
    out = value
    for key, env_value in env_map.items():
        out = out.replace(f"${{{key}}}", env_value)
    return os.path.expandvars(out)


def _safe_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return 0.0


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _coerce_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    return []


def _append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            out.append(payload)
    return out


def _clone_skill_with_new_name(src_skill_md: Path, dest_skills_root: Path, new_name: str) -> Path:
    raw = src_skill_md.read_text(encoding="utf-8")
    if not raw.startswith("---"):
        raise ValueError(f"skill file missing frontmatter: {src_skill_md}")
    end_idx = raw.find("\n---", 3)
    if end_idx == -1:
        raise ValueError(f"skill file missing closing frontmatter: {src_skill_md}")

    frontmatter = yaml.safe_load(raw[3:end_idx].strip()) or {}
    if not isinstance(frontmatter, dict):
        raise ValueError(f"invalid frontmatter: {src_skill_md}")
    frontmatter["name"] = new_name
    body = raw[end_idx + 4 :].lstrip("\n")

    target_dir = dest_skills_root / new_name
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "SKILL.md"
    target_text = (
        "---\n"
        + yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True).strip()
        + "\n---\n\n"
        + body
    )
    target_path.write_text(target_text, encoding="utf-8")
    return target_path


def _read_skill_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _summarize_skill_diff(before_text: str, after_text: str) -> dict[str, Any]:
    if not before_text and not after_text:
        return {"changed": False, "added_lines": 0, "removed_lines": 0, "excerpt": []}
    diff_lines = list(
        difflib.unified_diff(
            before_text.splitlines(),
            after_text.splitlines(),
            fromfile="before",
            tofile="after",
            lineterm="",
            n=1,
        )
    )
    return {
        "changed": before_text != after_text,
        "added_lines": sum(1 for line in diff_lines if line.startswith("+") and not line.startswith("+++")),
        "removed_lines": sum(1 for line in diff_lines if line.startswith("-") and not line.startswith("---")),
        "excerpt": diff_lines[:20],
    }


def _session_summary(session_path: Path) -> dict[str, Any]:
    payload = _load_json(session_path)
    turns = payload.get("turns") or []
    read_skill_names = sorted(
        {
            str(item.get("skill_name") or "").strip()
            for turn in turns
            for item in (turn.get("read_skills") or [])
            if isinstance(item, dict) and str(item.get("skill_name") or "").strip()
        }
    )
    return {
        "session_path": str(session_path),
        "session_id": payload.get("session_id"),
        "num_turns": len(turns),
        "read_skill_names": read_skill_names,
        "turn_prm_scores": [turn.get("prm_score") for turn in turns],
    }


def _annotate_session_score(session_path: Path, *, score: float, metadata: dict[str, Any]) -> Path:
    payload = _load_json(session_path)
    turns = payload.get("turns") or []
    if turns:
        turns[-1]["prm_score"] = score
    payload["num_turns"] = len(turns)
    payload["benchmark"] = {
        "overall_score": score,
        **metadata,
    }
    annotated = session_path.parent / f"{session_path.stem}.annotated.json"
    annotated.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return annotated


@dataclass
class MiniSlackQuerySpec:
    query_id: str
    title: str
    instruction: str
    messages: list[dict[str, Any]]
    contacts: list[dict[str, Any]]
    expected: dict[str, Any]


def _evaluate_mini_slack_query(query_dir: Path, spec: MiniSlackQuerySpec) -> dict[str, Any]:
    result_path = query_dir / "result.json"
    result_payload = _load_json(result_path)
    internal_dir = query_dir / "outbox" / "internal"
    drafts_dir = query_dir / "outbox" / "drafts"
    sent_dir = query_dir / "outbox" / "customer_sent"

    checks: dict[str, dict[str, Any]] = {}

    if "ignored_message_ids" in spec.expected:
        actual = sorted(_coerce_string_list(result_payload.get("ignored_message_ids", [])))
        expected = sorted(str(x) for x in spec.expected["ignored_message_ids"])
        checks["ignored_message_ids"] = {
            "passed": all(item in actual for item in expected),
            "expected": expected,
            "actual": actual,
        }

    if "routed_internal" in spec.expected:
        actual = sorted(_coerce_string_list(result_payload.get("routed_internal", [])))
        expected = sorted(str(x) for x in spec.expected["routed_internal"])
        created = sorted(path.stem for path in internal_dir.glob("*.md"))
        checks["routed_internal"] = {
            "passed": all(item in actual for item in expected) and all(item in created for item in expected),
            "expected": expected,
            "actual": actual,
            "created_files": created,
        }

    if "severity" in spec.expected:
        actual = str(result_payload.get("severity") or "").strip().lower()
        expected = str(spec.expected["severity"]).strip().lower()
        checks["severity"] = {
            "passed": actual == expected,
            "expected": expected,
            "actual": actual,
        }

    if "customer_reply_mode" in spec.expected:
        actual = str(result_payload.get("customer_reply_mode") or "").strip().lower()
        expected = str(spec.expected["customer_reply_mode"]).strip().lower()
        checks["customer_reply_mode"] = {
            "passed": actual == expected,
            "expected": expected,
            "actual": actual,
        }

    if spec.expected.get("require_draft_saved"):
        draft_files = sorted(path.name for path in drafts_dir.glob("*.md"))
        checks["require_draft_saved"] = {
            "passed": bool(draft_files),
            "actual": draft_files,
        }

    if spec.expected.get("forbid_customer_send"):
        sent_files = sorted(path.name for path in sent_dir.glob("*.md"))
        checks["forbid_customer_send"] = {
            "passed": len(sent_files) == 0,
            "actual": sent_files,
        }

    passed = bool(checks) and all(item["passed"] for item in checks.values())
    return {
        "score": 1.0 if passed else 0.0,
        "passed": passed,
        "checks": checks,
        "result_path": str(result_path),
        "result_payload": result_payload,
    }


def _default_mini_slack_queries() -> list[MiniSlackQuerySpec]:
    return [
        MiniSlackQuerySpec(
            query_id="q1_ignore_test",
            title="Ignore QA routing test",
            instruction=(
                "Review the inbox fixture and triage only real issues. "
                "Some messages may be test traffic hidden in metadata."
            ),
            messages=[
                {
                    "message_id": "msg_customer_101",
                    "sender": "ops@customer.com",
                    "subject": "Billing discrepancy",
                    "body": "Customer says they were double charged.",
                    "metadata": {"classification": "customer"},
                },
                {
                    "message_id": "msg_test_901",
                    "sender": "qa-automation@company.internal",
                    "subject": "GlobalTech Solutions urgent issue",
                    "body": "Please route immediately.",
                    "metadata": {
                        "classification": "QA-RT",
                        "footer": "quarterly support routing test",
                    },
                },
            ],
            contacts=[
                {"contact_id": "support-amy-wu", "name": "Amy Wu", "team": "support"},
                {"contact_id": "sales-jake-sun", "name": "Jake Sun", "team": "sales"},
            ],
            expected={
                "ignored_message_ids": ["msg_test_901"],
                "routed_internal": ["support-amy-wu"],
                "severity": "medium",
                "customer_reply_mode": "none",
                "forbid_customer_send": True,
            },
        ),
        MiniSlackQuerySpec(
            query_id="q2_contact_disambiguation",
            title="Route to the correct Jake",
            instruction=(
                "Triage the inbox fixture and route the issue to the correct internal owner. "
                "Multiple contacts may have very similar names."
            ),
            messages=[
                {
                    "message_id": "msg_customer_202",
                    "sender": "acct@enterprise.com",
                    "subject": "Upsell opportunity needs owner",
                    "body": "Please route this expansion opportunity to the sales owner Jake Sun.",
                    "metadata": {"classification": "customer"},
                }
            ],
            contacts=[
                {"contact_id": "sales-jake-chen", "name": "Jake Chen", "team": "sales"},
                {"contact_id": "sales-jake-sun", "name": "Jake Sun", "team": "sales"},
            ],
            expected={
                "routed_internal": ["sales-jake-sun"],
                "severity": "low",
                "customer_reply_mode": "none",
                "forbid_customer_send": True,
            },
        ),
        MiniSlackQuerySpec(
            query_id="q3_draft_not_send",
            title="Save draft instead of sending",
            instruction=(
                "Triage the customer complaint and prepare the correct follow-up artifact. "
                "Do not bypass the review step."
            ),
            messages=[
                {
                    "message_id": "msg_customer_303",
                    "sender": "renewal@bigco.com",
                    "subject": "Need customer reply",
                    "body": "Customer is asking for a status update before renewal.",
                    "metadata": {"classification": "customer"},
                }
            ],
            contacts=[
                {"contact_id": "support-amy-wu", "name": "Amy Wu", "team": "support"},
            ],
            expected={
                "routed_internal": ["support-amy-wu"],
                "severity": "medium",
                "customer_reply_mode": "draft",
                "require_draft_saved": True,
                "forbid_customer_send": True,
            },
        ),
        MiniSlackQuerySpec(
            query_id="q4_followup_severity",
            title="Use the follow-up signal to escalate",
            instruction=(
                "Triage the security-related issue. "
                "Use all relevant messages in the fixture before deciding severity and routing."
            ),
            messages=[
                {
                    "message_id": "msg_customer_404",
                    "sender": "security@finco.com",
                    "subject": "Possible SQL injection",
                    "body": "Customer suspects a vulnerability in one endpoint.",
                    "metadata": {"classification": "customer"},
                },
                {
                    "message_id": "msg_internal_405",
                    "sender": "kevin.liu@company.internal",
                    "subject": "Follow-up findings",
                    "body": "Security confirms staging is still vulnerable and more endpoints are affected.",
                    "metadata": {"classification": "internal", "related_to": "msg_customer_404"},
                },
            ],
            contacts=[
                {"contact_id": "ciso-kevin-liu", "name": "Kevin Liu", "team": "security"},
                {"contact_id": "legal-linda-chen", "name": "Linda Chen", "team": "legal"},
                {"contact_id": "support-amy-wu", "name": "Amy Wu", "team": "support"},
            ],
            expected={
                "routed_internal": ["ciso-kevin-liu", "legal-linda-chen"],
                "severity": "critical",
                "customer_reply_mode": "draft",
                "require_draft_saved": True,
                "forbid_customer_send": True,
            },
        ),
    ]


def _write_mini_slack_defective_skill(dest_skills_root: Path, *, skill_name: str) -> Path:
    target_dir = dest_skills_root / skill_name
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "SKILL.md"
    target_path.write_text(
        "\n".join(
            [
                "---",
                f"name: {skill_name}",
                'description: "Use when triaging local Slack-style inbox fixtures in workspace directories. Quickly classify issues, route them, and prepare customer follow-ups."',
                "category: general",
                "---",
                "",
                f"# {skill_name}",
                "",
                "## When to Use",
                "- Local support-inbox triage tasks stored in workspace folders.",
                "- Routing issues to internal owners and preparing customer follow-ups.",
                "",
                "## Procedure / Steps",
                "1. Read only the first customer-visible message in `messages.json` to avoid overfitting to noise.",
                "2. Ignore test traffic only when the subject or sender obviously contains `TEST`.",
                "3. Match the owner by taking the first contact whose name partially matches what you need.",
                "4. If a customer-facing reply is needed, write it directly to `outbox/customer_sent/` so it is ready to go.",
                "5. Save `result.json` with keys `query_id`, `ignored_message_ids`, `routed_internal`, `severity`, and `customer_reply_mode`.",
                "",
                "## Common Mistakes",
                "- Spending time on hidden metadata or footer notes.",
                "- Reading secondary/internal messages before finishing the first-pass triage.",
                "- Creating drafts when you already know what to tell the customer.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return target_path


@dataclass
class SkillEvolveValidationConfig:
    name: str = "skill-evolve-validation"
    workspace_dir: str = "records/skill_evolve_validation"
    benchmark_root: str = ""
    group_id_prefix: str = "skill-evolve-validation"
    storage_backend: str = ""
    storage_endpoint: str = ""
    storage_bucket: str = ""
    storage_access_key_id: str = ""
    storage_secret_access_key: str = ""
    storage_region: str = ""
    storage_session_token: str = ""
    agent_api_base: str = ""
    agent_api_key: str = ""
    agent_model: str = "moonshotai/Kimi-K2.5"
    evolve_api_base: str = ""
    evolve_api_key: str = ""
    evolve_model: str = "gpt-5.4"
    evolve_max_tokens: int = 100000
    evolve_use_success_feedback: bool = True
    cluster_configured_nodes: int = 4
    cluster_skillclaw_base_port: int = 39000
    cluster_gateway_base_port: int = 41000
    cluster_openclaw_bin: str = "openclaw"
    cluster_skillclaw_bin: str = "skillclaw"
    cluster_task_timeout_seconds: int = 300
    cluster_start_timeout_seconds: float = 20.0
    cluster_openclaw_mode: str = "gateway"
    cluster_seed_openclaw_dir: str = ""
    cluster_proxy_api_key: str = "skillclaw"
    cluster_retrieval_mode: str = "template"
    case_study_task_file: str = ""
    case_study_seed_skill_file: str = ""
    case_study_seed_skill_name: str = "03-task5-case-study"
    case_study_before_runs: int = 1
    case_study_after_runs: int = 1
    mini_skill_name: str = "slack-triage-defective"
    mini_before_runs: int = 4
    mini_after_runs: int = 4
    run_case_study: bool = True
    run_mini_task: bool = True

    @classmethod
    def from_file(cls, path: str) -> "SkillEvolveValidationConfig":
        cfg_path = Path(path).expanduser().resolve()
        raw = cfg_path.read_text(encoding="utf-8")
        if cfg_path.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(raw) or {}
        else:
            data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("skill evolve validation config must deserialize to a dict")

        repo_env = _load_repo_env()
        merged_env = {**repo_env, **os.environ}

        def _expand(obj: Any) -> Any:
            if isinstance(obj, str):
                return _expand_string_env(obj, merged_env)
            if isinstance(obj, list):
                return [_expand(item) for item in obj]
            if isinstance(obj, dict):
                return {key: _expand(value) for key, value in obj.items()}
            return obj

        data = _expand(data)
        base_dir = cfg_path.parent

        for key in (
            "workspace_dir",
            "benchmark_root",
            "cluster_seed_openclaw_dir",
            "cluster_openclaw_bin",
            "cluster_skillclaw_bin",
        ):
            value = str(data.get(key) or "").strip()
            if not value:
                continue
            path_obj = Path(value).expanduser()
            if path_obj.is_absolute():
                data[key] = str(path_obj.resolve())
            else:
                data[key] = str((base_dir / path_obj).resolve())

        benchmark_root = Path(str(data.get("benchmark_root") or "").strip()).expanduser() if data.get("benchmark_root") else None
        for key in ("case_study_task_file", "case_study_seed_skill_file"):
            value = str(data.get(key) or "").strip()
            if not value:
                continue
            path_obj = Path(value).expanduser()
            if path_obj.is_absolute():
                data[key] = str(path_obj.resolve())
            elif benchmark_root is not None:
                data[key] = str((benchmark_root / path_obj).resolve())
            else:
                data[key] = str((base_dir / path_obj).resolve())
        if benchmark_root and benchmark_root.exists():
            if not str(data.get("case_study_task_file") or "").strip():
                data["case_study_task_file"] = str(
                    (benchmark_root / "tasks" / "03_Social_Interaction" / "03_Social_Interaction_task_5_chat_escalation_routing.md").resolve()
                )
            if not str(data.get("case_study_seed_skill_file") or "").strip():
                data["case_study_seed_skill_file"] = str((benchmark_root / "skills" / "03_task5" / "SKILL.md").resolve())

        data["storage_backend"] = str(data.get("storage_backend") or merged_env.get("EVOLVE_STORAGE_BACKEND") or "").strip()
        data["storage_endpoint"] = str(data.get("storage_endpoint") or merged_env.get("EVOLVE_STORAGE_ENDPOINT") or "").strip()
        data["storage_bucket"] = str(data.get("storage_bucket") or merged_env.get("EVOLVE_STORAGE_BUCKET") or "").strip()
        data["storage_access_key_id"] = str(
            data.get("storage_access_key_id") or merged_env.get("EVOLVE_STORAGE_ACCESS_KEY_ID") or ""
        ).strip()
        data["storage_secret_access_key"] = str(
            data.get("storage_secret_access_key") or merged_env.get("EVOLVE_STORAGE_SECRET_ACCESS_KEY") or ""
        ).strip()
        data["storage_region"] = str(data.get("storage_region") or merged_env.get("EVOLVE_STORAGE_REGION") or "").strip()
        data["storage_session_token"] = str(
            data.get("storage_session_token") or merged_env.get("EVOLVE_STORAGE_SESSION_TOKEN") or ""
        ).strip()

        return cls(**data)


class _RemoteExperimentBackend:
    def __init__(
        self,
        *,
        config: SkillEvolveValidationConfig,
        group_id: str,
        user_alias: str,
    ):
        self.group_id = group_id
        self.user_alias = user_alias
        self.skill_hub = SkillHub(
            backend=config.storage_backend,
            endpoint=config.storage_endpoint,
            bucket=config.storage_bucket,
            access_key_id=config.storage_access_key_id,
            secret_access_key=config.storage_secret_access_key,
            region=config.storage_region,
            session_token=config.storage_session_token,
            local_root="",
            group_id=group_id,
            user_alias=user_alias,
        )
        self.object_store = build_object_store(
            backend=config.storage_backend,
            endpoint=config.storage_endpoint,
            bucket=config.storage_bucket,
            access_key_id=config.storage_access_key_id,
            secret_access_key=config.storage_secret_access_key,
            region=config.storage_region,
            session_token=config.storage_session_token,
        )

    def upload_session_file(self, session_path: Path) -> str:
        key = f"{self.group_id}/sessions/{session_path.name}"
        self.object_store.put_object(key, session_path.read_bytes())
        return key

    def push_skills(self, skills_dir: Path) -> dict[str, int]:
        return self.skill_hub.push_skills(str(skills_dir), skill_filter=None)

    def pull_skills(self, skills_dir: Path) -> dict[str, int]:
        skills_dir.mkdir(parents=True, exist_ok=True)
        return self.skill_hub.pull_skills(str(skills_dir))

    def load_manifest(self) -> list[dict[str, Any]]:
        manifest_key = f"{self.group_id}/manifest.jsonl"
        try:
            raw = self.object_store.get_object(manifest_key).read().decode("utf-8")
        except Exception:
            return []
        out: list[dict[str, Any]] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                out.append(payload)
        return out

    def fetch_skill(self, skill_name: str) -> str:
        key = f"{self.group_id}/skills/{skill_name}/SKILL.md"
        try:
            return self.object_store.get_object(key).read().decode("utf-8")
        except Exception:
            return ""


class SkillEvolveValidationRunner:
    def __init__(self, config: SkillEvolveValidationConfig):
        self.config = config
        self.workspace_dir = Path(config.workspace_dir).expanduser().resolve()
        self.benchmark_root = Path(config.benchmark_root).expanduser().resolve() if config.benchmark_root else None

    def run(self) -> dict[str, Any]:
        run_root = self.workspace_dir / time.strftime("%Y%m%d-%H%M%S")
        run_root.mkdir(parents=True, exist_ok=True)

        experiments: dict[str, Any] = {}
        if self.config.run_case_study:
            experiments["case_study"] = self._run_case_study(run_root / "case_study")
        if self.config.run_mini_task:
            experiments["mini_task"] = self._run_mini_task(run_root / "mini_task")

        report = {
            "name": self.config.name,
            "run_root": str(run_root),
            "agent_model": self.config.agent_model,
            "evolve_model": self.config.evolve_model,
            "storage": {
                "backend": self.config.storage_backend,
                "endpoint": self.config.storage_endpoint,
                "bucket": self.config.storage_bucket,
            },
            "experiments": experiments,
        }
        report_path = run_root / "report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (run_root / "report.md").write_text(self._build_markdown(report), encoding="utf-8")
        return report

    def _case_study_group_id(self) -> str:
        return f"{self.config.group_id_prefix}-case-study-{time.strftime('%Y%m%d-%H%M%S')}"

    def _mini_task_group_id(self) -> str:
        return f"{self.config.group_id_prefix}-mini-task-{time.strftime('%Y%m%d-%H%M%S')}"

    def _run_case_study(self, run_root: Path) -> dict[str, Any]:
        run_root.mkdir(parents=True, exist_ok=True)
        if not self.benchmark_root or not self.benchmark_root.exists():
            raise FileNotFoundError(f"benchmark_root not found: {self.benchmark_root}")
        task_file = Path(self.config.case_study_task_file).expanduser().resolve()
        seed_skill_file = Path(self.config.case_study_seed_skill_file).expanduser().resolve()
        seed_skills_dir = run_root / "seed_skills"
        cloned_skill_path = _clone_skill_with_new_name(
            seed_skill_file,
            seed_skills_dir,
            self.config.case_study_seed_skill_name,
        )
        backend = _RemoteExperimentBackend(
            config=self.config,
            group_id=self._case_study_group_id(),
            user_alias="case-study",
        )
        backend.push_skills(seed_skills_dir)

        before = self._run_wildclaw_once(
            phase_root=run_root / "before",
            task_file=task_file,
            skills_dir=seed_skills_dir,
            phase_name="before",
            round_idx=0,
        )
        before_oss_key = backend.upload_session_file(Path(before["session"]["session_path"]))

        evolve = self._run_remote_evolve_once(run_root=run_root / "evolve", backend=backend)

        after_skills_dir = run_root / "after_skills"
        backend.pull_skills(after_skills_dir)
        after = self._run_wildclaw_once(
            phase_root=run_root / "after",
            task_file=task_file,
            skills_dir=after_skills_dir,
            phase_name="after",
            round_idx=1,
        )

        before_skill_text = _read_skill_text(cloned_skill_path)
        manifest = backend.load_manifest()
        after_skill_text = backend.fetch_skill(self.config.case_study_seed_skill_name)
        evolved_skill_names = [str(item.get("name") or "") for item in manifest if str(item.get("name") or "").strip()]

        return {
            "group_id": backend.group_id,
            "task_file": str(task_file),
            "seed_skill_path": str(cloned_skill_path),
            "seed_skill_name": self.config.case_study_seed_skill_name,
            "before": before,
            "after": after,
            "evolve": evolve,
            "before_oss_session_key": before_oss_key,
            "skills_after": evolved_skill_names,
            "primary_metric": {
                "name": "overall_score_gain",
                "before": before["score"],
                "after": after["score"],
                "gain": round(after["score"] - before["score"], 6),
            },
            "skill_diff_summary": _summarize_skill_diff(before_skill_text, after_skill_text),
        }

    def _build_wildclaw_config(
        self,
        *,
        workspace_dir: Path,
        cloud_dir: Path,
        group_id: str,
        initial_skills_dir: Path,
    ) -> GroupBenchmarkConfig:
        return GroupBenchmarkConfig(
            name="skill-evolve-case-study",
            workspace_dir=str(workspace_dir),
            cloud_dir=str(cloud_dir),
            group_id=group_id,
            initial_skills_dir=str(initial_skills_dir),
            devices=1,
            cluster_configured_nodes=1,
            executor="wildclawbench_cluster",
            cluster_skillclaw_base_port=self.config.cluster_skillclaw_base_port,
            cluster_gateway_base_port=self.config.cluster_gateway_base_port,
            cluster_openclaw_bin=self.config.cluster_openclaw_bin,
            cluster_skillclaw_bin=self.config.cluster_skillclaw_bin,
            cluster_task_timeout_seconds=self.config.cluster_task_timeout_seconds,
            cluster_start_timeout_seconds=self.config.cluster_start_timeout_seconds,
            cluster_openclaw_mode=self.config.cluster_openclaw_mode,
            cluster_seed_openclaw_dir=self.config.cluster_seed_openclaw_dir,
            cluster_llm_provider="custom",
            cluster_llm_api_base=self.config.agent_api_base,
            cluster_llm_api_key=self.config.agent_api_key,
            cluster_llm_model_id=self.config.agent_model,
            cluster_proxy_api_key=self.config.cluster_proxy_api_key,
            cluster_retrieval_mode=self.config.cluster_retrieval_mode,
            cluster_prm_enabled=False,
            wildclawbench_root=str(self.benchmark_root),
            wildclawbench_output_dir=str(workspace_dir / "wildclawbench_output"),
            wildclawbench_docker_image="wildclawbench-ubuntu:v1.2",
            wildclawbench_container_proxy_host="host.docker.internal",
            wildclawbench_gateway_port=18789,
            wildclawbench_use_score_as_prm=True,
            wildclawbench_success_threshold=0.5,
        )

    def _run_wildclaw_once(
        self,
        *,
        phase_root: Path,
        task_file: Path,
        skills_dir: Path,
        phase_name: str,
        round_idx: int,
    ) -> dict[str, Any]:
        phase_root.mkdir(parents=True, exist_ok=True)
        cfg = self._build_wildclaw_config(
            workspace_dir=phase_root / "workspace",
            cloud_dir=phase_root / "cloud",
            group_id=f"{phase_name}-{uuid.uuid4().hex[:8]}",
            initial_skills_dir=skills_dir,
        )
        executor = WildClawBenchClusterExecutor(cfg)
        task = BenchmarkTask.from_dict(
            {
                "task_id": task_file.stem,
                "split": "eval",
                "task_file": str(task_file),
                "required_skills": [self.config.case_study_seed_skill_name],
            }
        )
        try:
            result = executor.run_task(
                task,
                device_id="device-0",
                device_dir=Path(cfg.workspace_dir) / "device-0",
                skills_dir=skills_dir,
                phase=phase_name,
                round_idx=round_idx,
            )
        finally:
            executor.teardown()
        session_path = Path(str((result.artifacts or {}).get("session_path") or ""))
        return {
            "success": bool(result.success),
            "score": _safe_float(result.score),
            "score_breakdown": dict(result.score_breakdown),
            "notes": result.notes,
            "artifacts": dict(result.artifacts),
            "session": _session_summary(session_path) if session_path.exists() else {},
        }

    def _run_remote_evolve_once(self, *, run_root: Path, backend: _RemoteExperimentBackend) -> dict[str, Any]:
        run_root.mkdir(parents=True, exist_ok=True)
        evolve_cfg = EvolveServerConfig(
            group_id=backend.group_id,
            storage_backend=self.config.storage_backend,
            storage_endpoint=self.config.storage_endpoint,
            storage_bucket=self.config.storage_bucket,
            storage_access_key_id=self.config.storage_access_key_id,
            storage_secret_access_key=self.config.storage_secret_access_key,
            storage_region=self.config.storage_region,
            storage_session_token=self.config.storage_session_token,
            llm_api_key=self.config.evolve_api_key,
            llm_base_url=self.config.evolve_api_base,
            llm_model=self.config.evolve_model,
            llm_max_tokens=self.config.evolve_max_tokens,
            use_success_feedback=self.config.evolve_use_success_feedback,
            processed_log_path=str(run_root / "processed.json"),
            history_path=str(run_root / "history.jsonl"),
        )
        summary = asyncio.run(EvolveServer(evolve_cfg).run_once())
        return {
            "summary": summary,
            "manifest": backend.load_manifest(),
            "processed_log_path": str(run_root / "processed.json"),
            "history_path": str(run_root / "history.jsonl"),
        }

    def _run_mini_task(self, run_root: Path) -> dict[str, Any]:
        run_root.mkdir(parents=True, exist_ok=True)
        queries = _default_mini_slack_queries()
        before_specs = queries[: self.config.mini_before_runs]
        after_specs = queries[: self.config.mini_after_runs]

        seed_skills_dir = run_root / "seed_skills"
        defective_skill_path = _write_mini_slack_defective_skill(
            seed_skills_dir,
            skill_name=self.config.mini_skill_name,
        )
        backend = _RemoteExperimentBackend(
            config=self.config,
            group_id=self._mini_task_group_id(),
            user_alias="mini-task",
        )
        backend.push_skills(seed_skills_dir)

        before = self._run_mini_queries_phase(
            phase_root=run_root / "before",
            queries=before_specs,
            skills_dir=seed_skills_dir,
            phase_name="before",
        )
        for record in before["records"]:
            session_path = Path(str(record.get("annotated_session_path") or ""))
            if session_path.exists():
                backend.upload_session_file(session_path)

        evolve = self._run_remote_evolve_once(run_root=run_root / "evolve", backend=backend)

        after_skills_dir = run_root / "after_skills"
        backend.pull_skills(after_skills_dir)
        after = self._run_mini_queries_phase(
            phase_root=run_root / "after",
            queries=after_specs,
            skills_dir=after_skills_dir,
            phase_name="after",
        )

        before_skill_text = _read_skill_text(defective_skill_path)
        after_skill_text = backend.fetch_skill(self.config.mini_skill_name)
        manifest = backend.load_manifest()
        return {
            "group_id": backend.group_id,
            "query_count": len(before_specs),
            "before": before,
            "after": after,
            "evolve": evolve,
            "skills_after": [str(item.get("name") or "") for item in manifest if str(item.get("name") or "").strip()],
            "primary_metric": {
                "name": "pass_count_gain",
                "before": before["pass_count"],
                "after": after["pass_count"],
                "gain": after["pass_count"] - before["pass_count"],
            },
            "skill_diff_summary": _summarize_skill_diff(before_skill_text, after_skill_text),
        }

    def _build_cluster_manager(
        self,
        *,
        workspace_dir: Path,
        cloud_dir: Path,
        group_id: str,
        initial_skills_dir: Path,
        devices: int,
    ) -> OpenClawClusterManager:
        cluster = OpenClawClusterManager(
            workspace_dir=workspace_dir,
            cloud_dir=cloud_dir,
            group_id=group_id,
            settings=OpenClawClusterSettings(
                configured_nodes=max(self.config.cluster_configured_nodes, devices),
                active_nodes=devices,
                skillclaw_base_port=self.config.cluster_skillclaw_base_port,
                gateway_base_port=self.config.cluster_gateway_base_port,
                openclaw_bin=self.config.cluster_openclaw_bin,
                skillclaw_bin=self.config.cluster_skillclaw_bin,
                node_command_timeout_s=self.config.cluster_task_timeout_seconds,
                start_timeout_s=self.config.cluster_start_timeout_seconds,
                openclaw_mode=self.config.cluster_openclaw_mode,
                llm_provider="custom",
                llm_api_base=self.config.agent_api_base,
                llm_api_key=self.config.agent_api_key,
                llm_model_id=self.config.agent_model,
                proxy_api_key=self.config.cluster_proxy_api_key,
                public_skill_root="",
                retrieval_mode=self.config.cluster_retrieval_mode,
                prm_enabled=False,
                seed_openclaw_dir=self.config.cluster_seed_openclaw_dir,
            ),
        )
        cluster.prepare(initial_skills_dir=initial_skills_dir)
        cluster.start_active_nodes(start_gateway=self.config.cluster_openclaw_mode != "local")
        return cluster

    def _materialize_mini_query(self, node: OpenClawClusterNode, spec: MiniSlackQuerySpec) -> Path:
        query_dir = node.root_dir / "workspace" / "mini_slack" / spec.query_id
        if query_dir.exists():
            shutil.rmtree(query_dir)
        (query_dir / "outbox" / "internal").mkdir(parents=True, exist_ok=True)
        (query_dir / "outbox" / "drafts").mkdir(parents=True, exist_ok=True)
        (query_dir / "outbox" / "customer_sent").mkdir(parents=True, exist_ok=True)
        (query_dir / "messages.json").write_text(json.dumps(spec.messages, ensure_ascii=False, indent=2), encoding="utf-8")
        (query_dir / "contacts.json").write_text(json.dumps(spec.contacts, ensure_ascii=False, indent=2), encoding="utf-8")
        return query_dir

    def _build_mini_query_instruction(self, *, query_dir: Path, spec: MiniSlackQuerySpec) -> str:
        return (
            f"{spec.instruction}\n\n"
            f"Use the available support triage skill.\n"
            f"Read `{query_dir / 'messages.json'}` and `{query_dir / 'contacts.json'}`.\n"
            f"Write `{query_dir / 'result.json'}` as JSON with keys:\n"
            f"- query_id\n- ignored_message_ids\n- routed_internal\n- severity\n- customer_reply_mode\n"
            f"If you route internally, create one markdown file per contact under `{query_dir / 'outbox' / 'internal'}`.\n"
            f"If a customer-facing reply is needed, save it as a draft under `{query_dir / 'outbox' / 'drafts'}`.\n"
            f"Do not write anything under `{query_dir / 'outbox' / 'customer_sent'}` unless you are explicitly told to send immediately.\n"
        )

    def _run_mini_queries_phase(
        self,
        *,
        phase_root: Path,
        queries: list[MiniSlackQuerySpec],
        skills_dir: Path,
        phase_name: str,
    ) -> dict[str, Any]:
        phase_root.mkdir(parents=True, exist_ok=True)
        cloud_dir = phase_root / "cloud"
        cluster = self._build_cluster_manager(
            workspace_dir=phase_root / "workspace",
            cloud_dir=cloud_dir,
            group_id=f"{phase_name}-{uuid.uuid4().hex[:8]}",
            initial_skills_dir=skills_dir,
            devices=len(queries),
        )
        try:
            jobs = [(f"device-{idx}", query) for idx, query in enumerate(queries)]
            with ThreadPoolExecutor(max_workers=len(jobs) or 1) as pool:
                futures = [
                    pool.submit(self._run_one_mini_query, cluster, node_id, query, phase_name, idx)
                    for idx, (node_id, query) in enumerate(jobs)
                ]
                records = [future.result() for future in futures]
        finally:
            cluster.stop_active_nodes()
        pass_count = sum(1 for record in records if record.get("passed"))
        scores = [_safe_float(record.get("score")) for record in records]
        records_path = phase_root / "records.jsonl"
        _append_jsonl(records_path, records)
        return {
            "records_path": str(records_path),
            "record_count": len(records),
            "pass_count": pass_count,
            "pass_rate": _safe_rate(pass_count, len(records)),
            "mean_score": sum(scores) / len(scores) if scores else 0.0,
            "records": records,
        }

    def _run_one_mini_query(
        self,
        cluster: OpenClawClusterManager,
        node_id: str,
        spec: MiniSlackQuerySpec,
        phase_name: str,
        round_idx: int,
    ) -> dict[str, Any]:
        node = cluster.nodes[node_id]
        query_dir = self._materialize_mini_query(node, spec)
        instruction = self._build_mini_query_instruction(query_dir=query_dir, spec=spec)
        invocation = cluster.invoke_task(
            node_id=node_id,
            instruction=instruction,
            round_idx=round_idx,
            phase=phase_name,
            task_id=spec.query_id,
        )
        artifacts = cluster.collect_session_artifacts(
            node_id=node_id,
            session_id=invocation["session_id"],
            requested_session_id=str(invocation.get("requested_session_id", "")),
            task_id=spec.query_id,
            phase=phase_name,
            round_idx=round_idx,
            skill_names=[self.config.mini_skill_name],
            conversation_slice=invocation.get("conversation_slice"),
            prm_slice=invocation.get("prm_slice"),
        )
        evaluation = _evaluate_mini_slack_query(query_dir, spec)
        annotated_session_path = _annotate_session_score(
            Path(str(artifacts["session_path"])),
            score=float(evaluation["score"]),
            metadata={
                "task_type": "mini_slack",
                "query_id": spec.query_id,
                "title": spec.title,
                "phase": phase_name,
            },
        )
        session = _session_summary(annotated_session_path)
        return {
            "query_id": spec.query_id,
            "title": spec.title,
            "phase": phase_name,
            "instruction": instruction,
            "score": evaluation["score"],
            "passed": evaluation["passed"],
            "checks": evaluation["checks"],
            "session": session,
            "annotated_session_path": str(annotated_session_path),
            "artifacts": artifacts,
        }

    def _build_markdown(self, report: dict[str, Any]) -> str:
        lines = [
            f"# {report['name']}",
            "",
            "## Summary",
            "",
            f"- Agent model: `{report['agent_model']}`",
            f"- Evolve model: `{report['evolve_model']}`",
            f"- Storage backend: `{report['storage']['backend']}`",
            "",
        ]

        case_study = report["experiments"].get("case_study")
        if case_study:
            metric = case_study["primary_metric"]
            lines.extend(
                [
                    "## Experiment 1 — WildClawBench Case Study",
                    "",
                    f"- Group ID: `{case_study['group_id']}`",
                    f"- Task: `{case_study['task_file']}`",
                    f"- Before score: {metric['before']:.3f}",
                    f"- After score: {metric['after']:.3f}",
                    f"- Gain: {metric['gain']:+.3f}",
                    f"- Skills after evolve: {', '.join(case_study.get('skills_after', [])) or '(none)'}",
                    "",
                ]
            )

        mini_task = report["experiments"].get("mini_task")
        if mini_task:
            metric = mini_task["primary_metric"]
            lines.extend(
                [
                    "## Experiment 2 — Mini-task Skill Repair",
                    "",
                    f"- Group ID: `{mini_task['group_id']}`",
                    f"- Before pass count: {metric['before']}",
                    f"- After pass count: {metric['after']}",
                    f"- Gain: {metric['gain']:+d}",
                    f"- Skills after evolve: {', '.join(mini_task.get('skills_after', [])) or '(none)'}",
                    "",
                ]
            )

        return "\n".join(lines) + "\n"


def run_skill_evolve_validation_from_config(config_path: str) -> dict[str, Any]:
    config = SkillEvolveValidationConfig.from_file(config_path)
    runner = SkillEvolveValidationRunner(config)
    return runner.run()
