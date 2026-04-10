"""
Round-based benchmark scaffold for distributed skill sharing experiments.

The benchmark intentionally separates three concerns:

- orchestration: task sharding, rounds, sync, evaluation, reporting
- execution: replay or external command-based task runner
- storage: local device workspaces and a local shared skill hub

This keeps the harness useful before the real OpenClaw rollout is wired in.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import random
import re
import shutil
import subprocess
import sys
import time
from contextlib import suppress
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol

import yaml

from .local_skill_hub import LocalSkillHub
from .openclaw_cluster import OpenClawClusterManager, OpenClawClusterSettings
from ..skill_manager import SkillManager

logger = logging.getLogger(__name__)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
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
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _mean_score(values: list[Any]) -> float:
    numeric = [float(v) for v in values if isinstance(v, (int, float))]
    if not numeric:
        return 0.0
    return sum(numeric) / len(numeric)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip()).strip("-").lower()
    return slug or "task"


def _stable_index(value: str, modulo: int) -> int:
    digest = sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def _copy_tree_if_exists(src: Path | None, dst: Path) -> None:
    if not src or not src.exists():
        dst.mkdir(parents=True, exist_ok=True)
        return
    shutil.copytree(src, dst, dirs_exist_ok=True)


@dataclass
class BenchmarkTask:
    task_id: str
    instruction: str
    split: str = "train"
    required_skills: list[str] = field(default_factory=list)
    discoverable_skills: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, split_field: str = "split") -> "BenchmarkTask":
        task_id = str(payload.get("task_id") or payload.get("id") or "").strip()
        if not task_id:
            raise ValueError("task is missing task_id")

        instruction = str(
            payload.get("instruction")
            or payload.get("prompt")
            or payload.get("task_prompt")
            or payload.get("task_file")
            or payload.get("task_path")
            or ""
        ).strip()
        split = str(payload.get(split_field) or payload.get("split") or "train")
        required = [str(x) for x in payload.get("required_skills", [])]

        raw_skills = payload.get("discoverable_skills")
        if raw_skills is None and required:
            raw_skills = list(required)

        discoverable: list[dict[str, Any]] = []
        for item in raw_skills or []:
            discoverable.append(_normalize_skill_spec(item, task_id=task_id))

        return cls(
            task_id=task_id,
            instruction=instruction,
            split=split,
            required_skills=required,
            discoverable_skills=discoverable,
            raw=dict(payload),
        )


def _normalize_skill_spec(spec: Any, *, task_id: str) -> dict[str, Any]:
    if isinstance(spec, str):
        name = spec.strip()
        return {
            "name": name,
            "description": f"Discovered from benchmark task {task_id}.",
            "content": f"# {name}\n\nDiscovered during benchmark task {task_id}.\n",
            "category": "general",
        }

    if not isinstance(spec, dict):
        raise TypeError(f"unsupported skill spec type: {type(spec)!r}")

    name = str(spec.get("name") or "").strip()
    if not name:
        raise ValueError(f"skill spec for task {task_id} is missing name")

    out = dict(spec)
    out.setdefault("description", f"Discovered from benchmark task {task_id}.")
    out.setdefault("content", f"# {name}\n\nDiscovered during benchmark task {task_id}.\n")
    out.setdefault("category", "general")
    return out


def _collect_payload_text(payload: dict[str, Any]) -> str:
    result = payload.get("result")
    if not isinstance(result, dict):
        return ""
    texts: list[str] = []
    for key in ("text", "output_text", "message"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            texts.append(value.strip())
    for item in result.get("payloads", []) or []:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str) and text:
            texts.append(text)
    for item in result.get("messages", []) or []:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str) and text:
            texts.append(text)
    return "\n".join(texts).strip()


@dataclass
class ExecutorResult:
    success: bool
    used_skills: list[str] = field(default_factory=list)
    discovered_skills: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0
    notes: str = ""
    artifacts: dict[str, Any] = field(default_factory=dict)
    score: float | None = None
    score_breakdown: dict[str, Any] = field(default_factory=dict)


class TaskExecutor(Protocol):
    def run_task(
        self,
        task: BenchmarkTask,
        *,
        device_id: str,
        device_dir: Path,
        skills_dir: Path,
        phase: str,
        round_idx: int,
    ) -> ExecutorResult:
        ...


class ReplayExecutor:
    """Offline executor driven by task annotations."""

    def run_task(
        self,
        task: BenchmarkTask,
        *,
        device_id: str,
        device_dir: Path,
        skills_dir: Path,
        phase: str,
        round_idx: int,
    ) -> ExecutorResult:
        skill_names = {p.parent.name for p in skills_dir.glob("*/SKILL.md")}
        required = set(task.required_skills)
        used = sorted(required & skill_names)
        success = required.issubset(skill_names)
        discovered = task.discoverable_skills if phase == "train" else []
        return ExecutorResult(
            success=success,
            used_skills=used,
            discovered_skills=list(discovered),
            latency_ms=0.0,
            notes=f"replay_executor device={device_id} round={round_idx}",
            artifacts={},
        )


class CommandExecutor:
    """Executes an external command and reads a JSON result from stdout."""

    def __init__(self, command: list[str]):
        if not command:
            raise ValueError("executor_command must not be empty when executor=command")
        self.command = list(command)

    def run_task(
        self,
        task: BenchmarkTask,
        *,
        device_id: str,
        device_dir: Path,
        skills_dir: Path,
        phase: str,
        round_idx: int,
    ) -> ExecutorResult:
        child_env = dict(os.environ)
        child_env.update(
            {
                "SKILLCLAW_DEVICE_ID": device_id,
                "SKILLCLAW_DEVICE_DIR": str(device_dir),
                "SKILLCLAW_SKILLS_DIR": str(skills_dir),
                "SKILLCLAW_PHASE": phase,
                "SKILLCLAW_ROUND": str(round_idx),
                "SKILLCLAW_TASK_JSON": json.dumps(task.raw, ensure_ascii=False),
                "SKILLCLAW_PYTHON": sys.executable,
            }
        )

        started = time.perf_counter()
        proc = subprocess.run(
            self.command,
            cwd=str(device_dir),
            env=child_env,
            capture_output=True,
            text=True,
            check=False,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        payload = {}
        stdout = (proc.stdout or "").strip()
        if stdout:
            try:
                payload = json.loads(stdout.splitlines()[-1])
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"executor did not emit valid JSON: {stdout[:200]}") from exc

        if proc.returncode != 0 and not payload:
            return ExecutorResult(
                success=False,
                latency_ms=elapsed_ms,
                notes=(proc.stderr or stdout or f"executor exit code {proc.returncode}")[:400],
            )

        discovered = [
            _normalize_skill_spec(item, task_id=task.task_id)
            for item in payload.get("discovered_skills", [])
        ]
        return ExecutorResult(
            success=bool(payload.get("success", False)),
            used_skills=[str(x) for x in payload.get("used_skills", [])],
            discovered_skills=discovered,
            latency_ms=float(payload.get("latency_ms", elapsed_ms)),
            notes=str(payload.get("notes", ""))[:400],
            artifacts={},
        )


class OpenClawClusterExecutor:
    """Runs benchmark tasks through isolated OpenClaw + SkillClaw nodes."""

    def __init__(self, config: "GroupBenchmarkConfig"):
        self.config = config
        self.cluster = OpenClawClusterManager(
            workspace_dir=Path(config.workspace_dir),
            cloud_dir=Path(config.cloud_dir),
            group_id=config.group_id,
            settings=OpenClawClusterSettings(
                configured_nodes=max(config.cluster_configured_nodes, config.devices),
                active_nodes=config.devices,
                skillclaw_base_port=config.cluster_skillclaw_base_port,
                gateway_base_port=config.cluster_gateway_base_port,
                openclaw_bin=config.cluster_openclaw_bin,
                skillclaw_bin=config.cluster_skillclaw_bin,
                node_command_timeout_s=config.cluster_task_timeout_seconds,
                start_timeout_s=config.cluster_start_timeout_seconds,
                openclaw_mode=config.cluster_openclaw_mode,
                llm_provider=config.cluster_llm_provider,
                llm_api_base=config.cluster_llm_api_base,
                llm_api_key=config.cluster_llm_api_key,
                llm_model_id=config.cluster_llm_model_id,
                proxy_api_key=config.cluster_proxy_api_key,
                public_skill_root="",
                retrieval_mode=config.cluster_retrieval_mode,
                prm_enabled=config.cluster_prm_enabled,
                prm_provider=config.cluster_prm_provider,
                prm_url=config.cluster_prm_url,
                prm_model=config.cluster_prm_model,
                prm_api_key=config.cluster_prm_api_key,
                seed_openclaw_dir=config.cluster_seed_openclaw_dir,
            ),
        )
        self._prepared = False
        self._started = False

    def setup(self) -> None:
        if self._prepared:
            return
        initial_skills = (
            Path(self.config.initial_skills_dir).expanduser().resolve()
            if self.config.initial_skills_dir else None
        )
        self.cluster.prepare(initial_skills_dir=initial_skills)
        self.cluster.start_active_nodes()
        self._prepared = True
        self._started = True

    def teardown(self) -> None:
        if not self._started:
            return
        with suppress(Exception):
            self.cluster.stop_active_nodes()
        self._started = False

    def run_task(
        self,
        task: BenchmarkTask,
        *,
        device_id: str,
        device_dir: Path,
        skills_dir: Path,
        phase: str,
        round_idx: int,
    ) -> ExecutorResult:
        if not self._prepared:
            self.setup()

        started = time.perf_counter()
        invocation = self.cluster.invoke_task(
            node_id=device_id,
            instruction=task.instruction,
            round_idx=round_idx,
            phase=phase,
            task_id=task.task_id,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        expected_skills = list(task.required_skills)
        target_skill = str(task.raw.get("target_skill", "")).strip()
        if target_skill and target_skill not in expected_skills:
            expected_skills.append(target_skill)

        artifacts = self.cluster.collect_session_artifacts(
            node_id=device_id,
            session_id=invocation["session_id"],
            requested_session_id=str(invocation.get("requested_session_id", "")),
            task_id=task.task_id,
            phase=phase,
            round_idx=round_idx,
            skill_names=expected_skills,
            conversation_slice=invocation.get("conversation_slice"),
            prm_slice=invocation.get("prm_slice"),
        )

        payload = invocation.get("payload", {})
        stdout = invocation.get("stdout", "")
        stderr = invocation.get("stderr", "")
        response_text = _collect_payload_text(payload)
        success = invocation["returncode"] == 0 and not bool(payload.get("error"))
        strict_failures: list[str] = []

        if self.config.strict_require_nonempty_response and not response_text.strip():
            success = False
            strict_failures.append("empty_response_text")

        if self.config.strict_require_conversation_records:
            if int(artifacts.get("conversation_count", 0)) <= 0:
                success = False
                strict_failures.append("no_conversation_records")

        if self.config.strict_require_skill_reads_from_local_dir:
            if artifacts.get("read_skill_paths_outside_expected"):
                success = False
                strict_failures.append("skill_path_outside_expected")

        forbidden_substrings = [
            str(x).strip()
            for x in (self.config.strict_forbidden_substrings + list(task.raw.get("forbidden_substrings", []) or []))
            if str(x).strip()
        ]

        combined_text = "\n".join(
            part for part in [response_text, str(payload.get("summary", "")).strip(), stdout, stderr] if part
        ).lower()

        hit_forbidden = [token for token in forbidden_substrings if token.lower() in combined_text]
        if hit_forbidden:
            success = False
            strict_failures.append(f"forbidden_substring_hit:{' | '.join(hit_forbidden[:3])}")

        expected_substrings = [
            str(x) for x in task.raw.get("expected_substrings", []) or []
            if str(x).strip()
        ]
        if expected_substrings:
            haystack = combined_text
            success = success and all(token.lower() in haystack for token in expected_substrings)
            if not all(token.lower() in haystack for token in expected_substrings):
                strict_failures.append("missing_expected_substrings")
        notes = (
            str(response_text or payload.get("result") or payload.get("summary") or stdout or stderr or "")
            [:400]
        )

        discovered = task.discoverable_skills if phase == "train" else []
        artifacts = {
            **artifacts,
            "response_text": response_text,
            "strict_validation": {
                "enabled": bool(
                    self.config.strict_require_nonempty_response
                    or self.config.strict_require_conversation_records
                    or self.config.strict_require_skill_reads_from_local_dir
                    or bool(self.config.strict_forbidden_substrings)
                    or bool(task.raw.get("forbidden_substrings"))
                ),
                "failures": strict_failures,
                "forbidden_substrings_checked": len(forbidden_substrings),
                "expected_substrings_checked": len(expected_substrings),
            },
        }
        return ExecutorResult(
            success=success,
            used_skills=expected_skills,
            discovered_skills=list(discovered),
            latency_ms=elapsed_ms,
            notes=notes,
            artifacts=artifacts,
        )


@dataclass
class GroupBenchmarkConfig:
    name: str = "group-benchmark"
    workspace_dir: str = "records/group_benchmark"
    tasks_path: str = "examples/group_benchmark_tasks.jsonl"
    cloud_dir: str = "records/group_benchmark/cloud"
    group_id: str = "benchmark"
    initial_skills_dir: str = ""
    devices: int = 4
    eval_devices: int = 0
    rounds: int = 4
    initial_eval_enabled: bool = True
    train_tasks_per_device_per_round: int = 2
    max_parallel_devices: int = 1
    sync_enabled: bool = True
    eval_every_round: bool = True
    split_field: str = "split"
    train_split: str = "train"
    eval_split: str = "eval"
    heldout_ratio: float = 0.25
    assignment_strategy: str = "round_robin"
    eval_assignment_strategy: str = "replicate"
    device_field: str = ""
    eval_device_field: str = ""
    seed: int = 0
    executor: str = "replay"
    executor_command: list[str] = field(default_factory=list)
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
    cluster_skills_enabled: bool = True
    cluster_retrieval_mode: str = "template"
    cluster_max_context_tokens: int = 20000
    cluster_prm_enabled: bool = False
    cluster_prm_provider: str = "openai"
    cluster_prm_url: str = "https://api.openai.com/v1"
    cluster_prm_model: str = "gpt-5.2"
    cluster_prm_api_key: str = ""
    evolve_enabled: bool = False
    evolve_api_base: str = ""
    evolve_api_key: str = ""
    evolve_model: str = ""
    evolve_max_tokens: int = 4096
    evolve_debug_dir: str = ""
    evolve_aggregate_sessions: bool = True
    evolve_strategy: str = "dynamic_edit_conservative"
    evolve_use_success_feedback: bool = True
    wildclawbench_root: str = ""
    wildclawbench_output_dir: str = ""
    cloud_session_payload_mode: str = "full"
    wildclawbench_gateway_port: int = 18789
    wildclawbench_container_proxy_host: str = "host.docker.internal"
    wildclawbench_container_proxy_port: str = ""
    wildclawbench_docker_image: str = ""
    wildclawbench_docker_memory_limit: str = ""
    wildclawbench_collect_task_output: bool = True
    wildclawbench_tmp_workspace: str = "/tmp_workspace"
    wildclawbench_task_timeout_seconds_override: int = 0
    wildclawbench_success_threshold: float = 0.5
    wildclawbench_use_score_as_prm: bool = True
    strict_require_nonempty_response: bool = False
    strict_require_conversation_records: bool = False
    strict_require_skill_reads_from_local_dir: bool = False
    strict_forbidden_substrings: list[str] = field(default_factory=list)
    resume_completed_tasks: bool = False
    rollouts_per_task: int = 1
    train_round_sampling_strategy: str = "sequential"
    train_round_global_sample_size: int = 0

    @classmethod
    def from_file(cls, path: str) -> "GroupBenchmarkConfig":
        cfg_path = Path(path).expanduser().resolve()
        raw = cfg_path.read_text(encoding="utf-8")
        if cfg_path.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(raw) or {}
        else:
            data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("benchmark config must deserialize to a dict")
        base_dir = cfg_path.parent

        def _expand_env(obj: Any) -> Any:
            if isinstance(obj, str):
                return os.path.expandvars(obj)
            if isinstance(obj, list):
                return [_expand_env(item) for item in obj]
            if isinstance(obj, dict):
                return {key: _expand_env(value) for key, value in obj.items()}
            return obj

        data = _expand_env(data)

        for key in (
            "workspace_dir",
            "tasks_path",
            "cloud_dir",
            "initial_skills_dir",
            "evolve_debug_dir",
            "wildclawbench_root",
            "wildclawbench_output_dir",
        ):
            value = data.get(key)
            if isinstance(value, str) and value and not Path(value).expanduser().is_absolute():
                data[key] = str((base_dir / value).resolve())
        seed_dir = data.get("cluster_seed_openclaw_dir")
        if isinstance(seed_dir, str) and seed_dir and not Path(seed_dir).expanduser().is_absolute():
            data["cluster_seed_openclaw_dir"] = str((base_dir / seed_dir).resolve())
        return cls(**data)


@dataclass
class DeviceState:
    device_id: str
    device_dir: Path
    skills_dir: Path
    train_tasks: list[BenchmarkTask]
    cursor: int = 0


class GroupBenchmarkRunner:
    def __init__(self, config: GroupBenchmarkConfig):
        self.config = config
        self.workspace_dir = Path(config.workspace_dir).expanduser().resolve()
        self.cloud_dir = Path(config.cloud_dir).expanduser().resolve()
        self.tasks_path = Path(config.tasks_path).expanduser().resolve()
        self.initial_skills_dir = (
            Path(config.initial_skills_dir).expanduser().resolve()
            if config.initial_skills_dir
            else None
        )
        self.executor = self._build_executor(config)
        self.hub = LocalSkillHub(
            root_dir=str(self.cloud_dir),
            group_id=config.group_id,
            user_alias="benchmark",
        )
        self._train_task_pool: list[BenchmarkTask] = []

    def _build_executor(self, config: GroupBenchmarkConfig) -> TaskExecutor:
        if config.executor == "replay":
            return ReplayExecutor()
        if config.executor == "command":
            return CommandExecutor(config.executor_command)
        if config.executor == "openclaw_cluster":
            return OpenClawClusterExecutor(config)
        if config.executor == "wildclawbench_cluster":
            from .wildclawbench_executor import WildClawBenchClusterExecutor

            return WildClawBenchClusterExecutor(config)
        raise ValueError(f"unsupported executor: {config.executor}")

    def run(self) -> dict[str, Any]:
        random.seed(self.config.seed)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        train_tasks, eval_tasks = self._load_tasks()
        self._train_task_pool = list(train_tasks)
        devices = self._prepare_devices(train_tasks)

        setup = getattr(self.executor, "setup", None)
        teardown = getattr(self.executor, "teardown", None)
        if callable(setup):
            setup()

        try:
            report: dict[str, Any] = {
                "name": self.config.name,
                "workspace_dir": str(self.workspace_dir),
                "cloud_dir": str(self.cloud_dir),
                "tasks_path": str(self.tasks_path),
                "devices": [d.device_id for d in devices],
                "train_task_count": len(train_tasks),
                "eval_task_count": len(eval_tasks),
                "cluster": {
                    "configured_nodes": max(self.config.cluster_configured_nodes, self.config.devices),
                    "active_nodes": self.config.devices,
                    "executor": self.config.executor,
                },
                "strict": {
                    "require_nonempty_response": self.config.strict_require_nonempty_response,
                    "require_conversation_records": self.config.strict_require_conversation_records,
                    "require_skill_reads_from_local_dir": self.config.strict_require_skill_reads_from_local_dir,
                    "forbidden_substrings": list(self.config.strict_forbidden_substrings),
                },
                "initial_eval": (
                    self._evaluate_all(devices, eval_tasks, round_idx=0)
                    if self.config.initial_eval_enabled
                    else {}
                ),
                "rounds": [],
            }

            for round_idx in range(1, self.config.rounds + 1):
                train_summary = self._run_round_training(devices, round_idx)
                if self.config.sync_enabled or self.config.evolve_enabled:
                    push_summary = self._push_devices(devices)
                    evolve_summary = (
                        self._run_evolution(round_idx, train_summary=train_summary)
                        if self.config.evolve_enabled
                        else {}
                    )
                    pull_summary = self._pull_devices(devices)
                    sync_summary = {
                        "push": push_summary,
                        "evolution": evolve_summary,
                        "pull": pull_summary,
                        "remote_skill_count": len(self.hub.list_remote()),
                    }
                else:
                    sync_summary = {"push": {}, "pull": {}}
                eval_summary = (
                    self._evaluate_all(devices, eval_tasks, round_idx=round_idx)
                    if self.config.eval_every_round
                    else {}
                )
                round_report = {
                    "round": round_idx,
                    "train": train_summary,
                    "sync": sync_summary,
                    "eval": eval_summary,
                    "feedback": self._collect_feedback_snapshot(devices),
                }
                report["rounds"].append(round_report)
                _append_jsonl(self.workspace_dir / "rounds.jsonl", round_report)

            report["feedback"] = self._collect_feedback_snapshot(devices)
            report["summary"] = self._build_summary(report)
            report_path = self.workspace_dir / "report.json"
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            return report
        finally:
            if callable(teardown):
                teardown()

    def _load_tasks(self) -> tuple[list[BenchmarkTask], list[BenchmarkTask]]:
        if not self.tasks_path.exists():
            raise FileNotFoundError(f"tasks file not found: {self.tasks_path}")

        tasks: list[BenchmarkTask] = []
        with self.tasks_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                tasks.append(BenchmarkTask.from_dict(json.loads(line), split_field=self.config.split_field))

        split_values = {getattr(t, "split", "train") for t in tasks}
        if self.config.train_split in split_values or self.config.eval_split in split_values:
            train = [t for t in tasks if t.split == self.config.train_split]
            eval_tasks = [t for t in tasks if t.split == self.config.eval_split]
            return train, eval_tasks

        shuffled = list(tasks)
        random.Random(self.config.seed).shuffle(shuffled)
        eval_count = max(1, int(len(shuffled) * self.config.heldout_ratio))
        eval_tasks = shuffled[:eval_count]
        train = shuffled[eval_count:]
        return train, eval_tasks

    def _prepare_devices(self, train_tasks: list[BenchmarkTask]) -> list[DeviceState]:
        devices: list[DeviceState] = []
        if self._uses_round_global_sampling():
            task_buckets: list[list[BenchmarkTask]] = [[] for _ in range(self.config.devices)]
        else:
            task_buckets = [[] for _ in range(self.config.devices)]
            for idx, task in enumerate(train_tasks):
                bucket_indices = self._assign_task_targets(task, fallback_idx=idx)
                for bucket_idx in bucket_indices:
                    task_buckets[bucket_idx].append(task)

        for idx in range(self.config.devices):
            device_id = f"device-{idx}"
            device_dir = self.workspace_dir / device_id
            skills_dir = device_dir / "skills"
            _copy_tree_if_exists(self.initial_skills_dir, skills_dir)
            devices.append(
                DeviceState(
                    device_id=device_id,
                    device_dir=device_dir,
                    skills_dir=skills_dir,
                    train_tasks=task_buckets[idx],
                )
            )
        return devices

    def _uses_round_global_sampling(self) -> bool:
        strategy = str(self.config.train_round_sampling_strategy or "sequential").strip().lower()
        return (
            strategy == "random_with_replacement"
            and int(self.config.train_round_global_sample_size or 0) > 0
        )

    def _build_sampled_round_task(
        self,
        task: BenchmarkTask,
        *,
        round_idx: int,
        sample_idx: int,
    ) -> BenchmarkTask:
        base_task_id = task.task_id
        sampled_task_id = f"r{round_idx}-s{sample_idx + 1}-{_slugify(base_task_id)}"
        raw = dict(task.raw)
        raw["sampled_from_task_id"] = base_task_id
        raw["sample_round"] = round_idx
        raw["sample_index"] = sample_idx + 1
        raw["task_id"] = sampled_task_id
        return BenchmarkTask(
            task_id=sampled_task_id,
            instruction=task.instruction,
            split=task.split,
            required_skills=list(task.required_skills),
            discoverable_skills=deepcopy(task.discoverable_skills),
            raw=raw,
        )

    def _sample_train_tasks_for_round(
        self,
        devices: list[DeviceState],
        round_idx: int,
    ) -> dict[str, Any]:
        if not self._train_task_pool:
            return {
                "sampled_tasks": [],
                "assignments": {device.device_id: [] for device in devices},
                "strategy": "none",
                "sample_size": 0,
                "pool_size": 0,
                "with_replacement": False,
            }

        sample_size = max(0, int(self.config.train_round_global_sample_size or 0))
        rng = random.Random(int(self.config.seed) + int(round_idx))
        sampled_tasks: list[BenchmarkTask] = []
        for sample_idx in range(sample_size):
            source = rng.choice(self._train_task_pool)
            sampled_tasks.append(
                self._build_sampled_round_task(
                    source,
                    round_idx=round_idx,
                    sample_idx=sample_idx,
                )
            )

        assignments: dict[str, list[BenchmarkTask]] = {device.device_id: [] for device in devices}
        for idx, task in enumerate(sampled_tasks):
            if not devices:
                break
            target_device = devices[idx % len(devices)]
            assignments[target_device.device_id].append(task)

        sampled_info = [
            {
                "task_id": task.task_id,
                "base_task_id": str(task.raw.get("sampled_from_task_id") or task.task_id),
                "sample_index": int(task.raw.get("sample_index", 0) or 0),
                "device_id": devices[idx % len(devices)].device_id if devices else "",
            }
            for idx, task in enumerate(sampled_tasks)
        ]
        return {
            "sampled_tasks": sampled_info,
            "assignments": assignments,
            "strategy": "random_with_replacement",
            "sample_size": sample_size,
            "pool_size": len(self._train_task_pool),
            "with_replacement": True,
        }

    def _assign_task_targets(self, task: BenchmarkTask, *, fallback_idx: int) -> list[int]:
        if self.config.device_field:
            value = str(task.raw.get(self.config.device_field, "")).strip()
            if value:
                if value.startswith("device-"):
                    try:
                        explicit_idx = int(value.split("-", 1)[1])
                    except ValueError:
                        explicit_idx = _stable_index(value, self.config.devices)
                    return [explicit_idx % self.config.devices]
                return [_stable_index(value, self.config.devices)]

        if self.config.assignment_strategy == "round_robin":
            return [fallback_idx % self.config.devices]

        if self.config.assignment_strategy == "broadcast":
            return list(range(self.config.devices))

        raise ValueError(f"unsupported assignment strategy: {self.config.assignment_strategy}")

    def _assign_eval_task_targets(
        self,
        task: BenchmarkTask,
        *,
        fallback_idx: int,
        eval_device_count: int,
    ) -> list[int]:
        if self.config.eval_device_field:
            value = str(task.raw.get(self.config.eval_device_field, "")).strip()
            if value:
                if value.startswith("device-"):
                    try:
                        explicit_idx = int(value.split("-", 1)[1])
                    except ValueError:
                        explicit_idx = _stable_index(value, eval_device_count)
                    return [explicit_idx % eval_device_count]
                return [_stable_index(value, eval_device_count)]

        strategy = self.config.eval_assignment_strategy
        if strategy == "replicate":
            return list(range(eval_device_count))
        if strategy in {"round_robin", "distribute"}:
            return [fallback_idx % eval_device_count]
        raise ValueError(f"unsupported eval assignment strategy: {strategy}")

    def _run_round_training(self, devices: list[DeviceState], round_idx: int) -> dict[str, Any]:
        sampling_summary: dict[str, Any] = {}
        if self._uses_round_global_sampling():
            sampled = self._sample_train_tasks_for_round(devices, round_idx)
            sampling_summary = {
                "strategy": sampled["strategy"],
                "sample_size": sampled["sample_size"],
                "pool_size": sampled["pool_size"],
                "with_replacement": sampled["with_replacement"],
                "sampled_tasks": sampled["sampled_tasks"],
            }
            assignments = sampled["assignments"]
            max_workers = max(1, self.config.max_parallel_devices)
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [
                    pool.submit(
                        self._run_device_train_batch,
                        device,
                        round_idx,
                        assignments.get(device.device_id, []),
                    )
                    for device in devices
                ]
                results = [f.result() for f in futures]
        else:
            max_workers = max(1, self.config.max_parallel_devices)
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [
                    pool.submit(self._run_device_train_slice, device, round_idx)
                    for device in devices
                ]
                results = [f.result() for f in futures]

        total_tasks = sum(r["tasks"] for r in results)
        total_success = sum(r["successes"] for r in results)
        total_new_skills = sum(r["new_skills"] for r in results)
        total_score = sum(float(r.get("total_score", 0.0)) for r in results)
        scored_tasks = sum(int(r.get("scored_tasks", 0)) for r in results)
        return {
            "devices": results,
            "tasks": total_tasks,
            "successes": total_success,
            "success_rate": _safe_rate(total_success, total_tasks),
            "new_skills": total_new_skills,
            "total_score": total_score,
            "scored_tasks": scored_tasks,
            "mean_score": _safe_rate(total_score, scored_tasks),
            "sampling": sampling_summary,
        }

    def _run_device_train_slice(self, device: DeviceState, round_idx: int) -> dict[str, Any]:
        start = device.cursor
        stop = min(
            len(device.train_tasks),
            start + self.config.train_tasks_per_device_per_round,
        )
        batch = device.train_tasks[start:stop]
        device.cursor = stop
        return self._run_device_train_batch(device, round_idx, batch)

    def _run_device_train_batch(
        self,
        device: DeviceState,
        round_idx: int,
        batch: list[BenchmarkTask],
    ) -> dict[str, Any]:
        device.device_dir.mkdir(parents=True, exist_ok=True)
        device.skills_dir.mkdir(parents=True, exist_ok=True)
        manager = SkillManager(str(device.skills_dir), retrieval_mode="template")

        successes = 0
        new_skills = 0
        run_records = []
        scores: list[float] = []
        existing_records = self._load_existing_phase_records(
            device.device_dir / "train_runs.jsonl",
            phase="train",
            round_idx=round_idx,
        )
        rollouts = max(1, self.config.rollouts_per_task)

        for task in batch:
            for rollout_idx in range(rollouts):
                key = self._record_key(task.task_id, rollout_idx)
                existing = existing_records.get(key)
                if existing is not None:
                    run_records.append(existing)
                    if bool(existing.get("success")):
                        successes += 1
                    if isinstance(existing.get("score"), (int, float)):
                        scores.append(float(existing["score"]))
                    continue
                result = self.executor.run_task(
                    task,
                    device_id=device.device_id,
                    device_dir=device.device_dir,
                    skills_dir=device.skills_dir,
                    phase="train",
                    round_idx=round_idx,
                )
                if result.used_skills:
                    manager.record_injection(result.used_skills)
                    manager.record_feedback(result.used_skills, 1.0 if result.success else -1.0)

                if result.discovered_skills:
                    new_skills += manager.add_skills(result.discovered_skills)

                if result.success:
                    successes += 1

                record = {
                    "round": round_idx,
                    "phase": "train",
                    "device_id": device.device_id,
                    "task_id": task.task_id,
                    "base_task_id": str(task.raw.get("sampled_from_task_id") or task.task_id),
                    "rollout_idx": rollout_idx,
                    "success": result.success,
                    "used_skills": result.used_skills,
                    "discovered_skills": [s["name"] for s in result.discovered_skills],
                    "latency_ms": result.latency_ms,
                    "notes": result.notes,
                    "score": result.score,
                    "score_breakdown": result.score_breakdown,
                    "artifacts": result.artifacts,
                }
                run_records.append(record)
                _append_jsonl(device.device_dir / "train_runs.jsonl", record)
                if isinstance(result.score, (int, float)):
                    scores.append(float(result.score))

        manager._save_stats()
        total_task_runs = len(batch) * rollouts
        return {
            "device_id": device.device_id,
            "tasks": total_task_runs,
            "successes": successes,
            "success_rate": _safe_rate(successes, total_task_runs),
            "new_skills": new_skills,
            "local_skill_count": manager.get_skill_count(),
            "total_score": sum(scores),
            "scored_tasks": len(scores),
            "mean_score": _mean_score(scores),
            "records": run_records,
        }

    def _push_devices(self, devices: list[DeviceState]) -> list[dict[str, Any]]:
        push_results = []
        for device in devices:
            hub = LocalSkillHub(
                root_dir=str(self.cloud_dir),
                group_id=self.config.group_id,
                user_alias=device.device_id,
            )
            res = hub.push_skills(str(device.skills_dir))
            push_results.append({"device_id": device.device_id, **res})
        return push_results

    def _pull_devices(self, devices: list[DeviceState]) -> list[dict[str, Any]]:
        pull_results = []
        for device in devices:
            hub = LocalSkillHub(
                root_dir=str(self.cloud_dir),
                group_id=self.config.group_id,
                user_alias=device.device_id,
            )
            res = hub.pull_skills(str(device.skills_dir))
            pull_results.append({"device_id": device.device_id, **res})
        return pull_results

    def _sync_devices(self, devices: list[DeviceState]) -> dict[str, Any]:
        push_results = self._push_devices(devices)
        pull_results = self._pull_devices(devices)
        remote = self.hub.list_remote()
        return {"push": push_results, "pull": pull_results, "remote_skill_count": len(remote)}

    def _run_evolution(self, round_idx: int, *, train_summary: dict[str, Any]) -> dict[str, Any]:
        from evolve_server.config import EvolveServerConfig
        from evolve_server.server import EvolveServer

        evolve_dir = self.workspace_dir / "evolve"
        evolve_dir.mkdir(parents=True, exist_ok=True)
        aggregate_info = self._prepare_evolution_backend(round_idx, train_summary=train_summary)
        config = EvolveServerConfig(
            group_id=self.config.group_id,
            local_root=str(aggregate_info["backend_root"]),
            llm_api_key=self.config.evolve_api_key or self.config.cluster_llm_api_key,
            llm_base_url=self.config.evolve_api_base or self.config.cluster_llm_api_base,
            llm_model=self.config.evolve_model or self.config.cluster_llm_model_id or "gpt-5.2",
            llm_max_tokens=self.config.evolve_max_tokens,
            use_success_feedback=self.config.evolve_use_success_feedback,
            debug_dump_dir=(
                self.config.evolve_debug_dir
                or str((self.workspace_dir / "evolve_debug").resolve())
            ),
            processed_log_path=str(Path(aggregate_info["round_dir"]) / "evolve_processed.json"),
            history_path=str(Path(aggregate_info["round_dir"]) / "evolve_history.jsonl"),
        )
        summary = asyncio.run(EvolveServer(config).run_once())
        self._sync_evolution_backend_to_cloud(Path(aggregate_info["backend_root"]))
        return {
            "round": round_idx,
            "summary": summary,
            "manifest_path": str(self.cloud_dir / self.config.group_id / "manifest.jsonl"),
            "registry_path": str(self.cloud_dir / self.config.group_id / "evolve_skill_registry.json"),
            "aggregation": {
                "enabled": self.config.evolve_aggregate_sessions,
                "backend_root": str(aggregate_info["backend_root"]),
                "source_session_count": aggregate_info["source_session_count"],
                "aggregated_session_count": aggregate_info["aggregated_session_count"],
                "aggregated_session_paths": aggregate_info["aggregated_session_paths"],
            },
        }

    def _prepare_evolution_backend(self, round_idx: int, *, train_summary: dict[str, Any]) -> dict[str, Any]:
        round_dir = self.workspace_dir / "evolve" / f"round-{round_idx}"
        backend_root = round_dir / "backend"
        group_dir = backend_root / self.config.group_id
        sessions_dir = group_dir / "sessions"
        skills_dir = group_dir / "skills"
        round_dir.mkdir(parents=True, exist_ok=True)
        if backend_root.exists():
            shutil.rmtree(backend_root)
        sessions_dir.mkdir(parents=True, exist_ok=True)
        skills_dir.mkdir(parents=True, exist_ok=True)

        source_group = self.cloud_dir / self.config.group_id
        if (source_group / "skills").exists():
            shutil.copytree(source_group / "skills", skills_dir, dirs_exist_ok=True)
        for name in ("manifest.jsonl", "evolve_skill_registry.json"):
            src = source_group / name
            if src.exists():
                shutil.copy2(src, group_dir / name)

        source_session_paths: list[Path] = []
        for device in train_summary.get("devices", []):
            for record in device.get("records", []):
                raw_path = str(((record.get("artifacts") or {}).get("session_path")) or "").strip()
                if raw_path:
                    source_session_paths.append(Path(raw_path))

        aggregated_paths: list[str] = []
        if self.config.evolve_aggregate_sessions:
            aggregated_payloads = self._build_aggregated_sessions(round_idx, train_summary=train_summary)
            for payload in aggregated_payloads:
                path = sessions_dir / f"{payload['session_id']}.json"
                path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                aggregated_paths.append(str(path))
        else:
            for src in source_session_paths:
                if src.exists():
                    shutil.copy2(src, sessions_dir / src.name)
                    aggregated_paths.append(str(sessions_dir / src.name))

        return {
            "round_dir": str(round_dir),
            "backend_root": str(backend_root),
            "source_session_count": len(source_session_paths),
            "aggregated_session_count": len(aggregated_paths),
            "aggregated_session_paths": aggregated_paths,
        }

    def _build_aggregated_sessions(self, round_idx: int, *, train_summary: dict[str, Any]) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for device in train_summary.get("devices", []):
            for record in device.get("records", []):
                grouped.setdefault(str(record.get("task_id") or "unknown"), []).append(record)

        payloads: list[dict[str, Any]] = []
        for task_id, records in grouped.items():
            session_id = f"agg-r{round_idx}-{_slugify(task_id)}"
            scores = [
                float(rec["score"])
                for rec in records
                if isinstance(rec.get("score"), (int, float))
            ]
            mean_score = _mean_score(scores)
            best_record = max(
                records,
                key=lambda rec: (
                    float(rec.get("score", 0.0)) if isinstance(rec.get("score"), (int, float)) else -1.0,
                    1 if rec.get("success") else 0,
                ),
            )
            prompt_text = self._infer_aggregate_prompt(records)
            source_sessions = [self._load_source_session(rec) for rec in records]
            read_skill_names = sorted(
                {
                    name
                    for rec, session in zip(records, source_sessions)
                    for name in (
                        self._collect_session_skill_names(session)
                        or [str(raw) for raw in rec.get("used_skills", []) or [] if str(raw).strip()]
                    )
                    if str(name).strip()
                }
            )

            turns: list[dict[str, Any]] = []
            has_source_turns = any(bool(sess.get("turns")) for sess in source_sessions)
            if has_source_turns and len(records) > 1:
                for rollout_idx, (rec, source_session) in enumerate(zip(records, source_sessions)):
                    rollout_score = rec.get("score")
                    rollout_success = rec.get("success")
                    source_turns = source_session.get("turns") or []
                    if source_turns:
                        for source_turn in source_turns:
                            turn = dict(source_turn)
                            normalized_read_skills = []
                            for item in turn.get("read_skills", []) or []:
                                if isinstance(item, dict):
                                    normalized_read_skills.append(
                                        {
                                            "skill_id": str(item.get("skill_id") or ""),
                                            "skill_name": str(item.get("skill_name") or ""),
                                        }
                                    )
                                else:
                                    normalized_read_skills.append(
                                        {"skill_id": "", "skill_name": str(item).strip()}
                                    )
                            if normalized_read_skills:
                                turn["read_skills"] = normalized_read_skills
                            turn["turn_num"] = len(turns) + 1
                            turn["_rollout_idx"] = rollout_idx
                            turn["_rollout_score"] = rollout_score
                            turn["_rollout_success"] = rollout_success
                            turns.append(turn)
                    else:
                        score_text = (
                            f"{float(rollout_score):.3f}"
                            if isinstance(rollout_score, (int, float))
                            else "n/a"
                        )
                        turns.append(
                            {
                                "turn_num": len(turns) + 1,
                                "prompt_text": prompt_text,
                                "response_text": f"Rollout {rollout_idx}: score={score_text}, success={bool(rollout_success)}",
                                "tool_calls": [],
                                "tool_results": [],
                                "tool_observations": [],
                                "tool_errors": [],
                                "read_skills": [{"skill_id": "", "skill_name": name} for name in read_skill_names],
                                "injected_skills": [],
                                "prm_score": rollout_score if isinstance(rollout_score, (int, float)) else None,
                                "_rollout_idx": rollout_idx,
                                "_rollout_score": rollout_score,
                                "_rollout_success": rollout_success,
                            }
                        )
            else:
                turns = [
                    {
                        "turn_num": 1,
                        "prompt_text": prompt_text,
                        "response_text": self._build_aggregate_response(task_id, records),
                        "tool_calls": [],
                        "tool_results": [],
                        "tool_observations": [],
                        "tool_errors": [],
                        "read_skills": [{"skill_id": "", "skill_name": name} for name in read_skill_names],
                        "injected_skills": [],
                        "prm_score": mean_score if scores else None,
                    }
                ]

            success_count = sum(1 for rec in records if rec.get("success"))
            fail_count = len(records) - success_count
            score_std = 0.0
            if len(scores) > 1:
                avg = sum(scores) / len(scores)
                score_std = (sum((score - avg) ** 2 for score in scores) / len(scores)) ** 0.5
            payloads.append(
                {
                    "session_id": session_id,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "user_alias": "aggregator",
                    "num_turns": len(turns),
                    "task_id": task_id,
                    "phase": "aggregate",
                    "round": round_idx,
                    "turns": turns,
                    "aggregate": {
                        "task_id": task_id,
                        "device_count": len(records),
                        "rollout_count": len(records),
                        "source_session_paths": [
                            str(((rec.get("artifacts") or {}).get("session_path")) or "")
                            for rec in records
                        ],
                        "scores": scores,
                        "mean_score": mean_score if scores else None,
                        "score_std": round(score_std, 4),
                        "success_count": success_count,
                        "fail_count": fail_count,
                        "stability": (
                            "all_success"
                            if fail_count == 0
                            else "all_fail" if success_count == 0 else "unstable"
                        ),
                        "best_device_id": best_record.get("device_id"),
                    },
                }
            )
        return payloads

    def _load_source_session(self, record: dict[str, Any]) -> dict[str, Any]:
        raw_path = str(((record.get("artifacts") or {}).get("session_path")) or "").strip()
        if not raw_path:
            return {}
        path = Path(raw_path)
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _collect_session_skill_names(self, session_payload: dict[str, Any]) -> list[str]:
        turns = session_payload.get("turns") or []
        names: set[str] = set()
        if not isinstance(turns, list):
            return []
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            for item in turn.get("read_skills", []) or []:
                if isinstance(item, dict):
                    name = str(item.get("skill_name") or "").strip()
                else:
                    name = str(item).strip()
                if name:
                    names.add(name)
        return sorted(names)

    def _infer_aggregate_prompt(self, records: list[dict[str, Any]]) -> str:
        for rec in records:
            artifacts = rec.get("artifacts") or {}
            wildbench = artifacts.get("wildclawbench") or {}
            task_file = str(wildbench.get("task_file") or "").strip()
            if not task_file:
                continue
            path = Path(task_file)
            if not path.exists():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except Exception:
                continue
            match = re.search(r"## Prompt\s*\n(.*?)(?:\n## |\Z)", text, re.DOTALL)
            if match:
                return match.group(1).strip()
        return records[0].get("task_id", "benchmark task")

    def _build_aggregate_response(self, task_id: str, records: list[dict[str, Any]]) -> str:
        lines = [f"Aggregated benchmark summary for task `{task_id}` ({len(records)} rollout(s)).", ""]
        for idx, rec in enumerate(records):
            score = rec.get("score")
            score_text = f"{float(score):.3f}" if isinstance(score, (int, float)) else "n/a"
            rollout_label = f"Rollout {rec.get('rollout_idx', idx)}"
            lines.append(
                f"{rollout_label} (device {rec.get('device_id', '?')}): success={bool(rec.get('success'))} score={score_text}"
            )
            used_skills = [str(name) for name in rec.get("used_skills", []) or [] if str(name).strip()]
            if used_skills:
                lines.append(f"  Used skills: {', '.join(used_skills)}")
            notes = str(rec.get("notes") or "").strip()
            if notes:
                lines.append(f"  Notes: {notes[:800]}")
            score_breakdown = rec.get("score_breakdown") or {}
            if isinstance(score_breakdown, dict) and score_breakdown:
                breakdown = ", ".join(
                    f"{key}={value}"
                    for key, value in score_breakdown.items()
                    if key != "mode"
                )
                if breakdown:
                    lines.append(f"  Score breakdown: {breakdown}")
            lines.append("")

        if len(records) > 1:
            scores = [float(rec["score"]) for rec in records if isinstance(rec.get("score"), (int, float))]
            success_count = sum(1 for rec in records if rec.get("success"))
            fail_count = len(records) - success_count
            lines.append("--- Cross-rollout analysis ---")
            if fail_count == 0:
                lines.append(f"Stability: ALL {len(records)} rollouts SUCCEEDED")
            elif success_count == 0:
                lines.append(f"Stability: ALL {len(records)} rollouts FAILED")
            else:
                lines.append(f"Stability: UNSTABLE — {success_count} succeeded, {fail_count} failed")
            if scores:
                avg = sum(scores) / len(scores)
                score_std = (sum((score - avg) ** 2 for score in scores) / len(scores)) ** 0.5
                lines.append(
                    f"Scores: min={min(scores):.3f} max={max(scores):.3f} mean={avg:.3f} std={score_std:.3f}"
                )
            lines.append("")
        return "\n".join(lines).strip()

    def _sync_evolution_backend_to_cloud(self, backend_root: Path) -> None:
        src_group = backend_root / self.config.group_id
        dst_group = self.cloud_dir / self.config.group_id
        dst_group.mkdir(parents=True, exist_ok=True)
        for name in ("skills", "manifest.jsonl", "evolve_skill_registry.json"):
            src = src_group / name
            dst = dst_group / name
            if not src.exists():
                continue
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)

    def _evaluate_all(
        self,
        devices: list[DeviceState],
        eval_tasks: list[BenchmarkTask],
        *,
        round_idx: int,
    ) -> dict[str, Any]:
        eval_device_count = self.config.eval_devices or len(devices)
        eval_targets = devices[: max(0, min(len(devices), eval_device_count))]
        if self.config.eval_assignment_strategy == "distribute" and not self.config.eval_device_field:
            return self._evaluate_all_distributed(eval_targets, eval_tasks, round_idx=round_idx)
        eval_task_buckets: list[list[BenchmarkTask]] = [[] for _ in eval_targets]
        for idx, task in enumerate(eval_tasks):
            for target_idx in self._assign_eval_task_targets(
                task,
                fallback_idx=idx,
                eval_device_count=len(eval_targets),
            ):
                if 0 <= target_idx < len(eval_targets):
                    eval_task_buckets[target_idx].append(task)
        max_workers = max(1, self.config.max_parallel_devices)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(self._evaluate_device, device, eval_task_buckets[idx], round_idx)
                for idx, device in enumerate(eval_targets)
            ]
            results = [f.result() for f in futures]

        total_tasks = sum(r["tasks"] for r in results)
        total_success = sum(r["successes"] for r in results)
        total_score = sum(float(r.get("total_score", 0.0)) for r in results)
        scored_tasks = sum(int(r.get("scored_tasks", 0)) for r in results)
        return {
            "devices": results,
            "eval_devices": len(eval_targets),
            "tasks": total_tasks,
            "successes": total_success,
            "success_rate": _safe_rate(total_success, total_tasks),
            "total_score": total_score,
            "scored_tasks": scored_tasks,
            "mean_score": _safe_rate(total_score, scored_tasks),
        }

    def _evaluate_all_distributed(
        self,
        eval_targets: list[DeviceState],
        eval_tasks: list[BenchmarkTask],
        *,
        round_idx: int,
    ) -> dict[str, Any]:
        existing_records = self._load_existing_phase_records_across_devices(
            eval_targets,
            phase="eval",
            round_idx=round_idx,
        )
        rollouts = max(1, self.config.rollouts_per_task)
        task_queue: queue.Queue[tuple[BenchmarkTask, int]] = queue.Queue()
        for task in eval_tasks:
            for rollout_idx in range(rollouts):
                key = self._record_key(task.task_id, rollout_idx)
                if key not in existing_records:
                    task_queue.put((task, rollout_idx))

        max_workers = max(1, min(self.config.max_parallel_devices, len(eval_targets)))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(
                    self._evaluate_device_distributed,
                    device,
                    task_queue,
                    round_idx,
                    self._records_for_device(existing_records.values(), device.device_id),
                )
                for device in eval_targets
            ]
            results = [f.result() for f in futures]

        total_tasks = sum(r["tasks"] for r in results)
        total_success = sum(r["successes"] for r in results)
        total_score = sum(float(r.get("total_score", 0.0)) for r in results)
        scored_tasks = sum(int(r.get("scored_tasks", 0)) for r in results)
        return {
            "devices": results,
            "eval_devices": len(eval_targets),
            "tasks": total_tasks,
            "successes": total_success,
            "success_rate": _safe_rate(total_success, total_tasks),
            "total_score": total_score,
            "scored_tasks": scored_tasks,
            "mean_score": _safe_rate(total_score, scored_tasks),
        }

    def _evaluate_device_distributed(
        self,
        device: DeviceState,
        task_queue: "queue.Queue[tuple[BenchmarkTask, int]]",
        round_idx: int,
        existing_records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        successes = 0
        records = list(existing_records)
        scores: list[float] = []
        for record in existing_records:
            if bool(record.get("success")):
                successes += 1
            if isinstance(record.get("score"), (int, float)):
                scores.append(float(record["score"]))

        while True:
            try:
                task, rollout_idx = task_queue.get_nowait()
            except queue.Empty:
                break
            result = self.executor.run_task(
                task,
                device_id=device.device_id,
                device_dir=device.device_dir,
                skills_dir=device.skills_dir,
                phase="eval",
                round_idx=round_idx,
            )
            if result.success:
                successes += 1
            record = {
                "round": round_idx,
                "phase": "eval",
                "device_id": device.device_id,
                "task_id": task.task_id,
                "rollout_idx": rollout_idx,
                "success": result.success,
                "used_skills": result.used_skills,
                "latency_ms": result.latency_ms,
                "notes": result.notes,
                "score": result.score,
                "score_breakdown": result.score_breakdown,
                "artifacts": result.artifacts,
            }
            records.append(record)
            _append_jsonl(device.device_dir / "eval_runs.jsonl", record)
            if isinstance(result.score, (int, float)):
                scores.append(float(result.score))

        return {
            "device_id": device.device_id,
            "tasks": len(records),
            "successes": successes,
            "success_rate": _safe_rate(successes, len(records)),
            "total_score": sum(scores),
            "scored_tasks": len(scores),
            "mean_score": _mean_score(scores),
            "records": records,
        }

    def _evaluate_device(
        self,
        device: DeviceState,
        eval_tasks: list[BenchmarkTask],
        round_idx: int,
    ) -> dict[str, Any]:
        successes = 0
        records = []
        scores: list[float] = []
        existing_records = self._load_existing_phase_records(
            device.device_dir / "eval_runs.jsonl",
            phase="eval",
            round_idx=round_idx,
        )
        rollouts = max(1, self.config.rollouts_per_task)
        for task in eval_tasks:
            for rollout_idx in range(rollouts):
                key = self._record_key(task.task_id, rollout_idx)
                existing = existing_records.get(key)
                if existing is not None:
                    records.append(existing)
                    if bool(existing.get("success")):
                        successes += 1
                    if isinstance(existing.get("score"), (int, float)):
                        scores.append(float(existing["score"]))
                    continue
                result = self.executor.run_task(
                    task,
                    device_id=device.device_id,
                    device_dir=device.device_dir,
                    skills_dir=device.skills_dir,
                    phase="eval",
                    round_idx=round_idx,
                )
                if result.success:
                    successes += 1
                record = {
                    "round": round_idx,
                    "phase": "eval",
                    "device_id": device.device_id,
                    "task_id": task.task_id,
                    "rollout_idx": rollout_idx,
                    "success": result.success,
                    "used_skills": result.used_skills,
                    "latency_ms": result.latency_ms,
                    "notes": result.notes,
                    "score": result.score,
                    "score_breakdown": result.score_breakdown,
                    "artifacts": result.artifacts,
                }
                records.append(record)
                _append_jsonl(device.device_dir / "eval_runs.jsonl", record)
                if isinstance(result.score, (int, float)):
                    scores.append(float(result.score))

        total_task_runs = len(eval_tasks) * rollouts
        return {
            "device_id": device.device_id,
            "tasks": total_task_runs,
            "successes": successes,
            "success_rate": _safe_rate(successes, total_task_runs),
            "total_score": sum(scores),
            "scored_tasks": len(scores),
            "mean_score": _mean_score(scores),
            "records": records,
        }

    @staticmethod
    def _record_key(task_id: str, rollout_idx: int = 0) -> str:
        return f"{task_id}::{rollout_idx}"

    def _load_existing_phase_records(
        self,
        path: Path,
        *,
        phase: str,
        round_idx: int,
    ) -> dict[str, dict[str, Any]]:
        if not self.config.resume_completed_tasks or not path.exists():
            return {}
        records: dict[str, dict[str, Any]] = {}
        for payload in _load_jsonl(path):
            if str(payload.get("phase") or "") != phase:
                continue
            if int(payload.get("round", -1)) != int(round_idx):
                continue
            task_id = str(payload.get("task_id") or "").strip()
            if not task_id:
                continue
            rollout_idx = int(payload.get("rollout_idx", 0))
            records[self._record_key(task_id, rollout_idx)] = payload
        return records

    def _load_existing_phase_records_across_devices(
        self,
        devices: list[DeviceState],
        *,
        phase: str,
        round_idx: int,
    ) -> dict[str, dict[str, Any]]:
        if not self.config.resume_completed_tasks:
            return {}
        records: dict[str, dict[str, Any]] = {}
        filename = "eval_runs.jsonl" if phase == "eval" else "train_runs.jsonl"
        for device in devices:
            path = device.device_dir / filename
            for key, payload in self._load_existing_phase_records(path, phase=phase, round_idx=round_idx).items():
                records[key] = payload
        return records

    @staticmethod
    def _records_for_device(records: Any, device_id: str) -> list[dict[str, Any]]:
        return [
            dict(record)
            for record in records
            if str(record.get("device_id") or "") == device_id
        ]

    @staticmethod
    def _build_summary(report: dict[str, Any]) -> dict[str, Any]:
        initial_rate = float(report.get("initial_eval", {}).get("success_rate", 0.0))
        initial_score = float(report.get("initial_eval", {}).get("mean_score", 0.0))
        final_rate = initial_rate
        final_score = initial_score
        if report.get("rounds"):
            final_rate = float(report["rounds"][-1].get("eval", {}).get("success_rate", initial_rate))
            final_score = float(report["rounds"][-1].get("eval", {}).get("mean_score", initial_score))
        return {
            "initial_eval_success_rate": initial_rate,
            "final_eval_success_rate": final_rate,
            "absolute_gain": final_rate - initial_rate,
            "initial_eval_mean_score": initial_score,
            "final_eval_mean_score": final_score,
            "mean_score_gain": final_score - initial_score,
            "num_rounds": len(report.get("rounds", [])),
        }

    def _collect_feedback_snapshot(self, devices: list[DeviceState]) -> dict[str, Any]:
        per_device = []
        per_skill: dict[str, dict[str, Any]] = {}

        for device in devices:
            stats_path = device.skills_dir / "skill_stats.json"
            skill_stats = _load_json(stats_path)
            convo_path = device.device_dir / "records" / "conversations.jsonl"
            prm_path = device.device_dir / "records" / "prm_scores.jsonl"
            conversations = _load_jsonl(convo_path)
            prm_scores = _load_jsonl(prm_path)

            per_device.append(
                {
                    "device_id": device.device_id,
                    "conversation_count": len(conversations),
                    "prm_count": len(prm_scores),
                    "skill_stats": skill_stats,
                }
            )
            for skill_name, entry in skill_stats.items():
                info = per_skill.setdefault(
                    skill_name,
                    {
                        "devices": {},
                        "total_injections": 0,
                        "total_positive": 0,
                        "total_negative": 0,
                    },
                )
                info["devices"][device.device_id] = entry
                info["total_injections"] += int(entry.get("inject_count", 0))
                info["total_positive"] += int(entry.get("positive_count", 0))
                info["total_negative"] += int(entry.get("negative_count", 0))

        return {
            "devices": per_device,
            "skills": per_skill,
            "session_upload_dir": str(self.cloud_dir / self.config.group_id / "sessions"),
        }


def run_benchmark_from_config(config_path: str) -> dict[str, Any]:
    config = GroupBenchmarkConfig.from_file(config_path)
    runner = GroupBenchmarkRunner(config)
    return runner.run()
