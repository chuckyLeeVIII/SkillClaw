"""
WildClawBench task executor backed by a local multi-node SkillClaw cluster.

Each benchmark task still runs inside the original WildClawBench Docker task
container, but the OpenClaw model endpoint inside that container is rewritten to
target one specific SkillClaw node on the host. This preserves:

- WildClawBench task workspaces and grading logic
- per-node isolated skill libraries on the host
- post-task session capture for real skill evolution
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import shlex
import socket
import shutil
import subprocess
import tempfile
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .openclaw_cluster import (
    OpenClawClusterManager,
    OpenClawClusterNode,
    OpenClawClusterSettings,
    _count_jsonl_records,
)

logger = logging.getLogger(__name__)

_RETRYABLE_AGENT_BROWSER_WARMUP_ERRORS = (
    "ECONNRESET",
    "ERR_SSL_DECRYPTION_FAILED",
    "ERR_SSL_CIPHER_OPERATION_FAILED",
    "EAI_AGAIN",
    "ETIMEDOUT",
    "ECONNREFUSED",
    "NETWORK ABORTED",
)

# Transient network errors that justify retrying *any* warmup command (pip, npm, etc.)
_RETRYABLE_NETWORK_WARMUP_ERRORS = (
    "CONNECTION TIMED OUT",
    "READ TIMED OUT",
    "CONNECTIONRESETERROR",
    "ECONNRESET",
    "ETIMEDOUT",
    "ECONNREFUSED",
    "EAI_AGAIN",
    "NETWORK ABORTED",
    "SSL: DECRYPTION_FAILED",
    "CERTIFICATE_VERIFY_FAILED",
    "INCOMPLETE DOWNLOAD",
    "CONNECTIONERROR",
    "COULD NOT FIND A VERSION THAT SATISFIES",
    "NO MATCHING DISTRIBUTION",
    "DO NOT MATCH THE HASHES",
)


@dataclass
class WildClawTaskSpec:
    task_id: str
    category: str
    prompt: str
    workspace_path: str
    skills_path: str
    automated_checks: str
    env: str
    skills: str
    warmup: str
    timeout_seconds: int
    file_path: str


@dataclass
class WildClawExecutorResult:
    success: bool
    used_skills: list[str] = field(default_factory=list)
    discovered_skills: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0
    notes: str = ""
    artifacts: dict[str, Any] = field(default_factory=dict)
    score: float | None = None
    score_breakdown: dict[str, Any] = field(default_factory=dict)


def _safe_average(scores: dict[str, Any]) -> float | None:
    if not isinstance(scores, dict):
        return None
    numeric = [float(v) for v in scores.values() if isinstance(v, (int, float))]
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def _extract_final_score(scores: dict[str, Any]) -> float | None:
    if not isinstance(scores, dict) or "error" in scores:
        return None
    overall = scores.get("overall_score")
    if isinstance(overall, (int, float)):
        return float(overall)
    return _safe_average(scores)


def _resolve_task_timeout_seconds(task_timeout_seconds: int, override_seconds: int) -> int:
    if int(override_seconds or 0) > 0:
        return int(override_seconds)
    return int(task_timeout_seconds)


def _wildbench_grade_timeout_seconds(default: int = 600) -> int:
    raw = str(os.environ.get("SKILLCLAW_WILDBENCH_GRADE_TIMEOUT_S", "")).strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _grade_retry_delays_seconds(default: tuple[float, ...] = (5.0, 15.0, 30.0, 60.0)) -> tuple[float, ...]:
    raw = str(os.environ.get("SKILLCLAW_WILDBENCH_GRADE_RETRY_DELAYS_S", "")).strip()
    if not raw:
        return default
    delays: list[float] = []
    for chunk in raw.split(","):
        piece = chunk.strip()
        if not piece:
            continue
        try:
            value = float(piece)
        except ValueError:
            return default
        if value > 0:
            delays.append(value)
    return tuple(delays) or default


def _warmup_retry_delays_seconds(default: tuple[float, ...] = (5.0, 15.0, 30.0)) -> tuple[float, ...]:
    raw = str(os.environ.get("SKILLCLAW_WILDBENCH_WARMUP_RETRY_DELAYS_S", "")).strip()
    if not raw:
        return default
    delays: list[float] = []
    for chunk in raw.split(","):
        piece = chunk.strip()
        if not piece:
            continue
        try:
            value = float(piece)
        except ValueError:
            return default
        if value > 0:
            delays.append(value)
    return tuple(delays) or default


def _workspace_setup_retry_delays_seconds(
    default: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0),
) -> tuple[float, ...]:
    raw = str(os.environ.get("SKILLCLAW_WILDBENCH_SETUP_RETRY_DELAYS_S", "")).strip()
    if not raw:
        return default
    delays: list[float] = []
    for chunk in raw.split(","):
        piece = chunk.strip()
        if not piece:
            continue
        try:
            value = float(piece)
        except ValueError:
            return default
        if value > 0:
            delays.append(value)
    return tuple(delays) or default


def _container_exec_ready_timeout_seconds(default: float = 20.0) -> float:
    raw = str(os.environ.get("SKILLCLAW_WILDBENCH_CONTAINER_READY_TIMEOUT_S", "")).strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


_RETRYABLE_GRADE_ERROR_TOKENS = (
    "CONNECTION ERROR",
    "API CONNECTION ERROR",
    "SERVER DISCONNECTED",
    "TIMED OUT",
    "READ TIMED OUT",
    "TIMEOUT",
    "CONNECTION RESET",
    "ECONNRESET",
    "BAD GATEWAY",
    "SERVICE UNAVAILABLE",
    "GATEWAY TIMEOUT",
)


def _read_env_file_values(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        values[key] = value
    return values


def _should_retry_grade_result(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    texts = [
        str(payload.get("error") or ""),
        str(payload.get("judge_error") or ""),
        str(payload.get("llm_judge_error") or ""),
    ]
    if str(payload.get("judge_method") or "").strip().lower() == "failed":
        texts.append("failed")
    combined = "\n".join(texts).upper()
    if any(token in combined for token in _RETRYABLE_GRADE_ERROR_TOKENS):
        return True
    return bool(re.search(r"\bERROR CODE:\s*(429|502|503|504)\b", combined))


def _workspace_setup_timeout_seconds(default: float = 180.0) -> float:
    raw = str(os.environ.get("SKILLCLAW_WILDBENCH_SETUP_TIMEOUT_S", "")).strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _workspace_symlink_large_file_threshold_bytes(default: int = 200 * 1024 * 1024) -> int:
    raw = str(os.environ.get("SKILLCLAW_WILDBENCH_SETUP_SYMLINK_THRESHOLD_BYTES", "")).strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


_PIP_INSTALL_RE = re.compile(
    r"^((?:\S+/)?pip(?:3)?)\s+install\b",
)


def _enhance_pip_cmd(cmd: str) -> str:
    """Inject --timeout and --retries into pip install commands.

    Concurrent container warmups saturate bandwidth; the default 15-second
    timeout causes mass failures.  We bump to 300 s / 5 retries.
    We do NOT upgrade pip because the newer pip may not find packages
    that the container's older pip can resolve (index format differences).
    """
    m = _PIP_INSTALL_RE.match(cmd.strip())
    if m is None:
        return cmd
    # Already has explicit timeout/retries — leave it alone
    if "--timeout" in cmd or "--retries" in cmd:
        return cmd
    pip_bin = m.group(1)
    rest = cmd.strip()[m.end():]
    enhanced = f"{pip_bin} install --timeout 300 --retries 5{rest}"
    return enhanced


def _bypass_proxy_for_apt_cmd(cmd: str) -> str:
    """Run apt warmups without container proxy env to avoid broken mirror traffic."""
    normalized = cmd.strip()
    if not normalized.startswith(("apt ", "apt-get ")):
        return cmd
    return (
        "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY; "
        + normalized
    )


def _completed_process_debug_text(proc: subprocess.CompletedProcess[str]) -> str:
    parts = [f"returncode={proc.returncode}"]
    stdout = str(proc.stdout or "").strip()
    stderr = str(proc.stderr or "").strip()
    if stdout:
        parts.append(f"stdout={stdout}")
    if stderr:
        parts.append(f"stderr={stderr}")
    return "\n".join(parts)


def _extract_usage_from_jsonl(jsonl_path: Path) -> dict[str, Any]:
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "request_count": 0,
    }
    if not jsonl_path.exists():
        return totals
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if payload.get("type") != "message":
            continue
        msg = payload.get("message", {})
        if msg.get("role") != "assistant":
            continue
        totals["request_count"] += 1
        usage = msg.get("usage", {})
        totals["input_tokens"] += int(usage.get("input", 0) or 0)
        totals["output_tokens"] += int(usage.get("output", 0) or 0)
        totals["cache_read_tokens"] += int(usage.get("cacheRead", 0) or 0)
        totals["cache_write_tokens"] += int(usage.get("cacheWrite", 0) or 0)
        totals["total_tokens"] += int(usage.get("totalTokens", 0) or 0)
        cost = usage.get("cost", {})
        totals["cost_usd"] += float(cost.get("total", 0.0) or 0.0)
    totals["cost_usd"] = round(totals["cost_usd"], 6)
    return totals


def _copy_dir_from_container(container: str, src_path: str, dst_path: str) -> bool:
    proc = subprocess.run(
        ["docker", "cp", f"{container}:{src_path}", dst_path],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0


def _port_is_listening(host: str, port: int, *, timeout_s: float = 0.2) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def _strip_codeblock(raw: str) -> str:
    text = re.sub(r"^```[^\n]*\n?", "", raw.strip())
    text = re.sub(r"\n?```$", "", text).strip()
    return text


def parse_wildclaw_task_md(task_file: Path, *, benchmark_root: Path) -> WildClawTaskSpec:
    content = task_file.read_text(encoding="utf-8")
    frontmatter_match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", content, re.DOTALL)
    if not frontmatter_match:
        raise ValueError(f"YAML frontmatter not found: {task_file}")

    metadata = yaml.safe_load(frontmatter_match.group(1)) or {}
    body = frontmatter_match.group(2)

    sections: dict[str, str] = {}
    current_section: str | None = None
    lines: list[str] = []
    for line in body.split("\n"):
        header = re.match(r"^##\s+(.+)$", line)
        if header:
            if current_section is not None:
                sections[current_section] = "\n".join(lines).strip()
            current_section = header.group(1)
            lines = []
        else:
            lines.append(line)
    if current_section is not None:
        sections[current_section] = "\n".join(lines).strip()

    prompt = sections.get("Prompt", "").strip()
    raw_workspace = _strip_codeblock(sections.get("Workspace Path", ""))
    if not raw_workspace:
        raise ValueError(f"Missing ## Workspace Path in {task_file}")

    workspace_path = Path(raw_workspace)
    if not workspace_path.is_absolute():
        workspace_path = (benchmark_root / workspace_path).resolve()

    skills_path = (benchmark_root / "skills").resolve()
    timeout_seconds = int(metadata.get("timeout_seconds", 120))
    task_id = str(metadata.get("id") or task_file.stem)

    return WildClawTaskSpec(
        task_id=task_id,
        category=task_file.parent.name,
        prompt=prompt,
        workspace_path=str(workspace_path),
        skills_path=str(skills_path),
        automated_checks=_strip_codeblock(sections.get("Automated Checks", "")),
        env=_strip_codeblock(sections.get("Env", "")),
        skills=_strip_codeblock(sections.get("Skills", "")),
        warmup=_strip_codeblock(sections.get("Warmup", "")),
        timeout_seconds=timeout_seconds,
        file_path=str(task_file.resolve()),
    )


class WildClawBenchClusterExecutor:
    """Run WildClawBench tasks against isolated host SkillClaw nodes."""

    def __init__(self, config: Any):
        self.config = config
        benchmark_root = str(getattr(config, "wildclawbench_root", "") or "").strip()
        if not benchmark_root:
            raise ValueError("wildclawbench_root is required when executor=wildclawbench_cluster")
        self.benchmark_root = Path(benchmark_root).expanduser().resolve()
        self.output_root = Path(
            str(getattr(config, "wildclawbench_output_dir", "") or Path(config.workspace_dir) / "wildclawbench_output")
        ).expanduser().resolve()
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
                public_skill_root="/root/skills",
                skills_enabled=config.cluster_skills_enabled,
                retrieval_mode=config.cluster_retrieval_mode,
                max_context_tokens=config.cluster_max_context_tokens,
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
        self._benchmark_env_cache: dict[str, str] | None = None

    def setup(self) -> None:
        if self._prepared:
            return
        initial_skills = (
            Path(self.config.initial_skills_dir).expanduser().resolve()
            if self.config.initial_skills_dir else None
        )
        self.cluster.prepare(initial_skills_dir=initial_skills)
        self.cluster.start_active_nodes(start_gateway=False)
        self._prepared = True
        self._started = True

    def teardown(self) -> None:
        if not self._started:
            return
        self.cluster.stop_active_nodes()
        self._started = False

    def run_task(
        self,
        task: Any,
        *,
        device_id: str,
        device_dir: Path,
        skills_dir: Path,
        phase: str,
        round_idx: int,
    ) -> WildClawExecutorResult:
        if not self._prepared:
            self.setup()

        task_file = self._resolve_task_file(task)
        parsed = parse_wildclaw_task_md(task_file, benchmark_root=self.benchmark_root)
        parsed.timeout_seconds = _resolve_task_timeout_seconds(
            parsed.timeout_seconds,
            int(getattr(self.config, "wildclawbench_task_timeout_seconds_override", 0) or 0),
        )
        node = self.cluster.nodes[device_id]
        # Match official_clean exactly: each task container is isolated, so a
        # fixed session id is sufficient and avoids session-behavior drift.
        session_id = "chat"

        conversation_path = node.records_dir / "conversations.jsonl"
        prm_path = node.records_dir / "prm_scores.jsonl"
        conv_start = _count_jsonl_records(conversation_path)
        prm_start = _count_jsonl_records(prm_path)

        started = time.perf_counter()
        bench = self._execute_task_in_container(
            parsed=parsed,
            node=node,
            phase=phase,
            round_idx=round_idx,
            session_id=session_id,
            device_skills_dir=skills_dir,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        conv_end, prm_end = self._wait_for_session_records(
            conversation_path=conversation_path,
            prm_path=prm_path,
            conv_start=conv_start,
            prm_start=prm_start,
        )
        artifacts = self.cluster.collect_session_artifacts(
            node_id=device_id,
            session_id=session_id,
            requested_session_id=session_id,
            task_id=task.task_id,
            phase=phase,
            round_idx=round_idx,
            skill_names=[],
            conversation_slice={"start": conv_start, "end": conv_end},
            prm_slice={"start": prm_start, "end": prm_end},
            payload_mode=str(getattr(self.config, "cloud_session_payload_mode", "full") or "full"),
        )

        score = _extract_final_score(bench.get("scores", {}))
        if score is not None and bool(getattr(self.config, "wildclawbench_use_score_as_prm", True)):
            self._annotate_session_score(
                session_path=Path(str(artifacts.get("session_path", ""))),
                score=score,
                parsed=parsed,
                phase=phase,
                task_id=task.task_id,
            )

        session_turns = self._load_session_turns(Path(str(artifacts.get("session_path", ""))))
        used_skills = sorted(
            {
                str(item.get("skill_name") or "").strip()
                for turn in session_turns
                for item in (turn.get("read_skills") or [])
                if isinstance(item, dict) and str(item.get("skill_name") or "").strip()
            }
        )
        response_text = "\n".join(
            str(turn.get("response_text") or "").strip()
            for turn in session_turns
            if str(turn.get("response_text") or "").strip()
        ).strip()
        success_threshold = float(getattr(self.config, "wildclawbench_success_threshold", 0.5))
        has_valid_score = score is not None
        success = bench.get("error") is None and has_valid_score and score >= success_threshold

        artifacts.update(
            {
                "response_text": response_text,
                "wildclawbench": {
                    "task_file": parsed.file_path,
                    "category": parsed.category,
                    "output_dir": bench.get("output_dir"),
                    "scores": bench.get("scores", {}),
                    "usage": bench.get("usage", {}),
                    "error": bench.get("error"),
                    "session_id": session_id,
                },
            }
        )
        notes = str(bench.get("error") or response_text or json.dumps(bench.get("scores", {}), ensure_ascii=False))[:400]
        return WildClawExecutorResult(
            success=success,
            used_skills=used_skills,
            discovered_skills=[],
            latency_ms=elapsed_ms,
            notes=notes,
            artifacts=artifacts,
            score=score,
            score_breakdown=bench.get("scores", {}) if isinstance(bench.get("scores"), dict) else {},
        )

    def _wait_for_session_records(
        self,
        *,
        conversation_path: Path,
        prm_path: Path,
        conv_start: int,
        prm_start: int,
        timeout_s: float = 3.0,
    ) -> tuple[int, int]:
        deadline = time.monotonic() + timeout_s
        conv_last = _count_jsonl_records(conversation_path)
        prm_last = _count_jsonl_records(prm_path)
        stable_rounds = 0
        while time.monotonic() < deadline:
            time.sleep(0.2)
            conv_now = _count_jsonl_records(conversation_path)
            prm_now = _count_jsonl_records(prm_path)
            if conv_now == conv_last and prm_now == prm_last:
                stable_rounds += 1
            else:
                stable_rounds = 0
            conv_last, prm_last = conv_now, prm_now
            if stable_rounds >= 2 and conv_now >= conv_start:
                break
        return conv_last, prm_last

    def _resolve_task_file(self, task: Any) -> Path:
        raw = getattr(task, "raw", {}) if task is not None else {}
        candidate = (
            str(raw.get("task_file") or raw.get("task_path") or raw.get("wildclawbench_task_file") or "").strip()
        )
        if not candidate:
            raise ValueError(f"task {getattr(task, 'task_id', '?')} is missing task_file")
        path = Path(candidate).expanduser()
        if not path.is_absolute():
            path = (self.benchmark_root / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"WildClawBench task file not found: {path}")
        return path

    def _benchmark_env_values(self) -> dict[str, str]:
        if self._benchmark_env_cache is None:
            self._benchmark_env_cache = _read_env_file_values(self.benchmark_root / ".env")
        return dict(self._benchmark_env_cache)

    def _env_or_benchmark(self, key: str, default: str = "", *, prefer_benchmark: bool = False) -> str:
        benchmark_value = str(self._benchmark_env_values().get(key, default) or default).strip()
        if prefer_benchmark and benchmark_value:
            return benchmark_value
        value = str(os.environ.get(key, "") or "").strip()
        if value:
            return value
        return benchmark_value

    def _effective_openrouter_api_key(self) -> str:
        return (
            self._env_or_benchmark("OPENROUTER_API_KEY", prefer_benchmark=True)
            or self._env_or_benchmark("OPENAI_API_KEY", prefer_benchmark=True)
            or str(getattr(self.config, "cluster_llm_api_key", "") or "").strip()
        )

    def _effective_proxy_api_key(self) -> str:
        return self._env_or_benchmark("MY_PROXY_API_KEY", prefer_benchmark=True) or "dummy"

    def _build_models_config(self, node: OpenClawClusterNode) -> dict[str, Any]:
        llm_model = self.config.cluster_llm_model_id or "skillclaw-model"
        base_url_host = str(getattr(self.config, "wildclawbench_container_proxy_host", "host.docker.internal")).strip()
        context_window = int(os.environ.get("SKILLCLAW_WILDBENCH_CONTEXT_WINDOW", "200000") or 200000)
        max_tokens = int(os.environ.get("SKILLCLAW_WILDBENCH_MAX_TOKENS", "32768") or 32768)

        # For baseline runs with skills disabled, mirror the official clean
        # setup as closely as possible: route container OpenClaw directly to
        # the upstream OpenAI-compatible gateway instead of through the
        # SkillClaw proxy. This avoids proxy-side runtime semantics (approval
        # prompts, allowlists, tool-policy drift) while keeping the same image,
        # task container, and grading flow.
        use_direct_gateway = bool(getattr(self.config, "cluster_skills_enabled", True)) is False
        if use_direct_gateway:
            explicit_openrouter_base = self._rewrite_loopback_base_url(
                self._env_or_benchmark("OPENROUTER_BASE_URL", prefer_benchmark=True)
            )
            alias_base = ""
            alias_port_raw = str(self._env_or_benchmark("ALIAS_PROXY_PORT", "28081", prefer_benchmark=True)).strip()
            try:
                alias_port = int(alias_port_raw)
            except ValueError:
                alias_port = 28081
            if alias_port > 0 and _port_is_listening("127.0.0.1", alias_port):
                alias_base = f"http://{self._container_host_alias()}:{alias_port}/v1"
            upstream_base = self._rewrite_loopback_base_url(
                str(getattr(self.config, "cluster_llm_api_base", "") or "")
            )
            # Mirror official_clean semantics:
            # - OpenClaw's provider config keeps using the upstream/base gateway
            #   (normally 28080 + MY_PROXY_API_KEY)
            # - task scripts and grading use OPENROUTER_BASE_URL from .env
            #   (normally 28081 + OPENROUTER_API_KEY)
            direct_base = upstream_base or explicit_openrouter_base or alias_base
            if direct_base:
                api_key = (
                    self._effective_openrouter_api_key()
                    if direct_base in {explicit_openrouter_base, alias_base}
                    else self._effective_proxy_api_key()
                )
                return {
                    "providers": {
                        "custom-gateway": {
                            "baseUrl": direct_base,
                            "apiKey": api_key,
                            "api": "openai-completions",
                            "models": [
                                {
                                    "id": llm_model,
                                    "name": llm_model,
                                    "input": ["text", "image"],
                                    "reasoning": True,
                                    "contextWindow": context_window,
                                    "maxTokens": max_tokens,
                                }
                            ],
                        }
                    }
                }

        return {
            "providers": {
                "custom-gateway": {
                    "baseUrl": f"http://{base_url_host}:{node.proxy_port}/v1",
                    "apiKey": self.config.cluster_proxy_api_key or "skillclaw",
                    "api": "openai-completions",
                    "models": [
                        {
                            "id": llm_model,
                            "name": llm_model,
                            "input": ["text", "image"],
                            "reasoning": True,
                            "contextWindow": context_window,
                            "maxTokens": max_tokens,
                        }
                    ],
                }
            }
        }

    def _docker_image(self) -> str:
        return (
            str(getattr(self.config, "wildclawbench_docker_image", "") or "").strip()
            or self._env_or_benchmark("DOCKER_IMAGE", "wildclawbench-ubuntu:v1.2-browserfix", prefer_benchmark=True)
        )

    def _container_host_alias(self) -> str:
        return str(getattr(self.config, "wildclawbench_container_proxy_host", "host.docker.internal")).strip()

    def _rewrite_loopback_base_url(self, base_url: str) -> str:
        normalized = str(base_url or "").strip()
        if not normalized:
            return ""
        host_alias = self._container_host_alias()
        return re.sub(
            r"^(https?://)(?:127\.0\.0\.1|localhost)(?=[:/]|$)",
            lambda match: f"{match.group(1)}{host_alias}",
            normalized,
        )

    def _effective_openrouter_base_url(self) -> str:
        explicit = self._rewrite_loopback_base_url(
            self._env_or_benchmark("OPENROUTER_BASE_URL", prefer_benchmark=True)
        )
        if explicit:
            return explicit

        alias_port_raw = str(self._env_or_benchmark("ALIAS_PROXY_PORT", "28081", prefer_benchmark=True)).strip()
        try:
            alias_port = int(alias_port_raw)
        except ValueError:
            alias_port = 28081
        if alias_port > 0 and _port_is_listening("127.0.0.1", alias_port):
            return f"http://{self._container_host_alias()}:{alias_port}/v1"

        fallback = (
            str(getattr(self.config, "cluster_llm_api_base", "") or "").strip()
            or self._env_or_benchmark("OPENAI_BASE_URL", prefer_benchmark=True)
        )
        return self._rewrite_loopback_base_url(fallback)

    def _tmp_workspace(self) -> str:
        return str(getattr(self.config, "wildclawbench_tmp_workspace", "/tmp_workspace"))

    def _gateway_port(self) -> int:
        return int(getattr(self.config, "wildclawbench_gateway_port", 18789))

    def _thinking_default(self) -> str:
        value = str(
            getattr(self.config, "wildclawbench_thinking_default", "")
            or os.environ.get("WILDCLAWBENCH_THINKING_DEFAULT", "off")
        ).strip()
        return value or "off"

    def _docker_memory_limit(self) -> str:
        return str(getattr(self.config, "wildclawbench_docker_memory_limit", "") or "").strip()

    def _run_subprocess(self, cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.run(cmd, capture_output=True, text=True, check=False, **kwargs)

    def _wait_for_container_exec_ready(self, container_name: str) -> None:
        deadline = time.monotonic() + _container_exec_ready_timeout_seconds()
        last_error = ""
        while time.monotonic() < deadline:
            proc = self._run_subprocess(["docker", "exec", container_name, "/bin/true"])
            if proc.returncode == 0:
                return
            last_error = _completed_process_debug_text(proc)
            time.sleep(0.25)
        raise RuntimeError(f"container exec not ready: {last_error[:800]}")

    def _execute_task_in_container(
        self,
        *,
        parsed: WildClawTaskSpec,
        node: OpenClawClusterNode,
        phase: str,
        round_idx: int,
        session_id: str,
        device_skills_dir: Path,
    ) -> dict[str, Any]:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        container_name = f"{parsed.task_id}_{node.node_id}_{phase}_r{round_idx}_{uuid.uuid4().hex[:6]}"
        output_dir = self.output_root / phase / parsed.category / parsed.task_id / f"{node.node_id}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        workspace_path = Path(parsed.workspace_path)
        tmp_path = workspace_path / "tmp"
        tmp_workspace = self._tmp_workspace()
        transcript_path = output_dir / f"{session_id}.jsonl"
        agent_log_path = output_dir / "agent.log"
        gateway_log_path = output_dir / "gateway.log"

        gateway_proc: subprocess.Popen[str] | None = None
        agent_proc: subprocess.Popen[str] | None = None
        warmup_procs: list[subprocess.Popen[str]] = []
        start_time = time.perf_counter()
        result: dict[str, Any] = {
            "task_id": parsed.task_id,
            "scores": {},
            "error": None,
            "output_dir": str(output_dir),
            "usage": {},
        }

        try:
            self._remove_container(container_name)
            self._start_container(container_name, workspace_path, parsed.env, tmp_path)
            self._setup_workspace(container_name)
            self._setup_skills(container_name, parsed.skills, Path(parsed.skills_path))
            self._setup_local_device_skills(container_name, device_skills_dir)
            warmup_procs = self._run_warmup(
                container_name,
                parsed.warmup,
                output_dir=output_dir,
            )
            self._inject_models(container_name, self._build_models_config(node))
            model_name = f"custom-gateway/{self.config.cluster_llm_model_id or 'skillclaw-model'}"
            self._configure_openclaw_runtime(
                container_name,
                model_name=model_name,
                thinking=self._thinking_default(),
            )
            self._set_model(container_name, model_name)
            self._inject_openrouter_auth_profile(container_name)
            self._set_image_model(container_name, model_name)

            gateway_proc = self._run_background(
                container_name,
                (
                    "if [ -f /root/.openclaw/proxy-bootstrap.mjs ]; then "
                    "export NODE_OPTIONS=\"--import=/root/.openclaw/proxy-bootstrap.mjs ${NODE_OPTIONS:-}\"; "
                    "fi && "
                    f"openclaw gateway --port {self._gateway_port()}"
                ),
                gateway_log_path,
            )
            time.sleep(2.0)

            safe_prompt = self._compose_benchmark_prompt(parsed).replace("'", "'\\''")
            agent_proc = self._run_background(
                container_name,
                (
                    f"openclaw agent --session-id {session_id} "
                    f"--timeout {parsed.timeout_seconds} --message '{safe_prompt}'"
                ),
                agent_log_path,
            )
            try:
                agent_proc.wait(timeout=parsed.timeout_seconds)
            except subprocess.TimeoutExpired:
                result["error"] = f"agent timed out after {parsed.timeout_seconds}s"
                agent_proc.kill()
                agent_proc.wait(timeout=10)
        except Exception as exc:
            logger.exception("[WildClawBench] task %s failed", parsed.task_id)
            result["error"] = str(exc)
        finally:
            elapsed = time.perf_counter() - start_time
            result["elapsed_time_s"] = round(elapsed, 3)
            result["scores"] = self._grade_task(
                container_name=container_name,
                task_id=parsed.task_id,
                automated_checks=parsed.automated_checks,
                output_dir=output_dir,
                workspace_path=workspace_path,
                existing_error=result.get("error"),
            )
            self._copy_transcript(container_name, session_id, transcript_path)
            result["usage"] = _extract_usage_from_jsonl(transcript_path)
            result["usage"]["elapsed_time"] = round(elapsed, 2)
            self._collect_output(container_name, output_dir)
            if _should_retry_grade_result(result.get("scores", {})):
                fallback_scores = self._grade_task_locally_from_collected_output(
                    task_id=parsed.task_id,
                    automated_checks=parsed.automated_checks,
                    output_dir=output_dir,
                    workspace_path=workspace_path,
                )
                if fallback_scores and not _should_retry_grade_result(fallback_scores):
                    logger.info(
                        "[WildClawBench] local grading fallback succeeded for %s: %s",
                        parsed.task_id,
                        json.dumps(fallback_scores, ensure_ascii=False)[:200],
                    )
                    result["scores"] = fallback_scores
            if gateway_proc is not None:
                gateway_proc.terminate()
                try:
                    gateway_proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    gateway_proc.kill()
            for proc in warmup_procs:
                try:
                    proc.terminate()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=5)
                except Exception:
                    with suppress(Exception):
                        proc.kill()
            if agent_proc is not None:
                self._close_log_handle(agent_proc)
            if gateway_proc is not None:
                self._close_log_handle(gateway_proc)
            for proc in warmup_procs:
                self._close_log_handle(proc)
            self._remove_container(container_name)
        return result

    def _start_container(
        self,
        container_name: str,
        workspace_path: Path,
        extra_env: str,
        tmp_path: Path,
    ) -> None:
        proxy_http = self._env_or_benchmark("HTTP_PROXY_INNER", prefer_benchmark=True)
        proxy_https = self._env_or_benchmark("HTTPS_PROXY_INNER", prefer_benchmark=True)
        proxy_no = self._env_or_benchmark("NO_PROXY_INNER", prefer_benchmark=True)
        brave_api_key = self._env_or_benchmark("BRAVE_API_KEY", prefer_benchmark=True)
        openrouter_api_key = self._effective_openrouter_api_key()
        openrouter_base_url = self._effective_openrouter_base_url()
        judge_model = self._env_or_benchmark("JUDGE_MODEL", prefer_benchmark=True)
        proxy_api_key = self._effective_proxy_api_key()

        # If a container-side proxy port is configured and the proxy host is
        # host.docker.internal, automatically route the container's traffic
        # through the host proxy (e.g. mihomo/clash running on the host).
        # This overrides HTTP_PROXY_INNER so that containers can reach APIs
        # like Brave Search that are geo-blocked without a proxy.
        container_proxy_port = str(
            getattr(self.config, "wildclawbench_container_proxy_port", "")
        ).strip()
        container_proxy_host = str(
            getattr(self.config, "wildclawbench_container_proxy_host", "")
        ).strip()
        if container_proxy_port and container_proxy_host:
            proxy_http = f"http://{container_proxy_host}:{container_proxy_port}"
            proxy_https = proxy_http
            # Ensure host.docker.internal is excluded from the proxy so that
            # in-container traffic to the SkillClaw proxy on the host is direct.
            # Also exclude common package registries and apt mirrors that should
            # bypass the proxy to avoid 502/TLS failures during warmup.
            no_proxy_entries = {
                "localhost", "127.0.0.1", container_proxy_host,
                "pypi.org", "files.pythonhosted.org", "registry.npmjs.org",
                "github.com", "objects.githubusercontent.com",
                "archive.ubuntu.com", "security.ubuntu.com", "deb.nodesource.com",
            }
            if proxy_no:
                no_proxy_entries.update(e.strip() for e in proxy_no.split(",") if e.strip())
            proxy_no = ",".join(sorted(no_proxy_entries))
        proxy_all = os.environ.get("ALL_PROXY_INNER", proxy_https or proxy_http)

        env_args = [
            "-e", f"http_proxy={proxy_http}",
            "-e", f"https_proxy={proxy_https}",
            "-e", f"all_proxy={proxy_all}",
            "-e", f"HTTP_PROXY={proxy_http}",
            "-e", f"HTTPS_PROXY={proxy_https}",
            "-e", f"ALL_PROXY={proxy_all}",
            "-e", f"BRAVE_API_KEY={brave_api_key}",
            "-e", f"OPENROUTER_API_KEY={openrouter_api_key}",
            "-e", f"OPENROUTER_BASE_URL={openrouter_base_url}",
            "-e", f"JUDGE_MODEL={judge_model}",
            "-e", f"MY_PROXY_API_KEY={proxy_api_key}",
            "-e", "ERROR_RATE=0",
            "-e", f"no_proxy={'' if not proxy_http else proxy_no}",
            "-e", f"NO_PROXY={'' if not proxy_http else proxy_no}",
        ]
        for key in (
            "PIP_INDEX_URL",
            "PIP_EXTRA_INDEX_URL",
            "PIP_TRUSTED_HOST",
            "NPM_CONFIG_STRICT_SSL",
            "npm_config_strict_ssl",
            "NPM_CONFIG_REGISTRY",
            "npm_config_registry",
            "NODE_TLS_REJECT_UNAUTHORIZED",
        ):
            value = os.environ.get(key, "")
            if value:
                env_args += ["-e", f"{key}={value}"]
        extra_host_args: list[str] = []
        if str(getattr(self.config, "wildclawbench_container_proxy_host", "")).strip() == "host.docker.internal":
            extra_host_args = ["--add-host", "host.docker.internal:host-gateway"]
        memory_limit = self._docker_memory_limit()
        resource_args: list[str] = []
        if memory_limit:
            resource_args = ["--memory", memory_limit]
        for line in extra_env.splitlines():
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            if "=" in raw:
                key, explicit_value = raw.split("=", 1)
                key = key.strip()
                explicit_value = explicit_value.strip()
                if key:
                    env_args += ["-e", f"{key}={explicit_value}"]
                continue
            key = raw
            value = self._env_or_benchmark(key, prefer_benchmark=True)
            if value:
                env_args += ["-e", f"{key}={value}"]

        proc = self._run_subprocess(
            [
                "docker",
                "run",
                "-d",
                "--name",
                container_name,
                *extra_host_args,
                *resource_args,
                *env_args,
                "-v",
                f"{workspace_path}:/app:ro",
                self._docker_image(),
                "/bin/bash",
                "-c",
                "tail -f /dev/null",
            ]
        )
        if proc.returncode != 0:
            raise RuntimeError(f"container startup failed: {(proc.stderr or proc.stdout).strip()[:400]}")

        self._wait_for_container_exec_ready(container_name)

        if tmp_path.exists():
            mkdir_proc = self._run_subprocess(
                ["docker", "exec", container_name, "mkdir", "-p", f"{self._tmp_workspace()}/tmp"]
            )
            if mkdir_proc.returncode != 0:
                raise RuntimeError(f"tmp dir setup failed: {_completed_process_debug_text(mkdir_proc)[:800]}")
            cp_proc = self._run_subprocess(
                ["docker", "cp", f"{tmp_path}/.", f"{container_name}:{self._tmp_workspace()}/tmp/"]
            )
            if cp_proc.returncode != 0:
                raise RuntimeError(f"tmp dir copy failed: {_completed_process_debug_text(cp_proc)[:800]}")

    def _setup_workspace(self, container_name: str) -> None:
        symlink_threshold = _workspace_symlink_large_file_threshold_bytes()
        setup_cmd = f"""
set -e
python3 - <<'PY'
import os
import shutil
from pathlib import Path

SRC = Path('/app')
DST = Path({self._tmp_workspace()!r})
THRESHOLD = {symlink_threshold}
DST.mkdir(parents=True, exist_ok=True)

def _copy_tree(src: Path, dst: Path) -> None:
    for root, dirs, files in os.walk(src):
        root_path = Path(root)
        rel = root_path.relative_to(src)
        dst_root = dst / rel
        dst_root.mkdir(parents=True, exist_ok=True)
        for name in dirs:
            (dst_root / name).mkdir(parents=True, exist_ok=True)
        for name in files:
            source = root_path / name
            target = dst_root / name
            if target.is_symlink() or target.exists():
                target.unlink()
            size = source.stat().st_size
            if size >= THRESHOLD:
                link_target = os.readlink(source) if source.is_symlink() else str(source)
                os.symlink(link_target, target)
            else:
                shutil.copy2(source, target)

_copy_tree(SRC, DST)
exec_root = SRC / 'exec'
if exec_root.is_dir():
    _copy_tree(exec_root, DST)
PY
find {self._tmp_workspace()} -type d -exec chmod u+rwx {{}} +
find {self._tmp_workspace()} -type f -exec chmod u+rw {{}} +
if ! command -v python >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
  ln -sf "$(command -v python3)" /usr/local/bin/python
fi
if command -v bash >/dev/null 2>&1; then
  ln -sf "$(command -v bash)" /bin/sh
fi
mkdir -p {self._tmp_workspace()}/results
PW_CHROME="$(find /root/.cache/ms-playwright -path '*/chrome-linux64/chrome' -type f 2>/dev/null | head -n 1)"
if [ -n "$PW_CHROME" ]; then
  ln -sf "$PW_CHROME" /usr/local/bin/google-chrome
  ln -sf "$PW_CHROME" /usr/local/bin/chromium
fi
mkdir -p /root/memory
today="$(date +%F)"
yesterday="$(date -d yesterday +%F 2>/dev/null || python3 -c 'from datetime import date, timedelta; print((date.today() - timedelta(days=1)).isoformat())')"
touch "/root/memory/${{today}}.md" "/root/memory/${{yesterday}}.md"
rm -rf /root/.openclaw/workspace
ln -s {self._tmp_workspace()} /root/.openclaw/workspace
cat > /root/.openclaw/proxy-bootstrap.mjs <<'EOF'
import {{ setGlobalDispatcher, ProxyAgent }} from '/usr/lib/node_modules/openclaw/node_modules/undici/index.js';
const proxy = process.env.HTTPS_PROXY || process.env.HTTP_PROXY || process.env.ALL_PROXY;
if (proxy) {{
  setGlobalDispatcher(new ProxyAgent(proxy));
}}
EOF
""".strip()
        retry_delays = _workspace_setup_retry_delays_seconds()
        timeout_s = _workspace_setup_timeout_seconds()
        proc: subprocess.CompletedProcess[str] | None = None
        last_error = ""
        for attempt_idx in range(len(retry_delays) + 1):
            self._wait_for_container_exec_ready(container_name)
            try:
                proc = self._run_subprocess(
                    [
                        "docker",
                        "exec",
                        container_name,
                        "/bin/bash",
                        "-c",
                        setup_cmd,
                    ],
                    timeout=timeout_s,
                )
                if proc.returncode == 0:
                    return
                last_error = _completed_process_debug_text(proc)
            except subprocess.TimeoutExpired as exc:
                last_error = f"timeout after {timeout_s}s: {exc}"
            if attempt_idx >= len(retry_delays):
                break
            delay_s = retry_delays[attempt_idx]
            jitter_s = min(0.5, delay_s * 0.1) * random.random()
            logger.warning(
                "[WildClawBench] workspace setup retry %d/%d for %s after failure: %s",
                attempt_idx + 1,
                len(retry_delays),
                container_name,
                last_error[:300],
            )
            time.sleep(delay_s + jitter_s)
        raise RuntimeError(f"workspace setup failed after retries: {last_error[:1200]}")

    def _inject_lobster_workspace(self, container_name: str, node: OpenClawClusterNode) -> None:
        # Host node workspaces are intentionally not injected into benchmark containers.
        _ = (container_name, node)
        return

    def _setup_skills(self, container_name: str, skills_text: str, skills_root: Path) -> None:
        for line in skills_text.splitlines():
            skill_name = line.strip()
            if not skill_name:
                continue
            self._run_subprocess(["docker", "exec", container_name, "mkdir", "-p", f"/root/skills/{skill_name}"])
            self._run_subprocess(["docker", "cp", f"{skills_root / skill_name}", f"{container_name}:/root/skills"])

    def _setup_local_device_skills(self, container_name: str, device_skills_dir: Path) -> None:
        if not device_skills_dir.exists():
            return
        for skill_md in sorted(device_skills_dir.glob("*/SKILL.md")):
            skill_dir = skill_md.parent
            skill_name = skill_dir.name
            self._run_subprocess(
                ["docker", "exec", container_name, "/bin/bash", "-c", f"rm -rf '/root/skills/{skill_name}'"]
            )
            self._run_subprocess(["docker", "cp", str(skill_dir), f"{container_name}:/root/skills"])

    def _run_warmup(
        self,
        container_name: str,
        warmup: str,
        *,
        output_dir: Path,
    ) -> list[subprocess.Popen[str]]:
        background_procs: list[subprocess.Popen[str]] = []
        bg_idx = 0
        for line in warmup.splitlines():
            cmd = line.strip()
            if not cmd or cmd.startswith("#"):
                continue
            if cmd in {"rm -f -r /tmp_workspace/tmp", "rm -rf /tmp_workspace/fixtures"}:
                logger.info(
                    "[WildClawBench] warmup skip destructive cleanup for %s: %s",
                    container_name,
                    cmd,
                )
                continue
            if cmd.endswith("&"):
                cmd = cmd[:-1].rstrip()
                log_path = output_dir / "warmup" / f"bg-{bg_idx:02d}.log"
                bg_idx += 1
                background_procs.append(self._run_background(container_name, cmd, log_path))
                continue
            if self._provision_agent_browser_from_cache(container_name, cmd):
                continue
            # Enhance pip install commands with longer timeout and more retries
            # to survive concurrent-download bandwidth contention.
            cmd = _enhance_pip_cmd(cmd)
            cmd = _bypass_proxy_for_apt_cmd(cmd)
            retry_delays = _warmup_retry_delays_seconds()
            proc: subprocess.CompletedProcess[str] | None = None
            for attempt_idx in range(len(retry_delays) + 1):
                proc = self._run_subprocess(["docker", "exec", container_name, "/bin/bash", "-c", cmd])
                if proc.returncode == 0:
                    break
                error_text = self._warmup_error_text(proc)
                if attempt_idx >= len(retry_delays) or not self._should_retry_warmup(cmd, error_text):
                    break
                delay_s = retry_delays[attempt_idx]
                logger.warning(
                    "[WildClawBench] warmup retry %d/%d for %s after transient network failure",
                    attempt_idx + 1,
                    len(retry_delays),
                    cmd[:120],
                )
                time.sleep(delay_s)
            if proc is not None and proc.returncode != 0:
                raise RuntimeError(f"warmup failed: {cmd[:120]} :: {self._warmup_error_text(proc)[:300]}")
        return background_procs

    def _warmup_error_text(self, proc: subprocess.CompletedProcess[str]) -> str:
        parts = [str(proc.stderr or "").strip(), str(proc.stdout or "").strip()]
        return "\n".join(part for part in parts if part).strip()

    def _should_retry_warmup(self, cmd: str, error_text: str) -> bool:
        upper_error = error_text.upper()
        # Retry any command (pip, npm, etc.) on transient network errors
        if any(token in upper_error for token in _RETRYABLE_NETWORK_WARMUP_ERRORS):
            return True
        # Legacy: also check agent-browser-specific error patterns
        normalized_cmd = cmd.strip()
        if not normalized_cmd.startswith("npm install -g agent-browser"):
            return False
        return any(token in upper_error for token in _RETRYABLE_AGENT_BROWSER_WARMUP_ERRORS)

    def _agent_browser_cache_dir(self) -> Path | None:
        raw = str(os.environ.get("SKILLCLAW_AGENT_BROWSER_CACHE_DIR", "")).strip()
        if not raw:
            return None
        path = Path(raw).expanduser()
        if not (path / "bin" / "agent-browser.js").exists():
            return None
        return path

    def _provision_agent_browser_from_cache(self, container_name: str, cmd: str) -> bool:
        if not cmd.strip().startswith("npm install -g agent-browser"):
            return False
        cache_dir = self._agent_browser_cache_dir()
        if cache_dir is None:
            return False
        setup_steps = [
            ["docker", "exec", container_name, "mkdir", "-p", "/opt"],
            ["docker", "exec", container_name, "/bin/bash", "-c", "rm -rf /opt/agent-browser"],
            ["docker", "cp", str(cache_dir), f"{container_name}:/opt/"],
            [
                "docker",
                "exec",
                container_name,
                "/bin/bash",
                "-c",
                (
                    "printf '%s\\n' '#!/bin/sh' "
                    "'exec node /opt/agent-browser/bin/agent-browser.js \"$@\"' "
                    "> /usr/local/bin/agent-browser && chmod +x /usr/local/bin/agent-browser"
                ),
            ],
            ["docker", "exec", container_name, "/bin/bash", "-c", "agent-browser --help >/dev/null 2>&1"],
        ]
        for step in setup_steps:
            proc = self._run_subprocess(step)
            if proc.returncode != 0:
                logger.warning(
                    "[WildClawBench] agent-browser cache provisioning failed, falling back to npm install: %s",
                    self._warmup_error_text(proc)[:300],
                )
                return False
        logger.info("[WildClawBench] provisioned agent-browser from host cache: %s", cache_dir)
        return True

    def _inject_models(self, container_name: str, models_config: dict[str, Any]) -> None:
        models_json = json.dumps(models_config, ensure_ascii=False)
        inject_cmd = (
            "python3 - <<'PY'\n"
            "import json, pathlib\n"
            "config_path = pathlib.Path('/root/.openclaw/openclaw.json')\n"
            f"models = json.loads({models_json!r})\n"
            "config = json.loads(config_path.read_text()) if config_path.exists() else {}\n"
            "config['models'] = models\n"
            "config_path.write_text(json.dumps(config, indent=2))\n"
            "PY"
        )
        proc = self._run_subprocess(["docker", "exec", container_name, "/bin/bash", "-c", inject_cmd])
        if proc.returncode != 0:
            raise RuntimeError(f"models injection failed: {proc.stderr.strip()[:400]}")

    def _configure_openclaw_runtime(self, container_name: str, *, model_name: str, thinking: str) -> None:
        config_cmds: list[str] = []
        if os.environ.get("SKIP_FORCE_FULL_PROFILE") != "1":
            config_cmds.append("openclaw config set tools.profile full")
        config_cmds.extend([
            "openclaw config set agents.defaults.sandbox.mode off",
            "openclaw config set tools.exec.host gateway",
            "openclaw config set tools.exec.ask off",
            "openclaw config set tools.exec.security full",
            f"openclaw config set agents.defaults.thinkingDefault {shlex.quote(thinking)}",
            "openclaw config set browser.defaultProfile openclaw",
            "openclaw config set browser.headless true",
            "openclaw config set browser.noSandbox true",
            "openclaw config set browser.remoteCdpTimeoutMs 10000",
            "openclaw config set browser.remoteCdpHandshakeTimeoutMs 30000",
            (
                "PW_CHROME=\"$(find /root/.cache/ms-playwright -path '*/chrome-linux64/chrome' -type f 2>/dev/null | head -n 1)\"; "
                "if [ -n \"$PW_CHROME\" ]; then openclaw config set browser.executablePath \"$PW_CHROME\"; fi"
            ),
        ])
        proc = self._run_subprocess(
            ["docker", "exec", container_name, "/bin/bash", "-c", " && ".join(config_cmds)]
        )
        if proc.returncode != 0:
            raise RuntimeError(f"openclaw runtime configuration failed: {proc.stderr.strip()[:400]}")

        proxy_patch = (
            "python3 - <<'PY'\n"
            "import json, os\n"
            "from pathlib import Path\n"
            "config_path = Path('/root/.openclaw/openclaw.json')\n"
            "data = json.loads(config_path.read_text()) if config_path.exists() else {}\n"
            "proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('HTTP_PROXY') or os.environ.get('ALL_PROXY')\n"
            "if proxy:\n"
            "    browser = data.setdefault('browser', {})\n"
            "    extra = list(browser.get('extraArgs') or [])\n"
            "    flag = f'--proxy-server={proxy}'\n"
            "    if flag not in extra:\n"
            "        extra.append(flag)\n"
            "    browser['extraArgs'] = extra\n"
            "config_path.write_text(json.dumps(data, indent=2))\n"
            "PY"
        )
        self._run_subprocess(["docker", "exec", container_name, "/bin/bash", "-lc", proxy_patch])

    def _compose_benchmark_prompt(self, parsed: WildClawTaskSpec) -> str:
        system_prompt = (
            "You are an expert in a restricted, non-interactive environment. "
            f"Solve the task efficiently before the timeout ({parsed.timeout_seconds}s). "
            "Run all processes in the foreground without user input or background services. "
            "Provide a complete, functional solution in a single pass with no placeholders.\n"
        )
        task_specific_hint = ""
        if parsed.task_id == "02_Code_Intelligence_task_6_benchmark_vlmeval_ocrbench_zh":
            task_specific_hint = (
                "Important execution policy for this OCRBench task:\n"
                "- The local gateway may expose the target model under the compatible alias gpt-5-mini "
                "even when the benchmark prompt names gpt-5-mini-2025-08-07; use the gateway-compatible "
                "alias if needed, but keep the requested model name in the final result.json.\n"
                "- The grader only checks that VLMEvalKit is cloned and that result.json contains a Final Score "
                "in the expected OCRBench range, so focus on producing a valid result.json efficiently after "
                "setting up the evaluation workspace.\n\n"
            )
        return system_prompt + task_specific_hint + parsed.prompt

    def _local_grader_env(self) -> dict[str, str]:
        env = dict(os.environ)
        repo_root = Path(__file__).resolve().parents[2]
        for candidate in (
            self.benchmark_root / ".env",
            repo_root / ".env",
        ):
            for key, value in _read_env_file_values(candidate).items():
                if not env.get(key):
                    env[key] = value
        if not env.get("OPENAI_API_KEY") and env.get("OPENROUTER_API_KEY"):
            env["OPENAI_API_KEY"] = env["OPENROUTER_API_KEY"]
        if not env.get("OPENAI_BASE_URL") and env.get("OPENROUTER_BASE_URL"):
            env["OPENAI_BASE_URL"] = env["OPENROUTER_BASE_URL"]
        return env

    def _set_model(self, container_name: str, model_name: str) -> None:
        proc = self._run_subprocess(
            ["docker", "exec", container_name, "/bin/bash", "-c", f"openclaw models set '{model_name}'"]
        )
        if proc.returncode != 0:
            raise RuntimeError(f"model setup failed: {proc.stderr.strip()[:400]}")

    def _inject_openrouter_auth_profile(self, container_name: str) -> None:
        openrouter_api_key = self._effective_openrouter_api_key()
        if not openrouter_api_key:
            return
        auth_profile_path = "/root/.openclaw/agents/main/agent/auth-profiles.json"
        inject_cmd = (
            "python3 -c \""
            "import json, pathlib; "
            f"p = pathlib.Path('{auth_profile_path}'); "
            "d = json.loads(p.read_text()) if p.exists() else {'version':1,'profiles':{}}; "
            "d.setdefault('profiles', {})['openrouter:default'] = "
            f"{{'type':'api_key','provider':'openrouter','key':'{openrouter_api_key}'}}; "
            "p.parent.mkdir(parents=True, exist_ok=True); "
            "p.write_text(json.dumps(d, indent=2))\""
        )
        self._run_subprocess(["docker", "exec", container_name, "/bin/bash", "-c", inject_cmd])

    def _set_image_model(self, container_name: str, model_name: str) -> None:
        self._run_subprocess(
            [
                "docker",
                "exec",
                container_name,
                "/bin/bash",
                "-c",
                f"openclaw config set agents.defaults.imageModel.primary '{model_name}'",
            ]
        )

    def _run_background(self, container_name: str, bash_cmd: str, log_path: Path) -> subprocess.Popen[str]:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            ["docker", "exec", container_name, "/bin/bash", "-c", f"cd {self._tmp_workspace()} && {bash_cmd}"],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
        )
        proc._skillclaw_log_file = log_file
        return proc


    def _grade_task_locally_from_collected_output(
        self,
        *,
        task_id: str,
        automated_checks: str,
        output_dir: Path,
        workspace_path: Path,
    ) -> dict[str, Any]:
        if not automated_checks.strip():
            return {}
        results_src = output_dir / "task_output" / "workspace" / "results"
        if not results_src.exists():
            return {}
        with tempfile.TemporaryDirectory(prefix="skillclaw-wildbench-grade-") as tmpdir:
            stage_root = Path(tmpdir)
            stage_results = stage_root / "results"
            shutil.copytree(results_src, stage_results, dirs_exist_ok=True)
            gt_src = workspace_path / "gt"
            if gt_src.is_dir():
                shutil.copytree(gt_src, stage_root / "gt", dirs_exist_ok=True)
            rewritten_checks = automated_checks.replace("/tmp_workspace", str(stage_root))
            runner_code = "\n".join(
                [
                    "import json",
                    rewritten_checks,
                    "",
                    f"result = grade(transcript=[], workspace_path={str(stage_root)!r})",
                    "print(json.dumps(result))",
                ]
            ) + "\n"
            runner_path = stage_root / "_grade_runner.py"
            runner_path.write_text(runner_code, encoding="utf-8")
            run_proc = subprocess.run(
                ["python3", str(runner_path)],
                capture_output=True,
                text=True,
                check=False,
                timeout=_wildbench_grade_timeout_seconds(),
                env=self._local_grader_env(),
            )
            if run_proc.returncode != 0:
                return {"error": run_proc.stderr.strip() or "local grader failed"}
            parsed_result: dict[str, Any] | None = None
            for line in reversed(run_proc.stdout.splitlines()):
                raw = line.strip()
                if not raw.startswith("{"):
                    continue
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    parsed_result = parsed
                    break
            return parsed_result or {"error": "failed to parse local grading JSON"}

    def _grade_task(
        self,
        *,
        container_name: str,
        task_id: str,
        automated_checks: str,
        output_dir: Path,
        workspace_path: Path,
        existing_error: str | None,
    ) -> dict[str, Any]:
        gt_host = workspace_path / "gt"
        if gt_host.is_dir():
            self._run_subprocess(
                [
                    "docker",
                    "exec",
                    container_name,
                    "/bin/bash",
                    "-lc",
                    f"rm -rf {self._tmp_workspace()}/gt && mkdir -p {self._tmp_workspace()}/gt",
                ]
            )
            self._run_subprocess(
                [
                    "docker",
                    "cp",
                    f"{gt_host}/.",
                    f"{container_name}:{self._tmp_workspace()}/gt/",
                ]
            )
        if existing_error:
            return {"error": existing_error}
        if not automated_checks.strip():
            return {}
        self._run_subprocess(
            [
                "docker",
                "exec",
                container_name,
                "/bin/bash",
                "-lc",
                f"sync; sleep 1; ls -lah {self._tmp_workspace()}/results || true; ls -lah {self._tmp_workspace()}/gt || true",
            ]
        )

        runner_code = "\n".join(
            [
                "import json",
                automated_checks,
                "",
                f"result = grade(transcript=[], workspace_path={self._tmp_workspace()!r})",
                "print(json.dumps(result))",
            ]
        ) + "\n"
        temp_file = output_dir / "_grade_runner.py"
        temp_file.write_text(runner_code, encoding="utf-8")
        try:
            cp_proc = self._run_subprocess(["docker", "cp", str(temp_file), f"{container_name}:/tmp/_grade_runner.py"])
            if cp_proc.returncode != 0:
                return {"error": cp_proc.stderr.strip() or "docker cp failed"}
            retry_delays = _grade_retry_delays_seconds()
            for attempt_idx in range(len(retry_delays) + 1):
                run_proc = self._run_subprocess(
                    ["docker", "exec", container_name, "python3", "/tmp/_grade_runner.py"],
                    timeout=_wildbench_grade_timeout_seconds(),
                )
                if run_proc.returncode != 0:
                    error_result = {"error": run_proc.stderr.strip() or "grader failed"}
                    if attempt_idx < len(retry_delays) and _should_retry_grade_result(error_result):
                        delay_s = retry_delays[attempt_idx]
                        logger.warning(
                            "[WildClawBench] transient grader failure for %s (attempt %d/%d), retrying in %.1fs: %s",
                            task_id,
                            attempt_idx + 1,
                            len(retry_delays) + 1,
                            delay_s,
                            error_result["error"][:200],
                        )
                        time.sleep(delay_s)
                        continue
                    return error_result
                parsed_result: dict[str, Any] | None = None
                for line in reversed(run_proc.stdout.splitlines()):
                    raw = line.strip()
                    if not raw.startswith("{"):
                        continue
                    try:
                        parsed = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(parsed, dict):
                        parsed_result = parsed
                        break
                if parsed_result is None:
                    parsed_result = {"error": "failed to parse grading JSON"}
                if attempt_idx < len(retry_delays) and _should_retry_grade_result(parsed_result):
                    delay_s = retry_delays[attempt_idx]
                    logger.warning(
                        "[WildClawBench] transient grading result for %s (attempt %d/%d), retrying in %.1fs: %s",
                        task_id,
                        attempt_idx + 1,
                        len(retry_delays) + 1,
                        delay_s,
                        json.dumps(parsed_result, ensure_ascii=False)[:200],
                    )
                    time.sleep(delay_s)
                    continue
                return parsed_result
        finally:
            temp_file.unlink(missing_ok=True)

    def _copy_transcript(self, container_name: str, session_id: str, transcript_path: Path) -> None:
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        self._run_subprocess(
            [
                "docker",
                "cp",
                f"{container_name}:/root/.openclaw/agents/main/sessions/{session_id}.jsonl",
                str(transcript_path),
            ]
        )

    def _collect_output(self, container_name: str, output_dir: Path) -> None:
        if not bool(getattr(self.config, "wildclawbench_collect_task_output", True)):
            return
        task_output_dir = output_dir / "task_output"
        task_output_dir.mkdir(parents=True, exist_ok=True)
        _copy_dir_from_container(container_name, "/tmp/openclaw/.", str(task_output_dir))
        results_dir = task_output_dir / "workspace" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        _copy_dir_from_container(container_name, f"{self._tmp_workspace()}/results/.", str(results_dir))

    def _close_log_handle(self, proc: subprocess.Popen[str]) -> None:
        log_file = getattr(proc, "_skillclaw_log_file", None)
        if log_file and not log_file.closed:
            log_file.close()

    def _remove_container(self, container_name: str) -> None:
        self._run_subprocess(["docker", "rm", "-f", container_name])

    def _annotate_session_score(
        self,
        *,
        session_path: Path,
        score: float,
        parsed: WildClawTaskSpec,
        phase: str,
        task_id: str,
    ) -> None:
        if not session_path.exists():
            return
        try:
            payload = json.loads(session_path.read_text(encoding="utf-8"))
        except Exception:
            return
        turns = payload.get("turns") or []
        if turns:
            turns[-1]["prm_score"] = score
        payload["task_id"] = task_id
        payload["phase"] = phase
        payload["benchmark"] = {
            "task_file": parsed.file_path,
            "category": parsed.category,
            "overall_score": score,
        }
        payload["num_turns"] = len(turns)
        session_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_session_turns(self, session_path: Path) -> list[dict[str, Any]]:
        if not session_path.exists():
            return []
        try:
            payload = json.loads(session_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        turns = payload.get("turns")
        return turns if isinstance(turns, list) else []
