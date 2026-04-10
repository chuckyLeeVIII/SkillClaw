"""
Single-host multi-node OpenClaw cluster simulator.

Each simulated node gets:

- an isolated HOME / OPENCLAW_HOME
- its own SkillClaw config + proxy port
- its own skills and records directories

This lets benchmarks exercise "many machines using OpenClaw + SkillClaw"
without needing actual remote hosts.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import subprocess
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..cli import _healthz_ready

logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _count_jsonl_records(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
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


def _parse_json_from_stdout(stdout: str) -> dict[str, Any]:
    text = (stdout or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            payload = json.loads(text.splitlines()[-1])
        except Exception:
            return {}
    return payload if isinstance(payload, dict) else {}


def _extract_runtime_session_id(payload: dict[str, Any]) -> str:
    """Extract the real OpenClaw session id from a command JSON payload."""
    if not isinstance(payload, dict):
        return ""
    for key in ("sessionId", "session_id"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    result = payload.get("result")
    if isinstance(result, dict):
        meta = result.get("meta")
        if isinstance(meta, dict):
            agent_meta = meta.get("agentMeta")
            if isinstance(agent_meta, dict):
                for key in ("sessionId", "session_id"):
                    value = agent_meta.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
    return ""


def _is_port_available(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _resolve_available_port(
    preferred_port: int,
    *,
    used_ports: set[int],
    max_offset: int = 200,
) -> int:
    for offset in range(max_offset + 1):
        candidate = preferred_port + offset
        if candidate in used_ports:
            continue
        if _is_port_available(candidate):
            used_ports.add(candidate)
            return candidate
    raise RuntimeError(f"no available port near {preferred_port}")


def _wait_for_port_available(port: int, *, timeout_s: float = 10.0, poll_s: float = 0.2) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _is_port_available(port):
            return True
        time.sleep(poll_s)
    return _is_port_available(port)


def _extract_read_skill_paths(conversation_records: list[dict[str, Any]]) -> list[str]:
    paths: set[str] = set()
    for rec in conversation_records:
        tool_calls = rec.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            continue
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            func = tc.get("function") or {}
            if not isinstance(func, dict):
                continue
            tool_name = str(func.get("name") or "").strip().lower()
            if tool_name not in {"read", "file_read", "read_file", "readfile"}:
                continue
            try:
                args = json.loads(func.get("arguments") or "{}")
            except Exception:
                args = {}
            if not isinstance(args, dict):
                continue
            raw_path = str(args.get("path") or args.get("file") or "").strip()
            if not raw_path or not raw_path.endswith("SKILL.md"):
                continue
            paths.add(os.path.realpath(raw_path))
    return sorted(paths)


def _extract_read_skills(conversation_record: dict[str, Any]) -> list[dict[str, str]]:
    skill_names: list[str] = []
    tool_calls = conversation_record.get("tool_calls") or []
    if not isinstance(tool_calls, list):
        return []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        func = tc.get("function") or {}
        if not isinstance(func, dict):
            continue
        tool_name = str(func.get("name") or "").strip().lower()
        if tool_name not in {"read", "file_read", "read_file", "readfile"}:
            continue
        try:
            args = json.loads(func.get("arguments") or "{}")
        except Exception:
            args = {}
        if not isinstance(args, dict):
            continue
        raw_path = str(args.get("path") or args.get("file") or "").strip()
        if not raw_path or not raw_path.endswith("SKILL.md"):
            continue
        skill_name = Path(raw_path).parent.name.strip()
        if skill_name and skill_name not in skill_names:
            skill_names.append(skill_name)
    return [{"skill_id": "", "skill_name": name} for name in skill_names]


def _build_evolve_turns(
    conversation_records: list[dict[str, Any]],
    prm_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    prm_by_turn: dict[int, float] = {}
    for rec in prm_records:
        try:
            turn_num = int(rec.get("turn"))
            prm_by_turn[turn_num] = float(rec.get("score", 0.0))
        except Exception:
            continue

    def _norm_call_id(raw: str) -> str:
        """Normalize tool_call_id for matching (strip underscores/hyphens)."""
        return raw.replace("-", "").replace("_", "")

    # Build a map from normalized tool_call_id -> tool_name across all records
    call_id_to_name: dict[str, str] = {}
    for rec in conversation_records:
        for tc in rec.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            call_id = _norm_call_id(str(tc.get("id") or ""))
            func = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            name = str(func.get("name") or "")
            if call_id and name:
                call_id_to_name[call_id] = name

    _RESULT_CONTENT_MAX = 800

    def _extract_tool_results_from_messages(
        messages: list[dict[str, Any]],
        prev_msg_count: int,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """Extract tool result entries from NEW messages only (skip first prev_msg_count)."""
        tool_results: list[dict] = []
        tool_observations: list[dict] = []
        tool_errors: list[dict] = []
        for msg in messages[prev_msg_count:]:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            if role not in ("tool", "toolResult"):
                continue
            content_raw = msg.get("content")
            if isinstance(content_raw, list):
                content = " ".join(
                    str(c.get("text", "")) if isinstance(c, dict) else str(c)
                    for c in content_raw
                ).strip()
            else:
                content = str(content_raw or "")
            call_id = _norm_call_id(str(
                msg.get("tool_call_id")
                or msg.get("toolCallId")
                or ""
            ))
            tool_name = str(
                msg.get("toolName")
                or msg.get("name")
                or msg.get("tool_name")
                or call_id_to_name.get(call_id, "")
                or "unknown"
            )
            truncated = content[:_RESULT_CONTENT_MAX]
            has_error = bool(
                msg.get("is_error")
                or "Traceback" in content[:500]
                or "Error" in content[:500]
                or "error" in content[:200]
            )
            entry = {
                "tool_name": tool_name,
                "tool_call_id": call_id,
                "content": truncated,
                "has_error": has_error,
            }
            tool_results.append(entry)
            tool_observations.append(dict(entry))
            if has_error:
                tool_errors.append(dict(entry))
        return tool_results, tool_observations, tool_errors

    turns: list[dict[str, Any]] = []
    prev_msg_count = 0
    for idx, rec in enumerate(conversation_records, start=1):
        try:
            turn_num = int(rec.get("turn"))
        except Exception:
            turn_num = idx

        messages = rec.get("messages") or []

        # Extract only the NEW tool result messages (messages are cumulative).
        tr, to, te = _extract_tool_results_from_messages(messages, prev_msg_count)

        # Tool results in this record's messages belong to the PREVIOUS turn's
        # tool calls.  Attach them there.
        if tr and turns:
            prev = turns[-1]
            prev["tool_results"] = tr
            prev["tool_observations"] = to
            prev["tool_errors"] = te

        turns.append(
            {
                "turn_num": turn_num,
                "prompt_text": str(rec.get("instruction_text") or rec.get("prompt_text") or ""),
                "response_text": str(rec.get("response_text") or ""),
                "tool_calls": rec.get("tool_calls") or [],
                "tool_results": [],
                "tool_observations": [],
                "tool_errors": [],
                "read_skills": _extract_read_skills(rec),
                "injected_skills": [],
                "prm_score": prm_by_turn.get(turn_num),
            }
        )
        prev_msg_count = len(messages)

    return turns


def _is_under_root(path: str, root: Path) -> bool:
    try:
        path_obj = Path(path).resolve()
        root_obj = root.resolve()
        path_obj.relative_to(root_obj)
        return True
    except Exception:
        return False


@dataclass(frozen=True)
class OpenClawClusterNode:
    node_id: str
    index: int
    root_dir: Path
    home_dir: Path
    openclaw_dir: Path
    skills_dir: Path
    records_dir: Path
    log_dir: Path
    skillclaw_config_path: Path
    skillclaw_log_path: Path
    gateway_log_path: Path
    proxy_port: int
    gateway_port: int


@dataclass(frozen=True)
class OpenClawClusterSettings:
    configured_nodes: int = 8
    active_nodes: int = 3
    skillclaw_base_port: int = 39000
    gateway_base_port: int = 41000
    openclaw_bin: str = "openclaw"
    skillclaw_bin: str = "skillclaw"
    node_command_timeout_s: int = 180
    start_timeout_s: float = 20.0
    openclaw_mode: str = "gateway"
    llm_provider: str = "custom"
    llm_api_base: str = ""
    llm_api_key: str = ""
    llm_model_id: str = ""
    proxy_api_key: str = ""
    public_skill_root: str = ""
    skills_enabled: bool = True
    retrieval_mode: str = "template"
    max_context_tokens: int = 20000
    prm_enabled: bool = False
    prm_provider: str = "openai"
    prm_url: str = "https://api.openai.com/v1"
    prm_model: str = "gpt-5.2"
    prm_api_key: str = ""
    seed_openclaw_dir: str = ""


class OpenClawClusterManager:
    """Provision, start, stop, and query isolated OpenClaw nodes."""

    def __init__(
        self,
        *,
        workspace_dir: Path,
        cloud_dir: Path,
        group_id: str,
        settings: OpenClawClusterSettings,
    ):
        self.workspace_dir = workspace_dir.expanduser().resolve()
        self.cloud_dir = cloud_dir.expanduser().resolve()
        self.group_id = group_id
        self.settings = settings
        self.nodes: dict[str, OpenClawClusterNode] = {}
        self._active_ids: list[str] = []
        self._gateway_processes: dict[str, subprocess.Popen] = {}

    def prepare(self, *, initial_skills_dir: Path | None = None) -> dict[str, OpenClawClusterNode]:
        configured = max(self.settings.configured_nodes, self.settings.active_nodes)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.cloud_dir.mkdir(parents=True, exist_ok=True)
        used_proxy_ports: set[int] = set()
        used_gateway_ports: set[int] = set()

        for idx in range(configured):
            node_id = f"device-{idx}"
            root_dir = self.workspace_dir / node_id
            home_dir = root_dir / "home"
            openclaw_dir = home_dir / ".openclaw"
            skills_dir = root_dir / "skills"
            records_dir = root_dir / "records"
            log_dir = root_dir / "logs"
            preferred_proxy_port = self.settings.skillclaw_base_port + idx
            # OpenClaw may spawn extra listeners around gateway_port (observed
            # collisions at +3 on some versions), so keep a wider stride.
            preferred_gateway_port = self.settings.gateway_base_port + (idx * 10)
            proxy_port = _resolve_available_port(
                preferred_proxy_port,
                used_ports=used_proxy_ports,
            )
            gateway_port = _resolve_available_port(
                preferred_gateway_port,
                used_ports=used_gateway_ports,
            )
            node = OpenClawClusterNode(
                node_id=node_id,
                index=idx,
                root_dir=root_dir,
                home_dir=home_dir,
                openclaw_dir=openclaw_dir,
                skills_dir=skills_dir,
                records_dir=records_dir,
                log_dir=log_dir,
                skillclaw_config_path=home_dir / ".skillclaw" / "config.yaml",
                skillclaw_log_path=log_dir / "skillclaw.log",
                gateway_log_path=log_dir / "openclaw_gateway.log",
                proxy_port=proxy_port,
                gateway_port=gateway_port,
            )
            self._prepare_node(node, initial_skills_dir=initial_skills_dir)
            self.nodes[node_id] = node

        self._active_ids = [f"device-{idx}" for idx in range(self.settings.active_nodes)]
        return dict(self.nodes)

    def start_active_nodes(self, *, start_gateway: bool = True) -> None:
        started_ids: list[str] = []
        try:
            for node_id in self._active_ids:
                self.start_node(node_id, start_gateway=start_gateway)
                started_ids.append(node_id)
        except Exception:
            for started_id in reversed(started_ids):
                with suppress(Exception):
                    self.stop_node(started_id)
            raise

    def start_node(self, node_id: str, *, start_gateway: bool = True, _retries: int = 3) -> None:
        node = self.nodes[node_id]
        env = self._node_env(node)
        last_err: Exception | None = None
        for attempt in range(1, _retries + 1):
            with suppress(Exception):
                self.stop_node(node_id)
            _wait_for_port_available(node.proxy_port, timeout_s=self.settings.start_timeout_s)
            cmd = [
                self.settings.skillclaw_bin,
                "start",
                "--daemon",
                "--port",
                str(node.proxy_port),
                "--log-file",
                str(node.skillclaw_log_path),
            ]
            logger.info("[Cluster] starting %s on proxy :%d (attempt %d/%d)", node_id, node.proxy_port, attempt, _retries)
            proc = subprocess.run(
                cmd,
                cwd=str(node.root_dir),
                env=env,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.settings.node_command_timeout_s,
            )
            if proc.returncode != 0:
                last_err = RuntimeError(
                    f"failed to start {node_id}: "
                    f"{(proc.stderr or proc.stdout or '').strip()[:400]}"
                )
                if attempt < _retries:
                    logger.warning("[Cluster] %s start failed (attempt %d/%d), retrying in 5s …", node_id, attempt, _retries)
                    time.sleep(5)
                    continue
                raise last_err
            if not _healthz_ready(node.proxy_port, timeout=self.settings.start_timeout_s):
                last_err = RuntimeError(f"skillclaw proxy for {node_id} did not become healthy")
                if attempt < _retries:
                    logger.warning("[Cluster] %s healthz failed (attempt %d/%d), retrying in 5s …", node_id, attempt, _retries)
                    time.sleep(5)
                    continue
                raise last_err
            # Success
            break
        if start_gateway:
            self._start_gateway(node, env=env)

    def stop_active_nodes(self) -> None:
        for node_id in reversed(self._active_ids):
            self.stop_node(node_id)

    def stop_node(self, node_id: str) -> None:
        node = self.nodes[node_id]
        env = self._node_env(node)
        subprocess.run(
            [self.settings.skillclaw_bin, "stop"],
            cwd=str(node.root_dir),
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=self.settings.node_command_timeout_s,
        )
        gateway_proc = self._gateway_processes.pop(node_id, None)
        if gateway_proc is not None:
            gateway_proc.terminate()
            try:
                gateway_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                gateway_proc.kill()
                gateway_proc.wait(timeout=10)
        _wait_for_port_available(node.proxy_port, timeout_s=self.settings.start_timeout_s)

    def invoke_task(
        self,
        *,
        node_id: str,
        instruction: str,
        round_idx: int,
        phase: str,
        task_id: str,
    ) -> dict[str, Any]:
        node = self.nodes[node_id]
        requested_session_id = f"{phase}-{task_id}-r{round_idx}-{uuid.uuid4().hex[:8]}"
        env = self._node_env(node)
        conversation_path = node.records_dir / "conversations.jsonl"
        prm_path = node.records_dir / "prm_scores.jsonl"
        conv_start = _count_jsonl_records(conversation_path)
        prm_start = _count_jsonl_records(prm_path)
        started = self._run_openclaw_agent_once(
            node=node,
            env=env,
            session_id=requested_session_id,
            instruction=instruction,
        )
        conv_end = _count_jsonl_records(conversation_path)
        prm_end = _count_jsonl_records(prm_path)
        if started.returncode == 0 and conv_end <= conv_start:
            for _ in range(20):
                time.sleep(0.1)
                conv_end = _count_jsonl_records(conversation_path)
                prm_end = _count_jsonl_records(prm_path)
                if conv_end > conv_start:
                    break
        stdout = (started.stdout or "").strip()
        payload = _parse_json_from_stdout(stdout)
        runtime_session_id = _extract_runtime_session_id(payload)
        if started.returncode == 0 and conv_end <= conv_start:
            # Some OpenClaw/SkillClaw paths flush the previous pending turn only
            # when the next request for the same session arrives.
            flush_session_id = runtime_session_id or requested_session_id
            with suppress(Exception):
                self._run_openclaw_agent_once(
                    node=node,
                    env=env,
                    session_id=flush_session_id,
                    instruction="__skillclaw_flush_turn__",
                )
                for _ in range(30):
                    time.sleep(0.1)
                    conv_end = _count_jsonl_records(conversation_path)
                    prm_end = _count_jsonl_records(prm_path)
                    if conv_end > conv_start:
                        break
        return {
            "node_id": node_id,
            "session_id": runtime_session_id or requested_session_id,
            "requested_session_id": requested_session_id,
            "runtime_session_id": runtime_session_id,
            "returncode": started.returncode,
            "stdout": stdout,
            "stderr": (started.stderr or "").strip(),
            "payload": payload,
            "conversation_slice": {"start": conv_start, "end": conv_end},
            "prm_slice": {"start": prm_start, "end": prm_end},
        }

    def _run_openclaw_agent_once(
        self,
        *,
        node: OpenClawClusterNode,
        env: dict[str, str],
        session_id: str,
        instruction: str,
    ) -> subprocess.CompletedProcess[str]:
        cmd = [
            self.settings.openclaw_bin,
            "agent",
            "--session-id",
            session_id,
            "--agent",
            "main",
            "--message",
            instruction,
            "--json",
            "--timeout",
            str(self.settings.node_command_timeout_s),
        ]
        if self.settings.openclaw_mode == "local":
            cmd.append("--local")
        return subprocess.run(
            cmd,
            cwd=str(node.root_dir),
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=self.settings.node_command_timeout_s,
        )

    def collect_session_artifacts(
        self,
        *,
        node_id: str,
        session_id: str,
        requested_session_id: str,
        task_id: str,
        phase: str,
        round_idx: int,
        skill_names: list[str],
        conversation_slice: dict[str, int] | None = None,
        prm_slice: dict[str, int] | None = None,
        payload_mode: str = "full",
    ) -> dict[str, Any]:
        node = self.nodes[node_id]
        conversation_path = node.records_dir / "conversations.jsonl"
        prm_path = node.records_dir / "prm_scores.jsonl"
        stats_path = node.skills_dir / "skill_stats.json"
        all_conversations = _load_jsonl(conversation_path)
        all_prm = _load_jsonl(prm_path)

        conv_start = int((conversation_slice or {}).get("start", 0))
        conv_end = int((conversation_slice or {}).get("end", 0))
        prm_start = int((prm_slice or {}).get("start", 0))
        prm_end = int((prm_slice or {}).get("end", 0))

        conversation_records: list[dict[str, Any]]
        conversation_source = "session_id"
        if 0 <= conv_start <= conv_end <= len(all_conversations) and conv_end > conv_start:
            conversation_records = all_conversations[conv_start:conv_end]
            conversation_source = "slice"
        else:
            conversation_records = [
                rec for rec in all_conversations
                if rec.get("session_id") == session_id
            ]
            if not conversation_records and requested_session_id:
                conversation_records = [
                    rec for rec in all_conversations
                    if rec.get("session_id") == requested_session_id
                ]
                conversation_source = "requested_session_id"

        prm_records: list[dict[str, Any]]
        if 0 <= prm_start <= prm_end <= len(all_prm) and prm_end > prm_start:
            prm_records = all_prm[prm_start:prm_end]
        else:
            prm_records = [
                rec for rec in all_prm
                if rec.get("session_id") == session_id
            ]
            if not prm_records and requested_session_id:
                prm_records = [
                    rec for rec in all_prm
                    if rec.get("session_id") == requested_session_id
                ]
        skill_stats = _load_json(stats_path)
        read_skill_paths = _extract_read_skill_paths(conversation_records)
        expected_root = node.skills_dir.resolve()
        out_of_root_paths = [
            p for p in read_skill_paths
            if not _is_under_root(p, expected_root)
        ]
        artifact_id = f"{phase}-{task_id}-{node_id}-r{round_idx}-{uuid.uuid4().hex[:8]}"
        evolve_turns = _build_evolve_turns(conversation_records, prm_records)
        payload_mode_normalized = str(payload_mode or "full").strip().lower()
        if payload_mode_normalized not in {"full", "minimal"}:
            raise ValueError(f"unsupported session payload mode: {payload_mode}")
        skill_stats_payload = {
            name: skill_stats.get(name, {})
            for name in skill_names
        }
        session_payload = {
            "artifact_id": artifact_id,
            "session_id": session_id,
            "requested_session_id": requested_session_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "user_alias": node_id,
            "num_turns": len(evolve_turns),
            "turns": evolve_turns,
            "task_id": task_id,
            "phase": phase,
            "round": round_idx,
            "device_id": node_id,
            "conversation_source": conversation_source,
            "conversation_count": len(conversation_records),
            "prm_count": len(prm_records),
        }
        if payload_mode_normalized == "full":
            session_payload.update(
                {
                    "conversation_records": conversation_records,
                    "prm_records": prm_records,
                    "skill_stats": skill_stats_payload,
                    "read_skill_paths": read_skill_paths,
                    "expected_skill_roots": [str(expected_root)],
                    "read_skill_paths_outside_expected": out_of_root_paths,
                }
            )
        session_dir = self.cloud_dir / self.group_id / "sessions"
        session_dir.mkdir(parents=True, exist_ok=True)
        session_path = session_dir / f"{artifact_id}.json"
        session_path.write_text(
            json.dumps(session_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return {
            "artifact_id": artifact_id,
            "session_id": session_id,
            "requested_session_id": requested_session_id,
            "conversation_count": len(conversation_records),
            "prm_count": len(prm_records),
            "session_path": str(session_path),
            "skill_stats": skill_stats_payload,
            "conversation_source": conversation_source,
            "read_skill_paths": read_skill_paths,
            "expected_skill_roots": [str(expected_root)],
            "read_skill_paths_outside_expected": out_of_root_paths,
        }

    def _prepare_node(
        self,
        node: OpenClawClusterNode,
        *,
        initial_skills_dir: Path | None = None,
    ) -> None:
        if node.home_dir.exists():
            shutil.rmtree(node.home_dir)
        if node.records_dir.exists():
            shutil.rmtree(node.records_dir)
        if node.log_dir.exists():
            shutil.rmtree(node.log_dir)
        if initial_skills_dir and node.skills_dir.exists():
            shutil.rmtree(node.skills_dir)

        node.home_dir.mkdir(parents=True, exist_ok=True)
        node.openclaw_dir.mkdir(parents=True, exist_ok=True)
        node.skills_dir.mkdir(parents=True, exist_ok=True)
        node.records_dir.mkdir(parents=True, exist_ok=True)
        node.log_dir.mkdir(parents=True, exist_ok=True)

        self._patch_openclaw_config(node)
        self._initialize_clean_agent_state(node)
        if initial_skills_dir and initial_skills_dir.exists():
            shutil.copytree(initial_skills_dir, node.skills_dir, dirs_exist_ok=True)
        self._write_skillclaw_config(node)

    def _initialize_clean_agent_state(self, node: OpenClawClusterNode) -> None:
        agents_root = node.openclaw_dir / "agents"
        agents_root.mkdir(parents=True, exist_ok=True)
        known_agents = {p.name for p in agents_root.iterdir() if p.is_dir()}
        known_agents.add("main")
        for agent_name in sorted(known_agents):
            sessions_dir = agents_root / agent_name / "sessions"
            sessions_dir.mkdir(parents=True, exist_ok=True)
            sessions_path = sessions_dir / "sessions.json"
            sessions_path.write_text("{}\n", encoding="utf-8")

    def _patch_openclaw_config(self, node: OpenClawClusterNode) -> None:
        config_path = node.openclaw_dir / "openclaw.json"
        data = _load_json(config_path)
        data.setdefault("gateway", {})
        data["gateway"]["mode"] = "local"
        data["gateway"]["bind"] = "loopback"
        data["gateway"]["port"] = node.gateway_port
        auth = data["gateway"].setdefault("auth", {})
        auth.setdefault("mode", "token")
        auth.setdefault("token", uuid.uuid4().hex)
        # Fresh nodes should not inherit a stale remote gateway target.
        data["gateway"].pop("remote", None)

        llm_model = self.settings.llm_model_id or "skillclaw-model"
        models = data.setdefault("models", {})
        providers = models.setdefault("providers", {})
        providers["skillclaw"] = {
            "api": "openai-completions",
            "baseUrl": f"http://127.0.0.1:{node.proxy_port}/v1",
            "apiKey": self.settings.proxy_api_key or "skillclaw",
            "models": [
                {
                    "id": llm_model,
                    "name": llm_model,
                    "reasoning": False,
                    "input": ["text"],
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                    "contextWindow": 32768,
                    "maxTokens": 8192,
                }
            ],
        }

        agents = data.setdefault("agents", {})
        defaults = agents.setdefault("defaults", {})
        model_defaults = defaults.setdefault("model", {})
        model_defaults["primary"] = f"skillclaw/{llm_model}"
        sandbox_defaults = defaults.setdefault("sandbox", {})
        sandbox_defaults["mode"] = "off"
        defaults["workspace"] = str((node.root_dir / "workspace").resolve())

        for entry in agents.get("list", []) or []:
            if not isinstance(entry, dict):
                continue
            agent_id = str(entry.get("id") or "main").strip() or "main"
            if agent_id == "main":
                entry["workspace"] = defaults["workspace"]
            else:
                entry["workspace"] = str((node.root_dir / f"workspace-{agent_id}").resolve())
            entry["agentDir"] = str((node.openclaw_dir / "agents" / agent_id / "agent").resolve())

        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self._initialize_workspace_dirs(node, data)

    def _initialize_workspace_dirs(self, node: OpenClawClusterNode, config: dict[str, Any]) -> None:
        agents = config.get("agents", {}) if isinstance(config, dict) else {}
        defaults = agents.get("defaults", {}) if isinstance(agents, dict) else {}
        default_workspace = str(defaults.get("workspace", "") or "")
        if default_workspace:
            Path(default_workspace).mkdir(parents=True, exist_ok=True)
        for entry in agents.get("list", []) or []:
            if not isinstance(entry, dict):
                continue
            workspace = str(entry.get("workspace", "") or "")
            agent_dir = str(entry.get("agentDir", "") or "")
            if workspace:
                Path(workspace).mkdir(parents=True, exist_ok=True)
            if agent_dir:
                Path(agent_dir).mkdir(parents=True, exist_ok=True)

    def _write_skillclaw_config(self, node: OpenClawClusterNode) -> None:
        config = {
            "mode": "skills_only",
            "claw_type": "none",
            "configure_openclaw": False,
            "llm": {
                "provider": self.settings.llm_provider,
                "model_id": self.settings.llm_model_id,
                "api_base": self.settings.llm_api_base,
                "api_key": self.settings.llm_api_key,
            },
            "max_context_tokens": int(self.settings.max_context_tokens or 20000),
            "proxy": {
                "port": node.proxy_port,
                # Nodes may be reached from WildClawBench task containers via
                # host.docker.internal, so bind on all interfaces.
                "host": "0.0.0.0",
                "api_key": self.settings.proxy_api_key,
            },
            "skills": {
                "enabled": bool(self.settings.skills_enabled),
                "dir": str(node.skills_dir),
                "public_root": self.settings.public_skill_root,
                "retrieval_mode": self.settings.retrieval_mode,
                "top_k": 6,
            },
            "prm": {
                "enabled": self.settings.prm_enabled,
                "provider": self.settings.prm_provider,
                "url": self.settings.prm_url,
                "model": self.settings.prm_model,
                "api_key": self.settings.prm_api_key,
            },
            "sharing": {
                "enabled": False,
            },
        }
        node.skillclaw_config_path.parent.mkdir(parents=True, exist_ok=True)
        with node.skillclaw_config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    def _node_env(self, node: OpenClawClusterNode) -> dict[str, str]:
        env = dict(os.environ)
        env["HOME"] = str(node.home_dir)
        env["OPENCLAW_HOME"] = str(node.home_dir)
        env["OPENCLAW_CONFIG_PATH"] = str(node.openclaw_dir / "openclaw.json")
        env["OPENCLAW_PROFILE"] = node.node_id
        env["SKILLCLAW_DAEMON_READY_TIMEOUT_S"] = str(self.settings.start_timeout_s)
        # Skip slow HuggingFace tokenizer download attempts — the tokenizer
        # is optional (used only for rough token counting) and gpt-5.4 does
        # not have a public HF repo.  Without this, each daemon wastes ~96s
        # retrying unreachable HF endpoints, which can cause health-check
        # timeouts and spurious port-bind failures.
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        return env

    def _start_gateway(self, node: OpenClawClusterNode, *, env: dict[str, str]) -> None:
        logger.info("[Cluster] starting %s gateway on :%d", node.node_id, node.gateway_port)
        node.gateway_log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = node.gateway_log_path.open("ab")
        cmd = [self.settings.openclaw_bin, "--profile", node.node_id, "gateway"]
        proc = subprocess.Popen(
            cmd,
            cwd=str(node.root_dir),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            close_fds=True,
        )
        log_handle.close()
        self._gateway_processes[node.node_id] = proc
        self._wait_for_gateway_ready(node, proc)

    def _wait_for_gateway_ready(self, node: OpenClawClusterNode, proc: subprocess.Popen) -> None:
        deadline = time.monotonic() + self.settings.start_timeout_s
        gateway_url = f"http://127.0.0.1:{node.gateway_port}/"
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"gateway for {node.node_id} exited early; see {node.gateway_log_path}"
                )
            try:
                from urllib.request import urlopen

                with urlopen(gateway_url, timeout=1.0) as resp:
                    if resp.status == 200:
                        return
            except Exception:
                time.sleep(0.2)
        raise RuntimeError(
            f"gateway for {node.node_id} did not become reachable; see {node.gateway_log_path}"
        )
