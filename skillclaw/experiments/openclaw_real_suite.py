"""
Real OpenClaw + SkillClaw regression suite.

Runs a small set of live scenarios against a local SkillClaw proxy while
driving a real OpenClaw agent process. The suite captures:

- single-turn and multi-turn sessions
- skill loading / skill read behavior
- a direct PRM probe
- failure-driven skill evolution via EvolveServer
- post-evolution re-run

The runner writes a JSON report plus a Markdown summary.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import socket
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import httpx
import yaml

from evolve_server.config import EvolveServerConfig
from evolve_server.server import EvolveServer

from ..api_server import SkillClawAPIServer
from ..config import SkillClawConfig
from ..data_formatter import ConversationSample
from ..prm_scorer import PRMScorer
from ..skill_manager import SkillManager

logger = logging.getLogger(__name__)


def _now_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _parse_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except Exception:
        try:
            payload = json.loads(text.splitlines()[-1])
        except Exception:
            return {}
    return payload if isinstance(payload, dict) else {}


def _extract_agent_response(stdout: str) -> str:
    payload = _parse_json_object(stdout)
    if not payload:
        return stdout.strip()
    payloads = payload.get("payloads") or []
    if isinstance(payloads, list):
        texts = [
            str(item.get("text") or "")
            for item in payloads
            if isinstance(item, dict) and item.get("text")
        ]
        if texts:
            return "\n".join(texts).strip()
    return stdout.strip()


def _build_session_payload(session_id: str, turns: list[dict[str, Any]], user_alias: str) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "user_alias": user_alias,
        "num_turns": len(turns),
        "turns": turns,
    }


def _match_skill_names(turn: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for item in turn.get("read_skills", []) or []:
        if isinstance(item, dict):
            name = str(item.get("skill_name") or "").strip()
            if name:
                out.append(name)
    if out:
        return out

    tool_calls = turn.get("tool_calls", []) or []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        func = tc.get("function") or {}
        if not isinstance(func, dict):
            continue
        if str(func.get("name") or "").strip() != "read":
            continue
        args_raw = func.get("arguments", "{}")
        if not isinstance(args_raw, str):
            try:
                args_raw = json.dumps(args_raw, ensure_ascii=False)
            except Exception:
                args_raw = "{}"
        try:
            args = json.loads(args_raw)
        except Exception:
            args = {}
        path = str(args.get("path") or "")
        if not path:
            continue
        if "/skills/" in path and path.endswith("/SKILL.md"):
            skill_name = path.split("/skills/", 1)[1].split("/", 1)[0].strip()
            if skill_name:
                out.append(skill_name)
        elif path.endswith("/SKILL.md"):
            skill_name = Path(path).parent.name.strip()
            if skill_name:
                out.append(skill_name)
    return out


def _session_turns_from_state(session_turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for turn in session_turns:
        if not isinstance(turn, dict):
            continue
        out.append({
            "turn_num": int(turn.get("turn_num", 0) or 0),
            "prompt_text": turn.get("prompt_text", ""),
            "response_text": turn.get("response_text", ""),
            "tool_calls": turn.get("tool_calls", []) or [],
            "read_skills": turn.get("read_skills", []) or [],
            "tool_results": turn.get("tool_results", []) or [],
            "tool_observations": turn.get("tool_observations", []) or [],
            "tool_errors": turn.get("tool_errors", []) or [],
            "injected_skills": turn.get("injected_skills", []) or [],
            "prm_score": turn.get("prm_score"),
        })
    return out


@dataclass
class ScenarioTurn:
    message: str
    session_done: bool = False
    turn_type: str = "main"


@dataclass
class ScenarioSpec:
    name: str
    session_id: str
    turns: list[ScenarioTurn]
    note: str = ""
    expect_read_skill: str = ""


@dataclass
class OpenClawRealSuiteConfig:
    name: str = "openclaw-real-suite"
    workspace_dir: str = "records/openclaw_real_suite"
    group_id: str = "real-suite"
    agent_workspace_name: str = "openclaw-workspace"
    openclaw_home_dir: str = ""
    openclaw_bin: str = "openclaw"
    openclaw_agent_name: str = "live"
    openclaw_model_id: str = "skillclaw/gpt-5.2-1211-global"
    openclaw_base_model: str = "anthropic/claude-opus-4-6"
    llm_api_base: str = ""
    llm_api_key: str = ""
    llm_model_id: str = "gpt-5.2-1211-global"
    prm_api_base: str = ""
    prm_api_key: str = ""
    prm_model_id: str = "gpt-5.2-1211-global"
    prm_m: int = 1
    prm_temperature: float = 0.2
    prm_max_tokens: int = 768
    retrieval_mode: str = "template"
    run_evolution: bool = True
    evolve_model_id: str = ""
    evolve_max_tokens: int = 100000
    scenario_timeout_s: int = 300
    proxy_port: int = 0

    @classmethod
    def from_file(cls, path: str) -> "OpenClawRealSuiteConfig":
        cfg_path = Path(path).expanduser().resolve()
        raw = cfg_path.read_text(encoding="utf-8")
        if cfg_path.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(raw) or {}
        else:
            data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("real suite config must deserialize to a dict")
        base_dir = cfg_path.parent
        for key in ("workspace_dir", "openclaw_home_dir"):
            value = data.get(key)
            if isinstance(value, str) and value and not Path(value).expanduser().is_absolute():
                data[key] = str((base_dir / value).resolve())
        return cls(**data)


@dataclass
class ScenarioResult:
    scenario: ScenarioSpec
    runtime_session_id: str = ""
    returncodes: list[int] = field(default_factory=list)
    assistant_texts: list[str] = field(default_factory=list)
    stdout_snippets: list[str] = field(default_factory=list)
    stderr_snippets: list[str] = field(default_factory=list)
    session_turns: list[dict[str, Any]] = field(default_factory=list)
    record_turns: list[dict[str, Any]] = field(default_factory=list)
    success: bool = False
    notes: str = ""


class OpenClawRealSuiteRunner:
    def __init__(self, config: OpenClawRealSuiteConfig):
        self.config = config
        self.base_root = Path(config.workspace_dir).expanduser().resolve()
        self.run_root = _ensure_dir(self.base_root / _now_stamp())
        self.report_dir = _ensure_dir(self.run_root / "reports")
        self.artifact_dir = _ensure_dir(self.run_root / "artifacts")
        self.agent_workspace = _ensure_dir(self.run_root / config.agent_workspace_name)
        self.temp_root = Path(config.openclaw_home_dir).expanduser().resolve() if config.openclaw_home_dir else None
        self.openclaw_home = self.temp_root or Path(tempfile.mkdtemp(prefix="openclaw-suite-home-", dir=str(self.run_root)))
        _ensure_dir(self.openclaw_home)
        self.openclaw_home_state = _ensure_dir(self.openclaw_home / ".openclaw")
        self.skills_backend_root = _ensure_dir(self.run_root / "evolve_backend")
        self.skills_dir = _ensure_dir(self.skills_backend_root / config.group_id / "skills")
        self.sessions_dir = _ensure_dir(self.skills_backend_root / config.group_id / "sessions")
        self.processed_log = self.artifact_dir / "evolve_processed.json"
        self.history_log = self.artifact_dir / "evolve_history.jsonl"
        self.server_port = int(config.proxy_port or 0)
        self.record_dir = _ensure_dir(self.run_root / "skillclaw-records")
        self._submission_enabled = threading.Event()
        self._submission_enabled.set()
        self.skill_manager = SkillManager(
            skills_dir=str(self.skills_dir),
            public_skill_root=str(self.skills_dir),
            retrieval_mode=config.retrieval_mode,
        )
        self.skillclaw_config = SkillClawConfig(
            claw_type="openclaw",
            llm_provider="custom",
            llm_api_base=config.llm_api_base or os.environ.get("OPENAI_BASE_URL", ""),
            llm_api_key=config.llm_api_key or os.environ.get("OPENAI_API_KEY", ""),
            llm_model_id=config.llm_model_id,
            prm_provider="openai",
            prm_url=config.prm_api_base or config.llm_api_base or os.environ.get("OPENAI_BASE_URL", ""),
            prm_api_key=config.prm_api_key or config.llm_api_key or os.environ.get("OPENAI_API_KEY", ""),
            prm_model=config.prm_model_id,
            prm_m=config.prm_m,
            prm_temperature=config.prm_temperature,
            prm_max_new_tokens=config.prm_max_tokens,
            use_prm=True,
            use_skills=True,
            skills_dir=str(self.skills_dir),
            retrieval_mode=config.retrieval_mode,
            record_enabled=True,
            record_dir=str(self.record_dir),
            sharing_enabled=False,
            proxy_host="127.0.0.1",
            proxy_port=self.server_port,
            proxy_api_key="skillclaw",
            served_model_name=config.llm_model_id,
        )
        self.prm_scorer = PRMScorer(
            prm_url=self.skillclaw_config.prm_url,
            prm_model=self.skillclaw_config.prm_model,
            api_key=self.skillclaw_config.prm_api_key,
            prm_m=self.skillclaw_config.prm_m,
            temperature=self.skillclaw_config.prm_temperature,
            max_new_tokens=self.skillclaw_config.prm_max_new_tokens,
        )
        self.server: Optional[SkillClawAPIServer] = None
        self.scenario_results: list[ScenarioResult] = []
        self.prm_probe_result: dict[str, Any] = {}
        self.evolve_result: dict[str, Any] = {}
        self.direct_records: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Setup                                                               #
    # ------------------------------------------------------------------ #

    def _write_skill(self, name: str, description: str, content: str, category: str = "general") -> Path:
        skill_dir = _ensure_dir(self.skills_dir / name)
        frontmatter = [
            "---",
            f"name: {name}",
            f"description: {description}",
            "metadata:",
            "  skillclaw:",
            f"    category: {category}",
            "---",
            "",
            content.strip(),
            "",
        ]
        skill_path = skill_dir / "SKILL.md"
        skill_path.write_text("\n".join(frontmatter), encoding="utf-8")
        return skill_path

    def _seed_workspace_files(self) -> None:
        _ensure_dir(self.agent_workspace / "notes")
        (self.agent_workspace / "README.md").write_text(
            "# OpenClaw Real Suite\n\nThis workspace is used to verify live sessions.\n",
            encoding="utf-8",
        )
        (self.agent_workspace / "notes" / "reference.md").write_text(
            "# Reference Note\n\nThis file exists so the skill-read scenario has a stable target.\n",
            encoding="utf-8",
        )
        (self.agent_workspace / "notes" / "multi_turn.md").write_text(
            "# Multi-turn Note\n\nThe second turn should summarize the same idea more compactly.\n",
            encoding="utf-8",
        )
        self._write_skill(
            name="workspace-file-reader",
            description=(
                "Use when the user asks to inspect, quote, or summarize a file in the current workspace. "
                "NOT for tasks that already provide the needed file contents."
            ),
            content=(
                "# Workspace File Reader\n\n"
                "## When to Use\n"
                "- The user asks for the first paragraph, summary, or exact quote from a workspace file.\n"
                "- The user names a file in the current workspace and expects you to inspect it.\n\n"
                "## When NOT to Use\n"
                "- The file contents are already provided in the prompt.\n"
                "- The task is not about reading workspace files.\n\n"
                "## Procedure\n"
                "1. Use `read` on the exact workspace path.\n"
                "2. If the file is missing, say the path was not found and ask for the correct location.\n"
                "3. Quote or summarize only what was actually read.\n\n"
                "## Common Mistakes\n"
                "- Guessing the file contents.\n"
                "- Reading the wrong directory.\n"
                "- Ignoring a missing-file error.\n"
            ),
            category="general",
        )
        # Reload after seeding so the live proxy injects the custom skill set.
        self.skill_manager.reload()

    def _configure_openclaw(self) -> None:
        agent_dir = self.openclaw_home_state / "agents" / self.config.openclaw_agent_name / "agent"
        _ensure_dir(agent_dir)
        env = os.environ.copy()
        env["HOME"] = str(self.openclaw_home)

        self._run_cmd([
            self.config.openclaw_bin,
            "agents",
            "add",
            self.config.openclaw_agent_name,
            "--non-interactive",
            "--workspace",
            str(self.agent_workspace),
            "--agent-dir",
            str(agent_dir),
            "--model",
            self.config.openclaw_base_model,
            "--json",
        ], env=env, cwd=self.agent_workspace)

        provider_cfg = {
            "api": "openai-completions",
            "baseUrl": f"http://127.0.0.1:{self.server_port}/v1",
            "apiKey": "skillclaw",
            "models": [
                {
                    "id": self.config.llm_model_id,
                    "name": self.config.llm_model_id,
                    "reasoning": False,
                    "input": ["text"],
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                    "contextWindow": 32768,
                    "maxTokens": 8192,
                    "api": "openai-completions",
                }
            ],
        }

        self._run_cmd([
            self.config.openclaw_bin,
            "config",
            "set",
            "models.providers.skillclaw",
            json.dumps(provider_cfg, ensure_ascii=False),
        ], env=env, cwd=self.agent_workspace)
        self._run_cmd([
            self.config.openclaw_bin,
            "config",
            "set",
            "agents.list.1.model",
            json.dumps(self.config.openclaw_model_id, ensure_ascii=False),
        ], env=env, cwd=self.agent_workspace)

    def _start_proxy(self) -> None:
        if not self.server_port:
            self.server_port = _free_port()
            self.skillclaw_config.proxy_port = self.server_port
        self.server = SkillClawAPIServer(
            config=self.skillclaw_config,
            output_queue=queue.Queue(),
            submission_enabled=self._submission_enabled,
            skill_manager=self.skill_manager,
            prm_scorer=self.prm_scorer,
        )
        self.server.start()
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            try:
                with httpx.Client(timeout=2.0) as client:
                    resp = client.get(f"http://127.0.0.1:{self.server_port}/healthz")
                    if resp.status_code == 200 and resp.json().get("ok") is True:
                        return
            except Exception:
                time.sleep(0.5)
        raise RuntimeError("SkillClaw proxy did not become healthy in time")

    def _run_cmd(self, cmd: list[str], *, env: dict[str, str], cwd: Path) -> subprocess.CompletedProcess[str]:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            timeout=self.config.scenario_timeout_s,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"command failed ({proc.returncode}): {' '.join(cmd)}\n"
                f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
            )
        return proc

    def _run_agent_turn(self, *, session_id: str, message: str, cwd: Path) -> dict[str, Any]:
        env = os.environ.copy()
        env["HOME"] = str(self.openclaw_home)
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
        proc = self._run_cmd(
            [
                self.config.openclaw_bin,
                "agent",
                "--local",
                "--agent",
                self.config.openclaw_agent_name,
                "--session-id",
                session_id,
                "--message",
                message,
                "--json",
                "--timeout",
                str(self.config.scenario_timeout_s),
            ],
            env=env,
            cwd=cwd,
        )
        payload = _parse_json_object(proc.stdout)
        meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
        agent_meta = meta.get("agentMeta", {}) if isinstance(meta, dict) else {}
        runtime_session_id = ""
        for key in ("sessionId", "session_id"):
            value = ""
            if isinstance(agent_meta, dict):
                value = str(agent_meta.get(key) or "").strip()
            if not value and isinstance(meta, dict):
                value = str(meta.get(key) or "").strip()
            if not value and isinstance(payload, dict):
                value = str(payload.get(key) or "").strip()
            if value:
                runtime_session_id = value
                break
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "assistant_text": _extract_agent_response(proc.stdout),
            "payload": payload,
            "session_id": runtime_session_id or session_id,
        }

    def _snapshot_turns(self, session_id: str) -> list[dict[str, Any]]:
        if self.server is None:
            return []
        turns = self.server._session_turns.get(session_id, [])
        return _session_turns_from_state(turns)

    def _conversation_records(self) -> list[dict[str, Any]]:
        return _load_jsonl(self.record_dir / "conversations.jsonl")

    def _merge_prm_scores(self, records: list[dict[str, Any]]) -> None:
        score_map: dict[tuple[str, int], dict[str, Any]] = {}
        for row in _load_jsonl(self.record_dir / "prm_scores.jsonl"):
            session_id = str(row.get("session_id") or "")
            turn = row.get("turn")
            try:
                turn_num = int(turn)
            except Exception:
                continue
            if session_id:
                score_map[(session_id, turn_num)] = row

        for record in records:
            session_id = str(record.get("session_id") or "")
            turn = record.get("turn")
            try:
                turn_num = int(turn)
            except Exception:
                continue
            score_row = score_map.get((session_id, turn_num))
            if not score_row:
                continue
            record["prm_score"] = score_row.get("score")
            if "votes" in score_row:
                record["prm_votes"] = score_row.get("votes")

    def _write_session_snapshots(self, sessions: dict[str, list[dict[str, Any]]]) -> list[Path]:
        out_paths: list[Path] = []
        session_root = _ensure_dir(self.sessions_dir)
        for session_id, turns in sessions.items():
            payload = _build_session_payload(session_id, turns, user_alias="real-suite")
            path = session_root / f"{session_id}.json"
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            out_paths.append(path)
        return out_paths

    def _run_prm_probe(self) -> dict[str, Any]:
        session_id = f"prm-probe-{uuid.uuid4().hex[:8]}"
        payload = {
            "model": self.config.llm_model_id,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "请用一句话回答：1+1等于几？"},
            ],
            "max_tokens": 32,
            "session_id": session_id,
            "session_done": True,
        }
        headers = {
            "Authorization": "Bearer skillclaw",
            "X-Session-Id": session_id,
            "X-Turn-Type": "main",
            "X-Session-Done": "true",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=180.0) as client:
            resp = client.post(
                f"http://127.0.0.1:{self.server_port}/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            body = resp.json()
        probe = {
            "session_id": session_id,
            "response_text": body["choices"][0]["message"]["content"],
            "http_status": resp.status_code,
            "conversation_record_count": len(_load_jsonl(self.record_dir / "conversations.jsonl")),
            "prm_record_count": len(_load_jsonl(self.record_dir / "prm_scores.jsonl")),
        }
        self.prm_probe_result = probe
        return probe

    def _run_evolution(self, session_ids: list[str]) -> dict[str, Any]:
        evolve_root = self.skills_backend_root
        evolve_cfg = EvolveServerConfig(
            group_id=self.config.group_id,
            local_root=str(evolve_root),
            llm_api_key=self.skillclaw_config.llm_api_key,
            llm_base_url=self.skillclaw_config.llm_api_base,
            llm_model=self.config.evolve_model_id or self.skillclaw_config.llm_model_id,
            llm_max_tokens=self.config.evolve_max_tokens,
            processed_log_path=str(self.processed_log),
            history_path=str(self.history_log),
        )
        server = EvolveServer(evolve_cfg)
        summary = asyncio.run(server.run_once())
        self.skill_manager.reload()
        self.evolve_result = {
            "summary": summary,
            "skills_dir": str(self.skills_dir),
            "manifest_path": str(evolve_root / self.config.group_id / "manifest.jsonl"),
            "registry_path": str(evolve_root / self.config.group_id / "evolve_skill_registry.json"),
            "session_ids": session_ids,
        }
        return self.evolve_result

    def _build_scenarios(self) -> list[ScenarioSpec]:
        return [
            ScenarioSpec(
                name="simple-single-turn",
                session_id="real-simple-1",
                turns=[
                    ScenarioTurn(message="请直接回答：12 * 12 等于多少？只输出数字。", session_done=True),
                ],
                note="单轮简单题，验证基础请求链路。",
            ),
            ScenarioSpec(
                name="multi-turn-follow-up",
                session_id="real-multi-1",
                turns=[
                    ScenarioTurn(message="请阅读 notes/multi_turn.md，并先用三点概括要点。"),
                    ScenarioTurn(message="把刚才的三点概括压缩成一句话，保留核心信息。", session_done=True),
                ],
                note="同一 session 的两轮对话，验证上下文延续。",
            ),
            ScenarioSpec(
                name="skill-read",
                session_id="real-skill-read-1",
                turns=[
                    ScenarioTurn(
                        message=(
                            "请使用 workspace-file-reader 技能，阅读 notes/reference.md 的第一段，"
                            "并原样返回这一段。"
                        ),
                        session_done=True,
                    ),
                ],
                note="验证 skill catalog 注入和 skill 文件读取。",
                expect_read_skill="workspace-file-reader",
            ),
            ScenarioSpec(
                name="failure-for-evolution",
                session_id="real-evolve-seed-1",
                turns=[
                    ScenarioTurn(
                        message=(
                            "请先读取 notes/missing_spec.md 第一段，然后告诉我标题。"
                        ),
                        session_done=True,
                    ),
                ],
                note="故意触发缺文件错误，作为演化输入。",
            ),
        ]

    def _scenario_result_from_turns(
        self,
        scenario: ScenarioSpec,
        turn_outputs: list[dict[str, Any]],
        runtime_session_id: str,
        record_turns: list[dict[str, Any]] | None = None,
    ) -> ScenarioResult:
        session_turns = (
            record_turns
            if record_turns is not None
            else self._snapshot_turns(runtime_session_id or scenario.session_id)
        )
        matched_skill = scenario.expect_read_skill
        if matched_skill:
            read_names = [
                name
                for turn in session_turns
                for name in _match_skill_names(turn)
            ]
            success = matched_skill in read_names
        elif scenario.name == "simple-single-turn":
            success = bool(turn_outputs and turn_outputs[0]["assistant_text"].strip())
        elif scenario.name == "multi-turn-follow-up":
            success = len(session_turns) >= 2
        elif scenario.name == "failure-for-evolution":
            success = any(
                bool(turn.get("tool_errors"))
                or "not found" in str(turn.get("response_text", "")).lower()
                or "不存在" in str(turn.get("response_text", ""))
                for turn in session_turns
            )
        else:
            success = all(rc == 0 for rc in [int(x["returncode"]) for x in turn_outputs])
        result = ScenarioResult(
            scenario=scenario,
            runtime_session_id=(
                runtime_session_id
                or (str(session_turns[0].get("session_id") or "") if session_turns else "")
                or scenario.session_id
            ),
            returncodes=[int(x["returncode"]) for x in turn_outputs],
            assistant_texts=[str(x["assistant_text"]) for x in turn_outputs],
            stdout_snippets=[str(x["stdout"])[:1200] for x in turn_outputs],
            stderr_snippets=[str(x["stderr"])[:800] for x in turn_outputs],
            session_turns=session_turns,
            record_turns=session_turns,
            success=success,
            notes=scenario.note,
        )
        return result

    def _run_scenarios(self) -> list[ScenarioResult]:
        results: list[ScenarioResult] = []
        for scenario in self._build_scenarios():
            before_records = self._conversation_records()
            turn_outputs: list[dict[str, Any]] = []
            runtime_session_id = ""
            for turn in scenario.turns:
                turn_output = (
                    self._run_agent_turn(
                        session_id=scenario.session_id,
                        message=turn.message,
                        cwd=self.agent_workspace,
                    )
                )
                runtime_session_id = str(turn_output.get("session_id") or runtime_session_id or scenario.session_id)
                turn_outputs.append(turn_output)
                time.sleep(1.0)
            after_records = self._conversation_records()
            record_turns = after_records[len(before_records):]
            actual_session_id = (
                str(record_turns[0].get("session_id") or "")
                if record_turns
                else runtime_session_id
            )
            result = self._scenario_result_from_turns(
                scenario,
                turn_outputs,
                actual_session_id or runtime_session_id,
                record_turns=record_turns,
            )
            results.append(result)
        self.scenario_results = results
        return results

    def _build_report(self) -> dict[str, Any]:
        sessions: dict[str, list[dict[str, Any]]] = {}
        for result in self.scenario_results:
            session_key = result.runtime_session_id or result.scenario.session_id
            sessions.setdefault(session_key, []).extend(result.record_turns)
        if self.prm_probe_result:
            probe_session = self.prm_probe_result["session_id"]
            sessions[probe_session] = self._snapshot_turns(probe_session)
        direct_records = []
        for session_id, turns in sessions.items():
            for turn in turns:
                direct_records.append({
                    "session_id": session_id,
                    "turn_num": turn.get("turn_num"),
                    "prompt_text": turn.get("prompt_text", ""),
                    "response_text": turn.get("response_text", ""),
                    "read_skills": [r.get("skill_name", "") for r in turn.get("read_skills", []) if isinstance(r, dict)],
                    "tool_error_count": len(turn.get("tool_errors", []) or []),
                    "tool_call_count": len(turn.get("tool_calls", []) or []),
                    "prm_score": turn.get("prm_score"),
                })
        self.direct_records = direct_records
        return {
            "name": self.config.name,
            "run_root": str(self.run_root),
            "workspace_dir": str(self.agent_workspace),
            "skills_dir": str(self.skills_dir),
            "record_dir": str(self.record_dir),
            "prm_probe": self.prm_probe_result,
            "scenarios": [
                {
                    "name": r.scenario.name,
                    "session_id": r.scenario.session_id,
                    "runtime_session_id": r.runtime_session_id,
                    "note": r.notes,
                    "returncodes": r.returncodes,
                    "assistant_texts": r.assistant_texts,
                    "success": r.success,
                    "turn_count": len(r.session_turns),
                    "read_skills": sorted({name for turn in r.session_turns for name in _match_skill_names(turn)}),
                    "tool_error_count": sum(len(turn.get("tool_errors", []) or []) for turn in r.session_turns),
                    "tool_call_count": sum(len(turn.get("tool_calls", []) or []) for turn in r.session_turns),
                    "prm_scores": [turn.get("prm_score") for turn in r.session_turns],
                }
                for r in self.scenario_results
            ],
            "evolution": self.evolve_result,
            "records": direct_records,
            "timestamps": {
                "generated_at": _now_stamp(),
            },
        }

    def _build_report_markdown(self, report: dict[str, Any]) -> str:
        lines = [
            f"# {report['name']}",
            "",
            "## 总览",
            "",
            f"- Workspace: `{report['workspace_dir']}`",
            f"- Skills dir: `{report['skills_dir']}`",
            f"- Records dir: `{report['record_dir']}`",
            f"- PRM probe session: `{report.get('prm_probe', {}).get('session_id', '-')}`",
            f"- Evolve skills evolved: {report.get('evolution', {}).get('summary', {}).get('skills_evolved', 0)}",
            "",
            "## 场景结果",
            "",
        ]
        for scenario in report.get("scenarios", []):
            lines.extend([
                f"### {scenario['name']}",
                f"- session_id: `{scenario['session_id']}`",
                f"- success: `{scenario['success']}`",
                f"- turns: `{scenario['turn_count']}`",
                f"- tool calls: `{scenario['tool_call_count']}`",
                f"- tool errors: `{scenario['tool_error_count']}`",
                f"- PRM scores: `{scenario['prm_scores']}`",
                f"- read skills: `{scenario['read_skills']}`",
                f"- note: {scenario['note']}",
                "- assistant outputs:",
            ])
            for txt in scenario.get("assistant_texts", []):
                lines.append(f"  - {txt[:300]}")
            lines.append("")

        lines.extend([
            "## PRM 探针",
            "",
            f"- HTTP status: {report.get('prm_probe', {}).get('http_status', '-')}",
            f"- response: {report.get('prm_probe', {}).get('response_text', '')}",
            f"- conversations.jsonl count: {report.get('prm_probe', {}).get('conversation_record_count', 0)}",
            f"- prm_scores.jsonl count: {report.get('prm_probe', {}).get('prm_record_count', 0)}",
            "",
            "## 演化结果",
            "",
            f"- summary: `{report.get('evolution', {}).get('summary', {})}`",
            f"- manifest: `{report.get('evolution', {}).get('manifest_path', '')}`",
            f"- registry: `{report.get('evolution', {}).get('registry_path', '')}`",
            "",
            "## 记录摘要",
            "",
        ])
        for rec in report.get("records", []):
            lines.append(
                f"- {rec['session_id']}#{rec['turn_num']} "
                f"prm={rec['prm_score']} "
                f"tool_calls={rec['tool_call_count']} "
                f"tool_errors={rec['tool_error_count']} "
                f"skills={rec['read_skills']}"
            )
        lines.append("")
        return "\n".join(lines)

    def run(self) -> dict[str, Any]:
        if not self.skillclaw_config.llm_api_base or not self.skillclaw_config.llm_api_key:
            raise RuntimeError(
                "OpenClaw real suite requires OPENAI_BASE_URL and OPENAI_API_KEY "
                "(or explicit llm_api_base / llm_api_key in the config)."
            )
        self._seed_workspace_files()
        self._start_proxy()
        self._configure_openclaw()

        try:
            self._run_scenarios()
            self._run_prm_probe()
            for result in self.scenario_results:
                self._merge_prm_scores(result.record_turns)

            sessions_for_evolve: dict[str, list[dict[str, Any]]] = {}
            for result in self.scenario_results:
                if not any(
                    turn.get("tool_errors") or float(turn.get("prm_score") or 0.0) <= 0.0
                    for turn in result.record_turns
                ):
                    continue
                session_key = result.runtime_session_id or result.scenario.session_id
                sessions_for_evolve.setdefault(session_key, []).extend(result.record_turns)
            session_paths = self._write_session_snapshots(sessions_for_evolve)
            if self.config.run_evolution and session_paths:
                self._run_evolution(list(sessions_for_evolve.keys()))

            post_scenario = ScenarioSpec(
                name="post-evolve-retry",
                session_id="real-post-evolve-1",
                turns=[
                    ScenarioTurn(
                        message=(
                            "请再次处理 notes/missing_spec.md。先检查文件是否存在；如果不存在，"
                            "不要假装读取成功，直接说明缺少路径即可。"
                        ),
                        session_done=True,
                    ),
                ],
                note="复测演化后的缺文件处理。",
                expect_read_skill="workspace-file-reader",
            )
            post_before_records = self._conversation_records()
            post_turn_outputs = [
                self._run_agent_turn(
                    session_id=post_scenario.session_id,
                    message=post_scenario.turns[0].message,
                    cwd=self.agent_workspace,
                )
            ]
            post_after_records = self._conversation_records()
            post_record_turns = post_after_records[len(post_before_records):]
            post_result = self._scenario_result_from_turns(
                post_scenario,
                post_turn_outputs,
                str(post_record_turns[0].get("session_id") or post_turn_outputs[0].get("session_id") or post_scenario.session_id)
                if post_record_turns
                else str(post_turn_outputs[0].get("session_id") or post_scenario.session_id),
                record_turns=post_record_turns or None,
            )
            self.scenario_results.append(post_result)

            report = self._build_report()
            report_path = self.report_dir / "report.json"
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            md_path = self.report_dir / "report.md"
            md_path.write_text(self._build_report_markdown(report), encoding="utf-8")
            _append_jsonl(self.artifact_dir / "suite_records.jsonl", report)
            return report
        finally:
            if self.server is not None:
                self.server.stop()


def run_openclaw_real_suite_from_config(config_path: str | None = None) -> dict[str, Any]:
    if config_path:
        config = OpenClawRealSuiteConfig.from_file(config_path)
    else:
        config = OpenClawRealSuiteConfig()
    runner = OpenClawRealSuiteRunner(config)
    return runner.run()
