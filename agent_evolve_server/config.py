"""
Configuration dataclass for the Agent Evolve Server.

Reuses storage and scheduling fields from
:class:`evolve_server.config.EvolveServerConfig` and adds OpenClaw-specific
settings (binary path, home directory, fresh mode, timeout).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from evolve_server.config import EvolveServerConfig

_PACKAGE_DIR = Path(__file__).resolve().parent
_DEFAULT_AGENT_EVOLVE_BASE_URL = "http://localhost:28080"
_DEFAULT_AGENT_EVOLVE_MODEL = "claude-opus-4-6"


def _first_nonempty(*names: str, default: str = "") -> str:
    for name in names:
        value = str(os.environ.get(name, "") or "").strip()
        if value:
            return value
    return default


@dataclass
class AgentEvolveServerConfig(EvolveServerConfig):
    # OpenClaw agent settings
    openclaw_bin: str = "openclaw"
    openclaw_home: str = ""
    fresh: bool = True
    agent_timeout: int = 600
    workspace_root: str = ""
    agents_md_path: str = ""
    llm_api_type: str = "anthropic-messages"

    def __post_init__(self):
        if not self.openclaw_home:
            self.openclaw_home = str(_PACKAGE_DIR / ".openclaw_home")
        if not self.workspace_root:
            self.workspace_root = str(_PACKAGE_DIR / "agent_workspace")

    @classmethod
    def from_env(cls) -> "AgentEvolveServerConfig":
        base = EvolveServerConfig.from_env()
        base_fields = {f.name: getattr(base, f.name) for f in base.__dataclass_fields__.values()}
        base_fields["llm_api_key"] = _first_nonempty(
            "AGENT_EVOLVE_LLM_API_KEY",
            "AGENT_EVOLVE_API_KEY",
            "OPENAI_API_KEY",
            default=base.llm_api_key,
        )
        base_fields["llm_base_url"] = _first_nonempty(
            "AGENT_EVOLVE_LLM_BASE_URL",
            "AGENT_EVOLVE_BASE_URL",
            default=_DEFAULT_AGENT_EVOLVE_BASE_URL,
        )
        base_fields["llm_model"] = _first_nonempty(
            "AGENT_EVOLVE_MODEL",
            default=_DEFAULT_AGENT_EVOLVE_MODEL,
        )
        return cls(
            **base_fields,
            openclaw_bin=os.environ.get("AGENT_EVOLVE_OPENCLAW_BIN", "openclaw"),
            openclaw_home=os.environ.get("AGENT_EVOLVE_OPENCLAW_HOME", ""),
            fresh=os.environ.get("AGENT_EVOLVE_FRESH", "1").lower() not in {"0", "false", "no"},
            agent_timeout=int(os.environ.get("AGENT_EVOLVE_TIMEOUT", "600")),
            workspace_root=os.environ.get("AGENT_EVOLVE_WORKSPACE_ROOT", ""),
            agents_md_path=os.environ.get("AGENT_EVOLVE_AGENTS_MD", ""),
            llm_api_type=os.environ.get("AGENT_EVOLVE_LLM_API_TYPE", "anthropic-messages"),
        )

    @classmethod
    def from_skillclaw_config(cls, config) -> "AgentEvolveServerConfig":
        base = EvolveServerConfig.from_skillclaw_config(config)
        base_fields = {f.name: getattr(base, f.name) for f in base.__dataclass_fields__.values()}
        base_fields["llm_api_key"] = _first_nonempty(
            "AGENT_EVOLVE_LLM_API_KEY",
            "AGENT_EVOLVE_API_KEY",
            "OPENAI_API_KEY",
            default=(config.llm_api_key or config.prm_api_key or base.llm_api_key),
        )
        base_fields["llm_base_url"] = _first_nonempty(
            "AGENT_EVOLVE_LLM_BASE_URL",
            "AGENT_EVOLVE_BASE_URL",
            default=_DEFAULT_AGENT_EVOLVE_BASE_URL,
        )
        base_fields["llm_model"] = _first_nonempty(
            "AGENT_EVOLVE_MODEL",
            default=_DEFAULT_AGENT_EVOLVE_MODEL,
        )
        return cls(
            **base_fields,
            openclaw_bin=os.environ.get("AGENT_EVOLVE_OPENCLAW_BIN", "openclaw"),
            openclaw_home=os.environ.get("AGENT_EVOLVE_OPENCLAW_HOME", ""),
            fresh=os.environ.get("AGENT_EVOLVE_FRESH", "1").lower() not in {"0", "false", "no"},
            agent_timeout=int(os.environ.get("AGENT_EVOLVE_TIMEOUT", "600")),
            workspace_root=os.environ.get("AGENT_EVOLVE_WORKSPACE_ROOT", ""),
            agents_md_path=os.environ.get("AGENT_EVOLVE_AGENTS_MD", ""),
            llm_api_type=os.environ.get("AGENT_EVOLVE_LLM_API_TYPE", "anthropic-messages"),
        )
