"""
Loader for the AGENTS.md that guides the OpenClaw evolution agent.

The canonical AGENTS.md lives alongside this file at
``agent_evolve_server/AGENTS.md``.  Edit that file directly to change the
evolution methodology.
"""

from __future__ import annotations

from pathlib import Path

_AGENTS_MD_PATH = Path(__file__).resolve().parent / "EVOLVE_AGENTS.md"


def load_agents_md() -> str:
    """Read the built-in AGENTS.md file and return its content."""
    return _AGENTS_MD_PATH.read_text(encoding="utf-8")
