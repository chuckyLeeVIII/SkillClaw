"""
Shared constants and enums for the evolve server.
"""

from __future__ import annotations

import re
from enum import IntEnum

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,}$")


class FailureType(IntEnum):
    """Five-way failure taxonomy for a bad turn."""
    SKILL_CONTENT_STALE = 1
    SKILL_MISSELECT = 2
    SKILL_GAP = 3
    TOOL_ERROR = 4
    MODEL_BASELINE = 5


FAILURE_LABELS: dict[int, str] = {
    FailureType.SKILL_CONTENT_STALE: "Skill 内容失效",
    FailureType.SKILL_MISSELECT: "Skill 误选",
    FailureType.SKILL_GAP: "Skill 缺口",
    FailureType.TOOL_ERROR: "工具使用错误",
    FailureType.MODEL_BASELINE: "模型基础能力",
}


NO_SKILL_KEY = "__no_skill__"


class DecisionAction:
    """Allowed evolution-decision action identifiers."""
    CREATE = "create_skill"
    IMPROVE = "improve_skill"
    OPTIMIZE_DESC = "optimize_description"
    SKIP = "skip"
