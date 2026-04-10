"""
evolve_server — **population-based** asynchronous skill evolution engine
for SkillClaw.

Periodically (or on-demand) fetches session interaction data from shared
storage, builds a dual-layer representation for each session (lossless
programmatic trajectory + LLM trajectory-aware analysis), aggregates
sessions by skill, and lets the LLM decide whether to evolve, optimise,
or skip each skill — with full trajectory context.

Pipeline::

    Shared Storage Sessions
      → Dual-layer preprocessing:
          A. Programmatic trajectory (step-by-step path, zero info loss)
          B. LLM trajectory-aware analysis (causal chains, skill effectiveness)
          C. Metadata extraction (skills_referenced, avg_prm, tool_errors)
      → Aggregate sessions by skill
      → Per-skill evolution (LLM sees trajectory + analysis; decides: improve / optimize / skip)
      → No-skill session handling (LLM sees trajectory + analysis; decides: create / skip)
      → Shared Storage Skills

Usage::

    python -m evolve_server                       # periodic (default 10 min)
    python -m evolve_server --once                 # single pass
    python -m evolve_server --port 8787            # with HTTP trigger
"""

from .config import EvolveServerConfig
from .constants import DecisionAction, FailureType, FAILURE_LABELS, NO_SKILL_KEY
from .llm_client import AsyncLLMClient
from .mock_bucket import LocalBucket, MockBucket
from .server import EvolveServer
from .skill_registry import SkillIDRegistry

__all__ = [
    "EvolveServer",
    "EvolveServerConfig",
    "AsyncLLMClient",
    "LocalBucket",
    "MockBucket",
    "SkillIDRegistry",
    "FailureType",
    "DecisionAction",
    "FAILURE_LABELS",
    "NO_SKILL_KEY",
]
