"""
Data structures for conversation samples collected by the API proxy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ConversationSample:
    """One sample collected from the API proxy."""

    session_id: str
    turn_num: int
    prompt_tokens: list[int]
    response_tokens: list[int]
    response_logprobs: list[float]
    loss_mask: list[int]
    reward: float
    prompt_text: str = ""
    response_text: str = ""
    teacher_logprobs: Optional[list[float]] = None
    skill_generation: int = 0
