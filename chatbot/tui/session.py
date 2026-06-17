"""Session state for the TUI chat REPL.

The session is a plain dataclass that holds in-memory runtime overrides
and tracking data. Persistent configuration (the source of truth) lives
in :mod:`chatbot.config`; the TUI never mutates it. Snapshots stored
under named keys are persisted through the configured ``Store`` backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


def _none_str() -> Optional[str]:
    return None


def _none_float() -> Optional[float]:
    return None


def _none_int() -> Optional[int]:
    return None


@dataclass
class Session:
    """Runtime session state for one user of the TUI."""

    user_id: str

    # Runtime overrides. ``None`` means "use the value from chatbot.config".
    model: Optional[str] = field(default_factory=_none_str)
    system_prompt: Optional[str] = field(default_factory=_none_str)
    temperature: Optional[float] = field(default_factory=_none_float)
    num_ctx: Optional[int] = field(default_factory=_none_int)

    # Feature toggles.
    streaming: bool = True

    # Tracking for /retry and /copy.
    last_user_message: Optional[str] = field(default_factory=_none_str)
    last_assistant_reply: Optional[str] = field(default_factory=_none_str)

    # Tracks whether the in-memory history diverges from what is on disk.
    unsaved_changes: bool = False

    # ---- Effective values (override or default) ----------------------------

    def effective_model(self, default: str) -> str:
        return self.model if self.model is not None else default

    def effective_system(self, default: str) -> str:
        return self.system_prompt if self.system_prompt is not None else default

    def effective_temperature(self, default: float) -> float:
        return self.temperature if self.temperature is not None else default

    def effective_num_ctx(self, default: int) -> int:
        return self.num_ctx if self.num_ctx is not None else default

    # ---- Mutation helpers --------------------------------------------------

    def mark_message(self, user_msg: str, assistant_msg: str) -> None:
        """Record the most recent turn (used by /retry and /copy)."""
        self.last_user_message = user_msg
        self.last_assistant_reply = assistant_msg
        self.unsaved_changes = True

    def reset_last(self) -> None:
        self.last_user_message = None
        self.last_assistant_reply = None
