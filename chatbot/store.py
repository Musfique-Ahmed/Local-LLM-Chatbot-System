"""Abstract store interface and factory for the conversation history backend."""

from __future__ import annotations

from typing import Protocol

from chatbot import config


class Store(Protocol):
    """Protocol every history backend must implement."""

    def get_history(self, user_id: str) -> list[dict]: ...
    def save_history(self, user_id: str, history: list[dict]) -> None: ...
    def clear_history(self, user_id: str) -> None: ...


def make_store() -> Store:
    """Return a Store instance based on config.STORE_BACKEND."""
    backend = config.STORE_BACKEND.lower()
    if backend == "redis":
        from chatbot.redis_store import RedisStore  # local import avoids cycles

        return RedisStore()
    if backend == "mongo":
        from chatbot.mongo_store import MongoStore

        return MongoStore()
    if backend == "memory":
        from chatbot.memory_store import MemoryStore

        return MemoryStore()
    raise ValueError(
        f"Unknown STORE_BACKEND={config.STORE_BACKEND!r} "
        "(expected 'redis', 'mongo', or 'memory')"
    )


def probe_store() -> tuple[bool, str]:
    """Try to construct a store and perform a no-op call.

    Returns ``(ok, message)``. Used at CLI startup so we can warn
    clearly when the configured backend is unavailable, instead of
    crashing on the first user message.
    """
    try:
        store = make_store()
        store.get_history("__probe__")
    except Exception as exc:  # noqa: BLE001 — any backend error is a probe failure
        return False, f"{type(exc).__name__}: {exc}"
    return True, "ok"

