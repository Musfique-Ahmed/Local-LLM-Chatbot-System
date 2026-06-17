"""In-memory store backend.

Useful for running the chatbot without Redis or MongoDB installed.
History lives in a process-local dict and is lost when the CLI exits.
Use the ``memory`` backend by setting ``STORE_BACKEND=memory`` in the
environment, or by passing ``store=MemoryStore()`` directly.
"""

from __future__ import annotations

from threading import RLock


class MemoryStore:
    """Thread-safe, process-local history store.

    Implements the same shape as ``RedisStore`` and ``MongoStore``:
    ``get_history``, ``save_history``, ``clear_history``. All return
    new lists so callers can mutate freely.
    """

    def __init__(self) -> None:
        self._data: dict[str, list[dict]] = {}
        self._lock = RLock()

    def get_history(self, user_id: str) -> list[dict]:
        with self._lock:
            return list(self._data.get(user_id, []))

    def save_history(self, user_id: str, history: list[dict]) -> None:
        with self._lock:
            self._data[user_id] = list(history)

    def clear_history(self, user_id: str) -> None:
        with self._lock:
            self._data.pop(user_id, None)
