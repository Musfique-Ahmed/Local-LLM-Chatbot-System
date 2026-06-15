"""MongoDB-backed conversation history wrapper (Atlas M0 friendly)."""

from __future__ import annotations

from chatbot import config


class MongoStore:
    """MongoDB implementation of the Store protocol.

    Document shape:
        {
            "_id": "<user_id>",
            "messages": [{"role": "user"|"assistant", "content": "..."}, ...]
        }
    """

    def __init__(self) -> None:
        if not config.MONGO_URI:
            raise RuntimeError(
                "MONGO_URI is empty. Set the MONGO_URI environment variable "
                "(e.g. 'mongodb+srv://user:pass@cluster0.xxxxx.mongodb.net/...') "
                "or define it in chatbot/config.py."
            )
        # Lazy import so Redis-only installs don't need pymongo installed.
        from pymongo import MongoClient

        self._client = MongoClient(config.MONGO_URI, serverSelectionTimeoutMS=5000)
        self._coll = self._client[config.MONGO_DB][config.MONGO_COLLECTION]

    def get_history(self, user_id: str) -> list[dict]:
        """Return the conversation history list for a user, or [] if none."""
        doc = self._coll.find_one({"_id": user_id})
        if not doc:
            return []
        return list(doc.get("messages", []))

    def save_history(self, user_id: str, history: list[dict]) -> None:
        """Persist the history, trimmed to the last MAX_MESSAGES entries."""
        trimmed = history[-config.MAX_MESSAGES :]
        self._coll.replace_one(
            {"_id": user_id},
            {"_id": user_id, "messages": trimmed},
            upsert=True,
        )

    def clear_history(self, user_id: str) -> None:
        """Delete the history document for a user."""
        self._coll.delete_one({"_id": user_id})
