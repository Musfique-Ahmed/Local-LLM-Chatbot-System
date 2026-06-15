"""Redis-backed conversation history wrapper."""

import json

import redis

from chatbot import config

# Module-level client — only allowed global mutable state in the project.
client = redis.Redis(
    host=config.REDIS_HOST,
    port=config.REDIS_PORT,
    db=config.REDIS_DB,
    decode_responses=True,
)


def _key(user_id: str) -> str:
    """Build the Redis key for a user's history."""
    return f"chat:{user_id}"


def get_history(user_id: str) -> list[dict]:
    """Return the conversation history list for a user, or [] if none."""
    raw = client.get(_key(user_id))
    if raw is None:
        return []
    return json.loads(raw)


def save_history(user_id: str, history: list[dict]) -> None:
    """Persist the history, trimmed to the last MAX_MESSAGES entries."""
    trimmed = history[-config.MAX_MESSAGES:]
    client.set(_key(user_id), json.dumps(trimmed))


def clear_history(user_id: str) -> None:
    """Delete the history key for a user."""
    client.delete(_key(user_id))
