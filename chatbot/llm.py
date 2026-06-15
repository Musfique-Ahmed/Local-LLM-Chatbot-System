"""Prompt construction and Ollama HTTP client."""

import requests

from chatbot import config


def build_prompt(history: list[dict], new_message: str) -> str:
    """Build a plain-text multi-turn prompt ending with 'Assistant:'."""
    lines: list[str] = [f"System: {config.SYSTEM_PROMPT}", ""]
    for entry in history:
        role = entry.get("role", "").capitalize()
        content = entry.get("content", "")
        lines.append(f"{role}: {content}")
    lines.append(f"User: {new_message}")
    lines.append("Assistant:")
    return "\n".join(lines)


def call_ollama(prompt: str) -> str:
    """POST to Ollama and return the stripped model response text."""
    payload = {
        "model": config.MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_gpu": config.NUM_GPU,
            "num_ctx": config.NUM_CTX,
            "temperature": config.TEMPERATURE,
            "repeat_penalty": config.REPEAT_PENALTY,
        },
    }
    try:
        resp = requests.post(config.OLLAMA_URL, json=payload, timeout=300)
    except requests.RequestException as exc:
        raise RuntimeError(f"Ollama HTTP request failed: {exc}") from exc

    if resp.status_code != 200:
        raise RuntimeError(
            f"Ollama returned HTTP {resp.status_code}: {resp.text[:200]}"
        )

    try:
        data = resp.json()
    except ValueError as exc:
        raise RuntimeError(f"Ollama response was not valid JSON: {exc}") from exc

    if "response" not in data:
        raise RuntimeError(f"Ollama response missing 'response' field: {data}")

    return data["response"].strip()
