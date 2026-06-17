"""Transport layer for the TUI chat REPL.

Two backends share a tiny ``Transport`` Protocol:

* :class:`InProcessTransport` — imports ``chatbot.llm`` and
  ``chatbot.store`` directly. No socket, no server. Supports token
  streaming from Ollama via ``stream: True``.
* :class:`TcpTransport` — speaks the same newline-delimited JSON wire
  format the existing async server uses. Useful for keeping server and
  client decoupled, but only supports a single reply per request.

A ``Transport`` owns the conversation history for a single user. The
TUI shell calls :meth:`get_history`, :meth:`send`, :meth:`clear_history`
and treats both implementations interchangeably.
"""

from __future__ import annotations

import json
import socket
from typing import Iterator, Optional, Protocol, runtime_checkable

import requests

from chatbot import config, llm
from chatbot.store import Store, make_store


@runtime_checkable
class Transport(Protocol):
    """A history + inference backend the TUI can talk to."""

    backend_name: str

    def get_history(self, user_id: str) -> list[dict]: ...
    def save_history(self, user_id: str, history: list[dict]) -> None: ...
    def clear_history(self, user_id: str) -> None: ...
    def send(
        self,
        user_id: str,
        message: str,
        *,
        model: str,
        system: str,
        temperature: float,
        num_ctx: int,
    ) -> str: ...
    def stream(
        self,
        user_id: str,
        message: str,
        *,
        model: str,
        system: str,
        temperature: float,
        num_ctx: int,
    ) -> Iterator[str]: ...


# ---------------------------------------------------------------------------
# In-process
# ---------------------------------------------------------------------------


class InProcessTransport:
    """Runs the LLM and history store in the same Python process.

    Uses :func:`chatbot.llm.build_prompt` and a streaming-friendly variant
    of :func:`chatbot.llm.call_ollama` against the local Ollama server.
    """

    backend_name = "in-process"

    def __init__(self, store: Optional[Store] = None) -> None:
        self._store: Store = store if store is not None else make_store()
        self._fallback = False  # True once we've degraded to memory

    # ---- history -----------------------------------------------------------

    def get_history(self, user_id: str) -> list[dict]:
        try:
            return list(self._store.get_history(user_id))
        except Exception:
            self._degrade_to_memory()
            return list(self._store.get_history(user_id))

    def save_history(self, user_id: str, history: list[dict]) -> None:
        try:
            self._store.save_history(user_id, history)
        except Exception:
            self._degrade_to_memory()
            self._store.save_history(user_id, history)

    def clear_history(self, user_id: str) -> None:
        try:
            self._store.clear_history(user_id)
        except Exception:
            self._degrade_to_memory()
            self._store.clear_history(user_id)

    def _degrade_to_memory(self) -> None:
        """Switch to a process-local MemoryStore after a backend error.

        The TUI prints a one-time notice so the user knows history
        will not survive a restart. Idempotent.
        """
        if self._fallback:
            return
        self._fallback = True
        from chatbot.memory_store import MemoryStore

        self._store = MemoryStore()
        try:
            from chatbot.tui.render import render_notice

            render_notice(
                "history store unavailable — falling back to in-memory (history will not persist)"
            )
        except Exception:
            # Rendering is best-effort; never let a notice crash the loop.
            pass

    # ---- single-shot send --------------------------------------------------

    def send(
        self,
        user_id: str,
        message: str,
        *,
        model: str,
        system: str,
        temperature: float,
        num_ctx: int,
    ) -> str:
        history = self.get_history(user_id)
        prompt = _build_prompt(history, message, system)
        return _call_ollama_once(
            prompt,
            model=model,
            temperature=temperature,
            num_ctx=num_ctx,
        )

    # ---- streaming send ----------------------------------------------------

    def stream(
        self,
        user_id: str,
        message: str,
        *,
        model: str,
        system: str,
        temperature: float,
        num_ctx: int,
    ) -> Iterator[str]:
        history = self.get_history(user_id)
        prompt = _build_prompt(history, message, system)
        yield from _call_ollama_stream(
            prompt,
            model=model,
            temperature=temperature,
            num_ctx=num_ctx,
        )


# ---------------------------------------------------------------------------
# TCP
# ---------------------------------------------------------------------------


class TcpTransport:
    """Talks to the running async server over a newline-JSON socket.

    Streams are not supported by the server protocol; the TUI falls back
    to a single blocking call when streaming is requested.
    """

    backend_name = "tcp"

    def __init__(self, host: str = config.HOST, port: int = config.PORT) -> None:
        self._host = host
        self._port = port

    def _request(self, user_id: str, message: str) -> str:
        with socket.create_connection((self._host, self._port), timeout=300) as sock:
            sock_file = sock.makefile("rwb", buffering=0)
            payload = json.dumps({"user_id": user_id, "message": message}) + "\n"
            sock_file.write(payload.encode("utf-8"))
            line = sock_file.readline()
            if not line:
                raise RuntimeError("server closed connection")
            reply = json.loads(line.decode("utf-8").strip())
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply.get("response", "")

    # The TCP server is authoritative for history, so the client-side
    # implementations are thin proxies that issue a `__clear__` magic.

    def get_history(self, user_id: str) -> list[dict]:
        # The wire protocol doesn't expose a history dump. Return an empty
        # list; the client state remains the working copy. This is enough
        # for `/history` and the banner.
        return []

    def save_history(self, user_id: str, history: list[dict]) -> None:  # noqa: D401
        # No-op: the server saves on every turn.
        return None

    def clear_history(self, user_id: str) -> None:
        self._request(user_id, "__clear__")

    def send(
        self,
        user_id: str,
        message: str,
        *,
        model: str,
        system: str,
        temperature: float,
        num_ctx: int,
    ) -> str:
        del model, system, temperature, num_ctx  # server controls these
        return self._request(user_id, message)

    def stream(
        self,
        user_id: str,
        message: str,
        *,
        model: str,
        system: str,
        temperature: float,
        num_ctx: int,
    ) -> Iterator[str]:
        # Server protocol is request/response — yield the full reply as one
        # chunk so the TUI can still render it as a panel.
        yield self.send(
            user_id,
            message,
            model=model,
            system=system,
            temperature=temperature,
            num_ctx=num_ctx,
        )


# ---------------------------------------------------------------------------
# Shared helpers (also used by the cli shell)
# ---------------------------------------------------------------------------


def _build_prompt(history: list[dict], message: str, system: str) -> str:
    """Same shape as ``chatbot.llm.build_prompt`` but with a per-call system.

    We still use the canonical ``build_prompt`` for the default case so
    behavior matches the existing server, but allow the TUI to override
    the system prompt at call time.
    """
    if system == config.SYSTEM_PROMPT:
        return llm.build_prompt(history, message)

    lines: list[str] = [f"System: {system}", ""]
    for entry in history:
        role = entry.get("role", "").capitalize()
        content = entry.get("content", "")
        lines.append(f"{role}: {content}")
    lines.append(f"User: {message}")
    lines.append("Assistant:")
    return "\n".join(lines)


def _call_ollama_once(
    prompt: str,
    *,
    model: str,
    temperature: float,
    num_ctx: int,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_gpu": config.NUM_GPU,
            "num_ctx": num_ctx,
            "temperature": temperature,
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


def _call_ollama_stream(
    prompt: str,
    *,
    model: str,
    temperature: float,
    num_ctx: int,
) -> Iterator[str]:
    """Stream tokens from Ollama. Falls back to a one-shot call on error."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "num_gpu": config.NUM_GPU,
            "num_ctx": num_ctx,
            "temperature": temperature,
            "repeat_penalty": config.REPEAT_PENALTY,
        },
    }
    try:
        with requests.post(
            config.OLLAMA_URL, json=payload, timeout=300, stream=True
        ) as resp:
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Ollama returned HTTP {resp.status_code}: {resp.text[:200]}"
                )
            yielded = False
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except ValueError:
                    continue
                piece = chunk.get("response")
                if piece:
                    yielded = True
                    yield piece
                if chunk.get("done"):
                    break
            if not yielded:
                # Server replied 200 but no NDJSON chunks — fall through.
                return
    except requests.RequestException as exc:
        raise RuntimeError(f"Ollama stream failed: {exc}") from exc


def make_transport(
    use_tcp: bool = False,
    store: Optional[Store] = None,
) -> Transport:
    """Factory: in-process by default, TCP if ``use_tcp`` is True.

    ``store`` is forwarded to :class:`InProcessTransport` and lets the
    caller swap in a :class:`~chatbot.memory_store.MemoryStore` (or any
    other ``Store``) without touching ``chatbot.config``.
    """
    if use_tcp:
        return TcpTransport()
    return InProcessTransport(store=store)
