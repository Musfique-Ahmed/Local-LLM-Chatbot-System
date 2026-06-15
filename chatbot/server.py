"""Async TCP server that proxies chat to a local Ollama instance."""

import asyncio
import json
import logging
import os
import time
from logging.handlers import RotatingFileHandler

from chatbot import config, llm
from chatbot.store import make_store

# Single Store instance for the lifetime of the process; selected by STORE_BACKEND.
_history = make_store()

log = logging.getLogger("chatbot.server")


def _configure_logging() -> None:
    """Configure rotating file logging and stdout mirroring."""
    os.makedirs("logs", exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    file_handler = RotatingFileHandler(
        "logs/server.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(stream_handler)


async def _send(writer: asyncio.StreamWriter, payload: dict) -> None:
    """Write a JSON object as a single newline-terminated line."""
    data = (json.dumps(payload) + "\n").encode("utf-8")
    writer.write(data)
    await writer.drain()


async def _read_request(
    reader: asyncio.StreamReader,
) -> dict | None:
    """Read one newline-terminated JSON object from the client."""
    raw = await reader.readline()
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8").strip())
    except json.JSONDecodeError:
        return None


async def handle_client(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    """Handle a single TCP connection: one request, one response."""
    peer = writer.get_extra_info("peername")
    start = time.perf_counter()
    user_id = "<unknown>"

    try:
        request = await _read_request(reader)
        if request is None:
            await _send(writer, {"error": "Invalid payload"})
            log.warning("Bad JSON from %s", peer)
            return

        user_id_raw = request.get("user_id")
        message_raw = request.get("message")
        if (
            not isinstance(user_id_raw, str)
            or not isinstance(message_raw, str)
            or not user_id_raw
            or not message_raw
        ):
            await _send(writer, {"error": "Invalid payload"})
            log.warning("Invalid payload from %s: %r", peer, request)
            return

        user_id = user_id_raw
        message = message_raw
        log.info(
            "recv user_id=%s msg_len=%d from %s",
            user_id, len(message), peer,
        )

        if message == "__clear__":
            _history.clear_history(user_id)
            await _send(writer, {"response": "History cleared."})
            log.info("cleared history for user_id=%s", user_id)
            return

        history = _history.get_history(user_id)

        try:
            prompt = llm.build_prompt(history, message)
            response = await asyncio.to_thread(llm.call_ollama, prompt)
        except Exception:
            log.exception("LLM call failed for user_id=%s", user_id)
            await _send(writer, {"error": "LLM unavailable"})
            return

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        _history.save_history(user_id, history)

        await _send(writer, {"response": response})

        latency_ms = (time.perf_counter() - start) * 1000.0
        log.info(
            "done user_id=%s msg_len=%d resp_len=%d latency_ms=%.1f",
            user_id, len(message), len(response), latency_ms,
        )
    except Exception:
        log.exception("Unhandled error for user_id=%s peer=%s", user_id, peer)
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


async def main() -> None:
    """Start the TCP server and run forever."""
    _configure_logging()
    server = await asyncio.start_server(handle_client, config.HOST, config.PORT)
    log.info("Chatbot server listening on %s:%d (backend=%s)", config.HOST, config.PORT, config.STORE_BACKEND)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
