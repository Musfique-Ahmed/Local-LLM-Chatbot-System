"""Unit tests for the TUI command registry and session state.

These tests avoid touching real stores or LLM calls. The transport
parameter is a tiny fake that records calls in memory.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest

from chatbot import config
from chatbot.tui import commands
from chatbot.tui.commands import CommandResult
from chatbot.tui.session import Session


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeTransport:
    """In-memory transport that records every call for assertions."""

    def __init__(self) -> None:
        self.backend_name = "in-process"
        self.cleared: list[str] = []
        self.saved: dict[str, list[dict]] = {}
        self.sent: list[tuple[str, str]] = []
        self.replies: list[str] = []

    def get_history(self, user_id: str) -> list[dict]:
        return list(self.saved.get(user_id, []))

    def save_history(self, user_id: str, history: list[dict]) -> None:
        self.saved[user_id] = list(history)

    def clear_history(self, user_id: str) -> None:
        self.cleared.append(user_id)
        self.saved.pop(user_id, None)

    def send(self, user_id, message, **_):
        self.sent.append((user_id, message))
        reply = self.replies.pop(0) if self.replies else f"echo: {message}"
        return reply

    def stream(self, user_id, message, **_):
        self.sent.append((user_id, message))
        reply = self.replies.pop(0) if self.replies else f"echo: {message}"
        yield reply


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def transport() -> FakeTransport:
    return FakeTransport()


@pytest.fixture
def session() -> Session:
    return Session(user_id="alice")


def _quiet(capsys):
    """Helper: capture and discard stdout from a command."""
    return capsys.readouterr().out


# ---------------------------------------------------------------------------
# is_command / parse_command
# ---------------------------------------------------------------------------


def test_is_command_accepts_known_names():
    assert commands.is_command("/help")
    assert commands.is_command("/clear")
    assert commands.is_command("/quit")
    assert commands.is_command("   /help with args")  # leading whitespace ok


def test_is_command_rejects_plain_text():
    assert not commands.is_command("hello world")
    assert not commands.is_command("not /a command")
    assert not commands.is_command("/")
    assert not commands.is_command("")


def test_parse_command_returns_command_with_args():
    cmd = commands.parse_command("/temp 0.5")
    assert cmd is not None
    assert cmd.name == "temp"
    assert cmd.args == "0.5"


def test_parse_command_with_no_args():
    cmd = commands.parse_command("/clear")
    assert cmd is not None
    assert cmd.name == "clear"
    assert cmd.args == ""


def test_parse_command_unknown_name():
    assert commands.parse_command("/nope") is None


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------


def test_dispatch_quit_returns_exit(transport, session):
    result = commands.dispatch("/quit", session, transport)
    assert result is CommandResult.EXIT


def test_dispatch_exit_alias_works(transport, session):
    result = commands.dispatch("/exit", session, transport)
    assert result is CommandResult.EXIT


def test_dispatch_clear_clears_history(transport, session):
    transport.save_history("alice", [{"role": "user", "content": "hi"}])
    result = commands.dispatch("/clear", session, transport)
    assert result is CommandResult.CONTINUE
    assert "alice" in transport.cleared


def test_dispatch_clear_alias(transport, session):
    transport.save_history("alice", [{"role": "user", "content": "hi"}])
    result = commands.dispatch("/cls", session, transport)
    assert result is CommandResult.CONTINUE
    assert "alice" in transport.cleared


def test_dispatch_model_shows_default(transport, session, capsys):
    commands.dispatch("/model", session, transport)
    out = _quiet(capsys)
    assert config.MODEL_NAME in out


def test_dispatch_model_sets_value(transport, session, capsys):
    commands.dispatch("/model llama3.2:3b", session, transport)
    assert session.model == "llama3.2:3b"
    # Subsequent show reflects the override.
    commands.dispatch("/model", session, transport)
    out = _quiet(capsys)
    assert "llama3.2:3b" in out


def test_dispatch_temp_validates_range(transport, session, capsys):
    result = commands.dispatch("/temp 5.0", session, transport)
    assert result is CommandResult.ERROR
    # Valid value should set the override.
    result = commands.dispatch("/temp 0.2", session, transport)
    assert result is CommandResult.CONTINUE
    assert session.temperature == 0.2


def test_dispatch_temp_rejects_non_number(transport, session, capsys):
    result = commands.dispatch("/temp hot", session, transport)
    assert result is CommandResult.ERROR


def test_dispatch_ctx_rejects_tiny_value(transport, session, capsys):
    result = commands.dispatch("/ctx 16", session, transport)
    assert result is CommandResult.ERROR
    result = commands.dispatch("/ctx 4096", session, transport)
    assert result is CommandResult.CONTINUE
    assert session.num_ctx == 4096


def test_dispatch_system_sets_prompt(transport, session):
    commands.dispatch("/system You are a pirate.", session, transport)
    assert session.system_prompt == "You are a pirate."


def test_dispatch_stream_toggles(transport, session):
    assert session.streaming is True
    commands.dispatch("/stream off", session, transport)
    assert session.streaming is False
    commands.dispatch("/stream on", session, transport)
    assert session.streaming is True
    commands.dispatch("/stream", session, transport)
    assert session.streaming is False


def test_dispatch_stream_rejects_garbage(transport, session, capsys):
    result = commands.dispatch("/stream maybe", session, transport)
    assert result is CommandResult.ERROR


def test_dispatch_user_switches(transport, session):
    commands.dispatch("/user bob", session, transport)
    assert session.user_id == "bob"


def test_dispatch_user_requires_id(transport, session, capsys):
    result = commands.dispatch("/user", session, transport)
    assert result is CommandResult.ERROR


def test_dispatch_save_and_load_round_trip(transport, session):
    transport.save_history("alice", [{"role": "user", "content": "old"}])
    commands.dispatch("/save backup", session, transport)
    assert "alice:backup" in transport.saved
    # Mutate history, then load snapshot.
    transport.save_history("alice", [{"role": "user", "content": "new"}])
    commands.dispatch("/load backup", session, transport)
    history = transport.get_history("alice")
    assert history == [{"role": "user", "content": "old"}]


def test_dispatch_save_requires_name(transport, session, capsys):
    result = commands.dispatch("/save", session, transport)
    assert result is CommandResult.ERROR


def test_dispatch_new_clears_current_user(transport, session):
    transport.save_history("alice", [{"role": "user", "content": "x"}])
    commands.dispatch("/new", session, transport)
    assert "alice" in transport.cleared


def test_dispatch_help_shows_all(transport, session, capsys):
    commands.dispatch("/help", session, transport)
    out = _quiet(capsys)
    for name in ("help", "clear", "quit", "model", "system", "temp", "ctx"):
        assert f"/{name}" in out


def test_dispatch_help_for_one_command(transport, session, capsys):
    commands.dispatch("/help temp", session, transport)
    out = _quiet(capsys)
    assert "temp" in out
    assert "temperature" in out.lower() or "0.0" in out


def test_dispatch_unknown_command(transport, session, capsys):
    # Goes through the manual branch in dispatch since parse_command returns None.
    result = commands.dispatch("/nope", session, transport)
    assert result is CommandResult.ERROR


def test_dispatch_retry_with_no_history(transport, session, capsys):
    # No last_user_message: the handler refuses and returns ERROR. The
    # CLI loop checks session.last_user_message before calling
    # dispatch, so this ERROR path is the safety net.
    result = commands.dispatch("/retry", session, transport)
    assert result is CommandResult.ERROR


def test_dispatch_retry_with_history_continues(transport, session, capsys):
    session.mark_message("hi", "hello")
    result = commands.dispatch("/retry", session, transport)
    assert result is CommandResult.CONTINUE
    # needs_retry is a pure shape check on the line.
    assert commands.needs_retry("/retry") is True


# ---------------------------------------------------------------------------
# session.mark_message
# ---------------------------------------------------------------------------


def test_session_mark_message_tracks_last(session):
    session.mark_message("hi", "hello there")
    assert session.last_user_message == "hi"
    assert session.last_assistant_reply == "hello there"
    assert session.unsaved_changes is True


def test_session_effective_values_fall_back_to_config():
    s = Session(user_id="x")
    assert s.effective_model("gemma3:4b") == "gemma3:4b"
    assert s.effective_temperature(0.7) == 0.7
    s.model = "llama3:3b"
    s.temperature = 0.1
    assert s.effective_model("gemma3:4b") == "llama3:3b"
    assert s.effective_temperature(0.7) == 0.1


def test_session_reset_last_clears_state(session):
    session.mark_message("hi", "hello")
    session.reset_last()
    assert session.last_user_message is None
    assert session.last_assistant_reply is None
