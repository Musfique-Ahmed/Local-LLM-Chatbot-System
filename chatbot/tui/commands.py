"""Slash-command registry for the TUI chat REPL.

Each command is a small function that takes the parsed ``Command``
plus the live :class:`~chatbot.tui.session.Session` and a
:class:`chatbot.tui.transport.Transport`, and returns a
:class:`CommandResult` indicating whether the REPL should keep going.

Commands are registered by decoration. ``/help`` builds its output
dynamically from the registry, so adding a new command only requires
implementing a handler and decorating it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from chatbot import config
from chatbot.tui.session import Session
from chatbot.tui.transport import Transport


class CommandResult(str, Enum):
    """Outcome of dispatching a slash command."""

    CONTINUE = "continue"  # keep the REPL loop going
    EXIT = "exit"          # user asked to leave
    ERROR = "error"        # the command failed; print the error and continue


@dataclass
class Command:
    """A single slash command invocation."""

    name: str
    args: str  # raw argument string (may be empty)


@dataclass
class _Entry:
    name: str
    aliases: tuple[str, ...]
    summary: str
    usage: str
    handler: Callable[[Command, Session, Transport], CommandResult]


_REGISTRY: dict[str, _Entry] = {}


def _register(
    name: str,
    *,
    aliases: tuple[str, ...] = (),
    summary: str,
    usage: str,
) -> Callable[[Callable[[Command, Session, Transport], CommandResult]], Callable[[Command, Session, Transport], CommandResult]]:
    def wrap(fn: Callable[[Command, Session, Transport], CommandResult]) -> Callable[[Command, Session, Transport], CommandResult]:
        _REGISTRY[name] = _Entry(name, aliases, summary, usage, fn)
        return fn

    return wrap


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@_register("help", summary="Show this help (or detail for one command).", usage="/help [command]")
def _help(cmd: Command, session: Session, transport: Transport) -> CommandResult:
    from chatbot.tui.render import render_help, render_command_detail

    target = cmd.args.strip()
    if target:
        entry = _REGISTRY.get(target.lstrip("/"))
        if entry is None:
            print(f"unknown command: /{target.lstrip('/')}")
            return CommandResult.ERROR
        render_command_detail(entry)
    else:
        render_help(_REGISTRY.values())
    return CommandResult.CONTINUE


@_register(
    "clear",
    aliases=("cls",),
    summary="Clear the current conversation history.",
    usage="/clear",
)
def _clear(cmd: Command, session: Session, transport: Transport) -> CommandResult:
    transport.clear_history(session.user_id)
    session.reset_last()
    session.unsaved_changes = False
    return CommandResult.CONTINUE


@_register(
    "quit",
    aliases=("exit",),
    summary="Exit the chatbot.",
    usage="/quit",
)
def _quit(cmd: Command, session: Session, transport: Transport) -> CommandResult:
    return CommandResult.EXIT


@_register(
    "history",
    summary="Show the last N turns (default 10).",
    usage="/history [n]",
)
def _history(cmd: Command, session: Session, transport: Transport) -> CommandResult:
    from chatbot.tui.render import render_history

    raw = cmd.args.strip()
    n = 10
    if raw:
        try:
            n = int(raw)
        except ValueError:
            print(f"history: expected integer, got {raw!r}")
            return CommandResult.ERROR
    history = transport.get_history(session.user_id)
    render_history(history, last_n=n)
    return CommandResult.CONTINUE


@_register(
    "model",
    summary="Show or set the model used for the next turn.",
    usage="/model [name]",
)
def _model(cmd: Command, session: Session, transport: Transport) -> CommandResult:
    arg = cmd.args.strip()
    if not arg:
        print(f"model: {session.effective_model(config.MODEL_NAME)}")
        return CommandResult.CONTINUE
    session.model = arg
    return CommandResult.CONTINUE


@_register(
    "system",
    summary="Show or set the system prompt for the next turn.",
    usage="/system [text]",
)
def _system(cmd: Command, session: Session, transport: Transport) -> CommandResult:
    arg = cmd.args.strip()
    if not arg:
        print(f"system: {session.effective_system(config.SYSTEM_PROMPT)}")
        return CommandResult.CONTINUE
    session.system_prompt = arg
    return CommandResult.CONTINUE


@_register(
    "temp",
    aliases=("temperature",),
    summary="Show or set the sampling temperature (0.0-2.0).",
    usage="/temp <0.0-2.0>",
)
def _temp(cmd: Command, session: Session, transport: Transport) -> CommandResult:
    arg = cmd.args.strip()
    if not arg:
        print(f"temp: {session.effective_temperature(config.TEMPERATURE):.2f}")
        return CommandResult.CONTINUE
    try:
        value = float(arg)
    except ValueError:
        print(f"temp: expected a number, got {arg!r}")
        return CommandResult.ERROR
    if not 0.0 <= value <= 2.0:
        print(f"temp: {value} out of range (expected 0.0-2.0)")
        return CommandResult.ERROR
    session.temperature = value
    return CommandResult.CONTINUE


@_register(
    "ctx",
    aliases=("num_ctx",),
    summary="Show or set the context window size in tokens.",
    usage="/ctx <int>",
)
def _ctx(cmd: Command, session: Session, transport: Transport) -> CommandResult:
    arg = cmd.args.strip()
    if not arg:
        print(f"ctx: {session.effective_num_ctx(config.NUM_CTX)}")
        return CommandResult.CONTINUE
    try:
        value = int(arg)
    except ValueError:
        print(f"ctx: expected an integer, got {arg!r}")
        return CommandResult.ERROR
    if value < 128:
        print(f"ctx: {value} is suspiciously small (minimum 128)")
        return CommandResult.ERROR
    session.num_ctx = value
    return CommandResult.CONTINUE


@_register(
    "save",
    summary="Save the current history under a named snapshot.",
    usage="/save <name>",
)
def _save(cmd: Command, session: Session, transport: Transport) -> CommandResult:
    name = cmd.args.strip()
    if not name:
        print("save: provide a snapshot name")
        return CommandResult.ERROR
    snapshot_key = f"{session.user_id}:{name}"
    history = transport.get_history(session.user_id)
    transport.save_history(snapshot_key, history)
    session.unsaved_changes = False
    return CommandResult.CONTINUE


@_register(
    "load",
    summary="Replace current history with a previously saved snapshot.",
    usage="/load <name>",
)
def _load(cmd: Command, session: Session, transport: Transport) -> CommandResult:
    name = cmd.args.strip()
    if not name:
        print("load: provide a snapshot name")
        return CommandResult.ERROR
    snapshot_key = f"{session.user_id}:{name}"
    snapshot = transport.get_history(snapshot_key)
    transport.save_history(session.user_id, snapshot)
    session.reset_last()
    session.unsaved_changes = False
    return CommandResult.CONTINUE


@_register(
    "user",
    summary="Switch to a different user id (loads that user's history).",
    usage="/user <id>",
)
def _user(cmd: Command, session: Session, transport: Transport) -> CommandResult:
    new_id = cmd.args.strip()
    if not new_id:
        print("user: provide a new user id")
        return CommandResult.ERROR
    session.user_id = new_id
    session.reset_last()
    session.unsaved_changes = False
    return CommandResult.CONTINUE


@_register(
    "new",
    summary="Start a fresh empty history for the current user.",
    usage="/new",
)
def _new(cmd: Command, session: Session, transport: Transport) -> CommandResult:
    transport.clear_history(session.user_id)
    session.reset_last()
    session.unsaved_changes = False
    return CommandResult.CONTINUE


@_register(
    "stream",
    summary="Toggle token streaming (in-process only).",
    usage="/stream [on|off]",
)
def _stream(cmd: Command, session: Session, transport: Transport) -> CommandResult:
    arg = cmd.args.strip().lower()
    if arg == "on":
        session.streaming = True
    elif arg == "off":
        session.streaming = False
    elif not arg:
        session.streaming = not session.streaming
    else:
        print(f"stream: expected on/off, got {arg!r}")
        return CommandResult.ERROR
    return CommandResult.CONTINUE


@_register(
    "copy",
    summary="Copy the last assistant reply to the clipboard.",
    usage="/copy",
)
def _copy(cmd: Command, session: Session, transport: Transport) -> CommandResult:
    if not session.last_assistant_reply:
        print("copy: nothing to copy yet")
        return CommandResult.ERROR
    try:
        import pyperclip  # type: ignore
    except ImportError:
        print("copy: pyperclip is not installed (`pip install pyperclip`)")
        return CommandResult.ERROR
    try:
        pyperclip.copy(session.last_assistant_reply)
    except pyperclip.PyperclipException as exc:
        print(f"copy: {exc}")
        return CommandResult.ERROR
    return CommandResult.CONTINUE


@_register(
    "retry",
    summary="Re-send the last user message and replace the assistant reply.",
    usage="/retry",
)
def _retry(cmd: Command, session: Session, transport: Transport) -> CommandResult:
    # Handled by the REPL loop, not the dispatcher — the loop owns the
    # render path for streaming. Surface a clear error if invoked with no
    # last message.
    if not session.last_user_message:
        print("retry: no previous user message to resend")
        return CommandResult.ERROR
    # Mark for the REPL: it inspects the command name and re-issues.
    # The dispatcher normally returns CONTINUE; the loop keys off the
    # command's name (set by the parser) to know this was a /retry.
    return CommandResult.CONTINUE


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _resolve_name(name: str) -> Optional[str]:
    """Return the primary command name for ``name`` (handles aliases)."""
    if name in _REGISTRY:
        return name
    for primary, entry in _REGISTRY.items():
        if name in entry.aliases:
            return primary
    return None


def is_command(line: str) -> bool:
    """Return True if the line begins with a recognized command prefix."""
    stripped = line.lstrip()
    if not stripped.startswith("/"):
        return False
    body = stripped[1:]
    if not body:
        return False
    name = body.split(maxsplit=1)[0]
    return _resolve_name(name) is not None


def parse_command(line: str) -> Optional[Command]:
    """Parse a slash line into a :class:`Command`. Returns None if invalid."""
    stripped = line.lstrip()
    if not stripped.startswith("/"):
        return None
    body = stripped[1:]
    if not body:
        return None
    name, _, args = body.partition(" ")
    primary = _resolve_name(name)
    if primary is None:
        return None
    return Command(name=primary, args=args.strip())


def dispatch(line: str, session: Session, transport: Transport) -> CommandResult:
    """Parse and run a slash command, returning its :class:`CommandResult`."""
    cmd = parse_command(line)
    if cmd is None:
        first = line.lstrip().split(maxsplit=1)[0]
        print(f"unknown command: {first}")
        return CommandResult.ERROR
    return _REGISTRY[cmd.name].handler(cmd, session, transport)


def command_names() -> list[str]:
    """Return all primary command names, sorted alphabetically."""
    return sorted(_REGISTRY)


def needs_retry(line: str) -> bool:
    """True if the line is ``/retry`` and there's something to retry."""
    cmd = parse_command(line)
    return cmd is not None and cmd.name == "retry"
