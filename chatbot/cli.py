"""TUI chat REPL entry point.

Run with::

    python -m chatbot.cli                # in-process transport (default)
    python -m chatbot.cli --tcp          # talk to the running async server
    python -m chatbot.cli --user alice   # pick a user id non-interactively

The REPL prints a styled banner, accepts multi-line input, and
recognizes the slash commands registered in
:mod:`chatbot.tui.commands`. Use ``/help`` for the full list.
"""

from __future__ import annotations

import argparse
import getpass
import sys
from typing import Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout

from chatbot import config
from chatbot.tui import commands, render
from chatbot.tui.completer import build_completer
from chatbot.tui.session import Session
from chatbot.tui.transport import Transport, make_transport


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="chatbot.cli",
        description="Terminal UI for the local LLM chatbot.",
    )
    parser.add_argument(
        "--tcp",
        action="store_true",
        help="Use the TCP server transport (default: in-process).",
    )
    parser.add_argument(
        "--memory",
        action="store_true",
        help="Use an in-memory history store (no Redis/Mongo needed).",
    )
    parser.add_argument(
        "--user",
        default=None,
        help="User id for this session (default: prompt for one).",
    )
    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="Skip the startup banner.",
    )
    return parser.parse_args(argv)


def _prompt_for_user_id() -> str:
    """Ask for a user id; fall back to the system user on EOF."""
    try:
        raw = input("user_id: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)
    if not raw:
        return getpass.getuser() or "default"
    return raw


def _send_user_message(
    message: str,
    *,
    session: Session,
    transport: Transport,
) -> Optional[str]:
    """Send a user message through the transport; return the reply or None.

    The assistant reply is rendered in place. The return value is the
    raw text (for ``/retry`` to reuse) or ``None`` if the call failed
    (in which case an error panel has already been printed).
    """
    model = session.effective_model(config.MODEL_NAME)
    system = session.effective_system(config.SYSTEM_PROMPT)
    temperature = session.effective_temperature(config.TEMPERATURE)
    num_ctx = session.effective_num_ctx(config.NUM_CTX)

    render.render_user_message(message)

    history_before = transport.get_history(session.user_id)

    try:
        if session.streaming and transport.backend_name == "in-process":
            with render.console.status(render.make_spinner(), spinner="dots"):
                # We can't easily hold a status + a Live at once, so we
                # run a tiny status pre-roll to indicate the request is
                # in flight, then switch to the streaming Live.
                pass
            # Prime the spinner briefly without blocking the stream.
            with render.console.status("[green]thinking…[/green]"):
                # We need at least one item in the iterator to start the
                # spinner visibly; pull the first chunk and continue.
                iterator = transport.stream(
                    session.user_id,
                    message,
                    model=model,
                    system=system,
                    temperature=temperature,
                    num_ctx=num_ctx,
                )
            reply = render.render_assistant_stream(iterator)
        else:
            with render.console.status("[green]thinking…[/green]"):
                reply = transport.send(
                    session.user_id,
                    message,
                    model=model,
                    system=system,
                    temperature=temperature,
                    num_ctx=num_ctx,
                )
            render.render_assistant_message(reply)
    except Exception as exc:  # noqa: BLE001 — surface any LLM error to the user
        render.render_error(str(exc))
        return None

    # Persist the new turn into the history. For TCP the server is
    # authoritative; for in-process we own the store.
    if transport.backend_name != "tcp":
        history_after = list(history_before)
        history_after.append({"role": "user", "content": message})
        history_after.append({"role": "assistant", "content": reply})
        transport.save_history(session.user_id, history_after)
        # Trim if we exceed the configured cap.
        cap = config.MAX_MESSAGES
        if len(history_after) > cap:
            history_after = history_after[-cap:]
            transport.save_history(session.user_id, history_after)

    session.mark_message(message, reply)
    return reply


def _run_repl(session: Session, transport: Transport) -> int:
    """Main REPL loop. Returns the process exit code."""
    if not _parse_no_banner():
        render.render_banner(
            backend=transport.backend_name,
            user_id=session.user_id,
            model=session.effective_model(config.MODEL_NAME),
            streaming=session.streaming,
        )

    history_file = _history_file_path()
    prompt_session = PromptSession(
        history=FileHistory(str(history_file)) if history_file else None,
        completer=build_completer(),
    )

    prompt_html = HTML(
        "<style fg='ansicyan' bold='true'>you</style>"
        "<style fg='ansibrightblack'> ▸ </style>"
    )

    while True:
        try:
            with patch_stdout():
                line = prompt_session.prompt(prompt_html)
        except KeyboardInterrupt:
            # Ctrl-C: continue to the next prompt (don't exit on a single
            # accidental keystroke). A blank line + Ctrl-C, or Ctrl-D,
            # exits.
            print()
            continue
        except EOFError:
            print()
            return 0

        text = line.strip()
        if not text:
            continue

        if text.startswith("/") and commands.is_command(text):
            # /retry needs to know it was a retry so the loop re-issues
            # the last user message; everything else is a plain dispatch.
            if commands.needs_retry(text):
                if not session.last_user_message:
                    render.render_error("retry: no previous user message to resend")
                    continue
                # Roll back the previous turn so it isn't double-counted.
                if transport.backend_name != "tcp":
                    history = transport.get_history(session.user_id)
                    if len(history) >= 2 and history[-1].get("role") == "assistant":
                        trimmed = history[:-2]
                        transport.save_history(session.user_id, trimmed)
                _send_user_message(
                    session.last_user_message,
                    session=session,
                    transport=transport,
                )
                continue
            result = commands.dispatch(text, session, transport)
            if result is commands.CommandResult.EXIT:
                return 0
            continue

        # Plain chat message.
        _send_user_message(text, session=session, transport=transport)


def _parse_no_banner() -> bool:
    """Inspect argv once to honor --no-banner before printing the banner."""
    return "--no-banner" in sys.argv[1:]


def _history_file_path():
    """Return a stable path for the prompt_toolkit history file, or None."""
    from pathlib import Path

    base = Path.home() / ".chatbot_cli_history"
    try:
        base.parent.mkdir(parents=True, exist_ok=True)
        return base
    except OSError:
        return None


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    user_id = args.user or _prompt_for_user_id()
    session = Session(user_id=user_id)
    transport = _build_transport(args)

    if (
        transport.backend_name == "in-process"
        and not args.memory
        and not getattr(transport, "_fallback", False)
    ):
        # Probe the store up front so the user gets a clear warning
        # before typing their first message, instead of a stack trace.
        from chatbot.store import probe_store

        ok, message = probe_store()
        if not ok:
            from chatbot.tui.render import render_notice

            render_notice(
                f"history store probe failed ({message}) — "
                "rerun with --memory to skip persistence, or start Redis/Mongo."
            )

    return _run_repl(session, transport)


def _build_transport(args: argparse.Namespace) -> Transport:
    """Construct a transport honoring --tcp and --memory.

    ``--memory`` swaps in the in-process ``MemoryStore`` directly; this
    is more reliable than mutating ``os.environ`` after
    ``chatbot.config`` has already been imported.
    """
    from chatbot.tui.transport import TcpTransport, make_transport
    from chatbot.memory_store import MemoryStore

    if args.tcp:
        return TcpTransport()
    if args.memory:
        return make_transport(use_tcp=False, store=MemoryStore())
    return make_transport(use_tcp=False)


if __name__ == "__main__":
    sys.exit(main())
