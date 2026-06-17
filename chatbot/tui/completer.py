"""prompt_toolkit completer for the TUI chat REPL.

A :class:`~prompt_toolkit.completion.WordCompleter` populated from the
command registry. Completes both primary names (``/help``) and aliases
(``/exit``). The completer is case-insensitive and uses a leading
slash as the trigger pattern so the user can still type prose that
happens to contain command-shaped words.
"""

from __future__ import annotations

from prompt_toolkit.completion import WordCompleter

from chatbot.tui import commands


def build_completer() -> WordCompleter:
    """Return a WordCompleter covering every command name and alias."""
    words: set[str] = set()
    # ``command_names`` only exposes primary names; pull aliases directly
    # from the registry via a small helper round-trip.
    registry = getattr(commands, "_REGISTRY", {})
    for name, entry in registry.items():
        words.add(f"/{name}")
        for alias in entry.aliases:
            words.add(f"/{alias}")
    return WordCompleter(
        words=sorted(words),
        pattern=r"/\w+",
        ignore_case=True,
        sentence=True,
    )
