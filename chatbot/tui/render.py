"""Rich-based rendering helpers for the TUI chat REPL.

All output goes through this module so the look-and-feel stays
consistent: a styled banner on launch, colored Markdown panels for
chat turns, dim italic notices, and red error boxes.

The module is import-safe on systems without a TTY (e.g. when piped) —
``rich`` falls back to plain text automatically.
"""

from __future__ import annotations

from typing import Iterable, Optional

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

console = Console()


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------


BANNER_ART = r"""
 _      __    __                     __          __
| | /| / /__ / /  ___ ____  ___ ____/ /  ___ ___/ /
| |/ |/ / -_) _ \/ _ `/ _ \/ _ `/ _  /  / _ `/ _  /
|__/|__/\__/_.__/\_,_/_//_/\_,_/\_,_/   \_,_/\_,_/
"""


def render_banner(
    *,
    backend: str,
    user_id: str,
    model: str,
    streaming: bool,
) -> None:
    """Print the launch banner with session metadata."""
    console.print(Text(BANNER_ART, style="bold cyan"), highlight=False)
    info = Table.grid(padding=(0, 1))
    info.add_column(style="dim", justify="right")
    info.add_column(style="bold")
    info.add_row("backend", backend)
    info.add_row("user", user_id)
    info.add_row("model", model)
    info.add_row("stream", "on" if streaming else "off")
    console.print(Panel(info, border_style="cyan", title="[bold]session[/bold]", title_align="left"))
    console.print(
        "[dim]type [/dim][bold]/help[/bold][dim] for commands, [/dim][bold]/quit[/bold][dim] to leave.[/dim]"
    )
    console.print()


# ---------------------------------------------------------------------------
# Chat turns
# ---------------------------------------------------------------------------


def render_user_message(text: str) -> None:
    """Render a user message in a cyan-bordered panel."""
    console.print(Panel(Text(text, style="white"), border_style="cyan", title="[bold cyan]you[/bold cyan]", title_align="left"))


def render_assistant_message(text: str) -> None:
    """Render a complete assistant reply, parsing Markdown where possible."""
    body = _safe_markdown(text)
    console.print(Panel(body, border_style="green", title="[bold green]assistant[/bold green]", title_align="left"))


def render_error(text: str) -> None:
    """Render an error message in a red box."""
    console.print(Panel(Text(text, style="bold red"), border_style="red", title="[bold red]error[/bold red]", title_align="left"))


def render_notice(text: str) -> None:
    """Render a dimmed system notice (e.g. saved snapshot, switched user)."""
    console.print(Text(f"  · {text}", style="dim italic"))


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def render_assistant_stream(chunks: Iterable[str]) -> str:
    """Render a streamed assistant reply, updating the panel as tokens arrive.

    Returns the full reply text. Uses ``rich.live.Live`` so the panel
    redraws in place rather than scrolling.
    """
    pieces: list[str] = []
    with Live(console=console, refresh_per_second=20, transient=False) as live:
        for chunk in chunks:
            pieces.append(chunk)
            body = _safe_markdown("".join(pieces) + "▍")
            live.update(
                Panel(
                    body,
                    border_style="green",
                    title="[bold green]assistant[/bold green]",
                    title_align="left",
                )
            )
        # Final render without the cursor.
        body = _safe_markdown("".join(pieces))
        live.update(
            Panel(
                body,
                border_style="green",
                title="[bold green]assistant[/bold green]",
                title_align="left",
            )
        )
    return "".join(pieces)


def make_spinner() -> Spinner:
    """Return a 'dots' spinner for the wait-for-first-token phase."""
    return Spinner("dots", text="thinking…", style="green")


# ---------------------------------------------------------------------------
# History view
# ---------------------------------------------------------------------------


def render_history(history: list[dict], *, last_n: int) -> None:
    """Print the last ``last_n`` turns as colored panels."""
    if not history:
        render_notice("no history yet")
        return
    # Each turn is two entries (user, assistant). Slice in pairs.
    tail = history[-(last_n * 2):]
    for entry in tail:
        role = entry.get("role", "")
        content = entry.get("content", "")
        if role == "user":
            render_user_message(content)
        elif role == "assistant":
            render_assistant_message(content)


# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------


def render_help(entries: Iterable["object"]) -> None:
    """Print a two-column command summary."""
    table = Table(title="commands", title_style="bold cyan", border_style="cyan", show_lines=False)
    table.add_column("command", style="bold green", no_wrap=True)
    table.add_column("aliases", style="dim")
    table.add_column("summary", style="white")

    for entry in sorted(entries, key=lambda e: e.name):
        aliases = ", ".join(f"/{a}" for a in entry.aliases) if entry.aliases else "—"
        table.add_row(f"/{entry.name}", aliases, entry.summary)

    console.print(table)
    console.print("[dim]use [/dim][bold]/help <command>[/bold][dim] for details.[/dim]")


def render_command_detail(entry: "object") -> None:
    """Print the usage line and summary for a single command."""
    aliases = ", ".join(f"/{a}" for a in entry.aliases) if entry.aliases else "(none)"
    body = Text()
    body.append("usage  ", style="dim")
    body.append(entry.usage + "\n")
    body.append("aliases  ", style="dim")
    body.append(aliases + "\n")
    body.append("summary  ", style="dim")
    body.append(entry.summary)
    console.print(Panel(body, border_style="cyan", title=f"[bold]/ {entry.name}[/bold]", title_align="left"))


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _safe_markdown(text: str) -> object:
    """Return a Markdown renderable, falling back to Text on parse issues."""
    try:
        return Markdown(text)
    except Exception:
        return Text(text)
