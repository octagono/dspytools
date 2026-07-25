"""Rich console output utilities for beautiful CLI.

All CLI commands should use these helpers instead of raw click.echo()
for consistent, beautiful output.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table

console = Console()


def section(title: str) -> None:
    """Print a prominent section divider/header with a rule line."""
    console.print()
    console.print(Rule(f"[bold cyan]{title}[/bold cyan]"))
    console.print()


def url(text: str) -> None:
    """Print a clickable URL."""
    console.print(f"  {text}")


def code(text: str) -> str:
    """Format text as inline code."""
    return f"[dim]{text}[/]"


def kv_block(data: dict[str, str | int | float | None], indent: int = 2) -> None:
    """Print a multi-line key/value block."""
    padding = " " * indent
    for key, value in data.items():
        if value is not None:
            console.print(f"{padding}[bold]{key}:[/] {value}")


def header(text: str) -> None:
    """Print a section header."""
    console.print()
    console.print(Rule(f"[bold][cyan]{text}[/cyan][/bold]"))


def info(text: str) -> None:
    """Print an info message."""
    console.print(f"  [cyan]*[/] {text}")


def ok(text: str) -> None:
    """Print a success message."""
    console.print(f"  [bold][green][*][/green][/bold] {text}")


def fail(text: str) -> None:
    """Print a failure message."""
    console.print(f"  [bold][red][-][/red][/bold] {text}")


def warn(text: str) -> None:
    """Print a warning message."""
    console.print(f"  [bold][yellow][!][/yellow][/bold] {text}")


def error(text: str) -> None:
    """Print an error message."""
    console.print(f"  [bold][red][x][/red][/bold] {text}")


def key_value(key: str, value: str, indent: int = 2) -> None:
    """Print a key/value pair."""
    padding = " " * indent
    console.print(f"{padding}[bold]{key}:[/] {value}")


def panel(title: str, content: str, border_style: str = "cyan") -> None:
    """Print content in a panel."""
    console.print(
        Panel(content, title=title, border_style=border_style, padding=(1, 2))
    )


def rule(title: str = "") -> None:
    """Print a horizontal rule."""
    console.print(Rule(title=title))


def syntax(code: str, lang: str = "python") -> None:
    """Print syntax-highlighted code."""
    console.print(Syntax(code, lang, theme="monokai", line_numbers=True))


def table(title: str, columns: list[str], rows: list[list[str]], **kwargs: Any) -> None:
    """Print a table."""
    tbl = Table(title=title, title_style="bold cyan", **kwargs)
    for col in columns:
        tbl.add_column(col, style="cyan", no_wrap=False)
    for row in rows:
        tbl.add_row(*row)
    console.print(tbl)


def progress(description: str = "Working...") -> Progress:
    """Create a progress bar context."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    )


def spinner(description: str = "Working...") -> Progress:
    """Create a spinner for indeterminate progress.

    Usage::

        sp = spinner("Loading...")
        with sp:
            sp.add_task("Loading...")
            do_work()
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    )


def status_text(status: str) -> str:
    """Return colored status text with bracket notation."""
    status_lower = status.lower()
    colors = {
        "running": "green",
        "stopped": "red",
        "starting": "yellow",
        "completed": "green",
        "failed": "red",
        "queued": "yellow",
        "cancelled": "yellow",
        "pending": "yellow",
        "success": "green",
        "active": "green",
        "inactive": "red",
    }
    style = colors.get(status_lower, "dim")
    return f"[bold][{style}][{status}][/][/bold]" if style else status


F = TypeVar("F", bound=Callable[..., Any])


def label_option(f: F) -> F:
    """Decorator: adds --label Click option for run labeling."""
    return click.option(
        "--label",
        default=None,
        type=str,
        help="Run label for registry",
    )(f)


def force_option(f: F) -> F:
    """Decorator: adds --force Click option to force recompilation."""
    return click.option(
        "--force/--no-force",
        default=False,
        help="Force recompilation even if cached result exists",
    )(f)
