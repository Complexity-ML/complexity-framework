"""
CLI Utilities - Console, spinners, formatting.
"""

from typing import Optional
from contextlib import contextmanager

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.syntax import Syntax
    from rich.markdown import Markdown
    from rich import print as rprint
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


# =============================================================================
# Console
# =============================================================================

class FallbackConsole:
    """Fallback console when rich is not available."""

    def print(self, *args, **kwargs):
        # Strip rich markup
        text = " ".join(str(a) for a in args)
        text = text.replace("[bold]", "").replace("[/bold]", "")
        text = text.replace("[dim]", "").replace("[/dim]", "")
        text = text.replace("[green]", "").replace("[/green]", "")
        text = text.replace("[red]", "").replace("[/red]", "")
        text = text.replace("[yellow]", "").replace("[/yellow]", "")
        text = text.replace("[blue]", "").replace("[/blue]", "")
        text = text.replace("[cyan]", "").replace("[/cyan]", "")
        print(text)

    def input(self, prompt: str = "") -> str:
        return input(prompt.replace("[bold cyan]", "").replace("[/bold cyan]", ""))


console = Console() if HAS_RICH else FallbackConsole()


# =============================================================================
# Spinner
# =============================================================================

@contextmanager
def spinner(message: str = "Loading..."):
    """
    Show a spinner while doing work.

    Usage:
        with spinner("Loading model..."):
            load_heavy_model()
    """
    if HAS_RICH:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task(message, total=None)
            yield
    else:
        print(message)
        yield


# =============================================================================
# Panels & Tables
# =============================================================================

def print_panel(content: str, title: Optional[str] = None, style: str = "green"):
    """Print a panel."""
    if HAS_RICH:
        console.print(Panel(content, title=title, border_style=style))
    else:
        if title:
            print(f"\n=== {title} ===")
        print(content)
        print()


def print_table(title: str, columns: list, rows: list):
    """Print a table."""
    if HAS_RICH:
        table = Table(title=title)
        for col in columns:
            table.add_column(col["name"], style=col.get("style", ""))
        for row in rows:
            table.add_row(*[str(v) for v in row])
        console.print(table)
    else:
        print(f"\n{title}")
        print("-" * 50)
        header = " | ".join(c["name"] for c in columns)
        print(header)
        print("-" * 50)
        for row in rows:
            print(" | ".join(str(v) for v in row))
        print()


def print_code(code: str, language: str = "python"):
    """Print syntax-highlighted code."""
    if HAS_RICH:
        console.print(Syntax(code, language, theme="monokai"))
    else:
        print(code)


def print_markdown(text: str):
    """Print markdown."""
    if HAS_RICH:
        console.print(Markdown(text))
    else:
        print(text)


# =============================================================================
# Colors & Formatting
# =============================================================================

def success(text: str) -> str:
    """Green success text."""
    return f"[bold green]✓ {text}[/bold green]" if HAS_RICH else f"✓ {text}"


def error(text: str) -> str:
    """Red error text."""
    return f"[bold red]✗ {text}[/bold red]" if HAS_RICH else f"✗ {text}"


def warning(text: str) -> str:
    """Yellow warning text."""
    return f"[yellow]⚠ {text}[/yellow]" if HAS_RICH else f"⚠ {text}"


def info(text: str) -> str:
    """Blue info text."""
    return f"[blue]ℹ {text}[/blue]" if HAS_RICH else f"ℹ {text}"


def dim(text: str) -> str:
    """Dimmed text."""
    return f"[dim]{text}[/dim]" if HAS_RICH else text


def bold(text: str) -> str:
    """Bold text."""
    return f"[bold]{text}[/bold]" if HAS_RICH else text
