import json
from typing import Any, Dict
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

console = Console()

def display_welome_banner(mode: str = "confirm", provider: str = "ollama") -> None:
    """
    Renders the startup banner when CLIA launches.
    """
    banner_text = Text()
    banner_text.append("CLIA", style="bold cyan")
    banner_text.append(" - Command Line Interfacce Assistant\n", style="bold white")
    banner_text.append(f"Provider: {provider} | Mode: {mode}\n", style="dim white")
    banner_text.append("Type 'exit' or 'quit' to end the session.", style="italic dim white")

    panel = Panel(
        banner_text,
        title="[bold cyan]Welcome[/bold cyan]",
        border_style="cyan"
        expand=False
    )
    console.print(panel)

def display_tool_confirmation(tool_name: str, arguments: dict[str, Any]) -> None:
    """
    Renders a formatted panel inspecting a proposed tool call before asking user permission.
    """
    formatted_args = json.dumps(arguments, indent=2)
    args_syntax = Syntax(formatted_args, "json", theme="monokai", word_wrap=True)

    panel = Panel(
        args_syntax,
        title=f"[bold yellow]Proposed Tool Execution: {tool_name}[/bold yellow]",
        subtitle="[dim]Requires approval[/dim]"
        border_style="yellow",
        expand=False
    )
    console.print(panel)

def display_tool_results(tool_name: str, result: str, status: str = "success") -> None:
    """
    Renders the observation output resulting from a tool execution.
    """
    border_color = "green" if status == "success" else "red"
    title_color = "bold green" if status == "success" else "bold red"

    max_len = 2000
    display_text = result if len(result) <= max_len else result[:max_len] + "\n... [truncated]"

    panel = Panel(
        display_text,
        title=f"[{title_color}]Tool Result ({status}): {tool_name}[/{title_color}]",
        border_style=border_color,
        expand=False
    )
    console.print(panel)

def print_status(message: str) -> None:
    """
    Prints a status update message.
    """
    console.print(f"[dim cyam]-> {message}[/dim cyan]")