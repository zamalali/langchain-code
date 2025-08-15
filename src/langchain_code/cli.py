from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from .config import resolve_provider
from .agent.bootstrap import build_agent
from .workflows.feature_impl import FEATURE_INSTR
from .workflows.bug_fix import BUGFIX_INSTR


from rich.text import Text

def banner(provider: str, project_dir: Path) -> Panel:
    title = f"langchain-code  •  provider: {provider.upper()}"
    body = Text()
    body.append("Project: ", style="bold")
    body.append(str(project_dir))
    body.append("\n\n")
    body.append("Type your request. Press Ctrl+C to exit.\n", style="dim")
    return Panel.fit(body, title=title, subtitle="ReAct • Tools • Safe Edits", border_style="magenta")

app = typer.Typer(add_completion=False)
console = Console()

@app.command(help="Open an interactive chat with the agent.")
def chat(
    llm: Optional[str] = typer.Option(None, "--llm", help='anthropic | gemini'),
    project_dir: Path = typer.Option(Path.cwd(), "--project-dir", exists=True, file_okay=False),
):
    provider = resolve_provider(llm)
    agent = build_agent(provider=provider, project_dir=project_dir)

    # console.print(Panel.fit(f"[b]langchain-code[/b] — provider: [magenta]{provider}[/magenta]\nProject: {project_dir}"))
    # console.print("[dim]Type your request. Ctrl+C to exit.[/dim]")
    console.print(banner(provider, project_dir))

    try:
        while True:
            user = typer.prompt("> ")
            if not user.strip():
                continue
            result = agent.invoke({"input": user})
            console.print(result["output"])
    except (KeyboardInterrupt, EOFError):
        console.print("\n[bold]Goodbye![/bold]")

@app.command(help="POC Task 1: Implement a feature end-to-end (plan → search → edit → verify).")
def feature(
    request: str = typer.Argument(..., help='e.g. "Add a dark mode toggle in settings"'),
    llm: Optional[str] = typer.Option(None, "--llm", help='anthropic | gemini'),
    project_dir: Path = typer.Option(Path.cwd(), "--project-dir", exists=True, file_okay=False),
    test_cmd: Optional[str] = typer.Option(None, "--test-cmd", help='e.g. "pytest -q" or "npm test"'),
    apply: bool = typer.Option(False, "--apply", help="Apply writes and run commands without interactive confirm."),
):
    provider = resolve_provider(llm)
    agent = build_agent(provider=provider, project_dir=project_dir, apply=apply, test_cmd=test_cmd, instruction_seed=FEATURE_INSTR)
    console.print(Panel.fit(f"[b]Feature Implementation[/b] using [magenta]{provider}[/magenta]"))
    result = agent.invoke({"input": request})
    console.print(result["output"])

@app.command(help="POC Task 2: Diagnose & fix a bug (trace → pinpoint → patch → test).")
def fix(
    request: Optional[str] = typer.Argument(None, help='e.g. "Fix crash on image upload"'),
    log: Optional[Path] = typer.Option(None, "--log", exists=True, help="Path to error log or stack trace."),
    llm: Optional[str] = typer.Option(None, "--llm", help='anthropic | gemini'),
    project_dir: Path = typer.Option(Path.cwd(), "--project-dir", exists=True, file_okay=False),
    test_cmd: Optional[str] = typer.Option(None, "--test-cmd", help='e.g. "pytest -q"'),
    apply: bool = typer.Option(False, "--apply", help="Apply writes and run commands without interactive confirm."),
):
    provider = resolve_provider(llm)
    agent = build_agent(provider=provider, project_dir=project_dir, apply=apply, test_cmd=test_cmd, instruction_seed=BUGFIX_INSTR)

    bug_input = request or ""
    if log:
        bug_input += "\n\n--- ERROR LOG ---\n" + Path(log).read_text(encoding="utf-8")

    console.print(Panel.fit(f"[b]Bug Fix[/b] using [magenta]{provider}[/magenta]"))
    result = agent.invoke({"input": bug_input.strip() or "Fix the bug using the provided log."})
    console.print(result["output"])


def main():
    app()

if __name__ == "__main__":
    main()
# This code is part of the langchain-code package and provides a command-line interface