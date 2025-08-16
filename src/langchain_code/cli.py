from __future__ import annotations
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.style import Style
from rich.text import Text
from pyfiglet import Figlet
from .config import resolve_provider
from .agent.react import build_react_agent
from .workflows.feature_impl import FEATURE_INSTR
from .workflows.bug_fix import BUGFIX_INSTR


def print_langcode_ascii(console, text="LangCode", font="ansi_shadow", gradient="dark_to_light"):
    """Print a single-shot 'LangCode' ASCII banner with a strong green L→R gradient."""
    def _hex_to_rgb(h): h = h.lstrip("#"); return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    def _lerp(a, b, t): return int(a + (b - a) * t)
    def _interpolate_palette(palette, width):
        if width <= 1: return [palette[0]]
        out, steps_total = [], width - 1
        for x in range(width):
            pos = x / steps_total
            seg = min(int(pos * (len(palette) - 1)), len(palette) - 2)
            seg_start, seg_end = seg / (len(palette) - 1), (seg + 1) / (len(palette) - 1)
            local_t = (pos - seg_start) / (seg_end - seg_start + 1e-9)
            c1, c2 = _hex_to_rgb(palette[seg]), _hex_to_rgb(palette[seg + 1])
            rgb = tuple(_lerp(a, b, local_t) for a, b in zip(c1, c2))
            out.append("#{:02x}{:02x}{:02x}".format(*rgb))
        return out

    def _print_block_with_horizontal_gradient(lines, palette):
        width = max(len(line) for line in lines) if lines else 0
        ramp = _interpolate_palette(palette, width)
        for line in lines:
            t = Text()
            padded = line.ljust(width)
            for j, ch in enumerate(padded):
                t.append(ch if ch == " " else ch, Style(color=ramp[j], bold=True))
            console.print(t)

    fig = Figlet(font=font)
    lines = fig.renderText(text).rstrip("\n").splitlines()

    palette = ["#052e1e", "#064e3b", "#065f46", "#047857", "#059669", "#16a34a", "#22c55e", "#34d399"]
    if gradient == "light_to_dark":
        palette = list(reversed(palette))

    _print_block_with_horizontal_gradient(lines, palette)


def banner(provider: str, project_dir: Path, title_text: str, interactive: bool = False) -> Panel:
    """Display a banner with the project information."""
    title = Text(title_text, style="bold magenta")
    
    body = Text()
    body.append("Provider: ", style="bold")
    body.append(provider.upper())
    body.append("\n")
    body.append("Project:  ", style="bold")
    body.append(str(project_dir))
    
    if interactive:
        body.append("\n\n")
        body.append("Type your request. Press Ctrl+C to exit.\n", style="dim")
    
    return Panel(
        body,
        title=title,
        subtitle=Text("ReAct • Tools • Safe Edits", style="dim"),
        border_style="green",
        padding=(1, 2),
    )

app = typer.Typer(add_completion=False)
console = Console()

@app.command(help="Open an interactive chat with the agent.")
def chat(
    llm: Optional[str] = typer.Option(None, "--llm", help='anthropic | gemini'),
    project_dir: Path = typer.Option(Path.cwd(), "--project-dir", exists=True, file_okay=False),
):
    provider = resolve_provider(llm)
    agent = build_react_agent(provider=provider, project_dir=project_dir)

    print_langcode_ascii(console, text="LangCode", font="ansi_shadow", gradient="dark_to_light")

    console.print(banner(provider, project_dir, "LangChain Code Agent", interactive=True))
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
    agent = build_react_agent(provider=provider, project_dir=project_dir, apply=apply, test_cmd=test_cmd, instruction_seed=FEATURE_INSTR)
    console.print(banner(provider, project_dir, "Feature Implementation"))
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
    agent = build_react_agent(provider=provider, project_dir=project_dir, apply=apply, test_cmd=test_cmd, instruction_seed=BUGFIX_INSTR)

    bug_input = request or ""
    if log:
        bug_input += "\n\n--- ERROR LOG ---\n" + Path(log).read_text(encoding="utf-8")

    console.print(banner(provider, project_dir, "Bug Fix"))
    result = agent.invoke({"input": bug_input.strip() or "Fix the bug using the provided log."})
    console.print(result["output"])


def main():
    app()

if __name__ == "__main__":
    main()
