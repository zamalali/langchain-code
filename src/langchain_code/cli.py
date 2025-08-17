from __future__ import annotations
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.style import Style
from rich.text import Text
from rich.markdown import Markdown
from rich.rule import Rule
from rich import box
from pyfiglet import Figlet

from .config import resolve_provider
from .agent.react import build_react_agent
from .workflows.feature_impl import FEATURE_INSTR
from .workflows.bug_fix import BUGFIX_INSTR
from langchain_core.messages import HumanMessage, AIMessage


app = typer.Typer(add_completion=False, help="LangCode – ReAct + tools code agent CLI.")
console = Console()
PROMPT = "[bold green]langcode[/bold green] [dim]›[/dim] "


def print_langcode_ascii(console: Console, text: str = "LangCode", font: str = "ansi_shadow", gradient: str = "dark_to_light") -> None:
    """
    Render a single-shot ASCII banner with a left-to-right green gradient.
    """
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


def session_banner(provider: Optional[str], project_dir: Path, title_text: str, interactive: bool = False, apply: bool = False, test_cmd: Optional[str] = None, tips: Optional[list[str]] = None) -> Panel:
    """
    Build a framed status panel showing provider, project, and optional session tips.
    """
    title = Text(title_text, style="bold magenta")
    body = Text()

    body.append("Provider: ", style="bold")
    if provider and provider.strip() and " " not in provider:
        body.append(provider.upper())
    else:
        body.append((provider or "not set"), style="dim")
    body.append("\n")

    body.append("Project:  ", style="bold")
    body.append(str(project_dir))

    badge = Text()
    if apply:
        badge.append("  [APPLY MODE]", style="bold red")
    if test_cmd:
        badge.append(f"  tests: {test_cmd}", style="italic")
    if badge:
        body.append("\n")
        body.append_text(badge)

    if interactive:
        body.append("\n\n")
        body.append("Type your request. /clear to redraw, /exit to quit. Ctrl+C also exits.\n", style="dim")

    if tips:
        body.append("\n")
        for t in tips:
            body.append(t + "\n", style="dim")

    return Panel(
        body,
        title=title,
        subtitle=Text("ReAct • Tools • Safe Edits", style="dim"),
        border_style="green",
        padding=(1, 2),
        box=box.HEAVY,
    )


def _print_session_header(title: str, provider: Optional[str], project_dir: Path, *, interactive: bool = False, apply: bool = False, test_cmd: Optional[str] = None, tips: Optional[list[str]] = None) -> None:
    """
    Clear the screen and draw the LangCode header, banner, and a separator rule.
    """
    console.clear()
    print_langcode_ascii(console, text="LangCode", font="ansi_shadow", gradient="dark_to_light")
    console.print(session_banner(provider, project_dir, title, interactive=interactive, apply=apply, test_cmd=test_cmd, tips=tips))
    console.print(Rule(style="green"))


def _panel_agent_output(text: str, title: str = "Agent") -> Panel:
    """
    Wrap model output in a cyan panel to visually stay inside the LangCode session.
    """
    body = Markdown(text) if ("```" in text or "\n#" in text) else Text(text)
    return Panel.fit(body, title=title, border_style="cyan", box=box.ROUNDED, padding=(0, 1))


def _maybe_coerce_img_command(raw: str) -> str:
    """
    Convert '/img <p1> <p2> :: <prompt>' to a clear tool instruction for process_multimodal.
    """
    s = raw.strip()
    if not s.startswith("/img"):
        return raw
    try:
        rest = s[len("/img"):].strip()
        if "::" in rest:
            paths_part, prompt_text = rest.split("::", 1)
            prompt_text = prompt_text.strip()
        else:
            paths_part, prompt_text = rest, ""
        paths = [p for p in paths_part.split() if p]
        return (
            f'Please call the tool "process_multimodal" with '
            f"image_paths={paths} and text={prompt_text!r}. After the tool returns, summarize the result."
        )
    except Exception:
        return raw


@app.callback(invoke_without_command=True)
def _root(ctx: typer.Context):
    """
    Show the overview when run without subcommands.
    """
    if ctx.invoked_subcommand is None:
        provider_hint = "set via --llm anthropic|gemini"
        project_dir = Path.cwd()
        _print_session_header(
            "LangChain Code Agent",
            provider_hint,
            project_dir,
            interactive=False,
            tips=[
                "Quick start:",
                "• chat         Open an interactive session with the agent. (supports --apply)",
                "• feature      Plan → search → edit → verify. (supports --apply)",
                "• fix          Diagnose & patch a bug (use --log PATH). (supports --apply)",
                "Tip: run any command with --help for details.",
            ],
        )
        typer.echo(ctx.get_help())
        raise typer.Exit()


@app.command(help="Open an interactive chat with the agent.")
def chat(
    llm: Optional[str] = typer.Option(None, "--llm", help="anthropic | gemini"),
    project_dir: Path = typer.Option(Path.cwd(), "--project-dir", exists=True, file_okay=False),
    apply: bool = typer.Option(False, "--apply", help="Apply writes and run commands without interactive confirm."),
):
    """
    Start a bounded LangCode chat session with persistent visuals and safe controls.
    """
    provider = resolve_provider(llm)
    agent = build_react_agent(provider=provider, project_dir=project_dir, apply=apply)

    _print_session_header("LangChain Code Agent • Chat", provider, project_dir, interactive=True, apply=apply)

    history: list = []

    try:
        while True:
            user = console.input(PROMPT).strip()
            if not user:
                continue

            low = user.lower()
            if low in {"cls", "clear", "/clear"}:
                _print_session_header("LangChain Code Agent • Chat", provider, project_dir, interactive=True, apply=apply)
                continue
            if low in {"exit", "quit", ":q", "/exit"}:
                console.print("\n[bold]Goodbye![/bold]")
                break

            coerced = _maybe_coerce_img_command(user)
            res = agent.invoke({"input": coerced, "chat_history": history})
            output = res.get("output", "") if isinstance(res, dict) else str(res)
            console.print(_panel_agent_output(output))

            history.append(HumanMessage(content=coerced))
            history.append(AIMessage(content=output))
    except (KeyboardInterrupt, EOFError):
        console.print("\n[bold]Goodbye![/bold]")


@app.command(help="Implement a feature end-to-end (plan → search → edit → verify).")
def feature(
    request: str = typer.Argument(..., help='e.g. "Add a dark mode toggle in settings"'),
    llm: Optional[str] = typer.Option(None, "--llm", help="anthropic | gemini"),
    project_dir: Path = typer.Option(Path.cwd(), "--project-dir", exists=True, file_okay=False),
    test_cmd: Optional[str] = typer.Option(None, "--test-cmd", help='e.g. "pytest -q" or "npm test"'),
    apply: bool = typer.Option(False, "--apply", help="Apply writes and run commands without interactive confirm."),
):
    """
    Run the feature workflow once and render the result within the session frame.
    """
    provider = resolve_provider(llm)
    agent = build_react_agent(provider=provider, project_dir=project_dir, apply=apply, test_cmd=test_cmd, instruction_seed=FEATURE_INSTR)

    _print_session_header("LangChain Code Agent • Feature", provider, project_dir, interactive=False, apply=apply, test_cmd=test_cmd)
    res = agent.invoke({"input": request})
    output = res.get("output", "") if isinstance(res, dict) else str(res)
    console.print(_panel_agent_output(output, title="Feature Result"))


@app.command(help="Diagnose & fix a bug (trace → pinpoint → patch → test).")
def fix(
    request: Optional[str] = typer.Argument(None, help='e.g. "Fix crash on image upload"'),
    log: Optional[Path] = typer.Option(None, "--log", exists=True, help="Path to error log or stack trace."),
    llm: Optional[str] = typer.Option(None, "--llm", help="anthropic | gemini"),
    project_dir: Path = typer.Option(Path.cwd(), "--project-dir", exists=True, file_okay=False),
    test_cmd: Optional[str] = typer.Option(None, "--test-cmd", help='e.g. "pytest -q"'),
    apply: bool = typer.Option(False, "--apply", help="Apply writes and run commands without interactive confirm."),
):
    """
    Run the bug-fix workflow once, with optional log input, and render the result within the session frame.
    """
    provider = resolve_provider(llm)
    agent = build_react_agent(provider=provider, project_dir=project_dir, apply=apply, test_cmd=test_cmd, instruction_seed=BUGFIX_INSTR)

    bug_input = request or ""
    if log:
        bug_input += "\n\n--- ERROR LOG ---\n" + Path(log).read_text(encoding="utf-8")

    _print_session_header("LangChain Code Agent • Fix", provider, project_dir, interactive=False, apply=apply, test_cmd=test_cmd)
    res = agent.invoke({"input": bug_input.strip() or "Fix the bug using the provided log."})
    output = res.get("output", "") if isinstance(res, dict) else str(res)
    console.print(_panel_agent_output(output, title="Fix Result"))


def main() -> None:
    """
    Entrypoint for the langcode console script.
    """
    app()


if __name__ == "__main__":
    main()
