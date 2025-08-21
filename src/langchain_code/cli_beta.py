from __future__ import annotations
from pathlib import Path
from typing import Optional, List
import sys
import os
import time
import threading
from contextlib import contextmanager

import typer
from rich.console import Console
from rich.panel import Panel
from rich.style import Style
from rich.text import Text
from rich.markdown import Markdown
from rich.rule import Rule
from rich import box
from rich.live import Live
from pyfiglet import Figlet

from .config import resolve_provider
from .agent.react import build_react_agent, build_deep_agent
from .workflows.feature_impl import FEATURE_INSTR
from .workflows.bug_fix import BUGFIX_INSTR
from .workflows.auto import AUTO_DEEP_INSTR
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.messages import ToolMessage


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


def session_banner(provider: Optional[str], project_dir: Path, title_text: str, interactive: bool = False, apply: bool = False, test_cmd: Optional[str] = None, tips: Optional[List[str]] = None) -> Panel:
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
        subtitle=Text("ReAct • Deep • Tools • Safe Edits", style="dim"),
        border_style="green",
        padding=(1, 2),
        box=box.HEAVY,
    )


def _print_session_header(title: str, provider: Optional[str], project_dir: Path, *, interactive: bool = False, apply: bool = False, test_cmd: Optional[str] = None, tips: Optional[List[str]] = None) -> None:
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


def _panel_user_input(text: str) -> Panel:
    """Pretty green panel to echo submitted user input in the transcript."""
    content = Markdown(text) if ("```" in text or "\n#" in text) else Text(text, style="bold")
    return Panel(
        content,
        title=Text("You", style="bold green"),
        border_style="green",
        padding=(1, 2),
        box=box.HEAVY,
    )


# =============== LIVE TYPING INPUT BOX (inside a green boundary) ===============
def _render_typing_panel(buffer: str) -> Panel:
    """Render the live typing panel with current buffer."""
    caret = "▏"  # slim caret
    text = Text.from_markup("")
    text.append_text(Text.from_markup(PROMPT))
    text.append(buffer)
    text.append(caret, style="bold green")
    return Panel(
        text,
        title=Text("Type your request", style="bold green"),
        border_style="green",
        padding=(1, 2),
        box=box.HEAVY,
    )


def _readline_in_panel() -> str:
    """
    Read a line from stdin while rendering a green panel around the input.
    Falls back to Console.input when a TTY isn't available.
    Supports basic editing: backspace, Ctrl+C, Enter. Ignores arrow keys.
    """
    if not sys.stdin.isatty():
        # Fallback (no live boundary possible)
        return console.input(PROMPT).strip()

    # Cross-platform char reading
    on_windows = os.name == "nt"
    if on_windows:
        try:
            import msvcrt  # type: ignore
        except Exception:
            return console.input(PROMPT).strip()

        buf: List[str] = []
        with Live(_render_typing_panel(""), refresh_per_second=30, console=console) as live:
            while True:
                ch = msvcrt.getwch()
                if ch in ("\r", "\n"):
                    console.print() 
                    return "".join(buf).strip()
                if ch == "\003":  # Ctrl+C
                    raise KeyboardInterrupt()
                if ch in ("\b", "\x7f"):
                    if buf:
                        buf.pop()
                elif ch == "\xe0" or ch == "\x00":
                    _ = msvcrt.getwch()  
                else:
                    buf.append(ch)
                live.update(_render_typing_panel("".join(buf)))  

    else:
        import termios, tty

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        buf: List[str] = []
        try:
            tty.setraw(fd)
            with Live(_render_typing_panel(""), refresh_per_second=30, console=console):
                while True:
                    ch = sys.stdin.read(1)
                    if ch in ("\r", "\n"):
                        console.print()
                        return "".join(buf).strip()
                    if ch == "\x03":  # Ctrl+C
                        raise KeyboardInterrupt()
                    if ch in ("\x7f", "\b"):  # Backspace
                        if buf:
                            buf.pop()
                    elif ch == "\x1b":
                        # ESC sequence (arrows etc.) – read the next two chars if present, ignore.
                        _ = sys.stdin.read(1)
                        _ = sys.stdin.read(1)
                    else:
                        # Filter non-printable controls
                        if ord(ch) >= 32:
                            buf.append(ch)
                    console.update(_render_typing_panel("".join(buf)))
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


# =================== MINIMAL LEFT-ALIGNED DOT SPINNER (no box) =================
class _DotSpinner:
    """
    Left-aligned circular spinner: a ring of dots with a black filled dot (●) rotating
    among white hollow dots (○). No panel/box—just the spinner line itself.
    """
    def __init__(self, n: int = 8, speed: float = 0.08):
        self.n = max(4, n)
        self.speed = max(0.03, float(speed))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _frame(self, i: int) -> Text:
        idx = i % self.n
        chars = []
        for k in range(self.n):
            chars.append("●" if k == idx else "○")
        line = " ".join(chars)
        # pure left-aligned line, dim to keep it subtle
        return Text(line, style="bold")

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        i = 0
        with Live(self._frame(i), refresh_per_second=30, console=console) as live:
            while not self._stop.is_set():
                i += 1
                live.update(self._frame(i)) 
                time.sleep(self.speed)

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)


@contextmanager
def _with_spinner():
    sp = _DotSpinner()
    try:
        sp.start()
        yield
    finally:
        sp.stop()


# =============================== UTILITIES =====================================
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


def _extract_last_content(messages: list) -> str:
    """Best-effort to get string content of the last message."""
    if not messages:
        return ""
    last = messages[-1]
    c = getattr(last, "content", None)

    if isinstance(c, str):
        return c.strip()

    if isinstance(c, list):
        parts = []
        for p in c:
            if isinstance(p, dict):
                if 'text' in p and isinstance(p['text'], str):
                    parts.append(p['text'])
                elif p.get('type') == 'text' and isinstance(p.get('data') or p.get('content'), str):
                    parts.append(p.get('data') or p.get('content'))
            elif isinstance(p, str):
                parts.append(p)
        return "\n".join(parts).strip()

    if isinstance(last, dict):
        c = last.get("content", "")
        if isinstance(c, str):
            return c.strip()
        if isinstance(c, list):
            return "\n".join(str(x) for x in c if isinstance(x, str)).strip()

    return (str(c) if c is not None else str(last)).strip()


def _collect_recent_tool_summaries(messages: list, max_items: int = 3, max_chars_each: int = 400) -> list[str]:
    """Grab the most recent ToolMessage snippets to show something meaningful."""
    out = []
    for m in reversed(messages or []):
        is_tool = isinstance(m, ToolMessage) or (isinstance(m, dict) and m.get("type") == "tool")
        if not is_tool:
            continue
        text = getattr(m, "content", None)
        if not isinstance(text, str):
            if isinstance(m, dict):
                text = m.get("content")
        if isinstance(text, str) and text.strip():
            out.append(text.strip()[:max_chars_each])
            if len(out) >= max_items:
                break
    return list(reversed(out))


def _synthesize_fallback_final(messages: list) -> str:
    """Guaranteed, concise FINAL message when the model outputs nothing."""
    tools = _collect_recent_tool_summaries(messages)
    tool_lines = "\n".join(f"  - {t}" for t in tools) if tools else "  - (no recent tool output)"
    return (
        "FINAL:\n"
        "- Completed TODOS with statuses: (unknown; model returned empty output)\n"
        "- Files changed: (unknown)\n"
        "- Important command outputs (short):\n"
        f"{tool_lines}\n"
        "- Follow-ups/blockers: Model produced a blank response. Re-run the action or inspect tool logs above."
    )


# ================================ ROOT =========================================
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


# ================================ COMMANDS =====================================
@app.command(help="Open an interactive chat with the agent.")
def chat(
    llm: Optional[str] = typer.Option(None, "--llm", help="anthropic | gemini"),
    project_dir: Path = typer.Option(Path.cwd(), "--project-dir", exists=True, file_okay=False),
    apply: bool = typer.Option(False, "--apply", help="Apply writes and run commands without interactive confirm."),
    mode: str = typer.Option("react", "--mode", help="react | deep"),
    auto: bool = typer.Option(False, "--auto", help="Autonomy mode: plan+act with no questions (deep mode only)."),
):
    provider = resolve_provider(llm)
    mode = (mode or "react").lower()
    if mode not in {"react", "deep"}:
        mode = "react"

    if mode == "deep":
        seed = AUTO_DEEP_INSTR if auto else None
        agent = build_deep_agent(provider=provider, project_dir=project_dir, apply=apply, instruction_seed=seed)
        session_title = "LangChain Code Agent • Deep Chat"
    else:
        agent = build_react_agent(provider=provider, project_dir=project_dir, apply=apply)
        session_title = "LangChain Code Agent • Chat"

    _print_session_header(session_title, provider, project_dir, interactive=True, apply=apply)

    history: list = []  # for ReAct
    msgs: list = []     # for Deep

    try:
        while True:
            # ---- LIVE INPUT INSIDE GREEN BOUNDARY ----
            user = _readline_in_panel()
            if not user:
                continue

            low = user.lower()
            if low in {"cls", "clear", "/clear"}:
                _print_session_header(session_title, provider, project_dir, interactive=True, apply=apply)
                history.clear()
                msgs.clear()
                continue
            if low in {"exit", "quit", ":q", "/exit"}:
                console.print("\n[bold]Goodbye![/bold]")
                break

            coerced = _maybe_coerce_img_command(user)

            # Keep a static transcript panel of what was submitted
            console.print(_panel_user_input(user))

            # ---- INFERENCE WITH LEFT-ALIGNED DOT SPINNER (NO BOX) ----
            if mode == "deep":
                msgs.append({"role": "user", "content": coerced})
                with _with_spinner():
                    # Simple retry logic - only one retry if empty response
                    for attempt in range(2):
                        res = agent.invoke({"messages": msgs})
                        if isinstance(res, dict) and "messages" in res:
                            msgs = res["messages"]
                            last_content = _extract_last_content(msgs).strip()
                            if last_content:
                                break
                            elif attempt == 0:
                                msgs.append({
                                    "role": "system",
                                    "content": "You MUST use your tools to complete the request. Do NOT ask questions - act directly. Provide a FINAL answer."
                                })
                output = _extract_last_content(msgs).strip() or "No response generated."
                console.print(_panel_agent_output(output))

            else:
                with _with_spinner():
                    res = agent.invoke({"input": coerced, "chat_history": history})
                output = res.get("output", "") if isinstance(res, dict) else str(res)
                if not output.strip():
                    output = "No response generated. Try rephrasing your request."
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

    console.print(_panel_user_input(request))
    with _with_spinner():
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

    console.print(_panel_user_input(bug_input.strip() or "Fix the bug using the provided log."))
    with _with_spinner():
        res = agent.invoke({"input": bug_input.strip() or "Fix the bug using the provided log."})
    output = res.get("output", "") if isinstance(res, dict) else str(res)
    console.print(_panel_agent_output(output, title="Fix Result"))


@app.command(help="Analyze any codebase and generate insights.")
def analyze(
    request: str = typer.Argument(..., help='e.g. "What are the main components of this project?"'),
    llm: Optional[str] = typer.Option(None, "--llm", help="anthropic | gemini"),
    project_dir: Path = typer.Option(Path.cwd(), "--project-dir", exists=True, file_okay=False),
):
    """
    Run the deep agent to analyze the codebase and generate insights.
    """
    provider = resolve_provider(llm)
    agent = build_deep_agent(provider=provider, project_dir=project_dir, apply=False)

    _print_session_header("LangChain Code Agent • Analyze", provider, project_dir, interactive=False, apply=False)

    console.print(_panel_user_input(request))
    with _with_spinner():
        res = agent.invoke({"messages": [{"role": "user", "content": request}]})
    output = _extract_last_content(res.get("messages", [])).strip() if isinstance(res, dict) and "messages" in res else str(res)
    console.print(_panel_agent_output(output, title="Analysis Result"))


def main() -> None:
    """
    Entrypoint for the langcode console script.
    """
    app()


if __name__ == "__main__":
    main()
