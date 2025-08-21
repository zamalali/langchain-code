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
from .agent.react import build_react_agent, build_deep_agent
from .workflows.feature_impl import FEATURE_INSTR
from .workflows.bug_fix import BUGFIX_INSTR
from .workflows.auto import AUTO_DEEP_INSTR
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.messages import ToolMessage


app = typer.Typer(add_completion=False, help="LangCode – ReAct + tools code agent CLI.")
console = Console()
PROMPT = "[bold green]langcode[/bold green] [dim]›[/dim] "


def print_langcode_ascii(
    console: Console,
    text: str = "LangCode",
    font: str = "ansi_shadow",
    gradient: str = "dark_to_light",
) -> None:
    """
    Render a single-shot ASCII banner with a left-to-right green gradient.
    """
    def _hex_to_rgb(h):
        h = h.lstrip("#")
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    def _lerp(a, b, t):
        return int(a + (b - a) * t)

    def _interpolate_palette(palette, width):
        if width <= 1:
            return [palette[0]]
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


def session_banner(
    provider: Optional[str],
    project_dir: Path,
    title_text: str,
    *,
    interactive: bool = False,
    apply: bool = False,
    test_cmd: Optional[str] = None,
    tips: Optional[list[str]] = None,
) -> Panel:
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


def _print_session_header(
    title: str,
    provider: Optional[str],
    project_dir: Path,
    *,
    interactive: bool = False,
    apply: bool = False,
    test_cmd: Optional[str] = None,
    tips: Optional[list[str]] = None,
) -> None:
    """
    Clear the screen and draw the LangCode header, banner, and a separator rule.
    """
    console.clear()
    print_langcode_ascii(console, text="LangCode", font="ansi_shadow", gradient="dark_to_light")
    console.print(
        session_banner(
            provider,
            project_dir,
            title,
            interactive=interactive,
            apply=apply,
            test_cmd=test_cmd,
            tips=tips,
        )
    )
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


MAX_RECOVERY_STEPS = 2  # (currently unused but kept for future guardrail logic)


def _looks_like_requesting_paths(text: str) -> bool:
    """Detect model asking user for file paths or permission instead of acting."""
    t = (text or "").lower()
    needles = [
        "provide the file path",
        "please provide the path",
        "i need the path",
        "cannot access local files",
        "share the file contents",
        "may i proceed",
        "ask for confirmation",
        "do you want me to",
    ]
    return any(n in t for n in needles)


def _self_heal_directive(user_goal: str) -> str:
    """Strong, generic directive that forces discovery + action without questions."""
    return (
        "SELF-HEAL DIRECTIVE:\n"
        "You returned an empty or non-actionable reply (or asked the user for paths). "
        "Do NOT ask questions. Discover and act with tools immediately.\n\n"
        "Required now (no chatter):\n"
        "1) Discover relevant files: use list_dir, glob('**/*'), glob('docs/**/*.md'), glob('src/**/*'), "
        "   and grep for key symbols. Read targets with read_file.\n"
        "2) Perform the requested changes end-to-end using edit_by_diff / write_file.\n"
        "3) Run at least one run_cmd (git status/diff/add/commit/push or tests) and capture stdout/stderr.\n"
        "4) If deletion is requested, use delete_file (or write_file with empty content as last resort).\n"
        "5) Attempt git add/commit and push (determine branch via run_cmd). Do not ask for credentials—attempt and report.\n"
        "6) Produce exactly one final message starting with 'FINAL:' summarizing todos, files changed, "
        "   important command outputs, and follow-ups.\n\n"
        f"User goal/context: {user_goal}\n"
        "Execute now."
    )


def _extract_last_content(messages: list) -> str:
    """Best-effort to get string content of the last message."""
    if not messages:
        return ""
    last = messages[-1]
    c = getattr(last, "content", None)

    # If regular string, return it
    if isinstance(c, str):
        return c.strip()

    if isinstance(c, list):
        parts = []
        for p in c:
            # Common LC/Graph part formats:
            # {'type': 'text', 'text': '...'}   or   {'text': '...'}
            if isinstance(p, dict):
                if 'text' in p and isinstance(p['text'], str):
                    parts.append(p['text'])
                elif p.get('type') == 'text' and isinstance(p.get('data') or p.get('content'), str):
                    parts.append(p.get('data') or p.get('content'))
            elif isinstance(p, str):
                parts.append(p)
        return "\n".join(parts).strip()

    # Dict message
    if isinstance(last, dict):
        c = last.get("content", "")
        if isinstance(c, str):
            return c.strip()
        if isinstance(c, list):
            return "\n".join(str(x) for x in c if isinstance(x, str)).strip()

    # Fallback
    return (str(c) if c is not None else str(last)).strip()


def _has_tool_activity(messages: list) -> bool:
    """Detect any tool IO in the transcript."""
    for m in messages or []:
        if isinstance(m, ToolMessage):
            return True
        if getattr(m, "type", "") == "tool":
            return True
        if isinstance(m, dict) and m.get("type") == "tool":
            return True
    return False


def _saw_run_cmd(messages: list) -> bool:
    """Heuristic for your run_cmd tool (prints '$ <cmd>' and '(exit N)')."""
    for m in messages or []:
        c = getattr(m, "content", None)
        if isinstance(c, str) and ("\n(exit " in c or c.startswith("$ ")):
            return True
        if isinstance(m, dict):
            c = m.get("content")
            if isinstance(c, str) and ("\n(exit " in c or c.startswith("$ ")):
                return True
    return False


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
            apply=False,
            test_cmd=None,
            tips=[
                "Quick start:",
                "• chat         Open an interactive session with the agent. (supports --apply in some modes)",
                "• feature      Plan → search → edit → verify. (supports --apply)",
                "• fix          Diagnose & patch a bug (use --log PATH). (supports --apply)",
                "Tip: run any command with --help for details.",
            ],
        )
        typer.echo(ctx.get_help())
        raise typer.Exit()


def _mentions_write_todos_validation(messages: list) -> bool:
    """
    Detect common pydantic/validation complaints around write_todos / state.files.
    """
    needles = (
        "validation error for write_todos",
        "state.files",
        "Field required",
        "errors.pydantic.dev",
    )
    for m in messages or []:
        c = getattr(m, "content", None)
        if isinstance(c, str):
            lc = c.lower()
            if any(n in lc for n in needles):
                return True
        elif isinstance(m, dict):
            c = m.get("content")
            if isinstance(c, str) and any(n in c.lower() for n in needles):
                return True
    return False


@app.command(help="Open an interactive chat with the agent.")
def chat(
    llm: Optional[str] = typer.Option(None, "--llm", help="anthropic | gemini"),
    project_dir: Path = typer.Option(Path.cwd(), "--project-dir", exists=True, file_okay=False),
    mode: str = typer.Option("react", "--mode", help="react | deep"),
    auto: bool = typer.Option(False, "--auto", help="Autonomy mode: plan+act with no questions (deep mode only)."),
):
    provider = resolve_provider(llm)
    mode = (mode or "react").lower()
    if mode not in {"react", "deep"}:
        mode = "react"

    # Build the agent
    if mode == "deep":
        seed = AUTO_DEEP_INSTR if auto else None
        agent = build_deep_agent(provider=provider, project_dir=project_dir, instruction_seed=seed, apply=auto,)
        session_title = "LangChain Code Agent • Deep Chat"
        if auto:
            session_title += " (Auto)"
    else:
        agent = build_react_agent(provider=provider, project_dir=project_dir)
        session_title = "LangChain Code Agent • Chat"

    _print_session_header(session_title, provider, project_dir, interactive=True)

    history: list = []   # for ReAct
    msgs: list = []      # for Deep (LangGraph-style dict messages)

    try:
        while True:
            user = console.input(PROMPT).strip()
            if not user:
                continue

            low = user.lower()
            if low in {"cls", "clear", "/clear"}:
                _print_session_header(session_title, provider, project_dir, interactive=True)
                history.clear()
                msgs.clear()
                continue
            if low in {"exit", "quit", ":q", "/exit"}:
                console.print("\n[bold]Goodbye![/bold]")
                break

            coerced = _maybe_coerce_img_command(user)

            # ----------------------------
            # Deep (LangGraph) mode
            # ----------------------------
            if mode == "deep":
                msgs.append({"role": "user", "content": coerced})

                if auto:
                    msgs.append({
                        "role": "system",
                        "content": (
                            "AUTOPILOT: Start now. Discover files (glob/list_dir/grep), read targets (read_file), "
                            "perform edits (edit_by_diff/write_file), and run at least one run_cmd (git/tests) "
                            "capturing stdout/stderr + exit code. Then produce one 'FINAL:' report and STOP. No questions."
                        )
                    })

                config = {"configurable": {"recursion_limit": 50}}
                try:
                    res = agent.invoke({"messages": msgs}, config=config)
                    if isinstance(res, dict) and "messages" in res:
                        msgs = res["messages"]
                    else:
                        console.print(_panel_agent_output("Error: Invalid response format from agent"))
                        continue
                except Exception as e:
                    if "recursion" in str(e).lower():
                        console.print(_panel_agent_output(f"Agent hit recursion limit. Last response: {_extract_last_content(msgs)}"))
                    else:
                        console.print(_panel_agent_output(f"Agent error: {e}"))
                    continue

                # Simple retry for empty responses (max 1 retry)
                last_content = _extract_last_content(msgs).strip()
                if not last_content:
                    msgs.append({
                        "role": "system",
                        "content": "You must provide a response. Use your tools to complete the request and give a clear answer."
                    })
                    try:
                        res = agent.invoke({"messages": msgs}, config=config)
                        if isinstance(res, dict) and "messages" in res:
                            msgs = res["messages"]
                        last_content = _extract_last_content(msgs).strip()
                    except Exception as e:
                        last_content = f"Agent failed after retry: {e}"

                output = last_content or "No response generated."
                console.print(_panel_agent_output(output))

            # ----------------------------
            # ReAct (LangChain) mode
            # ----------------------------
            else:
                try:
                    res = agent.invoke({"input": coerced, "chat_history": history})
                    output = res.get("output", "") if isinstance(res, dict) else str(res)

                    # Blank-output guardrail
                    if not output.strip():
                        steps = res.get("intermediate_steps") if isinstance(res, dict) else None
                        if steps:
                            previews = []
                            for pair in steps[-3:]:
                                try:
                                    previews.append(str(pair))
                                except Exception:
                                    continue
                            output = "Model returned empty output. Recent steps:\n" + "\n".join(previews)
                        else:
                            output = "No response generated. Try rephrasing your request."

                    console.print(_panel_agent_output(output))
                    history.append(HumanMessage(content=coerced))
                    history.append(AIMessage(content=output))

                except Exception as e:
                    console.print(_panel_agent_output(f"ReAct agent error: {e}"))

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
    agent = build_react_agent(
        provider=provider,
        project_dir=project_dir,
        apply=apply,
        test_cmd=test_cmd,
        instruction_seed=FEATURE_INSTR,
    )

    _print_session_header(
        "LangChain Code Agent • Feature",
        provider,
        project_dir,
        interactive=False,
        apply=apply,
        test_cmd=test_cmd,
    )
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
    agent = build_react_agent(
        provider=provider,
        project_dir=project_dir,
        apply=apply,
        test_cmd=test_cmd,
        instruction_seed=BUGFIX_INSTR,
    )

    bug_input = request or ""
    if log:
        bug_input += "\n\n--- ERROR LOG ---\n" + Path(log).read_text(encoding="utf-8")

    _print_session_header(
        "LangChain Code Agent • Fix",
        provider,
        project_dir,
        interactive=False,
        apply=apply,
        test_cmd=test_cmd,
    )
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

    _print_session_header(
        "LangChain Code Agent • Analyze",
        provider,
        project_dir,
        interactive=False,
        apply=False,
    )
    res = agent.invoke({"messages": [{"role": "user", "content": request}]})
    output = (
        _extract_last_content(res.get("messages", [])).strip()
        if isinstance(res, dict) and "messages" in res
        else str(res)
    )
    console.print(_panel_agent_output(output, title="Analysis Result"))


def main() -> None:
    """
    Entrypoint for the langcode console script.
    """
    app()


if __name__ == "__main__":
    main()
