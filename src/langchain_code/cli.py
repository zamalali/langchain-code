from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import warnings
from collections import OrderedDict

import typer
from rich.console import Console
from rich.panel import Panel
from rich.style import Style
from rich.text import Text
from rich.markdown import Markdown
from rich.rule import Rule
from rich import box
from pyfiglet import Figlet
from rich.progress import Progress, SpinnerColumn, TextColumn

# Silence specific noisy warnings (Pydantic "typing.NotRequired" spam)
warnings.filterwarnings("ignore", message=r"typing\.NotRequired is not a Python type.*")
warnings.filterwarnings("ignore", category=UserWarning, module=r"pydantic\._internal.*")

# --- router-aware helpers from config ---
from .config import resolve_provider as _resolve_provider_base
from .config import get_model, get_model_info

from .agent.react import build_react_agent, build_deep_agent
from .workflows.feature_impl import FEATURE_INSTR
from .workflows.bug_fix import BUGFIX_INSTR
from .workflows.auto import AUTO_DEEP_INSTR
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.messages import ToolMessage


APP_HELP = """
LangCode – ReAct + Tools + Deep (LangGraph) code agent CLI.

Use it to chat with an agent, implement features, fix bugs, or analyze a codebase.

Key flags (for `chat`):
  • --mode [react|deep]   Choose the reasoning engine (default: react).
      - react  : Classic ReAct agent with tools.
      - deep   : LangGraph-style multi-step agent.
  • --auto                 Autopilot (deep mode only): plan + act with no questions.
  • --router               Auto-route to the most efficient LLM per query (uses Gemini if --llm not provided).
  • --priority             Router priority: balanced | cost | speed | quality (default: balanced)
  • --verbose              Show router model-selection panels.

Examples:
  • langcode chat --llm anthropic --mode react
  • langcode chat --llm gemini --mode deep --auto
  • langcode chat --router --priority cost --verbose
  • langcode feature "Add a dark mode toggle" --router --priority quality
  • langcode fix --log error.log --test-cmd "pytest -q" --router
"""

app = typer.Typer(add_completion=False, help=APP_HELP.strip())
console = Console()
PROMPT = "[bold green]langcode[/bold green] [dim]›[/dim] "

# ---------- tiny in-memory agent cache (per model/provider/mode/project) ----------
_AGENT_CACHE: "OrderedDict[Tuple[str, str, str, str, bool], Any]" = OrderedDict()
_AGENT_CACHE_MAX = 6

def _agent_cache_get(key: Tuple[str, str, str, str, bool]):
    if key in _AGENT_CACHE:
        _AGENT_CACHE.move_to_end(key)
        return _AGENT_CACHE[key]
    return None

def _agent_cache_put(key: Tuple[str, str, str, str, bool], value: Any) -> None:
    _AGENT_CACHE[key] = value
    _AGENT_CACHE.move_to_end(key)
    while len(_AGENT_CACHE) > _AGENT_CACHE_MAX:
        _AGENT_CACHE.popitem(last=False)

# ---------- UI helpers ----------
def print_langcode_ascii(
    console: Console,
    text: str = "LangCode",
    font: str = "ansi_shadow",
    gradient: str = "dark_to_light",
) -> None:
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
    model_info: Optional[Dict[str, Any]] = None,
    router_enabled: bool = False,
) -> Panel:
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
    if router_enabled:
        badge.append("  [ROUTER ON]", style="bold green")
    if apply:
        badge.append("  [APPLY MODE]", style="bold red")
    if test_cmd:
        badge.append(f"  tests: {test_cmd}", style="italic")
    if badge:
        body.append("\n")
        body.append_text(badge)

    if model_info:
        body.append("\n")
        model_line = (
            f"Model: {model_info.get('model_name', '(unknown)')} "
            f"[{model_info.get('langchain_model_name', '?')}]"
            f" • priority={model_info.get('priority_used','balanced')}"
        )
        body.append(model_line, style="dim")

    if interactive:
        body.append("\n\n")
        body.append("Type your request. /clear to redraw, /exit or /quit to quit. Ctrl+C also exits.\n", style="dim")

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
    model_info: Optional[Dict[str, Any]] = None,
    router_enabled: bool = False,
) -> None:
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
            model_info=model_info,
            router_enabled=router_enabled,
        )
    )
    console.print(Rule(style="green"))

def _panel_agent_output(text: str, title: str = "Agent") -> Panel:
    body = Markdown(text) if ("```" in text or "\n#" in text) else Text(text)
    return Panel.fit(body, title=title, border_style="cyan", box=box.ROUNDED, padding=(0, 1))

def _panel_router_choice(info: Dict[str, Any]) -> Panel:
    if not info:
        body = Text("Router active, but no model info available.", style="dim")
    else:
        name = info.get("model_name", "(unknown)")
        langchain_name = info.get("langchain_model_name", "?")
        provider = info.get("provider", "?").upper()
        priority = info.get("priority_used", "balanced")
        latency = info.get("latency_tier", "?")
        rs = info.get("reasoning_strength", "?")
        ic = info.get("input_cost_per_million", "?")
        oc = info.get("output_cost_per_million", "?")
        ctx = info.get("context_window", "?")
        body = Text.from_markup(
            f"[bold]Router:[/bold] {provider} → [bold]{name}[/bold] [dim]({langchain_name})[/dim]\n"
            f"[dim]priority={priority} • latency_tier={latency} • reasoning={rs}/10 • "
            f"cost=${ic}M in/${oc}M out • ctx={ctx} tokens[/dim]"
        )
    return Panel.fit(body, title="Model Selection", border_style="green", box=box.ROUNDED, padding=(0, 1))

def _show_loader() -> Progress:
    progress = Progress(
        SpinnerColumn(spinner_name="dots", style=Style(color="green")),
        TextColumn("[progress.description]{task.description}", style=Style(color="white")),
        transient=True,
    )
    progress.add_task("[bold]Processing...", total=None)
    return progress

def _maybe_coerce_img_command(raw: str) -> str:
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

MAX_RECOVERY_STEPS = 2

def _extract_last_content(messages: list) -> str:
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

# =========================
# Provider & router helpers
# =========================
def _resolve_provider(llm_opt: Optional[str], router: bool) -> str:
    if llm_opt:
        return _resolve_provider_base(llm_opt)
    if router:
        return "gemini"
    return _resolve_provider_base(None)

def _build_react_agent_with_optional_llm(provider: str, project_dir: Path, llm=None, **kwargs):
    try:
        if llm is not None:
            return build_react_agent(provider=provider, project_dir=project_dir, llm=llm, **kwargs)
    except TypeError:
        pass
    return build_react_agent(provider=provider, project_dir=project_dir, **kwargs)

def _build_deep_agent_with_optional_llm(provider: str, project_dir: Path, llm=None, **kwargs):
    try:
        if llm is not None:
            return build_deep_agent(provider=provider, project_dir=project_dir, llm=llm, **kwargs)
    except TypeError:
        pass
    return build_deep_agent(provider=provider, project_dir=project_dir, **kwargs)

# =========================
# Root callback
# =========================
@app.callback(invoke_without_command=True)
def _root(ctx: typer.Context):
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
                "• chat         Open an interactive session (use --mode react|deep; add --auto with deep).",
                "• feature      Plan → search → edit → verify. (supports --apply and --test-cmd)",
                "• fix          Diagnose & patch a bug (use --log PATH, --test-cmd). (supports --apply)",
                "• analyze      Deep agent insights over the codebase.",
                "Examples:",
                "  - langcode chat --llm anthropic --mode react",
                "  - langcode chat --llm gemini --mode deep --auto",
                "  - langcode chat --router --priority cost --verbose",
                "Exit anytime with /exit, /quit, or Ctrl+C. Use /clear to redraw.",
            ],
        )
        typer.echo(ctx.get_help())
        raise typer.Exit()

# =========================
# Commands
# =========================
@app.command(help="Open an interactive chat with the agent. Modes: react | deep (default: react). Use --auto in deep mode for full autopilot (plan+act with no questions).")
def chat(
    llm: Optional[str] = typer.Option(None, "--llm", help="anthropic | gemini"),
    project_dir: Path = typer.Option(Path.cwd(), "--project-dir", exists=True, file_okay=False),
    mode: str = typer.Option("react", "--mode", help="react | deep"),
    auto: bool = typer.Option(False, "--auto", help="Autonomy mode: plan+act with no questions (deep mode only)."),
    router: bool = typer.Option(False, "--router", help="Auto-route to the most efficient LLM per query."),
    priority: str = typer.Option("balanced", "--priority", help="Router priority: balanced | cost | speed | quality"),
    verbose: bool = typer.Option(False, "--verbose", help="Show model selection panels."),
):
    priority = (priority or "balanced").lower()
    if priority not in {"balanced", "cost", "speed", "quality"}:
        priority = "balanced"

    provider = _resolve_provider(llm, router)
    mode = (mode or "react").lower()
    if mode not in {"react", "deep"}:
        mode = "react"

    session_title = "LangChain Code Agent • Deep Chat" if mode == "deep" else "LangChain Code Agent • Chat"
    if mode == "deep" and auto:
        session_title += " (Auto)"
    _print_session_header(session_title, provider, project_dir, interactive=True, router_enabled=router)

    history: list = []  # for ReAct
    msgs: list = []     # for Deep

    # Static agent only when router is off
    static_agent = None
    if not router:
        if mode == "deep":
            seed = AUTO_DEEP_INSTR if auto else None
            static_agent = build_deep_agent(provider=provider, project_dir=project_dir, instruction_seed=seed, apply=auto)
        else:
            static_agent = build_react_agent(provider=provider, project_dir=project_dir)

    prio_limits = {"speed": 30, "cost": 35, "balanced": 45, "quality": 60}

    try:
        while True:
            user = console.input(PROMPT).strip()
            if not user:
                continue

            low = user.lower()
            if low in {"cls", "clear", "/clear"}:
                _print_session_header(session_title, provider, project_dir, interactive=True, router_enabled=router)
                history.clear()
                msgs.clear()
                continue
            if low in {"exit", "quit", ":q", "/exit", "/quit"}:
                console.print("\n[bold]Goodbye![/bold]")
                break

            coerced = _maybe_coerce_img_command(user)

            # Show spinner immediately for zero perceived latency
            with _show_loader():
                agent = static_agent
                model_info = None
                chosen_llm = None

                if router:
                    provider = _resolve_provider(llm, router=True)
                    model_info = get_model_info(provider, coerced, priority)
                    chosen_llm = get_model(provider, coerced, priority)

                    model_key = model_info.get("langchain_model_name") if model_info else "default"
                    cache_key = (
                        "deep" if mode == "deep" else "react",
                        provider,
                        model_key,
                        str(project_dir.resolve()),
                        bool(auto) if mode == "deep" else False,
                    )
                    cached = _agent_cache_get(cache_key)
                    if cached is not None:
                        agent = cached
                    else:
                        if verbose and model_info:
                            console.print(_panel_router_choice(model_info))
                        if mode == "deep":
                            seed = AUTO_DEEP_INSTR if auto else None
                            agent = _build_deep_agent_with_optional_llm(
                                provider=provider,
                                project_dir=project_dir,
                                llm=chosen_llm,
                                instruction_seed=seed,
                                apply=auto,
                            )
                        else:
                            agent = _build_react_agent_with_optional_llm(
                                provider=provider,
                                project_dir=project_dir,
                                llm=chosen_llm,
                            )
                        _agent_cache_put(cache_key, agent)

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

                    config = {"configurable": {"recursion_limit": prio_limits.get(priority, 45)}}
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

                else:
                    # ReAct mode
                    try:
                        res = agent.invoke({"input": coerced, "chat_history": history})
                        output = res.get("output", "") if isinstance(res, dict) else str(res)

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
                        # keep history small for speed and cost
                        if len(history) > 20:
                            history[:] = history[-20:]

                    except Exception as e:
                        console.print(_panel_agent_output(f"ReAct agent error: {e}"))

    except (KeyboardInterrupt, EOFError):
        console.print("\n[bold]Goodbye![/bold]")

@app.command(help="Implement a feature end-to-end (plan → search → edit → verify). Supports --apply and optional --test-cmd (e.g., 'pytest -q').")
def feature(
    request: str = typer.Argument(..., help='e.g. "Add a dark mode toggle in settings"'),
    llm: Optional[str] = typer.Option(None, "--llm", help="anthropic | gemini"),
    project_dir: Path = typer.Option(Path.cwd(), "--project-dir", exists=True, file_okay=False),
    test_cmd: Optional[str] = typer.Option(None, "--test-cmd", help='e.g. "pytest -q" or "npm test"'),
    apply: bool = typer.Option(False, "--apply", help="Apply writes and run commands without interactive confirm."),
    router: bool = typer.Option(False, "--router", help="Auto-route to the most efficient LLM for this request."),
    priority: str = typer.Option("balanced", "--priority", help="balanced | cost | speed | quality"),
    verbose: bool = typer.Option(False, "--verbose", help="Show model selection panel."),
):
    priority = (priority or "balanced").lower()
    if priority not in {"balanced", "cost", "speed", "quality"}:
        priority = "balanced"

    provider = _resolve_provider(llm, router)
    model_info = None
    chosen_llm = None

    if router:
        model_info = get_model_info(provider, request, priority)
        chosen_llm = get_model(provider, request, priority)

    _print_session_header(
        "LangChain Code Agent • Feature",
        provider,
        project_dir,
        interactive=False,
        apply=apply,
        test_cmd=test_cmd,
        model_info=(model_info if (router and verbose) else None),
        router_enabled=router,
    )
    if router and verbose and model_info:
        console.print(_panel_router_choice(model_info))

    # Cache for feature run too
    model_key = (model_info or {}).get("langchain_model_name", "default")
    cache_key = ("react", provider, model_key, str(project_dir.resolve()), False)
    cached = _agent_cache_get(cache_key)
    if cached is None:
        agent = _build_react_agent_with_optional_llm(
            provider=provider,
            project_dir=project_dir,
            llm=chosen_llm,
            apply=apply,
            test_cmd=test_cmd,
            instruction_seed=FEATURE_INSTR,
        )
        _agent_cache_put(cache_key, agent)
    else:
        agent = cached

    with _show_loader():
        res = agent.invoke({"input": request})
        output = res.get("output", "") if isinstance(res, dict) else str(res)
    console.print(_panel_agent_output(output, title="Feature Result"))

@app.command(help="Diagnose & fix a bug (trace → pinpoint → patch → test). Accepts --log, --test-cmd, and supports --apply.")
def fix(
    request: Optional[str] = typer.Argument(None, help='e.g. "Fix crash on image upload"'),
    log: Optional[Path] = typer.Option(None, "--log", exists=True, help="Path to error log or stack trace."),
    llm: Optional[str] = typer.Option(None, "--llm", help="anthropic | gemini"),
    project_dir: Path = typer.Option(Path.cwd(), "--project-dir", exists=True, file_okay=False),
    test_cmd: Optional[str] = typer.Option(None, "--test-cmd", help='e.g. "pytest -q"'),
    apply: bool = typer.Option(False, "--apply", help="Apply writes and run commands without interactive confirm."),
    router: bool = typer.Option(False, "--router", help="Auto-route to the most efficient LLM for this request."),
    priority: str = typer.Option("balanced", "--priority", help="balanced | cost | speed | quality"),
    verbose: bool = typer.Option(False, "--verbose", help="Show model selection panel."),
):
    priority = (priority or "balanced").lower()
    if priority not in {"balanced", "cost", "speed", "quality"}:
        priority = "balanced"

    provider = _resolve_provider(llm, router)

    bug_input = (request or "").strip()
    if log:
        bug_input += "\n\n--- ERROR LOG ---\n" + Path(log).read_text(encoding="utf-8")
    bug_input = bug_input.strip() or "Fix the bug using the provided log."

    model_info = None
    chosen_llm = None
    if router:
        model_info = get_model_info(provider, bug_input, priority)
        chosen_llm = get_model(provider, bug_input, priority)

    _print_session_header(
        "LangChain Code Agent • Fix",
        provider,
        project_dir,
        interactive=False,
        apply=apply,
        test_cmd=test_cmd,
        model_info=(model_info if (router and verbose) else None),
        router_enabled=router,
    )
    if router and verbose and model_info:
        console.print(_panel_router_choice(model_info))

    model_key = (model_info or {}).get("langchain_model_name", "default")
    cache_key = ("react", provider, model_key, str(project_dir.resolve()), False)
    cached = _agent_cache_get(cache_key)
    if cached is None:
        agent = _build_react_agent_with_optional_llm(
            provider=provider,
            project_dir=project_dir,
            llm=chosen_llm,
            apply=apply,
            test_cmd=test_cmd,
            instruction_seed=BUGFIX_INSTR,
        )
        _agent_cache_put(cache_key, agent)
    else:
        agent = cached

    with _show_loader():
        res = agent.invoke({"input": bug_input})
        output = res.get("output", "") if isinstance(res, dict) else str(res)
    console.print(_panel_agent_output(output, title="Fix Result"))

@app.command(help="Analyze any codebase and generate insights (deep agent).")
def analyze(
    request: str = typer.Argument(..., help='e.g. "What are the main components of this project?"'),
    llm: Optional[str] = typer.Option(None, "--llm", help="anthropic | gemini"),
    project_dir: Path = typer.Option(Path.cwd(), "--project-dir", exists=True, file_okay=False),
    router: bool = typer.Option(False, "--router", help="Auto-route to the most efficient LLM for this request."),
    priority: str = typer.Option("balanced", "--priority", help="balanced | cost | speed | quality"),
    verbose: bool = typer.Option(False, "--verbose", help="Show model selection panel."),
):
    priority = (priority or "balanced").lower()
    if priority not in {"balanced", "cost", "speed", "quality"}:
        priority = "balanced"

    provider = _resolve_provider(llm, router)

    model_info = None
    chosen_llm = None
    if router:
        model_info = get_model_info(provider, request, priority)
        chosen_llm = get_model(provider, request, priority)

    _print_session_header(
        "LangChain Code Agent • Analyze",
        provider,
        project_dir,
        interactive=False,
        apply=False,
        model_info=(model_info if (router and verbose) else None),
        router_enabled=router,
    )
    if router and verbose and model_info:
        console.print(_panel_router_choice(model_info))

    model_key = (model_info or {}).get("langchain_model_name", "default")
    cache_key = ("deep", provider, model_key, str(project_dir.resolve()), False)
    cached = _agent_cache_get(cache_key)
    if cached is None:
        agent = _build_deep_agent_with_optional_llm(
            provider=provider,
            project_dir=project_dir,
            llm=chosen_llm,
            apply=False,
        )
        _agent_cache_put(cache_key, agent)
    else:
        agent = cached

    with _show_loader():
        res = agent.invoke({"messages": [{"role": "user", "content": request}]},
                           config={"configurable": {"recursion_limit": 45}})
        output = (
            _extract_last_content(res.get("messages", [])).strip()
            if isinstance(res, dict) and "messages" in res
            else str(res)
        )
    console.print(_panel_agent_output(output, title="Analysis Result"))

def main() -> None:
    app()

if __name__ == "__main__":
    main()
