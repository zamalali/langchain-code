from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import warnings
import sys
import os
import click  # used for fallback editing if no terminal editor
import platform
from collections import OrderedDict
import shutil
import subprocess
import difflib
from datetime import datetime

# cross-platform raw key input
try:
    import termios
    import tty
except Exception:  # windows
    termios = None
    tty = None
try:
    import msvcrt  # type: ignore
except Exception:
    msvcrt = None

import typer
from rich.console import Console, Group
from rich.panel import Panel
from rich.style import Style
from rich.text import Text
from rich.markdown import Markdown
from rich.rule import Rule
from rich.align import Align
from rich.columns import Columns
from rich.table import Table
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


APP_HELP = """
LangCode – ReAct + Tools + Deep (LangGraph) code agent CLI.

Use it to chat with an agent, implement features, fix bugs, or analyze a codebase.

Key flags (for `chat`):
  • --mode [react|deep]   Choose the reasoning engine (default: react).
      - react  : Classic ReAct agent with tools.
      - deep   : LangGraph-style multi-step agent.
  • --auto                 Autopilot (deep mode only). The deep agent will plan+act end-to-end
                           WITHOUT asking questions (it still uses tools safely). Think “hands-off planning”.
  • --apply                Write changes to disk and run commands for you (feature/fix flows).
                           If OFF, the agent proposes diffs only. Think “permission to execute”.

  • --router               Auto-route to the most efficient LLM per query (uses Gemini if --llm not provided).
  • --priority             Router priority: balanced | cost | speed | quality (default: balanced)
  • --verbose              Show router model-selection panels.

Examples:
  • langcode chat --llm anthropic --mode react
  • langcode chat --llm gemini --mode deep --auto
  • langcode chat --router --priority cost --verbose
  • langcode feature "Add a dark mode toggle" --router --priority quality
  • langcode fix --log error.log --test-cmd "pytest -q" --router

Custom instructions:
  • Put project-specific rules in .langcode/langcode.md (created automatically).
  • From the launcher, select “Custom Instructions” to open your editor; or run `langcode instr`.

NEW:
  • Just run `langcode` to open a beautiful interactive launcher.
    Use ↑/↓ to move, ←/→ to change values, Enter to start, h for help, q to quit.
"""

app = typer.Typer(add_completion=False, help=APP_HELP.strip())
console = Console()
PROMPT = "[bold green]langcode[/bold green] [dim]›[/dim] "

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

# =========================
# Custom Instructions (.langcode/langcode.md)
# =========================

LANGCODE_DIRNAME = ".langcode"
LANGCODE_FILENAME = "langcode.md"

def _ensure_langcode_md(project_dir: Path) -> Path:
    """
    Ensure .langcode/langcode.md exists. Return its Path.
    """
    cfg_dir = project_dir / LANGCODE_DIRNAME
    cfg_dir.mkdir(parents=True, exist_ok=True)
    md_path = cfg_dir / LANGCODE_FILENAME
    if not md_path.exists():
        template = f"""# LangCode — Project Custom Instructions

Use this file to add project-specific guidance for the agent.
These notes are appended to the base system prompt in both ReAct and Deep agents.

**Tips**
- Keep it concise and explicit.
- Prefer bullet points and checklists.
- Mention repo conventions, must/shouldn’t rules, style guides, and gotchas.

## Project Rules
- [ ] e.g., All edits must run `pytest -q` and pass.
- [ ] e.g., Use Ruff & Black for Python formatting.

## Code Style & Architecture
- e.g., Follow existing module boundaries in `src/...`

## Tooling & Commands
- e.g., Use `make test` to run the test suite.

---
_Created {datetime.now().strftime('%Y-%m-%d %H:%M')} by LangCode CLI_
"""
        md_path.write_text(template, encoding="utf-8")
    return md_path

def _inline_capture_editor(initial_text: str) -> str:
    """
    Minimal inline editor fallback. Type/paste, then finish with a line: EOF
    (Only used if no terminal editor is available.)
    """
    console.print(Panel.fit(
        Text("Inline editor: Type/paste your content below. End with a line containing only: EOF", style="bold"),
        border_style="cyan",
        title="Inline Editor"
    ))
    if initial_text.strip():
        console.print(Text("---- CURRENT CONTENT (preview) ----", style="dim"))
        console.print(Markdown(initial_text))
        console.print(Text("---- START TYPING (new content will replace file) ----", style="dim"))
    lines: List[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "EOF":
            break
        lines.append(line)
    return "\n".join(lines)

def _pick_terminal_editor() -> Optional[List[str]]:
    """
    Choose a terminal editor command list, prioritizing Vim.
    Order:
      1) $LANGCODE_EDITOR (split on spaces)
      2) $VISUAL
      3) $EDITOR
      4) nvim, vim, vi, nano (first found on PATH)
      5) Windows-specific: check common Vim installation paths
    Returns argv list or None if nothing available.
    """
    if os.environ.get("LANGCODE_EDITOR"):
        return os.environ["LANGCODE_EDITOR"].split()
    
    for var in ("VISUAL", "EDITOR"):
        v = os.environ.get(var)
        if v:
            return v.split()
    
    # Check PATH first
    for cand in ("nvim", "vim", "vi", "nano"):
        if shutil.which(cand):
            return [cand]
    
    # Windows-specific: check common Vim installation paths
    if platform.system().lower() == "windows":
        common_vim_paths = [
            r"C:\Program Files\Vim\vim91\vim.exe",
            r"C:\Program Files\Vim\vim90\vim.exe",
            r"C:\Program Files\Vim\vim82\vim.exe",
            r"C:\Program Files (x86)\Vim\vim91\vim.exe",
            r"C:\Program Files (x86)\Vim\vim90\vim.exe",
            r"C:\Program Files (x86)\Vim\vim82\vim.exe",
            r"C:\tools\vim\vim91\vim.exe",  # Chocolatey
            r"C:\tools\vim\vim90\vim.exe",
            r"C:\Users\{}\scoop\apps\vim\current\vim.exe".format(os.environ.get("USERNAME", "")),  # Scoop
        ]
        
        for vim_path in common_vim_paths:
            if os.path.exists(vim_path):
                return [vim_path]
                
        # Also check if 'vim.exe' is in PATH but not found by shutil.which
        try:
            result = subprocess.run(["where", "vim"], capture_output=True, text=True, check=False)
            if result.returncode == 0 and result.stdout.strip():
                vim_exe = result.stdout.strip().split('\n')[0]
                if os.path.exists(vim_exe):
                    return [vim_exe]
        except Exception:
            pass
    
    return None


def _open_in_terminal_editor(file_path: Path) -> bool:
    """
    Open the file in a terminal editor and block until it exits.
    Returns True if the editor launched, False otherwise.
    """
    cmd = _pick_terminal_editor()
    if not cmd:
        return False
    
    try:
        # On Windows, we might need to handle the console properly
        if platform.system().lower() == "windows":
            # Use shell=False to avoid issues with paths containing spaces
            subprocess.run(cmd + [str(file_path)], check=False)
        else:
            subprocess.run([*cmd, str(file_path)], check=False)
        return True
    except Exception as e:
        console.print(f"[yellow]Failed to launch editor: {e}[/yellow]")
        return False


def _diff_stats(before: str, after: str) -> Dict[str, int]:
    """
    Compute a simple added/removed stat using difflib.ndiff semantics.
    """
    before_lines = before.splitlines()
    after_lines = after.splitlines()
    added = removed = 0
    for line in difflib.ndiff(before_lines, after_lines):
        if line.startswith("+ "):
            added += 1
        elif line.startswith("- "):
            removed += 1
    return {
        "added": added,
        "removed": removed,
        "total_after": len(after_lines),
    }

def _edit_langcode_md(project_dir: Path) -> None:
    """
    Open .langcode/langcode.md in a terminal editor (Vim-first).
    Falls back to $VISUAL/$EDITOR, then click.edit, then inline capture if nothing available.
    After exit, show short stats: lines added/removed and new total lines.
    """
    md_path = _ensure_langcode_md(project_dir)
    original = md_path.read_text(encoding="utf-8")

    launched = _open_in_terminal_editor(md_path)
    edited_text: Optional[str] = None

    if not launched:
        edited_text = click.edit(original, require_save=False)
        if edited_text is None:
            edited_text = _inline_capture_editor(original)
        if edited_text is not None and edited_text != original:
            md_path.write_text(edited_text, encoding="utf-8")
    else:
        edited_text = md_path.read_text(encoding="utf-8")

    if edited_text is None:
        console.print(Panel.fit(Text("No changes saved.", style="yellow"), border_style="yellow"))
        return
    if edited_text == original:
        console.print(Panel.fit(Text("No changes saved (file unchanged).", style="yellow"), border_style="yellow"))
        return

    stats = _diff_stats(original, edited_text)
    console.print(Panel.fit(
        Text.from_markup(
            f"Saved [bold]{md_path}[/bold]\n"
            f"[green]+{stats['added']}[/green] / [red]-{stats['removed']}[/red] • total {stats['total_after']} lines"
        ),
        border_style="green"
    ))

# =========================
# UI helpers
# =========================

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
    tips: Optional[List[str]] = None,
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
    tips: Optional[List[str]] = None,
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
# Interactive Launcher
# =========================

class _Key:
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    ENTER = "ENTER"
    ESC = "ESC"
    Q = "Q"
    H = "H"
    OTHER = "OTHER"

def _read_key() -> str:
    # Windows
    if msvcrt:
        ch = msvcrt.getch()
        if ch in (b'\x00', b'\xe0'):
            ch2 = msvcrt.getch()
            codes = {b'H': _Key.UP, b'P': _Key.DOWN, b'K': _Key.LEFT, b'M': _Key.RIGHT}
            return codes.get(ch2, _Key.OTHER)
        if ch in (b'\r', b'\n'):
            return _Key.ENTER
        if ch in (b'\x1b',):
            return _Key.ESC
        if ch in (b'q', b'Q'):
            return _Key.Q
        if ch in (b'h', b'H', b'?'):
            return _Key.H
        return _Key.OTHER

    # POSIX
    if not sys.stdin.isatty() or not termios or not tty:
        # Fallback: just press Enter to continue
        return _Key.ENTER

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        seq = os.read(fd, 3)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

    if seq == b'\x1b[A':
        return _Key.UP
    if seq == b'\x1b[B':
        return _Key.DOWN
    if seq == b'\x1b[D':
        return _Key.LEFT
    if seq == b'\x1b[C':
        return _Key.RIGHT
    if seq in (b'\r', b'\n'):
        return _Key.ENTER
    if seq == b'\x1b':
        return _Key.ESC
    if seq in (b'q', b'Q'):
        return _Key.Q
    if seq in (b'h', b'H', b'?'):
        return _Key.H
    return _Key.OTHER

def _render_choice(label: str, value: str, focused: bool, enabled: bool = True) -> Text:
    chev = "▶ " if focused else "  "
    style = "bold white" if focused and enabled else ("dim" if not enabled else "white")
    val_style = "bold cyan" if enabled else "dim"
    t = Text(chev + label + ": ", style=style)
    t.append(value, style=val_style)
    return t

def _info_panel_for(field: str) -> Panel:
    info_map = {
        "Command": "[bold]What to do[/bold]: [cyan]chat[/cyan] (interactive), [cyan]feature[/cyan] (plan→edit→verify), [cyan]fix[/cyan] (trace→patch→test), [cyan]analyze[/cyan] (deep code insights).",
        "Engine": "[bold]Reasoning engine[/bold]: [cyan]react[/cyan] = fast tool-using ReAct. [cyan]deep[/cyan] = LangGraph multi-step planner.",
        "Router": "[bold]Router[/bold]: Auto-picks the most efficient LLM per prompt (priority aware). Toggle on for smarter cost/speed/quality.",
        "Priority": "[bold]Routing priority[/bold]: [cyan]balanced[/cyan] | cost | speed | quality – influences model choice.",
        "Autopilot": "[bold]Autopilot[/bold] (deep chat only): Agent plans and executes end-to-end [italic]without asking questions[/italic]. It still uses tools safely, but won't seek confirmation.",
        "Apply": "[bold]Apply[/bold] (feature/fix only): If ON, the agent is allowed to [bold]write files[/bold] and [bold]run commands[/bold]. If OFF, it proposes diffs and commands but does not execute them.",
        "LLM": "[bold]LLM provider[/bold]: Explicitly force [cyan]anthropic[/cyan] or [cyan]gemini[/cyan]. Leave blank to follow router/defaults.",
        "Project": "[bold]Project directory[/bold]: Where the agent operates.",
        "Custom Instructions": "[bold].langcode/langcode.md[/bold]: Project-specific instructions appended to the base prompt. Press Enter to open Vim (or $VISUAL/$EDITOR). Exit with :wq. A shortstat will be shown.",
        "Tests": "[bold]Test command[/bold]: e.g. [cyan]pytest -q[/cyan] or [cyan]npm test[/cyan]. Used by feature/fix flows.",
        "Start": "[bold]Enter[/bold] to launch with the current configuration.",
        "Help": "[bold]h[/bold] to toggle help, [bold]q[/bold] or [bold]Esc[/bold] to quit.",
    }
    text = info_map.get(field, "Use ↑/↓ to move, ←/→ to change values, Enter to start, h for help, q to quit.")
    return Panel(Text.from_markup(text), title="Info", border_style="green", box=box.ROUNDED)

def _help_content() -> Panel:
    return Panel(
        Markdown(APP_HELP.strip()),
        title="LangCode Help",
        border_style="magenta",
        box=box.HEAVY,
        padding=(1, 2),
    )

def _draw_launcher(state: Dict[str, Any], focus_index: int, show_help: bool = False) -> None:
    console.clear()
    print_langcode_ascii(console, text="LangCode", font="ansi_shadow", gradient="dark_to_light")

    header = Align.center(Text("ReAct • Deep • Tools • Safe Edits", style="dim"))
    console.print(header)
    console.print(Rule(style="green"))

    rows = []

    # Build display values
    cmd_val = state["command"]
    engine_val = state["engine"]
    router_val = "on" if state["router"] else "off"
    prio_val = state["priority"]
    auto_enabled = (state["command"] == "chat" and state["engine"] == "deep")
    autopilot_val = "on" if (state["autopilot"] and auto_enabled) else ("off" if auto_enabled else "n/a")
    apply_enabled = state["command"] in ("feature", "fix")
    apply_val = "on" if (state["apply"] and apply_enabled) else ("off" if apply_enabled else "n/a")
    llm_val = state["llm"] or "(auto)"
    proj_val = str(state["project_dir"])
    tests_val = state["test_cmd"] or "(none)"

    # Custom instructions status
    md_path = (state["project_dir"] / LANGCODE_DIRNAME / LANGCODE_FILENAME)
    if md_path.exists():
        try:
            txt = md_path.read_text(encoding="utf-8")
            lines = txt.count("\n") + (1 if txt and not txt.endswith("\n") else 0)
        except Exception:
            lines = 0
        instr_val = f"edit…  ({LANGCODE_DIRNAME}/{LANGCODE_FILENAME}, {lines} lines)"
    else:
        instr_val = f"create…  ({LANGCODE_DIRNAME}/{LANGCODE_FILENAME})"

    labels = [
        ("Command", cmd_val, True),
        ("Engine", engine_val, True),
        ("Router", router_val, True),
        ("Priority", prio_val, state["router"]),
        ("Autopilot", autopilot_val, auto_enabled),
        ("Apply", apply_val, apply_enabled),
        ("LLM", llm_val, True),
        ("Project", proj_val, True),
        ("Custom Instructions", instr_val, True),
        ("Tests", tests_val, state["command"] in ("feature", "fix")),
        ("Start", "Press Enter", True),
    ]

    for idx, (label, value, enabled) in enumerate(labels):
        rows.append(_render_choice(label, value, focused=(idx == focus_index), enabled=enabled))

    left_panel = Panel(
        Align.left(Text.assemble(*[r + Text("\n") for r in rows])),
        title="Launcher",
        border_style="cyan",
        box=box.ROUNDED,
        padding=(1, 2),
    )

    right_panel = _info_panel_for(labels[focus_index][0])

    # main = Columns([left_panel, right_panel], expand=True, equal=True)
    # console.print(main)
    console.print(Group(left_panel, right_panel))

    footer_items = [
        Text("↑/↓ move  ", style="dim"),
        Text("←/→ change  ", style="dim"),
        Text("Enter start  ", style="dim"),
        Text("h for help  ", style="dim"),
        Text("q to quit", style="dim"),
    ]
    console.print(Align.center(Text.assemble(*footer_items)))
    console.print(Rule(style="green"))
    if show_help:
        console.print(_help_content())

def _launcher_loop(initial_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    state = dict(initial_state)
    focus_index = 0
    show_help = False
    fields_order = [
        "Command", "Engine", "Router", "Priority", "Autopilot", "Apply",
        "LLM", "Project", "Custom Instructions", "Tests", "Start"
    ]

    def toggle(field: str, direction: int) -> None:
        if field == "Command":
            opts = ["chat", "feature", "fix", "analyze"]
            i = (opts.index(state["command"]) + direction) % len(opts)
            state["command"] = opts[i]
        elif field == "Engine":
            opts = ["react", "deep"]
            i = (opts.index(state["engine"]) + direction) % len(opts)
            state["engine"] = opts[i]
        elif field == "Router":
            state["router"] = not state["router"]
        elif field == "Priority":
            if not state["router"]:
                return
            opts = ["balanced", "cost", "speed", "quality"]
            i = (opts.index(state["priority"]) + direction) % len(opts)
            state["priority"] = opts[i]
        elif field == "Autopilot":
            # only applicable for deep chat
            if state["command"] == "chat" and state["engine"] == "deep":
                state["autopilot"] = not state["autopilot"]
        elif field == "Apply":
            # only for feature/fix
            if state["command"] in ("feature", "fix"):
                state["apply"] = not state["apply"]
        elif field == "LLM":
            # cycle: (auto) -> anthropic -> gemini -> (auto)
            opts = [None, "anthropic", "gemini"]
            cur = state["llm"]
            i = (opts.index(cur) + direction) % len(opts)
            state["llm"] = opts[i]
        elif field == "Project":
            # prompt for path
            path = console.input("[bold]Project directory[/bold] (Enter to keep current): ").strip()
            if path:
                p = Path(path).expanduser().resolve()
                if p.exists() and p.is_dir():
                    state["project_dir"] = p
                else:
                    console.print(Panel.fit(Text(f"Invalid directory: {p}", style="bold red"), border_style="red"))
                    console.input("Press Enter to continue...")
        elif field == "Custom Instructions":
            try:
                _edit_langcode_md(state["project_dir"])
            except Exception as e:
                console.print(Panel.fit(Text(f"Failed to edit custom instructions: {e}", style="bold red"), border_style="red"))
        elif field == "Tests":
            if state["command"] in ("feature", "fix"):
                t = console.input("[bold]Test command[/bold] (e.g. pytest -q) – empty to clear: ").strip()
                state["test_cmd"] = t or None

    while True:
        _draw_launcher(state, focus_index, show_help=show_help)
        key = _read_key()

        if key in (_Key.Q, _Key.ESC):
            return None
        if key == _Key.H:
            show_help = not show_help
            continue
        if key == _Key.UP:
            focus_index = (focus_index - 1) % len(fields_order)
            continue
        if key == _Key.DOWN:
            focus_index = (focus_index + 1) % len(fields_order)
            continue
        if key == _Key.LEFT:
            toggle(fields_order[focus_index], -1)
            continue
        if key == _Key.RIGHT:
            toggle(fields_order[focus_index], +1)
            continue
        if key == _Key.ENTER:
            field = fields_order[focus_index]
            if field in ("Project", "Custom Instructions", "Tests"):
                # Enter acts like edit for these fields
                toggle(field, +1)
                continue
            if field == "Start":
                return state
            # For toggles and cycling fields, Enter also advances
            toggle(field, +1)
            continue

# =========================
# Root callback
# =========================
@app.callback(invoke_without_command=True)
def _root(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        # Interactive launcher on bare `langcode`
        default_state = {
            "command": "chat",
            "engine": "react",
            "router": False,
            "priority": "balanced",
            "autopilot": False,
            "apply": False,  # diff-only unless turned on for feature/fix
            "llm": None,     # None | "anthropic" | "gemini"
            "project_dir": Path.cwd(),
            "test_cmd": None,
        }
        chosen = _launcher_loop(default_state)
        if not chosen:
            console.print("\n[bold]Goodbye![/bold]")
            raise typer.Exit()

        # Route to the selected command
        cmd = chosen["command"]
        if cmd == "chat":
            chat(
                llm=chosen["llm"],
                project_dir=chosen["project_dir"],
                mode=chosen["engine"],
                auto=bool(chosen["autopilot"] and chosen["engine"] == "deep"),
                router=chosen["router"],
                priority=chosen["priority"],
                verbose=False,
            )
        elif cmd == "feature":
            # Ask for a one-liner request
            req = console.input("[bold]Feature request[/bold] (e.g. Add a dark mode toggle): ").strip()
            if not req:
                console.print("[yellow]Aborted: no request provided.[/yellow]")
                raise typer.Exit()
            feature(
                request=req,
                llm=chosen["llm"],
                project_dir=chosen["project_dir"],
                test_cmd=chosen["test_cmd"],
                apply=chosen["apply"],
                router=chosen["router"],
                priority=chosen["priority"],
                verbose=False,
            )
        elif cmd == "fix":
            # Ask for a one-liner or load a log
            req = console.input("[bold]Bug summary[/bold] (e.g. Fix crash on image upload) [Enter to skip]: ").strip() or None
            log_path = console.input("[bold]Path to error log[/bold] [Enter to skip]: ").strip()
            log = Path(log_path) if log_path else None
            fix(
                request=req,
                log=log if log and log.exists() else None,
                llm=chosen["llm"],
                project_dir=chosen["project_dir"],
                test_cmd=chosen["test_cmd"],
                apply=chosen["apply"],
                router=chosen["router"],
                priority=chosen["priority"],
                verbose=False,
            )
        else:  # analyze
            req = console.input("[bold]Analysis question[/bold] (e.g. What are the main components?): ").strip()
            if not req:
                console.print("[yellow]Aborted: no question provided.[/yellow]")
                raise typer.Exit()
            analyze(
                request=req,
                llm=chosen["llm"],
                project_dir=chosen["project_dir"],
                router=chosen["router"],
                priority=chosen["priority"],
                verbose=False,
            )
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

@app.command(name="instr", help="Open or create project-specific instructions (.langcode/langcode.md) in your editor.")
def edit_instructions(
    project_dir: Path = typer.Option(Path.cwd(), "--project-dir", exists=True, file_okay=False)
):
    _print_session_header(
        "LangChain Code Agent • Custom Instructions",
        provider=None,
        project_dir=project_dir,
        interactive=False
    )
    _edit_langcode_md(project_dir)

def main() -> None:
    app()

if __name__ == "__main__":
    main()
