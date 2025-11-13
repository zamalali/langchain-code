from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import termios  # type: ignore
    import tty  # type: ignore
except Exception:  # pragma: no cover (Windows)
    termios = None  # type: ignore
    tty = None  # type: ignore

try:
    import msvcrt  # type: ignore
except Exception:  # pragma: no cover (POSIX)
    msvcrt = None  # type: ignore

from rich import box
from rich.align import Align
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from .constants import LANGCODE_DIRNAME, LANGCODE_FILENAME
from .display import langcode_ascii_renderable
from .env import (
    edit_env_file,
    load_env_files,
    edit_global_env_file,
    load_global_env,
    env_status_label,
    global_env_status_label,
)
from .instructions import edit_langcode_md
from .mcp import edit_mcp_json, mcp_status_label
from .state import console, edit_feedback


class Key:
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    ENTER = "ENTER"
    ESC = "ESC"
    Q = "Q"
    H = "H"
    OTHER = "OTHER"


def read_key() -> str:
    if msvcrt:
        ch = msvcrt.getch()
        if ch in (b"\x00", b"\xe0"):
            second = msvcrt.getch()
            codes = {b"H": Key.UP, b"P": Key.DOWN, b"K": Key.LEFT, b"M": Key.RIGHT}
            return codes.get(second, Key.OTHER)
        if ch in (b"\r", b"\n"):
            return Key.ENTER
        if ch in (b"\x1b",):
            return Key.ESC
        if ch in (b"q", b"Q"):
            return Key.Q
        if ch in (b"h", b"H", b"?"):
            return Key.H
        return Key.OTHER

    if not sys.stdin.isatty() or not termios or not tty:
        return Key.ENTER

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        seq = os.read(fd, 3)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

    if seq == b"\x1b[A":
        return Key.UP
    if seq == b"\x1b[B":
        return Key.DOWN
    if seq == b"\x1b[D":
        return Key.LEFT
    if seq == b"\x1b[C":
        return Key.RIGHT
    if seq in (b"\r", b"\n"):
        return Key.ENTER
    if seq == b"\x1b":
        return Key.ESC
    if seq in (b"q", b"Q"):
        return Key.Q
    if seq in (b"h", b"H", b"?"):
        return Key.H
    return Key.OTHER


def render_choice(label: str, value: str, focused: bool, enabled: bool = True) -> Text:
    prefix = ">> " if focused else "   "
    style = "bold white" if focused and enabled else ("dim" if not enabled else "white")
    val_style = "bold cyan" if enabled else "dim"
    line = Text(prefix + label + ": ", style=style)
    line.append(str(value), style=val_style)
    return line


def help_content() -> Panel:
    commands = Table.grid(padding=(0, 2))
    commands.add_column("Command", style="bold cyan")
    commands.add_column("Description", style="dim")
    commands.add_row("chat", "Interactive agent chat (react or deep).")
    commands.add_row("feature", "Plan -> edit -> verify with optional --apply/--test-cmd.")
    commands.add_row("fix", "Trace -> patch -> test a bug; accepts logs & apply mode.")
    commands.add_row("analyze", "Deep code insights for the project.")
    commands.add_row("instr", "Open .langcode/langcode.md for project rules.")

    navigation = Table.grid(padding=(0, 2))
    navigation.add_column("Key", style="bold cyan")
    navigation.add_column("Action", style="dim")
    navigation.add_row("up/down", "Move between fields")
    navigation.add_row("left/right", "Toggle or cycle values")
    navigation.add_row("Enter", "Activate the focused field / start command")
    navigation.add_row("h", "Toggle this help overlay")
    navigation.add_row("q / Esc", "Exit the launcher")

    return Panel(
        Align.center(Group(commands, Rule(style="green"), navigation)),
        title="Help",
        border_style="green",
        box=box.ROUNDED,
        padding=(1, 2),
    )


def info_panel_for(field: str) -> Panel:
    info = {
        "Command": "[bold]What to do[/bold]: chat (interactive), feature (plan->edit->verify), fix (trace->patch->test), analyze (deep insights).",
        "Engine": "[bold]Reasoning engine[/bold]: react = fast tool agent, deep = LangGraph planner.",
        "Router": "[bold]Router[/bold]: auto-pick the best LLM per prompt.",
        "Priority": "[bold]Routing priority[/bold]: balanced | cost | speed | quality.",
        "Autopilot": "[bold]Autopilot[/bold] (deep chat): execute without asking questions.",
        "Apply": "[bold]Apply[/bold] (feature/fix): allow writes & running commands.",
        "LLM": "[bold]LLM provider[/bold]: anthropic | gemini | openai | ollama. Use left/right to cycle. Enter on ollama to manage installed models.",
        "Model": "[bold]Model override[/bold]: cycle with left/right to pick a provider model. Enter resets to provider default.",
        "Project": "[bold]Project directory[/bold]: path where the agent operates.",
        "Custom Instructions": "[bold].langcode/langcode.md[/bold]: project notes and rules.",
        "MCP Config": "[bold].langcode/mcp.json[/bold]: configure MCP servers/tools.",
        "Environment": "[bold].env[/bold]: manage project API keys and settings.",
        "Tests": "[bold]Test command[/bold]: e.g. pytest -q, npm test.",
        "Start": "[bold]Enter[/bold] to launch with the current configuration.",
        "Help": "[bold]h[/bold] toggles help, q/Esc exits the launcher.",
    }
    text = info.get(field, "Use arrow keys to navigate, Enter to activate.")
    return Panel(Text.from_markup(text), title="Info", border_style="green", box=box.ROUNDED, padding=(1, 1))


def list_ollama_models() -> List[str]:
    try:
        proc = subprocess.run(
            ["ollama", "list", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            try:
                data = json.loads(proc.stdout)
                if isinstance(data, list):
                    models: List[str] = []
                    for entry in data:
                        name = (entry.get("name") or entry.get("model") or "").strip()
                        if name:
                            models.append(name)
                    return models
            except Exception:
                pass

        proc_plain = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=2)
        if proc_plain.returncode == 0 and proc_plain.stdout:
            out: List[str] = []
            for row in proc_plain.stdout.splitlines()[1:]:
                row = row.strip()
                if not row:
                    continue
                name = row.split()[0]
                if name:
                    out.append(name)
            return out
    except Exception:
        pass
    return []


def provider_model_choices(provider: Optional[str]) -> List[str]:
    if not provider:
        return []
    if provider == "gemini":
        return [
            "gemini-2.0-flash-lite",
            "gemini-2.0-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
        ]
    if provider == "anthropic":
        return [
            "claude-3-5-haiku-20241022",
            "claude-3-7-sonnet-20250219",
            "claude-opus-4-1-20250805",
        ]
    if provider == "openai":
        return ["gpt-4o-mini", "gpt-4o"]
    if provider == "ollama":
        return list_ollama_models()
    return []


def build_launcher_view(state: Dict[str, Any], focus_index: int, show_help: bool) -> Group:
    width = getattr(console.size, "width", 80)

    cmd_val = state["command"]
    engine_val = state["engine"]
    router_val = "on" if state["router"] else "off"
    prio_val = state["priority"]
    auto_enabled = state["command"] == "chat" and state["engine"] == "deep"
    autopilot_val = "on" if (state["autopilot"] and auto_enabled) else ("off" if auto_enabled else "n/a")
    apply_enabled = state["command"] in ("feature", "fix")
    apply_val = "on" if (state["apply"] and apply_enabled) else ("off" if apply_enabled else "n/a")
    llm_val = state["llm"] or "(auto)"
    if state["llm"] == "ollama" and state.get("ollama_model") and not state.get("model_override"):
        model_val = state["ollama_model"]
    else:
        model_val = state.get("model_override", "(default)")

    proj_val = str(state["project_dir"])
    tests_val = state["test_cmd"] or "(none)"
    ollama_names = list_ollama_models() if state["llm"] == "ollama" else []
    start_enabled = True
    if state["llm"] == "ollama":
        if not ollama_names:
            start_enabled = False
        elif state.get("ollama_model") and state["ollama_model"] not in ollama_names:
            start_enabled = False

    md_path = state["project_dir"] / LANGCODE_DIRNAME / LANGCODE_FILENAME
    if md_path.exists():
        try:
            text = md_path.read_text(encoding="utf-8")
            lines = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
        except Exception:
            lines = 0
        instr_val = f"edit...  ({LANGCODE_DIRNAME}/{LANGCODE_FILENAME}, {lines} lines)"
    else:
        instr_val = f"create...  ({LANGCODE_DIRNAME}/{LANGCODE_FILENAME})"

    env_val = env_status_label(state["project_dir"])
    mcp_val = mcp_status_label(state["project_dir"])
    model_enabled = bool(state["llm"]) and not state["router"]

    labels: List[Tuple[str, str, bool]] = [
        ("Command", cmd_val, True),
        ("Engine", engine_val, True),
        ("Router", router_val, True),
        ("Priority", prio_val, state["router"]),
        ("Autopilot", autopilot_val, auto_enabled),
        ("Apply", apply_val, apply_enabled),
        ("LLM", llm_val, True),
        ("Model", model_val, model_enabled),
        ("Project", proj_val, True),
        ("Environment", env_val, True),
        ("Global Environment", global_env_status_label(), True),
        ("Custom Instructions", instr_val, True),
        ("MCP Config", mcp_val, True),
        ("Tests", tests_val, state["command"] in ("feature", "fix")),
        ("Start", "Press Enter", start_enabled),
    ]

    rows = [render_choice(label, value, focused=(idx == focus_index), enabled=enabled) for idx, (label, value, enabled) in enumerate(labels)]
    left_panel = Panel(
        Align.left(Text.assemble(*[row + Text("\n") for row in rows])),
        title="Launcher",
        border_style="cyan",
        box=box.ROUNDED,
        padding=(1, 2),
    )

    field_name = labels[focus_index][0]
    right_panel = info_panel_for(field_name)

    if field_name == "LLM" and state["llm"] == "ollama":
        names = list_ollama_models()
        body = Text()
        body.append("Ollama models detected on this machine:\n", style="bold")
        if names:
            for idx, name in enumerate(names, 1):
                mark = "-> " if name == state.get("ollama_model") else "   "
                body.append(f"{mark}{idx}. {name}\n")
            body.append("\nPress Enter on this field to choose a model.\n", style="dim")
        else:
            body.append(" (None found)\n", style="yellow")
            body.append(" Tip: run `ollama pull llama3.1` to get a default model.\n", style="dim")
            body.append("\n[Start is disabled until a model is installed.]\n", style="dim")
        right_panel = Panel(body, title="Ollama", border_style="cyan", box=box.ROUNDED, padding=(1, 1))

    footer_items = [
        Text("up/down move  ", style="dim"),
        Text("left/right change  ", style="dim"),
        Text("Enter start  ", style="dim"),
        Text("h for help  ", style="dim"),
        Text("q to quit", style="dim"),
    ]

    components: List[Any] = [
        langcode_ascii_renderable(width),
        Align.center(Text("ReAct | Deep | Tools | Safe Edits", style="dim"), width=width),
        Rule(style="green"),
        Group(left_panel, right_panel),
        Align.center(Text.assemble(*footer_items), width=width),
        Rule(style="green"),
    ]
    status_msg = state.get("_status")
    if status_msg:
        status_text = Text(status_msg, style="yellow")
        status_text.no_wrap = False
        status_text.overflow = "fold"
        components.append(Align.center(status_text, width=width))
    if show_help:
        components.append(help_content())
    return Group(*components)


def default_state() -> Dict[str, Any]:
    return {
        "command": "chat",
        "engine": "react",
        "router": False,
        "priority": "balanced",
        "autopilot": False,
        "apply": False,
        "llm": None,
        "project_dir": Path.cwd(),
        "test_cmd": None,
        "ollama_model": None,
        "model_override": None,
    }


def launcher_loop(initial_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    state = dict(initial_state)
    focus_index = 0
    show_help = False
    fields_order = [
        "Command",
        "Engine",
        "Router",
        "Priority",
        "Autopilot",
        "Apply",
        "LLM",
        "Model",
        "Project",
        "Environment",
        "Global Environment",
        "Custom Instructions",
        "MCP Config",
        "Tests",
        "Start",
    ]

    with Live(build_launcher_view(state, focus_index, show_help), console=console, refresh_per_second=24, screen=True) as live:
        def refresh() -> None:
            live.update(build_launcher_view(state, focus_index, show_help))

        def toggle(field: str, direction: int) -> None:
            state.pop("_status", None)
            if field == "Command":
                options = ["chat", "feature", "fix", "analyze"]
                idx = (options.index(state["command"]) + direction) % len(options)
                state["command"] = options[idx]
            elif field == "Engine":
                options = ["react", "deep"]
                idx = (options.index(state["engine"]) + direction) % len(options)
                state["engine"] = options[idx]
            elif field == "Router":
                state["router"] = not state["router"]
            elif field == "Priority" and state["router"]:
                options = ["balanced", "cost", "speed", "quality"]
                idx = (options.index(state["priority"]) + direction) % len(options)
                state["priority"] = options[idx]
            elif field == "Autopilot" and state["command"] == "chat" and state["engine"] == "deep":
                state["autopilot"] = not state["autopilot"]
            elif field == "Apply" and state["command"] in ("feature", "fix"):
                state["apply"] = not state["apply"]
            elif field == "LLM":
                options = [None, "anthropic", "gemini", "openai", "ollama"]
                idx = (options.index(state["llm"]) + direction) % len(options)
                state["llm"] = options[idx]
                state["model_override"] = None
                os.environ.pop("LANGCODE_MODEL_OVERRIDE", None)
            elif field == "Model":
                if state.get("router"):
                    state["_status"] = "Disable router to choose a specific model."
                    return
                if not state.get("llm"):
                    state["_status"] = "Select an LLM provider first."
                    return
                choices = provider_model_choices(state["llm"])
                if not choices:
                    state["_status"] = "No models available for this provider."
                    state["model_override"] = None
                    os.environ.pop("LANGCODE_MODEL_OVERRIDE", None)
                    return
                options: List[Optional[str]] = [None, *choices]
                current = state.get("model_override")
                idx = options.index(current) if current in options else 0
                idx = (idx + direction) % len(options)
                selected = options[idx]
                if selected is None:
                    state["model_override"] = None
                    os.environ.pop("LANGCODE_MODEL_OVERRIDE", None)
                    state["_status"] = "Model reset to provider default."
                else:
                    state["model_override"] = selected
                    os.environ["LANGCODE_MODEL_OVERRIDE"] = selected
                    state["_status"] = f"Model set to {selected}."

        while True:
            refresh()
            key = read_key()

            if key in (Key.Q, Key.ESC):
                return None
            if key == Key.H:
                show_help = not show_help
                continue
            if key == Key.UP:
                focus_index = (focus_index - 1) % len(fields_order)
                if fields_order[focus_index] == "Model":
                    state["_status"] = "Use left/right to cycle models; Enter resets to default."
                else:
                    state.pop("_status", None)
                continue
            if key == Key.DOWN:
                focus_index = (focus_index + 1) % len(fields_order)
                if fields_order[focus_index] == "Model":
                    state["_status"] = "Use left/right to cycle models; Enter resets to default."
                else:
                    state.pop("_status", None)
                continue
            if key == Key.LEFT:
                toggle(fields_order[focus_index], -1)
                continue
            if key == Key.RIGHT:
                toggle(fields_order[focus_index], +1)
                continue

            if key != Key.ENTER:
                continue

            field = fields_order[focus_index]

            if field == "LLM":
                if state["llm"] != "ollama":
                    toggle("LLM", +1)
                    continue
                names = list_ollama_models()
                if not names:
                    live.stop()
                    console.print(Panel.fit(Text("No Ollama models detected. Install one (e.g., `ollama pull llama3.1`).", style="bold yellow"), border_style="yellow"))
                    console.input("Press Enter to continue...")
                    live.start(refresh=True)
                    continue
                live.stop()
                console.print(Panel.fit(Text("Select an Ollama model by number:", style="bold"), border_style="cyan"))
                for idx, name in enumerate(names, 1):
                    console.print(f"  {idx}. {name}")
                choice = console.input("Your choice: ").strip()
                live.start(refresh=True)
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(names):
                        state["ollama_model"] = names[idx]
                        os.environ["LANGCODE_OLLAMA_MODEL"] = names[idx]
                        state["_status"] = f"Ollama model set to {names[idx]}."
                    else:
                        state["_status"] = "Invalid selection."
                except Exception:
                    state["_status"] = "Invalid input."
                continue

            if field == "Model":
                if state.get("router"):
                    state["_status"] = "Disable router to pick a specific model."
                    continue
                if not state.get("llm"):
                    state["_status"] = "Select an LLM provider first."
                    continue
                choices = provider_model_choices(state["llm"])
                if not choices:
                    state["_status"] = "No models available; using provider default."
                    state["model_override"] = None
                    os.environ.pop("LANGCODE_MODEL_OVERRIDE", None)
                    continue
                state["model_override"] = None
                os.environ.pop("LANGCODE_MODEL_OVERRIDE", None)
                state["_status"] = "Model reset to provider default."
                continue

            if field == "Project":
                live.stop()
                inp = console.input("[bold]Project directory[/bold] (Enter to keep current): ").strip()
                if inp:
                    project = Path(inp).expanduser().resolve()
                    if project.exists() and project.is_dir():
                        state["project_dir"] = project
                    else:
                        console.print(Panel.fit(Text(f"Invalid directory: {project}", style="bold red"), border_style="red"))
                        console.input("Press Enter to continue...")
                live.start(refresh=True)
                continue

            if field == "Environment":
                live.stop()
                try:
                    with edit_feedback():
                        edit_env_file(state["project_dir"])
                        load_env_files(state["project_dir"], override_existing=False)
                except Exception as exc:
                    console.print(Panel.fit(Text(f"Failed to edit environment: {exc}", style="bold red"), border_style="red"))
                    console.input("Press Enter to continue...")
                live.start(refresh=True)
                continue

            if field == "Global Environment":
                live.stop()
                try:
                    with edit_feedback():
                        edit_global_env_file()
                        load_global_env(override_existing=False)
                except Exception as exc:
                    console.print(Panel.fit(Text(f"Failed to edit global environment: {exc}", style="bold red"), border_style="red"))
                    console.input("Press Enter to continue...")
                live.start(refresh=True)
                continue

            if field == "Custom Instructions":
                live.stop()
                try:
                    with edit_feedback():
                        edit_langcode_md(state["project_dir"])
                except Exception as exc:
                    console.print(Panel.fit(Text(f"Failed to edit custom instructions: {exc}", style="bold red"), border_style="red"))
                    console.input("Press Enter to continue...")
                live.start(refresh=True)
                continue

            if field == "MCP Config":
                live.stop()
                try:
                    with edit_feedback():
                        edit_mcp_json(state["project_dir"])
                except Exception as exc:
                    console.print(Panel.fit(Text(f"Failed to edit MCP config: {exc}", style="bold red"), border_style="red"))
                    console.input("Press Enter to continue...")
                live.start(refresh=True)
                continue

            if field == "Tests":
                if state["command"] in ("feature", "fix"):
                    live.stop()
                    cmd = console.input("[bold]Test command[/bold] (e.g. pytest -q) – empty to clear: ").strip()
                    state["test_cmd"] = cmd or None
                    live.start(refresh=True)
                continue

            if field == "Start":
                if state["llm"] == "ollama" and not list_ollama_models():
                    live.stop()
                    console.print(Panel.fit(Text("Cannot start: no Ollama models installed.", style="bold red"), border_style="red"))
                    console.input("Press Enter to continue...")
                    live.start(refresh=True)
                    continue
                return state

