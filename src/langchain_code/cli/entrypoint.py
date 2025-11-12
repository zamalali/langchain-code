from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import typer

from .commands.chat import chat
from .commands.flows import feature, fix, analyze
from .commands.system import wrap, shell, doctor
from .commands.configure import env, edit_instructions
from ..cli_components.app import app
from ..cli_components.launcher import launcher_loop, default_state, list_ollama_models
from ..cli_components.state import console, set_selection_hub_active
from ..cli_components.env import bootstrap_env


def _unwrap_exc(e: BaseException) -> BaseException:
    """Drill down through ExceptionGroup/TaskGroup, __cause__, and __context__ to the root error."""
    seen = set()
    while True:
        inner = getattr(e, "exceptions", None)
        if inner:
            e = inner[0]
            continue
        if getattr(e, "__cause__", None) and e.__cause__ not in seen:
            seen.add(e)
            e = e.__cause__
            continue
        if getattr(e, "__context__", None) and e.__context__ not in seen:
            seen.add(e)
            e = e.__context__
            continue
        return e


def _friendly_agent_error(e: BaseException) -> str:
    root = _unwrap_exc(e)
    name = root.__class__.__name__
    msg = (str(root) or "").strip() or "(no details)"
    return "Sorry, a tool run failed. Please try again :)\n\n" f"| {name}: {msg}\n\n"


def _dispatch_from_state(chosen: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """Dispatch the chosen launcher state and report navigation plus status text."""
    if chosen.get("llm") == "ollama" and not list_ollama_models():
        return {"nav": "select", "info": "Cannot start: no Ollama models installed."}

    try:
        cmd = chosen["command"]
        if cmd == "chat":
            nav = chat(
                llm=chosen["llm"],
                project_dir=chosen["project_dir"],
                mode=chosen["engine"],
                auto=bool(chosen["autopilot"] and chosen["engine"] == "deep"),
                router=chosen["router"],
                priority=chosen["priority"],
                verbose=False,
            )
            return {"nav": nav, "info": None}

        if cmd == "feature":
            req = console.input("[bold]Feature request[/bold] (e.g. Add a dark mode toggle): ").strip()
            if not req:
                return {"nav": "select", "info": "Feature request aborted (empty input)."}
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
            return {"nav": "select", "info": "Feature workflow completed."}

        if cmd == "fix":
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
            return {"nav": "select", "info": "Fix workflow completed."}

        req = console.input("[bold]Analysis question[/bold] (e.g. What are the main components?): ").strip()
        if not req:
            return {"nav": "select", "info": "Analysis aborted (empty question)."}
        analyze(
            request=req,
            llm=chosen["llm"],
            project_dir=chosen["project_dir"],
            router=chosen["router"],
            priority=chosen["priority"],
            verbose=False,
        )
        return {"nav": "select", "info": "Analysis results provided."}

    except RuntimeError as exc:
        return {"nav": "select", "info": str(exc)}
    except Exception as exc:
        return {"nav": "select", "info": _friendly_agent_error(exc)}


def selection_hub(initial_state: Optional[Dict[str, Any]] = None) -> None:
    """Persistent launcher loop so users can switch modes without restarting the CLI."""
    state = dict(initial_state or default_state())

    try:
        bootstrap_env(state["project_dir"], interactive_prompt_if_missing=True)
    except Exception:
        pass

    set_selection_hub_active(True)
    try:
        while True:
            chosen = launcher_loop(state)
            if not chosen:
                return
            state.update(chosen)
            result = _dispatch_from_state(chosen)
            nav = result.get("nav") if result else None
            info = result.get("info") if result else None
            if info:
                state["_status"] = info
            elif "_status" in state:
                state.pop("_status", None)
            if nav == "quit":
                console.print("\n[bold]Goodbye![/bold]")
                return
            if nav == "select":
                continue
    finally:
        set_selection_hub_active(False)


# Typer command registration -------------------------------------------------
app.command(help="Run a command inside a PTY and capture output to a session log (used by fix --from-tty).")(wrap)
app.command(help="Open a logged subshell. Anything you run here is captured for fix --from-tty.")(shell)
app.command(help="Run environment checks for providers, tools, and MCP.")(doctor)
app.command(
    help="Open an interactive chat with the agent. Modes: react | deep (default: react). Use --auto in deep mode for full autopilot (plan+act with no questions)."
)(chat)
app.command(
    help="Implement a feature end-to-end (plan → search → edit → verify). Supports --apply and optional --test-cmd (e.g., 'pytest -q')."
)(feature)
app.command(help="Diagnose & fix a bug (trace → pinpoint → patch → test). Accepts --log, --test-cmd, and supports --apply.")(fix)
app.command(help="Analyze any codebase and generate insights (deep agent).")(analyze)
app.command(help="Edit environment. Use --global to edit your global env (~/.config/langcode/.env).")(env)
app.command(name="instr", help="Open or create project-specific instructions (.langcode/langcode.md) in your editor.")(edit_instructions)


@app.callback(invoke_without_command=True)
def _default_entry(ctx: typer.Context):
    """Launch the selection hub when no explicit subcommand is provided."""
    if ctx.invoked_subcommand:
        return
    selection_hub()


def main() -> None:
    app()


__all__ = ["selection_hub", "main"]
