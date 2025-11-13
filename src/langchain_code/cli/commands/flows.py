from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import typer
from rich.panel import Panel
from rich.text import Text

from ...cli_components.display import (
    print_session_header,
    panel_agent_output,
    panel_router_choice,
    show_loader,
    pause_if_in_launcher,
)
from ...cli_components.env import bootstrap_env
from ...cli_components.state import console
from ...cli_components.env import tail_bytes, extract_error_block, tty_log_path
from ...cli_components.runtime import extract_last_content, thread_id_for
from ...cli_components.agents import (
    agent_cache_get,
    agent_cache_put,
    resolve_provider,
    build_react_agent_with_optional_llm,
    build_deep_agent_with_optional_llm,
)
from ...config_core import get_model, get_model_info
from ...workflows.feature_impl import FEATURE_INSTR
from ...workflows.bug_fix import BUGFIX_INSTR
from ..constants_runtime import (
    FEATURE_SESSION_TITLE,
    FIX_SESSION_TITLE,
    ANALYZE_SESSION_TITLE,
    FIX_FALLBACK_PROMPT,
)


def feature(
    request: str = typer.Argument(..., help='e.g. "Add a dark mode toggle in settings"'),
    llm: Optional[str] = typer.Option(None, "--llm", help="anthropic | gemini | openai | ollama"),
    project_dir: Path = typer.Option(Path.cwd(), "--project-dir", exists=True, file_okay=False),
    test_cmd: Optional[str] = typer.Option(None, "--test-cmd", help='e.g. "pytest -q" or "npm test"'),
    apply: bool = typer.Option(False, "--apply", help="Apply writes and run commands without interactive confirm."),
    router: bool = typer.Option(False, "--router", help="Auto-route to the most efficient LLM for this request."),
    priority: str = typer.Option("balanced", "--priority", help="balanced | cost | speed | quality"),
    verbose: bool = typer.Option(False, "--verbose", help="Show model selection panel."),
):
    bootstrap_env(project_dir, interactive_prompt_if_missing=True)

    priority = (priority or "balanced").lower()
    if priority not in {"balanced", "cost", "speed", "quality"}:
        priority = "balanced"

    provider = resolve_provider(llm, router)
    model_info = None
    chosen_llm = None

    if router:
        model_info = get_model_info(provider, request, priority)
        chosen_llm = get_model(provider, request, priority)

    print_session_header(
        FEATURE_SESSION_TITLE,
        provider,
        project_dir,
        interactive=False,
        apply=apply,
        test_cmd=test_cmd,
        model_info=(model_info if (router and verbose) else None),
        router_enabled=router,
    )
    if router and verbose and model_info:
        console.print(panel_router_choice(model_info))

    model_key = (model_info or {}).get("langchain_model_name", "default")
    cache_key = ("react", provider, model_key, str(project_dir.resolve()), False)
    cached = agent_cache_get(cache_key)
    if not router and provider in {"openai", "ollama"}:
        chosen_llm = get_model(provider)
    if cached is None:
        agent = build_react_agent_with_optional_llm(
            provider=provider,
            project_dir=project_dir,
            llm=chosen_llm,
            apply=apply,
            test_cmd=test_cmd,
            instruction_seed=FEATURE_INSTR,
        )
        agent_cache_put(cache_key, agent)
    else:
        agent = cached

    with show_loader():
        res = agent.invoke({"input": request, "chat_history": []})
        output = res.get("output", "") if isinstance(res, dict) else str(res)
    console.print(panel_agent_output(output, title="Feature Result"))
    pause_if_in_launcher()


def fix(
    request: Optional[str] = typer.Argument(None, help='e.g. "Fix crash on image upload"'),
    log: Optional[Path] = typer.Option(None, "--log", exists=True, help="Path to error log or stack trace."),
    llm: Optional[str] = typer.Option(None, "--llm", help="anthropic | gemini | openai | ollama"),
    project_dir: Path = typer.Option(Path.cwd(), "--project-dir", exists=True, file_okay=False),
    test_cmd: Optional[str] = typer.Option(None, "--test-cmd", help='e.g. "pytest -q"'),
    apply: bool = typer.Option(False, "--apply", help="Apply writes and run commands without interactive confirm."),
    router: bool = typer.Option(False, "--router", help="Auto-route to the most efficient LLM for this request."),
    priority: str = typer.Option("balanced", "--priority", help="balanced | cost | speed | quality"),
    verbose: bool = typer.Option(False, "--verbose", help="Show model selection panel."),
    from_tty: bool = typer.Option(
        False,
        "--from-tty",
        help="Use most recent output from the current logged terminal session (run your command via `langcode wrap ...` or `langcode shell`).",
    ),
    tty_id: Optional[str] = typer.Option(None, "--tty-id", help="Which session to read; defaults to current TTY."),
):
    bootstrap_env(project_dir, interactive_prompt_if_missing=True)

    priority = (priority or "balanced").lower()
    if priority not in {"balanced", "cost", "speed", "quality"}:
        priority = "balanced"

    provider = resolve_provider(llm, router)

    bug_input = (request or "").strip()
    if log:
        bug_input += "\n\n--- ERROR LOG ---\n" + Path(log).read_text(encoding="utf-8", errors="ignore")
    elif from_tty:
        tlog = os.environ.get("LANGCODE_TTY_LOG") or str(tty_log_path(tty_id))
        p = Path(tlog)
        if p.exists():
            recent = tail_bytes(p)
            block = extract_error_block(recent).strip()
            if block:
                bug_input += "\n\n--- ERROR LOG (from TTY) ---\n" + block
                console.print(Panel.fit(Text(f"Using error from session log: {p}", style="dim"), border_style="cyan"))
        else:
            console.print(
                Panel.fit(
                    Text("No TTY session log found. Run your failing command via `langcode wrap <cmd>` or `langcode shell`.", style="yellow"),
                    border_style="yellow",
                )
            )
    bug_input = bug_input.strip() or FIX_FALLBACK_PROMPT

    model_info = None
    chosen_llm = None
    if router:
        model_info = get_model_info(provider, bug_input, priority)
        chosen_llm = get_model(provider, bug_input, priority)

    print_session_header(
        FIX_SESSION_TITLE,
        provider,
        project_dir,
        interactive=False,
        apply=apply,
        test_cmd=test_cmd,
        model_info=(model_info if (router and verbose) else None),
        router_enabled=router,
    )
    if router and verbose and model_info:
        console.print(panel_router_choice(model_info))

    model_key = (model_info or {}).get("langchain_model_name", "default")
    cache_key = ("react", provider, model_key, str(project_dir.resolve()), False)
    cached = agent_cache_get(cache_key)
    if not router and provider in {"openai", "ollama"}:
        chosen_llm = get_model(provider)
    if cached is None:
        agent = build_react_agent_with_optional_llm(
            provider=provider,
            project_dir=project_dir,
            llm=chosen_llm,
            apply=apply,
            test_cmd=test_cmd,
            instruction_seed=BUGFIX_INSTR,
        )
        agent_cache_put(cache_key, agent)
    else:
        agent = cached

    with show_loader():
        res = agent.invoke({"input": bug_input, "chat_history": []})
        output = res.get("output", "") if isinstance(res, dict) else str(res)
    console.print(panel_agent_output(output, title="Fix Result"))
    pause_if_in_launcher()


def analyze(
    request: str = typer.Argument(..., help='e.g. "What are the main components of this project?"'),
    llm: Optional[str] = typer.Option(None, "--llm", help="anthropic | gemini | openai | ollama"),
    project_dir: Path = typer.Option(Path.cwd(), "--project-dir", exists=True, file_okay=False),
    router: bool = typer.Option(False, "--router", help="Auto-route to the most efficient LLM for this request."),
    priority: str = typer.Option("balanced", "--priority", help="balanced | cost | speed | quality"),
    verbose: bool = typer.Option(False, "--verbose", help="Show model selection panel."),
):
    bootstrap_env(project_dir, interactive_prompt_if_missing=True)

    priority = (priority or "balanced").lower()
    if priority not in {"balanced", "cost", "speed", "quality"}:
        priority = "balanced"

    provider = resolve_provider(llm, router)

    model_info = None
    chosen_llm = None
    if router:
        model_info = get_model_info(provider, request, priority)
        chosen_llm = get_model(provider, request, priority)

    print_session_header(
        ANALYZE_SESSION_TITLE,
        provider,
        project_dir,
        interactive=False,
        apply=False,
        model_info=(model_info if (router and verbose) else None),
        router_enabled=router,
    )
    if router and verbose and model_info:
        console.print(panel_router_choice(model_info))

    model_key = (model_info or {}).get("langchain_model_name", "default")
    cache_key = ("deep", provider, model_key, str(project_dir.resolve()), False)
    cached = agent_cache_get(cache_key)
    if not router and provider in {"openai", "ollama"}:
        chosen_llm = get_model(provider)
    if cached is None:
        agent = build_deep_agent_with_optional_llm(
            provider=provider,
            project_dir=project_dir,
            llm=chosen_llm,
            apply=False,
        )
        agent_cache_put(cache_key, agent)
    else:
        agent = cached

    with show_loader():
        output = ""
        try:
            res = agent.invoke(
                {"messages": [{"role": "user", "content": request}]},
                config={
                    "recursion_limit": 45,
                    "configurable": {"thread_id": thread_id_for(project_dir, "analyze")},
                },
            )
            output = (
                extract_last_content(res.get("messages", [])).strip()
                if isinstance(res, dict) and "messages" in res
                else str(res)
            )
        except Exception as e:
            output = f"Analyze error: {e}"
    console.print(panel_agent_output(output or "No response generated.", title="Analysis Result"))
    pause_if_in_launcher()


__all__ = ["feature", "fix", "analyze"]

