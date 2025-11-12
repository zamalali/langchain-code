from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import typer
from rich import box
from rich.live import Live
from rich.panel import Panel
from rich.table import Table as _Table
from rich.text import Text
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.callbacks import BaseCallbackHandler

from ...cli_components import state as cli_state
from ...cli_components.state import console, in_selection_hub
from ...cli_components.launcher import help_content
from ...cli_components.display import (
    print_session_header,
    panel_agent_output,
    panel_router_choice,
    show_loader,
    pause_if_in_launcher,
    normalize_chat_history_for_anthropic,
)
from ...cli_components.todos import render_todos_panel, diff_todos, _coerce_sequential_todos
from ...cli_components.runtime import (
    maybe_coerce_img_command,
    extract_last_content,
    thread_id_for,
)
from ...cli_components.agents import (
    agent_cache_get,
    agent_cache_put,
    resolve_provider,
    build_react_agent_with_optional_llm,
    build_deep_agent_with_optional_llm,
)
from ...cli_components.env import bootstrap_env
from ...workflows.auto import AUTO_DEEP_INSTR
from ...config_core import get_model, get_model_info, get_model_by_name
from ...cli_components.constants import PROMPT


_AGENT_EXECUTOR = ThreadPoolExecutor(max_workers=2)


def _todos_to_text_summary(todos: List[dict]) -> str:
    seq = _coerce_sequential_todos(todos)
    icon = {"pending": "○", "in_progress": "◔", "completed": "✓"}
    lines: List[str] = []
    for idx, item in enumerate(seq, 1):
        status = (item.get("status") or "pending").lower().replace("-", "_")
        mark = icon.get(status, "-")
        content = (item.get("content") or "").strip() or "(empty)"
        lines.append(f"{idx}. {mark} {content} [{status.replace('_', ' ')}]")
    header = "Agent steps:"
    return header + ("\n" + "\n".join(lines) if lines else "\n(no steps recorded)")


class TodoLiveMinimal(BaseCallbackHandler):
    """Updates the single TODO table in-place; no extra logs."""

    def __init__(self, live: Live):
        self.live = live
        self.todos: list[dict] = []
        self.seen = False

    def _extract_todos(self, payload) -> Optional[list]:
        import re as _re
        import ast as _ast
        import json as _json

        if isinstance(payload, dict):
            if isinstance(payload.get("todos"), list):
                return payload["todos"]
            upd = payload.get("update")
            if isinstance(upd, dict) and isinstance(upd.get("todos"), list):
                return upd["todos"]

        upd = getattr(payload, "update", None)
        if isinstance(upd, dict) and isinstance(upd.get("todos"), list):
            return upd["todos"]

        s = str(payload)
        m = _re.search(r"Command\([^)]*update=(\{.*\})\)?$", s, _re.S) or _re.search(r"update=(\{.*\})", s, _re.S)
        if m:
            try:
                data = _ast.literal_eval(m.group(1))
                if isinstance(data, dict) and isinstance(data.get("todos"), list):
                    return data["todos"]
            except Exception:
                pass
        jm = _re.search(r"(\{.*\"todos\"\s*:\s*\[.*\].*\})", s, _re.S)
        if jm:
            try:
                data = _json.loads(jm.group(1))
                if isinstance(data.get("todos"), list):
                    return data["todos"]
            except Exception:
                pass
        return None

    def _render(self, todos: list[dict]):
        todos = _coerce_sequential_todos(todos)
        self.todos = todos
        self.seen = True
        self.live.update(render_todos_panel(todos))

    def on_tool_end(self, output, **kwargs):
        t = self._extract_todos(output)
        if t is not None:
            self._render(t)

    def on_chain_end(self, outputs, **kwargs):
        t = self._extract_todos(outputs)
        if t is not None:
            self._render(t)


def chat(
    message: Optional[List[str]] = typer.Argument(None, help="Optional initial message to send (quotes not required)."),
    llm: Optional[str] = typer.Option(None, "--llm", help="anthropic | gemini | openai | ollama"),
    project_dir: Path = typer.Option(Path.cwd(), "--project-dir", exists=True, file_okay=False),
    mode: str = typer.Option("react", "--mode", help="react | deep"),
    auto: bool = typer.Option(False, "--auto", help="Autonomy mode: plan+act with no questions (deep mode only)."),
    router: bool = typer.Option(False, "--router", help="Auto-route to the most efficient LLM per query."),
    priority: str = typer.Option("balanced", "--priority", help="Router priority: balanced | cost | speed | quality"),
    verbose: bool = typer.Option(False, "--verbose", help="Show model selection panels (and deep logs)."),
    inline: bool = typer.Option(False, "--inline", help="Inline single-turn output (no banners/clear)."),
) -> Optional[str]:
    """
    Chat interface shared between standalone CLI invocation and the launcher.

    Returns:
      - "quit": user explicitly exited chat; caller should terminate program
      - "select": user requested to return to launcher
      - None: normal return (caller may continue)
    """
    bootstrap_env(project_dir, interactive_prompt_if_missing=True)

    priority = (priority or "balanced").lower()
    if priority not in {"balanced", "cost", "speed", "quality"}:
        priority = "balanced"

    provider = resolve_provider(llm, router)
    mode = (mode or "react").lower()
    if mode not in {"react", "deep"}:
        mode = "react"

    input_queue: List[str] = []
    if isinstance(message, list):
        first_msg = " ".join(message).strip()
        if first_msg:
            input_queue.append(first_msg)

    if inline and input_queue:
        first = input_queue.pop(0)
        coerced = maybe_coerce_img_command(first)
        use_loader = not (mode == "react" and verbose)
        cm = show_loader() if use_loader else nullcontext()
        with cm:
            model_info = None
            chosen_llm = None
            if router:
                model_info = get_model_info(provider, coerced, priority)
                chosen_llm = get_model(provider, coerced, priority)
                model_key = model_info.get("langchain_model_name") if model_info else "default"
            else:
                env_override = os.getenv("LANGCODE_MODEL_OVERRIDE")
                chosen_llm = get_model_by_name(provider, env_override) if env_override else get_model(provider)
                model_key = env_override or "default"

            cache_key = (
                "deep" if mode == "deep" else "react",
                provider,
                model_key,
                str(project_dir.resolve()),
                bool(auto) if mode == "deep" else False,
            )
            agent = agent_cache_get(cache_key)
            if agent is None:
                if mode == "deep":
                    seed = AUTO_DEEP_INSTR if auto else None
                    agent = build_deep_agent_with_optional_llm(
                        provider=provider, project_dir=project_dir, llm=chosen_llm, instruction_seed=seed, apply=auto
                    )
                else:
                    agent = build_react_agent_with_optional_llm(
                        provider=provider, project_dir=project_dir, llm=chosen_llm
                    )
                agent_cache_put(cache_key, agent)
            try:
                if mode == "deep":
                    res = agent.invoke(
                        {"messages": [{"role": "user", "content": coerced}]},
                        config={
                            "recursion_limit": 30 if priority in {"speed", "cost"} else 45,
                            "configurable": {"thread_id": thread_id_for(project_dir, "chat-inline")},
                        },
                    )
                    output = (
                        extract_last_content(res.get("messages", [])).strip()
                        if isinstance(res, dict) and "messages" in res
                        else str(res)
                    )
                else:
                    payload = {"input": coerced, "chat_history": []}
                    if provider == "anthropic":
                        payload["chat_history"] = normalize_chat_history_for_anthropic([])
                    if verbose:
                        from ...cli_components.runtime import RichDeepLogs

                        res = agent.invoke(payload, config={"callbacks": [RichDeepLogs(console)]})
                    else:
                        res = agent.invoke(payload)
                    output = res.get("output", "") if isinstance(res, dict) else str(res)
                    if provider == "anthropic":
                        from ...cli_components.display import to_text as _to_text

                        output = _to_text(output)
                output = (output or "").strip() or "No response generated."
            except Exception as e:  # pragma: no cover - defensive
                from ...cli_components.runtime import _friendly_agent_error as friendly_error  # type: ignore

                output = friendly_error(e)
        console.print(output)
        return None

    session_title = "LangChain Code Agent | Deep Chat" if mode == "deep" else "LangChain Code Agent | Chat"
    if mode == "deep" and auto:
        session_title += " (Auto)"
    print_session_header(
        session_title,
        provider,
        project_dir,
        interactive=True,
        router_enabled=router,
        deep_mode=(mode == "deep"),
        command_name="chat",
    )

    history: list = []
    msgs: list = []
    last_todos: list = []
    last_files: dict = {}
    user_turns = 0
    ai_turns = 0
    deep_thread_id = thread_id_for(project_dir, "chat")
    db_path = project_dir / ".langcode" / "memory.sqlite"

    static_agent = None
    static_agent_future = None
    if not router:
        from ...cli import _AGENT_EXECUTOR  # type: ignore

        def _build_static_agent() -> Any:
            env_override = os.getenv("LANGCODE_MODEL_OVERRIDE")
            chosen_llm = get_model_by_name(provider, env_override) if env_override else get_model(provider)
            if mode == "deep":
                seed = AUTO_DEEP_INSTR if auto else None
                return build_deep_agent_with_optional_llm(
                    provider=provider, project_dir=project_dir, llm=chosen_llm, instruction_seed=seed, apply=auto
                )
            return build_react_agent_with_optional_llm(
                provider=provider, project_dir=project_dir, llm=chosen_llm
            )

        static_agent_future = _AGENT_EXECUTOR.submit(_build_static_agent)

    prio_limits = {"speed": 30, "cost": 35, "balanced": 45, "quality": 60}

    try:
        while True:
            if input_queue:
                user = input_queue.pop(0)
            else:
                user = console.input(PROMPT).strip()
            if not user:
                continue

            low = user.lower()
            if low in {"cls", "clear", "/clear"}:
                print_session_header(
                    session_title,
                    provider,
                    project_dir,
                    interactive=True,
                    router_enabled=router,
                    deep_mode=(mode == "deep"),
                    command_name="chat",
                )
                history.clear()
                msgs.clear()
                last_todos = []
                last_files = {}
                continue

            if low in {"select", "/select", "/menu", ":menu"}:
                console.print("[cyan]Returning to launcher...[/cyan]")
                if in_selection_hub():
                    return "select"
                from ...cli.entrypoint import selection_hub  # local import to avoid cycles

                selection_hub(
                    {
                        "command": "chat",
                        "engine": mode,
                        "router": router,
                        "priority": priority,
                        "autopilot": bool(auto),
                        "apply": False,
                        "llm": llm,
                        "project_dir": project_dir,
                        "test_cmd": None,
                    }
                )
                return "quit"

            if low in {"exit", "quit", ":q", "/exit", "/quit"}:
                return "quit"

            if low in {"help", "/help", ":help"}:
                print_session_header(
                    session_title,
                    provider,
                    project_dir,
                    interactive=True,
                    router_enabled=router,
                    deep_mode=(mode == "deep"),
                    command_name="chat",
                )
                console.print(help_content())
                continue

            if low in {"/memory", "/stats"}:
                if mode != "deep":
                    console.print(
                        Panel.fit(
                            Text("Memory & stats are available in deep mode only.", style="yellow"),
                            border_style="yellow",
                        )
                    )
                    continue
                if low == "/memory":
                    t = _Table.grid(padding=(0, 2))
                    t.add_row(Text("Thread", style="bold"), Text(deep_thread_id))
                    t.add_row(Text("DB", style="bold"), Text(str(db_path)))
                    t.add_row(
                        Text("Todos", style="bold"),
                        Text(
                            ", ".join(
                                f"[{i+1}] {it.get('content','')}: {it.get('status','pending')}"
                                for i, it in enumerate(last_todos)
                            )
                            or "(none)"
                        ),
                    )
                    t.add_row(Text("Files", style="bold"), Text(", ".join(sorted(last_files.keys())) or "(none)"))
                    console.print(Panel(t, title="/memory", border_style="cyan", box=box.ROUNDED))
                else:
                    t = _Table.grid(padding=(0, 2))
                    t.add_row(Text("User turns", style="bold"), Text(str(user_turns)))
                    t.add_row(Text("Agent turns", style="bold"), Text(str(ai_turns)))
                    t.add_row(Text("Messages (current buffer)", style="bold"), Text(str(len(msgs))))
                    t.add_row(Text("Routing", style="bold"), Text(("on | priority=" + priority) if router else "off"))
                    t.add_row(Text("Checkpointer", style="bold"), Text(str(db_path)))
                    t.add_row(Text("Thread", style="bold"), Text(deep_thread_id))
                    console.print(Panel(t, title="/stats", border_style="cyan", box=box.ROUNDED))
                continue

            coerced = maybe_coerce_img_command(user)
            user_turns += 1

            pending_router_panel: Optional[Panel] = None
            pending_output_panel: Optional[Panel] = None
            react_history_update: Optional[Tuple[HumanMessage, AIMessage]] = None

            agent = static_agent
            model_info = None
            chosen_llm = None

            loader_cm = show_loader() if (router and not verbose) else nullcontext()
            with loader_cm:
                if router:
                    provider = resolve_provider(llm, router=True)
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
                    cached = agent_cache_get(cache_key)
                    if cached is not None:
                        agent = cached
                    else:
                        if verbose and model_info:
                            pending_router_panel = panel_router_choice(model_info)
                        if mode == "deep":
                            seed = AUTO_DEEP_INSTR if auto else None
                            agent = build_deep_agent_with_optional_llm(
                                provider=provider,
                                project_dir=project_dir,
                                llm=chosen_llm,
                                instruction_seed=seed,
                                apply=auto,
                            )
                        else:
                            agent = build_react_agent_with_optional_llm(
                                provider=provider,
                                project_dir=project_dir,
                                llm=chosen_llm,
                            )
                        agent_cache_put(cache_key, agent)

                else:
                    if agent is None:
                        if static_agent_future is not None:
                            future_done = static_agent_future.done()
                            build_cm = nullcontext() if (verbose or future_done) else show_loader()
                            try:
                                with build_cm:
                                    agent = static_agent_future.result()
                            except Exception:
                                agent = None
                            static_agent_future = None
                        if agent is None:
                            build_cm = nullcontext() if verbose else show_loader()
                            with build_cm:
                                env_override = os.getenv("LANGCODE_MODEL_OVERRIDE")
                                chosen_llm = get_model_by_name(provider, env_override) if env_override else get_model(provider)
                                if mode == "deep":
                                    seed = AUTO_DEEP_INSTR if auto else None
                                    agent = build_deep_agent_with_optional_llm(
                                        provider=provider,
                                        project_dir=project_dir,
                                        llm=chosen_llm,
                                        instruction_seed=seed,
                                        apply=auto,
                                    )
                                else:
                                    agent = build_react_agent_with_optional_llm(
                                        provider=provider,
                                        project_dir=project_dir,
                                        llm=chosen_llm,
                                    )
                    static_agent = agent

            if pending_router_panel:
                console.print(pending_router_panel)

            def _current_model_label() -> Optional[str]:
                if router:
                    if model_info and model_info.get("langchain_model_name"):
                        return model_info["langchain_model_name"]
                    return None
                env_override = os.getenv("LANGCODE_MODEL_OVERRIDE")
                if env_override:
                    return env_override
                try:
                    return get_model_info(provider).get("langchain_model_name")
                except Exception:
                    return None

            _model_label = _current_model_label()

            if mode == "react":
                try:
                    payload = {"input": coerced, "chat_history": history}
                    if provider == "anthropic":
                        payload["chat_history"] = normalize_chat_history_for_anthropic(payload["chat_history"])

                    if verbose:
                        from ...cli_components.runtime import RichDeepLogs

                        res = agent.invoke(payload, config={"callbacks": [RichDeepLogs(console)]})
                    else:
                        with show_loader():
                            res = agent.invoke(payload)

                    output = res.get("output", "") if isinstance(res, dict) else str(res)
                    if provider == "anthropic":
                        from ...cli_components.display import to_text as _to_text

                        output = _to_text(output)
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
                except Exception as e:
                    from ...cli_components.runtime import _friendly_agent_error as friendly_error  # type: ignore

                    output = friendly_error(e)

                pending_output_panel = panel_agent_output(output, model_label=_model_label)
                react_history_update = (HumanMessage(content=coerced), AIMessage(content=output))
                ai_turns += 1
            else:
                msgs.append({"role": "user", "content": coerced})
                if auto:
                    msgs.append(
                        {
                            "role": "system",
                            "content": (
                                "AUTOPILOT: Start now. Discover files (glob/list_dir/grep), read targets (read_file), "
                                "perform edits (edit_by_diff/write_file), and run at least one run_cmd (git/tests) "
                                "capturing stdout/stderr + exit code. Then produce one 'FINAL:' report and STOP. No questions."
                            ),
                        }
                    )

                deep_config: Dict[str, Any] = {
                    "recursion_limit": prio_limits.get(priority, 100),
                    "configurable": {"thread_id": deep_thread_id},
                }

                placeholder = Panel(
                    Text("Planning tasks...", style="dim"),
                    title="TODOs",
                    border_style="blue",
                    box=box.ROUNDED,
                    padding=(1, 1),
                    expand=True,
                )

                output: str = ""

                with Live(placeholder, refresh_per_second=8, transient=False) as live:
                    cli_state.current_live = live
                    todo_cb = TodoLiveMinimal(live)
                    deep_config["callbacks"] = [todo_cb]
                    res = {}
                    try:
                        res = agent.invoke({"messages": msgs}, config=deep_config)

                        if isinstance(res, dict) and "messages" in res:
                            msgs = res["messages"]
                            last_files = res.get("files") or last_files
                            last_content = extract_last_content(msgs).strip()
                        else:
                            last_content = ""
                            res = res if isinstance(res, dict) else {}

                    except Exception as e:
                        last_content = ""
                        if "recursion" in str(e).lower():
                            output = f"Agent hit recursion limit. Last response: {extract_last_content(msgs)}"
                        else:
                            output = f"Agent error: {e}"
                        res = {}
                    if not output:
                        if not last_content:
                            msgs.append(
                                {
                                    "role": "system",
                                    "content": "You must provide a response. Use your tools to complete the request and give a clear answer.",
                                }
                            )
                            try:
                                res2 = agent.invoke({"messages": msgs}, config=deep_config)
                                if isinstance(res2, dict) and "messages" in res2:
                                    msgs = res2["messages"]
                                last_content = extract_last_content(msgs).strip()
                            except Exception as e:
                                last_content = f"Agent failed after retry: {e}"
                        output = last_content or "No response generated."

                    final_todos = res.get("todos") if isinstance(res, dict) else None
                    if not isinstance(final_todos, list) or not final_todos:
                        final_todos = getattr(todo_cb, "todos", [])

                    if final_todos:
                        normalized_todos = _coerce_sequential_todos(final_todos)
                        animated_final = [{**todo, "status": todo.get("status", "pending")} for todo in normalized_todos]
                        any_completed = any(todo.get("status") == "completed" for todo in animated_final)
                        if not any_completed:
                            import time

                            for idx in range(len(animated_final)):
                                step_view: List[dict] = []
                                for j, todo in enumerate(animated_final):
                                    status = todo.get("status", "pending")
                                    if j < idx:
                                        status = "completed"
                                    elif j == idx:
                                        status = "in_progress" if status != "completed" else status
                                    step_view.append({**todo, "status": status})
                                live.update(render_todos_panel(step_view))
                                time.sleep(0.15)

                        completed_view = [{**todo, "status": "completed"} for todo in normalized_todos]

                        live.update(render_todos_panel(completed_view))
                        last_todos = completed_view

                        if not output.strip():
                            output = _todos_to_text_summary(completed_view)
                    else:
                        live.update(
                            Panel(
                                Text("No tasks were emitted by the agent.", style="dim"),
                                title="TODOs",
                                border_style="blue",
                                box=box.ROUNDED,
                                padding=(1, 1),
                                expand=True,
                            )
                        )

                cli_state.current_live = None

                pending_output_panel = panel_agent_output(output, model_label=_model_label)
                ai_turns += 1

            if pending_output_panel:
                console.print(pending_output_panel)

            if react_history_update:
                human_msg, ai_msg = react_history_update
                history.append(human_msg)
                history.append(ai_msg)
                if len(history) > 20:
                    history[:] = history[-20:]

    except (KeyboardInterrupt, EOFError):
        return "quit"


__all__ = ["chat"]
