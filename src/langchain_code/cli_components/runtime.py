from __future__ import annotations

import ast
import builtins
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any, List, Optional

from langchain_core.callbacks import BaseCallbackHandler
from rich import box
from rich.panel import Panel
from rich.text import Text

from . import state
from .todos import diff_todos, render_todos_panel, short


class InputPatch:
    def __init__(self, title: str = "Consent"):
        self.title = title
        self._orig_input = None

    def __enter__(self):
        self._orig_input = builtins.input

        def _rich_input(prompt: str = "") -> str:
            live = state.current_live
            cm = live.pause() if getattr(live, "pause", None) else nullcontext()
            with cm:
                body = Text()
                msg = prompt.strip() or "Action requires your confirmation."
                body.append(msg + "\n\n")
                body.append("Type ", style="dim")
                body.append("Y", style="bold green")
                body.append("/", style="dim")
                body.append("N", style="bold red")
                body.append(" and press Enter.", style="dim")

                panel = Panel(
                    body,
                    title=self.title,
                    border_style="yellow",
                    box=box.ROUNDED,
                    padding=(1, 2),
                )
                state.console.print(panel)

                answer = console.input("[bold yellow]›[/bold yellow] ").strip()
                return answer

        builtins.input = _rich_input
        return self

    def __exit__(self, exc_type, exc, tb):
        builtins.input = self._orig_input
        return False


class TodoLive(BaseCallbackHandler):
    """Stream TODO updates when planner tools return Command(update={'todos': ...})."""

    def __init__(self):
        self.prev: List[dict] = []

    def on_tool_end(self, output, **kwargs):
        text = str(output)
        if "Command(update=" not in text or "'todos':" not in text:
            return
        match = re.search(r"Command\(update=(\{.*\})\)$", text, re.S) or re.search(r"update=(\{.*\})", text, re.S)
        if not match:
            return
        try:
            data = ast.literal_eval(match.group(1))
            todos = data.get("todos")
            if not isinstance(todos, list):
                return
            changes = diff_todos(self.prev, todos)
            self.prev = todos
            if todos:
                state.console.print(render_todos_panel(todos))
            if changes:
                state.console.print(Panel(Text("\n".join(changes)), title="Progress", border_style="yellow", box=box.ROUNDED, expand=True))
        except Exception:
            pass


class RichDeepLogs(BaseCallbackHandler):
    """
    Minimal, pretty callback printer for deep (LangGraph) runs.
    Only logs the big milestones so it stays readable.
    Toggle by passing --verbose.
    """

    def on_chain_start(self, serialized, inputs, **kwargs):
        name = (serialized or {}).get("id") or (serialized or {}).get("name") or "chain"
        state.console.print(Panel.fit(Text.from_markup(f"▶ [bold]Start[/bold] {name}\n[dim]{short(str(inputs))}[/dim]"), border_style="cyan", title="Node", box=box.ROUNDED))

    def on_chain_end(self, outputs, **kwargs):
        state.console.print(Panel.fit(Text.from_markup(f"[bold]End[/bold]\n[dim]{short(str(outputs))}[/dim]"), border_style="cyan", title="Node", box=box.ROUNDED))

    def on_tool_start(self, serialized, tool_input, **kwargs):
        name = (serialized or {}).get("name") or "tool"
        state.console.print(Panel.fit(Text.from_markup(f"[bold]{name}[/bold]\n[dim]{short(str(tool_input))}[/dim]"), border_style="yellow", title="Tool", box=box.ROUNDED))

    def on_tool_end(self, output, **kwargs):
        state.console.print(Panel.fit(Text.from_markup(f" [bold]Tool result[/bold]\n{short(str(output))}"), border_style="yellow", title="Tool", box=box.ROUNDED))

    def on_llm_start(self, serialized, prompts, **kwargs):
        name = (serialized or {}).get("id") or (serialized or {}).get("name") or "llm"
        show = "\n---\n".join(short(p) for p in (prompts or [])[:1])
        state.console.print(Panel.fit(Text.from_markup(f"[bold]{name}[/bold]\n{show}"), border_style="green", title="LLM", box=box.ROUNDED))

    def on_llm_end(self, response, **kwargs):
        state.console.print(Panel.fit(Text("[dim]LLM complete[/dim]"), border_style="green", title="LLM", box=box.ROUNDED))


def maybe_coerce_img_command(raw: str) -> str:
    text = raw.strip()
    if not text.startswith("/img"):
        return raw
    try:
        rest = text[len("/img"):].strip()
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


def extract_last_content(messages: List) -> str:
    if not messages:
        return ""
    last = messages[-1]
    content = getattr(last, "content", None)

    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item and isinstance(item["text"], str):
                    parts.append(item["text"])
                elif item.get("type") == "text" and isinstance(item.get("data") or item.get("content"), str):
                    parts.append(item.get("data") or item.get("content"))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip()

    if isinstance(last, dict):
        content = last.get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return "\n".join(str(x) for x in content if isinstance(x, str)).strip()

    return (str(content) if content is not None else str(last)).strip()


def thread_id_for(project_dir: Path, purpose: str = "chat") -> str:
    """Stable thread id per project & purpose for LangGraph checkpointer."""
    return f"{purpose}@{project_dir.resolve()}"
