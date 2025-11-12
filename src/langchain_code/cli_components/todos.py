from __future__ import annotations

from typing import List, Optional

from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def _coerce_sequential_todos(todos: List[dict] | None) -> List[dict]:
    """Ensure visual progression is strictly sequential."""
    todos = list(todos or [])
    blocked = False
    out: List[dict] = []
    for item in todos:
        status = (item.get("status") or "pending").lower().replace("-", "_")
        if blocked and status in {"in_progress", "completed"}:
            status = "pending"
        if status != "completed":
            blocked = True
        out.append({**item, "status": status})
    return out


def render_todos_panel(todos: List[dict]) -> Panel:
    todos = _coerce_sequential_todos(todos)

    if not todos:
        return Panel(Text("No TODOs yet.", style="dim"), title="TODOs", border_style="blue", box=box.ROUNDED)
    table = Table.grid(padding=(0, 1))
    table.add_column(justify="right", width=3, no_wrap=True)
    table.add_column()

    ICON = {"pending": "○", "in_progress": "◔", "completed": "✓"}
    STYLE = {"pending": "dim", "in_progress": "yellow", "completed": "green"}

    for idx, item in enumerate(todos, 1):
        status = (item.get("status") or "pending").lower().replace("-", "_")
        status = status if status in ICON else "pending"
        content = (item.get("content") or "").strip() or "(empty)"
        style = STYLE[status]
        mark = ICON[status]
        text = Text(content, style=style)
        if status == "completed":
            text.stylize("strike")
        table.add_row(f"{idx}.", Text.assemble(Text(mark + " ", style=style), text))
    return Panel(table, title="TODOs", border_style="blue", box=box.ROUNDED, padding=(1, 1), expand=True)


def diff_todos(before: List[dict] | None, after: List[dict] | None) -> List[str]:
    before_list = _coerce_sequential_todos(before or [])
    after_list = _coerce_sequential_todos(after or [])
    changes: List[str] = []
    for idx in range(min(len(before_list), len(after_list))):
        prev = (before_list[idx].get("status") or "").lower()
        curr = (after_list[idx].get("status") or "").lower()
        if prev != curr:
            content = (after_list[idx].get("content") or before_list[idx].get("content") or "").strip()
            changes.append(f"[{idx + 1}] {content} -> {curr}")
    if len(after_list) > len(before_list):
        for j in range(len(before_list), len(after_list)):
            content = (after_list[j].get("content") or "").strip()
            changes.append(f"[+ ] {content} (added)")
    if len(before_list) > len(after_list):
        for j in range(len(after_list), len(before_list)):
            content = (before_list[j].get("content") or "").strip()
            changes.append(f"[- ] {content} (removed)")
    return changes


def complete_all_todos(todos: List[dict] | None) -> List[dict]:
    """
    Mark any non-completed TODOs as completed. Invoked right before rendering
    the final answer so the board reflects finished work the agent may have
    forgotten to mark as done.
    """
    todos = list(todos or [])
    out: List[dict] = []
    for item in todos:
        status = (item.get("status") or "pending").lower().replace("-", "_")
        if status != "completed":
            item = {**item, "status": "completed"}
        out.append(item)
    return out


def short(text: str, length: int = 280) -> str:
    text = text.replace("\r\n", "\n").strip()
    return text if len(text) <= length else text[:length] + " ..."

