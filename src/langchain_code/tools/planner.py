# src/langchain_code/tools/planner.py
from __future__ import annotations
from typing import Any, Annotated, Dict, List
import re

from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from ..agent.state import DeepAgentState

WRITE_TODOS_DESCRIPTION = """Create or update a structured todo list.

Accepted input formats (normalized automatically):
- String with bullet lines:
  "- Plan\n- Search\n- Edit\n- Verify"
- List of strings:
  ["Plan the work", "Search repo", "Edit files", "Run tests"]
- List of objects:
  [{"content":"Plan", "status":"pending"}, {"content":"Search", "status":"in_progress"}]
- Object with 'items':
  {"items": ["Plan", {"content":"Search","status":"pending"}]}

Also understands GitHub-style checkboxes:
- [ ] pending
- [x] completed

Valid statuses: "pending", "in_progress", "completed".
If omitted or invalid, defaults to "pending".
Maintain at most ONE item with status "in_progress".
"""

_ALLOWED = {"pending", "in_progress", "completed"}
_ALIAS = {
    "in-progress": "in_progress",
    "progress": "in_progress",
    "doing": "in_progress",
    "todo": "pending",
    "tbd": "pending",
    "done": "completed",
    "complete": "completed",
    "finished": "completed",
}

def _coerce_status(s: str | None) -> str:
    if not s:
        return "pending"
    s = s.strip().lower()
    s = _ALIAS.get(s, s)
    return s if s in _ALLOWED else "pending"

_checkbox_re = re.compile(r"^\s*(?:[-*+]|\d+[.)])?\s*(\[[ xX]\])?\s*(.+)$")

def _normalize_one(item: Any) -> Dict[str, str] | None:
    if item is None:
        return None

    if isinstance(item, str):
        # Handle "- [x] Do thing" / "* [ ] Task" / "1) [x] Task"
        m = _checkbox_re.match(item.strip())
        if m:
            box, rest = m.groups()
            content = (rest or "").strip().strip("-•*").strip()
            if not content:
                return None
            status = "completed" if (box and box.lower() == "[x]") else "pending"
            return {"content": content, "status": status}
        content = item.strip().lstrip("-•* ").strip()
        if not content:
            return None
        return {"content": content, "status": "pending"}

    if isinstance(item, dict):
        content = (
            item.get("content")
            or item.get("task")
            or item.get("title")
            or ""
        )
        content = content.strip()
        if not content:
            return None
        status = _coerce_status(item.get("status"))
        return {"content": content, "status": status}

    # Fallback to string coercion
    return _normalize_one(str(item))

def _normalize_list(raw: Any) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    if isinstance(raw, str):
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        for ln in lines:
            item = _normalize_one(ln)
            if item:
                out.append(item)
    elif isinstance(raw, list):
        for el in raw:
            item = _normalize_one(el)
            if item:
                out.append(item)
    elif isinstance(raw, dict) and "items" in raw:
        for el in raw["items"]:
            item = _normalize_one(el)
            if item:
                out.append(item)
    else:
        got = _normalize_one(raw)
        if got:
            out.append(got)
    # De-dup consecutive duplicates by content (compact)
    dedup: List[Dict[str, str]] = []
    last = None
    for it in out[:50]:
        if not last or it["content"] != last["content"]:
            dedup.append(it)
            last = it
    return dedup

def _enforce_single_in_progress(items: List[Dict[str, str]], prefer_index: int | None = None) -> None:
    # Keep at most one 'in_progress'. If multiple, keep the preferred index and downgrade others to 'pending'.
    ip_indices = [i for i, it in enumerate(items) if it.get("status") == "in_progress"]
    if len(ip_indices) <= 1:
        return
    keep = prefer_index if prefer_index is not None else ip_indices[0]
    for i in ip_indices:
        if i != keep:
            items[i]["status"] = "pending"

@tool(description=WRITE_TODOS_DESCRIPTION)
def write_todos(
    todos: Any = None,
    items: Any = None,
    value: Any = None,
    state: Annotated[DeepAgentState, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> Command:
    raw = None
    for candidate in (todos, items, value):
        if candidate is not None:
            raw = candidate
            break
    if raw is None:
        raw = []
    normalized = _normalize_list(raw)
    _enforce_single_in_progress(normalized, prefer_index=None)
    return Command(update={
        "todos": normalized,
        "messages": [ToolMessage(f"Updated todos ({len(normalized)} items).", tool_call_id=tool_call_id)],
    })

@tool(description="Append a new TODO (defaults to pending).")
def append_todo(
    content: str,
    status: str | None = None,
    state: Annotated[DeepAgentState, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> Command:
    content = (content or "").strip()
    if not content:
        return Command()  # no-op
    todos = state.get("todos", []) or []
    todos = list(todos)
    s = _coerce_status(status)
    todos.append({"content": content, "status": s})
    _enforce_single_in_progress(todos, prefer_index=len(todos) - 1 if s == "in_progress" else None)
    return Command(update={
        "todos": todos,
        "messages": [ToolMessage(f"Appended todo: {content}", tool_call_id=tool_call_id)],
    })

@tool(description="Update TODO status by index (0-based). Status = pending | in_progress | completed. Keeps only one in_progress.")
def update_todo_status(
    index: int,   
    status: str,
    state: Annotated[DeepAgentState, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> Command:
    try:
        idx = int(index)
    except Exception:
        idx = 0

    todos = list(state.get("todos", []) or [])
    if 0 <= idx < len(todos):
        todos[idx] = {**todos[idx], "status": _coerce_status(status)}
        _enforce_single_in_progress(todos, prefer_index=idx if todos[idx]["status"] == "in_progress" else None)
        msg = f"Set todo[{idx+1}] to {todos[idx]['status']}"
    else:
        msg = f"Ignored update_todo_status: index {idx} out of range (n={len(todos)})"
    return Command(update={
        "todos": todos,
        "messages": [ToolMessage(msg, tool_call_id=tool_call_id)],
    })

@tool(description="Clear all TODOs.")
def clear_todos(
    state: Annotated[DeepAgentState, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> Command:
    return Command(update={
        "todos": [],
        "messages": [ToolMessage("Cleared todos.", tool_call_id=tool_call_id)],
    })
