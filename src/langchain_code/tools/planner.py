from __future__ import annotations
from typing import Any, Annotated, Dict, List
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from ..agent.state import DeepAgentState

WRITE_TODOS_DESCRIPTION = """Create or update a structured todo list.

Accepted input formats (the tool normalizes them automatically):
- A single string with bullet lines:
  "- Plan\n- Search\n- Edit\n- Verify"
- A list of strings:
  ["Plan the work", "Search repo", "Edit files", "Run tests"]
- A list of objects:
  [{"content":"Plan", "status":"pending"}, {"content":"Search", "status":"in_progress"}]
- An object with 'items': {"items": ["Plan", {"content":"Search","status":"pending"}]}

Valid statuses: "pending", "in_progress", "completed".
If omitted or invalid, status defaults to "pending".
Keep only ONE item 'in_progress' at a time.
"""

def _normalize_one(item: Any) -> Dict[str, str] | None:
    if item is None:
        return None
    # String task
    if isinstance(item, str):
        content = item.strip().lstrip("-•* ").strip()
        if not content:
            return None
        return {"content": content, "status": "pending"}
    # Dict-like task
    if isinstance(item, dict):
        content = (
            item.get("content")
            or item.get("task")
            or item.get("title")
            or ""
        ).strip()
        if not content:
            return None
        status = (item.get("status") or "pending").lower().replace("in-progress", "in_progress")
        if status not in {"pending", "in_progress", "completed"}:
            status = "pending"
        return {"content": content, "status": status}
    # Fallback
    return _normalize_one(str(item))

def _normalize_todos(raw: Any) -> List[Dict[str, str]]:
    todos: List[Dict[str, str]] = []
    if isinstance(raw, str):
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        for ln in lines:
            item = _normalize_one(ln)
            if item:
                todos.append(item)
    elif isinstance(raw, list):
        for el in raw:
            item = _normalize_one(el)
            if item:
                todos.append(item)
    elif isinstance(raw, dict) and "items" in raw:
        for el in raw["items"]:
            item = _normalize_one(el)
            if item:
                todos.append(item)
    else:
        got = _normalize_one(raw)
        if got:
            todos.append(got)

    # Cap length just to be safe
    return todos[:50]

@tool(description=WRITE_TODOS_DESCRIPTION)
def write_todos(
    todos: Any = None,              # now optional
    items: Any = None,              # common alias models use
    value: Any = None,              # another alias fallback
    state: Annotated[DeepAgentState, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> Command:
    raw = None
    for candidate in (todos, items, value):
        if candidate is not None:
            raw = candidate
            break
    if raw is None:
        raw = []  # empty list is fine
    normalized = _normalize_todos(raw)
    return Command(update={
        "todos": normalized,
        "messages": [ToolMessage(f"Updated todos ({len(normalized)} items).", tool_call_id=tool_call_id)],
    })