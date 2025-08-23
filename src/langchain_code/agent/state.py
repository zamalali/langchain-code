from __future__ import annotations
from langgraph.prebuilt.chat_agent_executor import AgentState
from typing_extensions import TypedDict, NotRequired, Annotated
from typing import Literal

class Todo(TypedDict):
    content: str
    status: Literal["pending", "in_progress", "completed"]

def file_reducer(l, r):
    if l is None: return r
    if r is None: return l
    return {**l, **r}

class DeepAgentState(AgentState):
    todos: NotRequired[list[Todo]]
    files: Annotated[NotRequired[dict[str, str]], file_reducer]
