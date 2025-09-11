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

def replace_reducer(_, new):
    return new

class DeepAgentState(AgentState):
    todos: Annotated[NotRequired[list[Todo]], replace_reducer]   
    files: Annotated[NotRequired[dict[str, str]], file_reducer]
