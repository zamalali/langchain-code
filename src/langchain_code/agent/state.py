from __future__ import annotations
from langgraph.graph import MessagesState
from typing_extensions import TypedDict, NotRequired, Annotated
from typing import Literal, Any
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class Todo(TypedDict):
    content: str
    status: Literal["pending", "in_progress", "completed"]

def file_reducer(l, r):
    if l is None: return r
    if r is None: return l
    return {**l, **r}

def replace_reducer(_, new):
    return new

class DeepAgentState(MessagesState):
    """LangChain 1.0 compliant state schema for deep agent.
    
    Extends MessagesState with custom state channels:
    - remaining_steps: Required by langgraph.prebuilt.create_react_agent
    - todos: List of Todo objects (replaced completely on update)
    - files: Dict of file contents (merged incrementally on update)
    """
    remaining_steps: int
    todos: Annotated[NotRequired[list[Todo]], replace_reducer]
    files: Annotated[NotRequired[dict[str, str]], file_reducer]
