from __future__ import annotations
from typing import TypedDict, NotRequired, Annotated, Any, Dict, List
from langchain_core.tools import tool, BaseTool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.prebuilt import InjectedState, create_react_agent
from ..agent.state import DeepAgentState
from ..config_core import get_model

RESEARCH_SUBAGENT = {
    "name": "research-agent",
    "description": "In-depth web research and synthesis.",
    "prompt": "You are an expert researcher. Produce precise, sourced notes, then a short summary.",
    "include_files": True,
    "include_todos": True,
    "model_settings": {"provider": "gemini", "temperature": 0.1},
}

CRITIQUE_SUBAGENT = {
    "name": "critique-agent",
    "description": "Strict editor that critiques and improves final drafts.",
    "prompt": "Be terse and ruthless. Fix clarity, structure, and correctness.",
    "include_todos": False,
    "model_settings": {"provider": "gemini", "temperature": 0.1},
}

class SubAgent(TypedDict):
    name: str
    description: str
    prompt: str
    tools: NotRequired[list[str]]
    include_files: NotRequired[bool]
    include_todos: NotRequired[bool]
    model_settings: NotRequired[Dict[str, Any]]

def _maybe_model(base_model, ms: Dict[str, Any] | None):
    if not ms:
        return base_model
    model = base_model
    if "provider" in ms:  # gemini | anthropic
        model = get_model(ms["provider"])

    bind_args = {k: ms[k] for k in ("temperature", "max_tokens") if k in ms}
    if bind_args:
        try:
            model = model.bind(**bind_args)
        except Exception:
            pass
    return model

def _index_tools(tools: List[BaseTool]) -> Dict[str, BaseTool]:
    out: Dict[str, BaseTool] = {}
    for t in tools:
        if not isinstance(t, BaseTool):
            t = tool(t)
        out[t.name] = t
    return out

def create_task_tool(
    tools: List[BaseTool],
    instructions: str,
    subagents: List[SubAgent],
    base_model,
    state_schema,
):
    tools_by_name = _index_tools(tools)

    agents = {
        "general-purpose": create_react_agent(
            base_model, prompt=instructions, tools=tools, state_schema=state_schema
        )
    }
    configs = {"general-purpose": {"include_files": True, "include_todos": True}}

    for sa in subagents or []:
        model = _maybe_model(base_model, sa.get("model_settings"))
        allowed_tools = tools if "tools" not in sa else [tools_by_name[n] for n in sa["tools"] if n in tools_by_name]
        agents[sa["name"]] = create_react_agent(
            model, prompt=sa["prompt"], tools=allowed_tools, state_schema=state_schema
        )
        configs[sa["name"]] = {
            "include_files": sa.get("include_files", False),
            "include_todos": sa.get("include_todos", False),
        }

    other = "\n".join(
        f"- {n}: {sa.get('description','')}"
        for n, sa in ((s["name"], s) for s in (subagents or []))
    )

    @tool(description=f"""Launch a specialized sub-agent to execute a complex task.
Available agents:
- general-purpose: general research/exec agent (Tools: all)
{other}

Usage:
- description: detailed, self-contained instruction for the subagent
- subagent_type: one of {list(agents.keys())}""")
    async def task(
        description: str,
        subagent_type: str,
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ):
        if subagent_type not in agents:
            return f"Error: unknown subagent_type '{subagent_type}'. Allowed: {list(agents.keys())}"

        sub_state: Dict[str, Any] = {"messages": [{"role": "user", "content": description}]}
        cfg = configs.get(subagent_type, {})
        if cfg.get("include_files"):
            sub_state["files"] = state.get("files", {})
        if cfg.get("include_todos"):
            sub_state["todos"] = state.get("todos", [])

        result = await agents[subagent_type].ainvoke(sub_state)
        update: Dict[str, Any] = {
            "messages": [ToolMessage(result["messages"][-1].content, tool_call_id=tool_call_id)]
        }
        if cfg.get("include_files"):
            update["files"] = result.get("files", {})
        if cfg.get("include_todos"):
            update["todos"] = result.get("todos", [])
        return Command(update=update)

    return task
