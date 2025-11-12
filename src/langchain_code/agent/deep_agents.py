from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, List, Optional, Union

from langgraph.prebuilt import create_react_agent
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from ..agent.state import DeepAgentState
from ..agent.subagents import SubAgent, create_task_tool
from ..config_core import get_model
from ..mcp_loader import get_mcp_tools
from ..tools.fs_local import (
    make_delete_file_tool,
    make_edit_by_diff_tool,
    make_list_dir_tool,
    make_read_file_tool,
    make_write_file_tool,
)
from ..tools.mermaid import make_mermaid_tools
from ..tools.planner import write_todos, append_todo, update_todo_status, clear_todos
from ..tools.processor import make_process_multimodal_tool
from ..tools.shell import make_run_cmd_tool
from ..tools.script_exec import make_script_exec_tool
from ..tools.search import make_glob_tool, make_grep_tool
from ..static_values import BASE_DEEP_SUFFIX
from ..workflows.base_system import BASE_SYSTEM

def load_langcode_context(project_dir: Path) -> str:
    ctx_file = project_dir / ".langcode" / "langcode.md"
    if ctx_file.exists():
        try:
            return "\n\n# Project Context\n" + ctx_file.read_text(encoding="utf-8")
        except Exception as e:
            return f"\n\n# Project Context\n(Error reading langcode.md: {e})"
    return ""


try:
    from langchain_tavily import TavilySearch
except Exception:
    TavilySearch = None

async def _load_dynamic_tools(project_dir: Path, model, apply: bool, test_cmd: Optional[str]) -> List[BaseTool]:
    tools: List[BaseTool] = [
        make_glob_tool(str(project_dir)),
        make_grep_tool(str(project_dir)),
        make_list_dir_tool(str(project_dir)),
        make_read_file_tool(str(project_dir)),
        make_edit_by_diff_tool(str(project_dir), apply),
        make_write_file_tool(str(project_dir), apply),
        make_delete_file_tool(str(project_dir), apply),
        make_run_cmd_tool(str(project_dir), apply, test_cmd),
        make_script_exec_tool(str(project_dir), apply),
        make_process_multimodal_tool(str(project_dir), model),
        write_todos,
        append_todo,
        update_todo_status,
        clear_todos
        ]
    tools.extend(await get_mcp_tools(project_dir))

    if TavilySearch and os.getenv("TAVILY_API_KEY"):
        try:
            tools.append(
                TavilySearch(
                    max_results=5,
                    topic="general",
                    description=(
                        "Use TavilySearch for internet or websearch to answer questions "
                        "that require up-to-date information from the web. "
                        "Best for research, current events, general knowledge, news etc."
                    ),
                )
            )
        except Exception as e:
            print(f"[LangCode] Tavily disabled (reason: {e})")

    tools.extend(make_mermaid_tools(str(project_dir)))
    return tools


def create_deep_agent(
    *,
    provider: str,
    project_dir: Path,
    instructions: Optional[str] = None,
    subagents: Optional[List[SubAgent]] = None,
    apply: bool = False,
    test_cmd: Optional[str] = None,
    state_schema=DeepAgentState,
    checkpointer: Optional[Any] = None,
    llm: Optional[Any] = None,
):
    """
    Returns a LangGraph graph with planning + subagents.
    """
    model = llm or get_model(provider)
    project_context = load_langcode_context(project_dir)
    prompt = (BASE_SYSTEM + "\n" + (instructions or "") + "\n" + BASE_DEEP_SUFFIX + project_context).strip()

    tools = asyncio.run(_load_dynamic_tools(project_dir, model, apply, test_cmd))
    task_tool = create_task_tool(tools, instructions or BASE_SYSTEM, subagents or [], model, state_schema)
    all_tools: List[Union[BaseTool, Any]] = [*tools, task_tool]


    if checkpointer is None:
        checkpointer = MemorySaver()

    graph = create_react_agent(
        model,
        prompt=prompt,
        tools=all_tools,
        state_schema=state_schema,
        checkpointer=checkpointer,
    )
    graph.config = {"recursion_limit": 250}
    return graph
