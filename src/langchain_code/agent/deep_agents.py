from __future__ import annotations
from typing import Optional, List, Union, Any
from pathlib import Path
from langgraph.prebuilt import create_react_agent
from langgraph.types import Checkpointer
from langchain_core.tools import BaseTool
from ..agent.state import DeepAgentState
from ..agent.subagents import create_task_tool, SubAgent
from ..tools.planner import write_todos
from ..workflows.base_system import BASE_SYSTEM
from ..config import get_model
from ..tools.fs_local import make_list_dir_tool, make_read_file_tool, make_write_file_tool, make_edit_by_diff_tool, make_delete_file_tool
from ..tools.search import make_glob_tool, make_grep_tool
from ..tools.shell import make_run_cmd_tool
from ..tools.processor import make_process_multimodal_tool
from ..tools.mermaid import make_mermaid_tools
from ..mcp_loader import get_mcp_tools
import asyncio

try:
    from langchain_tavily import TavilySearch
except Exception:
    TavilySearch = None

BASE_DEEP_SUFFIX = """
## Planning & Tasking
- Use `write_todos` to outline steps, update statuses live, and mark completion IMMEDIATELY after finishing a step.
- For complex work or to quarantine context, call `task(description, subagent_type=...)` to launch a sub-agent.

## Subagents
- Prefer 'general-purpose' for iterative research/execution.
"""

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
        make_process_multimodal_tool(str(project_dir), model),
        write_todos,
    ]
    tools.extend(await get_mcp_tools())
    if TavilySearch:
        tools.append(TavilySearch(max_results=5, topic="general"))
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
    checkpointer: Optional[Checkpointer] = None,
    llm: Optional[Any] = None,
):
    """
    Returns a LangGraph graph (same as deepagents) with planning + subagents.

    Backward compatible:
    - If llm is provided, use it.
    - Else fall back to get_model(provider) (original behavior).
    """
    model = llm or get_model(provider)
    prompt = (BASE_SYSTEM + "\n" + (instructions or "") + "\n" + BASE_DEEP_SUFFIX).strip()

    tools = asyncio.run(_load_dynamic_tools(project_dir, model, apply, test_cmd))
    task_tool = create_task_tool(tools, instructions or BASE_SYSTEM, subagents or [], model, state_schema)
    all_tools: List[Union[BaseTool, Any]] = [*tools, task_tool]

    graph = create_react_agent(
        model,
        prompt=prompt,
        tools=all_tools,
        state_schema=state_schema,
        checkpointer=checkpointer,
    )
    graph.config = {"recursion_limit": 50}
    return graph
