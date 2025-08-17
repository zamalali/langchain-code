from __future__ import annotations
from typing import Optional, List, Dict
from pathlib import Path

from langchain_core.tools import BaseTool

from ..config import get_model
from ..tools.fs_local import (
    make_read_file_tool, make_write_file_tool, make_list_dir_tool, make_edit_by_diff_tool
)
from ..tools.search import make_glob_tool, make_grep_tool
from ..tools.shell import make_run_cmd_tool
from ..mcp_loader import get_mcp_tools
import asyncio

try:
    from langchain_tavily import TavilySearch
except Exception:
    TavilySearch = None

try:
    from deepagents import create_deep_agent
except Exception:
    create_deep_agent = None


def build_deep_agent(
    provider: str,
    project_dir: Path,
    apply: bool = False,
    test_cmd: Optional[str] = None,
    instruction_seed: Optional[str] = None,
):
    if create_deep_agent is None:
        raise RuntimeError("deepagents is not installed. Run: pip install deepagents")

    model = get_model(provider)

    tools: List[BaseTool] = [
        make_glob_tool(str(project_dir)),
        make_grep_tool(str(project_dir)),
        make_list_dir_tool(str(project_dir)),
        make_read_file_tool(str(project_dir)),
        make_edit_by_diff_tool(str(project_dir), apply),
        make_write_file_tool(str(project_dir), apply),
        make_run_cmd_tool(str(project_dir), apply, test_cmd),
    ]
    if TavilySearch:
        tools.append(TavilySearch(max_results=5, topic="general"))

    mcp_tools = asyncio.run(get_mcp_tools())
    tools.extend(mcp_tools)

    tool_map: Dict[str, BaseTool] = {getattr(t, "name", f"tool_{i}"): t for i, t in enumerate(tools)}
    fs_names = [n for n in tool_map if n in {"glob", "grep", "list_dir", "read_file", "edit_by_diff", "write_file"}]
    shell_names = [n for n in tool_map if n in {"run_cmd"}]
    search_names = [n for n in tool_map if "tavily" in n] 
    reserved = set(fs_names + shell_names + search_names)
    mcp_names = [n for n in tool_map if n not in reserved]

    research_subagent = {
        "name": "researcher",
        "description": "Web/MCP research and synthesis.",
        "prompt": "Plan → search → read → synthesize → cite sources clearly.",
        "tools": search_names + mcp_names,
    }
    coder_subagent = {
        "name": "coder",
        "description": "Read code, propose minimal diffs, run tests/commands.",
        "prompt": "Prefer small, reversible patches via edit_by_diff; verify with tests/commands when provided.",
        "tools": fs_names + shell_names,
    }
    subagents = [research_subagent, coder_subagent]

    instructions = (
        "You are a deep coding+research agent.\n"
        "Follow: (1) plan, (2) gather minimal context, (3) propose safe diffs, "
        "(4) verify with tests/commands if configured, (5) summarize results and next steps.\n"
    )
    if instruction_seed:
        instructions += f"\n--- Task Seed ---\n{instruction_seed}\n"

    agent = create_deep_agent(
        tools=list(tool_map.values()),
        instructions=instructions,
        subagents=subagents,
        model=model,
    )
    return agent
