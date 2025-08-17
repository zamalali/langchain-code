# from __future__ import annotations
# from typing import Optional, List, Dict
# from pathlib import Path

# from langchain_core.tools import BaseTool

# from ..config import get_model
# from ..tools.fs_local import (
#     make_read_file_tool, make_write_file_tool, make_list_dir_tool, make_edit_by_diff_tool
# )
# from ..tools.search import make_glob_tool, make_grep_tool
# from ..tools.shell import make_run_cmd_tool
# from ..mcp_loader import get_mcp_tools
# import asyncio

# try:
#     from langchain_tavily import TavilySearch
# except Exception:
#     TavilySearch = None

# try:
#     from deepagents import create_deep_agent
# except Exception:
#     create_deep_agent = None


# def build_deep_agent(
#     provider: str,
#     project_dir: Path,
#     apply: bool = False,
#     test_cmd: Optional[str] = None,
#     instruction_seed: Optional[str] = None,
# ):
#     if create_deep_agent is None:
#         raise RuntimeError("deepagents is not installed. Run: pip install deepagents")

#     model = get_model(provider)

#     tools: List[BaseTool] = [
#         make_glob_tool(str(project_dir)),
#         make_grep_tool(str(project_dir)),
#         make_list_dir_tool(str(project_dir)),
#         make_read_file_tool(str(project_dir)),
#         make_edit_by_diff_tool(str(project_dir), apply),
#         make_write_file_tool(str(project_dir), apply),
#         make_run_cmd_tool(str(project_dir), apply, test_cmd),
#     ]
#     if TavilySearch:
#         tools.append(TavilySearch(max_results=5, topic="general"))

#     mcp_tools = asyncio.run(get_mcp_tools())
#     tools.extend(mcp_tools)

#     tool_map: Dict[str, BaseTool] = {getattr(t, "name", f"tool_{i}"): t for i, t in enumerate(tools)}
#     fs_names = [n for n in tool_map if n in {"glob", "grep", "list_dir", "read_file", "edit_by_diff", "write_file"}]
#     shell_names = [n for n in tool_map if n in {"run_cmd"}]
#     search_names = [n for n in tool_map if "tavily" in n] 
#     reserved = set(fs_names + shell_names + search_names)
#     mcp_names = [n for n in tool_map if n not in reserved]

#     research_subagent = {
#         "name": "researcher",
#         "description": "Web/MCP research and synthesis.",
#         "prompt": "Plan → search → read → synthesize → cite sources clearly.",
#         "tools": search_names + mcp_names,
#     }
#     coder_subagent = {
#         "name": "coder",
#         "description": "Read code, propose minimal diffs, run tests/commands.",
#         "prompt": "Prefer small, reversible patches via edit_by_diff; verify with tests/commands when provided.",
#         "tools": fs_names + shell_names,
#     }
#     subagents = [research_subagent, coder_subagent]

#     instructions = (
#         "You are a deep coding+research agent.\n"
#         "Follow: (1) plan, (2) gather minimal context, (3) propose safe diffs, "
#         "(4) verify with tests/commands if configured, (5) summarize results and next steps.\n"
#     )
#     if instruction_seed:
#         instructions += f"\n--- Task Seed ---\n{instruction_seed}\n"

#     agent = create_deep_agent(
#         tools=list(tool_map.values()),
#         instructions=instructions,
#         subagents=subagents,
#         model=model,
#     )
#     return agent










# langcode/agent/deep.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Dict
from deepagents import create_deep_agent
from langchain_core.tools import BaseTool
from ..config import get_model
from ..tools.fs_local import make_read_file_tool, make_write_file_tool, make_list_dir_tool, make_edit_by_diff_tool
from ..tools.search import make_glob_tool, make_grep_tool
from ..tools.shell import make_run_cmd_tool
from ..tools.processor import make_process_multimodal_tool
from ..mcp_loader import get_mcp_tools
import asyncio

DEEP_SYSTEM_HINT = """You are a long-horizon coding agent. Always:
1) Write a short PLAN first using your planning tool.
2) Execute steps iteratively; after risky steps, verify with tests or static checks.
3) Prefer small, reviewable diffs; group related edits; write meaningful commit messages.
4) Use subagents when research or focused code edits would pollute main context.
5) Before writes or shell commands, summarize intent and expected outcome."""

def _base_tools(project_dir: Path, apply: bool, test_cmd: Optional[str]) -> List[BaseTool]:
    model = get_model("gemini")  # model is overridden below; placeholder for multimodal tool
    return [
        make_glob_tool(str(project_dir)),
        make_grep_tool(str(project_dir)),
        make_list_dir_tool(str(project_dir)),
        make_read_file_tool(str(project_dir)),
        make_edit_by_diff_tool(str(project_dir), apply),
        make_write_file_tool(str(project_dir), apply),
        make_run_cmd_tool(str(project_dir), apply, test_cmd),
        make_process_multimodal_tool(str(project_dir), model),
    ]

async def _mcp_tools() -> List[BaseTool]:
    return await get_mcp_tools()

def build_deep_agent(
    provider: str,
    project_dir: Path,
    apply: bool = False,
    test_cmd: Optional[str] = None,
    instruction_seed: Optional[str] = None,
):
    model = get_model(provider)
    tools = _base_tools(project_dir, apply, test_cmd)
    tools.extend(asyncio.run(_mcp_tools()))

    # Subagents: focused tool scopes
    subagents: List[Dict] = [
        {
            "name": "research-agent",
            "description": "Web + repo search; produce brief research notes & sources.",
            "prompt": "Be concise. Prefer authoritative sources; include citations.",
            "tools": [t.name for t in tools if t.name in {"glob", "grep", "TavilySearch", "web_fetch", "search"}],
        },
        {
            "name": "code-agent",
            "description": "Make and validate targeted code changes.",
            "prompt": "Edit with small diffs; run tests; summarize changes and risks.",
            "tools": [t.name for t in tools if t.name in {"read_file", "write_file", "edit_by_diff", "run_cmd", "glob", "grep"}],
        },
        {
            "name": "git-agent",
            "description": "Prepare commits/branches and PR descriptions.",
            "prompt": "Stage only relevant files; craft informative commit messages.",
            "tools": [t.name for t in tools if "git" in t.name or t.name in {"run_cmd"}],
        },
    ]

    instructions = (instruction_seed or "") + "\n\n" + DEEP_SYSTEM_HINT
    agent = create_deep_agent(
        tools=tools,
        instructions=instructions,
        subagents=subagents,
        model=model,
    )
    return agent
