from __future__ import annotations
from typing import Optional, List
from pathlib import Path

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool

from ..config import get_model
from ..tools.fs_local import (
    make_read_file_tool, make_write_file_tool, make_list_dir_tool, make_edit_by_diff_tool
)
from ..tools.search import make_glob_tool, make_grep_tool
from ..tools.shell import make_run_cmd_tool
from ..workflows.base_system import BASE_SYSTEM

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from ..mcp_loader import get_mcp_tools
import asyncio

try: 
    from langchain_tavily import TavilySearch
except Exception: 
    TavilySearch = None  
def build_prompt(instruction_seed: Optional[str]) -> ChatPromptTemplate:
    system_extra = ("\n\n" + instruction_seed) if instruction_seed else ""
    prompt = ChatPromptTemplate.from_messages([
        ("system", BASE_SYSTEM + system_extra),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    return prompt


def build_react_agent(
    provider: str,
    project_dir: Path,
    apply: bool = False,
    test_cmd: Optional[str] = None,
    instruction_seed: Optional[str] = None,
) -> AgentExecutor:
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

    mcp_tools = asyncio.run(get_mcp_tools())
    tools.extend(mcp_tools)
    if TavilySearch:
        tools.append(TavilySearch(max_results=5, topic="general"))

    prompt = build_prompt(instruction_seed)
    agent = create_tool_calling_agent(model, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return executor
