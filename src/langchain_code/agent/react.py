from __future__ import annotations
from typing import Optional, List
from pathlib import Path

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from .deep_agents import create_deep_agent
from .subagents import RESEARCH_SUBAGENT, CRITIQUE_SUBAGENT
from ..config import get_model
from ..tools.fs_local import (
    make_read_file_tool, make_write_file_tool, make_list_dir_tool, make_edit_by_diff_tool, make_delete_file_tool
)
from ..tools.search import make_glob_tool, make_grep_tool
from ..tools.shell import make_run_cmd_tool
from ..tools.processor import make_process_multimodal_tool

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
        MessagesPlaceholder("chat_history"),
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
    mcp_tools = asyncio.run(get_mcp_tools())
    
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
    ]
    
    # tools.extend(make_github_tools())
    tools.extend(mcp_tools)
    
    if TavilySearch:
        tools.append(TavilySearch(max_results=5, topic="general"))
    
    prompt = build_prompt(instruction_seed)
    agent = create_tool_calling_agent(model, tools, prompt)
    
    # Enhanced AgentExecutor configuration
    executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        max_iterations=20,  
        max_execution_time=300,  
        early_stopping_method="generate",  
        handle_parsing_errors=True,  
        return_intermediate_steps=True,  
    )
    
    return executor

def build_deep_agent(
    provider: str,
    project_dir: Path,
    apply: bool = False,
    test_cmd: Optional[str] = None,
    instruction_seed: Optional[str] = None,
    subagents: Optional[list] = None,
):
    return create_deep_agent(
        provider=provider,
        project_dir=project_dir,
        instructions=instruction_seed,
        subagents=subagents or [RESEARCH_SUBAGENT, CRITIQUE_SUBAGENT],
        apply=apply,
        test_cmd=test_cmd,
    )