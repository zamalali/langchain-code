# from __future__ import annotations
# from typing import Optional, List
# from pathlib import Path

# from langchain.agents import AgentExecutor, create_tool_calling_agent
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.tools import BaseTool

# from ..config import get_model
# from ..tools.fs_local import (
#     make_read_file_tool, make_write_file_tool, make_list_dir_tool, make_edit_by_diff_tool
# )
# from ..tools.git_tools import make_github_tools
# from ..tools.search import make_glob_tool, make_grep_tool
# from ..tools.shell import make_run_cmd_tool
# from ..tools.processor import make_process_multimodal_tool

# from ..workflows.base_system import BASE_SYSTEM
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from ..mcp_loader import get_mcp_tools
# import asyncio

# try: 
#     from langchain_tavily import TavilySearch
# except Exception: 
#     TavilySearch = None  
# def build_prompt(instruction_seed: Optional[str]) -> ChatPromptTemplate:
#     system_extra = ("\n\n" + instruction_seed) if instruction_seed else ""
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", BASE_SYSTEM + system_extra),
#         MessagesPlaceholder("chat_history"),
#         ("human", "{input}"),
#         MessagesPlaceholder("agent_scratchpad"),
#     ])
#     return prompt


# def build_react_agent(
#     provider: str,
#     project_dir: Path,
#     apply: bool = False,
#     test_cmd: Optional[str] = None,
#     instruction_seed: Optional[str] = None,
# ) -> AgentExecutor:
#     model = get_model(provider)
#     mcp_tools = asyncio.run(get_mcp_tools())
#     tools: List[BaseTool] = [
#         make_glob_tool(str(project_dir)),
#         make_grep_tool(str(project_dir)),
#         make_list_dir_tool(str(project_dir)),
#         make_read_file_tool(str(project_dir)),
#         make_edit_by_diff_tool(str(project_dir), apply),
#         make_write_file_tool(str(project_dir), apply),
#         make_run_cmd_tool(str(project_dir), apply, test_cmd),
#         make_process_multimodal_tool(str(project_dir), model),

#     ]
#     tools.extend(make_github_tools())
#     tools.extend(mcp_tools)
#     if TavilySearch:
#         tools.append(TavilySearch(max_results=5, topic="general"))
#     prompt = build_prompt(instruction_seed)
#     agent = create_tool_calling_agent(model, tools, prompt)
#     executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
#     return executor


































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
from ..tools.git_tools import make_github_tools
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
    
    # Enhanced system prompt to prevent early stopping
    enhanced_system = BASE_SYSTEM + """

CRITICAL EXECUTION RULES:
- You MUST complete all requested tasks before finishing
- If you see code changes with git diff, you MUST also update documentation and push code as requested
- Never finish the chain until ALL parts of the user's request are completed
- If you encounter any errors, try alternative approaches rather than stopping
- Always explain what you're doing and why at each step
- If unsure about next steps, ask for clarification rather than stopping
""" + system_extra

    prompt = ChatPromptTemplate.from_messages([
        ("system", enhanced_system),
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
        make_run_cmd_tool(str(project_dir), apply, test_cmd),
        make_process_multimodal_tool(str(project_dir), model),
    ]
    
    tools.extend(make_github_tools())
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
        max_iterations=20,  # Increase from default (usually 15)
        max_execution_time=300,  # 5 minutes timeout
        early_stopping_method="generate",  # Continue generating even if max_iterations reached
        handle_parsing_errors=True,  # Handle parsing errors gracefully
        return_intermediate_steps=True,  # Useful for debugging
    )
    
    return executor