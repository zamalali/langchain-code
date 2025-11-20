from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, List, Optional

from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from langgraph.types import Checkpointer
from ..config_core import get_model
from ..mcp_loader import get_mcp_tools
from ..static_values import RUNTIME_POLICY
from ..tools.fs_local import (
    make_delete_file_tool,
    make_edit_by_diff_tool,
    make_list_dir_tool,
    make_read_file_tool,
    make_write_file_tool,
)
from ..tools.mermaid import make_mermaid_tools
from ..tools.shell import make_run_cmd_tool, make_read_terminal_tool
from ..tools.processor import make_process_multimodal_tool
from ..tools.search import make_glob_tool, make_grep_tool
from ..tools.script_exec import make_script_exec_tool
from ..workflows.base_system import BASE_SYSTEM

logger = logging.getLogger(__name__)

try:
    from langchain_tavily import TavilySearch  
except Exception:  
    TavilySearch = None  


def _maybe_make_tavily_tool() -> Optional[BaseTool]:
    if TavilySearch is None or not os.getenv("TAVILY_API_KEY"):
        return None
    try:
        return TavilySearch(
            max_results=5,
            topic="general",
            description=(
                "Use TavilySearch for internet/web search when you need up-to-date info. "
                "Best for research, current events, and general knowledge."
            ),
        )
    except Exception as e:  
        logger.warning("Tavily disabled: %s", e)
        return None


def load_langcode_context(project_dir: Path) -> str:
    ctx_file = project_dir / ".langcode" / "langcode.md"
    if ctx_file.exists():
        try:
            return "\n\n# Project Context\n" + ctx_file.read_text(encoding="utf-8")
        except Exception as e:  
            return f"\n\n# Project Context\n(Error reading langcode.md: {e})"
    return ""


def _escape_braces(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


def build_prompt(instruction_seed: Optional[str], project_dir: Path) -> ChatPromptTemplate:
    system_extra = ("\n\n" + instruction_seed) if instruction_seed else ""
    project_context = load_langcode_context(project_dir)
    system_text = _escape_braces(BASE_SYSTEM + "\n\n" + RUNTIME_POLICY + system_extra + project_context)
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_text),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )


def build_react_agent(
    provider: str,
    project_dir: Path,
    apply: bool = False,
    test_cmd: Optional[str] = None,
    instruction_seed: Optional[str] = None,
    *,
    llm: Optional[Any] = None,
):
    """Build a ReAct agent using LangChain 1.0 create_agent API.
    
    This creates a fast-loop agent for chat, reads, and targeted edits.
    Uses the new LangChain 1.0 create_agent which handles tool calling
    identically across all providers (Anthropic, Gemini, OpenAI, Ollama).
    
    Args:
        provider: LLM provider ("anthropic", "gemini", "openai", "ollama")
        project_dir: Root directory for filesystem operations
        apply: Whether to write changes to disk
        test_cmd: Optional test command to run
        instruction_seed: Optional system prompt customization
        llm: Optional pre-configured LLM instance
        
    Returns:
        Compiled agent runnable supporting .invoke(), .stream(), etc.
    """
    model = llm or get_model(provider)
    try:
        mcp_tools: List[BaseTool] = asyncio.run(get_mcp_tools(project_dir))
    except RuntimeError:
        loop = asyncio.get_event_loop()
        mcp_tools = loop.run_until_complete(get_mcp_tools(project_dir))
    except Exception as e:  
        logger.warning("Failed to load MCP tools: %s", e)
        mcp_tools = []

    tool_list: List[BaseTool] = [
        make_glob_tool(str(project_dir)),
        make_grep_tool(str(project_dir)),
        make_list_dir_tool(str(project_dir)),
        make_read_file_tool(str(project_dir)),
        make_edit_by_diff_tool(str(project_dir), apply),
        make_write_file_tool(str(project_dir), apply),
        make_delete_file_tool(str(project_dir), apply),
        make_script_exec_tool(str(project_dir), apply, return_direct=False),
        make_process_multimodal_tool(str(project_dir), model),
        make_run_cmd_tool(str(project_dir), apply, test_cmd),
        make_read_terminal_tool()
    ]

    tool_list.extend(mcp_tools)
    t = _maybe_make_tavily_tool()
    if t:
        tool_list.append(t)
    tool_list.extend(make_mermaid_tools(str(project_dir)))

    system_prompt = build_prompt(instruction_seed, project_dir)

    # LangChain 1.0: create_agent is the new unified API
    # Returns a compiled runnable that works across all providers identically
    agent = create_agent(
        model,
        tools=tool_list,
        system_prompt=str(system_prompt)
    )
    
    return agent


def build_deep_agent(
    provider: str,
    project_dir: Path,
    apply: bool = False,
    test_cmd: Optional[str] = None,
    instruction_seed: Optional[str] = None,
    subagents: Optional[list] = None,
    checkpointer: Optional[Checkpointer] = None,
    *,
    llm: Optional[Any] = None,
):
    from .deep_agents import create_deep_agent  
    return create_deep_agent(
        provider=provider,
        project_dir=project_dir,
        instructions=instruction_seed,
        subagents=subagents or [],
        apply=apply,
        test_cmd=test_cmd,
        checkpointer=checkpointer,
        llm=llm,
    )
