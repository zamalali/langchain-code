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

BASE_SYSTEM = """You are a coding agent running in a terminal.
You can reason and act with tools until the task is complete.
Always:
1) Make a brief plan.
2) Use minimal tool calls to gather context (glob/grep/read).
3) Propose edits with small, safe changes. Prefer edit_by_diff.
4) If a test command is provided, run it to verify.
5) Summarize results and next steps.

Output diffs or concrete commands rather than long prose."""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def build_prompt(instruction_seed: Optional[str]) -> ChatPromptTemplate:
    system_extra = ("\n\n" + instruction_seed) if instruction_seed else ""
    # Removed the chat_history placeholder; AgentExecutor will still manage tool messages via agent_scratchpad
    prompt = ChatPromptTemplate.from_messages([
        ("system", BASE_SYSTEM + system_extra),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    return prompt


def build_agent(
    provider: str,
    project_dir: Path,
    apply: bool = False,
    test_cmd: Optional[str] = None,
    instruction_seed: Optional[str] = None,
) -> AgentExecutor:
    model = get_model(provider)

    # Build tools with captured config (no .bind())
    tools: List[BaseTool] = [
        make_glob_tool(str(project_dir)),
        make_grep_tool(str(project_dir)),
        make_list_dir_tool(str(project_dir)),
        make_read_file_tool(str(project_dir)),
        make_edit_by_diff_tool(str(project_dir), apply),
        make_write_file_tool(str(project_dir), apply),
        make_run_cmd_tool(str(project_dir), apply, test_cmd),
    ]

    prompt = build_prompt(instruction_seed)
    agent = create_tool_calling_agent(model, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return executor
