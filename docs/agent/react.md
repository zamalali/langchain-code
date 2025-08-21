# ReAct Agent

This module implements the ReAct-based coding agent.

## `build_prompt`

Constructs the core prompt.  Starts with `BASE_SYSTEM`, adding an optional `instruction_seed` for workflow specialization (e.g., feature implementation).

Includes placeholders for:
- `chat_history`
- `input`
- `agent_scratchpad`

## `build_react_agent`

Main factory function for creating the `AgentExecutor`.

**Arguments:**
- `provider`: LLM provider (e.g., `anthropic`, `gemini`).
- `project_dir`: Project root directory (scopes filesystem operations).
- `apply`: If `True`, tools modify files/run commands without confirmation.
- `test_cmd`: Optional command (e.g., `"pytest"`) for `{TEST_CMD}`.
- `instruction_seed`: Optional string to specialize agent behavior.

### Tools

The agent uses these tools:

- **Filesystem:** `glob`, `grep`, `list_dir`, `read_file`, `edit_by_diff`, `write_file`
- **Execution:** `run_cmd`
- **Processing:** `process_multimodal`
- **External:** `TavilySearch` (if available), `mcp_tools`

### Agent and Executor

1.  Gets the model (`get_model`).
2.  Assembles the tools.
3.  Creates the prompt (`build_prompt`).
4.  Creates the agent (`create_tool_calling_agent`).
5.  Creates the executor (`AgentExecutor`, `verbose=True`).
