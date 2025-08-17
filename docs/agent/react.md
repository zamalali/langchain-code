# ReAct Agent

This module is responsible for constructing and configuring the ReAct-based coding agent.

## `build_prompt`

This function constructs the core prompt for the agent. It starts with a `BASE_SYSTEM` prompt that provides the agent with its core instructions, persona, and constraints. An optional `instruction_seed` can be appended to specialize the agent for a particular workflow (e.g., feature implementation or bug fixing).

The final prompt template includes placeholders for:
- `chat_history`: To maintain conversation context.
- `input`: The user's current request.
- `agent_scratchpad`: The internal monologue and tool usage of the ReAct agent.

## `build_react_agent`

This is the main factory function for creating the `AgentExecutor`.

**Arguments:**
- `provider`: The LLM provider to use (e.g., `anthropic`, `gemini`).
- `project_dir`: The root directory of the project the agent will work on. This scopes all filesystem operations.
- `apply`: If `True`, tools that modify the filesystem (`edit_by_diff`, `write_file`) or run commands (`run_cmd`) will do so without interactive confirmation.
- `test_cmd`: An optional command (e.g., `"pytest"`) that the `run_cmd` tool can execute via the special placeholder `{TEST_CMD}`.
- `instruction_seed`: An optional string to specialize the agent's behavior.

### Tools

The agent is equipped with a suite of tools to interact with the local environment:

- **Filesystem Tools:**
    - `glob`: Find files using glob patterns.
    - `grep`: Search for content within files.
    - `list_dir`: List the contents of a directory.
    - `read_file`: Read the contents of a file.
    - `edit_by_diff`: Apply a change to a file using a diff format. This is a safe way to make targeted edits.
    - `write_file`: Write content to a file, overwriting it if it exists.

- **Execution Tools:**
    - `run_cmd`: Execute a shell command.

- **Processing Tools:**
    - `process_multimodal`: A tool for analyzing images and text together.

- **External Tools:**
    - `TavilySearch`: If available, this tool allows the agent to search the web for information.
    - `mcp_tools`: A set of tools for interacting with the Mission Control Platform.

### Agent and Executor

1.  **Model:** The `get_model` function is called to get the appropriate language model.
2.  **Tools:** The list of tools is assembled.
3.  **Prompt:** The `build_prompt` function is called to create the prompt.
4.  **Agent:** The `create_tool_calling_agent` function from LangChain is used to create the agent, binding the model, tools, and prompt together.
5.  **Executor:** An `AgentExecutor` is created to run the agent. The `verbose=True` setting allows for the agent's internal monologue to be printed to the console, which is useful for debugging.
