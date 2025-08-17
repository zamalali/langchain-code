# Deep Agent

This module constructs a more complex "deep" agent that orchestrates sub-agents for specialized tasks. It's designed for complex problems that require both research and coding.

## `build_deep_agent`

This function builds the deep agent.

**Arguments:**
- `provider`: The LLM provider.
- `project_dir`: The root directory for filesystem operations.
- `apply`: If `True`, filesystem and command tools will execute without confirmation.
- `test_cmd`: An optional test command for the `run_cmd` tool.
- `instruction_seed`: An optional string to specialize the agent's behavior.

### Sub-agents

The deep agent is composed of two sub-agents:

1.  **Researcher:** This sub-agent is responsible for web and Mission Control Platform (MCP) research. It has access to the `TavilySearch` tool and any MCP tools that are available.
2.  **Coder:** This sub-agent is responsible for reading and writing code, and running tests. It has access to the local filesystem and shell tools.

### Agent Construction

1.  **Tools:** The function assembles a list of tools, including filesystem tools, shell tools, and external tools like `TavilySearch` and MCP tools.
2.  **Sub-agents:** The `researcher` and `coder` sub-agents are defined with their respective tools and prompts.
3.  **Instructions:** A main prompt is constructed for the deep agent, which includes the `instruction_seed` if provided.
4.  **Agent:** The `create_deep_agent` function from the `deepagents` library is used to create the final agent. This function takes the tools, instructions, sub-agents, and model as input.
