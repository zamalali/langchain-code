<div align="center">
  <img src="../assets/logo.png" alt="LangCode Logo" width="180" />
  <h1 align="center">LangCode</h1>

  <p align="center"><i><b>The only CLI you'll ever need!</b></i></p>
</div>

# MCP Loader

The MCP Loader is responsible for loading and initializing the Multi-Code Pal (MCP) agent. It dynamically loads configurations, tools, and other components to construct the agent at runtime.

## Key Responsibilities

- **Dynamic Loading:** Loads agent configurations, tools, and other resources from specified paths.
- **Agent Initialization:** Initializes the MCP agent with the loaded components.
- **Extensibility:** Allows for easy extension and customization by adding new tools or modifying configurations without changing the core code.

## Usage

The `load_mcp` function is the main entry point for loading the MCP agent. It takes the necessary paths and configurations to initialize the agent.

```python
from langcode.mcp_loader import load_mcp

# Load the MCP agent with default configurations
mcp_agent = load_mcp()

# Execute a task
result = mcp_agent.run("Some task")
```
