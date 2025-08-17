# Command-Line Interface (CLI)

The `cli.py` module provides the command-line interface for LangCode. It uses the `typer` library to create a user-friendly and powerful CLI.

## Usage

The CLI is the main entry point for interacting with the LangCode agent. It provides commands for different tasks, such as chatting with the agent, implementing features, and fixing bugs.

### Commands

- **`chat`**: Opens an interactive chat session with the agent.
- **`feature`**: Implements a new feature from a given request.
- **`fix`**: Fixes a bug based on a request and an optional error log.

### Options

Each command supports a set of common options:

- **`--llm`**: Specifies the language model provider to use (e.g., `anthropic`, `gemini`).
- **`--project-dir`**: Sets the root directory for the project the agent will work on.
- **`--apply`**: Allows the agent to apply changes to files and run commands without interactive confirmation.
- **`--test-cmd`**: Provides a test command that the agent can run to verify its changes.

## Session UI

The CLI provides a rich and interactive session UI using the `rich` library. This includes:

- An ASCII banner for the LangCode application.
- A status panel showing the current provider, project directory, and other session information.
- Formatted output from the agent, including markdown and code blocks.
