# Command-Line Interface (CLI)

The `cli.py` module provides the command-line interface for LangCode. It uses the `typer` library to create a user-friendly and powerful CLI.

## Usage

The CLI is the main entry point for interacting with the LangCode agent. It provides commands for different tasks, such as chatting with the agent, implementing features, and fixing bugs.

### Commands

- **`chat`**: Opens an interactive chat session with the agent.
Example:
```bash
langcode chat --llm gemini --project-dir myproject
```
This starts a chat session using the Gemini language model and working within the `myproject` directory.

- **`feature`**: Implements a new feature from a given request.
Example:
```bash
langcode feature "Add a new user authentication system" --apply --test-cmd "pytest"
```
This requests the implementation of a new user authentication system, applies the changes automatically, and runs pytest to verify the changes.

- **`fix`**: Fixes a bug based on a request and an optional error log.
Example:
```bash
langcode fix "Resolve the issue with database connection" --project-dir myproject
```
This requests a fix for a database connection issue within the `myproject` directory.

### Options

Each command supports a set of common options:

- **`--llm`**: Specifies the language model provider to use (e.g., `anthropic`, `gemini`).
- **`--project-dir`**: Sets the root directory for the project the agent will work on.
- **`--apply`**: Allows the agent to apply changes to files and run commands without interactive confirmation.
- **`--test-cmd`**: Provides a test command that the agent can run to verify its changes.

## Error Handling

The CLI provides informative error messages to the user in case of failures.  These messages indicate the type of error and suggest potential solutions.

## Session UI

The CLI provides a rich and interactive session UI using the `rich` library. This includes:

- An ASCII banner for the LangCode application.
- A status panel showing the current provider, project directory, and other session information.
- Formatted output from the agent, including markdown and code blocks.
