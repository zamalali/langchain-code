# langchain-code

`langchain-code` is a command-line tool that provides an AI agent to help you with your codebase. It can chat with you about your code, implement new features, and fix bugs. It uses the LangChain ecosystem and supports multiple LLM providers like Anthropic and Gemini.

## Features

*   **Interactive Chat**: Have a conversation with the AI about your project.
*   **Feature Implementation**: Describe a feature, and the agent will plan, search, edit files, and verify the implementation.
*   **Bug Fixing**: Provide a bug description or an error log, and the agent will trace, pinpoint, patch, and test the fix.
*   **Safe Edits**: All file modifications are shown as diffs and require your confirmation by default.
*   **LLM Provider Support**: Choose between `anthropic` and `gemini` models.

## Usage

The main entry point is the `langchain-code` command, which provides the following sub-commands.

### `chat`

Open an interactive chat session with the agent in your project directory.

```bash
langchain-code chat
```

**Options:**
*   `--llm [anthropic|gemini]`: Specify the LLM provider to use.
*   `--project-dir <path>`: Set the project directory (defaults to current directory).

### `feature`

Request a new feature to be implemented.

```bash
langchain-code feature "Add a dark mode toggle in the settings"
```

**Arguments:**
*   `REQUEST`: A description of the feature you want to add (e.g., "Add a dark mode toggle in settings").

**Options:**
*   `--llm [anthropic|gemini]`: Specify the LLM provider.
*   `--project-dir <path>`: Set the project directory.
*   `--test-cmd <cmd>`: A command to run to verify the feature (e.g., `"pytest -q"`).
*   `--apply`: Apply file changes and run commands without asking for interactive confirmation.

### `fix`

Fix a bug in the codebase.

```bash
langchain-code fix "The app crashes on image upload" --log ./error.log
```

**Arguments:**
*   `REQUEST`: A description of the bug (optional if a log is provided).

**Options:**
*   `--log <path>`: Path to an error log or stack trace.
*   `--llm [anthropic|gemini]`: Specify the LLM provider.
*   `--project-dir <path>`: Set the project directory.
*   `--test-cmd <cmd>`: A command to run to verify the fix (e.g., `"pytest"`).
*   `--apply`: Apply file changes and run commands without asking for interactive confirmation.
