<div align="center">
<pre>
██╗      █████╗ ███╗   ██╗ ██████╗  ██████╗ ██████╗ ██████╗ ███████╗
██║     ██╔══██╗████╗  ██║██╔════╝ ██╔════╝██╔═══██╗██╔══██╗██╔════╝
██║     ███████║██╔██╗ ██║██║  ███╗██║     ██║   ██║██║  ██║█████╗  
██║     ██╔══██║██║╚██╗██║██║   ██║██║     ██║   ██║██║  ██║██╔══╝  
███████╗██║  ██║██║ ╚████║╚██████╔╝╚██████╗╚██████╔╝██████╔╝███████╗
╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝  ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝
</pre>
<p>
    <b>A command-line agent that writes and edits code based on your instructions.</b>
<p>
</div>

---

**LangCode** is an experimental AI coding agent that operates directly in your terminal. It uses a ReAct-style loop with a curated set of tools to understand your codebase, plan changes, and execute them safely. It can help you implement features, fix bugs, or just chat about your code.

## Features

-   **🤖 Interactive Chat:** Have a conversation with the agent about your project.
-   **✨ Feature Implementation:** Describe a new feature, and the agent will plan and implement it.
-  **🐞 Bug Fixes:** Provide a bug description or a stack trace, and the agent will diagnose and patch the code.
-   **🛠️ Tool-Based:** Uses a set of tools for file system operations, code editing, and running commands.
-   **🔒 Safe by Default:** Requires your confirmation before applying any file edits or running any commands. Use the `--apply` flag to override.
-   **🧠 Multi-Provider:** Supports multiple LLM providers (currently Anthropic and Google Gemini).

## Highlights

-   **Interactive Debugging:** Use `langcode chat` to navigate your codebase, ask questions, and plan changes with the agent.
-   **Test-Driven Development:** Provide a test command with your feature request (`--test-cmd`), and the agent will run it to verify its changes.
-   **Automated Fixes:** Pass an error log to `langcode fix` (`--log`), and the agent will use the stack trace to find and patch the bug.
-   **Safe and Controllable:** The agent previews all file edits and commands before executing them. Use the `--apply` flag for fully autonomous operation.
-   **LLM Flexibility:** Switch between supported LLM providers (like Anthropic and Gemini) using the `--llm` option.

## How It Works

LangCode operates using a sophisticated agent architecture that dynamically selects the best approach for a given task.

### Hybrid Intelligent LLM Router

At its core, LangCode uses a **Hybrid Intelligent LLM Router** to analyze incoming requests. This router assesses the task's complexity, context size, and requirements (e.g., speed, multimodality) to select the most suitable Large Language Model (LLM) from a registered pool. This ensures that simple, fast tasks are handled by nimble models, while complex, long-context tasks are routed to more powerful ones, optimizing for both performance and cost.

The router uses a combination of rule-based scoring and a multi-armed bandit algorithm, allowing it to learn and adapt over time based on performance feedback.

### Agent Architectures

Based on the task, the router can delegate to different agent architectures:

1.  **ReAct Agent:** A simple and efficient agent that uses a ReAct-style loop for straightforward tasks like answering questions or performing simple file operations.
2.  **Deep Agent:** For complex, long-horizon tasks like implementing features or fixing bugs, LangCode employs a **Deep Agent**. This agent uses a structured approach with specialized sub-agents:
    -   **`research-agent`**: Gathers context by searching the web and the local codebase.
    -   **`code-agent`**: Makes and validates code changes using small, verifiable diffs and running tests.
    -   **`git-agent`**: Manages version control by staging files and crafting informative commit messages.

This multi-agent system allows for a clear separation of concerns and more robust execution of complex plans.

### Available Tools

The agents have access to a curated set of tools to interact with your project:

-   **`list_dir`**: List files and directories.
-   **`read_file`**: Read a file's content.
-   **`edit_by_diff`**: Apply changes to a file using a diff, promoting safe and reviewable edits.
-   **`write_file`**: Create or overwrite a file.
-   **`glob`**: Find files using glob patterns.
-   **`grep`**: Search for regex patterns within files.
-   **`run_cmd`**: Execute shell commands, requiring user confirmation by default.
-   **`process_multimodal`**: Process text and images, enabling visual understanding.
-   **MCP Tools**: A suite of tools for interacting with version control, web search, and more.


## Getting Started

### 1. Installation

First, ensure you have Python 3.10 or newer. You can install LangCode using pip:

```bash
pip install langchain-code
```

### 2. Configuration

The agent requires API keys for your chosen LLM provider. It loads these from a `.env` file in your project directory.

1.  **Create a `.env` file** in the root of your project:
    ```bash
    touch .env
    ```

2.  **Add your API keys** to the file. For example:
    ```env
    # For Anthropic Claude
    ANTHROPIC_API_KEY="sk-ant-..."

    # For Google Gemini
    GOOGLE_API_KEY="AIzaSy..."
    ```

The agent will automatically detect which provider to use based on the available environment variables. You can also explicitly choose a provider with the `--llm` option (e.g., `--llm gemini`).

### 3. Usage

The main entry point is the `langcode` command, which has three sub-commands: `chat`, `feature`, and `fix`.

#### `langcode chat`

Opens an interactive session to chat with the agent about your project. It's useful for asking questions, understanding code, or planning changes.

**Usage:**
```bash
langcode chat [OPTIONS]
```

**Example:**
```bash
# Start a chat session in the current directory
langcode chat

# Start a session using a specific LLM provider
langcode chat --llm gemini
```

#### `langcode feature`

Implements a new feature from a single request. The agent will plan the changes, write the code, and optionally run tests to verify it.

**Usage:**
```bash
langcode feature [OPTIONS] <REQUEST>
```

**Example:**
```bash
# Request a new feature
langcode feature "Add a /health endpoint to the main API"

# Verify the implementation with a test command
langcode feature "Refactor the user model to use UUIDs" --test-cmd "pytest -q"

# Allow the agent to apply edits without confirmation
langcode feature "Add a dark mode toggle" --apply
```

#### `langcode fix`

Diagnoses and fixes a bug. You can describe the bug or provide a log file containing a stack trace.

**Usage:**
```bash
langcode fix [OPTIONS] [REQUEST]
```

**Example:**
```bash
# Describe the bug to fix
langcode fix "The login button is not working on the main page"

# Provide an error log file for context
langcode fix --log "errors.log"

# Combine a description with a log file and verify with a test
langcode fix "Fix the crash on image upload" --log "crash.log" --test-cmd "npm test"
```

### Image Support in Chat

You can ask the agent to analyze images by using the `/img` command in a chat session.

**Syntax:**
```
/img <path1> [<path2> ...] :: <prompt>
```

**Example:**
```
> /img assets/screenshot.png :: What's wrong with this UI?
```

The agent will use the `process_multimodal` tool to understand the image and respond to your prompt.

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Branches

### `main`

This is the main branch of the project. It contains the latest stable version of the code.

### `dev`

This is the development branch. It contains the latest features and bug fixes. It is not guaranteed to be stable.

### `draft-work`

This is a draft work branch. It is used for experimental features and code that is not yet ready for production. It is not guaranteed to be functional.

**Note:** This branch is a dumping ground for ideas and work-in-progress code. It is not guaranteed to be stable or even functional. For stable, working code, please refer to the `dev` and `main` branches.
