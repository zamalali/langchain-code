<div align="center">
  <img src="assets/logo.png" alt="LangCode Logo" width="180" />
  <h1><b>LangCode</b></h1>
  <p style="font-size: 1.5em; font-style: italic; color: #3498db;">The only CLI you need.</p>
</div>

---

LangCode is a terminal-first, agentic coding tool designed to deeply understand your codebase and automate coding tasks. It seamlessly integrates the best features of Google's Gemini CLI and Anthropic's Claude Code into a unified and extensible workflow, enabling you to navigate code, plan changes, apply safe diffs, run tests, and manage Git operations directly from the command line. LangCode empowers developers to focus on high-level problem-solving by automating repetitive and time-consuming coding tasks.

## At a glance

* **Terminal-first developer UX:** Provides a seamless command-line experience for interacting with your codebase.
* **Multi-provider LLM routing:** Intelligently routes requests to the most appropriate Large Language Model (LLM) provider (Anthropic, Google Gemini) based on task requirements.
* **Safe, reviewable edits by diff:** Ensures code changes are safe and easily reviewable by generating explicit diffs and requiring confirmation before applying modifications.
* **Long-horizon “Deep Agent”:** Enables the execution of complex, multi-step tasks that require sustained reasoning and planning.
* **ReAct loop:** Facilitates quick chats, code lookups, and targeted edits through an interactive Read-Act-Observe loop.
* **Built-in tools:** Offers a comprehensive suite of built-in tools (fs, grep, glob, cmds, tests) and supports Model Context Protocol (MCP) extension for enhanced functionality.
* **DEEP AUTOPILOT mode:** Provides an optional fully autonomous mode for end-to-end task execution with minimal human intervention.

## Installation

Requirements:

* Python 3.10+
* macOS, Linux, or Windows (WSL recommended)

Install from PyPI:

```bash
pip install langchain-code
```

Verify:

```bash
langcode --help
```

## Quick start

1.  **Installation:**
    ```bash
    pip install langchain-code
    ```

2.  **Set up API keys:**
    Create a `.env` file in your project root and add your API keys for Anthropic and/or Google Gemini:
    ```env
    ANTHROPIC_API_KEY="sk-ant-..."
    GOOGLE_API_KEY="AIzaSy..."
    ```

3.  **Start an interactive chat session:**
    ```bash
    langcode chat
    ```

4.  **Implement a feature:**
    ```bash
    langcode feature "Add a /health endpoint to the main API" --test-cmd "pytest -q"
    ```

5.  **Fix a bug:**
    ```bash
    langcode fix "Login button unresponsive on main page" --log errors.log
    ```

6.  **Run a complex task with the Deep Agent:**
    ```bash
    langcode deepagent "Implement JWT auth with refresh tokens and tests"
    ```

7.  **Select an LLM provider explicitly:**
    ```bash
    langcode chat --llm gemini
    # or
    langcode chat --llm anthropic
    ```

8.  **Run without confirmation (use with caution):**
    ```bash
    langcode feature "Add dark mode toggle" --apply
    ```

## Authentication & configuration

LangCode reads provider credentials from a `.env` in your project root (or environment variables).

Create `.env`:

```bash
touch .env
```

Populate with keys for the providers you use:

```env
# Anthropic
ANTHROPIC_API_KEY="sk-ant-..."

# Google Gemini (API or OAuth-backed key)
GOOGLE_API_KEY="AIzaSy..."
```

LLM selection:

* Automatic: LangCode picks based on available keys and router policy.
* Explicit: `--llm gemini` or `--llm anthropic`.

Project settings (optional):

* `./.langcode/mcp.json` – register MCP servers (see below)
* `./.langcode/` – put project‑specific context or defaults

## Commands

LangCode provides a set of commands to interact with your codebase. Here's a breakdown of each command and its usage:

### `langcode chat`

Starts an interactive chat session with the agent. This is useful for exploring the codebase, asking questions, and making scoped edits.

```bash
langcode chat [--llm <provider>] [--mode <react|deep>] [--auto] [--router] [--priority <balanced|cost|speed|quality>] [--project-dir <path>]
```

*   `--llm`: Specifies the LLM provider to use (anthropic or gemini). If not specified, LangCode will automatically select a provider based on availability and router policy.
*   `--mode`: Selects the reasoning engine to use: `react` (default) or `deep`. The `deep` mode enables more complex, multi-step reasoning.
*   `--auto`: (Deep mode only) Enables autopilot mode, where the agent plans and executes tasks end-to-end without asking for confirmation.
*   `--router`: Enables smart model routing, which automatically picks the most efficient LLM per prompt based on the specified priority.
*   `--priority`: Sets the routing priority: `balanced` (default), `cost`, `speed`, or `quality`. This influences the model choice when routing is enabled.
*   `--project-dir`: Specifies the project directory to operate in. Defaults to the current directory.

**Example:**
```bash
langcode chat --llm gemini --mode deep --auto
```
This command starts a chat session using the Gemini LLM in deep mode with autopilot enabled.

### `langcode feature`

Implements a feature end-to-end, following a plan -> edit -> verify workflow. This command is designed to automate the process of adding new features to your codebase.

```bash
langcode feature <request> [--test-cmd <command>] [--apply] [--llm <provider>] [--router] [--priority <balanced|cost|speed|quality>] [--project-dir <path>]
```

*   `<request>`: A description of the feature to implement (e.g., "Add a dark mode toggle").
*   `--test-cmd`: Specifies a command to run to verify the changes (e.g., `pytest -q`).
*   `--apply`: Applies the changes and runs the test command without prompting for confirmation.
*   `--llm`: Specifies the LLM provider to use (anthropic or gemini).
*   `--router`: Enables smart model routing.
*   `--priority`: Sets the routing priority.
*   `--project-dir`: Specifies the project directory.

**Example:**
```bash
langcode feature "Implement user authentication with JWT" --test-cmd "pytest -q" --apply
```
This command implements user authentication with JWT, runs the tests, and applies the changes without prompting.

### `langcode fix`

Diagnoses and fixes a bug, following a trace -> pinpoint -> patch -> test workflow. This command helps automate the process of fixing bugs in your codebase.

```bash
langcode fix <request> [--log <file>] [--test-cmd <command>] [--apply] [--llm <provider>] [--router] [--priority <balanced|cost|speed|quality>] [--project-dir <path>]
```

*   `<request>`: A description of the bug to fix (e.g., "Fix crash on image upload").
*   `--log`: Specifies the path to an error log or stack trace.
*   `--test-cmd`: Specifies a command to run to verify the fix.
*   `--apply`: Applies the changes and runs the test command without prompting for confirmation.
*   `--llm`: Specifies the LLM provider to use.
*   `--router`: Enables smart model routing.
*   `--priority`: Sets the routing priority.
*   `--project-dir`: Specifies the project directory.

**Example:**
```bash
langcode fix "Resolve memory leak in image processing module" --log memory_leak.log --test-cmd "pytest -q"
```
This command resolves a memory leak using the provided log file and runs the tests to verify the fix.

### `langcode analyze`

Analyzes the codebase and generates insights using the Deep Agent architecture. This command is useful for getting an overview of the project and understanding its architecture.

```bash
langcode analyze <request> [--llm <provider>] [--router] [--priority <balanced|cost|speed|quality>] [--project-dir <path>]
```

*   `<request>`: A question or request for analysis (e.g., "What are the main components of this project?").
*   `--llm`: Specifies the LLM provider to use.
*   `--router`: Enables smart model routing.
*   `--priority`: Sets the routing priority.
*   `--project-dir`: Specifies the project directory.

**Example:**
```bash
langcode analyze "Explain the data flow in the user authentication module"
```
This command analyzes the user authentication module and explains the data flow.

### `langcode instr`

Opens or creates the project-specific instructions file (`.langcode/langcode.md`) in your editor. This file allows you to provide custom instructions and guidelines for the agent to follow.

```bash
langcode instr [--project-dir <path>]
```

*   `--project-dir`: Specifies the project directory.

**Example:**
```bash
langcode instr
```
This command opens the `.langcode/langcode.md` file in your default editor.


## Workflows

* `auto` – DEEP AUTOPILOT end‑to‑end execution
* `bug_fix` – guided diagnosis → patch → verify
* `feature_impl` – plan → small diffs → test → review

## How it works

### Hybrid Intelligent LLM Router

A rule‑augmented, feedback‑aware router picks the right model for each task based on complexity, context size, latency, and cost. Use `--llm` to override.

### Agent architectures

## Agent architectures

### ReAct Agent

The ReAct Agent is a fast loop for chat, reads, and targeted edits. It follows the ReAct (Reasoning and Acting) framework, where the agent reasons about the current state, decides on an action, and then observes the result of that action. This process is repeated until the agent reaches a conclusion or the maximum number of iterations is reached.

### Deep Agent

The Deep Agent is a structured, multi-agent system for complex work. It uses a LangGraph-style architecture to execute multi-step, long-horizon tasks. The Deep Agent consists of several sub-agents, each responsible for a specific task, such as research, code generation, and Git operations.

* **ReAct Agent** – fast loop for chat, reads, and targeted edits.
* **Deep Agent** – structured, multi‑agent system for complex work:

  * `research-agent` – gathers local and web context
  * `code-agent` – generates minimal, verifiable diffs and runs tests
  * `git-agent` – stages changes and crafts commits

### Deep Agent state

* **Todos** – managed via `write_todos`
* **Virtual files** – stage edits safely before touching disk

### DEEP AUTOPILOT mode

Run fully autonomously with a single final report. Combine with tests for safe, end‑to‑end changes.

## Built‑in tools

* `list_dir`, `glob`, `grep` – fast project navigation
* `read_file`, `edit_by_diff`, `write_file` – safe, reviewable edits
* `run_cmd` – gated shell execution
* `process_multimodal` – text + image understanding in chat
* `write_todos` – persistent plan items for Deep Agent

## MCP integration

Extend LangCode with external capabilities using Model Context Protocol. Define servers in `~/.langcode/mcp.json` or `./.langcode/mcp.json`.

Example `mcp.json` snippet:

```json
{
  "servers": {
    "github": { "command": "mcp-github" },
    "search": { "command": "mcp-search" }
  }
}
```

Invoke tools naturally in chat or within Deep Agent plans once configured.

## Image support in chat

Use the `/img` directive to analyze images alongside your prompt:

```text
/img assets/screenshot.png :: What’s wrong with this UI?
```

## Safety & control

* Preview all diffs and commands before execution
* Explicit confirmation gates by default
* `--apply` for non‑interactive runs
* Minimal diffs for clear reviews and quick rollbacks

## Examples

Summarize a repo:

```bash
langcode chat -p "Summarize this codebase and list risky areas"
```

Refactor with tests:

```bash
langcode feature "Switch user IDs to UUID" --test-cmd "pytest -q"
```

Patch from stack trace:

```bash
langcode fix --log crash.log --test-cmd "npm test"
```

Multi‑step change:

```bash
deepagent "Introduce feature flags with config, implement two flags, add tests"
```

## Troubleshooting

* Run with `--verbose` to inspect tool calls and decisions
* Ensure provider keys are present in `.env`
* Narrow scope or provide `--include-directories` for large monorepos
* File an issue with a minimal reproduction

## Configuration

LangCode can be configured via environment variables and project-specific files. The following environment variables are supported:

* `ANTHROPIC_API_KEY`: API key for Anthropic Claude.
* `GOOGLE_API_KEY`: API key for Google Gemini.

Project-specific configuration files:

* `./.langcode/mcp.json`: Configuration for Model Context Protocol (MCP) servers.
* `./.langcode/`: Directory for project-specific context or defaults.

## Custom Tools

LangCode can be extended with custom tools using the Model Context Protocol (MCP). To create a custom tool, you need to define a server that implements the MCP protocol and register it in the `mcp.json` file. Once configured, you can invoke the tool naturally in chat or within Deep Agent plans.

## Contributing

Issues and PRs are welcome. Please open an issue to discuss substantial changes before submitting a PR. See `CONTRIBUTING.md` for guidelines.

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgments

LangCode draws inspiration from the design and developer experience of Google’s Gemini CLI and Anthropic’s Claude Code, unified into a single, streamlined tool.
