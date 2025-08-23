<div align="center">
  <img src="assets/logo.png" alt="LangCode Logo" width="180" />
  <h1><b>LangCode</b></h1>
  <p><b>The only CLI you need.</b></p>
</div>

---

LangCode is a terminal‑first, agentic coding tool that understands your codebase and gets work done. It combines the best ideas from Google’s Gemini CLI and Anthropic’s Claude Code into a single, extensible workflow: navigate code, plan changes, apply safe diffs, run tests, and manage git — all from the command line.

## At a glance

* Terminal‑first developer UX
* Multi‑provider LLM routing (Anthropic, Google Gemini)
* Safe, reviewable edits by diff with explicit apply gates
* Long‑horizon “Deep Agent” for complex, multi‑step tasks
* ReAct loop for quick chats, lookups, and edits
* Built‑in tools (fs, grep, glob, cmds, tests) and MCP extension support
* Optional fully autonomous DEEP AUTOPILOT mode

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

Start in the current directory:

```bash
langcode chat
```

Implement a feature:

```bash
langcode feature "Add a /health endpoint to the main API" --test-cmd "pytest -q"
```

Fix a bug:

```bash
langcode fix "Login button unresponsive on main page"
langcode fix --log errors.log
```

Run a complex task with the Deep Agent:

```bash
deepagent "Implement JWT auth with refresh tokens and tests"
```

Select an LLM provider explicitly:

```bash
langcode chat --llm gemini
# or
langcode chat --llm anthropic
```

Run without confirmation:

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

### `langcode chat`

Interactive session for exploration, Q\&A, and scoped edits.

```bash
langcode chat [--llm <provider>] [--include-directories <paths>] [--apply]
```

### `langcode feature`

Plan → edit‑by‑diff → test for a single feature request.

```bash
langcode feature [OPTIONS] <REQUEST>

Options:
  --test-cmd "pytest -q"    # optional test runner
  --apply                    # apply file edits and run commands without prompts
  --llm <provider>
```

### `langcode fix`

Diagnose and patch with a description or stack trace.

```bash
langcode fix [OPTIONS] [REQUEST]

Options:
  --log <FILE>               # error log or stack trace
  --test-cmd "npm test"      # optional test runner
  --apply
  --llm <provider>
```

### `deepagent`

Execute multi‑step, long‑horizon tasks using the Deep Agent architecture.

```bash
deepagent <REQUEST>
```

## Workflows

* `auto` – DEEP AUTOPILOT end‑to‑end execution
* `bug_fix` – guided diagnosis → patch → verify
* `feature_impl` – plan → small diffs → test → review

## How it works

### Hybrid Intelligent LLM Router

A rule‑augmented, feedback‑aware router picks the right model for each task based on complexity, context size, latency, and cost. Use `--llm` to override.

### Agent architectures

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

## Contributing

Issues and PRs are welcome. Please open an issue to discuss substantial changes before submitting a PR. See `CONTRIBUTING.md` for guidelines.

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgments

LangCode draws inspiration from the design and developer experience of Google’s Gemini CLI and Anthropic’s Claude Code, unified into a single, streamlined tool.
