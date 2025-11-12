"""Shared static values used across the LangCode package."""

from __future__ import annotations

from pathlib import Path

APP_HELP = """
LangCode - ReAct + Tools + Deep (LangGraph) code agent CLI.

Just type `langcode` and hit enter - it's the only CLI you'll ever need.
Toggle across everything without leaving the terminal!

Use it to chat with an agent, implement features, fix bugs, or analyze a codebase.

Key flags (for `chat`):
  \x07 --mode [react|deep]   Choose the reasoning engine (default: react).
      - react  : Classic ReAct agent with tools.
      - deep   : LangGraph-style multi-step agent.
  \x07 --auto                 Autopilot (deep mode only). The deep agent will plan+act end-to-end
                           WITHOUT asking questions (it still uses tools safely). Think "hands-off planning".
  \x07 --apply                Write changes to disk and run commands for you (feature/fix flows).
                           If OFF, the agent proposes diffs only. Think "permission to execute".

  \x07 --router               Auto-route to the most efficient LLM per query (uses Gemini if --llm not provided).
  \x07 --priority             Router priority: balanced | cost | speed | quality (default: balanced)
  \x07 --verbose              Show router model-selection panels.

Examples:
  \x07 langcode chat --llm anthropic --mode react
  \x07 langcode chat --llm gemini --mode deep --auto
  \x07 langcode chat --router --priority cost --verbose
  \x07 langcode feature "Add a dark mode toggle" --router --priority quality
  \x07 langcode fix --log error.log --test-cmd "pytest -q" --router
  \x07 langcode tell me what's going on in the codebase     (quick mode \x1a analyze) 
  \x07 langcode fix this                                    (quick mode \x1a fix; reads TTY log if available)
Custom instructions:
  \x07 Put project-specific rules in .langcode/langcode.md (created automatically).
  \x07 From the launcher, select "Custom Instructions" to open your editor; or run `langcode instr`.

NEW:
  \x07 Just run `langcode` to open a beautiful interactive launcher.
    Use \x18/\x19 to move, \x1b/\x1a to change values, Enter to start, h for help, q to quit.
  \x07 In chat, type /select to return to the launcher without exiting.
""".strip()

PROMPT = "[bold green]langcode[/bold green] [dim]>[/dim] "

ENV_FILENAMES = (".env", ".env.local")

GLOBAL_ENV_ENVVAR = "LANGCODE_GLOBAL_ENV"
LANGCODE_CONFIG_DIR_ENVVAR = "LANGCODE_CONFIG_DIR"

LANGCODE_DIRNAME = ".langcode"
LANGCODE_FILENAME = "langcode.md"
MCP_FILENAME = "mcp.json"
MCP_PROJECT_REL = Path("src") / "langchain_code" / "config" / MCP_FILENAME

BASE_SYSTEM = """You are LangCode, a coding assistant with access to filesystem, shell, and web tools.

## Core Behavior
- Use tools to discover information before acting
- Make changes autonomously - don't ask for permission or paths
- Always verify your changes by reading files after editing
- Provide clear, factual responses based on tool outputs

## Available Tools
- **Files**: list_dir, glob, read_file, edit_by_diff, write_file, delete_file
- **Search**: grep (find text in files)  
- **Shell**: run_cmd (git, tests, etc.)
- **Scripts**: script_exec (run short Python/Bash/PowerShell/Node scripts in the repo)
- **Web**: TavilySearch
- **Multimodal**: process_multimodal (for images)
- **Planning**: write_todos (track progress)
- **Terminal**: read_terminal (mandatory when the user asks things like *"what�?Ts this?"*, *"what�?Ts this error?"*, or *"fix this"* without details �?" the info is almost always in the terminal)

## Workflow
1. **Check context**: If the user refers vaguely to �?othis,�?? �?oerror,�?? or �?ofix this,�??
   immediately call `read_terminal` to capture the terminal contents before doing anything else.
2. **Discover**: Use glob/grep/list_dir to understand the codebase
3. **Read**: Use read_file on relevant files
4. **Act**: Make precise edits with edit_by_diff or create new files with write_file
5. **Verify**: Re-read files and run commands to confirm changes
6. **Commit**: Use git commands to save your work


## Rules
- Always use tools rather than guessing
- For vague user queries about issues, errors, or �?owhat�?Ts this,�??
  invoke `read_terminal` immediately �?" do not ask the user to paste errors.
- For file edits, show exactly what changed
- Include relevant command outputs in your response
- Keep responses focused and actionable
"""

RUNTIME_POLICY = """
## Runtime Discipline (LangCode)
- Explore first: `list_dir`, `glob`, `grep` to find targets. Never ask the user for paths.
- Directory handling: if a folder is missing, just write files to that nested path (parents are auto-created).
- Script fallback (when tools can't express the logic cleanly):
  1) Prefer **Python** short scripts; else bash/pwsh/node if truly needed.
  2) Run with `script_exec(language="python", code=..., timeout_sec=60)`.
  3) If exit != 0, read stderr/stdout, **fix the script**, and retry up to 2 times.
  4) Keep responses factual and include the key log lines (command, exit, brief stdout/stderr).
- Verification: after edits/scripts, re-read files or run a quick command to confirm.
- Never spawn background daemons; keep everything inside the project root.
""".strip()

BASE_DEEP_SUFFIX = """
## Planning & TODOs
- In your FIRST 1-2 tool calls, call `write_todos([...])` with ~3-8 concrete steps.
- Before working a step, call `update_todo_status(index, "in_progress")`.
- After finishing it, call `update_todo_status(index, "completed")`.
- If you discover new work, call `append_todo("...")` and execute it.
- Keep only one item "in_progress" at a time and keep todos verb-first and specific.

## Subagents
- Prefer 'general-purpose' for iterative research/execution.
"""

FEATURE_INSTR = """You are implementing a feature end-to-end.
- Plan steps first (files to inspect, edits to make).
- Use glob/grep to locate relevant files.
- Use read_file to inspect.
- Make targeted edits via edit_by_diff (preferred) or write_file for new files.
- If a test command is available, call run_cmd with command "{TEST_CMD}" to execute it.
- Present a concise summary of changes (list of files edited/created) and next steps."""

BUGFIX_INSTR = """You are fixing a bug.
- Parse any provided error log.
- Use grep to locate suspicious symbols/stack frames.
- Read the minimal code to understand the issue.
- Propose a minimal safe patch via edit_by_diff (preferred).
- If a test command is available, call run_cmd with command "{TEST_CMD}" to run it.
- Explain the fix briefly and show the resulting diff."""

AUTO_DEEP_INSTR = """Execute the request completely and autonomously. 

**MANDATORY TERMINATION RULE:**
After completing your work, you MUST output exactly one message starting with "FINAL:" and then STOP. Do not continue using tools after outputting FINAL:.

**Steps:**
1. Discover the codebase structure (glob, grep, read key files)
   - Always find files yourself using shell search (ls, find, grep, glob) and never rely on user hints/inputs. Walk the filesystem when needed.
2. Make the requested changes (edit_by_diff or write_file)  
3. Test/verify your changes (run_cmd)
4. For visual content (diagrams, charts, images), generate only the rendered outputs when requested
5. Output FINAL: report and STOP

**Termination Condition:**
Once you have:
- Used at least one discovery tool (glob/grep/read_file)
- Made the requested changes
- Generated all requested outputs (including rendered visual like PNG from mermaid diagrams). Avoid saving .mmd files
- Run at least one shell command
- Committed your work (or attempted to)

Then output your FINAL: report and do NOT use any more tools.

**Output Format:**
```
FINAL:
- Accomplished: [what you did]
- Files changed: [list of files]  
- Command results: [key outputs]
- Status: [complete/blocked and why]
```

**Rules:**
- No intermediate status updates
- Use tools for all facts
- Don't ask questions - act autonomously
- Complete ALL requested deliverables before terminating
- STOP after outputting FINAL: - do not continue
"""
 
CHAT_SESSION_TITLE = "LangChain Code Agent | Chat"
DEEP_CHAT_SESSION_TITLE = "LangChain Code Agent | Deep Chat"
AUTO_CHAT_SUFFIX = " (Auto)"

TODO_PANEL_TITLE = "TODOs"
TODO_PLANNING_TEXT = "Planning tasks..."
TODO_EMPTY_TEXT = "No tasks were emitted by the agent."
TODO_ANIMATION_DELAY = 0.15
TODO_STEP_HEADER = "Agent steps:"

AUTOPILOT_PROMPT = (
    "AUTOPILOT: Start now. Discover files (glob/list_dir/grep), read targets (read_file), "
    "perform edits (edit_by_diff/write_file), and run at least one run_cmd (git/tests) capturing stdout/"
    "stderr + exit code. Then produce one 'FINAL:' report and STOP. No questions."
)

FEATURE_SESSION_TITLE = "LangChain Code Agent | Feature"
FIX_SESSION_TITLE = "LangChain Code Agent | Fix"
ANALYZE_SESSION_TITLE = "LangChain Code Agent | Analyze"
GLOBAL_ENV_TITLE = "LangCode | Global Environment"
PROJECT_ENV_TITLE = "LangCode | Project Environment"
INSTRUCTIONS_TITLE = "LangChain Code Agent | Custom Instructions"

FIX_FALLBACK_PROMPT = "Fix the bug using the provided log."

PROVIDER_KEY_LABELS = {
    "OPENAI_API_KEY": "OpenAI",
    "ANTHROPIC_API_KEY": "Anthropic",
    "GOOGLE_API_KEY": "Gemini",
    "GEMINI_API_KEY": "Gemini (alt)",
    "GROQ_API_KEY": "Groq",
    "TOGETHER_API_KEY": "Together",
    "FIREWORKS_API_KEY": "Fireworks",
    "PERPLEXITY_API_KEY": "Perplexity",
    "DEEPSEEK_API_KEY": "DeepSeek",
    "TAVILY_API_KEY": "Tavily (web search)",
}

DOCTOR_FOOTER_TIP = "Tip: run 'langcode instr' to set project rules; edit environment via the launcher."
