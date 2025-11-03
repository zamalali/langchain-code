"""Shared constants for the LangCode CLI."""

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
"""

PROMPT = "[bold green]langcode[/bold green] [dim]>[/dim] "

ENV_FILENAMES = (".env", ".env.local")

GLOBAL_ENV_ENVVAR = "LANGCODE_GLOBAL_ENV"
LANGCODE_CONFIG_DIR_ENVVAR = "LANGCODE_CONFIG_DIR"

LANGCODE_DIRNAME = ".langcode"
LANGCODE_FILENAME = "langcode.md"
MCP_FILENAME = "mcp.json"
MCP_PROJECT_REL = Path("src") / "langchain_code" / "config" / MCP_FILENAME
