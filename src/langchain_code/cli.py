from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import warnings
import sys
import os
import click
import platform
from collections import OrderedDict
import shutil
import subprocess
import difflib
from datetime import datetime
import re
import json
import ast
from contextlib import nullcontext
from io import StringIO

try:
    import termios
    import tty
except Exception:  
    termios = None
    tty = None
try:
    import msvcrt  
except Exception:
    msvcrt = None

import typer
from rich.console import Console, Group
from rich.panel import Panel
from rich.style import Style
from rich.text import Text
from rich.markdown import Markdown
from rich.rule import Rule
from rich.align import Align
from rich.columns import Columns
from rich.table import Table
from rich import box
from pyfiglet import Figlet

warnings.filterwarnings("ignore", message=r"typing\.NotRequired is not a Python type.*")
warnings.filterwarnings("ignore", category=UserWarning, module=r"pydantic\._internal.*")

from .config_core import resolve_provider as _resolve_provider_base
from .config_core import get_model, get_model_info, get_model_by_name

from .agent.react import build_react_agent, build_deep_agent
from .workflows.feature_impl import FEATURE_INSTR
from .workflows.bug_fix import BUGFIX_INSTR
from .workflows.auto import AUTO_DEEP_INSTR
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.callbacks import BaseCallbackHandler
import logging
from typer.core import TyperGroup
from click.exceptions import UsageError

class _DefaultToChatGroup(TyperGroup):
    """Route unknown subcommands to `chat --inline ...` so `langcode "Hi"` works."""
    def resolve_command(self, ctx, args):
        if args and not args[0].startswith("-"):
            try:
                return super().resolve_command(ctx, args)
            except UsageError:
                chat_cmd = self.get_command(ctx, "chat")
                if chat_cmd is None:
                    raise
                if "--inline" not in args:
                    args = ["--inline", *args]
                return chat_cmd.name, chat_cmd, args
        return super().resolve_command(ctx, args)



for _name in (
    "langchain_google_genai",
    "langchain_google_genai.chat_models",
    "tenacity",
    "tenacity.retry",
    "httpx",
    "urllib3",
    "google",
):
    _log = logging.getLogger(_name)
    _log.setLevel(logging.CRITICAL)
    _log.propagate = False


APP_HELP = """
LangCode – ReAct + Tools + Deep (LangGraph) code agent CLI.

Just type `langcode` and hit enter – it's the only CLI you'll ever need.
Toggle across everything without leaving the terminal!

Use it to chat with an agent, implement features, fix bugs, or analyze a codebase.

Key flags (for `chat`):
  • --mode [react|deep]   Choose the reasoning engine (default: react).
      - react  : Classic ReAct agent with tools.
      - deep   : LangGraph-style multi-step agent.
  • --auto                 Autopilot (deep mode only). The deep agent will plan+act end-to-end
                           WITHOUT asking questions (it still uses tools safely). Think “hands-off planning”.
  • --apply                Write changes to disk and run commands for you (feature/fix flows).
                           If OFF, the agent proposes diffs only. Think “permission to execute”.

  • --router               Auto-route to the most efficient LLM per query (uses Gemini if --llm not provided).
  • --priority             Router priority: balanced | cost | speed | quality (default: balanced)
  • --verbose              Show router model-selection panels.

Examples:
  • langcode chat --llm anthropic --mode react
  • langcode chat --llm gemini --mode deep --auto
  • langcode chat --router --priority cost --verbose
  • langcode feature "Add a dark mode toggle" --router --priority quality
  • langcode fix --log error.log --test-cmd "pytest -q" --router
  • langcode tell me what’s going on in the codebase     (quick mode → analyze) 
  • langcode fix this                                    (quick mode → fix; reads TTY log if available)
Custom instructions:
  • Put project-specific rules in .langcode/langcode.md (created automatically).
  • From the launcher, select “Custom Instructions” to open your editor; or run `langcode instr`.

NEW:
  • Just run `langcode` to open a beautiful interactive launcher.
    Use ↑/↓ to move, ←/→ to change values, Enter to start, h for help, q to quit.
  • In chat, type /select to return to the launcher without exiting.
"""

app = typer.Typer( 
    cls=_DefaultToChatGroup,  
    add_completion=False, 
    help=APP_HELP.strip(), 
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}, 
)
console = Console()
PROMPT = "[bold green]langcode[/bold green] [dim]›[/dim] "

_AGENT_CACHE: "OrderedDict[Tuple[str, str, str, str, bool], Any]" = OrderedDict()
_AGENT_CACHE_MAX = 6

_IN_SELECTION_HUB = False

ENV_FILENAMES = (".env", ".env.local")

def _env_template_text(scope: str = "project") -> str:
    """Return the standard .env template for project/global scopes."""
    title = ".env — environment for LangCode" if scope == "project" else "Global .env — environment for LangCode"
    return f"""# {title}
# Fill only what you need; keep secrets safe.
# Examples:
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=...
# GEMINI_API_KEY=...
# GROQ_API_KEY=...
# LANGCHAIN_API_KEY=...
# TAVILY_API_KEY=... (For web search)
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_PROJECT=langcode

# Created {datetime.now().strftime('%Y-%m-%d %H:%M')} by LangCode
"""

GLOBAL_ENV_ENVVAR = "LANGCODE_GLOBAL_ENV"       
LANGCODE_CONFIG_DIR_ENVVAR = "LANGCODE_CONFIG_DIR"  

def _user_config_dir() -> Path:
    """Cross-platform config dir: $LANGCODE_CONFIG_DIR | XDG | APPDATA | ~/.config/langcode."""
    if os.getenv(LANGCODE_CONFIG_DIR_ENVVAR):
        return Path(os.environ[LANGCODE_CONFIG_DIR_ENVVAR]).expanduser()
    if platform.system().lower() == "windows":
        base = os.getenv("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / "LangCode"
    xdg = os.getenv("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "langcode"
    return Path.home() / ".config" / "langcode"

def _global_env_path() -> Path:
    """~/.config/langcode/.env (Linux/macOS), %APPDATA%/LangCode/.env (Windows), or $LANGCODE_GLOBAL_ENV."""
    override = os.getenv(GLOBAL_ENV_ENVVAR)
    if override:
        return Path(override).expanduser()
    if platform.system().lower() == "windows":
        base = os.getenv("APPDATA") or (Path.home() / "AppData" / "Roaming")
        return Path(base) / "LangCode" / ".env"
    xdg = os.getenv("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "langcode" / ".env"
    return Path.home() / ".config" / "langcode" / ".env"

SESSIONS_DIR = _user_config_dir() / "sessions" 
SESSIONS_DIR.mkdir(parents=True, exist_ok=True) 
 
def _current_tty_id() -> str: 
    try: 
        return os.path.basename(os.ttyname(sys.stdin.fileno())).replace("/", "_") 
    except Exception: 
        return f"ppid{os.getppid()}_pid{os.getpid()}" 
 
def _tty_log_path(tty_id: Optional[str] = None) -> Path: 
    name = (tty_id or os.environ.get("LANGCODE_TTY_ID") or _current_tty_id()).strip() or "default" 
    return SESSIONS_DIR / f"{name}.log" 
 
def _tail_bytes(path: Path, max_bytes: int = 200_000) -> str: 
    try: 
        with open(path, "rb") as f: 
            f.seek(0, os.SEEK_END) 
            size = f.tell() 
            f.seek(max(0, size - max_bytes)) 
            return f.read().decode("utf-8", "ignore") 
    except Exception: 
        return "" 
 
_ERR_PATTERNS = [ 
    r"Traceback \(most recent call last\):[\s\S]+?(?=\n{2,}|\Z)",             
    r"(?:UnhandledPromiseRejection|Error:|TypeError:|ReferenceError:)[\s\S]+?(?=\n{2,}|\Z)", 
    r"panic: [\s\S]+?(?=\n{2,}|\Z)",                                         
    r"Exception(?: in thread.*)?:[\s\S]+?(?=\n{2,}|\Z)",                     
    r"(FAIL|ERROR|E\s)[\s\S]+?(?=\n{2,}|\Z)",                                 
    r"Compilation (error|failed):[\s\S]+?(?=\n{2,}|\Z)",                    
] 
 
def _extract_error_block(text: str) -> str: 
    last = "" 
    for pat in _ERR_PATTERNS: 
        for m in re.finditer(pat, text, re.M): 
            last = m.group(0) 
    if last.strip(): 
        return last.strip() 
    # Fallback: last ~120 lines 
    lines = text.strip().splitlines() 
    return "\n".join(lines[-120:])


def _ensure_global_env_file() -> Path:
    """Create the global .env with the same template as local (if missing)."""
    path = _global_env_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(_env_template_text("global"), encoding="utf-8")
    return path

def _edit_global_env_file() -> None:
    """Open the global .env in an editor and show a diff stat after save."""
    md_path = _ensure_global_env_file()
    original = md_path.read_text(encoding="utf-8")

    launched = _open_in_terminal_editor(md_path)
    edited_text: Optional[str] = None

    if not launched:
        edited_text = click.edit(original, require_save=False)
        if edited_text is None:
            edited_text = _inline_capture_editor(original)
        if edited_text is not None and edited_text != original:
            md_path.write_text(edited_text, encoding="utf-8")
    else:
        edited_text = md_path.read_text(encoding="utf-8")

    if edited_text is None:
        console.print(Panel.fit(Text("No changes saved.", style="yellow"), border_style="yellow"))
        return
    if edited_text == original:
        console.print(Panel.fit(Text("No changes saved (file unchanged).", style="yellow"), border_style="yellow"))
        return

    stats = _diff_stats(original, edited_text)
    console.print(Panel.fit(
        Text.from_markup(
            f"Saved [bold]{md_path}[/bold]\n"
            f"[green]+{stats['added']}[/green] / [red]-{stats['removed']}[/red] • total {stats['total_after']} lines"
        ),
        border_style="green"
    ))

def _load_global_env(*, override_existing: bool = False) -> Dict[str, Any]:
    """Load the global env file into os.environ if it exists."""
    path = _global_env_path()
    applied = _load_env_file(path, override_existing=override_existing) if path.exists() else []
    return {"path": str(path), "found": path.exists(), "applied_keys": applied}

def _global_env_status_label() -> str:
    """Short status for launcher/doctor."""
    p = _global_env_path()
    if p.exists():
        try:
            k = _count_env_keys_in_file(p)
        except Exception:
            k = 0
        try:
            rel = os.path.relpath(str(p), str(Path.cwd()))
        except ValueError:
            # Handle cross-drive paths on Windows
            rel = str(p)
        return f"edit…  ({rel}, {k} keys)"
    return f"create…  ({_global_env_path()})"



def _parse_env_text(text: str) -> Dict[str, str]:
    """
    Minimal .env parser (no external deps).
    Supports:
      - KEY=VALUE (unquoted or single/double quoted)
      - export KEY=VALUE
      - # comments and blank lines
      - \n and \t escapes in values
    """
    env: Dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if (len(val) >= 2) and ((val[0] == val[-1]) and val[0] in ('"', "'")):
            val = val[1:-1]
        val = val.replace("\\n", "\n").replace("\\t", "\t")
        if key:
            env[key] = val
    return env


def _load_env_file(path: Path, override_existing: bool = False) -> List[str]:
    """
    Load env vars from a single file into os.environ.
    Returns list of keys set/updated.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return []
    parsed = _parse_env_text(text)
    applied: List[str] = []
    for k, v in parsed.items():
        if override_existing or (k not in os.environ):
            os.environ[k] = v
            applied.append(k)
    return applied


def _load_env_files(project_dir: Path, *, override_existing: bool = False) -> Dict[str, Any]:
    """
    Load .env + .env.local (if present) from project_dir.
    Precedence: .env then .env.local (later overrides if override_existing=True).
    Returns dict with status info.
    """
    project_dir = project_dir.resolve()
    files = [project_dir / name for name in ENV_FILENAMES]
    existing = [p for p in files if p.exists()]
    applied_keys: List[str] = []
    for p in existing:
        applied_keys.extend(_load_env_file(p, override_existing=override_existing))
    return {
        "project_dir": str(project_dir),
        "files_found": [str(p) for p in existing],
        "applied_keys": applied_keys,
    }


def _count_env_keys_in_file(path: Path) -> int:
    try:
        txt = path.read_text(encoding="utf-8")
    except Exception:
        return 0
    return len(_parse_env_text(txt))


def _env_status_label(project_dir: Path) -> str:
    """
    Build a short status line for the launcher.
    """
    p_env = project_dir / ".env"
    p_local = project_dir / ".env.local"
    have_env = p_env.exists()
    have_local = p_local.exists()
    parts = []
    if have_env:
        parts.append(f".env:{_count_env_keys_in_file(p_env)}")
    if have_local:
        parts.append(f".env.local:{_count_env_keys_in_file(p_local)}")
    if parts:
        return "edit…  (" + ", ".join(parts) + ")"
    return "create…  (.env)"


def _ensure_env_file(project_dir: Path) -> Path:
    """
    Ensure .env exists in project_dir (create with helpful template if missing).
    """
    path = (project_dir / ".env")
    if not path.exists():
        path.write_text(_env_template_text("project"), encoding="utf-8")
    return path

def _edit_env_file(project_dir: Path) -> None:
    """
    Open .env in a terminal editor (Vim-first) or inline capture.
    Show short stats after save.
    """
    md_path = _ensure_env_file(project_dir)
    original = md_path.read_text(encoding="utf-8")

    launched = _open_in_terminal_editor(md_path)
    edited_text: Optional[str] = None

    if not launched:
        edited_text = click.edit(original, require_save=False)
        if edited_text is None:
            edited_text = _inline_capture_editor(original)
        if edited_text is not None and edited_text != original:
            md_path.write_text(edited_text, encoding="utf-8")
    else:
        edited_text = md_path.read_text(encoding="utf-8")

    if edited_text is None:
        console.print(Panel.fit(Text("No changes saved.", style="yellow"), border_style="yellow"))
        return
    if edited_text == original:
        console.print(Panel.fit(Text("No changes saved (file unchanged).", style="yellow"), border_style="yellow"))
        return

    stats = _diff_stats(original, edited_text)
    console.print(Panel.fit(
        Text.from_markup(
            f"Saved [bold]{md_path}[/bold]\n"
            f"[green]+{stats['added']}[/green] / [red]-{stats['removed']}[/red] • total {stats['total_after']} lines"
        ),
        border_style="green"
    ))


def _unwrap_exc(e: BaseException) -> BaseException:
    """Drill down through ExceptionGroup/TaskGroup, __cause__, and __context__ to the root error."""
    seen = set()
    while True:
        # Python 3.11 ExceptionGroup (incl. TaskGroup)
        inner = getattr(e, "exceptions", None)
        if inner:
            e = inner[0]
            continue
        if getattr(e, "__cause__", None) and e.__cause__ not in seen:
            seen.add(e)
            e = e.__cause__
            continue
        if getattr(e, "__context__", None) and e.__context__ not in seen:
            seen.add(e)
            e = e.__context__
            continue
        return e

def _friendly_agent_error(e: BaseException) -> str:
    root = _unwrap_exc(e)
    name = root.__class__.__name__
    msg = (str(root) or "").strip() or "(no details)"
    return (
        "Sorry, a tool run failed. Please try again :)\n\n"
        f"• {name}: {msg}\n\n"
    )


def _bootstrap_env(project_dir: Path, *, interactive_prompt_if_missing: bool = True, announce: bool = False) -> None:
    """
    Load env from project_dir; if none exists, automatically fall back to the global env.
    Never prompt to create a project .env.
    """
    info = _load_env_files(project_dir, override_existing=False)
    if info["files_found"]:
        return

    gpath = _ensure_global_env_file()
    g = _load_global_env(override_existing=False)

    try:
        kcount = _count_env_keys_in_file(gpath)
    except Exception:
        kcount = len(g.get("applied_keys") or [])

    if announce:  
        console.print(Panel.fit(
            Text.from_markup(
                f"Using [bold]global .env[/bold] at [bold]{g['path']}[/bold] "
                f"([green]{kcount} keys[/green])."
            ),
            title="Environment",
            border_style="green",
            box=box.ROUNDED,
        ))

def _agent_cache_get(key: Tuple[str, str, str, str, bool]):
    if key in _AGENT_CACHE:
        _AGENT_CACHE.move_to_end(key)
        return _AGENT_CACHE[key]
    return None

def _agent_cache_put(key: Tuple[str, str, str, str, bool], value: Any) -> None:
    _AGENT_CACHE[key] = value
    _AGENT_CACHE.move_to_end(key)
    while len(_AGENT_CACHE) > _AGENT_CACHE_MAX:
        _AGENT_CACHE.popitem(last=False)


LANGCODE_DIRNAME = ".langcode"
LANGCODE_FILENAME = "langcode.md"
MCP_FILENAME = "mcp.json"

MCP_PROJECT_REL = Path("src") / "langchain_code" / "config" / MCP_FILENAME

def _mcp_target_path(project_dir: Path) -> Path:
    """
    Always prefer the repo MCP at src/langchain_code/config/mcp.json.
    (We’ll mirror to .langcode/mcp.json after saving for backward compatibility.)
    """
    prefer = project_dir / MCP_PROJECT_REL
    # Safety: keep path inside project_dir even on weird symlinks.
    try:
        _ = prefer.resolve().relative_to(project_dir.resolve())
    except Exception:
        return project_dir / LANGCODE_DIRNAME / MCP_FILENAME
    return prefer

def _ensure_mcp_json(project_dir: Path) -> Path:
    """
    Ensure MCP config exists. Prefer src/langchain_code/config/mcp.json.
    """
    mcp_path = _mcp_target_path(project_dir)
    mcp_path.parent.mkdir(parents=True, exist_ok=True)
    if not mcp_path.exists():
        template = {
            "servers": {
                "github": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-github"],
                    "transport": "stdio",
                    "env": {
                        "GITHUB_TOKEN": "$GITHUB_API_KEY",
                        # "GITHUB_TOOLSETS": "repos,issues,pull_requests,actions,code_security"
                    }
                }
            }
        }
        mcp_path.write_text(json.dumps(template, indent=2) + "\n", encoding="utf-8")
    return mcp_path


def _mcp_status_label(project_dir: Path) -> str:
    """
    Show status for MCP config, pointing to src/langchain_code/config/mcp.json (or legacy).
    """
    mcp_path = _mcp_target_path(project_dir)
    rel = os.path.relpath(mcp_path, project_dir)
    if not mcp_path.exists():
        return f"create…  ({rel})"
    try:
        data = json.loads(mcp_path.read_text(encoding="utf-8") or "{}")
        servers = data.get("servers", {}) or {}
        count = len(servers) if isinstance(servers, dict) else 0
        return f"edit…  ({rel}, {count} server{'s' if count != 1 else ''})"
    except Exception:
        return f"edit…  ({rel}, unreadable)"


def _edit_mcp_json(project_dir: Path) -> None:
    """
    Open MCP config in a terminal editor (Vim-first), fall back to click.edit / inline,
    and show a short diff stat after save. Prefers src/langchain_code/config/mcp.json.
    """
    mcp_path = _ensure_mcp_json(project_dir)
    original = mcp_path.read_text(encoding="utf-8")

    launched = _open_in_terminal_editor(mcp_path)
    edited_text: Optional[str] = None

    if not launched:
        edited_text = click.edit(original, require_save=False)
        if edited_text is None:
            edited_text = _inline_capture_editor(original)
        if edited_text is not None and edited_text != original:
            mcp_path.write_text(edited_text, encoding="utf-8")
    else:
        edited_text = mcp_path.read_text(encoding="utf-8")

    if edited_text is None:
        console.print(Panel.fit(Text("No changes saved.", style="yellow"), border_style="yellow"))
        return
    if edited_text == original:
        console.print(Panel.fit(Text("No changes saved (file unchanged).", style="yellow"), border_style="yellow"))
        return

    stats = _diff_stats(original, edited_text)
    console.print(Panel.fit(
        Text.from_markup(
            f"Saved [bold]{mcp_path}[/bold]\n"
            f"[green]+{stats['added']}[/green] / [red]-{stats['removed']}[/red] • total {stats['total_after']} lines"
        ),
        border_style="green"
    ))


def _ensure_langcode_md(project_dir: Path) -> Path:
    """
    Ensure .langcode/langcode.md exists. Return its Path.
    """
    cfg_dir = project_dir / LANGCODE_DIRNAME
    cfg_dir.mkdir(parents=True, exist_ok=True)
    md_path = cfg_dir / LANGCODE_FILENAME
    if not md_path.exists():
        template = f"""# LangCode — Project Custom Instructions

Use this file to add project-specific guidance for the agent.
These notes are appended to the base system prompt in both ReAct and Deep agents.

**Tips**
- Keep it concise and explicit.
- Prefer bullet points and checklists.
- Mention repo conventions, must/shouldn’t rules, style guides, and gotchas.

## Project Rules
- [ ] e.g., All edits must run `pytest -q` and pass.
- [ ] e.g., Use Ruff & Black for Python formatting.

## Code Style & Architecture
- e.g., Follow existing module boundaries in `src/...`

## Tooling & Commands
- e.g., Use `make test` to run the test suite.

---
_Created {datetime.now().strftime('%Y-%m-%d %H:%M')} by LangCode CLI_
"""
        md_path.write_text(template, encoding="utf-8")
    return md_path

def _inline_capture_editor(initial_text: str) -> str:
    """
    Minimal inline editor fallback. Type/paste, then finish with a line: EOF
    (Only used if no terminal editor is available.)
    """
    console.print(Panel.fit(
        Text("Inline editor: Type/paste your content below. End with a line containing only: EOF", style="bold"),
        border_style="cyan",
        title="Inline Editor"
    ))
    if initial_text.strip():
        console.print(Text("---- CURRENT CONTENT (preview) ----", style="dim"))
        console.print(Markdown(initial_text))
        console.print(Text("---- START TYPING (new content will replace file) ----", style="dim"))
    lines: List[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "EOF":
            break
        lines.append(line)
    return "\n".join(lines)

def _pick_terminal_editor() -> Optional[List[str]]:
    """
    Choose a terminal editor command list, prioritizing Vim.
    Order:
      1) $LANGCODE_EDITOR (split on spaces)
      2) $VISUAL
      3) $EDITOR
      4) nvim, vim, vi, nano (first found on PATH)
      5) Windows-specific: check common Vim installation paths
    Returns argv list or None if nothing available.
    """
    if os.environ.get("LANGCODE_EDITOR"):
        return os.environ["LANGCODE_EDITOR"].split()

    for var in ("VISUAL", "EDITOR"):
        v = os.environ.get(var)
        if v:
            return v.split()

    # Check PATH first
    for cand in ("nvim", "vim", "vi", "nano"):
        if shutil.which(cand):
            return [cand]

    # Windows-specific: check common Vim installation paths
    if platform.system().lower() == "windows":
        common_vim_paths = [
            r"C:\Program Files\Vim\vim91\vim.exe",
            r"C:\Program Files\Vim\vim90\vim.exe",
            r"C:\Program Files\Vim\vim82\vim.exe",
            r"C:\Program Files (x86)\Vim\vim91\vim.exe",
            r"C:\Program Files (x86)\Vim\vim90\vim.exe",
            r"C:\Program Files (x86)\Vim\vim82\vim.exe",
            r"C:\tools\vim\vim91\vim.exe",  # Chocolatey
            r"C:\tools\vim\vim90\vim.exe",
            r"C:\Users\{}\scoop\apps\vim\current\vim.exe".format(os.environ.get("USERNAME", "")),  # Scoop
        ]

        for vim_path in common_vim_paths:
            if os.path.exists(vim_path):
                return [vim_path]

        try:
            result = subprocess.run(["where", "vim"], capture_output=True, text=True, check=False)
            if result.returncode == 0 and result.stdout.strip():
                vim_exe = result.stdout.strip().split('\n')[0]
                if os.path.exists(vim_exe):
                    return [vim_exe]
        except Exception:
            pass

    return None


def _open_in_terminal_editor(file_path: Path) -> bool:
    """
    Open the file in a terminal editor and block until it exits.
    Returns True if the editor launched, False otherwise.
    """
    cmd = _pick_terminal_editor()
    if not cmd:
        return False

    try:
        if platform.system().lower() == "windows":
            subprocess.run(cmd + [str(file_path)], check=False)
        else:
            subprocess.run([*cmd, str(file_path)], check=False)
        return True
    except Exception as e:
        console.print(f"[yellow]Failed to launch editor: {e}[/yellow]")
        return False


def _diff_stats(before: str, after: str) -> Dict[str, int]:
    """
    Compute a simple added/removed stat using difflib.ndiff semantics.
    """
    before_lines = before.splitlines()
    after_lines = after.splitlines()
    added = removed = 0
    for line in difflib.ndiff(before_lines, after_lines):
        if line.startswith("+ "):
            added += 1
        elif line.startswith("- "):
            removed += 1
    return {
        "added": added,
        "removed": removed,
        "total_after": len(after_lines),
    }

def _edit_langcode_md(project_dir: Path) -> None:
    """
    Open .langcode/langcode.md in a terminal editor (Vim-first).
    Falls back to $VISUAL/$EDITOR, then click.edit, then inline capture if nothing available.
    After exit, show short stats: lines added/removed and new total lines.
    """
    md_path = _ensure_langcode_md(project_dir)
    original = md_path.read_text(encoding="utf-8")

    launched = _open_in_terminal_editor(md_path)
    edited_text: Optional[str] = None

    if not launched:
        edited_text = click.edit(original, require_save=False)
        if edited_text is None:
            edited_text = _inline_capture_editor(original)
        if edited_text is not None and edited_text != original:
            md_path.write_text(edited_text, encoding="utf-8")
    else:
        edited_text = md_path.read_text(encoding="utf-8")

    if edited_text is None:
        console.print(Panel.fit(Text("No changes saved.", style="yellow"), border_style="yellow"))
        return
    if edited_text == original:
        console.print(Panel.fit(Text("No changes saved (file unchanged).", style="yellow"), border_style="yellow"))
        return

    stats = _diff_stats(original, edited_text)
    console.print(Panel.fit(
        Text.from_markup(
            f"Saved [bold]{md_path}[/bold]\n"
            f"[green]+{stats['added']}[/green] / [red]-{stats['removed']}[/red] • total {stats['total_after']} lines"
        ),
        border_style="green"
    ))

# =========================
# UI helpers
# =========================

def print_langcode_ascii(
    console: Console,
    text: str = "LangCode",
    font: str = "ansi_shadow",
    gradient: str = "dark_to_light",
) -> None:
    def _hex_to_rgb(h):
        h = h.lstrip("#")
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    def _lerp(a, b, t):
        return int(a + (b - a) * t)

    def _interpolate_palette(palette, width):
        if width <= 1:
            return [palette[0]]
        out, steps_total = [], width - 1
        for x in range(width):
            pos = x / steps_total
            seg = min(int(pos * (len(palette) - 1)), len(palette) - 2)
            seg_start, seg_end = seg / (len(palette) - 1), (seg + 1) / (len(palette) - 1)
            local_t = (pos - seg_start) / (seg_end - seg_start + 1e-9)
            c1, c2 = _hex_to_rgb(palette[seg]), _hex_to_rgb(palette[seg + 1])
            rgb = tuple(_lerp(a, b, local_t) for a, b in zip(c1, c2))
            out.append("#{:02x}{:02x}{:02x}".format(*rgb))
        return out

    def _print_block_with_horizontal_gradient(lines, palette):
        width = max(len(line) for line in lines) if lines else 0
        ramp = _interpolate_palette(palette, width)
        for line in lines:
            t = Text()
            padded = line.ljust(width)
            for j, ch in enumerate(padded):
                t.append(ch if ch == " " else ch, Style(color=ramp[j], bold=True))
            console.print(t)

    fig = Figlet(font=font)
    lines = fig.renderText(text).rstrip("\n").splitlines()
    palette = ["#052e1e", "#064e3b", "#065f46", "#047857", "#059669", "#16a34a", "#22c55e", "#34d399"]
    if gradient == "light_to_dark":
        palette = list(reversed(palette))
    _print_block_with_horizontal_gradient(lines, palette)

def session_banner(
    provider: Optional[str],
    project_dir: Path,
    title_text: str,
    *,
    interactive: bool = False,
    apply: bool = False,
    test_cmd: Optional[str] = None,
    tips: Optional[List[str]] = None,
    model_info: Optional[Dict[str, Any]] = None,
    router_enabled: bool = False,
    deep_mode: bool = False,
    command_name: Optional[str] = None,
) -> Panel:
    title = Text(title_text, style="bold magenta")
    body = Text()

    body.append("Provider: ", style="bold")
    if provider and provider.strip() and " " not in provider:
        body.append(provider.upper())
    else:
        body.append((provider or "not set"), style="dim")
    body.append("\n")

    body.append("Project:  ", style="bold")
    body.append(str(project_dir))

    badge = Text()
    if router_enabled:
        badge.append("  [ROUTER ON]", style="bold green")
    if command_name: 
        badge.append(f"  [{command_name}]", style="bold blue") 
    if deep_mode: 
        badge.append("  [DEEP MODE]", style="bold magenta")
    if apply:
        badge.append("  [APPLY MODE]", style="bold red")
    if test_cmd:
        badge.append(f"  tests: {test_cmd}", style="italic")
    if badge:
        body.append("\n")
        body.append_text(badge)

    if model_info:
        body.append("\n")
        model_line = (
            f"Model: {model_info.get('model_name', '(unknown)')} "
            f"[{model_info.get('langchain_model_name', '?')}]"
            f" • priority={model_info.get('priority_used','balanced')}"
        )
        body.append(model_line, style="dim")

    if interactive:
        body.append("\n\n")
        # UPDATED: mention /select
        body.append("Type your request. /clear to redraw, /select to change mode, /exit or /quit to quit. Ctrl+C also exits.\n", style="dim")

    if tips:
        body.append("\n")
        for t in tips:
            body.append(t + "\n", style="dim")

    return Panel(
        body,
        title=title,
        subtitle=Text("ReAct • Deep • Tools • Safe Edits", style="dim"),
        border_style="green",
        padding=(1, 2),
        box=box.HEAVY,
    )

def _pause_if_in_launcher() -> None:
    """If we were launched from the selection hub, wait for Enter before redrawing it."""
    if _IN_SELECTION_HUB:
        console.print(Rule(style="green"))
        console.input("[dim]Press Enter to return to the launcher…[/dim]")

def _print_session_header(
    title: str,
    provider: Optional[str],
    project_dir: Path,
    *,
    interactive: bool = False,
    apply: bool = False,
    test_cmd: Optional[str] = None,
    tips: Optional[List[str]] = None,
    model_info: Optional[Dict[str, Any]] = None,
    router_enabled: bool = False,
    deep_mode: bool = False,
    command_name: Optional[str] = None,
) -> None:
    console.clear()
    print_langcode_ascii(console, text="LangCode", font="ansi_shadow", gradient="dark_to_light")
    console.print(
        session_banner(
            provider,
            project_dir,
            title,
            interactive=interactive,
            apply=apply,
            test_cmd=test_cmd,
            tips=tips,
            model_info=model_info,
            router_enabled=router_enabled,
            deep_mode=deep_mode,
            command_name=command_name
        )
    )
    console.print(Rule(style="green"))

def _looks_like_markdown(text: str) -> bool:
    """Heuristic: decide if the model output is Markdown."""
    if "```" in text:
        return True
    # headings like "#", "##", etc.
    if re.search(r"(?m)^\s{0,3}#{1,6}\s", text):
        return True
    # unordered lists: -, *, +
    if re.search(r"(?m)^\s{0,3}[-*+]\s+", text):
        return True
    # ordered lists: 1. 2. ...
    if re.search(r"(?m)^\s{0,3}\d+\.\s+", text):
        return True
    # inline code or emphasis/bold
    if re.search(r"`[^`]+`", text) or re.search(r"\*\*[^*]+\*\*", text):
        return True
    return False


def _to_text(content: Any) -> str:
    """Coerce Claude-style content blocks (list[dict|str]) into a single string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                t = p.get("text") or p.get("data") or p.get("content")
                if isinstance(t, str):
                    parts.append(t)
            elif isinstance(p, str):
                parts.append(p)
            else:
                parts.append(str(p))
        return "\n".join(parts)
    return str(content)

def _normalize_chat_history_for_anthropic(history: list) -> list:
    """Return a copy of history with str content only (prevents .strip() on lists)."""
    out = []
    for msg in history:
        try:
            c = getattr(msg, "content", "")
            out.append(msg.__class__(content=_to_text(c)))
        except Exception:
            # fall back to best-effort string
            out.append(msg.__class__(content=str(getattr(msg, "content", ""))))
    return out

def _panel_agent_output(text: str, title: str = "Agent", model_label: Optional[str] = None) -> Panel:
    """
    Render agent output full-width, with clean wrapping and proper Markdown
    when appropriate. This avoids the 'half-cut' panel look.
    """
    text = (text or "").rstrip()

    if _looks_like_markdown(text):
        body = Markdown(text)  
    else:
        t = Text.from_ansi(text) if "\x1b[" in text else Text(text)
        t.no_wrap = False
        t.overflow = "fold"
        body = t

    return Panel(
        body,
        title=title,
        border_style="cyan",
        box=box.ROUNDED,
        padding=(1, 2),
        expand=True,   
        subtitle=(Text(f"Model: {model_label}", style="dim") if model_label else None),
        subtitle_align="right", 
    )

def _panel_router_choice(info: Dict[str, Any]) -> Panel:
    if not info:
        body = Text("Router active, but no model info available.", style="dim")
    else:
        name = info.get("model_name", "(unknown)")
        langchain_name = info.get("langchain_model_name", "?")
        provider = info.get("provider", "?").upper()
        priority = info.get("priority_used", "balanced")
        latency = info.get("latency_tier", "?")
        rs = info.get("reasoning_strength", "?")
        ic = info.get("input_cost_per_million", "?")
        oc = info.get("output_cost_per_million", "?")
        ctx = info.get("context_window", "?")
        body = Text.from_markup(
            f"[bold]Router:[/bold] {provider} → [bold]{name}[/bold] [dim]({langchain_name})[/dim]\n"
            f"[dim]priority={priority} • latency_tier={latency} • reasoning={rs}/10 • "
            f"cost=${ic}M in/${oc}M out • ctx={ctx} tokens[/dim]"
        )
    return Panel.fit(body, title="Model Selection", border_style="green", box=box.ROUNDED, padding=(0, 1))

def _show_loader():
    """Spinner that doesn't interfere with interactive prompts (y/n, input, click.confirm)."""
    return console.status("[bold]Processing...[/bold]", spinner="dots", spinner_style="green")


import builtins

_CURRENT_LIVE = None 

class _InputPatch:
    def __init__(self, console: Console, title: str = "Consent"):
        self.console = console
        self.title = title
        self._orig_input = None

    def __enter__(self):
        self._orig_input = builtins.input

        def _rich_input(prompt: str = "") -> str:
            live = globals().get("_CURRENT_LIVE", None)
            cm = live.pause() if getattr(live, "pause", None) else nullcontext()
            with cm:
                body = Text()
                msg = prompt.strip() or "Action requires your confirmation."
                body.append(msg + "\n\n")
                body.append("Type ", style="dim")
                body.append("Y", style="bold green")
                body.append("/", style="dim")
                body.append("N", style="bold red")
                body.append(" and press Enter.", style="dim")

                panel = Panel(
                    body,
                    title=self.title,
                    border_style="yellow",
                    box=box.ROUNDED,
                    padding=(1, 2),
                )
                self.console.print(panel)

                answer = self.console.input("[bold yellow]›[/bold yellow] ").strip()
                return answer

        builtins.input = _rich_input
        return self

    def __exit__(self, exc_type, exc, tb):
        builtins.input = self._orig_input
        return False


def _coerce_sequential_todos(todos: list[dict] | None) -> list[dict]:
    """Ensure visual progression is strictly sequential."""
    todos = list(todos or [])
    blocked = False
    out: list[dict] = []
    for it in todos:
        st = (it.get("status") or "pending").lower().replace("-", "_")
        if blocked and st in {"in_progress", "completed"}:
            st = "pending"
        if st != "completed":
            blocked = True
        out.append({**it, "status": st})
    return out

def _render_todos_panel(todos: list[dict]) -> Panel:
    todos = _coerce_sequential_todos(todos)

    if not todos:
        return Panel(Text("No TODOs yet.", style="dim"), title="TODOs", border_style="blue", box=box.ROUNDED)
    table = Table.grid(padding=(0, 1))
    table.add_column(justify="right", width=3, no_wrap=True)
    table.add_column()

    ICON = {"pending": "○", "in_progress": "◔", "completed": "✓"}
    STYLE = {"pending": "dim", "in_progress": "yellow", "completed": "green"}

    for i, it in enumerate(todos, 1):
        status = (it.get("status") or "pending").lower().replace("-", "_")
        status = status if status in ICON else "pending"
        content = (it.get("content") or "").strip() or "(empty)"
        style = STYLE[status]
        mark = ICON[status]
        text = Text(content, style=style)
        if status == "completed":
            text.stylize("strike")
        table.add_row(f"{i}.", Text.assemble(Text(mark + " ", style=style), text))
    return Panel(table, title="TODOs", border_style="blue", box=box.ROUNDED, padding=(1,1), expand=True)

def _diff_todos(before: list[dict] | None, after: list[dict] | None) -> list[str]:
    before = _coerce_sequential_todos(before or [])
    after = _coerce_sequential_todos(after or [])
    changes: list[str] = []
    for i in range(min(len(before), len(after))):
        b = (before[i].get("status") or "").lower()
        a = (after[i].get("status") or "").lower()
        if b != a:
            content = (after[i].get("content") or before[i].get("content") or "").strip()
            changes.append(f"[{i+1}] {content} → {a}")
    if len(after) > len(before):
        for j in range(len(before), len(after)):
            content = (after[j].get("content") or "").strip()
            changes.append(f"[+ ] {content} (added)")
    if len(before) > len(after):
        for j in range(len(after), len(before)):
            content = (before[j].get("content") or "").strip()
            changes.append(f"[- ] {content} (removed)")
    return changes

def _complete_all_todos(todos: list[dict] | None) -> list[dict]: 
    """ 
    Mark any non-completed TODOs as completed. Invoked right before rendering 
    the final answer so the board reflects finished work the agent may have 
    forgotten to mark as done. 
    """ 
    todos = list(todos or []) 
    out: list[dict] = [] 
    for it in todos: 
        st = (it.get("status") or "pending").lower().replace("-", "_") 
        if st != "completed": 
            it = {**it, "status": "completed"} 
        out.append(it) 
    return out

def _short(s: str, n: int = 280) -> str:
    s = s.replace("\r\n", "\n").strip()
    return s if len(s) <= n else s[:n] + " …"



class _TodoLive(BaseCallbackHandler):
    """Stream TODO updates when planner tools return Command(update={'todos': ...})."""
    def __init__(self, console: Console):
        self.c = console
        self.prev: list[dict] = []

    def on_tool_end(self, output, **kwargs):
        s = str(output)
        if "Command(update=" not in s or "'todos':" not in s:
            return
        m = re.search(r"Command\\(update=(\\{.*\\})\\)$", s, re.S)
        if not m:
            m = re.search(r"update=(\\{.*\\})", s, re.S)
        if not m:
            return
        try:
            data = ast.literal_eval(m.group(1)) 
            todos = data.get("todos")
            if not isinstance(todos, list):
                return
            changes = _diff_todos(self.prev, todos)
            self.prev = todos
            if todos:
                self.c.print(_render_todos_panel(todos))
            if changes:
                self.c.print(Panel(Text("\\n".join(changes)),
                                   title="Progress",
                                   border_style="yellow",
                                   box=box.ROUNDED,
                                   expand=True))
        except Exception:
            pass


class _RichDeepLogs(BaseCallbackHandler):
    """
    Minimal, pretty callback printer for deep (LangGraph) runs.
    Only logs the big milestones so it stays readable.
    Toggle by passing --verbose.
    """
    def __init__(self, console: Console):
        self.c = console

    # Chains/graphs (LangGraph nodes surface as chains)
    def on_chain_start(self, serialized, inputs, **kwargs):
        name = (serialized or {}).get("id") or (serialized or {}).get("name") or "chain"
        self.c.print(Panel.fit(
            Text.from_markup(f"▶ [bold]Start[/bold] {name}\n[dim]{_short(str(inputs))}[/dim]"),
            border_style="cyan", title="Node", box=box.ROUNDED
        ))

    def on_chain_end(self, outputs, **kwargs):
        self.c.print(Panel.fit(
            Text.from_markup(f"[bold]End[/bold]\n[dim]{_short(str(outputs))}[/dim]"),
            border_style="cyan", title="Node", box=box.ROUNDED
        ))

    # Tools
    def on_tool_start(self, serialized, tool_input, **kwargs):
        name = (serialized or {}).get("name") or "tool"
        self.c.print(Panel.fit(
            Text.from_markup(f"[bold]{name}[/bold]\n[dim]{_short(str(tool_input))}[/dim]"),
            border_style="yellow", title="Tool", box=box.ROUNDED
        ))

    def on_tool_end(self, output, **kwargs):
        self.c.print(Panel.fit(
            Text.from_markup(f" [bold]Tool result[/bold]\n{_short(str(output))}"),
            border_style="yellow", title="Tool", box=box.ROUNDED
        ))

    def on_llm_start(self, serialized, prompts, **kwargs):
        name = (serialized or {}).get("id") or (serialized or {}).get("name") or "llm"
        show = "\n---\n".join(_short(p) for p in (prompts or [])[:1])
        self.c.print(Panel.fit(
            Text.from_markup(f"[bold]{name}[/bold]\n{show}"),
            border_style="green", title="LLM", box=box.ROUNDED
        ))

    def on_llm_end(self, response, **kwargs):
        self.c.print(Panel.fit(
            Text("[dim]LLM complete[/dim]"),
            border_style="green", title="LLM", box=box.ROUNDED
        ))

def _maybe_coerce_img_command(raw: str) -> str:
    s = raw.strip()
    if not s.startswith("/img"):
        return raw
    try:
        rest = s[len("/img"):].strip()
        if "::" in rest:
            paths_part, prompt_text = rest.split("::", 1)
            prompt_text = prompt_text.strip()
        else:
            paths_part, prompt_text = rest, ""
        paths = [p for p in paths_part.split() if p]
        return (
            f'Please call the tool "process_multimodal" with '
            f"image_paths={paths} and text={prompt_text!r}. After the tool returns, summarize the result."
        )
    except Exception:
        return raw

MAX_RECOVERY_STEPS = 2

def _extract_last_content(messages: list) -> str:
    if not messages:
        return ""
    last = messages[-1]
    c = getattr(last, "content", None)

    if isinstance(c, str):
        return c.strip()
    if isinstance(c, list):
        parts = []
        for p in c:
            if isinstance(p, dict):
                if 'text' in p and isinstance(p['text'], str):
                    parts.append(p['text'])
                elif p.get('type') == 'text' and isinstance(p.get('data') or p.get('content'), str):
                    parts.append(p.get('data') or p.get('content'))
            elif isinstance(p, str):
                parts.append(p)
        return "\n".join(parts).strip()

    if isinstance(last, dict):
        c = last.get("content", "")
        if isinstance(c, str):
            return c.strip()
        if isinstance(c, list):
            return "\n".join(str(x) for x in c if isinstance(x, str)).strip()

    return (str(c) if c is not None else str(last)).strip()

def _thread_id_for(project_dir: Path, purpose: str = "chat") -> str:
    """Stable thread id per project & purpose for LangGraph checkpointer."""
    return f"{purpose}@{project_dir.resolve()}"

def _resolve_provider(llm_opt: Optional[str], router: bool) -> str:
    if llm_opt:
        return _resolve_provider_base(llm_opt)
    if router:
        return "gemini"
    return _resolve_provider_base(None)

def _build_react_agent_with_optional_llm(provider: str, project_dir: Path, llm=None, **kwargs):
    try:
        if llm is not None:
            return build_react_agent(provider=provider, project_dir=project_dir, llm=llm, **kwargs)
    except TypeError:
        pass
    return build_react_agent(provider=provider, project_dir=project_dir, **kwargs)

def _build_deep_agent_with_optional_llm(provider: str, project_dir: Path, llm=None, **kwargs):
    try:
        if llm is not None:
            return build_deep_agent(provider=provider, project_dir=project_dir, llm=llm, **kwargs)
    except TypeError:
        pass
    return build_deep_agent(provider=provider, project_dir=project_dir, **kwargs)

class _Key:
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    ENTER = "ENTER"
    ESC = "ESC"
    Q = "Q"
    H = "H"
    OTHER = "OTHER"

def _read_key() -> str:
    # Windows
    if msvcrt:
        ch = msvcrt.getch()
        if ch in (b'\x00', b'\xe0'):
            ch2 = msvcrt.getch()
            codes = {b'H': _Key.UP, b'P': _Key.DOWN, b'K': _Key.LEFT, b'M': _Key.RIGHT}
            return codes.get(ch2, _Key.OTHER)
        if ch in (b'\r', b'\n'):
            return _Key.ENTER
        if ch in (b'\x1b',):
            return _Key.ESC
        if ch in (b'q', b'Q'):
            return _Key.Q
        if ch in (b'h', b'H', b'?'):
            return _Key.H
        return _Key.OTHER

    # POSIX
    if not sys.stdin.isatty() or not termios or not tty:
        # Fallback: just press Enter to continue
        return _Key.ENTER

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        seq = os.read(fd, 3)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

    if seq == b'\x1b[A':
        return _Key.UP
    if seq == b'\x1b[B':
        return _Key.DOWN
    if seq == b'\x1b[D':
        return _Key.LEFT
    if seq == b'\x1b[C':
        return _Key.RIGHT
    if seq in (b'\r', b'\n'):
        return _Key.ENTER
    if seq == b'\x1b':
        return _Key.ESC
    if seq in (b'q', b'Q'):
        return _Key.Q
    if seq in (b'h', b'H', b'?'):
        return _Key.H
    return _Key.OTHER

def _render_choice(label: str, value: str, focused: bool, enabled: bool = True) -> Text:
    chev = "▶ " if focused else "  "
    style = "bold white" if focused and enabled else ("dim" if not enabled else "white")
    val_style = "bold cyan" if enabled else "dim"
    t = Text(chev + label + ": ", style=style)
    t.append(str(value), style=val_style)
    return t

def _info_panel_for(field: str) -> Panel:
    info_map = {
        "Command": "[bold]What to do[/bold]: [cyan]chat[/cyan] (interactive), [cyan]feature[/cyan] (plan→edit→verify), [cyan]fix[/cyan] (trace→patch→test), [cyan]analyze[/cyan] (deep code insights).",
        "Engine": "[bold]Reasoning engine[/bold]: [cyan]react[/cyan] = fast tool-using ReAct. [cyan]deep[/cyan] = LangGraph multi-step planner.",
        "Router": "[bold]Router[/bold]: Auto-picks the most efficient LLM per prompt (priority aware). Toggle on for smarter cost/speed/quality.",
        "Priority": "[bold]Routing priority[/bold]: [cyan]balanced[/cyan] | cost | speed | quality – influences model choice.",
        "Autopilot": "[bold]Autopilot[/bold] (deep chat only): Agent plans and executes end-to-end [italic]without asking questions[/italic]. It still uses tools safely, but won't seek confirmation.",
        "Apply": "[bold]Apply[/bold] (feature/fix only): If ON, the agent is allowed to [bold]write files[/bold] and [bold]run commands[/bold]. If OFF, it proposes diffs and commands but does not execute them.",
        "LLM": "[bold]LLM provider[/bold]: Explicitly force [cyan]anthropic[/cyan], [cyan]gemini[/cyan], [cyan]openai[/cyan], or [cyan]ollama[/cyan]. Leave blank to follow router/defaults.",
        "Project": "[bold]Project directory[/bold]: Where the agent operates.",
        "Custom Instructions": "[bold].langcode/langcode.md[/bold]: Project-specific instructions appended to the base prompt. Press Enter to open Vim (or $VISUAL/$EDITOR). Exit with :wq.",
        "MCP Config": "[bold].langcode/mcp.json[/bold]: Configure MCP servers used by the agent/tools. Press Enter to open Vim (or $VISUAL/$EDITOR). Exit with :wq. File is created if missing.",
        "Environment": "[bold].env[/bold]: Create/edit env vars (API keys, config) right here. We'll auto-load from .env / .env.local. If none exists, you can create one inline.",
        "Tests": "[bold]Test command[/bold]: e.g. [cyan]pytest -q[/cyan] or [cyan]npm test[/cyan]. Used by feature/fix flows.",
        "Start": "[bold]Enter[/bold] to launch with the current configuration.",
        "Help": "[bold]h[/bold] to toggle help, [bold]q[/bold] or [bold]Esc[/bold] to quit.",
    }
    text = info_map.get(field, "Use ↑/↓ to move, ←/→ to change values, Enter to start, h for help, q to quit.")
    return Panel(Text.from_markup(text), title="Info", border_style="green", box=box.ROUNDED)

def _list_ollama_models() -> list[str]: 
    try: 
        p = subprocess.run(["ollama", "list", "--format", "json"], capture_output=True, text=True, timeout=2) 
        if p.returncode == 0 and p.stdout.strip(): 
            try: 
                data = json.loads(p.stdout) 
                if isinstance(data, list): 
                    names = [] 
                    for it in data: 
                        n = (it.get("name") or it.get("model") or "").strip() 
                        if n: 
                            names.append(n)
                    return list(dict.fromkeys(names)) 
            except Exception: 
                pass 
        p2 = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=2) 
        if p2.returncode == 0 and p2.stdout: 
            lines = [ln.strip() for ln in p2.stdout.splitlines() if ln.strip()] 
            out = [] 
            for ln in lines[1:]: 
                name = ln.split()[0] 
                if name: 
                    out.append(name)
            return list(dict.fromkeys(out)) 
    except Exception: 
        pass 
    return []

def _provider_model_choices(provider: Optional[str]) -> list[str]: 
    """Return model ids (LangChain names) to pick from for a provider.""" 
    if not provider: 
        return [] 
    if provider == "gemini": 
        return [
            "gemini-2.0-flash-lite", 
            "gemini-2.0-flash", 
            "gemini-2.5-flash-lite", 
            "gemini-2.5-flash", 
            "gemini-2.5-pro", 
        ] 
    if provider == "anthropic": 
        return [ 
            "claude-3-5-haiku-20241022", 
            "claude-3-7-sonnet-20250219", 
            "claude-opus-4-1-20250805", 
        ] 
    if provider == "openai": 
        return [ 
            "gpt-4o-mini", 
            "gpt-4o", 
        ] 
    if provider == "ollama": 
        return _list_ollama_models() 
    return []


def _help_content() -> Panel:
    """
    LangCode-themed, compact help with bright tints on keywords only.
    - Outer frame: green + HEAVY (matches banner)
    - Inner cards: cyan borders
    - Commands / flags / keys: bright shaded colors
    - Descriptions: plain white for readability
    """
    from rich.table import Table
    from rich.columns import Columns

    # --- small helpers -------------------------------------------------------
    def shade(token: str, color: str) -> Text:
        return Text(token, style=f"bold {color}")

    def opts_card(title: str, rows: list[tuple[str, str]], palette: list[str]) -> Panel:
        t = Table.grid(padding=(0, 2))
        t.add_column("Flag", justify="right", no_wrap=True)
        t.add_column("Description", style="white")
        for i, (flag, desc) in enumerate(rows):
            t.add_row(shade(flag, palette[i % len(palette)]), desc)
        return Panel(t, title=title, border_style="cyan", box=box.ROUNDED, padding=(1, 1), expand=True)

    # Palettes stay in the LangCode family (green/teal/cyan/blue).
    p_cmd  = ["#22c55e", "#10b981", "#06b6d4", "#3b82f6", "#67e8f9"]    
    p_glob = ["#7dd3fc", "#22d3ee", "#06b6d4", "#60a5fa", "#34d399"]     
    p_chat = ["#bbf7d0", "#34d399", "#10b981", "#059669"]               
    p_fx   = ["#93c5fd", "#60a5fa", "#3b82f6"]                           
    p_key  = ["#22c55e", "#06b6d4", "#67e8f9", "#34d399", "#60a5fa"]   

    cmd_tbl = Table.grid(padding=(0, 2))
    cmd_tbl.add_column("Command", no_wrap=True)
    cmd_tbl.add_column("What it does", style="white")
    cmd_rows = [
        ("chat",    "Interactive agent chat. --mode {react|deep}. Deep supports --auto."),
        ("feature", "Plan → edit → verify a feature. Use --apply and --test-cmd to run tests."),
        ("fix",     "Diagnose & patch from logs. Accepts --log. Supports --apply and --test-cmd."),
        ("analyze", "Deep repo insights (LangGraph). Great for overviews & architecture questions."),
        ("instr",   "Open or create .langcode/langcode.md (project rules/instructions)."),
    ]
    for i, (name, desc) in enumerate(cmd_rows):
        cmd_tbl.add_row(shade(name, p_cmd[i % len(p_cmd)]), desc)
    cmds = Panel(cmd_tbl, title="Commands", border_style="cyan", box=box.ROUNDED, padding=(1, 1), expand=True)

    global_opts = opts_card("Global options", [
        ("--llm",              "Force provider (anthropic | gemini | openai | ollama)."),
        ("--router",           "Smart model routing per prompt."),
        ("--priority",         "balanced | cost | speed | quality (default: balanced)."),
        ("--verbose",          "Show model-selection panel when routing."),
        ("--project-dir PATH", "Set working directory (default: current)."),
    ], p_glob)

    chat_opts = opts_card("Chat options", [
        ("--mode", "Reasoning engine: react (default) | deep."),
        ("--auto", "Deep-mode autopilot: plan+act without questions."),
    ], p_chat)

    fx_opts = opts_card("Feature / Fix options", [
        ("--apply",                  "Allow edits & run commands (otherwise propose only)."),
        ('--test-cmd "pytest -q"',   "Command to verify changes."),
        ("--log PATH",               "Fix only: path to error log / stack trace."),
    ], p_fx)

    keys_tbl = Table.grid(padding=(0, 2))
    keys_tbl.add_column("Key / Command", justify="right", no_wrap=True)
    keys_tbl.add_column("Action", style="white")
    shortcuts = [
        ("h",            "Toggle this help."),
        ("q / Esc",      "Quit launcher."),
        ("/clear",       "Clear chat screen."),
        ("/select",      "Return to launcher from chat."),
        ("/exit or /quit","Leave chat."),
    ]
    for i, (k, desc) in enumerate(shortcuts):
        keys_tbl.add_row(shade(k, p_key[i % len(p_key)]), desc)
    keys = Panel(keys_tbl, title="Shortcuts", border_style="cyan", box=box.ROUNDED, padding=(1, 1), expand=True)

    cfg_tbl = Table.grid(padding=(0, 2))
    cfg_tbl.add_column(style="white")
    line1 = Text("• Environment keys live in ")
    line1.append(".env", style="bold #06b6d4")
    line1.append(" / ")
    line1.append(".env.local", style="bold #67e8f9")
    line1.append(" (edited from “Environment” in the launcher).")
    line2 = Text("• Project rules live in ")
    line2.append(".langcode/langcode.md", style="bold #22c55e")
    line2.append(" (edited from “Custom Instructions”).")
    cfg_tbl.add_row(line1)
    cfg_tbl.add_row(line2)
    cfg = Panel(cfg_tbl, title="Config & files", border_style="cyan", box=box.ROUNDED, padding=(1, 1), expand=True)

    body = Group(
        Align.center(Text("Quick Reference", style="bold")),
        cmds,
        Rule(style="green"),
        Columns([global_opts, chat_opts, fx_opts], expand=True, equal=True),
        Rule(style="green"),
        Columns([keys, cfg], expand=True, equal=True),
    )

    return Panel(
        body,
        title="LangCode Help",
        border_style="green",
        box=box.HEAVY,
        padding=(1, 2),
        expand=True,
    )

def _draw_launcher(state: Dict[str, Any], focus_index: int, show_help: bool = False) -> None:
    console.clear()
    print_langcode_ascii(console, text="LangCode", font="ansi_shadow", gradient="dark_to_light")

    header = Align.center(Text("ReAct • Deep • Tools • Safe Edits", style="dim"))
    console.print(header)
    console.print(Rule(style="green"))

    rows = []

    cmd_val = state["command"]
    engine_val = state["engine"]
    router_val = "on" if state["router"] else "off"
    prio_val = state["priority"]
    auto_enabled = (state["command"] == "chat" and state["engine"] == "deep")
    autopilot_val = "on" if (state["autopilot"] and auto_enabled) else ("off" if auto_enabled else "n/a")
    apply_enabled = state["command"] in ("feature", "fix")
    apply_val = "on" if (state["apply"] and apply_enabled) else ("off" if apply_enabled else "n/a")
    llm_val = state["llm"] or "(auto)"
    if state["llm"] == "ollama" and state.get("ollama_model") and not state.get("model_override"): 
        model_val = state["ollama_model"]
    else:
        model_val = state.get("model_override", "(default)")

    proj_val = str(state["project_dir"])
    tests_val = state["test_cmd"] or "(none)"
    ollama_names = _list_ollama_models() if state["llm"] == "ollama" else [] 
    start_enabled = True 
    if state["llm"] == "ollama": 
        if not ollama_names: 
            start_enabled = False 
        elif state.get("ollama_model") and state["ollama_model"] not in ollama_names: 
            start_enabled = False
    md_path = (state["project_dir"] / LANGCODE_DIRNAME / LANGCODE_FILENAME)
    if md_path.exists():
        try:
            txt = md_path.read_text(encoding="utf-8")
            lines = txt.count("\n") + (1 if txt and not txt.endswith("\n") else 0)
        except Exception:
            lines = 0
        instr_val = f"edit…  ({LANGCODE_DIRNAME}/{LANGCODE_FILENAME}, {lines} lines)"
    else:
        instr_val = f"create…  ({LANGCODE_DIRNAME}/{LANGCODE_FILENAME})"

    env_val = _env_status_label(state["project_dir"])
    mcp_val = _mcp_status_label(state["project_dir"])
    model_enabled = bool(state["llm"]) and not state["router"]

    labels = [
        ("Command", cmd_val, True),
        ("Engine", engine_val, True),
        ("Router", router_val, True),
        ("Priority", prio_val, state["router"]),
        ("Autopilot", autopilot_val, auto_enabled),
        ("Apply", apply_val, apply_enabled),
        ("LLM", llm_val, True),
        ("Model", model_val, model_enabled),
        ("Project", proj_val, True),
        ("Environment", env_val, True),
        ("Global Environment", _global_env_status_label(), True),
        ("Custom Instructions", instr_val, True),
        ("MCP Config", mcp_val, True),  
        ("Tests", tests_val, state["command"] in ("feature", "fix")),
        ("Start", "Press Enter", start_enabled),
    ]

    for idx, (label, value, enabled) in enumerate(labels):
        rows.append(_render_choice(label, value, focused=(idx == focus_index), enabled=enabled))

    left_panel = Panel(
        Align.left(Text.assemble(*[r + Text("\n") for r in rows])),
        title="Launcher",
        border_style="cyan",
        box=box.ROUNDED,
        padding=(1, 2),
    )

    field_name = labels[focus_index][0] 
    right_panel = _info_panel_for(field_name) 
 
    if field_name == "LLM" and state["llm"] == "ollama": 
        names = _list_ollama_models() 
        body = Text() 
        body.append("Ollama models detected on this machine:\n", style="bold") 
        if names: 
            for i, n in enumerate(names, 1):
                mark = "  " if n != state.get("ollama_model") else "✓ "
                body.append(f"{mark}{i}. {n}\n")
            body.append("\nPress Enter on this field to choose a model.\n", style="dim")
        else: 
            body.append(" (None found)\n", style="yellow") 
            body.append(" Tip: run `ollama pull llama3.1` to get a default model.\n", style="dim") 
            body.append("\n[Start is disabled until a model is installed.]\n", style="dim")

        right_panel = Panel(body, title="Ollama", border_style="cyan", box=box.ROUNDED, padding=(1,1))
    console.print(Group(left_panel, right_panel))

    footer_items = [
        Text("↑/↓ move  ", style="dim"),
        Text("←/→ change  ", style="dim"),
        Text("Enter start  ", style="dim"),
        Text("h for help  ", style="dim"),
        Text("q to quit", style="dim"),
    ]
    console.print(Align.center(Text.assemble(*footer_items)))
    console.print(Rule(style="green"))
    if show_help:
        console.print(_help_content())

def _launcher_loop(initial_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    state = dict(initial_state)
    focus_index = 0
    show_help = False
    _FIELDS_ORDER = (
        "Command", "Engine", "Router", "Priority", "Autopilot", "Apply",
        "LLM", "Model", "Project", "Environment", "Global Environment",
        "Custom Instructions", "MCP Config", "Tests", "Start"
    )
    fields_order = list(_FIELDS_ORDER)

    def toggle(field: str, direction: int) -> None:
        if field == "Command":
            opts = ["chat", "feature", "fix", "analyze"]
            i = (opts.index(state["command"]) + direction) % len(opts)
            state["command"] = opts[i]
        elif field == "Engine":
            opts = ["react", "deep"]
            i = (opts.index(state["engine"]) + direction) % len(opts)
            state["engine"] = opts[i]
        elif field == "Router":
            state["router"] = not state["router"]
        elif field == "Priority":
            if not state["router"]:
                return
            opts = ["balanced", "cost", "speed", "quality"]
            i = (opts.index(state["priority"]) + direction) % len(opts)
            state["priority"] = opts[i]
        elif field == "Autopilot":
            if state["command"] == "chat" and state["engine"] == "deep":
                state["autopilot"] = not state["autopilot"]
        elif field == "Apply":
            if state["command"] in ("feature", "fix"):
                state["apply"] = not state["apply"]
        elif field == "LLM":
            opts = [None, "anthropic", "gemini", "openai", "ollama"]
            cur = state["llm"]
            i = (opts.index(cur) + direction) % len(opts)
            state["llm"] = opts[i]
            state["model_override"] = None
            os.environ.pop("LANGCODE_MODEL_OVERRIDE", None)
        elif field == "Model":
            return 
        elif field == "Project":
            path = console.input("[bold]Project directory[/bold] (Enter to keep current): ").strip()
            if path:
                p = Path(path).expanduser().resolve()
                if p.exists() and p.is_dir():
                    state["project_dir"] = p
                else:
                    console.print(Panel.fit(Text(f"Invalid directory: {p}", style="bold red"), border_style="red"))
                    console.input("Press Enter to continue...")
        elif field == "Environment":
            try:
                _edit_env_file(state["project_dir"])
                _load_env_files(state["project_dir"], override_existing=False)
            except Exception as e:
                console.print(Panel.fit(Text(f"Failed to edit environment: {e}", style="bold red"), border_style="red"))
        
        elif field == "Global Environment":
            try:
                _edit_global_env_file()
                _load_global_env(override_existing=False)
            except Exception as e:
                console.print(Panel.fit(Text(f"Failed to edit global environment: {e}", style="bold red"), border_style="red"))

        elif field == "Custom Instructions":
            try:
                _edit_langcode_md(state["project_dir"])
            except Exception as e:
                console.print(Panel.fit(Text(f"Failed to edit custom instructions: {e}", style="bold red"), border_style="red"))
        elif field == "MCP Config":
            try:
                _edit_mcp_json(state["project_dir"])
            except Exception as e:
                console.print(Panel.fit(Text(f"Failed to edit MCP config: {e}", style="bold red"), border_style="red"))
        elif field == "Tests":
            if state["command"] in ("feature", "fix"):
                t = console.input("[bold]Test command[/bold] (e.g. pytest -q) – empty to clear: ").strip()
                state["test_cmd"] = t or None

    while True:
        _draw_launcher(state, focus_index, show_help=show_help)
        key = _read_key()

        if key in (_Key.Q, _Key.ESC):
            return None
        if key == _Key.H:
            show_help = not show_help
            continue
        if key == _Key.UP:
            focus_index = (focus_index - 1) % len(fields_order)
            continue
        if key == _Key.DOWN:
            focus_index = (focus_index + 1) % len(fields_order)
            continue
        if key == _Key.LEFT:
            toggle(fields_order[focus_index], -1)
            continue
        if key == _Key.RIGHT:
            toggle(fields_order[focus_index], +1)
            continue

        if key == _Key.ENTER:
            field = fields_order[focus_index]

            if field in ("Project", "Environment", "Global Environment", "Custom Instructions", "MCP Config", "Tests"):
                toggle(field, +1)
                continue

            if field == "LLM": 
                if state["llm"] != "ollama": 
                    toggle("LLM", +1)
                    continue
                names = _list_ollama_models()
                if not names:
                    console.print(Panel.fit(Text("No Ollama models detected. Install one (e.g., `ollama pull llama3.1`).", style="bold yellow"), border_style="yellow"))
                    console.input("Press Enter to continue...")
                    continue
                if len(names) == 1:
                    state["ollama_model"] = names[0]
                    os.environ["LANGCODE_OLLAMA_MODEL"] = names[0]
                    console.print(Panel.fit(Text(f"Selected: {names[0]}", style="green"), border_style="green"))
                    console.input("Press Enter to continue...")
                    continue
                console.print(Panel.fit(Text("Select an Ollama model by number:", style="bold"), border_style="cyan"))
                for i, n in enumerate(names, 1):
                    console.print(f"  {i}. {n}")
                choice = console.input("Your choice: ").strip()
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(names):
                        state["ollama_model"] = names[idx]
                        os.environ["LANGCODE_OLLAMA_MODEL"] = names[idx]
                        console.print(Panel.fit(Text(f"Selected: {names[idx]}", style="green"), border_style="green"))
                    else:
                        console.print(Panel.fit(Text("Invalid selection.", style="yellow"), border_style="yellow"))
                except Exception:
                    console.print(Panel.fit(Text("Invalid input.", style="yellow"), border_style="yellow"))
                console.input("Press Enter to continue...")
                continue



            if field == "Model": 
                if not state["llm"] or state["router"]: 
                    continue 
                choices = _provider_model_choices(state["llm"]) 
                if not choices: 
                    console.print(Panel.fit(Text("No selectable models found for this provider.", style="yellow"), 
                                            border_style="yellow")) 
                    console.input("Press Enter to continue...") 
                    continue 
                console.print(Panel.fit(Text("Select a model by number (empty to clear to default):", style="bold"), 
                                        border_style="cyan")) 
                for i, name in enumerate(choices, 1): 
                    mark = "✓ " if name == state.get("model_override") else "  " 
                    console.print(f"  {mark}{i}. {name}") 
                choice = console.input("Your choice: ").strip() 
                if not choice: 
                    state["model_override"] = None 
                    os.environ.pop("LANGCODE_MODEL_OVERRIDE", None) 
                else: 
                    try: 
                        idx = int(choice) - 1 
                        if 0 <= idx < len(choices): 
                            state["model_override"] = choices[idx] 
                            os.environ["LANGCODE_MODEL_OVERRIDE"] = choices[idx] 
                            console.print(Panel.fit(Text(f"Selected: {choices[idx]}", style="green"), 
                                                    border_style="green")) 
                        else: 
                            console.print(Panel.fit(Text("Invalid selection.", style="yellow"), border_style="yellow")) 
                    except Exception: 
                        console.print(Panel.fit(Text("Invalid input.", style="yellow"), border_style="yellow")) 
                console.input("Press Enter to continue...") 
                continue








            if field == "Start":
                if state["llm"] == "ollama" and not _list_ollama_models():
                    console.print(Panel.fit(Text("Cannot start: no Ollama models installed.", style="bold red"), border_style="red"))
                    console.input("Press Enter to continue...")
                    continue
                return state

def _default_state() -> Dict[str, Any]:
    return {
        "command": "chat",
        "engine": "react",
        "router": False,
        "priority": "balanced",
        "autopilot": False,
        "apply": False,  
        "llm": None,    
        "project_dir": Path.cwd(),
        "test_cmd": None,
        "ollama_model": None,
        "model_override": None
    }

def _dispatch_from_state(chosen: Dict[str, Any]) -> Optional[str]:
    """
    Dispatch to the appropriate command based on chosen launcher state.
    Returns:
      - "quit" if the invoked sub-flow indicated a full exit
      - "select" to return to launcher
      - None to simply continue
    """
    if chosen.get("llm") == "ollama" and not _list_ollama_models(): 
        console.print(Panel.fit(Text("Cannot start: no Ollama models installed.", style="bold red"), border_style="red")) 
        return None 
    cmd = chosen["command"]
    if cmd == "chat":
        return chat(
            llm=chosen["llm"],
            project_dir=chosen["project_dir"],
            mode=chosen["engine"],
            auto=bool(chosen["autopilot"] and chosen["engine"] == "deep"),
            router=chosen["router"],
            priority=chosen["priority"],
            verbose=False,
        )
    elif cmd == "feature":
        req = console.input("[bold]Feature request[/bold] (e.g. Add a dark mode toggle): ").strip()
        if not req:
            console.print("[yellow]Aborted: no request provided.[/yellow]")
            return None
        feature(
            request=req,
            llm=chosen["llm"],
            project_dir=chosen["project_dir"],
            test_cmd=chosen["test_cmd"],
            apply=chosen["apply"],
            router=chosen["router"],
            priority=chosen["priority"],
            verbose=False,
        )
        return None
    elif cmd == "fix":
        req = console.input("[bold]Bug summary[/bold] (e.g. Fix crash on image upload) [Enter to skip]: ").strip() or None
        log_path = console.input("[bold]Path to error log[/bold] [Enter to skip]: ").strip()
        log = Path(log_path) if log_path else None
        fix(
            request=req,
            log=log if log and log.exists() else None,
            llm=chosen["llm"],
            project_dir=chosen["project_dir"],
            test_cmd=chosen["test_cmd"],
            apply=chosen["apply"],
            router=chosen["router"],
            priority=chosen["priority"],
            verbose=False,
        )
        return None
    else: 
        req = console.input("[bold]Analysis question[/bold] (e.g. What are the main components?): ").strip()
        if not req:
            console.print("[yellow]Aborted: no question provided.[/yellow]")
            return None
        analyze(
            request=req,
            llm=chosen["llm"],
            project_dir=chosen["project_dir"],
            router=chosen["router"],
            priority=chosen["priority"],
            verbose=False,
        )
        return None

def _selection_hub(initial_state: Optional[Dict[str, Any]] = None) -> None:
    """
    Persistent launcher loop so users can switch modes without restarting the CLI.
    """
    global _IN_SELECTION_HUB
    state = dict(initial_state or _default_state())

    try:
        _bootstrap_env(state["project_dir"], interactive_prompt_if_missing=True)
    except Exception:
        pass

    _IN_SELECTION_HUB = True
    try:
        while True:
            chosen = _launcher_loop(state)
            if not chosen:
                # console.print("\n[bold]Goodbye![/bold]")
                return
            state.update(chosen)
            outcome = _dispatch_from_state(chosen)
            if outcome == "quit":
                console.print("\n[bold]Goodbye![/bold]")
                return
    finally:
        _IN_SELECTION_HUB = False

@app.command(help="Run a command inside a PTY and capture output to a session log (used by `fix --from-tty`).") 
def wrap( 
    cmd: List[str] = typer.Argument(..., help="Command to run (e.g., pytest -q)"), 
    project_dir: Path = typer.Option(Path.cwd(), "--project-dir", exists=True, file_okay=False), 
    tty_id: Optional[str] = typer.Option(None, "--tty-id", help="Override session id (default: auto per TTY)"), 
): 
    log_path = _tty_log_path(tty_id) 
    log_path.parent.mkdir(parents=True, exist_ok=True) 
    console.print(Panel.fit(Text(f"Logging to: {log_path}", style="dim"), title="TTY Capture", border_style="cyan")) 
    os.chdir(project_dir) 
    if platform.system().lower().startswith("win"): 
        with open(log_path, "a", encoding="utf-8", errors="ignore") as f: 
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) 
            assert proc.stdout is not None 
            for line in proc.stdout: 
                sys.stdout.write(line) 
                f.write(line) 
            rc = proc.wait() 
            raise typer.Exit(rc) 
    else: 
        import pty, os as _os 
        with open(log_path, "a", encoding="utf-8", errors="ignore") as f: 
            old_env = dict(_os.environ) 
            _os.environ["LANGCODE_TTY_LOG"] = str(log_path) 
            _os.environ["LANGCODE_TTY_ID"] = tty_id or _current_tty_id() 
            def _tee(master_fd): 
                data = _os.read(master_fd, 1024) 
                if data: 
                    try: 
                        f.write(data.decode("utf-8", "ignore")) 
                        f.flush() 
                    except Exception: 
                        pass 
                return data 
            try: 
                status = pty.spawn(cmd, master_read=_tee) 
            finally: 
                _os.environ.clear(); _os.environ.update(old_env) 
            raise typer.Exit(status >> 8) 
 
@app.command(help="Open a logged subshell. Anything you run here is captured for `fix --from-tty`.") 
def shell( 
    project_dir: Path = typer.Option(Path.cwd(), "--project-dir", exists=True, file_okay=False), 
    tty_id: Optional[str] = typer.Option(None, "--tty-id", help="Override session id (default: auto per TTY)"), 
): 
    sh = os.environ.get("SHELL") if platform.system().lower() != "windows" else os.environ.get("COMSPEC", "cmd.exe") 
    if not sh: 
        sh = "/bin/bash" if platform.system().lower() != "windows" else "cmd.exe" 
    return wrap([sh], project_dir=project_dir, tty_id=tty_id)



@app.command(help="Run environment checks for providers, tools, and MCP.")
def doctor(
    project_dir: Path = typer.Option(Path.cwd(), "--project-dir", exists=True, file_okay=False)
):
    _bootstrap_env(project_dir, interactive_prompt_if_missing=False)

    def yes(x): return Text("✔ " + x, style="green")
    def no(x):  return Text("✖ " + x, style="red")
    rows = []

    rows.append(yes(f"Python {sys.version.split()[0]} on {platform.platform()}"))

    for tool in ["git", "npx", "node", "ollama"]:
        rows.append(yes(f"{tool} found") if shutil.which(tool) else no(f"{tool} missing"))

    provider_keys = {
        "OPENAI_API_KEY": "OpenAI",
        "ANTHROPIC_API_KEY": "Anthropic",
        "GOOGLE_API_KEY": "Gemini",
        "GEMINI_API_KEY": "Gemini (alt)",
        "GROQ_API_KEY": "Groq",
        "TOGETHER_API_KEY": "Together",
        "FIREWORKS_API_KEY": "Fireworks",
        "PERPLEXITY_API_KEY": "Perplexity",
        "DEEPSEEK_API_KEY": "DeepSeek",
        "TAVILY_API_KEY": "Tavily (web search)"
    }
    provider_panel = Table.grid(padding=(0,2))
    provider_panel.add_column("Provider")
    provider_panel.add_column("Status")
    for env, label in provider_keys.items():
        ok = env in os.environ and bool(os.environ.get(env, "").strip())
        provider_panel.add_row(label, ("[green]OK[/green]" if ok else "[red]missing[/red]") + f"  [dim]{env}[/dim]")

    # MCP config
    mcp_path = _mcp_target_path(project_dir)
    mcp_status = "exists" if mcp_path.exists() else "missing"
    mcp_card = Panel(Text(f"{mcp_status}: {os.path.relpath(mcp_path, project_dir)}"), title="MCP", border_style=("green" if mcp_path.exists() else "red"))

    ollama = shutil.which("ollama")
    if ollama:
        models = _list_ollama_models()
        oll_text = ", ".join(models[:6]) + (" …" if len(models) > 6 else "") if models else "(none installed)"
        oll_card = Panel(Text(oll_text), title="Ollama models", border_style=("green" if models else "yellow"))
    else:
        oll_card = Panel(Text("ollama not found"), title="Ollama", border_style="red")

    gpath = _global_env_path()
    gexists = gpath.exists()
    gkeys = _count_env_keys_in_file(gpath) if gexists else 0
    gmsg = f"{'exists' if gexists else 'missing'}: {gpath}\nkeys: {gkeys}"
    global_card = Panel(Text(gmsg), title="Global .env", border_style=("green" if gexists else "red"))

    console.print(Panel(Align.left(Text.assemble(*[r + Text("\n") for r in rows])), title="System", border_style="cyan"))
    console.print(Panel(provider_panel, title="Providers", border_style="cyan"))
    console.print(Columns([mcp_card, oll_card, global_card]))
    console.print(Panel(Text("Tip: run 'langcode instr' to set project rules; edit environment via the launcher."), border_style="blue"))




def _quick_route_free_text(free_text: str, project_dir: Path) -> Optional[str]: 
    """Route free-text to analyze/fix/chat based on simple intent heuristics.""" 
    t = free_text.lower().strip() 
    # Heuristics (intentionally simple; you can swap for an LLM classifier later) 
    if any(k in t for k in ["fix", "error", "traceback", "stack", "crash", "red tests", "failing", "broken"]): 
        # Try to pull recent error from TTY capture (see wrap/shell below) 
        return fix( 
            request=free_text, 
            log=None, 
            project_dir=project_dir, 
            from_tty=True,  # new flag added below 
            router=False, 
            verbose=False, 
        ) 
    if any(k in t for k in ["what's going on", "whats going on", "overview", "summary", "explain the codebase", "architecture"]): 
        analyze( 
            request=free_text, 
            project_dir=project_dir, 
            router=False, 
            verbose=False, 
        ) 
        return None 

    return chat( 
        message=[free_text], 
        project_dir=project_dir, 
        mode="react", 
        router=False, 
        verbose=False, 
        inline = True
    ) 
 
@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    project_dir: Path = typer.Option(Path.cwd(), "--project-dir", exists=True, file_okay=False),
):
    if ctx.invoked_subcommand is not None:
        return
    extras = [a for a in ctx.args if not a.startswith("-")]
    if extras:
        return chat(
            message=extras,
            project_dir=project_dir,
            mode="react",
            router=False,
            verbose=False,
            inline=True,  
        )
    _selection_hub()
    raise typer.Exit()

@app.command(help="Open an interactive chat with the agent. Modes: react | deep (default: react). Use --auto in deep mode for full autopilot (plan+act with no questions).")
def chat(
    message: Optional[List[str]] = typer.Argument(None, help="Optional initial message to send (quotes not required)."),
    llm: Optional[str] = typer.Option(None, "--llm", help="anthropic | gemini | openai | ollama"),
    project_dir: Path = typer.Option(Path.cwd(), "--project-dir", exists=True, file_okay=False),
    mode: str = typer.Option("react", "--mode", help="react | deep"),
    auto: bool = typer.Option(False, "--auto", help="Autonomy mode: plan+act with no questions (deep mode only)."),
    router: bool = typer.Option(False, "--router", help="Auto-route to the most efficient LLM per query."),
    priority: str = typer.Option("balanced", "--priority", help="Router priority: balanced | cost | speed | quality"),
    verbose: bool = typer.Option(False, "--verbose", help="Show model selection panels (and deep logs)."),
    inline: bool = typer.Option(False, "--inline", help="Inline single-turn output (no banners/clear)."),
) -> Optional[str]:
    """
    Returns:
      - "quit": user explicitly exited chat; caller should terminate program
      - "select": user requested to return to launcher
      - None: normal return (caller may continue)
    """
    from rich.live import Live 

    _bootstrap_env(project_dir, interactive_prompt_if_missing=True)

    priority = (priority or "balanced").lower()
    if priority not in {"balanced", "cost", "speed", "quality"}:
        priority = "balanced"

    provider = _resolve_provider(llm, router)
    mode = (mode or "react").lower()
    if mode not in {"react", "deep"}:
        mode = "react"

    input_queue: List[str] = []
    if isinstance(message, list):
        first_msg = " ".join(message).strip()
        if first_msg:
            input_queue.append(first_msg)

    if inline and input_queue:
        first = input_queue.pop(0)
        coerced = _maybe_coerce_img_command(first)
        use_loader = not (mode == "react" and verbose)
        cm = _show_loader() if use_loader else nullcontext()
        with cm:
            model_info = None
            chosen_llm = None
            if router:
                model_info = get_model_info(provider, coerced, priority)
                chosen_llm = get_model(provider, coerced, priority)
                model_key = model_info.get("langchain_model_name") if model_info else "default"
            else:
                env_override = os.getenv("LANGCODE_MODEL_OVERRIDE")
                chosen_llm = get_model_by_name(provider, env_override) if env_override else get_model(provider)
                model_key = env_override or "default"

            cache_key = (
                "deep" if mode == "deep" else "react",
                provider,
                model_key,
                str(project_dir.resolve()),
                bool(auto) if mode == "deep" else False,
            )
            agent = _agent_cache_get(cache_key)
            if agent is None:
                if mode == "deep":
                    seed = AUTO_DEEP_INSTR if auto else None
                    agent = _build_deep_agent_with_optional_llm(
                        provider=provider, project_dir=project_dir, llm=chosen_llm, instruction_seed=seed, apply=auto
                    )
                else:
                    agent = _build_react_agent_with_optional_llm(
                        provider=provider, project_dir=project_dir, llm=chosen_llm
                    )
                _agent_cache_put(cache_key, agent)
            try:
                if mode == "deep":
                    res = agent.invoke(
                        {"messages": [{"role": "user", "content": coerced}]},
                        config={
                            "recursion_limit": 30 if priority in {"speed", "cost"} else 45,
                            "configurable": {"thread_id": _thread_id_for(project_dir, "chat-inline")},
                        },
                    )
                    output = (
                        _extract_last_content(res.get("messages", [])).strip()
                        if isinstance(res, dict) and "messages" in res
                        else str(res)
                    )
                else:
                    payload = {"input": coerced, "chat_history": []}
                    if provider == "anthropic":
                        payload["chat_history"] = _normalize_chat_history_for_anthropic([])
                    if verbose:
                        res = agent.invoke(payload, config={"callbacks": [_RichDeepLogs(console)]})
                    else:
                        res = agent.invoke(payload)

                    output = res.get("output", "") if isinstance(res, dict) else str(res)
                    if provider == "anthropic":
                        output = _to_text(output)
                output = (output or "").strip() or "No response generated."
            except Exception as e:
                output = _friendly_agent_error(e)
        console.print(output)
        return None

    session_title = "LangChain Code Agent • Deep Chat" if mode == "deep" else "LangChain Code Agent • Chat"
    if mode == "deep" and auto:
        session_title += " (Auto)"
    _print_session_header(
        session_title,
        provider,
        project_dir,
        interactive=True,
        router_enabled=router,
        deep_mode=(mode == "deep"),
        command_name="chat",
    )

    history: list = []
    msgs: list = []
    last_todos: list = []
    last_files: dict = {}
    user_turns = 0
    ai_turns = 0
    deep_thread_id = _thread_id_for(project_dir, "chat")
    db_path = project_dir / ".langcode" / "memory.sqlite"

    static_agent = None
    if not router:
        env_override = os.getenv("LANGCODE_MODEL_OVERRIDE")
        chosen_llm = get_model_by_name(provider, env_override) if env_override else get_model(provider)
        if mode == "deep":
            seed = AUTO_DEEP_INSTR if auto else None
            static_agent = _build_deep_agent_with_optional_llm(
                provider=provider, project_dir=project_dir, llm=chosen_llm, instruction_seed=seed, apply=auto
            )
        else:
            static_agent = _build_react_agent_with_optional_llm(
                provider=provider, project_dir=project_dir, llm=chosen_llm
            )

    prio_limits = {"speed": 30, "cost": 35, "balanced": 45, "quality": 60}

    import re as _re, ast as _ast, json as _json
    from langchain_core.callbacks import BaseCallbackHandler

    class _TodoLiveMinimal(BaseCallbackHandler):
        """Updates the single TODO table in-place; no extra logs."""
        def __init__(self, live: Live):
            self.live = live
            self.todos: list[dict] = []
            self.seen = False

        def _extract_todos(self, payload) -> Optional[list]:
            if isinstance(payload, dict):
                if isinstance(payload.get("todos"), list):
                    return payload["todos"]
                upd = payload.get("update")
                if isinstance(upd, dict) and isinstance(upd.get("todos"), list):
                    return upd["todos"]

            upd = getattr(payload, "update", None)
            if isinstance(upd, dict) and isinstance(upd.get("todos"), list):
                return upd["todos"]

            s = str(payload)
            m = _re.search(r"Command\([^)]*update=(\{.*\})\)?$", s, _re.S) or _re.search(r"update=(\{.*\})", s, _re.S)
            if m:
                try:
                    data = _ast.literal_eval(m.group(1))
                    if isinstance(data, dict) and isinstance(data.get("todos"), list):
                        return data["todos"]
                except Exception:
                    pass
            jm = _re.search(r"(\{.*\"todos\"\s*:\s*\[.*\].*\})", s, _re.S)
            if jm:
                try:
                    data = _json.loads(jm.group(1))
                    if isinstance(data.get("todos"), list):
                        return data["todos"]
                except Exception:
                    pass
            return None

        def _render(self, todos: list[dict]):
            todos = _coerce_sequential_todos(todos)
            self.todos = todos
            self.seen = True
            self.live.update(_render_todos_panel(todos))

        def on_tool_end(self, output, **kwargs):
            t = self._extract_todos(output)
            if t is not None:
                self._render(t)

        def on_chain_end(self, outputs, **kwargs):
            t = self._extract_todos(outputs)
            if t is not None:
                self._render(t)

    try:
        while True:
            if input_queue:
                user = input_queue.pop(0)
            else:
                user = console.input(PROMPT).strip()

            if not user:
                continue

            low = user.lower()
            if low in {"cls", "clear", "/clear"}:
                _print_session_header(
                    session_title, provider, project_dir,
                    interactive=True, router_enabled=router,
                    deep_mode=(mode == "deep"), command_name="chat"
                )
                history.clear()
                msgs.clear()
                last_todos = []
                last_files = {}
                continue

            if low in {"select", "/select", "/menu", ":menu"}:
                console.print("[cyan]Returning to launcher…[/cyan]")
                if _IN_SELECTION_HUB:
                    return "select"
                _selection_hub({
                    "command": "chat",
                    "engine": mode,
                    "router": router,
                    "priority": priority,
                    "autopilot": bool(auto),
                    "apply": False,
                    "llm": llm,
                    "project_dir": project_dir,
                    "test_cmd": None,
                })
                return "quit"

            if low in {"exit", "quit", ":q", "/exit", "/quit"}:
                return "quit"

            if low in {"help", "/help", ":help"}:
                _print_session_header(
                    session_title, provider, project_dir, interactive=True,
                    router_enabled=router, deep_mode=(mode == "deep"),
                    command_name="chat"
                )
                console.print(_help_content())
                continue

            if low in {"/memory", "/stats"}:
                if mode != "deep":
                    console.print(Panel.fit(Text("Memory & stats are available in deep mode only.", style="yellow"),
                                            border_style="yellow"))
                    continue
                from rich.table import Table as _Table
                if low == "/memory":
                    t = _Table.grid(padding=(0, 2))
                    t.add_row(Text("Thread", style="bold"), Text(deep_thread_id))
                    t.add_row(Text("DB", style="bold"), Text(str(db_path)))
                    t.add_row(
                        Text("Todos", style="bold"),
                        Text(", ".join(f"[{i+1}] {it.get('content','')}: {it.get('status','pending')}"
                                       for i, it in enumerate(last_todos)) or "(none)")
                    )
                    t.add_row(Text("Files", style="bold"), Text(", ".join(sorted(last_files.keys())) or "(none)"))
                    console.print(Panel(t, title="/memory", border_style="cyan", box=box.ROUNDED))
                else:
                    t = _Table.grid(padding=(0, 2))
                    t.add_row(Text("User turns", style="bold"), Text(str(user_turns)))
                    t.add_row(Text("Agent turns", style="bold"), Text(str(ai_turns)))
                    t.add_row(Text("Messages (current buffer)", style="bold"), Text(str(len(msgs))))
                    t.add_row(Text("Routing", style="bold"),
                              Text(("on • priority=" + priority) if router else "off"))
                    t.add_row(Text("Checkpointer", style="bold"), Text(str(db_path)))
                    t.add_row(Text("Thread", style="bold"), Text(deep_thread_id))
                    console.print(Panel(t, title="/stats", border_style="cyan", box=box.ROUNDED))
                continue

            coerced = _maybe_coerce_img_command(user)
            user_turns += 1

            pending_router_panel: Optional[Panel] = None
            pending_output_panel: Optional[Panel] = None
            react_history_update: Optional[Tuple[HumanMessage, AIMessage]] = None

            agent = static_agent
            model_info = None
            chosen_llm = None

            loader_cm = _show_loader() if (router and not verbose) else nullcontext()
            with loader_cm:
                if router:
                    provider = _resolve_provider(llm, router=True)
                    model_info = get_model_info(provider, coerced, priority)
                    chosen_llm = get_model(provider, coerced, priority)

                    model_key = model_info.get("langchain_model_name") if model_info else "default"
                    cache_key = (
                        "deep" if mode == "deep" else "react",
                        provider,
                        model_key,
                        str(project_dir.resolve()),
                        bool(auto) if mode == "deep" else False,
                    )
                    cached = _agent_cache_get(cache_key)
                    if cached is not None:
                        agent = cached
                    else:
                        if verbose and model_info:
                            pending_router_panel = _panel_router_choice(model_info)
                        if mode == "deep":
                            seed = AUTO_DEEP_INSTR if auto else None
                            agent = _build_deep_agent_with_optional_llm(
                                provider=provider,
                                project_dir=project_dir,
                                llm=chosen_llm,
                                instruction_seed=seed,
                                apply=auto,
                            )
                        else:
                            agent = _build_react_agent_with_optional_llm(
                                provider=provider,
                                project_dir=project_dir,
                                llm=chosen_llm,
                            )
                        _agent_cache_put(cache_key, agent)

            if pending_router_panel:
                console.print(pending_router_panel)

            def _current_model_label() -> Optional[str]:
                if router:
                    if model_info and model_info.get("langchain_model_name"):
                        return model_info["langchain_model_name"]
                    return None
                env_override = os.getenv("LANGCODE_MODEL_OVERRIDE")
                if env_override:
                    return env_override
                try:
                    return get_model_info(provider).get("langchain_model_name")
                except Exception:
                    return None

            _model_label = _current_model_label()


            if mode == "react":
                try:
                    payload = {"input": coerced, "chat_history": history}
                    if provider == "anthropic":
                        payload["chat_history"] = _normalize_chat_history_for_anthropic(payload["chat_history"])

                    if verbose:
                        res = agent.invoke(payload, config={"callbacks": [_RichDeepLogs(console)]})
                    else:
                        with _show_loader():
                            res = agent.invoke(payload)

                    output = res.get("output", "") if isinstance(res, dict) else str(res)
                    if provider == "anthropic":
                        output = _to_text(output)
                    if not output.strip():
                        steps = res.get("intermediate_steps") if isinstance(res, dict) else None
                        if steps:
                            previews = []
                            for pair in steps[-3:]:
                                try:
                                    previews.append(str(pair))
                                except Exception:
                                    continue
                            output = "Model returned empty output. Recent steps:\n" + "\n".join(previews)
                        else:
                            output = "No response generated. Try rephrasing your request."
                except Exception as e:
                    output = _friendly_agent_error(e)

                pending_output_panel = _panel_agent_output(output, model_label=_model_label)
                react_history_update = (HumanMessage(content=coerced), AIMessage(content=output))
                ai_turns += 1
            else:
                msgs.append({"role": "user", "content": coerced})
                if auto:
                    msgs.append({
                        "role": "system",
                        "content": (
                            "AUTOPILOT: Start now. Discover files (glob/list_dir/grep), read targets (read_file), "
                            "perform edits (edit_by_diff/write_file), and run at least one run_cmd (git/tests) "
                            "capturing stdout/stderr + exit code. Then produce one 'FINAL:' report and STOP. No questions."
                        )
                    })

                deep_config: Dict[str, Any] = {
                    "recursion_limit": prio_limits.get(priority, 100),
                    "configurable": {"thread_id": deep_thread_id},
                }

                placeholder = Panel(
                    Text("Planning tasks…", style="dim"),
                    title="TODOs",
                    border_style="blue",
                    box=box.ROUNDED,
                    padding=(1, 1),
                    expand=True
                )

                output: str = ""

                with Live(placeholder, refresh_per_second=8, transient=True) as live:
                    todo_cb = _TodoLiveMinimal(live)
                    deep_config["callbacks"] = [todo_cb]
                    res = {}
                    try:
                        res = agent.invoke({"messages": msgs}, config=deep_config)
                        if isinstance(res, dict) and "messages" in res:
                            msgs = res["messages"]
                            last_files = res.get("files") or last_files
                            last_content = _extract_last_content(msgs).strip()
                        else:
                            last_content = ""
                            res = res if isinstance(res, dict) else {}

                    except Exception as e:
                        last_content = ""
                        output = (f"Agent hit recursion limit. Last response: {_extract_last_content(msgs)}"
                                  if "recursion" in str(e).lower()
                                  else f"Agent error: {e}")
                        res = {}
                    if not output:
                        if not last_content:
                            # one safety retry to force a response
                            msgs.append({
                                "role": "system",
                                "content": "You must provide a response. Use your tools to complete the request and give a clear answer."
                            })
                            try:
                                res2 = agent.invoke({"messages": msgs}, config=deep_config)
                                if isinstance(res2, dict) and "messages" in res2:
                                    msgs = res2["messages"]
                                last_content = _extract_last_content(msgs).strip()
                            except Exception as e:
                                last_content = f"Agent failed after retry: {e}"
                        output = last_content or "No response generated."

                    final_todos = res.get("todos") if isinstance(res, dict) else None
                    if not isinstance(final_todos, list) or not final_todos:
                        final_todos = getattr(todo_cb, "todos", [])

                    if final_todos:
                        if output and output.strip():
                            final_todos = _complete_all_todos(final_todos)
                        live.update(_render_todos_panel(final_todos))
                        last_todos = final_todos
                    else:
                        live.update(Panel(
                            Text("No tasks were emitted by the agent.", style="dim"),
                            title="TODOs",
                            border_style="blue",
                            box=box.ROUNDED,
                            padding=(1, 1),
                            expand=True
                        ))

                pending_output_panel = _panel_agent_output(output, model_label=_model_label)
                ai_turns += 1

            if pending_output_panel:
                console.print(pending_output_panel)

            if react_history_update:
                human_msg, ai_msg = react_history_update
                history.append(human_msg)
                history.append(ai_msg)
                if len(history) > 20:
                    history[:] = history[-20:]
    except (KeyboardInterrupt, EOFError):
        return "quit"


@app.command(help="Implement a feature end-to-end (plan → search → edit → verify). Supports --apply and optional --test-cmd (e.g., 'pytest -q').")
def feature(
    request: str = typer.Argument(..., help='e.g. "Add a dark mode toggle in settings"'),
    llm: Optional[str] = typer.Option(None, "--llm", help="anthropic | gemini | openai | ollama"),
    project_dir: Path = typer.Option(Path.cwd(), "--project-dir", exists=True, file_okay=False),
    test_cmd: Optional[str] = typer.Option(None, "--test-cmd", help='e.g. "pytest -q" or "npm test"'),
    apply: bool = typer.Option(False, "--apply", help="Apply writes and run commands without interactive confirm."),
    router: bool = typer.Option(False, "--router", help="Auto-route to the most efficient LLM for this request."),
    priority: str = typer.Option("balanced", "--priority", help="balanced | cost | speed | quality"),
    verbose: bool = typer.Option(False, "--verbose", help="Show model selection panel."),
):
    _bootstrap_env(project_dir, interactive_prompt_if_missing=True)

    priority = (priority or "balanced").lower()
    if priority not in {"balanced", "cost", "speed", "quality"}:
        priority = "balanced"

    provider = _resolve_provider(llm, router)
    model_info = None
    chosen_llm = None

    if router:
        model_info = get_model_info(provider, request, priority)
        chosen_llm = get_model(provider, request, priority)

    _print_session_header(
        "LangChain Code Agent • Feature",
        provider,
        project_dir,
        interactive=False,
        apply=apply,
        test_cmd=test_cmd,
        model_info=(model_info if (router and verbose) else None),
        router_enabled=router,
        
    )
    if router and verbose and model_info:
        console.print(_panel_router_choice(model_info))

    model_key = (model_info or {}).get("langchain_model_name", "default")
    cache_key = ("react", provider, model_key, str(project_dir.resolve()), False)
    cached = _agent_cache_get(cache_key)
    if not router and provider in {"openai", "ollama"}:
        chosen_llm = get_model(provider)
    if cached is None:
        agent = _build_react_agent_with_optional_llm(
            provider=provider,
            project_dir=project_dir,
            llm=chosen_llm,
            apply=apply,
            test_cmd=test_cmd,
            instruction_seed=FEATURE_INSTR,
        )
        _agent_cache_put(cache_key, agent)
    else:
        agent = cached

    with _show_loader():
        res = agent.invoke({"input": request, "chat_history": []})
        output = res.get("output", "") if isinstance(res, dict) else str(res)
    console.print(_panel_agent_output(output, title="Feature Result"))
    _pause_if_in_launcher()
@app.command(help="Diagnose & fix a bug (trace → pinpoint → patch → test). Accepts --log, --test-cmd, and supports --apply.")
def fix(
    request: Optional[str] = typer.Argument(None, help='e.g. "Fix crash on image upload"'),
    log: Optional[Path] = typer.Option(None, "--log", exists=True, help="Path to error log or stack trace."),
    llm: Optional[str] = typer.Option(None, "--llm", help="anthropic | gemini | openai | ollama"),
    project_dir: Path = typer.Option(Path.cwd(), "--project-dir", exists=True, file_okay=False),
    test_cmd: Optional[str] = typer.Option(None, "--test-cmd", help='e.g. "pytest -q"'),
    apply: bool = typer.Option(False, "--apply", help="Apply writes and run commands without interactive confirm."),
    router: bool = typer.Option(False, "--router", help="Auto-route to the most efficient LLM for this request."),
    priority: str = typer.Option("balanced", "--priority", help="balanced | cost | speed | quality"),
    verbose: bool = typer.Option(False, "--verbose", help="Show model selection panel."),
    from_tty: bool = typer.Option(False, "--from-tty", help="Use most recent output from the current logged terminal session (run your command via `langcode wrap ...` or `langcode shell`)."),
    tty_id: Optional[str] = typer.Option(None, "--tty-id", help="Which session to read; defaults to current TTY."),
 ):
    _bootstrap_env(project_dir, interactive_prompt_if_missing=True)

    priority = (priority or "balanced").lower()
    if priority not in {"balanced", "cost", "speed", "quality"}:
        priority = "balanced"

    provider = _resolve_provider(llm, router)

    bug_input = (request or "").strip() 
    if log: 
        bug_input += "\n\n--- ERROR LOG ---\n" + Path(log).read_text(encoding="utf-8", errors="ignore") 
    elif from_tty: 
        tlog = os.environ.get("LANGCODE_TTY_LOG") or str(_tty_log_path(tty_id)) 
        p = Path(tlog) 
        if p.exists(): 
            recent = _tail_bytes(p) 
            block = _extract_error_block(recent).strip() 
            if block: 
                bug_input += "\n\n--- ERROR LOG (from TTY) ---\n" + block 
                console.print(Panel.fit(Text(f"Using error from session log: {p}", style="dim"), border_style="cyan")) 
        else: 
            console.print(Panel.fit(Text("No TTY session log found. Run your failing command via `langcode wrap <cmd>` or `langcode shell`.", style="yellow"), border_style="yellow"))
    bug_input = bug_input.strip() or "Fix the bug using the provided log."

    model_info = None
    chosen_llm = None
    if router:
        model_info = get_model_info(provider, bug_input, priority)
        chosen_llm = get_model(provider, bug_input, priority)

    _print_session_header(
        "LangChain Code Agent • Fix",
        provider,
        project_dir,
        interactive=False,
        apply=apply,
        test_cmd=test_cmd,
        model_info=(model_info if (router and verbose) else None),
        router_enabled=router,
    )
    if router and verbose and model_info:
        console.print(_panel_router_choice(model_info))

    model_key = (model_info or {}).get("langchain_model_name", "default")
    cache_key = ("react", provider, model_key, str(project_dir.resolve()), False)
    cached = _agent_cache_get(cache_key)
    if not router and provider in {"openai", "ollama"}:
        chosen_llm = get_model(provider)
    if cached is None:
        agent = _build_react_agent_with_optional_llm(
            provider=provider,
            project_dir=project_dir,
            llm=chosen_llm,
            apply=apply,
            test_cmd=test_cmd,
            instruction_seed=BUGFIX_INSTR,
        )
        _agent_cache_put(cache_key, agent)
    else:
        agent = cached

    with _show_loader():
        res = agent.invoke({"input": bug_input, "chat_history": []})
        output = res.get("output", "") if isinstance(res, dict) else str(res)
    console.print(_panel_agent_output(output, title="Fix Result"))
    _pause_if_in_launcher()
@app.command(help="Analyze any codebase and generate insights (deep agent).")
def analyze(
    request: str = typer.Argument(..., help='e.g. "What are the main components of this project?"'),
    llm: Optional[str] = typer.Option(None, "--llm", help="anthropic | gemini | openai | ollama"),
    project_dir: Path = typer.Option(Path.cwd(), "--project-dir", exists=True, file_okay=False),
    router: bool = typer.Option(False, "--router", help="Auto-route to the most efficient LLM for this request."),
    priority: str = typer.Option("balanced", "--priority", help="balanced | cost | speed | quality"),
    verbose: bool = typer.Option(False, "--verbose", help="Show model selection panel."),
):
    _bootstrap_env(project_dir, interactive_prompt_if_missing=True)

    priority = (priority or "balanced").lower()
    if priority not in {"balanced", "cost", "speed", "quality"}:
        priority = "balanced"

    provider = _resolve_provider(llm, router)

    model_info = None
    chosen_llm = None
    if router:
        model_info = get_model_info(provider, request, priority)
        chosen_llm = get_model(provider, request, priority)

    _print_session_header(
        "LangChain Code Agent • Analyze",
        provider,
        project_dir,
        interactive=False,
        apply=False,
        model_info=(model_info if (router and verbose) else None),
        router_enabled=router,
    )
    if router and verbose and model_info:
        console.print(_panel_router_choice(model_info))

    model_key = (model_info or {}).get("langchain_model_name", "default")
    cache_key = ("deep", provider, model_key, str(project_dir.resolve()), False)
    cached = _agent_cache_get(cache_key)
    if not router and provider in {"openai", "ollama"}:
        chosen_llm = get_model(provider)
    if cached is None:
        agent = _build_deep_agent_with_optional_llm(
            provider=provider,
            project_dir=project_dir,
            llm=chosen_llm,
            apply=False,
        )
        _agent_cache_put(cache_key, agent)
    else:
        agent = cached

    with _show_loader():
        output = ""
        try:
            res = agent.invoke(
                {"messages": [{"role": "user", "content": request}]},
                config={
                    "recursion_limit": 45,
                    "configurable": {"thread_id": _thread_id_for(project_dir, "analyze")},
                },
            )
            output = (
                _extract_last_content(res.get("messages", [])).strip()
                if isinstance(res, dict) and "messages" in res
                else str(res)
            )
        except Exception as e:
            output = f"Analyze error: {e}"
    console.print(_panel_agent_output(output or "No response generated.", title="Analysis Result"))
    _pause_if_in_launcher()


@app.command(help="Edit environment. Use --global to edit your global env (~/.config/langcode/.env or $LANGCODE_GLOBAL_ENV).")
def env(
    global_: bool = typer.Option(False, "--global", "-g", help="Edit the global env file."),
    project_dir: Path = typer.Option(Path.cwd(), "--project-dir", exists=True, file_okay=False)
):
    if global_:
        _print_session_header("LangCode • Global Environment", provider=None, project_dir=project_dir, interactive=False)
        _edit_global_env_file()
        _load_global_env(override_existing=True)
        console.print(Panel.fit(Text("Global environment loaded.", style="green"), border_style="green"))
    else:
        _print_session_header("LangCode • Project Environment", provider=None, project_dir=project_dir, interactive=False)
        _edit_env_file(project_dir)
        _load_env_files(project_dir, override_existing=False)
        console.print(Panel.fit(Text("Project environment loaded.", style="green"), border_style="green"))




@app.command(name="instr", help="Open or create project-specific instructions (.langcode/langcode.md) in your editor.")
def edit_instructions(
    project_dir: Path = typer.Option(Path.cwd(), "--project-dir", exists=True, file_okay=False)
):
    _print_session_header(
        "LangChain Code Agent • Custom Instructions",
        provider=None,
        project_dir=project_dir,
        interactive=False
    )
    _edit_langcode_md(project_dir)

def main() -> None:
    app()

if __name__ == "__main__":
    main()