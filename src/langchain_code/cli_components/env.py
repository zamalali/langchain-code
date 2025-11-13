from __future__ import annotations

import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re

import click
from rich import box
from rich.panel import Panel
from rich.text import Text

from .constants import (
    ENV_FILENAMES,
    GLOBAL_ENV_ENVVAR,
    LANGCODE_CONFIG_DIR_ENVVAR,
    MCP_FILENAME,
    LANGCODE_DIRNAME,
)
from .editors import diff_stats, inline_capture_editor, open_in_terminal_editor
from .state import console, edit_feedback_enabled


def env_template_text(scope: str = "project") -> str:
    """Return the standard .env template for project/global scopes."""
    title = ".env - environment for LangCode" if scope == "project" else "Global .env - environment for LangCode"
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


def user_config_dir() -> Path:
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


def global_env_path() -> Path:
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


SESSIONS_DIR = user_config_dir() / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def current_tty_id() -> str:
    import sys

    try:
        return os.path.basename(os.ttyname(sys.stdin.fileno())).replace("/", "_")
    except Exception:
        return f"ppid{os.getppid()}_pid{os.getpid()}"


def tty_log_path(tty_id: Optional[str] = None) -> Path:
    name = (tty_id or os.environ.get("LANGCODE_TTY_ID") or current_tty_id()).strip() or "default"
    return SESSIONS_DIR / f"{name}.log"


def tail_bytes(path: Path, max_bytes: int = 200_000) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            return f.read().decode("utf-8", "ignore")
    except Exception:
        return ""


ERR_PATTERNS = [
    r"Traceback \(most recent call last\):[\s\S]+?(?=\n{2,}|\Z)",
    r"(?:UnhandledPromiseRejection|Error:|TypeError:|ReferenceError:)[\s\S]+?(?=\n{2,}|\Z)",
    r"panic: [\s\S]+?(?=\n{2,}|\Z)",
    r"Exception(?: in thread.*)?:[\s\S]+?(?=\n{2,}|\Z)",
    r"(FAIL|ERROR|E\s)[\s\S]+?(?=\n{2,}|\Z)",
    r"Compilation (error|failed):[\s\S]+?(?=\n{2,}|\Z)",
]


def extract_error_block(text: str) -> str:
    last = ""
    for pat in ERR_PATTERNS:
        for match in re.finditer(pat, text, re.M):
            last = match.group(0)
    if last.strip():
        return last.strip()
    lines = text.strip().splitlines()
    return "\n".join(lines[-120:])


def ensure_global_env_file() -> Path:
    """Create the global .env with the same template as local (if missing)."""
    path = global_env_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(env_template_text("global"), encoding="utf-8")
    return path


def edit_global_env_file() -> None:
    """Open the global .env in an editor and show a diff stat after save."""
    md_path = ensure_global_env_file()
    original = md_path.read_text(encoding="utf-8")

    launched = open_in_terminal_editor(md_path)
    edited_text: Optional[str] = None

    if not launched:
        edited_text = click.edit(original, require_save=False)
        if edited_text is None:
            edited_text = inline_capture_editor(original)
        if edited_text is not None and edited_text != original:
            md_path.write_text(edited_text, encoding="utf-8")
    else:
        edited_text = md_path.read_text(encoding="utf-8")

    if edited_text is None:
        if edit_feedback_enabled():
            console.print(Panel.fit(Text("No changes saved.", style="yellow"), border_style="yellow"))
        return
    if edited_text == original:
        if edit_feedback_enabled():
            console.print(Panel.fit(Text("No changes saved (file unchanged).", style="yellow"), border_style="yellow"))
        return

    stats = diff_stats(original, edited_text)
    if edit_feedback_enabled():
        console.print(Panel.fit(
            Text.from_markup(
                f"Saved [bold]{md_path}[/bold]\n"
                f"[green]+{stats['added']}[/green] / [red]-{stats['removed']}[/red] - total {stats['total_after']} lines"
            ),
            border_style="green"
        ))


def parse_env_text(text: str) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    buffered_key = None
    buffered_value: List[str] = []

    def _flush_buffer():
        nonlocal buffered_key, buffered_value
        if buffered_key is not None:
            parsed[buffered_key] = "\n".join(buffered_value)
            buffered_key = None
            buffered_value = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if buffered_key is not None:
            if raw_line.endswith("\\"):
                buffered_value.append(raw_line[:-1])
                continue
            buffered_value.append(raw_line)
            _flush_buffer()
            continue

        if "=" in line:
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if value.endswith("\\"):
                buffered_key = key
                buffered_value = [value[:-1]]
                continue
            parsed[key] = value

    _flush_buffer()
    return parsed


def load_env_file(path: Path, *, override_existing: bool = False) -> List[str]:
    """
    Load env vars from a single file into os.environ.
    Returns list of keys set/updated.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return []
    parsed = parse_env_text(text)
    applied: List[str] = []
    for key, value in parsed.items():
        if override_existing or (key not in os.environ):
            os.environ[key] = value
            applied.append(key)
    return applied


def load_env_files(project_dir: Path, *, override_existing: bool = False) -> Dict[str, Any]:
    """
    Load .env files from project_dir (precedence: .env.local -> .env).
    Returns dict describing files found and keys applied.
    """
    files = [project_dir / name for name in ENV_FILENAMES]
    found = [p for p in files if p.exists()]
    applied: Dict[str, Any] = {"files_found": [str(p) for p in found], "applied_keys": []}
    for path in reversed(found):
        keys = load_env_file(path, override_existing=override_existing)
        if keys:
            applied["applied_keys"].extend(keys)
    return applied


def ensure_env_file(project_dir: Path) -> Path:
    """
    Ensure .env exists in project_dir (create with helpful template if missing).
    """
    path = project_dir / ".env"
    if not path.exists():
        path.write_text(env_template_text("project"), encoding="utf-8")
    return path


def count_env_keys_in_file(path: Path) -> int:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return 0
    return len(parse_env_text(text))


def env_status_label(project_dir: Path) -> str:
    """
    Build a short status line for the launcher.
    """
    p_env = project_dir / ".env"
    p_local = project_dir / ".env.local"
    have_env = p_env.exists()
    have_local = p_local.exists()
    parts = []
    if have_env:
        parts.append(f".env:{count_env_keys_in_file(p_env)}")
    if have_local:
        parts.append(f".env.local:{count_env_keys_in_file(p_local)}")
    if parts:
        return "edit-  (" + ", ".join(parts) + ")"
    return "create-  (.env)"


def edit_env_file(project_dir: Path) -> None:
    """
    Open .env in a terminal editor (Vim-first) or inline capture.
    Show short stats after save.
    """
    md_path = ensure_env_file(project_dir)
    original = md_path.read_text(encoding="utf-8")

    launched = open_in_terminal_editor(md_path)
    edited_text: Optional[str] = None

    if not launched:
        edited_text = click.edit(original, require_save=False)
        if edited_text is None:
            edited_text = inline_capture_editor(original)
        if edited_text is not None and edited_text != original:
            md_path.write_text(edited_text, encoding="utf-8")
    else:
        edited_text = md_path.read_text(encoding="utf-8")

    if edited_text is None:
        if edit_feedback_enabled():
            console.print(Panel.fit(Text("No changes saved.", style="yellow"), border_style="yellow"))
        return
    if edited_text == original:
        if edit_feedback_enabled():
            console.print(Panel.fit(Text("No changes saved (file unchanged).", style="yellow"), border_style="yellow"))
        return

    stats = diff_stats(original, edited_text)
    if edit_feedback_enabled():
        console.print(Panel.fit(
            Text.from_markup(
                f"Saved [bold]{md_path}[/bold]\n"
                f"[green]+{stats['added']}[/green] / [red]-{stats['removed']}[/red] - total {stats['total_after']} lines"
            ),
            border_style="green"
        ))


def load_global_env(*, override_existing: bool = False) -> Dict[str, Any]:
    """Load the global env file into os.environ if it exists."""
    path = global_env_path()
    applied = load_env_file(path, override_existing=override_existing) if path.exists() else []
    return {"path": str(path), "applied_keys": applied}


def global_env_status_label() -> str:
    """Short status for launcher/doctor."""
    path = global_env_path()
    if path.exists():
        try:
            count = count_env_keys_in_file(path)
        except Exception:
            count = 0
        rel = os.path.relpath(str(path), str(Path.cwd()))
        return f"edit-  ({rel}, {count} keys)"
    return f"create-  ({global_env_path()})"


def bootstrap_env(project_dir: Path, *, interactive_prompt_if_missing: bool = True, announce: bool = False) -> None:
    """
    Load env from project_dir; if none exists, automatically fall back to the global env.
    Never prompt to create a project .env.
    """
    info = load_env_files(project_dir, override_existing=False)
    if info["files_found"]:
        return

    gpath = ensure_global_env_file()
    g = load_global_env(override_existing=False)

    try:
        key_count = count_env_keys_in_file(gpath)
    except Exception:
        key_count = len(g.get("applied_keys") or [])

    if announce:
        console.print(Panel.fit(
            Text.from_markup(
                f"Using [bold]global .env[/bold] at [bold]{g['path']}[/bold] "
                f"([green]{key_count} keys[/green])."
            ),
            title="Environment",
            border_style="green",
            box=box.ROUNDED,
        ))
