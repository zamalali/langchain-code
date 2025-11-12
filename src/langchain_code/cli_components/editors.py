from __future__ import annotations

import os
import platform
import shutil
import subprocess
import difflib
from pathlib import Path
from typing import Dict, List, Optional

from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from .state import console


def inline_capture_editor(initial_text: str) -> str:
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


def pick_terminal_editor() -> Optional[List[str]]:
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


def open_in_terminal_editor(file_path: Path) -> bool:
    """
    Open the file in a terminal editor and block until it exits.
    Returns True if the editor launched, False otherwise.
    """
    cmd = pick_terminal_editor()
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


def diff_stats(before: str, after: str) -> Dict[str, int]:
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

