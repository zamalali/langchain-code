from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import click
from rich.panel import Panel
from rich.text import Text

from .constants import LANGCODE_DIRNAME, MCP_FILENAME, MCP_PROJECT_REL
from .editors import diff_stats, inline_capture_editor, open_in_terminal_editor
from .state import console, edit_feedback_enabled


def mcp_target_path(project_dir: Path) -> Path:
    """
    Always prefer the repo MCP at src/langchain_code/config/mcp.json.
    (We’ll mirror to .langcode/mcp.json after saving for backward compatibility.)
    """
    prefer = project_dir / MCP_PROJECT_REL
    try:
        _ = prefer.resolve().relative_to(project_dir.resolve())
    except Exception:
        return project_dir / LANGCODE_DIRNAME / MCP_FILENAME
    return prefer


def ensure_mcp_json(project_dir: Path) -> Path:
    """
    Ensure MCP config exists. Prefer src/langchain_code/config/mcp.json.
    """
    mcp_path = mcp_target_path(project_dir)
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


def mcp_status_label(project_dir: Path) -> str:
    """
    Show status for MCP config, pointing to src/langchain_code/config/mcp.json (or legacy).
    """
    mcp_path = mcp_target_path(project_dir)
    rel = os.path.relpath(mcp_path, project_dir)
    if not mcp_path.exists():
        return f"create-  ({rel})"
    try:
        data = json.loads(mcp_path.read_text(encoding="utf-8") or "{}")
        servers = data.get("servers", {}) or {}
        count = len(servers) if isinstance(servers, dict) else 0
        return f"edit-  ({rel}, {count} server{'s' if count != 1 else ''})"
    except Exception:
        return f"edit-  ({rel}, unreadable)"


def edit_mcp_json(project_dir: Path) -> None:
    """
    Open MCP config in a terminal editor (Vim-first), fall back to click.edit / inline,
    and show a short diff stat after save. Prefers src/langchain_code/config/mcp.json.
    """
    mcp_path = ensure_mcp_json(project_dir)
    original = mcp_path.read_text(encoding="utf-8")

    launched = open_in_terminal_editor(mcp_path)
    edited_text: Optional[str] = None

    if not launched:
        edited_text = click.edit(original, require_save=False)
        if edited_text is None:
            edited_text = inline_capture_editor(original)
        if edited_text is not None and edited_text != original:
            mcp_path.write_text(edited_text, encoding="utf-8")
    else:
        edited_text = mcp_path.read_text(encoding="utf-8")

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
                f"Saved [bold]{mcp_path}[/bold]\n"
                f"[green]+{stats['added']}[/green] / [red]-{stats['removed']}[/red] - total {stats['total_after']} lines"
            ),
            border_style="green"
        ))

