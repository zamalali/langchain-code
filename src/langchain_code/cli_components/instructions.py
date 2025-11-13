from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import click
from rich.panel import Panel
from rich.text import Text

from .constants import LANGCODE_DIRNAME, LANGCODE_FILENAME
from .editors import diff_stats, inline_capture_editor, open_in_terminal_editor
from .state import console, edit_feedback_enabled


def ensure_langcode_md(project_dir: Path) -> Path:
    """
    Ensure .langcode/langcode.md exists. Return its Path.
    """
    cfg_dir = project_dir / LANGCODE_DIRNAME
    cfg_dir.mkdir(parents=True, exist_ok=True)
    md_path = cfg_dir / LANGCODE_FILENAME
    if not md_path.exists():
        template = f"""# LangCode - Project Custom Instructions

Use this file to add project-specific guidance for the agent.
These notes are appended to the base system prompt in both ReAct and Deep agents.

**Tips**
- Keep it concise and explicit.
- Prefer bullet points and checklists.
- Mention repo conventions, must/shouldn't rules, style guides, and gotchas.

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


def edit_langcode_md(project_dir: Path) -> None:
    """
    Open .langcode/langcode.md in a terminal editor (Vim-first).
    Falls back to $VISUAL/$EDITOR, then click.edit, then inline capture if nothing available.
    After exit, show short stats: lines added/removed and new total lines.
    """
    md_path = ensure_langcode_md(project_dir)
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
