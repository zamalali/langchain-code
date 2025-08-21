from __future__ import annotations
from pathlib import Path
from difflib import unified_diff
from langchain_core.tools import tool
from ..safety.confirm import confirm_action
import shutil

def _rooted(project_dir: str, path: str) -> Path:
    p = Path(project_dir).joinpath(path).resolve()
    if not str(p).startswith(str(Path(project_dir).resolve())):
        raise ValueError("Path escapes project_dir")
    return p

def make_list_dir_tool(project_dir: str):
    @tool("list_dir", return_direct=False)
    def list_dir(path: str = ".") -> str:
        """List files and directories at a path."""
        p = _rooted(project_dir, path)
        if not p.exists():
            return f"{path} not found."
        items = []
        for child in sorted(p.iterdir()):
            suffix = "/" if child.is_dir() else ""
            items.append(str(Path(path) / (child.name + suffix)))
        return "\n".join(items)
    return list_dir

def make_read_file_tool(project_dir: str):
    @tool("read_file", return_direct=False)
    def read_file(path: str) -> str:
        """Read a file and return its content."""
        p = _rooted(project_dir, path)
        if not p.exists() or not p.is_file():
            return f"{path} not found or not a file."
        try:
            return p.read_text(encoding="utf-8")
        except Exception as e:
            return f"Error reading {path}: {e}"
    return read_file

def make_write_file_tool(project_dir: str, apply: bool):
    @tool("write_file", return_direct=False)
    def write_file(path: str, content: str) -> str:
        """Overwrite a file with new content. Shows a diff and asks confirmation."""
        p = _rooted(project_dir, path)
        old = ""
        if p.exists():
            try:
                old = p.read_text(encoding="utf-8")
            except Exception:
                old = ""
        diff = "\n".join(unified_diff(
            old.splitlines(), content.splitlines(),
            fromfile=f"a/{path}", tofile=f"b/{path}", lineterm=""
        ))
        if not confirm_action(f"Apply write to {path}? (diff will be printed by the agent)", apply):
            return f"Write to {path} cancelled.\nDiff:\n{diff}"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} chars to {path}."
    return write_file

def make_edit_by_diff_tool(project_dir: str, apply: bool):
    @tool("edit_by_diff", return_direct=False)
    def edit_by_diff(path: str, original_snippet: str, replaced_snippet: str) -> str:
        """
        Edit a file by replacing an exact snippet with a new snippet (safe micro-edit).
        Returns a unified diff and applies after confirmation.
        """
        p = _rooted(project_dir, path)
        if not p.exists() or not p.is_file():
            return f"{path} not found or not a file."
        text = p.read_text(encoding="utf-8")
        if original_snippet not in text:
            return f"Original snippet not found in {path}."
        new_text = text.replace(original_snippet, replaced_snippet, 1)
        diff = "\n".join(unified_diff(
            text.splitlines(), new_text.splitlines(),
            fromfile=f"a/{path}", tofile=f"b/{path}", lineterm=""
        ))
        if not confirm_action(f"Apply edit to {path}? (see diff)", apply):
            return f"Edit cancelled for {path}.\nDiff:\n{diff}"
        p.write_text(new_text, encoding="utf-8")
        return f"Applied 1 edit to {path}.\nDiff:\n{diff}"
    return edit_by_diff

def make_delete_path_tool(project_dir: str, apply: bool):
    @tool("delete_path", return_direct=False)
    def delete_path(path: str) -> str:
        """
        Delete a file or directory (recursive).
        Prefer `git rm <path>` for tracked files when appropriate; this tool is for
        untracked paths or when you explicitly want to remove from the working tree.
        Shows what it will delete and asks confirmation unless apply=True.
        """
        p = _rooted(project_dir, path)
        if not p.exists():
            return f"{path} not found."
        if not confirm_action(f"Delete {path}? (recursive for directories)", apply):
            return f"Delete cancelled for {path}."
        try:
            if p.is_dir():
                shutil.rmtree(p)
                return f"Deleted directory {path}"
            else:
                p.unlink()
                return f"Deleted file {path}"
        except Exception as e:
            return f"Error deleting {path}: {e}"
    return delete_path