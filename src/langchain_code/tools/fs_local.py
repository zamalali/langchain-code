from __future__ import annotations
from pathlib import Path
from difflib import unified_diff
from langchain_core.tools import tool


def _rooted(project_dir: str, path: str) -> Path:
    p = Path(project_dir).joinpath(path).resolve()
    root = Path(project_dir).resolve()
    if not str(p).startswith(str(root)):
        raise ValueError("Path escapes project_dir")
    return p

def _clip(s: str, n: int = 24000) -> str:
    return s if len(s) <= n else s[:n] + "\n...[truncated]..."

def _should_apply(apply: bool, safety: str, is_mutation: bool) -> tuple[bool, str | None]:
    """
    Decide if a mutating operation is allowed.

    safety:
      - 'auto'    : allow only if not a mutation (always mutation here) → block when apply=False
      - 'require' : require apply=True for any mutation
      - 'force'   : bypass apply and run anyway
    """
    safety = (safety or "auto").strip().lower()
    if safety not in {"auto", "require", "force"}:
        safety = "auto"

    if safety == "force":
        return True, None
    if not is_mutation:
        return True, None
    if safety == "require" and not apply:
        return False, "Execution requires apply=True (explicit consent). Re-run with --apply."
    if safety == "auto" and not apply:
        return False, ("Declined without apply: this is a file modification. "
                       "Re-run with --apply, or set safety='force' if you intend to write.")
    return True, None


def make_list_dir_tool(project_dir: str):
    @tool("list_dir", return_direct=False)
    def list_dir(path: str = ".") -> str:
        """
        LIST DIRECTORY — CONTRACT FOR THE AGENT

        Purpose
        -------
        List the contents of a **directory** (relative to project root).

        Use when
        --------
        - You want to explore folders to discover files/subfolders.
        - Examples: `list_dir()` → project root, `list_dir("src/")`, `list_dir(".langcode/")`.

        Do NOT use when
        ---------------
        - The path is a **file** (e.g., ends with `.py`, `.md`, `.json`, etc.) → use `read_file`.
        - You are unsure if it’s a file or directory → use `glob` first (e.g., `glob("src/**/*.py")`)
          or call `list_dir` on the **parent directory** instead (e.g., `list_dir("src/")`).

        Return shape
        ------------
        - One item per line.
        - Directories end with a trailing `/`.
        - Paths are relative to the project root.

        Failure guidance
        ----------------
        - If you get "not a directory", **switch tools**: use `read_file` for files or `glob` for discovery.
        - Do not call `list_dir` again on the same file path.

        Examples
        --------
        GOOD: `list_dir("src/")`
        BAD : `list_dir("src/langchain_code/cli.py")`  # this is a file; use read_file
        """
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
        """
        Read the full contents of a file.

        Use when you need to inspect or modify a file:
          - `read_file("src/config.py")`
          - `read_file("README.md")`
        """
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
    def write_file(
        path: str,
        content: str,
        *,
        safety: str = "require",   # 'auto' | 'require' | 'force'
        report: str = "diff",      # 'diff' | 'summary'
    ) -> str:
        """
        Overwrite or create a file with new content.

        Default behavior requires apply=True for writes. You can override with safety='force'.

        Examples:
          - `write_file("Work/hello.py", "print('hi')")`
          - `write_file("Work/hello.py", "...", safety="force")`  # apply even if agent apply=False
        """
        report = (report or "diff").strip().lower()
        if report not in {"diff", "summary"}:
            report = "diff"

        p = _rooted(project_dir, path)
        p.parent.mkdir(parents=True, exist_ok=True)

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

        allowed, msg = _should_apply(apply, safety, is_mutation=True)
        if not allowed:
            hdr = f"dry-run (apply={apply}): write {path}"
            if report == "summary":
                return f"{hdr}\nChange size: {len(content)} chars\nReason: {msg}"
            return f"{hdr}\nDiff:\n{_clip(diff)}\nReason: {msg}"

        try:
            p.write_text(content, encoding="utf-8")
            if report == "summary":
                return f"Wrote {len(content)} chars to {path}."
            return f"Wrote {len(content)} chars to {path}.\nDiff:\n{_clip(diff)}"
        except Exception as e:
            return f"Error writing {path}: {type(e).__name__}: {e}"
    return write_file

def make_edit_by_diff_tool(project_dir: str, apply: bool):
    @tool("edit_by_diff", return_direct=False)
    def edit_by_diff(
        path: str,
        original_snippet: str,
        replaced_snippet: str,
        *,
        safety: str = "require",   # 'auto' | 'require' | 'force'
    ) -> str:
        """
        Edit a file by replacing an exact snippet with a new snippet (safe micro-edit).
        """
        p = _rooted(project_dir, path)
        if not p.exists() or not p.is_file():
            return f"{path} not found or not a file."
        try:
            text = p.read_text(encoding="utf-8")
        except Exception as e:
            return f"Error reading {path}: {e}"

        if original_snippet not in text:
            return f"Original snippet not found in {path}."

        new_text = text.replace(original_snippet, replaced_snippet, 1)
        diff = "\n".join(unified_diff(
            text.splitlines(), new_text.splitlines(),
            fromfile=f"a/{path}", tofile=f"b/{path}", lineterm=""
        ))

        allowed, msg = _should_apply(apply, safety, is_mutation=True)
        if not allowed:
            return f"dry-run (apply={apply}): edit {path}\nDiff:\n{_clip(diff)}\nReason: {msg}"

        try:
            p.write_text(new_text, encoding="utf-8")
            return f"Applied 1 edit to {path}.\nDiff:\n{_clip(diff)}"
        except Exception as e:
            return f"Error writing {path}: {type(e).__name__}: {e}"
    return edit_by_diff

def make_delete_file_tool(project_dir: str, apply: bool):
    @tool("delete_file", return_direct=False)
    def delete_file(path: str, *, safety: str = "require") -> str:
        """
        Delete a file. Mutating: requires apply=True unless safety='force'.
        """
        p = _rooted(project_dir, path)
        if not p.exists():
            return f"{path} not found."
        if not p.is_file():
            return f"{path} is not a file."

        allowed, msg = _should_apply(apply, safety, is_mutation=True)
        if not allowed:
            return f"dry-run (apply={apply}): delete {path}\nReason: {msg}"

        try:
            p.unlink()
            return f"Deleted {path}."
        except Exception as e:
            return f"Error deleting {path}: {type(e).__name__}: {e}"
    return delete_file