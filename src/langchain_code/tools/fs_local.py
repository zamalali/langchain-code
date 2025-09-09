from __future__ import annotations
from pathlib import Path
from difflib import unified_diff
from langchain_core.tools import tool
from ..hooks import get_hook_runner


def _rooted(project_dir: str, path: str) -> Path:
    """
    Resolve `path` inside the project root and refuse path traversal.
    """
    p = Path(project_dir).joinpath(path).resolve()
    root = Path(project_dir).resolve()
    if not str(p).startswith(str(root)):
        raise ValueError("Path escapes project_dir")
    return p


def _clip(s: str, n: int = 24000) -> str:
    """Trim long strings for readable tool outputs."""
    return s if len(s) <= n else s[:n] + "\n...[truncated]..."


def _should_apply(apply: bool, safety: str, is_mutation: bool) -> tuple[bool, str | None]:
    """
    Enforce write/exec safety gates.

    safety:
      - 'auto'    : allow non-mutations only (block mutations unless apply=True)
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


def _with_hooks(base: str, *hook_results) -> str:
    """Append any hook outputs as a `[hooks]` section."""
    chunks = []
    for hr in hook_results:
        if hr and getattr(hr, "outputs", None):
            chunks.extend([o for o in hr.outputs if o])
    if not chunks:
        return base
    return f"{base}\n\n[hooks]\n" + "\n\n".join(chunks)


def make_list_dir_tool(project_dir: str):
    @tool(
        "list_dir",
        description="List the contents of a directory relative to the project root. Directories end with '/'.",
        return_direct=False,
    )
    def list_dir(path: str = ".") -> str:
        """
        List a directory (relative to project root).

        Args:
            path: Directory path (default '.').
        Returns:
            One item per line, with directories suffixed by '/'.
        """
        p = _rooted(project_dir, path)
        if not p.exists():
            return f"{path} not found."
        if not p.is_dir():
            return f"{path} is not a directory."
        items = []
        for child in sorted(p.iterdir()):
            suffix = "/" if child.is_dir() else ""
            items.append(str(Path(path) / (child.name + suffix)))
        return "\n".join(items)

    return list_dir


def make_read_file_tool(project_dir: str):
    @tool(
        "read_file",
        description="Read and return the full text content of a file relative to the project root.",
        return_direct=False,
    )
    def read_file(path: str) -> str:
        """
        Read a text file.

        Args:
            path: File path (relative to project root).
        Returns:
            Entire file content as UTF-8 text or an error message.
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
    @tool(
        "write_file",
        description="Create or overwrite a file with given content. Respects --apply and fires pre/post hooks.",
        return_direct=False,
    )
    def write_file(
        path: str,
        content: str,
        *,
        safety: str = "require",
        report: str = "diff",
    ) -> str:
        """
        Write a file (create/overwrite).

        Args:
            path: Target file path (relative to project root).
            content: New file contents (UTF-8).
            safety: 'auto' | 'require' | 'force' (see safety policy).
            report: 'diff' to show unified diff, or 'summary'.

        Returns:
            Operation summary (and diff if requested). Appends a [hooks] section if hooks ran.
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

        diff = "\n".join(
            unified_diff(old.splitlines(), content.splitlines(), fromfile=f"a/{path}", tofile=f"b/{path}", lineterm="")
        )

        runner = get_hook_runner(project_dir)
        pre = runner.fire(
            "pre_write_file",
            {"path": path, "diff": diff, "size": len(content), "apply": bool(apply), "mutating": True},
        )
        if not pre.allowed:
            return _with_hooks(f"write {path} blocked.\n{pre.message}", pre)

        allowed, msg = _should_apply(apply, safety, is_mutation=True)
        if not allowed:
            hdr = f"dry-run (apply={apply}): write {path}"
            if report == "summary":
                base = f"{hdr}\nChange size: {len(content)} chars\nReason: {msg}"
            else:
                base = f"{hdr}\nDiff:\n{_clip(diff)}\nReason: {msg}"
            return _with_hooks(base, pre)

        try:
            p.write_text(content, encoding="utf-8")
            if report == "summary":
                base = f"Wrote {len(content)} chars to {path}."
            else:
                base = f"Wrote {len(content)} chars to {path}.\nDiff:\n{_clip(diff)}"
            post = runner.fire(
                "post_write_file",
                {"path": path, "diff": diff, "size": len(content), "apply": bool(apply), "mutating": True},
            )
            return _with_hooks(base, pre, post)
        except Exception as e:
            return _with_hooks(f"Error writing {path}: {type(e).__name__}: {e}", pre)

    return write_file


def make_edit_by_diff_tool(project_dir: str, apply: bool):
    @tool(
        "edit_by_diff",
        description="Replace an exact snippet in a file with a new snippet (single, safe micro-edit). Fires pre/post hooks.",
        return_direct=False,
    )
    def edit_by_diff(
        path: str,
        original_snippet: str,
        replaced_snippet: str,
        *,
        safety: str = "require",
    ) -> str:
        """
        Edit a file by replacing an exact snippet once.

        Args:
            path: File path (relative to project root).
            original_snippet: Exact text to find.
            replaced_snippet: Replacement text.
            safety: 'auto' | 'require' | 'force'.

        Returns:
            Operation summary with unified diff, plus [hooks] if any ran.
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
        diff = "\n".join(
            unified_diff(text.splitlines(), new_text.splitlines(), fromfile=f"a/{path}", tofile=f"b/{path}", lineterm="")
        )

        runner = get_hook_runner(project_dir)
        pre = runner.fire(
            "pre_edit_by_diff", {"path": path, "diff": diff, "apply": bool(apply), "mutating": True}
        )
        if not pre.allowed:
            return _with_hooks(f"edit {path} blocked.\n{pre.message}", pre)

        allowed, msg = _should_apply(apply, safety, is_mutation=True)
        if not allowed:
            base = f"dry-run (apply={apply}): edit {path}\nDiff:\n{_clip(diff)}\nReason: {msg}"
            return _with_hooks(base, pre)

        try:
            p.write_text(new_text, encoding="utf-8")
            base = f"Applied 1 edit to {path}.\nDiff:\n{_clip(diff)}"
            post = runner.fire(
                "post_edit_by_diff", {"path": path, "diff": diff, "apply": bool(apply), "mutating": True}
            )
            return _with_hooks(base, pre, post)
        except Exception as e:
            return _with_hooks(f"Error writing {path}: {type(e).__name__}: {e}", pre)

    return edit_by_diff


def make_delete_file_tool(project_dir: str, apply: bool):
    @tool(
        "delete_file",
        description="Delete a file. Respects --apply and fires pre/post hooks.",
        return_direct=False,
    )
    def delete_file(path: str, *, safety: str = "require") -> str:
        """
        Delete a file.

        Args:
            path: File path (relative to project root).
            safety: 'auto' | 'require' | 'force'.

        Returns:
            Operation summary, plus [hooks] if any ran.
        """
        p = _rooted(project_dir, path)
        if not p.exists():
            return f"{path} not found."
        if not p.is_file():
            return f"{path} is not a file."

        runner = get_hook_runner(project_dir)
        pre = runner.fire("pre_delete_file", {"path": path, "apply": bool(apply), "mutating": True})
        if not pre.allowed:
            return _with_hooks(f"delete {path} blocked.\n{pre.message}", pre)

        allowed, msg = _should_apply(apply, safety, is_mutation=True)
        if not allowed:
            return _with_hooks(f"dry-run (apply={apply}): delete {path}\nReason: {msg}", pre)

        try:
            p.unlink()
            base = f"Deleted {path}."
            post = runner.fire("post_delete_file", {"path": path, "apply": bool(apply), "mutating": True})
            return _with_hooks(base, pre, post)
        except Exception as e:
            return _with_hooks(f"Error deleting {path}: {type(e).__name__}: {e}", pre)

    return delete_file
