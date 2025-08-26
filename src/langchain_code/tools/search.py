# src/langchain_code/tools/search.py
from __future__ import annotations
from pathlib import Path
import fnmatch
import os
import re
import time
from typing import Iterable, List, Optional
from langchain_core.tools import tool

# --- Tunables / sane defaults ---
DEFAULT_EXCLUDE_DIRS = {
    ".git", ".hg", ".svn",
    ".venv", "venv", "env",
    "node_modules",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".tox", ".cache",
    "dist", "build", "target",
    ".idea", ".vscode", ".gradle",
}
DEFAULT_MAX_RESULTS = 500            # max paths returned by glob
DEFAULT_MAX_MATCHES = 500            # max lines returned by grep
DEFAULT_MAX_FILES_SCANNED = 5000     # cap on files walked
DEFAULT_MAX_BYTES_PER_FILE = 2_000_000  # ~2MB: skip very large files
DEFAULT_TIME_BUDGET_SEC = 8.0        # soft time budget to keep ReAct snappy

def _iter_files(
    root: Path,
    exclude_dirs: set[str],
    max_files: int,
    time_budget_sec: float,
) -> Iterable[Path]:
    """Iteratively walk files under root:
       - prunes excluded directories
       - does NOT follow symlinked directories
       - stops on time/file caps
    """
    start = time.time()
    stack = [root]
    seen = 0

    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for entry in it:
                    # time budget
                    if (time.time() - start) > max(0.1, time_budget_sec):
                        return
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            name = entry.name
                            if name in exclude_dirs:
                                continue
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            yield Path(entry.path)
                            seen += 1
                            if seen >= max(1, max_files):
                                return
                    except PermissionError:
                        continue
        except (FileNotFoundError, PermissionError):
            continue

def _relposix(root: Path, p: Path) -> str:
    try:
        return p.relative_to(root).as_posix()
    except Exception:
        return str(p.as_posix())

def _compile_regex(pattern: str, ignore_case: bool) -> re.Pattern:
    flags = re.IGNORECASE if ignore_case else 0
    try:
        return re.compile(pattern, flags)
    except re.error:
        # Fallback to literal search if the pattern is invalid
        return re.compile(re.escape(pattern), flags)

def make_glob_tool(project_dir: str):
    @tool("glob", return_direct=False)
    def glob(
        pattern: str,
        *,
        exclude_dirs: Optional[List[str]] = None,
        max_results: int = DEFAULT_MAX_RESULTS,
        max_files_scanned: int = DEFAULT_MAX_FILES_SCANNED,
        time_budget_sec: float = DEFAULT_TIME_BUDGET_SEC,
    ) -> str:
        """
        Find files in the project by glob pattern.

        Examples:
          - "**/*.py" → all Python files
          - "src/**/*.json" → JSON under src/
          - "*config*" → any file with "config" in its name

        Optional:
          - exclude_dirs: extra folder names to skip (in addition to defaults)
          - max_results: cap returned paths
          - max_files_scanned: cap files walked before stopping
          - time_budget_sec: soft ceiling to prevent long scans
        """
        root = Path(project_dir).resolve()
        excludes = set(DEFAULT_EXCLUDE_DIRS)
        if exclude_dirs:
            excludes.update(exclude_dirs)

        results: List[str] = []
        scanned = 0
        timed_out = False
        start = time.time()

        for f in _iter_files(root, excludes, max_files_scanned, time_budget_sec):
            scanned += 1
            rel = _relposix(root, f)
            if fnmatch.fnmatch(rel, pattern):
                results.append(rel)
                if len(results) >= max_results:
                    break

        if (time.time() - start) > time_budget_sec:
            timed_out = True

        if not results:
            return "(no matches)"
        out = "\n".join(results[:max_results])
        if timed_out or scanned >= max_files_scanned:
            out += f"\n[note] truncated: scanned≈{scanned}, returned={len(results)}"
        return out
    return glob

def make_grep_tool(project_dir: str):
    @tool("grep", return_direct=False)
    def grep(
        pattern: str,
        path: str = ".",
        *,
        ignore_case: bool = False,
        exclude_dirs: Optional[List[str]] = None,
        max_matches: int = DEFAULT_MAX_MATCHES,
        max_files_scanned: int = DEFAULT_MAX_FILES_SCANNED,
        max_bytes_per_file: int = DEFAULT_MAX_BYTES_PER_FILE,
        time_budget_sec: float = DEFAULT_TIME_BUDGET_SEC,
    ) -> str:
        """
        Search for a regex (or literal if invalid) inside files under a directory.

        Returns lines in the form:
          <file>:<line number>:<matched text>

        Optional:
          - ignore_case: case-insensitive search
          - exclude_dirs: extra folder names to skip (defaults already skip .venv, node_modules, .git, etc.)
          - max_matches: cap on total matches returned
          - max_files_scanned: cap files walked before stopping
          - max_bytes_per_file: skip very large files quickly
          - time_budget_sec: soft ceiling to prevent long scans
        """
        root = Path(project_dir).resolve().joinpath(path).resolve()
        proj_root = Path(project_dir).resolve()

        if not root.exists():
            return f"{path} not found."
        if not str(root).startswith(str(proj_root)):
            return f"{path} escapes project root."

        excludes = set(DEFAULT_EXCLUDE_DIRS)
        if exclude_dirs:
            excludes.update(exclude_dirs)

        rx = _compile_regex(pattern, ignore_case)

        matches: List[str] = []
        scanned = 0
        timed_out = False
        start = time.time()

        for f in _iter_files(root, excludes, max_files_scanned, time_budget_sec):
            scanned += 1

            # Skip very large files outright
            try:
                if f.stat().st_size > max_bytes_per_file:
                    continue
            except Exception:
                continue

            rel = _relposix(proj_root, f)
            try:
                with f.open("r", encoding="utf-8", errors="ignore") as fh:
                    for i, line in enumerate(fh, 1):
                        # Time budget check within file too
                        if (time.time() - start) > time_budget_sec:
                            timed_out = True
                            break
                        if rx.search(line):
                            matches.append(f"{rel}:{i}:{line.rstrip()}")
                            if len(matches) >= max_matches:
                                break
                if len(matches) >= max_matches or timed_out:
                    break
            except Exception:
                continue

        if not matches:
            return "(no matches)"
        out = "\n".join(matches[:max_matches])
        if timed_out or scanned >= max_files_scanned:
            out += f"\n[note] truncated: scanned≈{scanned}, returned={len(matches)}"
        return out
    return grep
