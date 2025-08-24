from __future__ import annotations
from pathlib import Path
import fnmatch
import re
from typing import List
from langchain_core.tools import tool

def _walk_files(root: Path):
    for p in root.rglob("*"):
        if p.is_file():
            yield p

def make_glob_tool(project_dir: str):
    @tool("glob", return_direct=False)
    def glob(pattern: str) -> str:
        """
        Find files in the project by glob pattern.

        Use for locating files when you know part of their name or extension:
          - `"**/*.py"` → all Python files
          - `"src/**/*.json"` → JSON files under src/
          - `"*config*"` → any file with "config" in its name

        Returns up to 500 relative paths, or "(no matches)" if none found.
        """
        root = Path(project_dir)
        files = [str(p.relative_to(root)) for p in _walk_files(root)
                 if fnmatch.fnmatch(str(p.relative_to(root)), pattern)]
        return "\n".join(files[:500]) if files else "(no matches)"
    return glob

def make_grep_tool(project_dir: str):
    @tool("grep", return_direct=False)
    def grep(pattern: str, path: str = ".") -> str:
        """
        Search for a regex pattern inside files under a directory.

        Use for finding where something is defined or referenced:
          - `grep("class IntelligentLLMRouter")`
          - `grep("def get_model", "src/")`
          - `grep("import re")`

        Returns lines in the form:
          <file>:<line number>:<matched text>

        Scans up to 500 matches across files. If no matches, returns "(no matches)".
        """        
        root = Path(project_dir).joinpath(path)
        if not root.exists():
            return f"{path} not found."
        rx = re.compile(pattern)
        out = []
        for p in _walk_files(root):
            rel = str(p.relative_to(Path(project_dir)))
            try:
                with p.open("r", encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f, 1):
                        if rx.search(line):
                            out.append(f"{rel}:{i}:{line.strip()}")
                            if len(out) >= 500:
                                return "\n".join(out)
            except Exception:
                continue
        return "\n".join(out) if out else "(no matches)"
    return grep
