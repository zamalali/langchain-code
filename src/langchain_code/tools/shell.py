from __future__ import annotations
import subprocess
from langchain_core.tools import tool
from ..safety.confirm import confirm_action

def make_run_cmd_tool(cwd: str, apply: bool, test_cmd: str | None):
    @tool("run_cmd", return_direct=False)
    def run_cmd(command: str, timeout_sec: int = 120) -> str:
        """
        Run a shell command in the project directory (`cwd`).

        Use this tool for tasks such as:
        - Listing files or directories (`ls -la`, `dir`)
        - Finding a file (`find . -name "config.py"`)
        - Searching file contents (`grep "Router" -r src/`)
        - Running project commands (`pytest`, `make build`)

        Notes:
        - Pass a single command string (chain with `&&` if needed).
        - `{TEST_CMD}` will be replaced with the configured test command if used.
        - Timeout defaults to 120s; increase for long tests.
        """
        cmd = command.strip()
        if cmd == "{TEST_CMD}" and test_cmd:
            cmd = test_cmd

        if not confirm_action(f"Run command: `{cmd}` ?", apply):
            return f"Command cancelled: {cmd}"

        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                shell=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=max(5, int(timeout_sec)),  # NEW
            )
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            code = result.returncode
            out = f"$ {cmd}\n(exit {code})\n"
            if stdout:
                out += f"\n[stdout]\n{stdout}\n"
            if stderr:
                out += f"\n[stderr]\n{stderr}\n"
            return out
        except subprocess.TimeoutExpired:
            return f"$ {cmd}\n(timeout after {timeout_sec}s)\n"
        except Exception as e:
            return f"Error running `{cmd}`: {e}"
    return run_cmd
