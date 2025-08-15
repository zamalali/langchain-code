from __future__ import annotations
import subprocess
from langchain_core.tools import tool
from ..safety.confirm import confirm_action

def make_run_cmd_tool(cwd: str, apply: bool, test_cmd: str | None):
    @tool("run_cmd", return_direct=False)
    def run_cmd(command: str) -> str:
        """
        Run a shell command in cwd. Confirmation required unless apply=True.
        Use '{TEST_CMD}' placeholder to run the provided test_cmd if configured.
        """
        cmd = command.strip()
        if cmd == "{TEST_CMD}" and test_cmd:
            cmd = test_cmd

        if not confirm_action(f"Run command: `{cmd}` ?", apply):
            return f"Command cancelled: {cmd}"

        try:
            # shell=True for Windows compatibility; user commands are echoed back.
            result = subprocess.run(cmd, cwd=cwd, shell=True, capture_output=True, text=True)
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            code = result.returncode
            out = f"$ {cmd}\n(exit {code})\n"
            if stdout:
                out += f"\n[stdout]\n{stdout}\n"
            if stderr:
                out += f"\n[stderr]\n{stderr}\n"
            return out
        except Exception as e:
            return f"Error running `{cmd}`: {e}"
    return run_cmd
