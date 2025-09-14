from __future__ import annotations
import subprocess
import os, platform, ctypes, struct
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
                timeout=max(5, int(timeout_sec)), 
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

def make_read_terminal_tool():
    def _read_terminal() -> str:
        """
        Always call this tool immediately if the user says something vague like
        "what's this", "what's this error", "fix this", or refers to 'this' without details.
        This captures the current visible terminal contents so you can answer correctly.
        Never ask the user to paste errors — rely on this tool instead.
        """
        system = platform.system()

        if system == "Windows":
            from ctypes import wintypes

            kernel32 = ctypes.windll.kernel32
            h = kernel32.GetStdHandle(-11) 

            csbi = ctypes.create_string_buffer(22)
            res = kernel32.GetConsoleScreenBufferInfo(h, csbi)
            if not res:
                return ""

            (bufx, bufy, curx, cury, wattr,
             left, top, right, bottom,
             maxx, maxy) = struct.unpack("hhhhHhhhhhh", csbi.raw)

            width = right - left + 1
            height = bottom - top + 1
            size = width * height

            chars = ctypes.create_unicode_buffer(size)
            read = ctypes.c_int(0)
            coord = wintypes._COORD(0, 0)

            kernel32.ReadConsoleOutputCharacterW(
                h, chars, size, coord, ctypes.byref(read)
            )

            return chars.value.strip()

        else:
            try:
                import re, shutil

                def _clean(s: str) -> str:
                    if not s:
                        return s
                    ansi = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -/]*[@-~]')
                    s = ansi.sub('', s).replace('\r', '')
                    return s.strip()

                if os.environ.get("TMUX") and shutil.which("tmux"):
                    r = subprocess.run(
                        ["tmux", "capture-pane", "-p"],
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=3,
                    )
                    out = _clean(r.stdout)
                    if out:
                        return out

                if (os.environ.get("STY") or os.environ.get("SCREEN") or os.path.exists("/var/run/screen")) and shutil.which("screen"):
                    tmp = "/tmp/screen_hardcopy.txt"
                    subprocess.run(["screen", "-X", "hardcopy", tmp], timeout=3)
                    try:
                        with open(tmp, "r", encoding="utf-8", errors="replace") as f:
                            out = _clean(f.read())
                        os.remove(tmp)
                        if out:
                            return out
                    except Exception:
                        pass

                history_chunks = []

                bash_hist = os.path.expanduser("~/.bash_history")
                if os.path.exists(bash_hist):
                    try:
                        with open(bash_hist, "r", encoding="utf-8", errors="replace") as f:
                            lines = f.readlines()
                        if lines:
                            history_chunks.append("".join(lines[-50:]))
                    except Exception:
                        pass

                zsh_hist = os.path.expanduser("~/.zsh_history")
                if os.path.exists(zsh_hist):
                    try:
                        with open(zsh_hist, "r", encoding="utf-8", errors="replace") as f:
                            lines = f.readlines()
                        if lines:
                            cleaned = [ln.split(";", 1)[-1] for ln in lines[-100:]]
                            history_chunks.append("".join(cleaned[-50:]))
                    except Exception:
                        pass

                fish_hist = os.path.expanduser("~/.local/share/fish/fish_history")
                if not os.path.exists(fish_hist):
                    fish_hist = os.path.expanduser("~/.config/fish/fish_history")
                if os.path.exists(fish_hist):
                    try:
                        with open(fish_hist, "r", encoding="utf-8", errors="replace") as f:
                            lines = f.readlines()
                        cmds = [ln.strip()[6:] for ln in lines if ln.strip().startswith("- cmd: ")]
                        if cmds:
                            history_chunks.append("\n".join(cmds[-50:]))
                    except Exception:
                        pass

                if history_chunks:
                    combined = _clean("\n".join(history_chunks).strip())
                    if combined:
                        return "(terminal buffer capture not supported; recent history)\n" + combined

                if shutil.which("bash"):
                    r = subprocess.run(
                        ["bash", "-ic", "history -r; history 50"],
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=3,
                    )
                    out = _clean(r.stdout)
                    if out:
                        return out

                if shutil.which("zsh"):
                    r = subprocess.run(
                        ["zsh", "-ic", "fc -l -n -50"],
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=3,
                    )
                    out = _clean(r.stdout)
                    if out:
                        return out

                if shutil.which("fish"):
                    r = subprocess.run(
                        ["fish", "-ic", "history | tail -n 50"],
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=3,
                    )
                    out = _clean(r.stdout)
                    if out:
                        return out

                return "(no recent history)"
            except Exception:
                return "(not supported on this system)"


    @tool("read_terminal", return_direct=False)
    def read_terminal() -> str:
        """
        Capture the current visible contents of the terminal.

        Use this tool when:
        - You need to know what text is currently displayed in the terminal window.
        - You want to confirm whether the screen is empty (after `clear`/`cls`).
        - You want to inspect recent command outputs still visible.

        Notes:
        - This does NOT fetch command history, only what is visible right now.
        - On Windows, it uses the console screen buffer API.
        - On Linux/macOS, output may be limited depending on terminal capabilities.
        """
        return _read_terminal()

    return read_terminal