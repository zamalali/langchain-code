from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Literal
import os, sys, time, shutil, subprocess, re, datetime as _dt

from langchain_core.tools import tool
from pydantic import BaseModel, Field
# ---------------- interpreters & basics ----------------
_INTERPS = {
    "python":      lambda: [(sys.executable or "python"), "-u"],  # unbuffered
    "bash":        lambda: [shutil.which("bash") or "bash"],
    "sh":          lambda: [shutil.which("sh") or "sh"],
    "powershell":  lambda: ["powershell" if os.name == "nt" else "pwsh", "-NoProfile", "-ExecutionPolicy", "Bypass"],
    "node":        lambda: [shutil.which("node") or "node"],
    "cmd":         lambda: ["cmd", "/c"],  # Windows only
}
_EXT = {"python": "py", "bash": "sh", "sh": "sh", "powershell": "ps1", "node": "js", "cmd": "cmd"}

_DENY = (
    "rm -rf /", "mkfs", "format ", "shutdown", "reboot", ":(){ :|:& };:",
)

def _rooted(project_dir: str, path: str) -> Path:
    p = (Path(project_dir) / (path or "")).resolve()
    root = Path(project_dir).resolve()
    if not str(p).startswith(str(root)):
        raise ValueError("Path escapes project root")
    return p

def _looks_read_only(lang: str, code: str) -> bool:
    s = code.lower()

    if lang in {"bash", "sh", "cmd"}:
        patterns = [
            r"\brm\b", r"\bmv\b", r"\bcp\b", r"\bchmod\b", r"\bchown\b",
            r"\bmkdir\b", r"\brmdir\b", r">\s*\S", r">>\s*\S", r"\btruncate\b",
            r"\btouch\b", r"\btee\b", r"\bsed\b.*\s-i(\s|$)", r"\bgit\s+(clean|reset|checkout\s+--)"
        ]
        return not any(re.search(p, s) for p in patterns)

    if lang == "python":
        if re.search(r"\bopen\s*\([^)]*[\"'](?:w|a|\+)[\"']", s):  # write modes
            return False
        if re.search(r"\b(os\.remove|os\.unlink|os\.rmdir|os\.rename|os\.chmod|os\.chown|os\.mkdir|os\.makedirs)\b", s):
            return False
        if re.search(r"\b(shutil\.(copy|copy2|copyfile|copytree|move|rmtree|make_archive))\b", s):
            return False
        return True

    if lang == "powershell":
        patterns = [r"\bremove-item\b", r"\bmove-item\b", r"\bcopy-item\b", r"\bnew-item\b", r"\bset-item\b", r">\s*\S", r">>\s*\S"]
        return not any(re.search(p, s) for p in patterns)

    if lang == "node":
        if re.search(r"\bfs\.(write|append|rename|chmod|chown|rm|rmdir|mkdir)\b", s):
            return False
        return True

    return True

def _janitor(tmp_dir: Path, hours: int = 2) -> None:
    """Best-effort cleanup of old temp scripts."""
    try:
        if not tmp_dir.exists():
            return
        cutoff = _dt.datetime.utcnow().timestamp() - hours * 3600
        for p in tmp_dir.glob("snippet_*.*"):
            try:
                if p.stat().st_mtime < cutoff:
                    p.unlink(missing_ok=True)
            except Exception:
                pass
    except Exception:
        pass

# ---------------- args schema (Gemini-safe) ----------------
class ScriptExecArgs(BaseModel):
    language: Literal["python", "bash", "sh", "powershell", "node", "cmd"] = Field(..., description="Interpreter.")
    code: str = Field(..., description="Short, self-contained script body.")
    argv: List[str] = Field(default_factory=list, description="Command-line arguments.")
    stdin: str = Field(default="", description="Optional standard input.")
    timeout_sec: int = Field(default=120, ge=5, le=900, description="Hard time limit (seconds).")
    save_as: Optional[str] = Field(default=None, description="Relative path to persist the script.")
    persist: bool = Field(default=False, description="Keep the script even if save_as is not provided.")
    report: Literal["stdout", "full", "auto"] = Field(default="stdout", description="Output format.")
    safety: Literal["auto", "require", "force"] = Field(
        default="auto",
        description="auto: allow read-only without apply; require: needs apply=True; force: skip consent gate."
    )

# ---------------- tool factory ----------------
def make_script_exec_tool(project_dir: str, apply: bool, *, return_direct: bool = False):
    tmp_dir = _rooted(project_dir, ".langcode/tmp_scripts")
    _janitor(tmp_dir)  # opportunistic cleanup

    @tool("script_exec", args_schema=ScriptExecArgs, return_direct=return_direct)
    def script_exec(**kwargs) -> str:
        """
        Run a short script in the project workspace.
        - No prompts: read-only scripts run when apply=False (safety='auto').
        - Mutating scripts require apply=True unless safety='force'.
        - Temp scripts are deleted unless you set save_as or persist=True.
        """
        args = ScriptExecArgs(**kwargs)  # validate + fill defaults

        # quick safety checks
        if args.language not in _INTERPS:
            return f"Unsupported language: {args.language}. Allowed: {', '.join(sorted(_INTERPS.keys()))}"
        for bad in _DENY:
            if bad in args.code:
                return f"Blocked for safety; found '{bad}'."

        lang = args.language
        base_cmd = _INTERPS[lang]()
        if any(x is None for x in base_cmd):
            return f"Interpreter for {lang} not found on PATH."

        read_only = _looks_read_only(lang, args.code)
        if args.safety == "require" and not apply:
            return "Execution requires apply=True (explicit consent). Re-run with --apply."
        if args.safety == "auto" and not apply and not read_only:
            return ("Declined without apply: script appears to modify files. "
                    "Re-run with --apply, or set safety='force' if you intend to run it.")

        # command building (prefer inline -c for small python)
        cmd: List[str]
        script_path: Optional[Path] = None
        use_inline = (lang == "python" and not args.save_as and not args.persist and len(args.code) < 8000)
        if use_inline:
            cmd = base_cmd + ["-c", args.code]
        else:
            ext = _EXT.get(lang, "txt")
            filename = args.save_as or f".langcode/tmp_scripts/snippet_{int(time.time()*1000)}.{ext}"
            script_path = _rooted(project_dir, filename)
            script_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                body = args.code
                if lang in {"bash", "sh"} and not body.lstrip().startswith("#!"):
                    body = "#!/usr/bin/env bash\nset -euo pipefail\n" + body
                script_path.write_text(body, encoding="utf-8")
                if lang in {"bash", "sh"}:
                    try:
                        os.chmod(script_path, 0o755)
                    except Exception:
                        pass
            except Exception as e:
                return f"Failed to write script: {type(e).__name__}: {e}"

            if lang == "cmd":
                if os.name != "nt":
                    try: script_path.unlink(missing_ok=True)
                    except Exception: pass
                    return "cmd is Windows-only."
            if lang == "powershell":
                cmd = base_cmd + ["-File", str(script_path)]
            else:
                cmd = base_cmd + [str(script_path)]

        if args.argv:
            cmd += [str(a) for a in args.argv]

        # hide console on Windows
        popen_kwargs = {}
        if os.name == "nt":
            try:
                popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
            except Exception:
                pass

        try:
            proc = subprocess.run(
                cmd,
                cwd=project_dir,
                input=args.stdin,
                text=True,
                capture_output=True,
                timeout=max(5, int(args.timeout_sec)),
                shell=False,
                encoding="utf-8",
                errors="replace",
                **popen_kwargs,
            )
            out = (proc.stdout or "").replace("\r\n", "\n").strip()
            err = (proc.stderr or "").replace("\r\n", "\n").strip()

            def _clip(s: str, n: int = 24000) -> str:
                return s if len(s) <= n else s[:n] + "\n...[truncated]..."

            if args.report == "stdout":
                return out or f"(no stdout)\n(exit {proc.returncode})"
            if args.report == "auto":
                if out and not err and len(out) <= 2000:
                    return out
                return f"$ {' '.join(cmd)}\n(exit {proc.returncode})\n[stdout]\n{_clip(out)}\n[stderr]\n{_clip(err)}"

            # full
            src = (script_path and script_path.relative_to(Path(project_dir))) or "<inline>"
            lines = [f"$ {' '.join(cmd)}", f"(exit {proc.returncode})", f"[script] {src}"]
            if out: lines += ["", "[stdout]", _clip(out)]
            if err: lines += ["", "[stderr]", _clip(err)]
            return "\n".join(lines)

        except subprocess.TimeoutExpired:
            return f"Timed out after {args.timeout_sec}s running: {' '.join(cmd)}"
        except Exception as e:
            return f"Error executing script: {type(e).__name__}: {e}"
        finally:
            if script_path and not (args.persist or args.save_as):
                try:
                    script_path.unlink(missing_ok=True)
                except Exception:
                    pass

    return script_exec