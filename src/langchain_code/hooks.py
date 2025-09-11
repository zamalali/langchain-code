from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import fnmatch
import os
import re
import shlex
import subprocess
import threading
import time

try:
    import yaml
except Exception:
    yaml = None


from .safety.confirm import confirm_action



HOOKS_PATH_REL = Path(".langcode/hooks.yml")



@dataclass
class HookAction:
    run: Optional[str] = None
    deny: Optional[str] = None
    confirm: Optional[str] = None
    set_env: Dict[str, str] = field(default_factory=dict)


@dataclass
class HookRule:
    on: str
    if_path: List[str] = field(default_factory=list)
    if_cmd: List[str] = field(default_factory=list)
    if_cmd_re: Optional[str] = None
    if_language: List[str] = field(default_factory=list)
    when_mutating: Optional[bool] = None
    require_apply: Optional[bool] = None
    if_exit_code: Optional[int] = None
    actions: List[HookAction] = field(default_factory=list)


@dataclass
class HookConfig:
    version: int = 1
    timeout_sec: int = 60
    on_error: str = "fail"
    hooks: List[HookRule] = field(default_factory=list)


class HookResult:
    def __init__(self, allowed: bool, message: str = "", outputs: Optional[List[str]] = None):
        self.allowed = allowed
        self.message = message
        self.outputs = outputs or []

    def with_output(self, text: str) -> "HookResult":
        self.outputs.append(text)
        return self



class _SafeDict(dict):
    
    def __missing__(self, k):
        return "{" + k + "}"


def _fmt(template: str, ctx: Dict[str, Any]) -> str:
    return template.format_map(_SafeDict(**{k: _stringify(v) for k, v in (ctx or {}).items()}))


def _stringify(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return str(v)
    return str(v)


def _split_cmd(cmd: str) -> List[str]:
    
    if os.name == "nt":
        return [cmd]
    return shlex.split(cmd)



class HookRunner:
    
    def __init__(self, project_dir: Path):
        self.root = Path(project_dir).resolve()
        self.cfg_path = (self.root / HOOKS_PATH_REL).resolve()
        self._cfg = HookConfig()
        self._mtime = 0.0
        self._lock = threading.RLock()


    def fire(self, event: str, ctx: Dict[str, Any] | None = None) -> HookResult:
        
        self._reload_if_changed()
        if not self._cfg.hooks:
            return HookResult(True)

        c = dict(ctx or {})
        c.setdefault("path", "")
        c.setdefault("cmd", "")
        c.setdefault("language", "")
        c.setdefault("read_only", False)
        c.setdefault("mutating", False)
        c.setdefault("apply", False)
        c.setdefault("exit_code", None)

        outputs: List[str] = []

        for rule in self._cfg.hooks:
            if not self._match(rule, event, c):
                continue

            local_env = os.environ.copy()

            for action in rule.actions:
                if action.set_env:
                    for k, v in action.set_env.items():
                        try:
                            local_env[k] = _fmt(v, c)
                        except Exception:
                            local_env[k] = str(v)

                if action.deny:
                    msg = _fmt(action.deny, c)
                    if event.startswith("pre_"):
                        return HookResult(False, f"Blocked by hook: {msg}", outputs)
                    else:
                        outputs.append(f"[hook:deny:post] {msg}")
                        continue

                if action.confirm:
                    msg = _fmt(action.confirm, c)
                    ok = confirm_action(msg, bool(c.get("apply")))
                    if not ok:
                        return HookResult(False, f"Cancelled by user: {msg}", outputs)

                if action.run:
                    cmd = _fmt(action.run, c)
                    res = self._run(cmd, env=local_env)
                    outputs.append(res["report"])
                    if res["failed"] and self._cfg.on_error == "fail" and event.startswith("pre_"):
                        return HookResult(False, f"Hook command failed: {cmd}", outputs)

        return HookResult(True, "", outputs)


    def _reload_if_changed(self) -> None:
        with self._lock:
            if yaml is None:  # pragma: no cover
                self._cfg = HookConfig()
                return
            if not self.cfg_path.exists():
                self._cfg = HookConfig()
                return
            try:
                m = self.cfg_path.stat().st_mtime
            except Exception:
                self._cfg = HookConfig()
                return
            if m <= self._mtime:
                return
            self._mtime = m
            try:
                data = yaml.safe_load(self.cfg_path.read_text(encoding="utf-8")) or {}
                self._cfg = self._parse_config(data)
            except Exception:
                self._cfg = HookConfig()

    def _parse_config(self, data: Dict[str, Any]) -> HookConfig:
        defaults = data.get("defaults", {}) or {}
        cfg = HookConfig(
            version=int(data.get("version", 1)),
            timeout_sec=int(defaults.get("timeout_sec", 60)),
            on_error=str(defaults.get("on_error", "fail")).lower(),
            hooks=[],
        )
        for raw in data.get("hooks", []) or []:
            actions: List[HookAction] = []
            for a in raw.get("actions", []) or []:
                actions.append(HookAction(
                    run=a.get("run"),
                    deny=a.get("deny"),
                    confirm=a.get("confirm"),
                    set_env=(a.get("set_env") or {}),
                ))
            cfg.hooks.append(HookRule(
                on=str(raw.get("on", "")),
                if_path=list(raw.get("if_path", []) or []),
                if_cmd=list(raw.get("if_cmd", []) or []),
                if_cmd_re=raw.get("if_cmd_re"),
                if_language=list(raw.get("if_language", []) or []),
                when_mutating=raw.get("when_mutating"),
                require_apply=raw.get("require_apply"),
                if_exit_code=raw.get("if_exit_code"),
                actions=actions,
            ))
        return cfg

    def _match(self, rule: HookRule, event: str, ctx: Dict[str, Any]) -> bool:
        if rule.on != event:
            return False

        if rule.if_path:
            p = str(ctx.get("path") or "").replace("\\", "/")
            if not any(fnmatch.fnmatch(p, pat) for pat in rule.if_path):
                return False

        if rule.if_cmd:
            c = str(ctx.get("cmd") or "")
            if not any(sub in c for sub in rule.if_cmd):
                return False

        if rule.if_cmd_re:
            c = str(ctx.get("cmd") or "")
            try:
                if not re.search(rule.if_cmd_re, c):
                    return False
            except re.error:
                return False

        if rule.if_language:
            lang = str(ctx.get("language") or "").lower()
            if lang not in [s.lower() for s in rule.if_language]:
                return False

        if rule.when_mutating is not None and bool(ctx.get("mutating")) != bool(rule.when_mutating):
            return False
        if rule.require_apply is not None and bool(ctx.get("apply")) != bool(rule.require_apply):
            return False

        if rule.if_exit_code is not None:
            try:
                if int(ctx.get("exit_code")) != int(rule.if_exit_code):
                    return False
            except Exception:
                return False

        return True

    def _run(self, cmd: str, *, env: Dict[str, str]) -> Dict[str, Any]:
        
        use_shell = os.name == "nt"
        argv = _split_cmd(cmd)

        try:
            proc = subprocess.run(
                argv if not use_shell else cmd,
                cwd=self.root,
                text=True,
                capture_output=True,
                timeout=max(5, int(self._cfg.timeout_sec)),
                shell=use_shell,
                env=env,
            )
            out = (proc.stdout or "").strip()
            err = (proc.stderr or "").strip()
            report = f"$ {cmd}\n(exit {proc.returncode})"
            if out:
                report += f"\n[stdout]\n{out}"
            if err:
                report += f"\n[stderr]\n{err}"
            return {"failed": proc.returncode != 0, "report": report}
        except subprocess.TimeoutExpired:
            return {"failed": True, "report": f"$ {cmd}\n(timeout after {self._cfg.timeout_sec}s)"}
        except Exception as e:
            return {"failed": True, "report": f"$ {cmd}\n(error: {type(e).__name__}: {e})"}



_RUNNERS: Dict[str, HookRunner] = {}


def get_hook_runner(project_dir: str | Path) -> HookRunner:
    root = str(Path(project_dir).resolve())
    runner = _RUNNERS.get(root)
    if runner is None:
        runner = HookRunner(Path(project_dir))
        _RUNNERS[root] = runner
    return runner