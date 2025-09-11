from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from importlib.resources import files as pkg_files
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

try:
    # LC >= 0.2
    from langchain_core.tools.structured import StructuredTool  # type: ignore
except Exception:  # pragma: no cover
    # LC < 0.2 fallback
    from langchain.tools import StructuredTool  # type: ignore

# ---------- dotenv (optional) ----------
try:
    from dotenv import load_dotenv, find_dotenv

    def _preload_dotenv() -> None:
        # find .env in or above CWD (typical CLI usage) without overriding already-set env
        path = find_dotenv(usecwd=True)
        if path:
            load_dotenv(path, override=False)
except Exception:  # pragma: no cover
    def _preload_dotenv() -> None:
        pass


logger = logging.getLogger("langcode.mcp")
logger.addHandler(logging.NullHandler())
_VERBOSE = os.getenv("LANGCODE_MCP_VERBOSE", "").lower() in {"1", "true", "yes"}
if _VERBOSE:
    logging.basicConfig(level=logging.DEBUG)
    logger.setLevel(logging.DEBUG)

# ---------- default on-disk locations ----------
_DEFAULT_LOCATIONS: List[Path] = [
    Path.cwd() / ".langcode" / "mcp.json",
    Path.cwd() / "mcp.json",
    Path.home() / ".langcode" / "mcp.json",
    Path.home() / ".config" / "langcode" / "mcp.json",
]

# ---------- packaged fallback (inside the wheel) ----------
def _load_packaged_cfg() -> Optional[Dict[str, Any]]:
    """Load mcp.json packaged at langchain_code/config/mcp.json (wheel-safe)."""
    try:
        res = pkg_files("langchain_code.config") / "mcp.json"
        if res and res.is_file():
            data = json.loads(res.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("servers"), dict):
                logger.debug("MCP: loaded packaged config langchain_code/config/mcp.json")
                return data
    except Exception as e:  # pragma: no cover
        logger.debug("MCP: failed to load packaged config: %s", e)
    return None

def _env_overrides() -> List[Path]:
    """Allow users to point to custom config via env."""
    paths: List[Path] = []
    j = os.getenv("LANGCODE_MCP_JSON", "").strip()
    if j:
        for chunk in j.split(os.pathsep):
            p = Path(chunk).expanduser()
            if p.is_file():
                paths.append(p)
    d = os.getenv("LANGCODE_MCP_DIR", "").strip()
    if d:
        p = Path(d).expanduser() / "mcp.json"
        if p.is_file():
            paths.append(p)
    return paths

def _project_locations(project_dir: Optional[Path]) -> List[Path]:
    if not project_dir:
        return []
    project_dir = project_dir.resolve()
    return [
        project_dir / ".langcode" / "mcp.json",
        project_dir / "mcp.json",
    ]

def _expand_env_placeholders(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Expand $VARS in server env blocks using process env (after .env preload)."""
    servers = cfg.get("servers", {}) or {}
    for _, server in servers.items():
        env_map = server.get("env", {}) or {}
        if not isinstance(env_map, dict):
            continue
        expanded = {}
        for k, v in env_map.items():
            expanded[k] = os.path.expandvars(os.path.expanduser(v)) if isinstance(v, str) else v
        server["env"] = expanded
    return cfg

def _normalize_commands_for_windows(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """On Windows, prefer npx.cmd for CreateProcess compatibility."""
    if os.name != "nt":
        return cfg
    servers = cfg.get("servers", {}) or {}
    for name, server in servers.items():
        cmd = server.get("command")
        if isinstance(cmd, str) and cmd.lower() == "npx":
            server["command"] = "npx.cmd"
            logger.debug("MCP: normalized server '%s' command to npx.cmd for Windows", name)
    return cfg

def _read_json_file(pathlike: Union[Path, Any]) -> Optional[Dict[str, Any]]:
    try:
        if isinstance(pathlike, Path):
            if not pathlike.exists() or not pathlike.is_file():
                return None
            return json.loads(pathlike.read_text(encoding="utf-8"))
        if hasattr(pathlike, "read_text"):
            return json.loads(pathlike.read_text(encoding="utf-8"))
    except Exception as e:  # pragma: no cover
        logger.debug("MCP: failed to read JSON from %s: %s", pathlike, e)
    return None

def _merge_server_cfgs_dicts(dicts: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {"servers": {}}
    for data in dicts:
        if not data:
            continue
        servers = data.get("servers", {})
        if isinstance(servers, dict):
            cfg["servers"].update(servers)
    cfg = _expand_env_placeholders(cfg)
    cfg = _normalize_commands_for_windows(cfg)
    return cfg

def _ensure_sync_invocation(tool: BaseTool) -> BaseTool:
    """Make async StructuredTool invokable in sync LC agents."""
    try:
        is_structured = isinstance(tool, StructuredTool)  # type: ignore[arg-type]
    except Exception:
        is_structured = False

    if is_structured and getattr(tool, "coroutine", None) and getattr(tool, "func", None) is None:
        async_coro = tool.coroutine

        def _sync_func(*args, **kwargs):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if not loop or not loop.is_running():
                return asyncio.run(async_coro(*args, **kwargs))

            result_holder: Dict[str, Any] = {}
            error_holder: Dict[str, BaseException] = {}
            done = threading.Event()

            def _runner():
                new_loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(new_loop)
                    result_holder["value"] = new_loop.run_until_complete(async_coro(*args, **kwargs))
                except BaseException as e:  # pragma: no cover
                    error_holder["error"] = e
                finally:
                    try:
                        new_loop.close()
                    finally:
                        done.set()

            t = threading.Thread(target=_runner, daemon=True)
            t.start()
            done.wait()
            if "error" in error_holder:
                raise error_holder["error"]
            return result_holder.get("value")

        tool.func = _sync_func  # type: ignore[attr-defined]
    return tool

def _sanitize_tool_schema(tool: BaseTool) -> BaseTool:
    s = getattr(tool, "args_schema", None)
    if s is None:
        return tool

    def _strip(obj: Any) -> Any:
        if isinstance(obj, dict):
            obj.pop("$schema", None)
            obj.pop("additionalProperties", None)
            for k, v in list(obj.items()):
                obj[k] = _strip(v)
        elif isinstance(obj, list):
            return [_strip(x) for x in obj]
        return obj

    try:
        if isinstance(s, dict):
            tool.args_schema = _strip(s)  # type: ignore[assignment]
    except Exception as e:  # pragma: no cover
        logger.debug("MCP: sanitize schema failed for %s: %s", getattr(tool, "name", "<unknown>"), e)
    return tool

async def get_mcp_tools(project_dir: Optional[Path] = None) -> List[BaseTool]:
    """
    Search order:
      1) project_dir/.langcode/mcp.json, project_dir/mcp.json  (if provided)
      2) LANGCODE_MCP_JSON paths, LANGCODE_MCP_DIR/mcp.json
      3) CWD/HOME defaults: ./.langcode/mcp.json, ./mcp.json, ~/.langcode/mcp.json, ~/.config/langcode/mcp.json
      4) Packaged fallback: langchain_code/config/mcp.json (inside the wheel)
    """
    _preload_dotenv()

    if not _VERBOSE:
        logging.getLogger("langchain_mcp_adapters").setLevel(logging.WARNING)
        logging.getLogger("langchain").setLevel(logging.WARNING)

    dicts: List[Dict[str, Any]] = []

    # 1–3: files on disk
    disk_locations = [
        *_project_locations(project_dir),
        *_env_overrides(),
        *_DEFAULT_LOCATIONS,
    ]
    looked: List[str] = []
    for loc in disk_locations:
        looked.append(str(loc))
        d = _read_json_file(loc)
        if d:
            dicts.append(d)

    # 4: packaged fallback
    packaged = _load_packaged_cfg()
    if packaged:
        dicts.append(packaged)
        looked.append("pkg:langchain_code/config/mcp.json")

    cfg = _merge_server_cfgs_dicts(dicts)
    if not cfg.get("servers"):
        logger.debug("MCP: no servers discovered. Looked in: %s", " | ".join(looked))
        return []

    try:
        client = MultiServerMCPClient(cfg["servers"])
        tools = await client.get_tools()
    except Exception as e:
        logger.debug("MCP: client/get_tools failed: %s", e)
        return []

    tools = [_sanitize_tool_schema(t) for t in tools]
    tools = [_ensure_sync_invocation(t) for t in tools]

    logger.debug("MCP: total tools loaded: %d", len(tools))
    return tools

