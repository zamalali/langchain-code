from __future__ import annotations
import json
import logging
import os
import asyncio
import threading
from pathlib import Path
from typing import Any, Dict, List
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
try:
    from langchain_core.tools.structured import StructuredTool
except Exception:
    from langchain.tools import StructuredTool

try:
    from dotenv import load_dotenv, find_dotenv
    def _preload_dotenv() -> None:
        path = find_dotenv(usecwd=True)
        if path:
            load_dotenv(path, override=False) 
except Exception:
    def _preload_dotenv() -> None:
        pass 

logger = logging.getLogger("langcode.mcp")
logger.addHandler(logging.NullHandler())
_VERBOSE = os.getenv("LANGCODE_MCP_VERBOSE", "").lower() in {"1", "true", "yes"}
if _VERBOSE:
    logging.basicConfig(level=logging.DEBUG)
    logger.setLevel(logging.DEBUG)

CONFIG_LOCATIONS = [
    Path.cwd() / ".langcode" / "mcp.json",
    Path.cwd() / "mcp.json",
    Path.home() / ".langcode" / "mcp.json",
]

def _expand_env_placeholders(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Expand $VARS (or ${VARS}) in server env blocks using process env (loaded from .env)."""
    servers = cfg.get("servers", {})
    for _, server in servers.items():
        env_map = server.get("env", {})
        if not isinstance(env_map, dict):
            continue
        expanded = {}
        for k, v in env_map.items():
            if isinstance(v, str):
                expanded[k] = os.path.expandvars(os.path.expanduser(v))
            else:
                expanded[k] = v
        server["env"] = expanded
    return cfg

def _load_config() -> Dict[str, Any]:
    cfg: Dict[str, Any] = {"servers": {}}
    looked = []
    for loc in CONFIG_LOCATIONS:
        looked.append(str(loc))
        if loc.exists():
            try:
                raw = json.loads(loc.read_text(encoding="utf-8"))
                if isinstance(raw, dict) and isinstance(raw.get("servers"), dict):
                    cfg["servers"].update(raw["servers"])
            except Exception as e:
                logger.debug("Failed to load config from %s: %s", loc, e)
    cfg = _expand_env_placeholders(cfg)
    if not cfg["servers"]:
        logger.debug("No servers found. Looked in: %s", ", ".join(looked))
    else:
        logger.debug("Loaded servers: %s", ", ".join(cfg["servers"].keys()))
    return cfg

def _strip_keys_recursive(obj: Any) -> Any:
    if isinstance(obj, dict):
        obj.pop("$schema", None)
        obj.pop("additionalProperties", None)
        for k, v in list(obj.items()):
            obj[k] = _strip_keys_recursive(v)
    elif isinstance(obj, list):
        return [_strip_keys_recursive(x) for x in obj]
    return obj

def _sanitize_tool_schema(tool: BaseTool) -> BaseTool:
    s = getattr(tool, "args_schema", None)
    if s is None:
        return tool
    try:
        if isinstance(s, dict):
            tool.args_schema = _strip_keys_recursive(s)
    except Exception as e:
        logger.debug("Failed to sanitize schema for tool %s: %s", getattr(tool, "name", "<unknown>"), e)
    return tool

def _ensure_sync_invocation(tool: BaseTool) -> BaseTool:
    if isinstance(tool, StructuredTool) and getattr(tool, "coroutine", None) and getattr(tool, "func", None) is None:
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
                except BaseException as e:
                    error_holder["error"] = e
                finally:
                    try:
                        new_loop.close()
                    finally:
                        done.set()
            t = threading.Thread(target=_runner, daemon=True)
            t.start(); done.wait()
            if "error" in error_holder:
                raise error_holder["error"]
            return result_holder.get("value")
        tool.func = _sync_func
    return tool

async def get_mcp_tools() -> List[BaseTool]:
    _preload_dotenv()

    if not _VERBOSE:
        logging.getLogger("langchain_mcp_adapters").setLevel(logging.WARNING)
        logging.getLogger("langchain").setLevel(logging.WARNING)

    cfg = _load_config()
    if not cfg["servers"]:
        return []
    client = MultiServerMCPClient(cfg["servers"])
    tools = await client.get_tools()
    tools = [_sanitize_tool_schema(t) for t in tools]
    tools = [_ensure_sync_invocation(t) for t in tools]
    logger.debug("Total tools loaded: %d", len(tools))
    return tools
