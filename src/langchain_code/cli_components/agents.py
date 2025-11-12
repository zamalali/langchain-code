from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

from ..config_core import resolve_provider as _resolve_provider_base
from .state import agent_cache, agent_cache_limit, AgentCacheKey


def agent_cache_get(key: AgentCacheKey):
    if key in agent_cache:
        agent_cache.move_to_end(key)
        return agent_cache[key]
    return None


def agent_cache_put(key: AgentCacheKey, value: Any) -> None:
    agent_cache[key] = value
    agent_cache.move_to_end(key)
    while len(agent_cache) > agent_cache_limit:
        agent_cache.popitem(last=False)


def resolve_provider(llm_opt: Optional[str], router: bool) -> str:
    if llm_opt:
        return _resolve_provider_base(llm_opt)
    if router:
        return "gemini"
    return _resolve_provider_base(None)


def build_react_agent_with_optional_llm(provider: str, project_dir: Path, llm=None, **kwargs):
    from ..agent.react import build_react_agent

    try:
        if llm is not None:
            return build_react_agent(provider=provider, project_dir=project_dir, llm=llm, **kwargs)
    except TypeError:
        pass
    return build_react_agent(provider=provider, project_dir=project_dir, **kwargs)


def build_deep_agent_with_optional_llm(provider: str, project_dir: Path, llm=None, **kwargs):
    from ..agent.react import build_deep_agent

    try:
        if llm is not None:
            return build_deep_agent(provider=provider, project_dir=project_dir, llm=llm, **kwargs)
    except TypeError:
        pass
    return build_deep_agent(provider=provider, project_dir=project_dir, **kwargs)
