"""Shared runtime state for the LangCode CLI."""

from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Tuple

from rich.console import Console

console = Console()

# Agent cache mirrors the legacy module-level `_AGENT_CACHE`.
AgentCacheKey = Tuple[str, str, str, str, bool]
agent_cache: "OrderedDict[AgentCacheKey, Any]" = OrderedDict()
agent_cache_limit = 6

# Launcher flag mirrors `_IN_SELECTION_HUB`.
_selection_hub_active = False

# Controls whether edit helpers should show status banners.
_edit_feedback_enabled = False

# Keeps track of the currently active Rich live display.
current_live = None


def set_selection_hub_active(active: bool) -> None:
    """Toggle whether the launcher is currently active."""
    global _selection_hub_active
    _selection_hub_active = active


def in_selection_hub() -> bool:
    """Return True when the launcher hub owns the terminal."""
    return _selection_hub_active


def edit_feedback_enabled() -> bool:
    """Return True if edit helpers should print status banners."""
    return _edit_feedback_enabled


@contextmanager
def edit_feedback():
    """Temporarily enable edit feedback panels while a user is editing."""
    global _edit_feedback_enabled
    prev = _edit_feedback_enabled
    _edit_feedback_enabled = True
    try:
        yield
    finally:
        _edit_feedback_enabled = prev
