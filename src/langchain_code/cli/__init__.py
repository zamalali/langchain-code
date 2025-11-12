from __future__ import annotations

"""
Modular CLI package that wires together the Typer app and individual commands.

This package exists to keep the top-level entrypoint (`langchain_code.cli`)
lightweight while allowing each command/session to live in its own module.
"""

from .entrypoint import app, main, selection_hub

__all__ = ["app", "main", "selection_hub"]
