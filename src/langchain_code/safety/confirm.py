from __future__ import annotations
import os
from rich.prompt import Confirm

def should_auto_approve() -> bool:
    return os.getenv("LANGCODE_AUTO_APPROVE", "").lower() in {"1","true","yes","y"}

def confirm_action(msg: str, apply: bool) -> bool:
    if apply or should_auto_approve():
        return True
    return Confirm.ask(msg)
