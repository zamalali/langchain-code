from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from functools import lru_cache
from rich import box
from rich.console import Console, Group
from rich.align import Align
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.style import Style
from rich.text import Text
from pyfiglet import Figlet

from .state import console, in_selection_hub


@lru_cache(maxsize=64)
def _ascii_gradient_lines(width: int, text: str, font: str, gradient: str) -> Group:
    def _hex_to_rgb(h: str) -> tuple[int, int, int]:
        h = h.lstrip("#")
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

    def _lerp(a: int, b: int, t: float) -> int:
        return int(a + (b - a) * t)

    def _interpolate_palette(palette: list[str], steps: int) -> list[str]:
        if steps <= 1:
            return [palette[0]]
        out, steps_total = [], steps - 1
        for x in range(steps):
            pos = x / steps_total if steps_total else 0
            seg = min(int(pos * (len(palette) - 1)), len(palette) - 2)
            seg_start = seg / (len(palette) - 1)
            seg_end = (seg + 1) / (len(palette) - 1)
            local_t = (pos - seg_start) / (seg_end - seg_start + 1e-9)
            c1, c2 = _hex_to_rgb(palette[seg]), _hex_to_rgb(palette[seg + 1])
            rgb = tuple(_lerp(a, b, local_t) for a, b in zip(c1, c2))
            out.append(f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}")
        return out

    fig = Figlet(font=font, width=width)
    lines = fig.renderText(text).rstrip("\n").splitlines()
    if not lines:
        return Group()

    max_len = max(len(line) for line in lines)
    palette = ["#052e1e", "#064e3b", "#065f46", "#047857", "#059669", "#16a34a", "#22c55e", "#34d399"]
    if gradient == "light_to_dark":
        palette = list(reversed(palette))
    ramp = _interpolate_palette(palette, max_len)

    rendered_lines: List[Any] = []
    for raw in lines:
        if not raw.strip():
            rendered_lines.append(Align.center(Text(""), width=width))
            continue
        styled = Text()
        for idx, ch in enumerate(raw):
            styled.append(ch, style=Style(color=ramp[idx], bold=(ch != " ")))
        rendered_lines.append(Align.center(styled, width=width))
    return Group(*rendered_lines)
def langcode_ascii_renderable(width: int, text: str = "LangCode", font: str = "ansi_shadow", gradient: str = "dark_to_light") -> Group:
    if width < 60:
        return Group(Align.center(Text(text, style="bold green"), width=width))
    return _ascii_gradient_lines(width, text, font, gradient)


def print_langcode_ascii(
    console: Console,
    text: str = "LangCode",
    font: str = "ansi_shadow",
    gradient: str = "dark_to_light",
) -> None:
    width = getattr(console.size, "width", 80)
    console.print(langcode_ascii_renderable(width, text, font, gradient))


def session_banner(
    provider: Optional[str],
    project_dir: Path,
    title_text: str,
    *,
    interactive: bool = False,
    apply: bool = False,
    test_cmd: Optional[str] = None,
    tips: Optional[List[str]] = None,
    model_info: Optional[Dict[str, Any]] = None,
    router_enabled: bool = False,
    deep_mode: bool = False,
    command_name: Optional[str] = None,
) -> Panel:
    title = Text(title_text, style="bold magenta")
    body = Text()

    body.append("Provider: ", style="bold")
    if provider and provider.strip() and " " not in provider:
        body.append(provider.upper())
    else:
        body.append((provider or "not set"), style="dim")
    body.append("\n")

    body.append("Project:  ", style="bold")
    body.append(str(project_dir))

    badge = Text()
    if router_enabled:
        badge.append("  [ROUTER ON]", style="bold green")
    if command_name:
        badge.append(f"  [{command_name}]", style="bold blue")
    if deep_mode:
        badge.append("  [DEEP MODE]", style="bold magenta")
    if apply:
        badge.append("  [APPLY MODE]", style="bold red")
    if test_cmd:
        badge.append(f"  tests: {test_cmd}", style="italic")
    if badge:
        body.append("\n")
        body.append_text(badge)

    if model_info:
        body.append("\n")
        model_line = (
            f"Model: {model_info.get('model_name', '(unknown)')} "
            f"[{model_info.get('langchain_model_name', '?')}]"
            f" | priority={model_info.get('priority_used','balanced')}"
        )
        body.append(model_line, style="dim")

    if interactive:
        body.append("\n\n")
        body.append("Type your request. /clear to redraw, /select to change mode, /exit or /quit to quit. Ctrl+C also exits.\n", style="dim")

    if tips:
        body.append("\n")
        for item in tips:
            body.append(item + "\n", style="dim")

    return Panel(
        body,
        title=title,
        subtitle=Text("ReAct | Deep | Tools | Safe Edits", style="dim"),
        border_style="green",
        padding=(1, 2),
        box=box.HEAVY,
    )


def pause_if_in_launcher() -> None:
    """If we were launched from the selection hub, wait for Enter before redrawing it."""
    if in_selection_hub():
        console.print(Rule(style="green"))
        console.input("[dim]Press Enter to return to the launcher...[/dim]")


def print_session_header(
    title: str,
    provider: Optional[str],
    project_dir: Path,
    *,
    interactive: bool = False,
    apply: bool = False,
    test_cmd: Optional[str] = None,
    tips: Optional[List[str]] = None,
    model_info: Optional[Dict[str, Any]] = None,
    router_enabled: bool = False,
    deep_mode: bool = False,
    command_name: Optional[str] = None,
) -> None:
    console.clear()
    print_langcode_ascii(console, text="LangCode", font="ansi_shadow", gradient="dark_to_light")
    console.print(
        session_banner(
            provider,
            project_dir,
            title,
            interactive=interactive,
            apply=apply,
            test_cmd=test_cmd,
            tips=tips,
            model_info=model_info,
            router_enabled=router_enabled,
            deep_mode=deep_mode,
            command_name=command_name,
        )
    )
    console.print(Rule(style="green"))


def looks_like_markdown(text: str) -> bool:
    """Heuristic: decide if the model output is Markdown."""
    if "```" in text:
        return True
    if re.search(r"(?m)^\s{0,3}#{1,6}\s", text):
        return True
    if re.search(r"(?m)^\s{0,3}[-*+]\s+", text):
        return True
    if re.search(r"(?m)^\s{0,3}\d+\.\s+", text):
        return True
    if re.search(r"`[^`]+`", text) or re.search(r"\*\*[^*]+\*\*", text):
        return True
    return False


def to_text(content: Any) -> str:
    """Coerce Claude-style content blocks (list[dict|str]) into a single string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                val = item.get("text") or item.get("data") or item.get("content")
                if isinstance(val, str):
                    parts.append(val)
            elif isinstance(item, str):
                parts.append(item)
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def normalize_chat_history_for_anthropic(history: List) -> List:
    """Return a copy of history with str content only (prevents .strip() on lists)."""
    out: List = []
    for msg in history:
        try:
            content = getattr(msg, "content", "")
            out.append(msg.__class__(content=to_text(content)))
        except Exception:
            out.append(msg.__class__(content=str(getattr(msg, "content", ""))))
    return out


def panel_agent_output(text: str, title: str = "Agent", model_label: Optional[str] = None) -> Panel:
    """
    Render agent output full-width, with clean wrapping and proper Markdown
    when appropriate. This avoids the 'half-cut' panel look.
    """
    text = (text or "").rstrip()

    if looks_like_markdown(text):
        body = Markdown(text)
    else:
        t = Text.from_ansi(text) if "\x1b[" in text else Text(text)
        t.no_wrap = False
        t.overflow = "fold"
        body = t

    return Panel(
        body,
        title=title,
        border_style="cyan",
        box=box.ROUNDED,
        padding=(1, 2),
        expand=True,
        subtitle=(Text(f"Model: {model_label}", style="dim") if model_label else None),
        subtitle_align="right",
    )


def panel_router_choice(info: Dict[str, Any]) -> Panel:
    if not info:
        body = Text("Router active, but no model info available.", style="dim")
    else:
        name = info.get("model_name", "(unknown)")
        langchain_name = info.get("langchain_model_name", "?")
        provider = info.get("provider", "?").upper()
        priority = info.get("priority_used", "balanced")
        latency = info.get("latency_tier", "?")
        rs = info.get("reasoning_strength", "?")
        ic = info.get("input_cost_per_million", "?")
        oc = info.get("output_cost_per_million", "?")
        ctx = info.get("context_window", "?")
        body = Text.from_markup(
            f"[bold]Router:[/bold] {provider} -> [bold]{name}[/bold] [dim]({langchain_name})[/dim]\n"
            f"[dim]priority={priority} | latency_tier={latency} | reasoning={rs}/10 | "
            f"cost=${ic}M in/${oc}M out | ctx={ctx} tokens[/dim]"
        )
    return Panel.fit(body, title="Model Selection", border_style="green", box=box.ROUNDED, padding=(0, 1))


def show_loader():
    """Spinner that doesn't interfere with interactive prompts (y/n, input, click.confirm)."""
    return console.status("[bold]Processing...[/bold]", spinner="dots", spinner_style="green")