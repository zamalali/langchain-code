from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Dict, Tuple

import typer
from rich.console import Console
from rich.panel import Panel
from rich.style import Style
from rich.text import Text
from rich.markdown import Markdown
from rich.rule import Rule
from rich import box
from rich.table import Table
from rich.theme import Theme
from pyfiglet import Figlet
from getpass import getpass

# Optional arrow-key UI
try:
    import questionary  # type: ignore
except Exception:
    questionary = None  # graceful fallback

from dotenv import load_dotenv, set_key, find_dotenv, dotenv_values

from .config import resolve_provider
from .agent.react import build_react_agent
from .workflows.feature_impl import FEATURE_INSTR
from .workflows.bug_fix import BUGFIX_INSTR
from langchain_core.messages import HumanMessage, AIMessage

# ----- Theme: LangCode green -------------------------------------------------
THEME = Theme(
    {
        "primary": "green3",
        "accent": "spring_green3",
        "muted": "grey58",
        "warn": "yellow3",
        "error": "red",
        "link": "cyan",
    }
)
console = Console(theme=THEME)

app = typer.Typer(add_completion=False, help="LangCode – ReAct + tools code agent CLI.")
PROMPT = "[primary]langcode[/primary] [muted]›[/muted] "

# Known API keys by provider (shown in status & key entry)
PROVIDERS: Dict[str, Tuple[str, str]] = {
    "gemini": ("GOOGLE_API_KEY", "Google Gemini"),
    "anthropic": ("ANTHROPIC_API_KEY", "Anthropic Claude"),
    "openai": ("OPENAI_API_KEY", "OpenAI"),
}

# ----- Visuals ---------------------------------------------------------------

def print_langcode_ascii(console: Console, text: str = "LangCode", font: str = "ansi_shadow", gradient: str = "dark_to_light") -> None:
    def _hex_to_rgb(h): h = h.lstrip("#"); return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    def _lerp(a, b, t): return int(a + (b - a) * t)
    def _interpolate_palette(palette, width):
        if width <= 1: return [palette[0]]
        out, steps_total = [], width - 1
        for x in range(width):
            pos = x / steps_total
            seg = min(int(pos * (len(palette) - 1)), len(palette) - 2)
            seg_start, seg_end = seg / (len(palette) - 1), (seg + 1) / (len(palette) - 1)
            local_t = (pos - seg_start) / (seg_end - seg_start + 1e-9)
            c1, c2 = _hex_to_rgb(palette[seg]), _hex_to_rgb(palette[seg + 1])
            rgb = tuple(_lerp(a, b, local_t) for a, b in zip(c1, c2))
            out.append("#{:02x}{:02x}{:02x}".format(*rgb))
        return out

    def _print_block_with_horizontal_gradient(lines, palette):
        width = max(len(line) for line in lines) if lines else 0
        ramp = _interpolate_palette(palette, width)
        for line in lines:
            t = Text()
            padded = line.ljust(width)
            for j, ch in enumerate(padded):
                t.append(ch if ch == " " else ch, Style(color=ramp[j], bold=True))
            console.print(t)

    fig = Figlet(font=font)
    lines = fig.renderText(text).rstrip("\n").splitlines()
    palette = ["#052e1e", "#064e3b", "#065f46", "#047857", "#059669", "#16a34a", "#22c55e", "#34d399"]
    if gradient == "light_to_dark":
        palette = list(reversed(palette))
    _print_block_with_horizontal_gradient(lines, palette)


def session_banner(provider: Optional[str], project_dir: Path, title_text: str, interactive: bool = False, apply: bool = False, tips: Optional[list[str]] = None) -> Panel:
    title = Text(title_text, style="bold magenta")
    body = Text()

    body.append("Provider: ", style="bold")
    if provider and provider.strip() and " " not in provider:
        body.append(provider.upper(), style="primary")
    else:
        body.append((provider or "not set"), style="muted")
    body.append("\n")

    body.append("Project:  ", style="bold")
    body.append(str(project_dir), style="primary")

    badge = Text()
    if apply:
        badge.append("  [APPLY MODE]", style="bold red")
    if badge:
        body.append("\n")
        body.append_text(badge)

    if interactive:
        body.append("\n\n")
        body.append("Type your request. /clear to redraw, /exit to quit. Ctrl+C also exits.\n", style="muted")

    if tips:
        body.append("\n")
        for t in tips:
            body.append(t + "\n", style="muted")

    return Panel(
        body,
        title=title,
        subtitle=Text("ReAct • Tools • Safe Edits", style="muted"),
        border_style="primary",
        padding=(1, 2),
        box=box.HEAVY,
    )


def _panel_agent_output(text: str, title: str = "Agent") -> Panel:
    body = Markdown(text) if ("```" in text or "\n#" in text) else Text(text)
    return Panel.fit(body, title=title, border_style="cyan", box=box.ROUNDED, padding=(0, 1))

# ----- Env / API key helpers -------------------------------------------------

def _env_path(project_dir: Path) -> Path:
    found = find_dotenv(usecwd=True)
    return Path(found) if found else project_dir / ".env"

def _mask(value: str) -> str:
    if not value:
        return ""
    return ("•" * max(len(value) - 4, 0)) + value[-4:]

def _status_table(active_provider: str, project_dir: Path) -> Panel:
    vals = dotenv_values(_env_path(project_dir))
    table = Table(box=box.MINIMAL_HEAVY_HEAD, show_lines=False, expand=True, header_style="bold")
    table.add_column("Provider", style="primary")
    table.add_column("Env Var", style="muted")
    table.add_column("Status")
    table.add_column("Value (masked)", style="muted")

    for key, (envvar, label) in PROVIDERS.items():
        present = os.getenv(envvar) or vals.get(envvar)
        status = "[green]✅[/green]" if present else "[error]❌[/error]"
        suffix = " [bold yellow](active)[/bold yellow]" if key == active_provider else ""
        table.add_row(
            f"{label}{suffix}",
            envvar,
            status,
            _mask(str(present) if present else "")
        )

    return Panel(table, title="API Keys", border_style="primary", padding=(1, 2), box=box.ROUNDED)

def _save_api_key(env_file: Path, env_var: str, secret: str) -> None:
    env_file.parent.mkdir(parents=True, exist_ok=True)
    set_key(str(env_file), env_var, secret)
    load_dotenv(dotenv_path=str(env_file), override=True)

# ----- Arrow-key UI helpers --------------------------------------------------

def _select(prompt: str, choices: list) -> str:
    # choices can be strings or questionary.Choice
    if questionary:
        return questionary.select(prompt, choices=choices, qmark="›").ask()
    # fallback: turn choices into plain strings (skip disabled)
    flat: list[str] = []
    for c in choices:
        title = getattr(c, "title", None)
        disabled = getattr(c, "disabled", None)
        if title is not None:
            if disabled:
                continue
            flat.append(title)
        else:
            flat.append(str(c))
    console.print(f"\n{prompt}")
    for i, c in enumerate(flat, 1):
        console.print(f"  {i}. {c}")
    while True:
        pick = typer.prompt("Enter number")
        if pick.isdigit() and 1 <= int(pick) <= len(flat):
            return flat[int(pick) - 1]
        console.print("[error]Invalid choice[/error]")

def _ask_text(prompt: str, default: str | None = None) -> str:
    if questionary:
        return questionary.text(prompt, default=default or "").ask()
    return typer.prompt(prompt, default=default or "")

def _ask_password(prompt: str) -> str:
    if questionary:
        return questionary.password(prompt).ask() or ""
    return getpass(prompt + ": ")

def _confirm(prompt: str, default: bool = False) -> bool:
    if questionary:
        return bool(questionary.confirm(prompt, default=default).ask())
    return typer.confirm(prompt, default=default)

# ----- Core agent utilities --------------------------------------------------

def _maybe_coerce_img_command(raw: str) -> str:
    s = raw.strip()
    if not s.startswith("/img"):
        return raw
    try:
        rest = s[len("/img"):].strip()
        if "::" in rest:
            paths_part, prompt_text = rest.split("::", 1)
            prompt_text = prompt_text.strip()
        else:
            paths_part, prompt_text = rest, ""
        paths = [p for p in paths_part.split() if p]
        return (
            f'Please call the tool "process_multimodal" with '
            f"image_paths={paths} and text={prompt_text!r}. After the tool returns, summarize the result."
        )
    except Exception:
        return raw

def _print_session_header(title: str, provider: Optional[str], project_dir: Path, *, interactive: bool = False, apply: bool = False, tips: Optional[list[str]] = None) -> None:
    console.clear()
    print_langcode_ascii(console, text="LangCode", font="ansi_shadow", gradient="dark_to_light")
    console.print(session_banner(provider, project_dir, title, interactive=interactive, apply=apply, tips=tips))
    console.print(Rule(style="primary"))

# ----- Interactive launcher --------------------------------------------------

def _ensure_provider_and_key(provider: str, project_dir: Path) -> str:
    envvar, label = PROVIDERS.get(provider, (None, None))
    if not envvar:
        return provider
    if os.getenv(envvar) or dotenv_values(_env_path(project_dir)).get(envvar):
        return provider
    console.print(Panel.fit(Text(f"{label} requires {envvar}.", style="bold"), border_style="warn", title="Missing API key"))
    if _confirm(f"Enter {label} API key now?"):
        secret = _ask_password(f"Enter {envvar}")
        if secret.strip():
            _save_api_key(_env_path(project_dir), envvar, secret.strip())
            console.print(Panel.fit(Text(f"{label} API key saved to .env", style="primary"), border_style="primary"))
        else:
            console.print("[error]Key not set.[/error]")
    return provider

def _provider_choices(project_dir: Path):
    vals = dotenv_values(_env_path(project_dir))
    have_openai = bool(os.getenv("OPENAI_API_KEY") or vals.get("OPENAI_API_KEY"))

    if questionary:
        items = [
            questionary.Choice(title="Gemini", value="gemini"),
            questionary.Choice(title="Anthropic", value="anthropic"),
        ]
        if have_openai:
            items.append(
                questionary.Choice(
                    title="OpenAI (coming soon)",
                    value="__openai_disabled__",
                    disabled="coming soon",
                )
            )
        items.append(questionary.Choice(title="Back", value="__back__"))
        return items
    else:
        items = ["Gemini", "Anthropic"]
        if have_openai:
            items.append("OpenAI (coming soon)")
        items.append("Back")
        return items

def _launcher() -> None:
    project_dir = Path.cwd()
    load_dotenv()
    provider = resolve_provider(None)  # from env or default
    apply = False

    while True:
        _print_session_header(
            "LangChain Code Agent • Launcher",
            provider,
            project_dir,
            interactive=False,
            apply=apply,
            tips=["↑/↓ to choose. Enter to confirm.", "Theme: LangChain green. Secure .env key management."],
        )
        console.print(_status_table(provider, project_dir))

        choice = _select(
            "What would you like to do?",
            [
                "Chat",
                "Feature",
                "Fix",
                "Choose Provider",
                "Enter API key",
                "Toggle Apply Mode",
                "Exit",
            ],
        )

        if choice == "Exit":
            console.print("\n[bold]Goodbye![/bold]")
            raise typer.Exit()

        if choice == "Choose Provider":
            pick = _select("Select provider", _provider_choices(project_dir))
            if pick in (None, "__back__", "Back"):
                continue
            if pick in ("__openai_disabled__", "OpenAI (coming soon)"):
                console.print(Panel.fit(Text("OpenAI selection is shown because a key exists, but provider support is not enabled yet in this build.", style="muted"), border_style="warn"))
                continue
            provider = _ensure_provider_and_key(pick.lower(), project_dir)
            # persist choice
            set_key(str(_env_path(project_dir)), "LLM_PROVIDER", provider)
            load_dotenv(dotenv_path=str(_env_path(project_dir)), override=True)
            continue

        if choice == "Enter API key":
            key_choice = _select(
                "Which provider key?",
                [
                    "Google Gemini (GOOGLE_API_KEY)",
                    "Anthropic Claude (ANTHROPIC_API_KEY)",
                    "OpenAI (OPENAI_API_KEY)",
                    "Custom...",
                    "Back",
                ],
            )
            if key_choice == "Back":
                continue
            if key_choice.startswith("Google"):
                envvar = "GOOGLE_API_KEY"
            elif key_choice.startswith("Anthropic"):
                envvar = "ANTHROPIC_API_KEY"
            elif key_choice.startswith("OpenAI"):
                envvar = "OPENAI_API_KEY"
            else:
                envvar = _ask_text("Enter env var name (e.g., HYPER_API_KEY)")
                if not envvar:
                    continue

            secret = _ask_password(f"Enter {envvar}")
            if not secret.strip():
                console.print("[error]No value entered. Key not saved.[/error]")
                continue
            _save_api_key(_env_path(project_dir), envvar, secret.strip())
            console.print(Panel.fit(Text(f"{envvar} saved to .env", style="primary"), border_style="primary"))
            continue

        if choice == "Toggle Apply Mode":
            apply = not apply
            state = "ON" if apply else "OFF"
            console.print(Panel.fit(Text(f"Apply mode is now {state}", style="bold"), border_style="magenta"))
            continue

        # Ensure provider key exists before starting a session
        provider = _ensure_provider_and_key(provider, project_dir)

        if choice == "Chat":
            _start_chat(provider, project_dir, apply)
        elif choice == "Feature":
            request = _ask_text('Describe the feature (e.g., "Add a dark mode toggle in settings")')
            if not request.strip():
                console.print("[error]Feature request cannot be empty[/error]")
                continue
            _run_feature(request, provider, project_dir, apply)
        elif choice == "Fix":
            request = _ask_text('Describe the bug (e.g., "Fix crash on image upload")')
            log_path = _ask_text("Path to error log or stack trace (optional)")
            log_path = log_path.strip()
            log = Path(log_path) if log_path else None
            if log and not log.exists():
                console.print(f"[warn]Warning:[/warn] {log} not found; continuing without a log.")
                log = None
            _run_fix(request or None, log, provider, project_dir, apply)

# ----- Public Typer entrypoints ----------------------------------------------

@app.callback(invoke_without_command=True)
def _root(ctx: typer.Context):
    """Run the interactive launcher when invoked without subcommands."""
    if ctx.invoked_subcommand is None:
        _launcher()

@app.command(help="Open an interactive chat with the agent.")
def chat(
    llm: Optional[str] = typer.Option(None, "--llm", help="anthropic | gemini"),
    project_dir: Path = typer.Option(Path.cwd(), "--project-dir", exists=True, file_okay=False),
    apply: bool = typer.Option(False, "--apply", help="Apply writes and run commands without interactive confirm."),
):
    _start_chat(resolve_provider(llm), project_dir, apply)

def _start_chat(provider: str, project_dir: Path, apply: bool) -> None:
    agent = build_react_agent(provider=provider, project_dir=project_dir, apply=apply)
    title = "LangChain Code Agent • Chat"
    _print_session_header(title, provider, project_dir, interactive=True, apply=apply)

    history: list = []
    try:
        while True:
            user = console.input(PROMPT).strip()
            if not user:
                continue
            low = user.lower()
            if low in {"cls", "clear", "/clear"}:
                _print_session_header(title, provider, project_dir, interactive=True, apply=apply)
                continue
            if low in {"exit", "quit", ":q", "/exit"}:
                console.print("\n[bold]Goodbye![/bold]")
                break

            coerced = _maybe_coerce_img_command(user)
            res = agent.invoke({"input": coerced, "chat_history": history})
            output = res.get("output", "") if isinstance(res, dict) else str(res)
            console.print(_panel_agent_output(output))

            if isinstance(res, dict) and res.get("router_meta"):
                meta = res["router_meta"]
                console.print(
                    Panel.fit(
                        Text(
                            f"Routed to: {meta.get('selected_llm')} "
                            f"(conf {meta.get('confidence'):.2f}, {meta.get('decision_source')})",
                            style="italic muted",
                        ),
                        border_style="magenta",
                        title="Router",
                        box=box.ROUNDED,
                    )
                )

            history.append(HumanMessage(content=coerced))
            history.append(AIMessage(content=output))
    except (KeyboardInterrupt, EOFError):
        console.print("\n[bold]Goodbye![/bold]")

@app.command(help="Implement a feature end-to-end (plan → search → edit → verify).")
def feature(
    request: str = typer.Argument(..., help='e.g. "Add a dark mode toggle in settings"'),
    llm: Optional[str] = typer.Option(None, "--llm", help="anthropic | gemini"),
    project_dir: Path = typer.Option(Path.cwd(), "--project-dir", exists=True, file_okay=False),
    apply: bool = typer.Option(False, "--apply", help="Apply writes and run commands without interactive confirm."),
):
    _run_feature(request, resolve_provider(llm), project_dir, apply)

def _run_feature(request: str, provider: str, project_dir: Path, apply: bool) -> None:
    agent = build_react_agent(provider=provider, project_dir=project_dir, apply=apply, instruction_seed=FEATURE_INSTR)
    _print_session_header("LangChain Code Agent • Feature", provider, project_dir, interactive=False, apply=apply)
    res = agent.invoke({"input": request})
    output = res.get("output", "") if isinstance(res, dict) else str(res)
    console.print(_panel_agent_output(output, title="Feature Result"))

@app.command(help="Diagnose & fix a bug (trace → pinpoint → patch → test).")
def fix(
    request: Optional[str] = typer.Argument(None, help='e.g. "Fix crash on image upload"'),
    log: Optional[Path] = typer.Option(None, "--log", exists=True, help="Path to error log or stack trace."),
    llm: Optional[str] = typer.Option(None, "--llm", help="anthropic | gemini"),
    project_dir: Path = typer.Option(Path.cwd(), "--project-dir", exists=True, file_okay=False),
    apply: bool = typer.Option(False, "--apply", help="Apply writes and run commands without interactive confirm."),
):
    _run_fix(request, log, resolve_provider(llm), project_dir, apply)

def _run_fix(
    request: Optional[str],
    log: Optional[Path],
    provider: str,
    project_dir: Path,
    apply: bool
) -> None:
    agent = build_react_agent(provider=provider, project_dir=project_dir, apply=apply, instruction_seed=BUGFIX_INSTR)
    bug_input = request or ""
    if log:
        try:
            bug_input += "\n\n--- ERROR LOG ---\n" + Path(log).read_text(encoding="utf-8")
        except Exception:
            console.print(f"[warn]Warning:[/warn] Unable to read {log}; continuing without its content.")

    _print_session_header("LangChain Code Agent • Fix", provider, project_dir, interactive=False, apply=apply)
    res = agent.invoke({"input": bug_input.strip() or "Fix the bug using the provided log."})
    output = res.get("output", "") if isinstance(res, dict) else str(res)
    console.print(_panel_agent_output(output, title="Fix Result"))

def main() -> None:
    app()

if __name__ == "__main__":
    main()
