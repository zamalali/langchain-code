from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich.columns import Columns
from rich.table import Table

from ...cli_components.state import console
from ...cli_components.env import (
    bootstrap_env,
    tty_log_path,
    current_tty_id,
    mcp_target_path,
    global_env_path,
    count_env_keys_in_file,
)
from ...cli_components.launcher import list_ollama_models


def wrap(
    cmd: List[str] = typer.Argument(..., help="Command to run (e.g., pytest -q)"),
    project_dir: Path = typer.Option(Path.cwd(), "--project-dir", exists=True, file_okay=False),
    tty_id: Optional[str] = typer.Option(None, "--tty-id", help="Override session id (default: auto per TTY)"),
):
    log_path = tty_log_path(tty_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    console.print(Panel.fit(Text(f"Logging to: {log_path}", style="dim"), title="TTY Capture", border_style="cyan"))
    os.chdir(project_dir)

    if platform.system().lower().startswith("win"):
        with open(log_path, "a", encoding="utf-8", errors="ignore") as f:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                f.write(line)
            rc = proc.wait()
            raise typer.Exit(rc)

    import pty  # type: ignore

    with open(log_path, "a", encoding="utf-8", errors="ignore") as f:
        old_env = dict(os.environ)
        os.environ["LANGCODE_TTY_LOG"] = str(log_path)
        os.environ["LANGCODE_TTY_ID"] = tty_id or current_tty_id()

        def _tee(master_fd):
            data = os.read(master_fd, 1024)
            if data:
                try:
                    f.write(data.decode("utf-8", "ignore"))
                    f.flush()
                except Exception:
                    pass
            return data

        try:
            status = pty.spawn(cmd, master_read=_tee)
        finally:
            os.environ.clear()
            os.environ.update(old_env)
        raise typer.Exit(status >> 8)


def shell(
    project_dir: Path = typer.Option(Path.cwd(), "--project-dir", exists=True, file_okay=False),
    tty_id: Optional[str] = typer.Option(None, "--tty-id", help="Override session id (default: auto per TTY)"),
):
    sh = os.environ.get("SHELL") if platform.system().lower() != "windows" else os.environ.get("COMSPEC", "cmd.exe")
    if not sh:
        sh = "/bin/bash" if platform.system().lower() != "windows" else "cmd.exe"
    return wrap([sh], project_dir=project_dir, tty_id=tty_id)


def doctor(
    project_dir: Path = typer.Option(Path.cwd(), "--project-dir", exists=True, file_okay=False),
):
    bootstrap_env(project_dir, interactive_prompt_if_missing=False)

    def yes(x):
        return Text("• " + x, style="green")

    def no(x):
        return Text("• " + x, style="red")

    rows = []
    rows.append(yes(f"Python {sys.version.split()[0]} on {platform.platform()}"))

    for tool in ["git", "npx", "node", "ollama"]:
        rows.append(yes(f"{tool} found") if shutil.which(tool) else no(f"{tool} missing"))

    provider_keys = {
        "OPENAI_API_KEY": "OpenAI",
        "ANTHROPIC_API_KEY": "Anthropic",
        "GOOGLE_API_KEY": "Gemini",
        "GEMINI_API_KEY": "Gemini (alt)",
        "GROQ_API_KEY": "Groq",
        "TOGETHER_API_KEY": "Together",
        "FIREWORKS_API_KEY": "Fireworks",
        "PERPLEXITY_API_KEY": "Perplexity",
        "DEEPSEEK_API_KEY": "DeepSeek",
        "TAVILY_API_KEY": "Tavily (web search)",
    }

    provider_panel = Table.grid(padding=(0, 2))
    provider_panel.add_column("Provider")
    provider_panel.add_column("Status")
    for env, label in provider_keys.items():
        ok = env in os.environ and bool(os.environ.get(env, "").strip())
        provider_panel.add_row(
            label,
            ("[green]OK[/green]" if ok else "[red]missing[/red]") + f"  [dim]{env}[/dim]",
        )

    mcp_path = mcp_target_path(project_dir)
    mcp_status = "exists" if mcp_path.exists() else "missing"
    mcp_card = Panel(
        Text(f"{mcp_status}: {os.path.relpath(mcp_path, project_dir)}"),
        title="MCP",
        border_style=("green" if mcp_path.exists() else "red"),
    )

    ollama = shutil.which("ollama")
    if ollama:
        models = list_ollama_models()
        if models:
            oll_text = ", ".join(models[:6]) + (" ..." if len(models) > 6 else "")
        else:
            oll_text = "(none installed)"
        oll_card = Panel(Text(oll_text), title="Ollama models", border_style=("green" if models else "yellow"))
    else:
        oll_card = Panel(Text("ollama not found"), title="Ollama", border_style="red")

    gpath = global_env_path()
    gexists = gpath.exists()
    gkeys = count_env_keys_in_file(gpath) if gexists else 0
    gmsg = f"{'exists' if gexists else 'missing'}: {gpath}\nkeys: {gkeys}"
    global_card = Panel(Text(gmsg), title="Global .env", border_style=("green" if gexists else "red"))

    console.print(Panel(Align.left(Text.assemble(*[r + Text("\n") for r in rows])), title="System", border_style="cyan"))
    console.print(Panel(provider_panel, title="Providers", border_style="cyan"))
    console.print(Columns([mcp_card, oll_card, global_card]))
    console.print(
        Panel(
            Text("Tip: run 'langcode instr' to set project rules; edit environment via the launcher."),
            border_style="blue",
        )
    )


__all__ = ["wrap", "shell", "doctor"]
