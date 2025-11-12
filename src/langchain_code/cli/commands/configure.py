from __future__ import annotations

from pathlib import Path

import typer
from rich.panel import Panel
from rich.text import Text

from ...cli_components.display import print_session_header
from ...cli_components.env import edit_global_env_file, edit_env_file, load_global_env, load_env_files
from ...cli_components.state import console
from ...cli_components.instructions import edit_langcode_md


def env(
    global_: bool = typer.Option(False, "--global", "-g", help="Edit the global env file."),
    project_dir: Path = typer.Option(Path.cwd(), "--project-dir", exists=True, file_okay=False),
):
    if global_:
        print_session_header("LangCode | Global Environment", provider=None, project_dir=project_dir, interactive=False)
        edit_global_env_file()
        load_global_env(override_existing=True)
        console.print(Panel.fit(Text("Global environment loaded.", style="green"), border_style="green"))
    else:
        print_session_header("LangCode | Project Environment", provider=None, project_dir=project_dir, interactive=False)
        edit_env_file(project_dir)
        load_env_files(project_dir, override_existing=False)
        console.print(Panel.fit(Text("Project environment loaded.", style="green"), border_style="green"))


def edit_instructions(
    project_dir: Path = typer.Option(Path.cwd(), "--project-dir", exists=True, file_okay=False),
):
    print_session_header(
        "LangChain Code Agent | Custom Instructions",
        provider=None,
        project_dir=project_dir,
        interactive=False,
    )
    edit_langcode_md(project_dir)


__all__ = ["env", "edit_instructions"]

