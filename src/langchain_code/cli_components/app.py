from __future__ import annotations

import logging
import warnings

import typer
from click.exceptions import UsageError
from typer.core import TyperGroup

from .constants import APP_HELP


warnings.filterwarnings("ignore", message=r"typing\.NotRequired is not a Python type.*")
warnings.filterwarnings("ignore", category=UserWarning, module=r"pydantic\._internal.*")


class _DefaultToChatGroup(TyperGroup):
    """Route unknown subcommands to `chat --inline ...` so `langcode \"Hi\"` works."""

    def resolve_command(self, ctx, args):
        if args and not args[0].startswith("-"):
            try:
                return super().resolve_command(ctx, args)
            except UsageError:
                chat_cmd = self.get_command(ctx, "chat")
                if chat_cmd is None:
                    raise
                if "--inline" not in args:
                    args = ["--inline", *args]
                return chat_cmd.name, chat_cmd, args
        return super().resolve_command(ctx, args)


for _name in (
    "langchain_google_genai",
    "langchain_google_genai.chat_models",
    "tenacity",
    "tenacity.retry",
    "httpx",
    "urllib3",
    "google",
):
    _log = logging.getLogger(_name)
    _log.setLevel(logging.CRITICAL)
    _log.propagate = False


app = typer.Typer(
    cls=_DefaultToChatGroup,
    add_completion=False,
    help=APP_HELP.strip(),
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)

