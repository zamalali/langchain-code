from __future__ import annotations
import mimetypes
import re
from pathlib import Path
from typing import AsyncIterable, Iterable, List, Optional, Callable, Any, Dict

import asyncio
import threading

from genai_processors import content_api, streams
from genai_processors.contrib.langchain_model import LangChainModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool

def _img_part(path: Path) -> content_api.ProcessorPart:
    mt, _ = mimetypes.guess_type(path.name)
    if mt not in {"image/png", "image/jpeg", "image/gif"}:
        raise ValueError(f"Unsupported image format: {path}. Supported: PNG, JPEG, GIF")
    data = path.read_bytes()
    return content_api.ProcessorPart(data, mimetype=mt, role="user")


def _text_part(text: str, role: str = "user") -> content_api.ProcessorPart:
    return content_api.ProcessorPart(text, mimetype="text/plain", role=role)


def build_processor(
    model: BaseChatModel,
    system_instruction: Optional[str] = None,
    prompt_template: Optional[ChatPromptTemplate] = None,
) -> LangChainModel:
    si = _text_part(system_instruction, role="system") if system_instruction else None
    return LangChainModel(
        model=model,
        system_instruction=(si,) if si else (),
        prompt_template=prompt_template,
    )


async def stream_processor(
    model: BaseChatModel,
    text: Optional[str] = None,
    images: Iterable[Path] = (),
    system_instruction: Optional[str] = None,
    prompt_template: Optional[ChatPromptTemplate] = None,
) -> AsyncIterable[str]:
    """Yield streamed text chunks from the LangChain model via GenAI Processors."""
    proc = build_processor(model, system_instruction=system_instruction, prompt_template=prompt_template)
    parts: List[content_api.ProcessorPart] = []
    if text:
        parts.append(_text_part(text))
    for p in images:
        parts.append(_img_part(Path(p)))
    input_stream = streams.stream_content(parts)
    async for out in proc(input_stream):
        if content_api.is_text(out.mimetype):
            yield out.text


def _run_coro_sync(coro_func: Callable[[], Any]) -> Any:
    """Run a coroutine safely from sync code, even if an event loop is already running."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if not loop or not loop.is_running():
        return asyncio.run(coro_func())

    result_box: Dict[str, Any] = {}
    error_box: Dict[str, BaseException] = {}
    done = threading.Event()

    def runner():
        new_loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(new_loop)
            result_box["value"] = new_loop.run_until_complete(coro_func())
        except BaseException as e:
            error_box["error"] = e
        finally:
            try:
                new_loop.close()
            finally:
                done.set()

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    done.wait()
    if "error" in error_box:
        raise error_box["error"]
    return result_box.get("value")


def run_stream_to_text(
    model: BaseChatModel,
    *,
    text: Optional[str],
    images: Iterable[Path] = (),
    system_instruction: Optional[str] = None,
    prompt_template: Optional[ChatPromptTemplate] = None,
) -> str:
    """Collect the async stream to a single string (for use as a LangChain tool return)."""
    async def _consume():
        chunks: List[str] = []
        async for piece in stream_processor(
            model,
            text=text,
            images=images,
            system_instruction=system_instruction,
            prompt_template=prompt_template,
        ):
            chunks.append(piece)
        return "".join(chunks)

    return _run_coro_sync(_consume)


def _discover_images(
    root: Path,
    raw_hints: List[str],
    text: Optional[str],
) -> List[Path]:
    """
    Resolve/locate image files using:
      1) Provided paths (absolute/relative) resolved against root.
      2) If any do not exist, search by stem and common image extensions.
      3) If no paths given, infer names from `text` (explicit filenames or 'the <stem> image').
    Returns a list of resolved Path objects (deduped), preferring the shallowest match.
    """
    exts = (".png", ".jpg", ".jpeg", ".gif")
    resolved: List[Path] = []

    def _resolve_one(raw: str) -> Optional[Path]:

        p = Path(raw)
        if not p.is_absolute():
            p = (root / p).resolve()
        if p.exists() and p.is_file():
            return p

        stem = Path(raw).stem or raw
        candidates: List[Path] = []

        name = Path(raw).name
        for found in root.rglob(name):
            if found.is_file():
                candidates.append(found)

        for ext in exts:
            for found in root.rglob(f"*{stem}*{ext}"):
                if found.is_file():
                    candidates.append(found)

        if not candidates:
            return None

        uniq, seen = [], set()
        for c in candidates:
            s = str(c)
            if s not in seen:
                seen.add(s)
                uniq.append(c)
        uniq.sort(key=lambda x: (len(x.relative_to(root).parts), len(str(x))))
        return uniq[0]

    hints = list(raw_hints or [])

    if not hints and text:
        filenames = re.findall(r'([\w.\-]+?\.(?:png|jpe?g|gif))', text, flags=re.I)
        if filenames:
            hints.extend(filenames)
        else:
            stems = re.findall(r'(?:\bimage\s+|\bthe\s+)([\w.\-]+)', text, flags=re.I)
            hints.extend(stems)

    for raw in hints:
        p = _resolve_one(raw)
        if p:
            resolved.append(p)

    deduped, seen = [], set()
    for p in resolved:
        s = str(p)
        if s not in seen:
            seen.add(s)
            deduped.append(p)
    return deduped


def make_process_multimodal_tool(project_dir: str, model: BaseChatModel):
    """
    Factory: returns a LangChain tool named `process_multimodal` that:
      - accepts `text` and optional `image_paths`
      - auto-discovers images by filename or stem anywhere under project_dir
      - resolves relative paths against project_dir
      - gracefully handles fabricated absolute paths (e.g., /tmp/...)
    """
    root = Path(project_dir).resolve()

    @tool("process_multimodal", return_direct=False)
    def process_multimodal(text: str, image_paths: List[str] = []) -> str:
        """
        Process text + optional images (PNG/JPEG/GIF) with the underlying LLM and
        return the streamed text as a single string.

        Smart path resolution:
        - Pass bare filenames (e.g., "deepgit.png") or just a stem (e.g., "deepgit").
        - If a provided path does not exist, we search the project directory recursively
          for a matching image by name/stem (.png/.jpg/.jpeg/.gif).
        - If `image_paths` is empty, we try to infer filenames from `text`.
        - Relative paths are resolved against the project root; fabricated /tmp paths are ignored.
        """
        resolved = _discover_images(root, image_paths or [], text)
        if not resolved:
            tried = image_paths or []
            return (
                "Image file(s) not found or discoverable. "
                f"Tried to resolve {tried or '[inferred from text]'} under {root}."
            )
        try:
            return run_stream_to_text(model, text=text, images=resolved)
        except ValueError as e:
            return f"Error processing multimodal input: {e}"

    return process_multimodal
