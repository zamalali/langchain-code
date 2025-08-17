from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json
import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

@dataclass
class RouteMeta:
    mode: str                 
    test_cmd: str | None = None
    apply: bool = False
    log_path: Path | None = None
    image_paths_count: int = 0

class Router:
    """
    Router that prefers an *LLM classifier* (cheap model) and falls back to fast heuristics.
    Sticky per-session; can escalate to 'deep' if signals warrant it.
    """
    def __init__(self, classifier_model: Optional[BaseChatModel] = None, debug: bool = False):
        self.classifier = classifier_model
        self.debug = debug

        self._HEAVY_LEN = 500
        self._HEAVY_LOG_BYTES = 50_000

        self._prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a router for a code assistant. Decide the engine to use:\n"
             "- 'react' for quick Q&A, small edits, single-file changes, short logs, no deep research.\n"
             "- 'deep'  for planning, multi-step work, web/MCP research, multi-file edits, large logs, refactors, ambiguous tasks.\n"
             "Return STRICT JSON only: {\"engine\": \"react\"|\"deep\", \"confidence\": 0..1, \"rationale\": \"<short>\"}"),
            ("user", "{task}\n\nMode: {mode}\nHints: test_cmd={test_cmd} apply={apply} log_bytes={log_bytes} images={images}"),
        ])

    # ---------- public API ----------
    def choose(self, text: str, meta: RouteMeta) -> str:
        choice = self._classify(text, meta) or self._heuristic(text, meta)
        if self.debug:
            print(f"[router] choose -> {choice}")
        return choice

    def escalate_if_needed(self, current: str, text: str, meta: RouteMeta) -> str:
        if current == "deep":
            return "deep"
        # If LLM available, re-query; otherwise use a stronger heuristic threshold.
        new_choice = self._classify(text, meta)
        if not new_choice:
            if self._heavy_score(text, meta) >= 2.0:
                new_choice = "deep"
            else:
                new_choice = current
        if self.debug and new_choice != current:
            print(f"[router] escalate {current} -> {new_choice}")
        return new_choice

    # ---------- internals ----------
    def _classify(self, text: str, meta: RouteMeta) -> Optional[str]:
        if not self.classifier:
            return None
        try:
            size = 0
            if meta.log_path and meta.log_path.exists():
                try:
                    size = meta.log_path.stat().st_size
                except Exception:
                    size = 0

            msg = self._prompt.format_messages(
                task=text,
                mode=meta.mode,
                test_cmd=meta.test_cmd or "none",
                apply=str(meta.apply),
                log_bytes=str(size),
                images=str(meta.image_paths_count),
            )
            out = self.classifier.invoke(msg)
            raw = out.content if hasattr(out, "content") else str(out)
            # Extract JSON (be permissive if model wraps codefences)
            m = re.search(r"\{.*\}", raw, re.S)
            data = json.loads(m.group(0)) if m else json.loads(raw)
            engine = str(data.get("engine", "")).lower().strip()
            if engine in {"react", "deep"}:
                return engine
        except Exception:
            return None
        return None

    def _heuristic(self, text: str, meta: RouteMeta) -> str:
        return "deep" if self._heavy_score(text, meta) >= 1.2 else "react"

    def _heavy_score(self, text: str, meta: RouteMeta) -> float:
        score = 0.0
        if len(text) >= self._HEAVY_LEN: score += 1.2
        t = text.lower()
        if any(k in t for k in ("research", "search", "web", "tavily", "docs", "spec", "design")):
            score += 1.0
        if meta.mode == "fix" and meta.log_path:
            try:
                if meta.log_path.stat().st_size >= self._HEAVY_LOG_BYTES:
                    score += 1.2
            except Exception:
                pass
        if meta.image_paths_count >= 3:
            score += 0.5
        return score
