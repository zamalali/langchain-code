from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()

def resolve_provider(cli_llm: str | None) -> str:
    if cli_llm:
        return cli_llm.lower()
    env = os.getenv("LLM_PROVIDER", "gemini").lower()
    if env not in {"gemini", "anthropic"}:
        env = "gemini"
    return env

def get_model(provider: str):
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-3-7-sonnet-2025-05-14", temperature=0.2)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)
    raise ValueError(f"Unknown provider: {provider}")
