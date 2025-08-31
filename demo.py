import os
import sys
import asyncio
import platform
from typing import AsyncIterable, List

from dotenv import load_dotenv, dotenv_values

# Flexible import: some envs publish as genai_processors vs genaiprocessors
try:
    from genai_processors import content_api
    from genai_processors.contrib.langchain_model import LangChainModel

except ImportError:
    from genaiprocessors import content_api  
    from genai_processors.contrib.langchain_model import LangChainModel


from langchain_groq import ChatGroq


# ---- Config ----
MODEL_NAME = "llama-3.1-8b-instant"
TEMPERATURE = 0.2
MAX_TURNS = 24  # keep context bounded


def ensure_groq_key() -> None:
    """Ensure GROQ_API_KEY is available (also accept 'GROQ API KEY')."""
    load_dotenv()
    if os.environ.get("GROQ_API_KEY"):
        return
    vals = dotenv_values()
    spaced = vals.get("GROQ API KEY")
    if spaced:
        os.environ["GROQ_API_KEY"] = spaced
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError(
            "Groq API key not found. Put it in .env as GROQ_API_KEY=<key> "
            "or GROQ API KEY=<key>."
        )


async def make_turn_stream(
    history_parts: List[content_api.ProcessorPart],
    user_text: str,
) -> AsyncIterable[content_api.ProcessorPart]:
    """Yield full chat history + new user turn as ProcessorParts."""
    for p in history_parts:
        yield p
    yield content_api.ProcessorPart(
        user_text,
        mimetype="text/plain",
        role="user",
        metadata={"source": "cli"},
    )


def build_processor() -> LangChainModel:
    # LangChain ChatGroq LLM (streaming supported)
    llm = ChatGroq(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        max_retries=2,
        # timeout=None,  # uncomment to set a custom timeout
        # reasoning_format=None,  # for reasoning models only
    )

    # Optional: prepend a system message
    system = (
        content_api.ProcessorPart(
            "You are a concise, helpful assistant. Stream partial tokens promptly.",
            mimetype="text/plain",
            role="system",
        ),
    )

    # Wrap the LangChain model so we can stream via GenAI Processors
    return LangChainModel(model=llm, system_instruction=system)


def trim_history(history: List[content_api.ProcessorPart]) -> None:
    """Keep only the last N user/assistant turns (plus any system parts)."""
    systems = [p for p in history if p.role == "system"]
    convo = [p for p in history if p.role in ("user", "model")]

    # group into user->model pairs
    pairs = []
    i = 0
    while i < len(convo):
        if convo[i].role == "user":
            if i + 1 < len(convo) and convo[i + 1].role == "model":
                pairs.append((convo[i], convo[i + 1]))
                i += 2
            else:
                pairs.append((convo[i],))
                i += 1
        else:
            pairs.append((convo[i],))
            i += 1

    if len(pairs) > MAX_TURNS:
        pairs = pairs[-MAX_TURNS:]

    trimmed = []
    for tup in pairs:
        trimmed.extend(list(tup))

    history.clear()
    history.extend(systems + trimmed)


async def chat_repl() -> None:
    # Windows safety for some Python builds
    if platform.system() == "Windows":
        try:
            policy = asyncio.WindowsSelectorEventLoopPolicy()  # type: ignore[attr-defined]
            asyncio.set_event_loop_policy(policy)
        except Exception:
            pass

    ensure_groq_key()
    processor = build_processor()
    history: List[content_api.ProcessorPart] = []

    print(f"Groq streaming chat (LangChain: {MODEL_NAME})")
    print("Type your message and press Enter. Commands: /reset, /exit\n")

    while True:
        try:
            user = input("🗨️  You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user:
            continue
        if user.lower() in {"/exit", "exit", "quit", "q"}:
            print("Bye!")
            break
        if user.lower() in {"/reset", "reset"}:
            history.clear()
            print("✅ Context cleared.\n")
            continue

        turn_stream = make_turn_stream(history, user)

        # Stream assistant tokens; also collect them to append to history
        assistant_chunks: List[str] = []
        try:
            async for part in processor.call(turn_stream):
                if content_api.is_text(part.mimetype) and part.role == "model":
                    print(part.text, end="", flush=True)  # live token stream
                    assistant_chunks.append(part.text)
            print()  # newline after the model finishes
        except Exception as e:
            print(f"\n❌ Streaming error: {e}\n")
            continue

        # update history
        history.append(
            content_api.ProcessorPart(
                user,
                mimetype="text/plain",
                role="user",
                metadata={"source": "cli"},
            )
        )
        history.append(
            content_api.ProcessorPart(
                "".join(assistant_chunks),
                mimetype="text/plain",
                role="model",
                metadata={"model": MODEL_NAME},
            )
        )
        trim_history(history)
        print()  # spacing between turns


def main() -> None:
    try:
        asyncio.run(chat_repl())
    except KeyboardInterrupt:
        print("\nBye!")


if __name__ == "__main__":
    main()
