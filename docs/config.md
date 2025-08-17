# Configuration

The configuration module is responsible for setting up the language model provider and selecting the appropriate model for the agent.

## `resolve_provider`

This function determines which LLM provider to use. It prioritizes the provider specified via the `--llm` command-line option. If that's not provided, it falls back to the `LLM_PROVIDER` environment variable. If neither is set, it defaults to `gemini`.

**Priority Order:**
1. `--llm` CLI option (`anthropic` or `gemini`)
2. `LLM_PROVIDER` environment variable (`anthropic` or `gemini`)
3. Default: `gemini`

## `get_model`

This function returns an instance of the appropriate LangChain chat model based on the resolved provider.

- **`anthropic`**: Returns a `ChatAnthropic` instance using the `claude-3-7-sonnet-2025-05-14` model.
- **`gemini`**: Returns a `ChatGoogleGenerativeAI` instance using the `gemini-2.5-pro` model.

Both models are initialized with a `temperature` of `0.2` to encourage more deterministic and focused outputs suitable for coding tasks.
