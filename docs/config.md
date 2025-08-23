<div align="center">
  <img src="../assets/logo.png" alt="LangCode Logo" width="180" />
  <h1><b>LangCode</b></h1>
  <p><b>The only CLI you need.</b></p>
</div>

# Configuration

The configuration module is responsible for configuring the language model provider, selecting the appropriate model for the agent, and managing model selection logic. It defines the `resolve_provider` and `get_model` functions, which are used to determine the LLM provider and retrieve a LangChain chat model instance, respectively. It also includes the `IntelligentLLMRouter` class, which provides intelligent routing of requests to different LLMs based on query complexity and other factors.

## `resolve_provider`

The `resolve_provider` function determines which LLM provider to use. It prioritizes the provider specified via the `--llm` command-line option. If that option is not provided, it falls back to the `LLM_PROVIDER` environment variable. If neither is set, it defaults to `gemini`. This function ensures that the LLM provider is consistently resolved across the application.

**Configuration precedence:**

The LLM provider is resolved in the following order of precedence:

1.  `--llm` CLI option (`anthropic` or `gemini`)
2.  `LLM_PROVIDER` environment variable (`anthropic` or `gemini`)
3.  Default: `gemini`

This allows users to override the default provider either through the command line or by setting an environment variable.



This function determines which LLM provider to use. It prioritizes the provider specified via the `--llm` command-line option. If that's not provided, it falls back to the `LLM_PROVIDER` environment variable. If neither is set, it defaults to `gemini`.

**Priority Order:**
1. `--llm` CLI option (`anthropic` or `gemini`)
2. `LLM_PROVIDER` environment variable (`anthropic` or `gemini`)
3. Default: `gemini`

## `get_model`

The `get_model` function returns an instance of the appropriate LangChain chat model based on the resolved provider and an optional query. If a query is provided, the function uses the `IntelligentLLMRouter` to select the optimal model based on the query complexity and the specified priority. If no query is provided, the function returns a default model for the given provider.

- **`anthropic`**: Returns a `ChatAnthropic` instance using the `claude-3-7-sonnet-2025-05-14` model as the default. This model is well-suited for tasks that require creative text generation and complex reasoning.
- **`gemini`**: Returns a `ChatGoogleGenerativeAI` instance using the `gemini-2.0-flash` model as the default. This model is known for its strong performance on a wide range of tasks, including code generation and natural language understanding.

All models are initialized with a `temperature` of `0.2` to encourage more deterministic and focused outputs suitable for coding tasks. A lower temperature value reduces the randomness of the model's output, making it more predictable and consistent. The function also utilizes a lightweight model cache to improve performance by reusing previously created model instances.


This function returns an instance of the appropriate LangChain chat model based on the resolved provider and an optional query. If a query is provided, the function uses the `IntelligentLLMRouter` to select the optimal model based on the query complexity and the specified priority. If no query is provided, the function returns a default model for the given provider.

- **`anthropic`**: Returns a `ChatAnthropic` instance using the `claude-3-7-sonnet-2025-05-14` model as the default.
- **`gemini`**: Returns a `ChatGoogleGenerativeAI` instance using the `gemini-2.0-flash` model as the default.

All models are initialized with a `temperature` of `0.2` to encourage more deterministic and focused outputs suitable for coding tasks. The function also utilizes a lightweight model cache to improve performance by reusing previously created model instances.
