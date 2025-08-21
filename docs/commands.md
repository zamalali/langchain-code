# CLI Commands

## Available commands

*   **chat**: Opens an interactive chat with the agent.
    *   `--llm`: (Optional) Specifies the language model to use (anthropic | gemini).
    *   `--project-dir`: (Optional) Specifies the project directory. Defaults to the current working directory.
    *   `--mode`: (Optional) Specifies the chat mode (react | deep). Defaults to react.
    *   `--auto`: (Optional) Enables autonomy mode (deep mode only).
    *   `--apply`: (Optional) Enables apply mode, which applies writes and runs commands without interactive confirmation.

*   **feature**: Implements a feature end-to-end (plan → search → edit → verify).
    *   `request`: (Required) A description of the feature to implement (e.g., "Add a dark mode toggle in settings").
    *   `--llm`: (Optional) Specifies the language model to use (anthropic | gemini).
    *   `--project-dir`: (Optional) Specifies the project directory. Defaults to the current working directory.
    *   `--test-cmd`: (Optional) Specifies a test command to run (e.g., "pytest -q" or "npm test").
    *   `--apply`: (Optional) Enables apply mode, which applies writes and runs commands without interactive confirmation.

*   **fix**: Diagnoses & fixes a bug (trace → pinpoint → patch → test).
    *   `request`: (Optional) A description of the bug to fix (e.g., "Fix crash on image upload").
    *   `--log`: (Optional) Path to an error log or stack trace.
    *   `--llm`: (Optional) Specifies the language model to use (anthropic | gemini).
    *   `--project-dir`: (Optional) Specifies the project directory. Defaults to the current working directory.
    *   `--test-cmd`: (Optional) Specifies a test command to run (e.g., "pytest -q").
    *   `--apply`: (Optional) Enables apply mode, which applies writes and runs commands without interactive confirmation.

*   **analyze**: Analyzes any codebase and generates insights.
    *   `request`: (Required) A description of the analysis to perform (e.g., "What are the main components of this project?").
    *   `--llm`: (Optional) Specifies the language model to use (anthropic | gemini).
    *   `--project-dir`: (Optional) Specifies the project directory. Defaults to the current working directory.

*   **chat**: Opens an interactive chat with the agent.
    *   `--llm`: (Optional) Specifies the language model to use (anthropic | gemini).
    *   `--project-dir`: (Optional) Specifies the project directory. Defaults to the current working directory.
    *   `--apply`: (Optional) Enables apply mode, which applies writes and runs commands without interactive confirmation.
    *   `--mode`: (Optional) Specifies the chat mode (react | deep). Defaults to react.
    *   `--auto`: (Optional) Enables autonomy mode (deep mode only).

*   **feature**: Implements a feature end-to-end (plan → search → edit → verify).
    *   `request`: (Required) A description of the feature to implement (e.g., "Add a dark mode toggle in settings").
    *   `--llm`: (Optional) Specifies the language model to use (anthropic | gemini).
    *   `--project-dir`: (Optional) Specifies the project directory. Defaults to the current working directory.
    *   `--test-cmd`: (Optional) Specifies a test command to run (e.g., "pytest -q" or "npm test").
    *   `--apply`: (Optional) Enables apply mode, which applies writes and runs commands without interactive confirmation.

*   **fix**: Diagnoses & fixes a bug (trace → pinpoint → patch → test).
    *   `request`: (Optional) A description of the bug to fix (e.g., "Fix crash on image upload").
    *   `--log`: (Optional) Path to an error log or stack trace.
    *   `--llm`: (Optional) Specifies the language model to use (anthropic | gemini).
    *   `--project-dir`: (Optional) Specifies the project directory. Defaults to the current working directory.
    *   `--test-cmd`: (Optional) Specifies a test command to run (e.g., "pytest -q").
    *   `--apply`: (Optional) Enables apply mode, which applies writes and runs commands without interactive confirmation.

*   **analyze**: Analyzes any codebase and generates insights.
    *   `request`: (Required) A description of the analysis to perform (e.g., "What are the main components of this project?").
    *   `--llm`: (Optional) Specifies the language model to use (anthropic | gemini).
    *   `--project-dir`: (Optional) Specifies the project directory. Defaults to the current working directory.

## Beta Commands

*   None
