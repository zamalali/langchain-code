<div align="center">
  <img src="../assets/logo.png" alt="LangCode Logo" width="180" />
  <h1><b>LangCode</b></h1>
  <p><b>The only CLI you need.</b></p>
</div>

# Command-Line Interface (CLI)

The `cli.py` module provides the command-line interface (CLI) for LangCode. It leverages the `typer` library to create a user-friendly, composable, and powerful CLI experience. The CLI provides commands for interacting with LangCode agents and workflows.

## Usage

The CLI serves as the primary entry point for interacting with LangCode agents and workflows. It provides a set of commands for various tasks, including chatting with an agent, implementing new features, fixing bugs, and analyzing codebases. The CLI is designed with a focus on user-friendliness and power, utilizing the `typer` library to ensure a seamless and intuitive experience. The `cli.py` and `cli_beta.py` files provide different implementations of the CLI, with `cli.py` offering a more stable and feature-rich experience, while `cli_beta.py` may contain experimental features or changes.

### Installation

To use the LangCode CLI, you must first install it. It is recommended to install it within a virtual environment. You can install it using pip:

```bash
pip install langcode
```

### Basic Concepts

Before diving into the commands, it's helpful to understand a few basic concepts:

- **LLM Provider**: LangCode uses Large Language Models (LLMs) to perform its tasks. You can choose between different providers, such as `anthropic` and `gemini`. The provider can be specified using the `--llm` option. If the `--router` option is enabled, LangCode will automatically select the most efficient LLM based on the query.
- **Project Directory**: LangCode operates within a project directory. This directory contains the codebase that the agent will work on. You can specify the project directory using the `--project-dir` option. It defaults to the current working directory.
- **Apply Mode**: When the `--apply` flag is set, LangCode automatically applies changes to the codebase and runs commands without requiring interactive confirmation. This mode is useful for automated workflows but should be used with caution as it bypasses manual review.
- **Test Command**: LangCode can run a test command to verify its changes. You can specify the test command using the `--test-cmd` option. This command is executed after the agent makes changes to the codebase, and the agent will analyze the output to determine if the changes were successful.
- **Router Mode**: When the `--router` flag is set, LangCode automatically routes the request to the most efficient LLM based on factors such as cost, speed, and quality. This allows LangCode to dynamically select the best LLM for each task, optimizing performance and cost-effectiveness.
- **Priority**: The `--priority` option allows you to specify the priority for the LLM router. Available options are `balanced`, `cost`, `speed`, and `quality`. This option allows you to fine-tune the router's behavior based on your specific needs.



The CLI provides a set of commands for interacting with LangCode agents and workflows. These commands enable you to perform various tasks, such as chatting with an agent, implementing new features, fixing bugs, and analyzing codebases.

### Commands

The LangCode CLI provides the following commands:

- **`chat`**: Opens an interactive chat session with the agent. This command initiates an interactive dialogue with the LangCode agent, allowing you to communicate in real-time by posing questions or providing instructions. The agent processes your input and responds accordingly, offering information, executing actions, or requesting further clarification. The `chat` command supports two modes of operation, `react` and `deep`, selectable via the `--mode` option. It also supports image processing via `/img` commands, enabling you to incorporate visual information into the conversation. When the `--router` option is enabled, LangCode automatically selects the most appropriate LLM for the task. 

  *Options:*
    * `--llm`: Specifies the language model provider (e.g., `anthropic`, `gemini`).
    * `--project-dir`: Sets the root directory for the project.
    * `--mode`: Selects the reasoning engine (`react` or `deep`). Defaults to `react`.
    * `--auto`: Enables autopilot mode (deep mode only), where the agent plans and acts without requiring user confirmation.
    * `--router`: Enables automatic routing to the most efficient LLM.
    * `--priority`: Sets the priority for the LLM router (`balanced`, `cost`, `speed`, or `quality`).
    * `--verbose`: Displays model selection panels when using the router.

  *Example:*
```bash
langcode chat --llm gemini --project-dir myproject
```
This starts a chat session using the Gemini language model and working within the `myproject` directory.

  *Example with router:*
```bash
langcode chat --router --priority cost --verbose
```
This starts a chat session using the LLM router with a cost priority, displaying model selection information.


- **`feature`**: Implements a new feature from a given request. - **`feature`**: Implements a new feature from a given request. This command automates the process of implementing a new feature in a codebase. You provide a description of the feature request, and the agent will automatically plan, search, edit, and verify the changes required to implement the feature. The agent leverages the ReAct framework to reason about the required steps and take actions accordingly. The `feature` command uses the `FEATURE_INSTR` instruction seed to guide the agent through the implementation process.

  *Options:*
    * `--llm`: Specifies the language model provider (e.g., `anthropic`, `gemini`).
    * `--project-dir`: Sets the root directory for the project.
    * `--test-cmd`: Specifies a test command to run after implementing the feature.
    * `--apply`: Applies the changes automatically without confirmation.
    * `--router`: Enables automatic routing to the most efficient LLM.
    * `--priority`: Sets the priority for the LLM router (`balanced`, `cost`, `speed`, or `quality`).
    * `--verbose`: Displays model selection panels when using the router.

  *Example:*
```bash
langcode feature "Add a new user authentication system" --apply --test-cmd "pytest"
```
This requests the implementation of a new user authentication system, applies the changes automatically, and runs pytest to verify the changes.
Example:
```bash
langcode feature "Add a new user authentication system" --apply --test-cmd "pytest"
```
This requests the implementation of a new user authentication system, applies the changes automatically, and runs pytest to verify the changes.

- **`fix`**: Fixes a bug based on a request and an optional error log. - **`fix`**: Fixes a bug based on a request and an optional error log. This command assists in diagnosing and fixing bugs within a codebase. You provide a description of the bug and an optional error log, and the agent will trace the error, pinpoint the cause, patch the code, and test the changes to ensure the bug is resolved. The agent leverages the ReAct framework to reason about the bug and take appropriate actions. The `fix` command uses the `BUGFIX_INSTR` instruction seed to guide the agent through the bug-fixing process.

  *Options:*
    * `--request`: Specifies the bug fix request (e.g., "Fix crash on image upload").
    * `--log`: Provides the path to an error log or stack trace.
    * `--llm`: Specifies the language model provider (e.g., `anthropic`, `gemini`).
    * `--project-dir`: Sets the root directory for the project.
    * `--test-cmd`: Specifies a test command to run after fixing the bug.
    * `--apply`: Applies the changes automatically without confirmation.
    * `--router`: Enables automatic routing to the most efficient LLM.
    * `--priority`: Sets the priority for the LLM router (`balanced`, `cost`, `speed`, or `quality`).
    * `--verbose`: Displays model selection panels when using the router.

  *Example:*
```bash
langcode fix "Resolve the issue with database connection" --project-dir myproject
```
This requests a fix for a database connection issue within the `myproject` directory.
Example:
```bash
langcode fix "Resolve the issue with database connection" --project-dir myproject
```
This requests a fix for a database connection issue within the `myproject` directory.



- **`chat`**: Opens an interactive chat session with the agent.  Supports image processing via `/img` commands.
Example:
```bash
langcode chat --llm gemini --project-dir myproject
```
This starts a chat session using the Gemini language model and working within the `myproject` directory.

- **`feature`**: Implements a new feature from a given request.
Example:
```bash
langcode feature "Add a new user authentication system" --apply --test-cmd "pytest"
```
This requests the implementation of a new user authentication system, applies the changes automatically, and runs pytest to verify the changes.

- **`fix`**: Fixes a bug based on a request and an optional error log.
Example:
```bash
langcode fix "Resolve the issue with database connection" --project-dir myproject
```
This requests a fix for a database connection issue within the `myproject` directory.

### Options

Each command supports a set of common options that allow you to customize the behavior of the agent and the execution of the command. These options provide fine-grained control over various aspects of the process. For detailed information about each option, refer to the command-specific documentation.

- **`--llm`**: Specifies the language model provider to use (e.g., `anthropic`, `gemini`). This option allows you to choose the LLM that best suits your needs. Different LLMs may have different strengths and weaknesses, so it's important to choose the right one for the task at hand.
- **`--project-dir`**: Sets the root directory for the project the agent will work on. This option allows you to specify the codebase that the agent will operate on. The agent will only be able to access files and directories within the project directory.
- **`--apply`**: Allows the agent to apply changes to files and run commands without interactive confirmation. This option enables automated workflows, but it should be used with caution, as it can lead to unintended consequences if the agent makes mistakes.
- **`--test-cmd`**: Provides a test command that the agent can run to verify its changes. This option allows you to specify a command that will be executed after the agent makes changes to the codebase. The agent will analyze the output of the test command to determine if the changes were successful.



Each command supports a set of common options:

- **`--llm`**: Specifies the language model provider to use (e.g., `anthropic`, `gemini`).
- **`--project-dir`**: Sets the root directory for the project the agent will work on.
- **`--apply`**: Allows the agent to apply changes to files and run commands without interactive confirmation.
- **`--test-cmd`**: Provides a test command that the agent can run to verify its changes.

## Error Handling

The CLI provides robust error handling to ensure a smooth user experience. In case of failures, the CLI provides informative error messages that are designed to be as helpful as possible. These messages indicate the type of error encountered and suggest potential solutions or debugging steps. The CLI leverages `typer`'s built-in error handling capabilities to provide clear and concise feedback to the user, making it easier to diagnose and resolve issues. The error handling is designed to be both informative and user-friendly, guiding you through the troubleshooting process.

### Common Errors

- **Invalid Command**: This error occurs when you enter an invalid command or option. The CLI will display a list of available commands and options to help you correct the error.
- **Missing Argument**: This error occurs when you forget to provide a required argument for a command. The CLI will display a message indicating which argument is missing.
- **File Not Found**: This error occurs when the CLI cannot find a specified file or directory. Make sure that the file or directory exists and that you have the correct path.
- **Permission Denied**: This error occurs when you do not have the necessary permissions to access a file or directory. Make sure that you have the correct permissions.
- **LLM Error**: This error occurs when there is an issue with the language model provider. Check your API key and make sure that the provider is available.



The CLI provides informative error messages to the user in case of failures. These messages are designed to be as helpful as possible, indicating the type of error encountered and suggesting potential solutions or debugging steps. The CLI leverages `typer`'s built-in error handling capabilities to provide clear and concise feedback to the user.

## Session UI

The CLI provides a rich and interactive session UI, powered by the `rich` library, to enhance the user experience. This includes:

- An engaging ASCII banner for the LangCode application, displayed upon startup. This banner adds a touch of personality to the CLI and helps to visually identify the application.
- A dynamic status panel that provides real-time information about the current provider, project directory, and other relevant session details. This panel keeps you informed about the current state of the CLI and helps you to quickly identify any issues.
- Formatted output from the agent, supporting markdown and code blocks for improved readability and clarity. This ensures that complex information, such as code snippets and detailed explanations, are presented in an easily digestible format.

### Customization

You can customize the appearance of the CLI by modifying the `rich` configuration. This allows you to change the colors, fonts, and other visual elements of the CLI to suit your preferences.



The CLI provides a rich and interactive session UI, powered by the `rich` library, to enhance the user experience. This includes:

- An engaging ASCII banner for the LangCode application, displayed upon startup.
- A dynamic status panel that provides real-time information about the current provider, project directory, and other relevant session details.
- Formatted output from the agent, supporting markdown and code blocks for improved readability and clarity. This ensures that complex information, such as code snippets and detailed explanations, are presented in an easily digestible format.
