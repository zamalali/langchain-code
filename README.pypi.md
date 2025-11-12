<p align="center">
  <img src="https://raw.githubusercontent.com/zamalali/langchain-code/main/assets/logo.png" alt="LangCode Logo" width="160" />
</p>

<h1 align="center">LangCode</h1>

<p align="center"><strong>The only CLI you'll ever need!</strong></p>

![LangCode Demo](https://raw.githubusercontent.com/zamalali/langchain-code/main/assets/cmd.png)

[![PyPI version](https://badge.fury.io/py/langchain-code.svg)](https://pypi.org/project/langchain-code/) [![PyPI Downloads](https://static.pepy.tech/personalized-badge/langchain-code?period=total&units=international_system&left_color=black&right_color=green&left_text=downloads)](https://pepy.tech/project/langchain-code) ![Python Versions](https://img.shields.io/pypi/pyversions/langchain-code.svg) [![License](https://img.shields.io/github/license/zamalali/langchain-code)](https://github.com/zamalali/langchain-code/blob/main/LICENSE) [![Docker Build](https://github.com/zamalali/langchain-code/actions/workflows/docker-build.yml/badge.svg)](https://github.com/zamalali/langchain-code/actions/workflows/docker-build.yml) [![Docker Pulls](https://img.shields.io/docker/pulls/at384/langchain-code)](https://hub.docker.com/r/at384/langchain-code) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

LangCode is the "one-key" developer CLI that unifies **Gemini, Anthropic Claude, OpenAI, and Ollama** with **ReAct & Deep modes**—fully inline, right in your terminal.

---

## Get Started

### Installation:

```bash
pip install langchain-code
```

### Launch the Interactive Launcher:

Just type `langcode` in your terminal and hit Enter. This opens a user-friendly interactive menu where you can easily configure your session and access various functionalities without needing to remember specific command-line arguments. See the image shown above.

---

## Interactive Mode

The interactive mode serves as the central hub for all your coding tasks. It allows you to:

* **Choose a Command:** Select what you want to do: `chat`, `feature`, `fix`, or `analyze`.
* **Configure the Engine:** Pick between `react` (fast and efficient) and `deep` (for complex tasks).
* **Enable Smart Routing:** Let LangCode automatically select the best LLM for each task.
* **Set the Priority:** Optimize for `cost`, `speed`, or `quality` when using smart routing.
* **Manage Autopilot:** Enable fully autonomous mode for the Deep Agent (use with caution!).
* **Toggle Apply Mode:** Allow LangCode to automatically write changes to your file system.
* **Select an LLM:** Explicitly choose between Anthropic and Google Gemini, or let LangCode decide.
* **Specify the Project Directory:** Tell LangCode where your codebase is located.
* **Edit Environment Variables:** Quickly add or modify API keys and other settings in your `.env` file.
* **Customize Instructions:** Open the `.langcode/langcode.md` file to add project-specific guidelines.
* **Configure MCP Servers:** Set up Model Context Protocol (MCP) servers for advanced tool integration.
* **Edit Language Code:** Modify the core language code directly from the main window.
* **Specify MCP Servers:** Configure Model Context Protocol (MCP) servers for advanced tool integration.
* **Define a Test Command:** Specify a command to run after making changes (e.g., `pytest -q`).
* **Access Help:** Press `h` to toggle help and `q` or `Esc` to quit.

---

## Core Commands

While the interactive launcher is the recommended way to use LangCode, you can also use the following commands directly from the terminal:

* `langcode chat`: Starts an interactive chat session.
* `langcode feature`: Implements a new feature.
* `langcode fix`: Fixes a bug.
* `langcode analyze`: Analyzes the codebase.
* `langcode instr`: Opens the project instructions file.

---

## Install & Run

```bash
pip install langchain-code
langcode
```

---

## Contributing

Issues and PRs are welcome. Please open an issue to discuss substantial changes before submitting a PR. See [CONTRIBUTING.md](https://github.com/zamalali/langchain-code/blob/main/docs/CONTRIBUTING.md) for guidelines.

---

## License

MIT. See [LICENSE](https://github.com/zamalali/langchain-code/blob/main/LICENSE).

---

## Acknowledgments

LangCode draws inspiration from the design and developer experience of Google's Gemini CLI and Anthropic's Claude Code, unified into a single, streamlined tool.
