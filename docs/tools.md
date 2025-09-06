<div align="center">
  <img src="../assets/logo.png" alt="LangCode Logo" width="180" />
  <h1 align="center">LangCode</h1>

  <p align="center"><i><b>The only CLI you'll ever need!</b></i></p>
</div>

# Tools

The tools module provides a collection of tools that the agent can use to perform various tasks. These tools enable the agent to interact with the environment, access information, and perform computations.

## Available Tools

- **Files:** Provides functionalities for listing directories, finding files using glob patterns, reading file contents, editing files by replacing snippets, writing new files, and deleting files.
- **Search:** Enables searching for regex patterns within files under a specified directory.
- **Shell:** Allows running shell commands in the project directory, facilitating tasks like listing files, searching file contents, and running project-specific commands.
- **Web:** Offers a search engine optimized for comprehensive, accurate, and trusted results, useful for answering questions about current events.
- **Multimodal:** Processes text and optional images with the underlying LLM.
- **Planning:** Creates and updates structured todo lists for task management.
- **GitHub:** Provides functionalities for interacting with GitHub repositories, including creating, updating, and searching repositories, managing issues and pull requests, and pushing files.
- **Mermaid:** Renders Mermaid syntax to a PNG file.

- **Files:** Provides functionalities for listing directories, finding files using glob patterns, reading file contents, editing files by replacing snippets, writing new files, and deleting files.
- **Search:** Enables searching for regex patterns within files under a specified directory.
- **Shell:** Allows running shell commands in the project directory, facilitating tasks like listing files, searching file contents, and running project-specific commands.
- **Web:** Offers a search engine optimized for comprehensive, accurate, and trusted results, useful for answering questions about current events.
- **Multimodal:** Processes text and optional images with the underlying LLM.
- **Planning:** Creates and updates structured todo lists for task management.
- **GitHub:** Provides functionalities for interacting with GitHub repositories, including creating, updating, and searching repositories, managing issues and pull requests, and pushing files.
- **Mermaid:** Renders Mermaid syntax to a PNG file.

## Tool Usage Guidelines

- Always use tools rather than guessing.
- For file edits, show exactly what changed.
- Include relevant command outputs in your response.
- Keep responses focused and actionable.

## Adding New Tools

To add a new tool, you need to define a new function or class that implements the desired functionality and integrates it with the agent's tool management system.

### Best Practices

- Keep tools simple and focused.
- Provide clear and concise documentation.
- Handle errors gracefully.
- Test your tools thoroughly.
