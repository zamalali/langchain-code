# Tools

The tools module provides a collection of tools that the agent can use to perform various tasks. These tools can include anything from simple utilities like a calculator or a search engine to more complex integrations with external APIs.

## Available Tools

- **Calculator:** A simple calculator for performing mathematical calculations.
- **Search:** A tool for searching the web and retrieving information.
- **Code Interpreter:** A tool for executing Python code and displaying the results.

## Adding New Tools

To add a new tool, you need to create a new Python file in the `langcode/tools` directory and define a class that inherits from the `BaseTool` class. You will also need to add the tool to the `__init__.py` file in the same directory.
