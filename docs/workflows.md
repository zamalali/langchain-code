# Workflows

The workflows module defines a set of predefined workflows that the agent can follow to accomplish specific tasks. Each workflow is a sequence of steps that the agent executes to achieve a particular goal.

## Available Workflows

- **Bug Fix:** A workflow for identifying and fixing bugs in code.
- **Feature Implementation:** A workflow for implementing new features in a codebase.
- **General Purpose:** A general-purpose workflow that can be adapted to a wide range of tasks.

## Creating New Workflows

To create a new workflow, you need to create a new Python file in the `langcode/workflows` directory and define a class that inherits from the `BaseWorkflow` class. You will also need to add the workflow to the `__init__.py` file in the same directory.
