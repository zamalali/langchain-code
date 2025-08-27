<div align="center">
  <img src="../assets/logo.png" alt="LangCode Logo" width="180" />
  <h1><b>LangCode</b></h1>
  <p><b>The only CLI you need.</b></p>
</div>

# Workflows

The workflows module defines a set of predefined workflows that the agent can follow to accomplish specific tasks. Each workflow is a sequence of steps that the agent executes to achieve a particular goal.

## Available Workflows

The following workflows are available for the agent to use:

- **Bug Fix:** A workflow for identifying and fixing bugs in code. This workflow typically involves the following steps: identifying the bug, reproducing the bug, diagnosing the bug, fixing the bug, and verifying the fix.
- **Feature Implementation:** A workflow for implementing new features in a codebase. This workflow typically involves the following steps: understanding the requirements, designing the feature, implementing the feature, testing the feature, and integrating the feature.
- **General Purpose:** A general-purpose workflow that can be adapted to a wide range of tasks. This workflow can be customized to fit the specific needs of the task at hand.



The following workflows are available for the agent to use:

- **Bug Fix:** A workflow for identifying and fixing bugs in code. This workflow typically involves the following steps: identifying the bug, reproducing the bug, diagnosing the bug, fixing the bug, and verifying the fix.
- **Feature Implementation:** A workflow for implementing new features in a codebase. This workflow typically involves the following steps: understanding the requirements, designing the feature, implementing the feature, testing the feature, and integrating the feature.
- **General Purpose:** A general-purpose workflow that can be adapted to a wide range of tasks. This workflow can be customized to fit the specific needs of the task at hand.



- **Bug Fix:** A workflow for identifying and fixing bugs in code.
- **Feature Implementation:** A workflow for implementing new features in a codebase.
- **General Purpose:** A general-purpose workflow that can be adapted to a wide range of tasks.

## Creating New Workflows

To create a new workflow, you need to create a new Python file in the `langcode/workflows` directory and define a class that inherits from the `BaseWorkflow` class. You will also need to add the workflow to the `__init__.py` file in the same directory.

### Workflow Structure

Each workflow should have a clear goal and a well-defined set of steps. The workflow should be able to accomplish a specific task and return a result that can be used by the agent. The workflow should also be well-documented, with clear instructions on how to use it.

### Best Practices

- Keep workflows simple and focused.
- Provide clear and concise documentation.
- Handle errors gracefully.
- Test your workflows thoroughly.



To create a new workflow, you need to create a new Python file in the `langcode/workflows` directory and define a class that inherits from the `BaseWorkflow` class. You will also need to add the workflow to the `__init__.py` file in the same directory.
