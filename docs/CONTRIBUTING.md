<div align="center">
  <img src="../assets/logo.png" alt="LangCode Logo" width="180" />
  <h1 align="center">LangCode</h1>

  <p align="center"><i><b>The only CLI you'll ever need!</b></i></p>
</div>

# Contributing to LangCode

We welcome contributions to LangCode! This guide outlines the process and best practices for contributing to the project.

## Code of Conduct

This project and everyone participating in it is governed by the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/). By participating, you are expected to uphold this code. Please report unacceptable behavior to [insert contact email or link here].

## How to Contribute

There are several ways you can contribute to LangCode:

*   **Report Bugs:** If you find a bug, please submit a detailed issue on GitHub.
*   **Suggest Enhancements:** Have an idea for a new feature or improvement? Open an issue to discuss it.
*   **Contribute Code:** Submit pull requests to fix bugs, add features, or improve existing code.
*   **Improve Documentation:** Help us make the documentation clearer, more concise, and more helpful.

## Getting Started

1.  **Fork the Repository:** Click the "Fork" button in the top right corner of the repository on GitHub.
2.  **Clone Your Fork:** Clone the repository to your local machine:

    ```bash
    git clone https://github.com/zamalali/langchain-code.git
    cd langchain-code
    ```

3.  **Create a Virtual Environment:** It's recommended to use a virtual environment to manage dependencies:

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows, use .\.venv\Scripts\activate
    ```

4.  **Install Dependencies:**

    ```bash
    pip install -e .
    ```

5.  **Create a Branch:** Create a new branch for your contribution:

    ```bash
    git checkout -b feature/your-feature-name
    ```

## Making Changes

1.  **Follow Coding Standards:** Adhere to the existing coding style and conventions.
2.  **Write Tests:** Ensure your changes are covered by unit tests. Add new tests if necessary.
3.  **Run Tests:** Run all tests to ensure everything is working correctly:

    ```bash
    pytest
    ```

4.  **Commit Your Changes:** Write clear and concise commit messages:

    ```bash
    git commit -m "feat: Add your feature description"
    ```

## Submitting a Pull Request

1.  **Push Your Branch:** Push your branch to your forked repository on GitHub:

    ```bash
    git push origin feature/your-feature-name
    ```

2.  **Create a Pull Request:** Go to your forked repository on GitHub and click the "Compare & pull request" button.
3.  **Pull Request Template:**

    Use the following template for your pull request description:

    ```markdown
    ## Description

    [Provide a brief description of the changes you've made.]

    ## Related Issue(s)

    [If applicable, link to the issue(s) this PR addresses. For example: "Fixes #123"]

    ## Checklist

    - [ ] I have tested these changes thoroughly.
    - [ ] I have added or updated unit tests.
    - [ ] I have updated the documentation (if applicable).
    - [ ] I have followed the coding standards.
    - [ ] My code is free of any warnings or errors.

    ## Additional Notes

    [Add any additional information or context that might be helpful for reviewers.]
    ```

4.  **Review Process:** Your pull request will be reviewed by the project maintainers. They may request changes or ask questions. Please respond to their feedback promptly.
5.  **Merge:** Once your pull request has been approved, it will be merged into the main branch.

## Best Practices

*   **Keep PRs Small:** Smaller pull requests are easier to review and merge.
*   **Focus on One Thing:** Each pull request should address a single issue or feature.
*   **Write Clear Commit Messages:** Use descriptive commit messages to explain your changes.
*   **Stay Up-to-Date:** Keep your branch up-to-date with the main branch by rebasing or merging.

## Documentation

When contributing code, please update the documentation accordingly. This includes:

*   **API Documentation:** Document any new functions, classes, or modules.
*   **User Guides:** Update the user guides to reflect any new features or changes.
*   **Examples:** Provide examples of how to use the new features.

Thank you for contributing to LangCode!