# Feature Implementation Workflow

This workflow guides the agent in implementing a new feature from a user's request.

## `FEATURE_INSTR`

This is the instruction seed provided to the ReAct agent to specialize it for feature implementation tasks. It outlines a clear, step-by-step process:

1.  **Plan:** The agent should first think about the necessary steps, including which files might need to be inspected and what changes are required.
2.  **Locate:** Use file system tools like `glob` and `grep` to find the relevant code.
3.  **Inspect:** Use `read_file` to understand the existing code before making changes.
4.  **Edit:** Make targeted changes using `edit_by_diff`. For new files, `write_file` is appropriate. `edit_by_diff` is preferred for its safety and precision.
5.  **Verify:** If a test command has been provided (via the `--test-cmd` option), the agent should run it using `run_cmd` with the `{TEST_CMD}` placeholder.
6.  **Summarize:** Conclude by presenting a clear summary of the changes made and suggesting any next steps.
