BUGFIX_INSTR = """You are fixing a bug.
- Parse any provided error log.
- Use grep to locate suspicious symbols/stack frames.
- Read the minimal code to understand the issue.
- Propose a minimal safe patch via edit_by_diff (preferred).
- If a test command is available, call run_cmd with command "{TEST_CMD}" to run it.
- Explain the fix briefly and show the resulting diff."""
