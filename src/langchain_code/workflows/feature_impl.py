FEATURE_INSTR = """You are implementing a feature end-to-end.
- Plan steps first (files to inspect, edits to make).
- Use glob/grep to locate relevant files.
- Use read_file to inspect.
- Make targeted edits via edit_by_diff (preferred) or write_file for new files.
- If a test command is available, call run_cmd with command "{TEST_CMD}" to execute it.
- Present a concise summary of changes (list of files edited/created) and next steps."""
