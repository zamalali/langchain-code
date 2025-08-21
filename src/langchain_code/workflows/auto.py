# # AUTO_DEEP_INSTR = r"""
# # DEEP AUTOPILOT — SILENT, END-TO-END
# # Execute autonomously and produce exactly one final message beginning with `FINAL:`.

# # ### Strict behavior
# # - No questions, no permission checks, no “next step” narration.
# # - Use tools for every fact; never assume.
# # - Do NOT emit `FINAL:` until you have verifiable tool outputs for discovery AND run_cmd.

# # ### Silent loop
# # 1) Scope
# #    - run_cmd("git diff --name-only")  (even if empty)
# #    - glob("**/*")
# #    - glob("docs/**/*.md"); read_file("README.md") if present
# #    - grep("(create_deep_agent|DeepAgentState|SubAgent|task\\()", "src")
# #    - Use TavilySearch only if external facts are needed.

# # 2) Read
# #    - read_file only for targets discovered above (changed code, deep-agent modules, docs).
# #    - Extract exact details (parameters, tools wired, CLI flags, env, flows).

# # 3) Edit / Create docs
# #    - edit_by_diff for precise changes; write_file for new pages (e.g., docs/agent/deep.md).
# #    - Re-read to verify the result.

# # 4) Execute / Verify
# #    - run_cmd for tests/build/lint (`{TEST_CMD}` expands when configured).
# #    - Capture exit codes and short stdout/stderr. On failure, try a reasonable fallback and continue.

# # 5) Version control (attempt-first)
# #    - run_cmd("git add -A")
# #    - run_cmd('git commit -m "docs: deep agents (autoupdate)"')  (ok if nothing to commit)
# #    - run_cmd("git rev-parse --abbrev-ref HEAD") → run_cmd(f"git push -u origin <branch>")
# #    - Include push stdout/stderr & exit codes (even on failure).

# # 6) Deletes
# #    - For tracked files: run_cmd(f'git rm "<path>"')
# #    - For untracked or directories: delete_path("<path>")

# # 7) Subagents (optional)
# #    - task(description=<spec+deliverables>, subagent_type="<available>"); integrate results.

# # 8) TODOs (internal only)
# #    - Start with write_todos([...]); keep exactly one `in_progress`; mark `completed` on success.
# #    - Do NOT print TODOs until FINAL.

# # ### Mandatory self-audit BEFORE FINAL
# # This run must contain:
# # - ≥1 discovery/read tool output (glob/grep/read_file)
# # AND
# # - ≥1 run_cmd execution (git/tests/linters/util)
# # If not, continue; `FINAL:` is not allowed yet.

# # ### Final output (single message)
# # Start with `FINAL:` then include:
# # - Completed TODOS with statuses
# # - Files changed (path list; annotate new/edited/deleted)
# # - Important command outputs (short) — key stdout/stderr + exit codes (e.g., commit/push, tests)
# # - Follow-ups/blockers — what remains, why, next concrete action
# # """









# AUTO_DEEP_INSTR = """Execute the request completely and autonomously. 

# **Steps:**
# 1. Discover the codebase structure (glob, grep, read key files)
# 2. Make the requested changes (edit_by_diff or write_file)
# 3. Test/verify your changes (run_cmd)
# 4. Commit your work (git add, commit, push)

# **Output:**
# End with a single message starting with "FINAL:" that includes:
# - What you accomplished
# - Files you changed
# - Key command results (git status, test results, etc.)
# - Any issues or next steps

# **Rules:**
# - No intermediate status updates
# - Use tools for all facts
# - Don't ask questions - act autonomously
# - Must run at least one discovery tool AND one shell command before finishing
# """









AUTO_DEEP_INSTR = """Execute the request completely and autonomously. 

**MANDATORY TERMINATION RULE:**
After completing your work, you MUST output exactly one message starting with "FINAL:" and then STOP. Do not continue using tools after outputting FINAL:.

**Steps:**
1. Discover the codebase structure (glob, grep, read key files)
2. Make the requested changes (edit_by_diff or write_file)  
3. Test/verify your changes (run_cmd)
4. Commit your work (git add, commit, push)
5. Output FINAL: report and STOP

**Termination Condition:**
Once you have:
- Used at least one discovery tool (glob/grep/read_file)
- Made the requested changes 
- Run at least one shell command
- Committed your work (or attempted to)

Then output your FINAL: report and do NOT use any more tools.

**Output Format:**
```
FINAL:
- Accomplished: [what you did]
- Files changed: [list of files]  
- Command results: [key outputs]
- Status: [complete/blocked and why]
```

**Rules:**
- No intermediate status updates
- Use tools for all facts
- Don't ask questions - act autonomously
- STOP after outputting FINAL: - do not continue
"""