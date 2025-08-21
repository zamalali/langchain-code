# Deep Agents

Deep agents are autonomous agents that can plan and execute complex tasks. They use a combination of tools and subagents to achieve their goals.

## Key Concepts

*   **Planning**: Deep agents use a planner to break down complex tasks into smaller, more manageable steps.
*   **Subagents**: Deep agents can delegate tasks to subagents, which are specialized agents that can perform specific tasks.

## create_deep_agent

The `create_deep_agent` function creates a deep agent.

```python
def create_deep_agent(
    *,
    provider: str,
    project_dir: Path,
    instructions: Optional[str] = None,
    subagents: Optional[List[SubAgent]] = None,
    apply: bool = False,
    test_cmd: Optional[str] = None,
    state_schema=DeepAgentState,
    checkpointer: Optional[Checkpointer] = None,
):
    \"\"\"\
    Returns a LangGraph graph (same as deepagents) with planning + subagents.
    \"\"\"
```

## BASE_DEEP_SUFFIX

The `BASE_DEEP_SUFFIX` is a string that is appended to the base system prompt for deep agents. It contains instructions for planning and tasking, as well as information about subagents.

```python
BASE_DEEP_SUFFIX = \"\"\"\
## Planning & Tasking
- Use `write_todos` to outline steps, update statuses live, and mark completion IMMEDIATELY after finishing a step.
- For complex work or to quarantine context, call `task(description, subagent_type=...)` to launch a sub-agent.

## Subagents
- Prefer 'general-purpose' for iterative research/execution.
\"\"\"
```
