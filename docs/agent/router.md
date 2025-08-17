# Router

This module is responsible for deciding which agent to use for a given task: the simpler `react` agent or the more complex `deep` agent.

## `Router`

The `Router` class uses a combination of a language model classifier and heuristics to make its decision.

### `choose`

This is the main method of the `Router`. It takes a user's request and some metadata, and returns either `"react"` or `"deep"`.

- **Classifier:** If a classifier model is provided, the router will first try to use it to classify the request. The classifier is a language model that has been prompted to choose between the `react` and `deep` agents based on the user's request and the provided metadata.
- **Heuristic:** If the classifier is not available or fails, the router falls back to a heuristic. The heuristic is a simple scoring system that adds points for things like the length of the request, the presence of keywords like "research" or "design", and the size of any provided log files.

### `escalate_if_needed`

This method allows the router to escalate from the `react` agent to the `deep` agent if the conversation becomes more complex. It uses the same classifier and heuristic as the `choose` method.

## `RouteMeta`

This dataclass holds metadata about the user's request that is used by the router to make its decision.

- `mode`: The current mode of the agent (e.g., `fix`, `feat`).
- `test_cmd`: The test command, if any.
- `apply`: Whether or not to apply changes automatically.
- `log_path`: The path to a log file, if any.
- `image_paths_count`: The number of images provided with the request.
