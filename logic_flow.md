```mermaid
graph LR
    User --> CLI: Input
    CLI --> Agent: Invoke
    Agent --> LLM: Prompt
    LLM --> Agent: Response
    Agent --> Tools: Action
    Tools --> Agent: Result
    Agent --> CLI: Output
    CLI --> User: Display
```