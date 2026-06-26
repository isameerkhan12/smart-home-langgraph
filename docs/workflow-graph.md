# Smart Home LangGraph Workflow Diagram

```mermaid
flowchart TD
    A([Start]) --> B[detect_intent]
    B --> C[retrieve_context]

    C --> D{should_summarize?}
    D -->|summarize| E[summarize]
    D -->|continue| F[generate_response]
    E --> F

    F --> G{tools_condition?}
    G -->|tools| H[tools]
    G -->|END / no tool calls| I[critique_response]

    H --> J[process_tool_result]
    J --> I

    I --> K{route_after_critique}
    K -->|memory| L[memory_writer]
    K -->|repair| M[repair_response]
    K -->|retrieve_error_memory| N[retrieve_error_memory]
    K -->|error_memory_writer| O[error_memory_writer]

    N --> M
    M --> G

    L --> Q[record_turn]
    O --> Q
    Q --> R((END))
```

## Notes

- Entry point: `detect_intent`.
- Tool execution is optional and controlled by `tools_condition`.
- Critique-repair loop exits to `memory_writer` on pass or max repair exhaustion.
- If tool failure persists at max repairs, flow writes to `error_memory_writer` before ending.
