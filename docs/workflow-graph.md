# Smart Home LangGraph Workflow Diagram

```mermaid
flowchart TD
    subgraph init["Initialization"]
        A([Start]) --> B[detect_intent]
        B --> C[retrieve_context]
    end

    subgraph context["Context Preparation"]
        C --> D{should_summarize?}
        D -->|"> 6 messages"| E[summarize]
        D -->|"≤ 6 messages"| P[planner]
        E --> P
    end

    subgraph generation["Response Generation"]
        P -->|"sets use_memory_only flag"| F[generate_response]
        F --> G{tools_condition?}
        G -->|"tool_calls present"| H[tools]
        G -->|"no tool_calls"| I[critique_response]
        H --> J[process_tool_result]
        J --> I
    end

    subgraph repair["Critique & Repair Loop"]
        I --> K{route_after_critique}
        K -->|"passed"| L[memory_writer]
        K -->|"failed, can repair"| M[repair_response]
        K -->|"tool failed, can repair"| N[retrieve_error_memory]
        K -->|"tool failed, max repairs"| O[error_memory_writer]
        N --> M
        M --> G2{tools_condition?}
        G2 -->|"tool_calls"| H
        G2 -->|"no tool_calls"| I
    end

    subgraph finish["Finalization"]
        L --> Q[record_turn]
        O --> Q
        Q --> R((END))
    end
```

## Planner Node

The **planner** node decides whether memory is sufficient to answer the user's question:

- **`use_memory_only = true`**: Memory contains the answer → `generate_response` and `repair_response` run **without tools bound** (LLM physically cannot call `python_repl`)
- **`use_memory_only = false`**: Computation needed → `generate_response` and `repair_response` run **with tools bound** (LLM can call `python_repl`)

This prevents redundant tool calls when the answer already exists in long-term memory.

## Notes

- Entry point: `detect_intent`.
- Planner gates tool access based on memory sufficiency.
- Tool execution is optional and controlled by `tools_condition`.
- Critique-repair loop exits to `memory_writer` on pass or max repair exhaustion.
- If tool failure persists at max repairs, flow writes to `error_memory_writer` before ending.
