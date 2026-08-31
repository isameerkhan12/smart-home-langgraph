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
        D -->|"≤ 6 messages"| P[memory_evaluator]
        E --> P
    end

    subgraph generation["Response Generation"]
        P -->|"sets tools_usage and memory_usage_mode"| F[generate_response]
        F --> G{tools_condition?}
        G -->|"tool_calls present"| H[tools]
        G -->|"no tool_calls"| I[critique_response]
        H --> J[process_tool_result]
        J --> K{route_after_tool_result}
        K -->|"ordinary tool result"| F
        K -->|"submit_final_answer"| I
    end

    subgraph repair["Critique & Repair Loop"]
        I --> S{route_after_critique}
        S -->|"passed"| L[memory_writer]
        S -->|"failed, can repair"| M[repair_response]
        S -->|"tool failed, can repair"| N[retrieve_error_memory]
        S -->|"tool failed, max repairs"| O[error_memory_writer]
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

## Memory Evaluator Node

The **memory_evaluator** node decides whether memory is sufficient to answer the user's question:

- **`memory_usage_mode = memory_only`**: Memory contains the answer → `generate_response` and `repair_response` run **without tools bound** (LLM physically cannot call analysis tools)
- **`memory_usage_mode = partial`**: Memory contains some facts → tools are available for missing facts, and the response combines memory with tool results
- **`memory_usage_mode = none`**: Computation is needed → tools are available for the full analysis

This prevents redundant tool calls when the answer already exists in long-term memory.

## Notes

- Entry point: `detect_intent`.
- Memory evaluator gates tool access based on memory sufficiency.
- Direct answers and explicit `submit_final_answer` results go to `critique_response`.
- Ordinary tool results loop back to `generate_response` so multi-step analysis can continue before critique.
- The critique-repair loop exits to `memory_writer` on pass or max repair exhaustion.
- If tool failure persists at max repairs, flow writes to `error_memory_writer` before ending.
