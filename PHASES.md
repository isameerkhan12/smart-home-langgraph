# Smart Home LangGraph — Phase-by-Phase Implementation Guide

This document grows with the project.
Every phase explains: what was built, which requirement it solves, and how the metrics connect to it.

---

## How to read this document

| Column | Meaning |
|---|---|
| **What was built** | Files and classes created in this phase |
| **Requirement solved** | Which of your original requirements this phase addresses |
| **Metric connection** | Which of the 4 metrics this phase affects and why |
| **Key concept** | The LangGraph / Python idea you learn in this phase |

---

## Phase 0 — Environment + Hello Graph

### What was built

| File | Purpose |
|---|---|
| `.venv/` | Isolated Python virtual environment so project packages do not conflict with your system Python |
| `requirements.txt` | Pinned list of every package the project needs (LangGraph, Gemini, pandas, pytest, …) |
| `.env.example` | Template showing which environment variables you need; copy to `.env` and fill in your key |
| `config/settings.py` | Reads `GEMINI_API_KEY` from `.env` safely so the key never appears in source code |
| `graph/state.py` | Defines `AgentState` — the shared notebook passed between every graph node |
| `graph/workflow.py` | Two-node LangGraph graph: `detect_intent` → `build_response` → END |
| `main.py` | Command-line entry point: `python -m smart_home_langgraph.main --query "..."` |
| `tests/test_phase0_workflow.py` | Verifies the graph produces the right intent and a non-empty plan |

### Requirement solved

> *"The user asks questions / gives tasks to the agent."*

Phase 0 is the skeleton. Without it nothing else can be built.
It proves you can send a natural-language question into a LangGraph and get a structured response back.

### Metric connection

None of the 4 metrics are recorded yet.
Phase 0 is purely infrastructure. It exists so every later phase has a running, testable foundation to build on.

### Key LangGraph concept

```
StateGraph  →  add_node  →  add_edge  →  compile  →  invoke
```

- `StateGraph(AgentState)` — creates the graph container
- `add_node("name", function)` — registers a step
- `add_edge("from", "to")` — sequential connection (no branching yet)
- `compile()` — validates and locks the graph into a runnable app
- `invoke(initial_state)` — runs the graph from entry point to END

---

## Phase 1 — Scope Definition + Metrics Tracker

### What was built

| File | Purpose |
|---|---|
| `evaluation/scope.yaml` | Declares the 4 in-scope task classes, out-of-scope items for v1, and the 4 success metrics with descriptions |
| `evaluation/metrics.py` | `EpisodeRecord` (one run's data) + `MetricsTracker` (computes % trends) + `load_scope()` helper |
| `tests/test_phase1_metrics.py` | Verifies scope loads correctly and metrics compute the right percentages |

### Requirement solved

> *"The agent learns from previous mistakes while generating a response."*
> *"The agent learns from previous preferences while generating a response."*
> *"Agent should not make mistakes after some experience."*
> *"Agent should use previous knowledge (recipe) that it had used previously."*

These 4 requirements all have the same hidden assumption: **you must be able to measure whether improvement is happening.** Without numbers you cannot tell if the agent is actually getting better or just appearing to.

Phase 1 defines the measurement system that every later phase will write into.

### The 4 metrics — what each one proves

| Metric | Target trend | What it proves |
|---|---|---|
| `critique_first_pass_rate` | ↑ increases | Agent generates better responses over time; needs fewer repair loops |
| `repeated_mistake_rate` | ↓ decreases | Mistake memory is working; agent stops repeating known errors |
| `preference_adherence_rate` | ↑ increases | Preference memory is working; agent respects what the user told it |
| `recipe_reuse_rate` | ↑ increases | Recipe memory is working; agent reuses proven solutions instead of reinventing |

### Why track in code and not manually?

In Phase 6 you run 30+ episodes and plot these four numbers as trend lines.
That plot is the **evidence** that the agent improved with experience — the kind of result you can show a supervisor or include in a research report.
`MetricsTracker` captures every episode outcome automatically so the data is always ready.

### Metric connection

Phase 1 **defines** all 4 metrics but does not record real values yet.
The fields in `EpisodeRecord` are placeholders (`True`/`False`) that the critique node (Phase 4) and memory writer (Phase 5) will fill in automatically.

### Key concept

```
EpisodeRecord  →  MetricsTracker.add_episode()  →  MetricsTracker.summary()
```

- `EpisodeRecord` — a dataclass holding one run's outcome (did critique pass? was a mistake repeated? …)
- `MetricsTracker` — collects records and computes percentages safely (returns `None` when no data yet to avoid divide-by-zero)
- `load_scope()` — reads `scope.yaml` with `yaml.safe_load` so tests can validate against the defined task classes

---

## Phase 2 — Simulated Data + Memory Stores

### What was built

| File | Purpose |
|---|---|
| `data/simulator.py` | `SmartHomeSimulator` — generates 15-minute interval sensor + appliance data for N days |
| `memory/store.py` | Three store classes: `PreferenceMemory`, `MistakeMemory`, `RecipeMemory` — each backed by a local JSON file |
| `memory/retriever.py` | `MemoryRetriever` — queries all three stores and assembles one context string for the LLM prompt |
| `tests/test_phase2_data_and_memory.py` | 12 tests covering row counts, value ranges, save/reload round-trips, best-recipe selection, and retriever context assembly |

### Requirement solved

> *"There should be time-series data that the model learns/trains on."*
> *"The agent learns from previous mistakes while generating a response."*
> *"The agent learns from previous preferences while generating a response."*
> *"Agent should not make mistakes after some experience."*
> *"Agent should use previous knowledge (recipe) that it had used previously."*

The simulator provides the sensor data.
The three memory stores are the persistent "brain" that the agent reads from and writes to.
Without them, every conversation starts from zero — the agent can never improve.

### The three memory stores — what each one holds

| Store | What it persists | Requirement it fills |
|---|---|---|
| `PreferenceMemory` | `key → value` pairs, e.g. `quiet_hours: 22:00-07:00` | Agent respects user preferences |
| `MistakeMemory` | `error_description + corrective_rule` per task class | Agent avoids repeating known mistakes |
| `RecipeMemory` | `strategy_steps + critique_score` per task class | Agent reuses proven solutions |

### Retrieval priority order

```
1. MistakeMemory   → AVOID known errors (safety first)
2. RecipeMemory    → REUSE what worked (efficiency)
3. PreferenceMemory → PERSONALISE the answer
```

This order is enforced in `MemoryRetriever.retrieve()`.
The result is a single text block injected into the LLM prompt in Phase 3.

### Metric connection

| Metric | How Phase 2 enables it |
|---|---|
| `repeated_mistake_rate` | `MistakeMemory` stores the errors; Phase 5 will check if the agent looked them up |
| `preference_adherence_rate` | `PreferenceMemory` stores the rules; Phase 5 will check if the response followed them |
| `recipe_reuse_rate` | `RecipeMemory` stores the recipes; Phase 5 will flag when `has_recipe()` was True and a recipe was used |

Phase 2 builds the storage layer. The metrics will get real values once the critique + memory-writer nodes (Phase 4 and 5) start writing outcomes back.

### Key concepts

```
dataclass + asdict()  →  json.dump  →  json.load  →  dataclass(**record)
```

- `dataclass` — auto-generates `__init__` and `__repr__` from field definitions
- `asdict()` — converts a dataclass instance to a plain dict so `json.dump` can write it
- `json.safe_load` / `json.load` — reads back the dict; `dataclass(**rec)` reconstructs the object
- `tmp_path` pytest fixture — gives each test its own isolated temp directory so JSON files never interfere between tests

---

## Phase 3 — Core LangGraph Pipeline with Gemini

*Coming after Phase 2. Will cover:*
- Real LLM calls via Gemini API replacing the keyword classifier
- Context retriever node pulling relevant sensor windows
- Requirement solved: *"the user asks questions / gives tasks to the agent"* — with grounded, real answers

---

## Phase 4 — Critique Node + Repair Loop

*Coming after Phase 3. Will cover:*
- Critique node with structured output (pass/fail, issues, severity, repair hints)
- Conditional edge: critique fail → repair → re-critique (max N retries)
- First real values written into `EpisodeRecord.critique_passed_first_try`
- Requirement solved: *"add a critique node to validate / evaluate the generated output"*

---

## Phase 5 — Learning from Experience

*Coming after Phase 4. Will cover:*
- Writing accepted critique fixes into mistake memory
- Writing successful strategies into recipe memory
- Retrieval priority at generation time: mistakes → recipes → preferences → domain context
- All 4 `EpisodeRecord` fields populated with real values
- Requirement solved: *"agent should not make mistakes after some experience"* + *"use previous knowledge (recipe)"*

---

## Phase 6 — Evaluation + Research Report

*Final phase. Will cover:*
- Repeated-task evaluation scenarios (same task class, 10+ episodes each)
- Trend plots for all 4 metrics showing improvement over episodes
- Documented limitations and future work (real device integration, voice interface)
- Requirement solved: proof that all learning requirements work end-to-end
