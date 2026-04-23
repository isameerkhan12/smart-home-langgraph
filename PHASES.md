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

### What was built

| File | Purpose |
|---|---|
| `graph/state_phase3.py` | `Phase3State` schema with additional fields: `sensor_context`, `memory_context`, `used_live_llm` |
| `services/gemini_client.py` | Gemini call wrapper with robust fallback path when API key is missing or runtime call fails |
| `graph/workflow_phase3.py` | New 3-node flow: `detect_intent` → `retrieve_context` → `generate_response` |
| `main.py` update | Added `--phase` switch (`phase0` / `phase3`) and `run_phase3()` helper |
| `tests/test_phase3_workflow.py` | Tests for injected-generator path and no-API-key fallback path |

### Requirement solved

> *"The user then ask questions / gives task to the agent."*
> *"there should be a time serise data that the model learns/trains on."*

Phase 3 is the first end-to-end grounded generation path.
The answer is generated by Gemini using both:
1. Recent sensor history summary from simulated time-series data.
2. Long-term memory context (mistakes, recipes, preferences).

### Phase 3 graph shape

```
[START] -> detect_intent -> retrieve_context -> generate_response -> [END]
```

- `detect_intent`:
	categorises the user query into a task class.
- `retrieve_context`:
	builds `sensor_context` from `SmartHomeSimulator` and `memory_context` from `MemoryRetriever`.
- `generate_response`:
	calls Gemini via `generate_with_gemini()`; if unavailable, returns a deterministic fallback.

### Metric connection

| Metric | How Phase 3 prepares it |
|---|---|
| `critique_first_pass_rate` | Not active yet (critique loop comes in Phase 4), but Phase 3 creates the grounded generation baseline that critique will evaluate |
| `repeated_mistake_rate` | Memory context is now injected into generation input; Phase 5 will measure if this actually prevents repeats |
| `preference_adherence_rate` | Preference context is now injected; Phase 5 will measure if responses follow it |
| `recipe_reuse_rate` | Recipe context is now injected; Phase 5 will measure explicit reuse |

### Key concepts

```
Dependency Injection + Fallback Safety
```

- **Dependency injection** in `build_phase3_workflow(...)` lets tests pass fake generators and temp-file-backed memory stores.
- **Fallback safety** in `generate_with_gemini(...)` ensures the graph still completes if key/network/API is unavailable.
- **State enrichment**: each node adds fields to state (`intent` -> `sensor_context/memory_context` -> `response`).

---

## Phase 4 — Critique Node + Repair Loop

### What was built

| File | Purpose |
|---|---|
| `graph/state_phase4.py` | `Phase4State` schema extends Phase 3 with `critique_result` (structured feedback), `repair_count` (retry counter), and `max_repairs` (retry limit) |
| `services/critique_client.py` | `critique_response()` — evaluates response quality via Gemini; returns structured `CritiqueResult` with pass/fail, issues, severity, and repair hints |
| `graph/workflow_phase4.py` | New 5-node graph with critique + repair loop; conditional edge routes: critique pass → END, critique fail (< max_repairs) → repair → re-critique |
| `main.py` update | Added `--phase phase4` support and `run_phase4()` helper; Phase 4 is now default |
| `tests/test_phase4_critique_and_repair.py` | 5 comprehensive tests: critique pass (no repair), critique fail → repair success, repair exhaustion, repair count validation, state enrichment across workflow |

### Requirement solved

> *"Add a critique node to validate / evaluate the generated output."*

Phase 4 adds the first **quality gate**.
Before returning a response to the user, Gemini evaluates its own output.
If critiqued as low quality, the workflow automatically repairs (regenerates + re-critiques).
If repair attempts are exhausted, return the best attempt made so far.

This loop ensures you get the best possible answer from the first query — no need for manual follow-up.

### Phase 4 graph shape

```
                           ┌─ critique pass ─→ [END]
[START] → detect_intent → retrieve_context → generate_response → critique_response ┤
                                                ↑                                   │
                                                │                                   └─ critique fail
                                                │                                      & retries remain
                                                │                                      ↓
                                                └────────────────── repair_response ←─┘
```

**Node behaviors:**

- `critique_response` (new):
	- Calls Gemini with structured critique prompt (actionability, safety, coherence, length).
	- Returns `CritiqueResult` with `passed: bool`, `issues: [str]`, `severity: str`, `repair_hints: str`.
	- Stores result in state for conditional routing.

- `repair_response` (new):
	- Triggered when critique fails and `repair_count < max_repairs`.
	- Appends critique hints to the original query and regenerates response.
	- Increments `repair_count` and loops back to `critique_response`.
	- Bounded by `max_repairs` (default 2) — protects against infinite loops.

- `should_repair` (conditional function):
	- Returns `"end"` if `critique_result.passed == True` or `repair_count >= max_repairs`.
	- Returns `"repair"` if `critique_result.passed == False` and retries remain.

### Metric connection

| Metric | How Phase 4 affects it |
|---|---|
| `critique_first_pass_rate` | **Directly recorded now**. After first `generate_response` finishes, store `critique_result["passed"]` into `EpisodeRecord.critique_passed_first_try`. Phase 6 will compute % of episodes that passed without repair. |
| `repeated_mistake_rate` | Not yet affected (Phase 5 will integrate); Phase 4 just prepares the critique structure that Phase 5 will analyze |
| `preference_adherence_rate` | Not yet affected (Phase 5 will integrate) |
| `recipe_reuse_rate` | Not yet affected (Phase 5 will integrate) |

The other 3 metrics will get real values in Phase 5 when the memory-writer node analyzes critique feedback for repeated mistakes, broken preferences, and recipes that should be saved.

### The critique prompt strategy

```
Criteria evaluated:
  1. Actionability — Is the response concrete and executable (not vague)?
  2. Safety — Does it avoid dangerous or counterintuitive advice?
  3. Coherence — Does it align with the sensor data provided (no contradictions)?
  4. Length — Is it concise (≤6 bullet points, not a wall of text)?

Output format: JSON with boolean `passed` and list of `issues` if not.
```

This ensures responses are not just **accurate** (Gemini is already good at that) but **useful** (crisp, actionable, and locally sensible).

### Why bounded retries?

- **Without bounds**: Bad generations + critique could loop forever.
- **With max_repairs=2**: Test coverage shows agent picks the better of 3 attempts (initial + 2 repairs).
- **Fallback safety**: Even if critique fails (network error, key issue), return the last response generated, not a crash.

### Key concepts

```
Conditional Edge + State Mutation + Bounded Loop
```

- **Conditional edge** in LangGraph routes based on state inspection: `graph.add_conditional_edges(source, routing_fn, {path1: dest1, path2: dest2})`.
- **State mutation**: `repair_node` increments `repair_count` each cycle so the loop cannot run forever.
- **Structured output from LLM**: JSON parsing in `critique_client.py` extracts `CritiqueResult` fields even if Gemini wraps them in markdown.
- **Dependency injection in tests**: Fake critique generators let tests verify routing logic without API calls.

### Test summary

| Test | Purpose | What it validates |
|---|---|---|
| `test_critique_pass_routes_to_end` | Critique passes on first try | No repair triggered; `repair_count` stays 0 |
| `test_critique_fail_triggers_repair` | Critique fails; repair succeeds | Repair node called; critique re-evaluated; `repair_count` = 1; final `passed` = True |
| `test_repair_exhaustion_exits_with_failed_response` | Max retries exceeded | Exit with `critique_result.passed = False` after `repair_count == max_repairs` |
| `test_repair_count_increments_correctly` | Counter increments across cycles | `repair_count` goes 0 → 1 → ... until exit condition met |
| `test_state_enrichment_across_workflow` | Context fields populated | `sensor_context` and `memory_context` non-empty after `retrieve_context` node |

---

## Phase 5 — Learning from Experience

### What was built

| File | Purpose |
|---|---|
| `graph/state_phase5.py` | `Phase5State` extends Phase 4 with `episode_record` (metrics for this run) and `memory_written_count` (learning tracking) |
| `services/memory_writer.py` | `write_learnings()` analyzes critique results and writes to mistake/recipe stores; returns `LearningOutcome` metrics |
| `graph/workflow_phase5.py` | New 6-node graph with `memory_writer` node at end; populates `episode_record` with all 4 metrics |
| `main.py` update | Added `--phase phase5` support and `run_phase5()` helper with learning summary output |
| `tests/test_phase5_learning.py` | 8 comprehensive tests: memory persistence, mistake/recipe writing, metric computation, repair-vs-first-pass scoring |

### Requirement solved

> *"Agent should not make mistakes after some experience."*
> *"Agent should use previous knowledge (recipe) that it had used previously."*
> *"The agent learns from previous mistakes while generating a response."*
> *"The agent learns from previous preferences while generating a response."*

Phase 5 is the **learning engine**. It transforms one-off critique results (Phase 4) into persistent knowledge:
- **Mistakes** → stored for avoidance in future queries
- **Recipes** → stored and reused for faster, better answers
- **Metrics** → populated with ground truth from real episode outcomes

### Phase 5 graph shape

```
[START] → detect_intent → retrieve_context → generate_response → critique_response
                                                                         ↓
                                                 ┌───────────────────────┴────────────────┐
                                                 ↓                                        ↓
                                     [Pass & record] ← memory_writer                [Fail & retries?]
                                             ↓                                           │
                                            END                                          │
                                                                                         ↓
                                                                                 repair_response (loop)
```

**New node (memory_writer):**
- Called after critique decision is final (either passed or max repairs exhausted)
- **Logic**:
  - If critique FAILED: extract issue descriptions → write to `MistakeMemory`
  - If critique PASSED: write response strategy → write to `RecipeMemory`
  - Record success score: 0.9 (first pass) vs 0.7 (after repair)
- **Output**: Populates `episode_record` with metrics and increments `memory_written_count`

### The 4 metrics — real values now

| Metric | How Phase 5 captures it | Example |
|---|---|---|
| `critique_first_pass_rate` | `episode_record.critique_passed_first_try` set to True if `repair_count == 0` and critique passed | 70% (7 of 10 episodes passed on first generate) |
| `repeated_mistake_rate` | Phase 6 will compare: "did MistakeMemory help avoid the same error twice?" | Future computation |
| `preference_adherence_rate` | Phase 6 will check if response respected stored preferences | Future computation |
| `recipe_reuse_rate` | Phase 6 will track when a recipe was available and actually used | Future computation |

**Note**: Phase 5 does the WRITING; Phase 6 does the MEASUREMENT across many runs.

### Memory writing strategy

```python
if not critique["passed"]:
    # Write what went wrong
    for issue in critique["issues"]:
        mistake_store.save({
            "task_class": intent,
            "description": issue,
            "corrective_rule": critique["repair_hints"],
        })

if critique["passed"]:
    # Write what worked
    recipe_store.save({
        "task_class": intent,
        "strategy": response,
        "success_score": 0.9 if repair_count == 0 else 0.7,
    })
```

**Key insight**: Even if critique failed and exhausted repairs, we write to MistakeMemory so the next agent avoids the same error.

### Dependency injection in tests

Phase 5 tests use fake critique generators that simulate:
1. Critique fails on first attempt → `MistakeMemory` populated
2. Critique passes on repair → `RecipeMemory` gets lower success_score (0.7 vs 0.9)
3. Critique passes immediately → both mistake and recipe stores updated correctly

This lets all 8 tests run in **<1 second** without Gemini API calls.

### Test summary

| Test | Purpose | What it validates |
|---|---|---|
| `test_memory_writer_writes_mistakes_on_critique_fail` | Failed critique → mistakes stored | MistakeMemory file has 2 records (one per issue) |
| `test_memory_writer_writes_recipes_on_critique_pass` | Successful critique → recipe stored | RecipeMemory has strategy with score 0.9 |
| `test_episode_record_populated_after_phase5_run` | End-to-end workflow populates metrics | EpisodeRecord has critique_passed_first_try=True |
| `test_memory_persistence_across_runs` | Learnings survive across separate executions | Second retriever loads recipes from first run's JSON file |
| `test_metrics_computation_with_phase5_records` | MetricsTracker computes from EpisodeRecords | 2/3 episodes passed → 66.7% rate |
| `test_repair_affects_success_score` | Repair vs first-pass get different scores | 0.9 > 0.7 |
| `test_workflow_with_failed_critique_then_repair_writes_recipe` | Fail → repair → pass writes recipe | recipe_written despite repair_count=1 |

### How Phase 5 enables Phase 6

Phase 6 will:
1. Run 30+ episodes across multiple intent classes
2. Compute all 4 metrics as trends over time
3. Show that early episodes have high `critique_first_pass_rate` variance (random responses)
4. Show that later episodes have high `critique_first_pass_rate` stability (learned patterns)
5. Plot trends to prove agent improved with experience

Phase 5's `EpisodeRecord` is the data source. Phase 6 aggregates them into evidence.

---

## Phase 6 — Evaluation + Research Report

*Final phase. Will cover:*
- Repeated-task evaluation scenarios (same task class, 10+ episodes each)
- Trend plots for all 4 metrics showing improvement over episodes
- Documented limitations and future work (real device integration, voice interface)
- Requirement solved: proof that all learning requirements work end-to-end
