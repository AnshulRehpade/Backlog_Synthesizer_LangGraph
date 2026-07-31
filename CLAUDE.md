# CLAUDE.md — Backlog Synthesizer LangGraph Migration

This file is read at the start of every session to provide immediate architectural context. Do not re-discover from source — trust this document.

---

## 1. Project Summary

This is a **LangGraph migration** of the [original Backlog Synthesizer](https://github.com/AnshulRehpade/backlog-synthesizer).

The original project is a multi-agent Python system that reads meeting transcripts, architecture documents, and existing backlog tickets, then produces structured user stories with acceptance criteria grouped into epics. This migration replaces the hand-rolled orchestrator with LangGraph's StateGraph, gaining built-in checkpointing, conditional routing, and time-travel debugging.

---

## 2. Key Migration Decisions

### Decision 1: LangGraph Checkpointer replaces Short-Term Memory

The original `memory/short_term.py` stored intermediate results between agent steps, keyed by session_id with a primary-store → dict fallback. LangGraph's Checkpointer does exactly this — persists full graph state after every node, keyed by thread_id. Using both would be redundant.

- `short_term.py` is **removed** from this migration
- `MemoryEngine` no longer accepts or delegates to `ShortTermMemory`
- For production: swap `MemorySaver` → `SqliteSaver` or `RedisSaver` with zero code changes

### Decision 2: Nested state over flattened state

PipelineState uses nested Pydantic models (`ExtractionResult`, `GapReport`, `StoryOutput`) rather than flattening into primitive fields. This:
- Preserves original project's convention of Pydantic models at agent boundaries
- Validates structure at each node boundary
- Keeps state readable and self-documenting

### Decision 3: Graph-based routing instead of internal retries

The LLM fallback node records its decision in `react_notes` (with an `action` field) and lets graph conditional edges route to the correct next node. This means:
- Retries are visible in the graph execution trace
- The checkpointer captures state before and after each retry
- No circular function calls inside nodes — the graph handles cycles cleanly

---

## 3. Three-Tier Decision Pattern

The original project's ReAct Orchestrator made 5 LLM-driven decisions at pipeline decision points. In LangGraph, this maps to a three-tier pattern:

| Level | Mechanism | Cost | When Used |
|-------|-----------|------|-----------|
| Level 1 | Conditional edges | Zero (deterministic) | Normal-path routing (e.g., parser → gap → story) |
| Level 2 | LLM fallback | LLM call | Unexpected situations (e.g., 0 items extracted, all duplicates) |
| Level 3 | Human in the loop | Interrupt | Critical decisions (e.g., ambiguous duplicates, conflicting data) |

### Level 2: LLM Fallback Node (`nodes/llm_fallback_node.py`)

When conditional edges cannot determine what to do (e.g., 0 epics generated, >50% stories
need refinement), the pipeline routes to the LLM Fallback node. This node:

1. Builds an observation summary from current state
2. Asks the LLM to choose from predefined actions
3. Records the chosen action in `react_notes[-1]["action"]`
4. Returns state — the graph's `route_after_llm_fallback` reads the action and routes accordingly

**retry_count field** (in `PipelineState`): Prevents infinite retry loops. The LLM Fallback
node increments `retry_count` before allowing a retry action. Routing functions check
`retry_count >= 1` before sending to fallback again — if a retry has already been attempted
and quality issues persist, routing sends to `error_node` instead. Maximum retries: **1**.

### Level 3: Human Review Node (`nodes/human_review_node.py`)

Uses LangGraph's `interrupt()` to pause the graph and wait for human input. The checkpointer
persists state at the interrupt point. The graph resumes when the caller provides
`Command(resume=<decision>)`. Human can:
- `"approve"` → continue to next pipeline node
- `"reject"` → terminate pipeline with permanent failure
- `"modify_threshold"` → re-run gap detection with adjusted parameters

### route_after_parser Logic

```
parser → routing:
  - permanent_failure or parser_output is None → error_node
  - 0 items extracted → llm_fallback_node (retry_count gate applies)
  - < 3 items extracted → llm_fallback_node (too few — vague transcript)
  - all items confidence < 0.3 → llm_fallback_node (parser uncertain)
  - otherwise → gap_detection (success)
```

Parser-specific LLM fallback actions:
- `retry_parser` — route back to parser node
- `request_better_transcript` — halt, ask user for clearer input
- `proceed_with_low_confidence` — route to gap_detection with warning

### route_after_gap_detection Logic

```
gap_detection → routing:
  - permanent_failure or gap_output is None → error_node
  - new_count == 0 and conflict_count == 0 (all duplicates) → llm_fallback_node
  - duplicate_count / total_items > 0.8 (>80% duplicates) → llm_fallback_node
  - otherwise → story_writer (success)
```

Gap-detection-specific LLM fallback actions:
- `lower_threshold` — route back to gap_detection with lower duplicate threshold
- `return_nothing_new` — halt gracefully, all items already in backlog
- `proceed_with_conflicts_only` — route to story_writer with conflict items only
- `request_review` — route to human_review node

### route_after_story_writer Logic

```
story_writer → routing:
  - permanent_failure or story_output is None → error_node
  - 0 epics generated → llm_fallback_node
  - >50% stories need refinement → llm_fallback_node
  - otherwise → END (success)
```

Story-writer-specific LLM fallback actions:
- `retry_story_writer` — route back to story_writer node
- `return_partial` — pipeline ends with partial results
- `halt` — pipeline ends with permanent failure

### route_after_llm_fallback Logic

```
llm_fallback → routing (reads react_notes[-1]["action"]):
  - retry_parser → parser
  - lower_threshold → gap_detection
  - retry_story_writer → story_writer
  - proceed_with_low_confidence → gap_detection
  - proceed_with_conflicts_only → story_writer
  - request_review → human_review
  - return_partial, return_nothing_new, halt, or anything else → END
```

### route_after_human_review Logic

```
human_review → routing (reads react_notes[-1]["action"]):
  - permanent_failure → END (human rejected)
  - approve_gap_detection or modify_threshold → gap_detection
  - approve_story_writer → story_writer
  - otherwise → END
```

**needs_human_review field** (in `PipelineState`): Set by `request_review` action.
Cleared by human_review_node after the human provides input. Defaults to `False`.

---

## 4. Full Graph Topology

```
START
  ↓
parser → [route_after_parser]
  ├── gap_detection (success)
  ├── llm_fallback_node (quality issues)
  └── error_node (permanent failure)

gap_detection → [route_after_gap_detection]
  ├── story_writer (success)
  ├── llm_fallback_node (all duplicates / >80% duplicates)
  └── error_node (permanent failure)

story_writer → [route_after_story_writer]
  ├── END (success)
  ├── llm_fallback_node (0 epics / >50% need refinement)
  └── error_node (permanent failure)

llm_fallback_node → [route_after_llm_fallback]
  ├── parser (retry_parser)
  ├── gap_detection (lower_threshold, proceed_with_low_confidence)
  ├── story_writer (retry_story_writer, proceed_with_conflicts_only)
  ├── human_review (request_review)
  └── END (return_partial, halt, return_nothing_new, etc.)

human_review → [route_after_human_review]
  ├── gap_detection (approve, modify_threshold)
  ├── story_writer (approve)
  └── END (reject)

error_node → END
```

---

## 5. File Structure

```
backlog_synthesizer_langgraph/
├── graph/                        # LangGraph-specific infrastructure
│   ├── state.py                  # PipelineState TypedDict definition
│   ├── builder.py                # StateGraph construction ✅ (full topology with all edges)
│   ├── routing.py                # All 5 routing functions ✅
│   └── checkpointer.py          # MemorySaver setup (enables interrupt/resume)
├── nodes/                        # LangGraph node functions
│   ├── parser_node.py            # Wraps ParserAgent ✅
│   ├── gap_detection_node.py     # Wraps GapDetectionAgent ✅
│   ├── story_writer_node.py      # Wraps StoryWriterAgent ✅
│   ├── llm_fallback_node.py      # Level 2 LLM fallback decisions ✅
│   ├── human_review_node.py      # Level 3 human-in-the-loop ✅
│   └── error_node.py             # Handles failures ✅
├── src/backlog_synthesizer_langgraph/
│   ├── agents/                   # Agent implementations
│   │   ├── parser.py            # ✅ Extracts items from documents
│   │   ├── gap_detection.py     # ✅ Classifies items against existing backlog
│   │   └── story_writer.py      # ✅ Generates user stories from gap report
│   ├── memory/                   # Long-Term Memory + Audit Log (short_term.py removed)
│   ├── tools/                    # Protocol interfaces + implementations (unchanged)
│   ├── models/                   # Pydantic data contracts (unchanged)
│   ├── observability/            # OpenTelemetry layer (unchanged)
│   ├── evaluation/               # Evaluation framework (unchanged)
│   └── config.py                 # PipelineConfig (unchanged)
├── evaluation/                   # Golden dataset + regression detection (top-level)
├── tests/
│   ├── test_story_writer_node.py      # 12 tests ✅
│   ├── test_llm_fallback_node.py      # 17 tests ✅
│   ├── test_parser_node.py            # 17 tests ✅
│   ├── test_gap_detection_node.py     # 20 tests ✅
│   ├── test_human_review_node.py      # 20 tests ✅
│   ├── test_full_pipeline.py          # 24 tests ✅ (e2e routing + graph structure)
│   └── test_app.py                    # 9 tests ✅ (UI pipeline runner smoke tests)
├── data/
│   ├── sample_transcript.txt          # Demo transcript for main.py
│   └── sample_backlog.json            # Demo backlog tickets (10 items)
├── app.py                        # Streamlit web UI ✅
├── main.py                       # CLI entry point ✅
├── pyproject.toml
├── .env.example
├── .gitignore
├── README.md
└── CLAUDE.md
```

---

## 6. What Changed vs What Stayed

### Changed
- **Orchestrator** → replaced by LangGraph StateGraph (graph/builder.py)
- **Short-Term Memory** → replaced by LangGraph Checkpointer (graph/checkpointer.py)
- **Retry/routing logic** → graph conditional edges + LLM fallback node + human review
- **MemoryEngine** → simplified facade (no ShortTermMemory delegation)
- **Package name** → `backlog_synthesizer_langgraph` (new namespace)

### Stayed the Same (copied unchanged)
- All **tool implementations** (Anthropic, OpenAI, embeddings, vector search, PDF parsing)
- All **Protocol interfaces** (`tools/interfaces.py`)
- All **Pydantic models** (inputs, extraction, gap_detection, output, memory, evaluation)
- **Long-Term Memory** (ChromaDB vector store)
- **Audit Log** (chronological agent invocation records)
- **Observability** (OpenTelemetry tracing + metrics)
- **Configuration** (`config.py` with env var resolution)
- **Error taxonomy** (ToolError → Transient | Permanent)

---

## 7. What Is NOT Yet Migrated

- [x] Parser node + agent ✅
- [x] Gap Detection node + agent ✅
- [x] Story Writer node + agent ✅
- [x] Error node ✅
- [x] LLM Fallback node (Level 2) ✅
- [x] Human Review node (Level 3) ✅
- [x] Full graph wiring with all edges ✅
- [x] route_after_parser, route_after_gap_detection, route_after_story_writer ✅
- [x] route_after_llm_fallback, route_after_human_review ✅
- [x] Checkpointer integration (enables interrupt/resume) ✅
- [x] CLI entry point (main.py) ✅
- [ ] Evaluation CLI integration with LangGraph pipeline
- [ ] Full regression test suite with golden datasets
- [ ] Streaming output support
- [ ] Production checkpointer (SqliteSaver / PostgresSaver)

---

## 8. Running the Project

```bash
# Create virtual environment and install
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Configure (add your Anthropic API key)
cp .env.example .env
# Edit .env → set ANTHROPIC_API_KEY=sk-ant-...

# Run the pipeline (CLI)
python main.py data/sample_transcript.txt --backlog data/sample_backlog.json

# Run the Streamlit UI
streamlit run app.py

# Run tests with coverage
python -m pytest tests/
```

### Streamlit UI (`app.py`)

Launch: `streamlit run app.py` → opens at http://localhost:8501

**What each tab shows:**
- **Stories** — Epics with expandable stories showing user story text, acceptance criteria, tags, and refinement warnings
- **Pipeline Stats** — Metrics (stories/epics/duplicates/conflicts/time) + per-agent breakdown table
- **Raw JSON** — Complete StoryOutput as formatted JSON with download button
- **Audit Log** — ReAct reasoning notes and error entries from each agent

**How human review works in the UI:**
- When the LLM fallback selects `request_review`, the pipeline pauses via `interrupt()`
- The UI shows a blue "Human review required" banner with Approve/Reject buttons
- On approve: pipeline resumes to gap_detection or story_writer
- On reject: pipeline terminates with permanent_failure

### Current Status: ✅ Full pipeline with Streamlit UI
- All packages install cleanly
- Graph compiles with 6 nodes: parser, gap_detection, story_writer, error_node, llm_fallback_node, human_review
- LLM fallback routes back to pipeline nodes via graph edges (not internal calls)
- Human review uses `interrupt()` for pause/resume with checkpointer persistence
- Checkpointer enables session state persistence and replay
- CLI: `python main.py data/sample_transcript.txt [--backlog data/sample_backlog.json]`
- UI: `streamlit run app.py` (sample transcript pre-filled, run immediately)
- **119 tests passing, 69% coverage** (threshold: 65%)
- Pipeline runs end-to-end with sample data (requires valid `ANTHROPIC_API_KEY` for LLM calls)
- Without API key: gracefully degrades through parser → llm_fallback → return_partial → END

### Known Differences from Original Project

1. **Retries are graph cycles** — original used internal function calls; this uses graph edges so retries show up in the execution trace and checkpointer captures each step.
2. **Human review pauses the graph** — original had no human-in-the-loop; this uses LangGraph's `interrupt()` which persists state and waits for external input.
3. **Decision recording** — LLM fallback records `action` in `react_notes` for downstream routing instead of immediately executing the action.
4. **Streamlit UI** — original was CLI-only; this adds a web interface with live progress and tabbed results.

---

## 9. Conventions (same as original)

1. **All external I/O behind Protocol interfaces** — never import concrete tools inside agents
2. **Error classification** — `ToolError` → `TransientToolError` | `PermanentToolError`
3. **Async throughout** — all agent methods are `async`; use `asyncio.to_thread()` for sync tools
4. **Pydantic models for all data contracts** — no raw dicts crossing agent boundaries
5. **Environment variables for all configuration** — loaded via `PipelineConfig.from_env()`
6. **Tags flow through the pipeline** — Parser → Gap Detection → Story Writer
7. **Tests mock all external tools** — no real API calls
8. **Nodes never raise** — every node catches all exceptions and returns state gracefully

---

## 10. Dependencies Added (vs original)

- `langgraph>=0.2.0` — StateGraph, MemorySaver, conditional edges, interrupt()
- `langchain-core>=0.2.0` — base abstractions required by LangGraph
