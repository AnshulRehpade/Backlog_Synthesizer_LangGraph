# Design Notes: Backlog Synthesizer (LangGraph)

A short write-up of what this agent does, how it is built, and why. The graph diagram is in the [README](README.md#graph-topology).

## What it does and who it's for

Product managers and engineering leads lose hours after every planning meeting turning discussion into backlog tickets, and they often create duplicates of tickets that already exist. Backlog Synthesizer takes a meeting transcript (and optionally architecture docs and the current backlog) and produces structured user stories with acceptance criteria, grouped into epics. Before writing anything, it checks each item against the existing backlog and flags duplicates and conflicts.

## How it's built

- **Framework:** LangGraph `StateGraph` with a typed `PipelineState`. This is a migration of my original hand-rolled orchestrator ([backlog-synthesizer](https://github.com/AnshulRehpade/backlog-synthesizer)), done incrementally with the Strangler Fig pattern.
- **Multi-agent:** three specialist agents run as graph nodes:
  - **Parser:** extracts features, pain points and decisions, each with a confidence score.
  - **Gap Detection:** classifies each item as new, duplicate or conflicting.
  - **Story Writer:** writes stories and groups them into epics.

  Three supporting nodes handle exceptions: an **LLM fallback** node, a **human review** node (`interrupt()`), and an **error** node. All routing logic lives in `graph/routing.py`, never inside nodes.
- **Tools:** document parsing (including PDF), embeddings, vector search, a few-shot example store, LLM generation, and exporters. All sit behind Protocol interfaces (`tools/interfaces.py`), so Anthropic and OpenAI can be swapped through config.
- **Memory and RAG:**
  - *Short-term:* LangGraph checkpointer (MemorySaver for dev, SqliteSaver for persistence). It saves state after every node and enables resume and time-travel debugging.
  - *Long-term:* a ChromaDB vector store.
  - *Retrieval:* Gap Detection uses semantic search over the existing backlog, with a similarity of 0.80 or higher treated as a duplicate and 0.50 or higher as a conflict.
  - *Audit log:* a chronological record of every invocation.
- **Observability:** OpenTelemetry tracing, plus optional LangSmith tracing (`LANGCHAIN_TRACING_V2`).

## Key design decision: a three-tier decision pattern instead of ReAct everywhere

The original version made a ReAct-style LLM call at every decision point, five per run. In the migration I replaced that with three tiers:

1. **Deterministic conditional edges** (zero cost) for the normal path.
2. **One LLM fallback call** only when something unexpected happens: zero items extracted, every item a duplicate, or zero epics produced.
3. **Human review** through `interrupt()` for decisions that need human judgment.

**Tradeoff:** most runs now use no routing LLM calls at all. Routing is predictable and easy to unit-test, because every edge is a plain function. The cost is flexibility. The thresholds (minimum item counts, confidence cut-offs, 0.80/0.50 similarity) are hand-tuned, so an unusual transcript can be misrouted where a ReAct agent might have adapted. The LLM fallback and human review tiers are the safety net for that, and the thresholds are environment variables so they can be tuned without code changes.

## How I tested and evaluated it

- **Tests:** 119 pytest tests across 7 files. They include unit tests for every node, including the fallback and human-review paths, and full-pipeline integration tests. A coverage threshold is enforced in `pyproject.toml`. The original version had 554 tests at 81% coverage and cost about $0.009 per run.
- **Evaluation harness** (`evaluation/run_eval.py`): runs the full graph against a golden dataset of 5 scenarios, each with expected story, duplicate and conflict counts. It reports the per-scenario result and the keyword overlap between the transcript and the generated stories (mean, min, max and standard deviation).
- **Honest limitations:** five scenarios is a small set, and keyword overlap is only a rough proxy for story quality. My next step would be an LLM-as-judge rubric (completeness, faithfulness, acceptance-criteria quality) run as a LangSmith experiment over a larger dataset.

## Demo

There is no hosted demo yet. Run it locally with `streamlit run app.py`. The UI shows live progress for each agent, tabs for stories, stats, raw JSON and the audit log, and Approve/Reject buttons when human review triggers. You can also use the CLI: `python main.py data/sample_transcript.txt --backlog data/sample_backlog.json`.

## Authorship

This is a solo project. I designed and built all of it.
