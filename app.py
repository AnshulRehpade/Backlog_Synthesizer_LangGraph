"""Streamlit UI for the Backlog Synthesizer LangGraph pipeline.

Run with: streamlit run app.py

Features:
- Streaming output: uses astream() for live node-by-node progress
- Cost tracking: displays token usage and estimated cost per agent
- Persistent sessions: checkpointer preserves state between runs
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, ".")

import streamlit as st

from graph.builder import compile_graph
from graph.state import PipelineState
from backlog_synthesizer_langgraph.models.inputs import (
    BacklogTicket,
    DocumentType,
    InputDocument,
    SessionInputs,
)

# ─── Page Config ──────────────────────────────────────────────────

st.set_page_config(
    page_title="Backlog Synthesizer",
    page_icon="🤖",
    layout="wide",
)

# ─── Constants ────────────────────────────────────────────────────

SAMPLE_TRANSCRIPT_PATH = Path("data/sample_transcript.txt")
SAMPLE_BACKLOG_PATH = Path("data/sample_backlog.json")

# Anthropic Haiku pricing (per 1M tokens)
_INPUT_COST_PER_M = 0.80  # $0.80 per 1M input tokens
_OUTPUT_COST_PER_M = 4.00  # $4.00 per 1M output tokens


# ─── Cost Calculation ─────────────────────────────────────────────


def _estimate_cost(input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD based on token counts (Haiku pricing)."""
    return (input_tokens * _INPUT_COST_PER_M / 1_000_000) + (
        output_tokens * _OUTPUT_COST_PER_M / 1_000_000
    )


# ─── Streaming Pipeline Runner ────────────────────────────────────


async def run_pipeline_streaming(
    transcript_text: str,
    backlog_tickets: list[BacklogTicket],
    filename: str = "transcript.txt",
    progress_callback=None,
) -> dict:
    """Run the pipeline with streaming node updates.

    Uses astream() to yield intermediate state after each node completes,
    calling progress_callback with (node_name, state) for live UI updates.

    Args:
        transcript_text: Raw transcript text.
        backlog_tickets: List of existing backlog tickets.
        filename: Name for the input document.
        progress_callback: Called with (node_name, state) after each node.

    Returns:
        Final pipeline state dict.
    """
    content = transcript_text.encode("utf-8")
    document = InputDocument(
        filename=filename,
        document_type=DocumentType.TRANSCRIPT_TXT,
        content=content,
        size_bytes=len(content),
    )

    session_id = str(uuid.uuid4())
    inputs = SessionInputs(
        session_id=session_id,
        documents=[document],
        backlog_tickets=backlog_tickets,
    )

    initial_state: PipelineState = {
        "inputs": inputs,
        "session_id": session_id,
        "parser_output": None,
        "gap_output": None,
        "story_output": None,
        "current_step": "start",
        "status": "in_progress",
        "errors": [],
        "react_notes": [],
        "retry_count": 0,
        "needs_human_review": False,
        "trace_id": str(uuid.uuid4()),
        "total_duration_ms": 0,
    }

    graph = compile_graph()
    config = {"configurable": {"thread_id": session_id}}

    # Stream node-by-node updates
    final_state = None
    async for chunk in graph.astream(initial_state, config=config):
        # chunk is a dict like {"node_name": state_update}
        for node_name, state_update in chunk.items():
            if progress_callback:
                progress_callback(node_name, state_update)
            final_state = state_update

    # If streaming produced no output, fall back to ainvoke
    if final_state is None:
        final_state = await graph.ainvoke(initial_state, config=config)

    return final_state


# Keep the non-streaming version for tests
async def run_pipeline_for_ui(
    transcript_text: str,
    backlog_tickets: list[BacklogTicket],
    filename: str = "transcript.txt",
) -> dict:
    """Run the pipeline and return the final state (non-streaming).

    Args:
        transcript_text: Raw transcript text.
        backlog_tickets: List of existing backlog tickets.
        filename: Name for the input document.

    Returns:
        Final pipeline state dict.
    """
    content = transcript_text.encode("utf-8")
    document = InputDocument(
        filename=filename,
        document_type=DocumentType.TRANSCRIPT_TXT,
        content=content,
        size_bytes=len(content),
    )

    session_id = str(uuid.uuid4())
    inputs = SessionInputs(
        session_id=session_id,
        documents=[document],
        backlog_tickets=backlog_tickets,
    )

    initial_state: PipelineState = {
        "inputs": inputs,
        "session_id": session_id,
        "parser_output": None,
        "gap_output": None,
        "story_output": None,
        "current_step": "start",
        "status": "in_progress",
        "errors": [],
        "react_notes": [],
        "retry_count": 0,
        "needs_human_review": False,
        "trace_id": str(uuid.uuid4()),
        "total_duration_ms": 0,
    }

    graph = compile_graph()
    config = {"configurable": {"thread_id": session_id}}
    final_state = await graph.ainvoke(initial_state, config=config)
    return final_state


def _load_sample_transcript() -> str:
    """Load the sample transcript from disk."""
    if SAMPLE_TRANSCRIPT_PATH.exists():
        return SAMPLE_TRANSCRIPT_PATH.read_text(encoding="utf-8")
    return ""


def _load_sample_backlog() -> list[BacklogTicket]:
    """Load sample backlog tickets from disk."""
    if not SAMPLE_BACKLOG_PATH.exists():
        return []
    data = json.loads(SAMPLE_BACKLOG_PATH.read_text(encoding="utf-8"))
    return [
        BacklogTicket(
            id=item["id"],
            title=item["title"],
            description=item.get("description", ""),
            status=item.get("status", "open"),
            tags=item.get("tags", []),
        )
        for item in data
    ]


def _parse_backlog_json(content: str) -> list[BacklogTicket]:
    """Parse backlog JSON string into tickets."""
    data = json.loads(content)
    if not isinstance(data, list):
        raise ValueError("Backlog JSON must be an array")
    return [
        BacklogTicket(
            id=item["id"],
            title=item["title"],
            description=item.get("description", ""),
            status=item.get("status", "open"),
            tags=item.get("tags", []),
        )
        for item in data
    ]


# ─── Section 1: Header ───────────────────────────────────────────

st.title("Backlog Synthesizer")
st.caption("Powered by LangGraph Multi-Agent Pipeline")

col1, col2, col3 = st.columns(3)
col1.markdown("🤖 **4 AI Agents**")
col2.markdown("🔄 **LangGraph Orchestration**")
col3.markdown("⚡ **ReAct + Human-in-Loop**")

st.divider()

# ─── Section 2: Input Panel (Sidebar) ────────────────────────────

with st.sidebar:
    st.header("Inputs")

    # Transcript input
    st.subheader("Transcript")
    transcript_tab1, transcript_tab2 = st.tabs(["Upload File", "Paste Text"])

    transcript_text = ""
    transcript_filename = "transcript.txt"

    with transcript_tab1:
        uploaded_files = st.file_uploader(
            "Upload transcript(s)",
            type=["txt", "md", "pdf"],
            key="transcript_upload",
            accept_multiple_files=True,
        )
        if uploaded_files:
            parts = []
            names = []
            for f in uploaded_files:
                parts.append(f.read().decode("utf-8", errors="replace"))
                names.append(f.name)
            transcript_text = "\n\n---\n\n".join(parts)
            transcript_filename = ", ".join(names)
            if len(uploaded_files) > 1:
                st.caption(f"{len(uploaded_files)} files loaded")

    with transcript_tab2:
        sample_text = _load_sample_transcript()
        pasted_text = st.text_area(
            "Paste transcript text",
            value=sample_text,
            height=200,
            key="transcript_paste",
        )
        if pasted_text.strip() and not transcript_text:
            transcript_text = pasted_text

    # Backlog input
    st.subheader("Backlog Tickets")
    backlog_tab1, backlog_tab2 = st.tabs(["Upload JSON", "Use Sample"])

    backlog_tickets: list[BacklogTicket] = []

    with backlog_tab1:
        backlog_file = st.file_uploader(
            "Upload backlog JSON",
            type=["json"],
            key="backlog_upload",
        )
        if backlog_file is not None:
            try:
                backlog_tickets = _parse_backlog_json(
                    backlog_file.read().decode("utf-8")
                )
                st.success(f"Loaded {len(backlog_tickets)} tickets")
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                st.error(f"Invalid backlog JSON: {e}")

    with backlog_tab2:
        use_sample = st.checkbox("Use sample backlog (10 tickets)", value=True)
        if use_sample and not backlog_tickets:
            backlog_tickets = _load_sample_backlog()
            st.info(f"{len(backlog_tickets)} sample tickets loaded")

    st.divider()

    # Threshold tuning
    st.subheader("Thresholds")
    dup_threshold = st.slider(
        "Duplicate threshold",
        min_value=0.50,
        max_value=0.95,
        value=0.80,
        step=0.05,
        help="Similarity score at or above which items are flagged as duplicates",
    )
    conflict_threshold = st.slider(
        "Conflict threshold",
        min_value=0.30,
        max_value=0.75,
        value=0.50,
        step=0.05,
        help="Similarity score at or above which items are flagged as conflicts",
    )

    # Apply thresholds via env vars so PipelineConfig.from_env() picks them up
    import os
    os.environ["GAP_DETECTION_DUPLICATE_THRESHOLD"] = str(dup_threshold)
    os.environ["GAP_DETECTION_CONFLICT_THRESHOLD"] = str(conflict_threshold)

    st.divider()

    # Run button
    can_run = bool(transcript_text.strip())
    run_clicked = st.button(
        "🚀 Generate Stories",
        disabled=not can_run,
        use_container_width=True,
        type="primary",
    )

    # Session info
    st.divider()
    st.caption("Session Info")
    if "session_id" in st.session_state:
        st.text(f"ID: {st.session_state['session_id'][:8]}...")
        st.text(f"Run: {st.session_state.get('run_timestamp', 'N/A')}")

# ─── Section 3 & 4: Main Area ────────────────────────────────────

# Show welcome content when no results yet
if "result" not in st.session_state or st.session_state.get("result") is None:
    if not (run_clicked and transcript_text.strip()):

        # Quick-start callout
        st.info(
            "👋 **Ready to go!** A sample transcript is pre-loaded in the sidebar. "
            "Just click **🚀 Generate Stories** to see the pipeline in action.",
            icon="⚡",
        )

        # Pipeline diagram
        st.subheader("Pipeline Architecture")
        st.markdown("""
```mermaid
graph LR
    A[📄 Parser] -->|items| B[🔍 Gap Detection]
    B -->|new + conflicts| C[✍️ Story Writer]
    C -->|stories| D[✅ Done]
    B -.->|all duplicates| E[🤖 LLM Fallback]
    C -.->|quality issues| E
    E -.->|retry| A
    E -.->|retry| B
    E -.->|retry| C
    E -.->|needs human| F[👤 Human Review]
```
        """)

        # How it works
        st.subheader("How It Works")
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.markdown("**1. Parse**")
            st.caption("Extracts decisions, features, pain points from your transcript")
        with col_b:
            st.markdown("**2. Detect Gaps**")
            st.caption("Compares against existing backlog — finds new items, duplicates, conflicts")
        with col_c:
            st.markdown("**3. Write Stories**")
            st.caption("Generates structured user stories with acceptance criteria and tags")
        with col_d:
            st.markdown("**4. Review**")
            st.caption("Quality checks via LLM fallback, optional human-in-the-loop")

        st.divider()

        # Sample output preview
        st.subheader("Example Output")
        st.caption("Here's what the pipeline generates from a sprint planning meeting:")

        with st.expander("Epic: Authentication & SSO", expanded=True):
            st.markdown("**Add SAML 2.0 SSO for Enterprise Authentication**")
            st.markdown("> As an enterprise admin, I want to configure SSO with SAML 2.0, "
                        "so that my team can log in with their corporate credentials")
            st.markdown("""
**Acceptance Criteria:**
- SAML 2.0 authentication flow completes successfully
- User is redirected to IdP and back to application
- Session is established after successful SAML assertion
- Configuration UI allows setting IdP metadata URL

**Tags:** `auth` `sso` `enterprise`
            """)

        with st.expander("Epic: Performance & Reliability"):
            st.markdown("**Fix Password Reset Email Delivery**")
            st.markdown("> As a user, I want password reset emails to arrive reliably, "
                        "so that I can regain access to my account when needed")
            st.markdown("""
**Acceptance Criteria:**
- Email delivery rate exceeds 99%
- Dedicated IP or reputable provider configured
- Delivery failures are monitored with alerts

**Tags:** `email` `reliability` `auth`
            """)

        st.divider()
        st.caption(
            "The full output includes JSON export, JIRA CSV import, and GitHub Issues markdown. "
            "Tune duplicate/conflict thresholds in the sidebar to adjust sensitivity."
        )

if run_clicked and transcript_text.strip():
    st.session_state["result"] = None
    st.session_state["session_id"] = str(uuid.uuid4())[:8]
    st.session_state["run_timestamp"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )

    # Pipeline progress with streaming updates
    st.subheader("Pipeline Progress")

    step_parser = st.status("Parser Agent", expanded=True)
    step_gap = st.status("Gap Detection Agent", expanded=False)
    step_story = st.status("Story Writer Agent", expanded=False)
    step_output = st.status("Output", expanded=False)

    step_parser.update(label="⏳ Parser Agent — extracting items...", state="running")

    # Track which nodes have completed for streaming progress
    completed_nodes = set()
    node_timings = {}
    stream_start = time.time()

    def _on_node_complete(node_name: str, state_update: dict):
        """Callback for streaming: update UI as each node completes."""
        completed_nodes.add(node_name)
        node_timings[node_name] = time.time() - stream_start

        if node_name == "parser":
            parser_out = state_update.get("parser_output")
            if parser_out is not None:
                item_count = len(parser_out.items)
                step_parser.update(
                    label=f"✅ Parser Agent — {item_count} items ({node_timings[node_name]:.1f}s)",
                    state="complete",
                )
                step_gap.update(
                    label="⏳ Gap Detection — analyzing...", state="running"
                )
            else:
                step_parser.update(label="❌ Parser Agent — failed", state="error")

        elif node_name == "gap_detection":
            gap_out = state_update.get("gap_output")
            if gap_out is not None:
                step_gap.update(
                    label=(
                        f"✅ Gap Detection — {gap_out.total_new} new, "
                        f"{gap_out.total_duplicates} dup ({node_timings[node_name]:.1f}s)"
                    ),
                    state="complete",
                )
                step_story.update(
                    label="⏳ Story Writer — generating...", state="running"
                )
            else:
                step_gap.update(label="❌ Gap Detection — failed", state="error")

        elif node_name == "story_writer":
            story_out = state_update.get("story_output")
            if story_out is not None:
                story_count = sum(len(e.stories) for e in story_out.epics)
                epic_count = len(story_out.epics)
                step_story.update(
                    label=f"✅ Story Writer — {story_count} stories, {epic_count} epics ({node_timings[node_name]:.1f}s)",
                    state="complete",
                )
            else:
                step_story.update(label="❌ Story Writer — failed", state="error")

    # Run pipeline with streaming
    start_time = time.time()
    try:
        result = asyncio.run(
            run_pipeline_streaming(
                transcript_text, backlog_tickets, transcript_filename,
                progress_callback=_on_node_complete,
            )
        )
    except Exception as e:
        st.error(f"Pipeline crashed: {e}")
        result = None

    elapsed = time.time() - start_time

    if result is not None:
        st.session_state["result"] = result

        # Final progress update for any nodes the stream didn't catch
        parser_output = result.get("parser_output")
        gap_output = result.get("gap_output")
        story_output = result.get("story_output")

        if "parser" not in completed_nodes:
            if parser_output is not None:
                step_parser.update(
                    label=f"✅ Parser Agent — {len(parser_output.items)} items",
                    state="complete",
                )
            else:
                step_parser.update(label="❌ Parser Agent — failed", state="error")

        if "gap_detection" not in completed_nodes:
            if gap_output is not None:
                step_gap.update(
                    label=f"✅ Gap Detection — {gap_output.total_new} new",
                    state="complete",
                )
            elif parser_output is not None:
                step_gap.update(label="❌ Gap Detection — failed", state="error")
            else:
                step_gap.update(label="⬜ Gap Detection — skipped", state="complete")

        if "story_writer" not in completed_nodes:
            if story_output is not None:
                story_count = sum(len(e.stories) for e in story_output.epics)
                step_story.update(
                    label=f"✅ Story Writer — {story_count} stories",
                    state="complete",
                )
            elif gap_output is not None:
                step_story.update(label="❌ Story Writer — failed", state="error")
            else:
                step_story.update(label="⬜ Story Writer — skipped", state="complete")

        step_output.update(
            label=f"✅ Done — {elapsed:.1f}s total", state="complete"
        )

# ─── Section 5: Error/Warning Banners ────────────────────────────

if "result" in st.session_state and st.session_state["result"] is not None:
    result = st.session_state["result"]
    status = result.get("status", "unknown")

    if status == "partial_failure":
        st.warning(
            "⚠️ Pipeline completed with warnings — partial results shown below"
        )
    elif status == "permanent_failure":
        st.error("❌ Pipeline failed — check your API key and try again")
        errors = result.get("errors", [])
        if errors:
            with st.expander("Error details"):
                for err in errors:
                    if isinstance(err, dict):
                        st.code(
                            f"[{err.get('node', '?')}] {err.get('message', 'unknown')}"
                        )

    if result.get("needs_human_review"):
        st.info("👤 Human review required")
        hr_col1, hr_col2 = st.columns(2)
        hr_col1.button("✅ Approve", key="hr_approve")
        hr_col2.button("❌ Reject", key="hr_reject")

    # ─── Section 4: Results ───────────────────────────────────────

    story_output = result.get("story_output")
    if story_output is not None:
        st.divider()

        tab_stories, tab_stats, tab_json, tab_audit = st.tabs(
            ["Stories", "Pipeline Stats", "Raw JSON", "Audit Log"]
        )

        # Tab 1 — Stories
        with tab_stories:
            for epic in story_output.epics:
                st.subheader(f"{epic.epic_title}")
                st.caption(f"{len(epic.stories)} stories")

                for story in epic.stories:
                    with st.expander(story.title):
                        st.markdown(f"**User Story:** {story.user_story}")

                        st.markdown("**Acceptance Criteria:**")
                        for ac in story.acceptance_criteria:
                            st.markdown(f"- {ac.description}")

                        tag_str = " ".join(f"`{tag}`" for tag in story.tags)
                        st.markdown(f"**Tags:** {tag_str}")

                        if story.needs_refinement:
                            st.warning("⚠️ This story needs refinement")

        # Tab 2 — Pipeline Stats with Cost Tracking
        with tab_stats:
            parser_output = result.get("parser_output")
            gap_output = result.get("gap_output")

            # Top-level metrics
            total_stories = sum(len(e.stories) for e in story_output.epics)
            total_epics = len(story_output.epics)
            duplicates = gap_output.total_duplicates if gap_output else 0
            conflicts = gap_output.total_conflicts if gap_output else 0
            total_time_ms = result.get("total_duration_ms", 0)

            # Compute token totals from metadata
            parser_meta = (
                parser_output.metadata if parser_output and hasattr(parser_output, "metadata") else {}
            )
            parser_input_tok = parser_meta.get("input_tokens", 0)
            parser_output_tok = parser_meta.get("output_tokens", 0)

            # Estimate story writer tokens (approx: ~150 input + ~120 output per story)
            sw_input_tok = total_stories * 150
            sw_output_tok = total_stories * 120

            total_input_tokens = parser_input_tok + sw_input_tok
            total_output_tokens = parser_output_tok + sw_output_tok
            total_cost = _estimate_cost(total_input_tokens, total_output_tokens)

            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Stories", total_stories)
            m2.metric("Epics", total_epics)
            m3.metric("Duplicates", duplicates)
            m4.metric("Conflicts", conflicts)
            m5.metric("Time", f"{total_time_ms / 1000:.1f}s")
            m6.metric("Cost", f"${total_cost:.4f}")

            # Token summary
            st.subheader("Token Usage")
            tok1, tok2, tok3 = st.columns(3)
            tok1.metric("Input Tokens", f"{total_input_tokens:,}")
            tok2.metric("Output Tokens", f"{total_output_tokens:,}")
            tok3.metric("Total Tokens", f"{total_input_tokens + total_output_tokens:,}")

            # Per-agent breakdown
            st.subheader("Per-Agent Breakdown")

            agents_data = []
            if parser_output is not None:
                p_cost = _estimate_cost(parser_input_tok, parser_output_tok)
                agents_data.append({
                    "Agent": "Parser",
                    "Items": len(parser_output.items),
                    "Input Tokens": parser_input_tok,
                    "Output Tokens": parser_output_tok,
                    "Cost ($)": f"{p_cost:.4f}",
                })
            if gap_output is not None:
                agents_data.append({
                    "Agent": "Gap Detection",
                    "Items": len(gap_output.entries),
                    "Input Tokens": 0,
                    "Output Tokens": 0,
                    "Cost ($)": "0.0000",
                })
            sw_cost = _estimate_cost(sw_input_tok, sw_output_tok)
            agents_data.append({
                "Agent": "Story Writer",
                "Items": total_stories,
                "Input Tokens": sw_input_tok,
                "Output Tokens": sw_output_tok,
                "Cost ($)": f"{sw_cost:.4f}",
            })

            st.table(agents_data)

        # Tab 3 — Export & Raw JSON
        with tab_json:
            from backlog_synthesizer_langgraph.tools.exporters import (
                export_github_markdown,
                export_jira_csv,
            )

            st.subheader("Export Formats")
            exp1, exp2, exp3 = st.columns(3)

            # JSON download
            output_json = story_output.model_dump_json(indent=2)
            exp1.download_button(
                "📥 JSON",
                data=output_json,
                file_name="story_output.json",
                mime="application/json",
            )

            # JIRA CSV download
            jira_csv = export_jira_csv(story_output)
            exp2.download_button(
                "📥 JIRA CSV",
                data=jira_csv,
                file_name="stories_jira.csv",
                mime="text/csv",
            )

            # GitHub Markdown download
            gh_md = export_github_markdown(story_output)
            exp3.download_button(
                "📥 GitHub MD",
                data=gh_md,
                file_name="stories_github.md",
                mime="text/markdown",
            )

            st.divider()
            st.subheader("Raw JSON")
            st.json(json.loads(output_json))

        # Tab 4 — Audit Log
        with tab_audit:
            react_notes = result.get("react_notes", [])
            if react_notes:
                for note in react_notes:
                    if isinstance(note, dict):
                        with st.expander(
                            f"[{note.get('node', '?')}] {note.get('timestamp', '')[:19]}"
                        ):
                            st.text(f"Node: {note.get('node', '?')}")
                            st.text(f"Note: {note.get('note', '')}")
                            if "action" in note:
                                st.text(f"Action: {note['action']}")
            else:
                st.info("No audit entries for this run.")

            errors = result.get("errors", [])
            if errors:
                st.subheader("Errors")
                for err in errors:
                    if isinstance(err, dict):
                        st.code(
                            f"[{err.get('node', '?')}] "
                            f"{err.get('error_type', '?')}: "
                            f"{err.get('message', '')}"
                        )

    elif status not in ("permanent_failure",):
        st.info("No stories generated. Check the Audit Log for details.")
