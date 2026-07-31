"""Streamlit UI for the Backlog Synthesizer LangGraph pipeline.

Run with: streamlit run app.py
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


# ─── Pipeline Runner ──────────────────────────────────────────────


async def run_pipeline_for_ui(
    transcript_text: str,
    backlog_tickets: list[BacklogTicket],
    filename: str = "transcript.txt",
) -> dict:
    """Run the pipeline and return the final state.

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
        uploaded_file = st.file_uploader(
            "Upload transcript",
            type=["txt", "md", "pdf"],
            key="transcript_upload",
        )
        if uploaded_file is not None:
            transcript_text = uploaded_file.read().decode("utf-8", errors="replace")
            transcript_filename = uploaded_file.name

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

if run_clicked and transcript_text.strip():
    st.session_state["result"] = None
    st.session_state["session_id"] = str(uuid.uuid4())[:8]
    st.session_state["run_timestamp"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )

    # Pipeline progress
    progress_container = st.container()
    with progress_container:
        st.subheader("Pipeline Progress")

        step_parser = st.status("Parser Agent", expanded=True)
        step_gap = st.status("Gap Detection Agent", expanded=False)
        step_story = st.status("Story Writer Agent", expanded=False)
        step_output = st.status("Output", expanded=False)

        step_parser.update(label="⏳ Parser Agent — extracting items...", state="running")

    # Run pipeline
    start_time = time.time()
    try:
        result = asyncio.run(
            run_pipeline_for_ui(transcript_text, backlog_tickets, transcript_filename)
        )
    except Exception as e:
        st.error(f"Pipeline crashed: {e}")
        result = None

    elapsed = time.time() - start_time

    if result is not None:
        st.session_state["result"] = result
        status = result.get("status", "unknown")

        # Update progress indicators based on what completed
        parser_output = result.get("parser_output")
        gap_output = result.get("gap_output")
        story_output = result.get("story_output")

        if parser_output is not None:
            item_count = len(parser_output.items)
            step_parser.update(
                label=f"✅ Parser Agent — {item_count} items extracted",
                state="complete",
            )
        else:
            step_parser.update(label="❌ Parser Agent — failed", state="error")

        if gap_output is not None:
            step_gap.update(
                label=(
                    f"✅ Gap Detection — {gap_output.total_new} new, "
                    f"{gap_output.total_duplicates} duplicates, "
                    f"{gap_output.total_conflicts} conflicts"
                ),
                state="complete",
            )
        elif parser_output is not None:
            step_gap.update(label="❌ Gap Detection — failed", state="error")
        else:
            step_gap.update(label="⬜ Gap Detection — skipped", state="complete")

        if story_output is not None:
            story_count = sum(len(e.stories) for e in story_output.epics)
            epic_count = len(story_output.epics)
            step_story.update(
                label=f"✅ Story Writer — {story_count} stories, {epic_count} epics",
                state="complete",
            )
        elif gap_output is not None:
            step_story.update(label="❌ Story Writer — failed", state="error")
        else:
            step_story.update(label="⬜ Story Writer — skipped", state="complete")

        step_output.update(
            label=f"✅ Done — {elapsed:.1f}s total ({status})", state="complete"
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

                        tag_str = " ".join(
                            f"`{tag}`" for tag in story.tags
                        )
                        st.markdown(f"**Tags:** {tag_str}")

                        if story.needs_refinement:
                            st.warning("⚠️ This story needs refinement")

        # Tab 2 — Pipeline Stats
        with tab_stats:
            parser_output = result.get("parser_output")
            gap_output = result.get("gap_output")

            # Top-level metrics
            total_stories = sum(len(e.stories) for e in story_output.epics)
            total_epics = len(story_output.epics)
            duplicates = gap_output.total_duplicates if gap_output else 0
            conflicts = gap_output.total_conflicts if gap_output else 0
            total_time_ms = result.get("total_duration_ms", 0)

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Stories", total_stories)
            m2.metric("Epics", total_epics)
            m3.metric("Duplicates Filtered", duplicates)
            m4.metric("Conflicts Detected", conflicts)
            m5.metric("Total Time", f"{total_time_ms / 1000:.1f}s")

            # Per-agent breakdown
            st.subheader("Per-Agent Breakdown")

            agents_data = []
            if parser_output is not None:
                meta = parser_output.metadata if hasattr(parser_output, "metadata") else {}
                agents_data.append({
                    "Agent": "Parser",
                    "Items Processed": len(parser_output.items),
                    "Input Tokens": meta.get("input_tokens", "—"),
                    "Output Tokens": meta.get("output_tokens", "—"),
                })
            if gap_output is not None:
                agents_data.append({
                    "Agent": "Gap Detection",
                    "Items Processed": len(gap_output.entries),
                    "Input Tokens": "—",
                    "Output Tokens": "—",
                })
            agents_data.append({
                "Agent": "Story Writer",
                "Items Processed": total_stories,
                "Input Tokens": "—",
                "Output Tokens": "—",
            })

            st.table(agents_data)

        # Tab 3 — Raw JSON
        with tab_json:
            output_json = story_output.model_dump_json(indent=2)
            st.download_button(
                "📥 Download JSON",
                data=output_json,
                file_name="story_output.json",
                mime="application/json",
            )
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
