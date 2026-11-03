"""Tests that the parser keeps talk about the meeting itself out of the items."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from backlog_synthesizer_langgraph.agents.parser import _CHUNK_SIZE, MEETING_NOTE, ParserAgent
from backlog_synthesizer_langgraph.models.inputs import DocumentType, InputDocument, SessionInputs
import graph.builder  # noqa: F401  (nodes.* must be imported after graph, or the import is circular)
from nodes.parser_node import parser_node


def _doc(text: str = "Sarah: We need CSV export. Let's size it later.") -> InputDocument:
    return InputDocument(
        filename="m.txt", document_type=DocumentType.TRANSCRIPT_TXT,
        content=text.encode(), size_bytes=len(text),
    )


def _llm(*items: dict) -> MagicMock:
    llm = MagicMock()
    llm.generate.return_value = json.dumps(list(items))
    return llm


_REQUIREMENT = {"item_type": "feature_request", "text": "Add CSV export", "confidence": 0.9}
_RECAP = {"item_type": MEETING_NOTE, "text": "Size CSV export in the second half", "confidence": 0.9}


async def test_meeting_notes_are_set_aside_and_recorded():
    result = await ParserAgent(_llm(_REQUIREMENT, _RECAP), MagicMock()).parse_documents([_doc()])

    assert [i.text for i in result.items] == ["Add CSV export"]
    assert result.metadata["meeting_notes"] == ["Size CSV export in the second half"]


async def test_prompt_defines_meeting_notes():
    llm = _llm(_REQUIREMENT)
    await ParserAgent(llm, MagicMock()).parse_documents([_doc()])
    prompt = llm.generate.call_args.args[0]
    assert '"meeting_note"' in prompt and "recaps or summaries" in prompt


async def test_typical_meeting_is_read_in_one_chunk():
    """A whole meeting in one chunk lets the model see a recap repeats earlier items."""
    meeting = "Sarah: something we need to build. " * 250  # ~9,000 characters
    assert len(meeting) < _CHUNK_SIZE
    llm = _llm(_REQUIREMENT)
    result = await ParserAgent(llm, MagicMock()).parse_documents([_doc(meeting)])
    assert result.metadata["chunks_created"] == 1
    assert llm.generate.call_count == 1


async def test_parser_node_uses_temperature_zero_and_reports_notes():
    with patch("nodes.parser_node.PipelineConfig") as config_cls:
        config = MagicMock()
        config.create_llm_tool.return_value = _llm(_REQUIREMENT, _RECAP)
        config_cls.from_env.return_value = config
        state = {
            "inputs": SessionInputs(session_id="s", documents=[_doc()], backlog_tickets=[]),
            "session_id": "s", "react_notes": [], "errors": [], "total_duration_ms": 0,
        }
        result = await parser_node(state)

    config.create_llm_tool.assert_called_once_with(temperature=0.0)
    assert "Set aside 1 meeting notes" in result["react_notes"][-1]["note"]
