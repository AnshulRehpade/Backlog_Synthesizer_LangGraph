"""Parser Agent for the Backlog Synthesizer system.

Reads raw documents (transcripts, architecture docs) and extracts structured
items (decisions, pain points, feature requests, constraints) using LLM-based
extraction with document parsing and chunking.

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6

Copied from original project pattern — uses Protocol interfaces for all I/O.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from backlog_synthesizer_langgraph.models.extraction import (
    DocumentError,
    ExtractionResult,
    ExtractedItem,
    TextChunk,
)
from backlog_synthesizer_langgraph.models.inputs import InputDocument
from backlog_synthesizer_langgraph.tools.interfaces import (
    DocumentParsingTool,
    LLMGenerationTool,
)

logger = logging.getLogger(__name__)

# Maximum chunk size in characters (approximate token boundary)
_CHUNK_SIZE = 2000
_CHUNK_OVERLAP = 200

SYSTEM_PROMPT = (
    "You are a technical document analyst. You extract structured items from "
    "meeting transcripts, architecture documents, and planning notes. "
    "You produce JSON output only, with no additional commentary or markdown formatting."
)

EXTRACTION_PROMPT_TEMPLATE = """Analyze the following text chunk and extract all actionable items.

Text chunk (from document "{filename}", chunk {chunk_index}):
---
{chunk_text}
---

Extract items in these categories:
- "feature_request": Something users/stakeholders want built
- "pain_point": A problem or frustration mentioned
- "decision": A decision that was made or proposed
- "constraint": A technical or business constraint mentioned

For each item, provide:
- "item_type": one of the categories above
- "text": the relevant text (verbatim or closely paraphrased, max 200 chars)
- "confidence": 0.0-1.0 how confident you are this is a real actionable item
- "stakeholder": who mentioned it (or null if unclear)
- "section_heading": the topic/section context (or null if unclear)
- "tags": 1-3 keyword tags for categorization (lowercase, hyphenated)

Return a JSON array of objects. If no items are found, return an empty array [].
Respond with ONLY the JSON array, no other text."""


def _chunk_text(text: str, chunk_size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> list[TextChunk]:
    """Split text into overlapping chunks for processing.

    Args:
        text: The full document text.
        chunk_size: Maximum characters per chunk.
        overlap: Character overlap between consecutive chunks.

    Returns:
        List of TextChunk objects with index, text, and approximate token count.
    """
    chunks: list[TextChunk] = []
    start = 0
    index = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]

        # Approximate token count (rough: 4 chars per token)
        token_count = len(chunk_text) // 4

        chunks.append(TextChunk(
            index=index,
            text=chunk_text,
            token_count=token_count,
        ))

        index += 1
        start = end - overlap
        if start >= len(text):
            break

    return chunks


def _extract_text_from_document(
    document: InputDocument,
    parser_tool: DocumentParsingTool,
) -> tuple[str | None, DocumentError | None]:
    """Extract text content from a document based on its type.

    Args:
        document: The input document to parse.
        parser_tool: Tool for PDF parsing.

    Returns:
        Tuple of (extracted_text, error). One will be None.
    """
    doc_type = document.document_type.value

    if doc_type in ("transcript_pdf",):
        try:
            text = parser_tool.pdf_to_text(document.content)
            return text, None
        except Exception as e:
            return None, DocumentError(
                filename=document.filename,
                reason=f"PDF parsing failed: {e}",
            )
    elif doc_type in ("transcript_txt", "transcript_md", "architecture_html"):
        # Text-based formats — decode directly
        try:
            text = document.content.decode("utf-8")
            return text, None
        except UnicodeDecodeError as e:
            return None, DocumentError(
                filename=document.filename,
                reason=f"Failed to decode document as UTF-8: {e}",
            )
    elif doc_type == "backlog_json":
        # JSON backlog — decode and convert to readable text
        try:
            text = document.content.decode("utf-8")
            # Parse to validate it's JSON, then use as-is for extraction
            json.loads(text)
            return text, None
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            return None, DocumentError(
                filename=document.filename,
                reason=f"Failed to parse backlog JSON: {e}",
            )
    else:
        return None, DocumentError(
            filename=document.filename,
            reason=f"Unsupported document type: {doc_type}",
        )


def _parse_extraction_response(response: str, chunk_index: int) -> list[ExtractedItem]:
    """Parse LLM extraction response into ExtractedItem objects.

    Args:
        response: Raw LLM JSON response.
        chunk_index: The source chunk index for these items.

    Returns:
        List of validated ExtractedItem objects. Invalid items are skipped.
    """
    # Strip markdown fences if present
    cleaned = response.strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.find("\n")
        if first_newline >= 0:
            cleaned = cleaned[first_newline + 1:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        logger.warning("Failed to parse LLM extraction response as JSON")
        return []

    if not isinstance(data, list):
        # Sometimes LLM wraps in an object
        if isinstance(data, dict) and "items" in data:
            data = data["items"]
        else:
            data = [data] if isinstance(data, dict) else []

    items: list[ExtractedItem] = []
    for raw_item in data:
        if not isinstance(raw_item, dict):
            continue

        try:
            item_type = raw_item.get("item_type", "feature_request")
            if item_type not in ("feature_request", "pain_point", "decision", "constraint"):
                item_type = "feature_request"

            text = str(raw_item.get("text", ""))[:200]
            if not text.strip():
                continue

            confidence = raw_item.get("confidence", 0.5)
            try:
                confidence = max(0.0, min(1.0, float(confidence)))
            except (TypeError, ValueError):
                confidence = 0.5

            tags = raw_item.get("tags", [])
            if not isinstance(tags, list):
                tags = []
            tags = [str(t).lower().strip() for t in tags if t][:3]

            items.append(ExtractedItem(
                item_type=item_type,
                text=text,
                source_chunk_index=chunk_index,
                confidence=confidence,
                stakeholder=raw_item.get("stakeholder"),
                section_heading=raw_item.get("section_heading"),
                tags=tags,
            ))
        except Exception as e:
            logger.warning("Skipping malformed extracted item: %s", e)
            continue

    return items


class ParserAgent:
    """Agent that extracts structured items from raw documents.

    Uses DocumentParsingTool for PDF conversion and LLMGenerationTool for
    intelligent extraction of decisions, pain points, feature requests,
    and constraints from document text.

    Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6
    """

    def __init__(
        self,
        llm_tool: LLMGenerationTool,
        document_parser: DocumentParsingTool,
    ) -> None:
        """Initialize ParserAgent with tool dependencies.

        Args:
            llm_tool: Tool for LLM text generation.
            document_parser: Tool for document parsing (PDF → text).
        """
        self._llm_tool = llm_tool
        self._document_parser = document_parser

    async def parse_documents(
        self,
        documents: list[InputDocument],
    ) -> ExtractionResult:
        """Parse multiple documents and extract structured items.

        Processes each document: extracts text, chunks it, then uses the LLM
        to extract structured items from each chunk. All chunks are processed
        concurrently for improved throughput.

        Args:
            documents: List of InputDocument objects to process.

        Returns:
            ExtractionResult containing all extracted items and any document errors.
        """
        all_items: list[ExtractedItem] = []
        all_errors: list[DocumentError] = []
        total_chunks = 0
        metadata: dict[str, Any] = {
            "document_count": len(documents),
            "chunks_created": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }

        for document in documents:
            text, error = _extract_text_from_document(document, self._document_parser)

            if error is not None:
                all_errors.append(error)
                continue

            if not text or not text.strip():
                all_errors.append(DocumentError(
                    filename=document.filename,
                    reason="Document produced empty text after parsing",
                ))
                continue

            # Chunk the text
            chunks = _chunk_text(text)
            total_chunks += len(chunks)

            # Process all chunks concurrently
            coros = [
                self._extract_from_chunk(chunk, document.filename)
                for chunk in chunks
            ]
            chunk_results = await asyncio.gather(*coros)

            for items in chunk_results:
                all_items.extend(items)

        metadata["chunks_created"] = total_chunks
        # Approximate token counts (4 chars per token)
        metadata["input_tokens"] = sum(len(item.text) // 4 for item in all_items)
        metadata["output_tokens"] = len(all_items) * 50  # rough estimate

        return ExtractionResult(
            items=all_items,
            errors=all_errors,
            metadata=metadata,
        )

    async def _extract_from_chunk(
        self,
        chunk: TextChunk,
        filename: str,
    ) -> list[ExtractedItem]:
        """Extract items from a single text chunk using the LLM.

        Args:
            chunk: The text chunk to process.
            filename: Source document filename (for prompt context).

        Returns:
            List of extracted items. Returns empty list on LLM failure.
        """
        prompt = EXTRACTION_PROMPT_TEMPLATE.format(
            filename=filename,
            chunk_index=chunk.index,
            chunk_text=chunk.text,
        )

        try:
            response = await asyncio.to_thread(
                self._llm_tool.generate,
                prompt,
                SYSTEM_PROMPT,
            )
        except Exception as e:
            logger.warning(
                "LLM extraction failed for chunk %d of %s: %s",
                chunk.index,
                filename,
                e,
            )
            return []

        return _parse_extraction_response(response, chunk.index)
