"""Concrete implementation of the DocumentParsingTool interface.

Uses PyMuPDF (fitz) for PDF-to-text conversion. All library-specific
exceptions are translated to interface-defined ToolError subtypes.

Note: Text chunking is handled internally by ParserAgent using tiktoken,
not by this tool. Chunking is pure in-memory computation, not external I/O.
"""

from __future__ import annotations

import fitz  # PyMuPDF

from backlog_synthesizer_langgraph.tools.errors import PermanentToolError, TransientToolError


class PyMuPDFDocumentParser:
    """DocumentParsingTool implementation backed by PyMuPDF.

    Satisfies the DocumentParsingTool Protocol defined in
    backlog_synthesizer_langgraph.tools.interfaces.
    """

    def pdf_to_text(self, content: bytes) -> str:
        """Convert PDF binary content to plain text.

        Uses PyMuPDF to extract text page-by-page, preserving paragraph
        boundaries by joining pages with double newlines.

        Args:
            content: Raw PDF file bytes.

        Returns:
            Extracted text content preserving paragraph boundaries.

        Raises:
            PermanentToolError: If the PDF is malformed or cannot be parsed.
            TransientToolError: If a transient I/O error occurs.
        """
        if not content:
            raise PermanentToolError(
                "Cannot parse empty PDF content", original_error=None
            )

        try:
            doc = fitz.open(stream=content, filetype="pdf")
        except Exception as e:
            raise PermanentToolError(
                f"Failed to open PDF document: {e}", original_error=e
            ) from e

        try:
            pages: list[str] = []
            for page in doc:
                page_text = page.get_text()
                if page_text.strip():
                    pages.append(page_text.strip())
            return "\n\n".join(pages)
        except MemoryError as e:
            raise TransientToolError(
                f"Insufficient memory to process PDF: {e}", original_error=e
            ) from e
        except Exception as e:
            raise PermanentToolError(
                f"Error extracting text from PDF: {e}", original_error=e
            ) from e
        finally:
            doc.close()
