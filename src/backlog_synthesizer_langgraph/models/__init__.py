"""Data models for the Backlog Synthesizer system."""

from backlog_synthesizer_langgraph.models.inputs import (
    BacklogTicket,
    DocumentType,
    InputDocument,
    SessionInputs,
)
from backlog_synthesizer_langgraph.models.extraction import (
    DocumentError,
    ExtractedItem,
    ExtractionResult,
    TextChunk,
)
from backlog_synthesizer_langgraph.models.gap_detection import (
    ConflictFlag,
    DuplicateFlag,
    GapReport,
    GapReportEntry,
)
from backlog_synthesizer_langgraph.models.output import (
    AcceptanceCriterion,
    Epic,
    OutputMetadata,
    StoryOutput,
    UserStory,
)
from backlog_synthesizer_langgraph.models.memory import (
    AuditEntry,
    SessionState,
)
from backlog_synthesizer_langgraph.models.evaluation import (
    EvaluationCaseResult,
    EvaluationReport,
    GoldenEntry,
    JudgeScores,
)

__all__ = [
    # Input models
    "DocumentType",
    "InputDocument",
    "BacklogTicket",
    "SessionInputs",
    # Extraction models
    "TextChunk",
    "ExtractedItem",
    "ExtractionResult",
    "DocumentError",
    # Gap detection models
    "DuplicateFlag",
    "ConflictFlag",
    "GapReportEntry",
    "GapReport",
    # Output models
    "AcceptanceCriterion",
    "UserStory",
    "Epic",
    "OutputMetadata",
    "StoryOutput",
    # Memory models
    "AuditEntry",
    "SessionState",
    # Evaluation models
    "GoldenEntry",
    "JudgeScores",
    "EvaluationCaseResult",
    "EvaluationReport",
]
