"""Evaluation framework data models for the Backlog Synthesizer system."""

from pydantic import BaseModel, Field

from backlog_synthesizer_langgraph.models.output import UserStory


class GoldenEntry(BaseModel):
    """A single entry in the golden dataset for evaluation."""

    transcript: str
    expected_stories: list[UserStory]


class GoldenDatasetEntry(BaseModel):
    """A single entry in the expanded golden dataset for CLI-based evaluation."""

    id: str
    description: str
    transcript: str
    existing_backlog: list[dict] = Field(default_factory=list)
    expected_stories_count: int = 0
    expected_duplicates: int = 0
    expected_conflicts: int = 0
    tags: list[str] = Field(default_factory=list)


class EvalRunResult(BaseModel):
    """Result of evaluating a single golden entry."""

    golden_id: str
    keyword_overlap: float = 0.0
    stories_generated: int = 0
    passed: bool = True
    failure_reason: str | None = None


class EvalSummary(BaseModel):
    """Aggregate evaluation results."""

    total_entries: int
    passed: int
    failed: int
    keyword_overlap_mean: float = 0.0
    keyword_overlap_std: float = 0.0
    keyword_overlap_min: float = 0.0
    keyword_overlap_max: float = 0.0
    success_rate: float = 0.0
    results: list[EvalRunResult] = Field(default_factory=list)


class JudgeScores(BaseModel):
    """LLM-as-judge scores for a generated user story."""

    relevance: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    clarity: int = Field(ge=1, le=5)


class EvaluationCaseResult(BaseModel):
    """Result of evaluating a single golden dataset entry."""

    case_index: int
    keyword_overlap_score: float = Field(ge=0.0, le=1.0)
    judge_scores: JudgeScores | None = None
    failure_reason: str | None = None


class EvaluationReport(BaseModel):
    """Complete evaluation report with per-case and aggregate metrics."""

    results: list[EvaluationCaseResult]
    aggregate_keyword_overlap_mean: float
    aggregate_keyword_overlap_min: float
    aggregate_relevance_mean: float | None = None
    aggregate_completeness_mean: float | None = None
    aggregate_clarity_mean: float | None = None
