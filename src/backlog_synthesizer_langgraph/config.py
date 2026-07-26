"""Dependency injection configuration for the Backlog Synthesizer pipeline.

Provides PipelineConfig which binds concrete tool implementations to their
Protocol interfaces based on environment variables or an optional config file.
Swapping tool implementations requires ONLY changing the config — zero changes
to agent source code.

Copied from original project (unchanged).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backlog_synthesizer_langgraph.tools.interfaces import (
    DocumentParsingTool,
    EmbeddingTool,
    LLMGenerationTool,
    VectorSearchTool,
)


@dataclass
class PipelineConfig:
    """Configuration that binds concrete tool implementations to interfaces.

    Reads settings from environment variables with sensible defaults.
    Optionally loads overrides from a JSON config file specified by the
    ``BACKLOG_SYNTHESIZER_CONFIG`` environment variable or passed explicitly.

    Usage::

        config = PipelineConfig.from_env()
        llm = config.create_llm_tool()
        parser = config.create_document_parser()

    Swapping implementations (e.g., a different embedding model) requires
    only changing the relevant environment variable or config file entry —
    no agent code changes needed.
    """

    # LLM provider: "openai" or "anthropic"
    llm_provider: str = "anthropic"

    # OpenAI settings
    openai_api_key: str | None = field(default=None, repr=False)
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str | None = None
    openai_timeout: float = 60.0

    # Anthropic settings
    anthropic_api_key: str | None = field(default=None, repr=False)
    anthropic_model: str = "claude-haiku-4-20250414"
    anthropic_max_tokens: int = 4096
    anthropic_timeout: float = 60.0

    # Embedding settings
    embedding_model: str = "all-MiniLM-L6-v2"

    # ChromaDB / Vector search settings
    chroma_collection: str = "backlog_items"
    chroma_persist_dir: str | None = None

    # Gap detection thresholds
    gap_detection_duplicate_threshold: float = 0.80
    gap_detection_conflict_threshold: float = 0.50

    # ReAct reasoning
    react_reasoning_enabled: bool = True

    # Tokenizer
    tokenizer_model: str = "cl100k_base"

    # Evaluation thresholds
    eval_keyword_threshold: float = 0.60
    eval_f1_threshold: float = 0.60
    eval_regression_threshold: float = 0.05

    # Few-shot prompting
    few_shot_enabled: bool = True

    # Observability
    otel_enabled: bool = True
    runs_dir: str = "runs"

    @classmethod
    def from_env(cls, config_path: str | Path | None = None) -> PipelineConfig:
        """Create a PipelineConfig from environment variables and optional config file.

        Resolution order (later sources override earlier ones):
        1. Defaults defined in this class
        2. Environment variables
        3. Config file (JSON) — path from ``config_path`` param or
           ``BACKLOG_SYNTHESIZER_CONFIG`` env var

        Args:
            config_path: Optional path to a JSON config file. If not provided,
                falls back to the ``BACKLOG_SYNTHESIZER_CONFIG`` environment variable.

        Returns:
            A fully-resolved PipelineConfig instance.
        """
        # Start with env var values
        env_values: dict[str, Any] = {}

        llm_provider = os.environ.get("LLM_PROVIDER")
        if llm_provider:
            env_values["llm_provider"] = llm_provider

        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            env_values["openai_api_key"] = api_key

        model = os.environ.get("OPENAI_MODEL")
        if model:
            env_values["openai_model"] = model

        base_url = os.environ.get("OPENAI_BASE_URL")
        if base_url:
            env_values["openai_base_url"] = base_url

        timeout = os.environ.get("OPENAI_TIMEOUT")
        if timeout:
            env_values["openai_timeout"] = float(timeout)

        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        if anthropic_key:
            env_values["anthropic_api_key"] = anthropic_key

        anthropic_model = os.environ.get("ANTHROPIC_MODEL")
        if anthropic_model:
            env_values["anthropic_model"] = anthropic_model

        anthropic_max_tokens = os.environ.get("ANTHROPIC_MAX_TOKENS")
        if anthropic_max_tokens:
            env_values["anthropic_max_tokens"] = int(anthropic_max_tokens)

        anthropic_timeout = os.environ.get("ANTHROPIC_TIMEOUT")
        if anthropic_timeout:
            env_values["anthropic_timeout"] = float(anthropic_timeout)

        embedding_model = os.environ.get("EMBEDDING_MODEL")
        if embedding_model:
            env_values["embedding_model"] = embedding_model

        chroma_collection = os.environ.get("CHROMA_COLLECTION")
        if chroma_collection:
            env_values["chroma_collection"] = chroma_collection

        chroma_persist_dir = os.environ.get("CHROMA_PERSIST_DIR")
        if chroma_persist_dir:
            env_values["chroma_persist_dir"] = chroma_persist_dir

        gap_dup_threshold = os.environ.get("GAP_DETECTION_DUPLICATE_THRESHOLD")
        if gap_dup_threshold:
            env_values["gap_detection_duplicate_threshold"] = float(gap_dup_threshold)

        gap_conflict_threshold = os.environ.get("GAP_DETECTION_CONFLICT_THRESHOLD")
        if gap_conflict_threshold:
            env_values["gap_detection_conflict_threshold"] = float(gap_conflict_threshold)

        react_enabled = os.environ.get("REACT_REASONING_ENABLED")
        if react_enabled is not None:
            env_values["react_reasoning_enabled"] = react_enabled.lower() in ("true", "1", "yes")

        tokenizer_model = os.environ.get("TOKENIZER_MODEL")
        if tokenizer_model:
            env_values["tokenizer_model"] = tokenizer_model

        otel_enabled = os.environ.get("OTEL_ENABLED")
        if otel_enabled is not None:
            env_values["otel_enabled"] = otel_enabled.lower() in ("true", "1", "yes")

        runs_dir = os.environ.get("RUNS_DIR")
        if runs_dir:
            env_values["runs_dir"] = runs_dir

        eval_keyword_threshold = os.environ.get("EVAL_KEYWORD_THRESHOLD")
        if eval_keyword_threshold:
            env_values["eval_keyword_threshold"] = float(eval_keyword_threshold)

        eval_f1_threshold = os.environ.get("EVAL_F1_THRESHOLD")
        if eval_f1_threshold:
            env_values["eval_f1_threshold"] = float(eval_f1_threshold)

        eval_regression_threshold = os.environ.get("EVAL_REGRESSION_THRESHOLD")
        if eval_regression_threshold:
            env_values["eval_regression_threshold"] = float(eval_regression_threshold)

        few_shot_enabled = os.environ.get("FEW_SHOT_ENABLED")
        if few_shot_enabled is not None:
            env_values["few_shot_enabled"] = few_shot_enabled.lower() in ("true", "1", "yes")

        # Load config file overrides
        resolved_path = config_path or os.environ.get("BACKLOG_SYNTHESIZER_CONFIG")
        if resolved_path:
            file_values = cls._load_config_file(Path(resolved_path))
            env_values.update(file_values)

        return cls(**env_values)

    @staticmethod
    def _load_config_file(path: Path) -> dict[str, Any]:
        """Load configuration overrides from a JSON file."""
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        try:
            content = path.read_text(encoding="utf-8")
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file {path}: {e}") from e

        if not isinstance(data, dict):
            raise ValueError(f"Config file {path} must contain a JSON object")

        valid_keys = {
            "llm_provider",
            "openai_api_key",
            "openai_model",
            "openai_base_url",
            "openai_timeout",
            "anthropic_api_key",
            "anthropic_model",
            "anthropic_max_tokens",
            "anthropic_timeout",
            "embedding_model",
            "chroma_collection",
            "chroma_persist_dir",
            "gap_detection_duplicate_threshold",
            "gap_detection_conflict_threshold",
            "react_reasoning_enabled",
            "tokenizer_model",
            "otel_enabled",
            "runs_dir",
            "eval_keyword_threshold",
            "eval_f1_threshold",
            "eval_regression_threshold",
            "few_shot_enabled",
        }

        return {k: v for k, v in data.items() if k in valid_keys}

    def create_document_parser(self) -> DocumentParsingTool:
        """Create a DocumentParsingTool instance."""
        from backlog_synthesizer_langgraph.tools.document_parsing import PyMuPDFDocumentParser

        return PyMuPDFDocumentParser()

    def create_embedding_tool(self) -> EmbeddingTool:
        """Create an EmbeddingTool instance configured with the specified model."""
        from backlog_synthesizer_langgraph.tools.embedding import SentenceTransformerEmbeddingTool

        return SentenceTransformerEmbeddingTool(model_name=self.embedding_model)

    def create_vector_search_tool(self) -> VectorSearchTool:
        """Create a VectorSearchTool instance configured with collection and persist settings."""
        from backlog_synthesizer_langgraph.tools.vector_search import ChromaVectorSearchTool

        return ChromaVectorSearchTool(
            collection_name=self.chroma_collection,
            persist_directory=self.chroma_persist_dir,
        )

    def create_llm_tool(self) -> LLMGenerationTool:
        """Create an LLMGenerationTool instance based on the configured provider."""
        if self.llm_provider == "anthropic":
            from backlog_synthesizer_langgraph.tools.anthropic_generation import AnthropicGenerationTool

            return AnthropicGenerationTool(
                api_key=self.anthropic_api_key,
                model=self.anthropic_model,
                max_tokens=self.anthropic_max_tokens,
                timeout=self.anthropic_timeout,
            )
        elif self.llm_provider == "openai":
            from backlog_synthesizer_langgraph.tools.llm_generation import OpenAIGenerationTool

            return OpenAIGenerationTool(
                api_key=self.openai_api_key,
                model=self.openai_model,
                base_url=self.openai_base_url,
                timeout=self.openai_timeout,
            )
        else:
            raise ValueError(
                f"Unknown LLM provider: '{self.llm_provider}'. "
                f"Supported providers: 'openai', 'anthropic'."
            )
