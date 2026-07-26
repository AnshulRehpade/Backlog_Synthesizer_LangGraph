"""Concrete EmbeddingTool implementation using sentence-transformers.

Satisfies Requirement 9.2 (embedding computation through Tool interface)
and Requirement 9.6 (translate implementation-specific exceptions to ToolError).
"""

from backlog_synthesizer_langgraph.tools.errors import PermanentToolError, TransientToolError


class SentenceTransformerEmbeddingTool:
    """EmbeddingTool implementation backed by sentence-transformers.

    Generates embedding vectors using a specified sentence-transformers model.
    All library-specific exceptions are translated to ToolError subtypes.

    Args:
        model_name: The sentence-transformers model to load.
            Defaults to "all-MiniLM-L6-v2".
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = self._load_model()

    def _load_model(self):  # noqa: ANN202
        """Load the sentence-transformers model.

        Raises:
            PermanentToolError: If the model cannot be loaded.
        """
        try:
            from sentence_transformers import SentenceTransformer

            return SentenceTransformer(self._model_name)
        except ImportError as e:
            raise PermanentToolError(
                "sentence-transformers library is not installed. "
                "Install it with: pip install sentence-transformers",
                original_error=e,
            ) from e
        except OSError as e:
            raise PermanentToolError(
                f"Failed to load model '{self._model_name}': model not found or "
                f"cannot be downloaded. Error: {e}",
                original_error=e,
            ) from e
        except Exception as e:
            raise PermanentToolError(
                f"Failed to load model '{self._model_name}': {e}",
                original_error=e,
            ) from e

    def generate_embedding(self, text: str) -> list[float]:
        """Generate an embedding vector for the given text.

        Args:
            text: The input text to embed.

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            PermanentToolError: If the input is invalid or the model is misconfigured.
            TransientToolError: If inference fails due to a transient issue.
        """
        if not isinstance(text, str):
            raise PermanentToolError(
                f"Expected text input of type str, got {type(text).__name__}"
            )

        try:
            embedding = self._model.encode(text, show_progress_bar=False)
            return embedding.tolist()
        except RuntimeError as e:
            raise TransientToolError(
                f"Embedding inference failed (possibly transient): {e}",
                original_error=e,
            ) from e
        except MemoryError as e:
            raise TransientToolError(
                f"Embedding inference failed due to memory exhaustion: {e}",
                original_error=e,
            ) from e
        except Exception as e:
            raise PermanentToolError(
                f"Embedding generation failed for input text: {e}",
                original_error=e,
            ) from e
