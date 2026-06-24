"""Embedding strategy for memory system."""

import numpy as np
from pydantic import BaseModel, Field

from research_engineer.models.memory import MemoryType
from research_engineer.tools.base import ToolError


class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation."""

    default_model: str = Field("sentence-transformers/all-mpnet-base-v2", description="Default embedding model")
    batch_size: int = Field(32, description="Batch size for embedding")
    max_length: int = Field(512, description="Maximum text length")
    cache_embeddings: bool = Field(True, description="Whether to cache embeddings")
    cache_path: str = Field("data/embedding_cache", description="Cache directory")


try:
    from sentence_transformers import SentenceTransformer

    class EmbeddingStrategy:
        """Strategy for generating embeddings."""

        EMBEDDING_MODELS = {
            MemoryType.PAPER: "sentence-transformers/all-mpnet-base-v2",
            MemoryType.REPOSITORY: "sentence-transformers/all-mpnet-base-v2",
            MemoryType.EXPERIMENT_PLAN: "sentence-transformers/all-mpnet-base-v2",
            MemoryType.PATCH: "microsoft/codebert-base",
            MemoryType.ARCHITECTURE_DECISION: "sentence-transformers/all-mpnet-base-v2",
            MemoryType.RESEARCH_INSIGHT: "sentence-transformers/all-mpnet-base-v2",
            MemoryType.FAILED_APPROACH: "sentence-transformers/all-mpnet-base-v2",
            MemoryType.SUCCESSFUL_APPROACH: "sentence-transformers/all-mpnet-base-v2",
        }

        def __init__(self, config: EmbeddingConfig | None = None):
            self.config = config or EmbeddingConfig()
            self.models: dict[str, SentenceTransformer] = {}
            self.cache: dict[str, np.ndarray] = {}

        def _get_model(self, memory_type: MemoryType) -> SentenceTransformer:
            """Get or create embedding model for memory type."""
            model_name = self.EMBEDDING_MODELS.get(memory_type, self.config.default_model)

            if model_name not in self.models:
                self.models[model_name] = SentenceTransformer(model_name)

            return self.models[model_name]

        async def embed(self, content: str, memory_type: MemoryType) -> np.ndarray:
            """Generate embedding for content."""
            cache_key = f"{memory_type.value}:{hash(content)}"

            if self.config.cache_embeddings and cache_key in self.cache:
                return self.cache[cache_key]

            try:
                model = self._get_model(memory_type)

                truncated = content[:self.config.max_length] if len(content) > self.config.max_length else content

                embedding = model.encode([truncated], convert_to_numpy=True, show_progress_bar=False)[0]

                if self.config.cache_embeddings:
                    self.cache[cache_key] = embedding

                return embedding

            except Exception as e:
                raise ToolError(f"Failed to generate embedding: {e}", None, e)

        async def embed_batch(self, contents: list[str], memory_type: MemoryType) -> np.ndarray:
            """Generate embeddings for batch of contents."""
            try:
                model = self._get_model(memory_type)

                truncated = [
                    c[:self.config.max_length] if len(c) > self.config.max_length else c
                    for c in contents
                ]

                embeddings = model.encode(
                    truncated,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    batch_size=self.config.batch_size
                )

                return embeddings

            except Exception as e:
                raise ToolError(f"Failed to generate batch embeddings: {e}", None, e)

        def embed_text(self, text: str, model_name: str | None = None) -> np.ndarray:
            """Synchronous embedding for simple cases."""
            model_name = model_name or self.config.default_model

            if model_name not in self.models:
                self.models[model_name] = SentenceTransformer(model_name)

            model = self.models[model_name]
            return model.encode([text], convert_to_numpy=True)[0]

        async def clear_cache(self):
            """Clear embedding cache."""
            self.cache.clear()

        def get_embedding_dim(self, memory_type: MemoryType) -> int:
            """Get embedding dimension for memory type."""
            model_name = self.EMBEDDING_MODELS.get(memory_type, self.config.default_model)

            if model_name in self.models:
                return self.models[model_name].get_sentence_embedding_dimension()

            temp_model = SentenceTransformer(model_name)
            return temp_model.get_sentence_embedding_dimension()

except ImportError:
    class EmbeddingStrategy:
        """Fallback when sentence-transformers is not available."""

        def __init__(self, config: EmbeddingConfig | None = None):
            raise ImportError(
                "sentence-transformers is required for EmbeddingStrategy. "
                "Install with: pip install sentence-transformers"
            )
