"""Vector store for semantic embeddings."""

import numpy as np
from pydantic import BaseModel, Field

from research_engineer.tools.base import Tool, ToolError


class VectorStoreConfig(BaseModel):
    """Configuration for vector store."""

    store_type: str = Field("chroma", description="Type of vector store (chroma, faiss)")
    embedding_model: str = Field("sentence-transformers/all-mpnet-base-v2", description="Embedding model")
    embedding_dim: int = Field(768, description="Embedding dimension")
    index_type: str = Field("Flat", description="Index type (Flat, IVF, HNSW)")
    distance_metric: str = Field("cosine", description="Distance metric (cosine, l2, ip)")
    persist_directory: str = Field("data/vector_store", description="Persistence directory")


class VectorSearchResult(BaseModel):
    """Result from vector search."""

    id: str = Field(..., description="Item ID")
    score: float = Field(..., description="Similarity score")
    metadata: dict = Field(default_factory=dict, description="Associated metadata")
    distance: float = Field(..., description="Distance value")


class VectorStore(Tool):
    """Abstract base class for vector stores."""

    async def add(self, ids: list[str], embeddings: np.ndarray, metadatas: list[dict] | None = None) -> bool:
        """Add embeddings to store."""
        raise NotImplementedError

    async def search(
        self,
        query_embedding: np.ndarray,
        limit: int = 10,
        filters: dict | None = None
    ) -> list[VectorSearchResult]:
        """Search for similar embeddings."""
        raise NotImplementedError

    async def delete(self, ids: list[str]) -> bool:
        """Delete embeddings by ID."""
        raise NotImplementedError

    async def update(self, ids: list[str], embeddings: np.ndarray, metadatas: list[dict] | None = None) -> bool:
        """Update embeddings."""
        raise NotImplementedError


try:
    import chromadb
    from chromadb.config import Settings

    class ChromaVectorStore(VectorStore):
        """Chroma-based vector store."""

        def __init__(self, config: VectorStoreConfig):
            self.config = config
            self.persist_directory = Path(config.persist_directory)
            self.persist_directory.mkdir(parents=True, exist_ok=True)

            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory)
            )

            self.collections: dict[str, chromadb.Collection] = {}

        def get_or_create_collection(self, name: str) -> chromadb.Collection:
            """Get or create a collection."""
            if name not in self.collections:
                self.collections[name] = self.client.get_or_create_collection(
                    name=name,
                    metadata={"embedding_dim": self.config.embedding_dim}
                )
            return self.collections[name]

        async def add(
            self,
            ids: list[str],
            embeddings: np.ndarray,
            metadatas: list[dict] | None = None,
            collection_name: str = "default"
        ) -> bool:
            """Add embeddings to collection."""
            try:
                collection = self.get_or_create_collection(collection_name)

                if metadatas is None:
                    metadatas = [{} for _ in ids]

                collection.add(
                    ids=ids,
                    embeddings=embeddings.tolist(),
                    metadatas=metadatas
                )

                return True

            except Exception as e:
                raise ToolError(f"Failed to add embeddings: {e}", None, e)

        async def search(
            self,
            query_embedding: np.ndarray,
            limit: int = 10,
            filters: dict | None = None,
            collection_name: str = "default"
        ) -> list[VectorSearchResult]:
            """Search for similar embeddings."""
            try:
                collection = self.get_or_create_collection(collection_name)

                where = filters if filters else None

                results = collection.query(
                    query_embeddings=query_embedding.tolist(),
                    n_results=limit,
                    where=where,
                    include=["distances", "metadatas"]
                )

                search_results = []
                if results["ids"] and results["ids"][0]:
                    for i, id_ in enumerate(results["ids"][0]):
                        search_results.append(
                            VectorSearchResult(
                                id=id_,
                                score=1.0 - results["distances"][0][i] if results["distances"] else 1.0,
                                metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                                distance=results["distances"][0][i] if results["distances"] else 0.0
                            )
                        )

                return search_results

            except Exception as e:
                raise ToolError(f"Failed to search embeddings: {e}", None, e)

        async def delete(self, ids: list[str], collection_name: str = "default") -> bool:
            """Delete embeddings by ID."""
            try:
                collection = self.get_or_create_collection(collection_name)
                collection.delete(ids=ids)
                return True

            except Exception as e:
                raise ToolError(f"Failed to delete embeddings: {e}", None, e)

        async def update(
            self,
            ids: list[str],
            embeddings: np.ndarray,
            metadatas: list[dict] | None = None,
            collection_name: str = "default"
        ) -> bool:
            """Update embeddings."""
            try:
                collection = self.get_or_create_collection(collection_name)

                if metadatas is None:
                    metadatas = [{} for _ in ids]

                collection.update(
                    ids=ids,
                    embeddings=embeddings.tolist(),
                    metadatas=metadatas
                )

                return True

            except Exception as e:
                raise ToolError(f"Failed to update embeddings: {e}", None, e)

        async def get_stats(self, collection_name: str = "default") -> dict:
            """Get collection statistics."""
            try:
                collection = self.get_or_create_collection(collection_name)
                count = collection.count()

                return {
                    "collection_name": collection_name,
                    "count": count,
                    "embedding_dim": self.config.embedding_dim
                }

            except Exception as e:
                raise ToolError(f"Failed to get stats: {e}", None, e)

except ImportError:
    class ChromaVectorStore:
        """Fallback when chromadb is not available."""

        def __init__(self, config: VectorStoreConfig):
            raise ImportError("chromadb is required for ChromaVectorStore. Install with: pip install chromadb")


try:
    import faiss

    class FAISSVectorStore(VectorStore):
        """FAISS-based vector store."""

        def __init__(self, config: VectorStoreConfig):
            self.config = config
            self.index = None
            self.id_map: dict[int, str] = {}
            self.metadata_map: dict[int, dict] = {}
            self.current_id = 0

            self._init_index()

        def _init_index(self):
            """Initialize FAISS index."""
            if self.config.distance_metric == "cosine":
                self.index = faiss.IndexFlatIP(self.config.embedding_dim)
            elif self.config.distance_metric == "l2":
                self.index = faiss.IndexFlatL2(self.config.embedding_dim)
            else:
                self.index = faiss.IndexFlatIP(self.config.embedding_dim)

        async def add(
            self,
            ids: list[str],
            embeddings: np.ndarray,
            metadatas: list[dict] | None = None,
        ) -> bool:
            """Add embeddings to index."""
            try:
                if self.config.distance_metric == "cosine":
                    faiss.normalize_L2(embeddings)

                self.index.add(embeddings)

                for i, id_ in enumerate(ids):
                    self.id_map[self.current_id] = id_
                    if metadatas:
                        self.metadata_map[self.current_id] = metadatas[i]
                    self.current_id += 1

                return True

            except Exception as e:
                raise ToolError(f"Failed to add embeddings: {e}", None, e)

        async def search(
            self,
            query_embedding: np.ndarray,
            limit: int = 10,
            filters: dict | None = None,
        ) -> list[VectorSearchResult]:
            """Search for similar embeddings."""
            try:
                if self.config.distance_metric == "cosine":
                    faiss.normalize_L2(query_embedding)

                query = query_embedding.reshape(1, -1)

                distances, indices = self.index.search(query, limit)

                results = []
                for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
                    if idx == -1:
                        continue

                    id_ = self.id_map.get(idx, str(idx))
                    metadata = self.metadata_map.get(idx, {})

                    if filters and not self._matches_filters(metadata, filters):
                        continue

                    results.append(
                        VectorSearchResult(
                            id=id_,
                            score=float(1.0 - dist) if self.config.distance_metric == "cosine" else float(-dist),
                            metadata=metadata,
                            distance=float(dist)
                        )
                    )

                return results

            except Exception as e:
                raise ToolError(f"Failed to search embeddings: {e}", None, e)

        def _matches_filters(self, metadata: dict, filters: dict) -> bool:
            """Check if metadata matches filters."""
            for key, value in filters.items():
                if key not in metadata or metadata[key] != value:
                    return False
            return True

        async def delete(self, ids: list[str]) -> bool:
            """Delete embeddings (not supported in FAISS without rebuilding)."""
            raise ToolError("FAISS does not support deletion without rebuilding index", None, None)

        async def update(
            self,
            ids: list[str],
            embeddings: np.ndarray,
            metadatas: list[dict] | None = None,
        ) -> bool:
            """Update embeddings (not supported in FAISS without rebuilding)."""
            raise ToolError("FAISS does not support update without rebuilding index", None, None)

except ImportError:
    class FAISSVectorStore:
        """Fallback when faiss is not available."""

        def __init__(self, config: VectorStoreConfig):
            raise ImportError("faiss is required for FAISSVectorStore. Install with: pip install faiss-cpu")


from pathlib import Path

VectorStoreImpl = ChromaVectorStore
