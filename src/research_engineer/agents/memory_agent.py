"""Memory Agent for persistent knowledge management."""


from pydantic import BaseModel, Field

from research_engineer.llm import LLMProvider
from research_engineer.models.coding import GeneratedPatch, ReviewResult
from research_engineer.models.memory import (
    ArchitectureDecisionMemory,
    ExperimentPlanMemory,
    FailedApproachMemory,
    MemoryFilters,
    MemoryRelationship,
    MemoryResult,
    MemoryStats,
    MemoryType,
    PaperMemory,
    PatchMemory,
    RelationshipType,
    RepositoryMemory,
    ResearchInsightMemory,
    SuccessfulApproachMemory,
)
from research_engineer.models.paper import Paper
from research_engineer.models.plan import EngineeringReport
from research_engineer.models.planner import PlanResult
from research_engineer.models.repo import Repository, RepositorySummary
from research_engineer.models.summary import ResearchSummary
from research_engineer.tools.base import ToolError
from research_engineer.tools.embedding_strategy import (
    EmbeddingConfig,
    EmbeddingStrategy,
)
from research_engineer.tools.memory_graph import MemoryKnowledgeGraph
from research_engineer.tools.memory_query import QueryProcessor, SemanticQuery
from research_engineer.tools.memory_storage import MemoryStorageTool
from research_engineer.tools.relationship_detector import RelationshipDetector
from research_engineer.tools.retrieval_strategies import (
    HybridSearchStrategy,
    RetrievalQuery,
)
from research_engineer.tools.vector_store import ChromaVectorStore, VectorStoreConfig


class MemoryConfig(BaseModel):
    """Configuration for Memory Agent."""

    db_path: str = Field("data/research_engineer.db", description="Database path")
    vector_store_path: str = Field("data/vector_store", description="Vector store path")
    embedding_model: str = Field("sentence-transformers/all-mpnet-base-v2", description="Embedding model")
    auto_consolidate: bool = Field(True, description="Auto-consolidate memories")
    auto_detect_relationships: bool = Field(True, description="Auto-detect relationships")
    min_relationship_confidence: float = Field(0.7, description="Minimum relationship confidence")
    batch_size: int = Field(32, description="Batch size for operations")
    log_access: bool = Field(True, description="Log memory access")


class MemoryAgent:
    """Central agent for memory operations."""

    def __init__(self, config: MemoryConfig | None = None, llm: LLMProvider | None = None):
        self.agent_name: str = "MemoryAgent"
        self.config = config or MemoryConfig()

        self.storage = MemoryStorageTool(db_path=self.config.db_path)

        vector_config = VectorStoreConfig(
            persist_directory=self.config.vector_store_path,
            embedding_model=self.config.embedding_model
        )
        try:
            self.vector_store = ChromaVectorStore(vector_config)
        except (ImportError, Exception):
            self.vector_store = None

        try:
            self.embedding_strategy = EmbeddingStrategy(EmbeddingConfig())
        except (ImportError, Exception):
            self.embedding_strategy = None

        self.query_processor = QueryProcessor()

        self.graph: MemoryKnowledgeGraph | None = None
        try:
            self.graph = MemoryKnowledgeGraph()
        except ImportError:
            self.graph = None

        self.relationship_detector = RelationshipDetector(storage=self.storage)

        self.hybrid_strategy = HybridSearchStrategy()
        from research_engineer.agents._llm_support import resolve_llm
        self.llm_provider = resolve_llm(self.agent_name, llm)

    async def store_paper(
        self,
        paper: Paper,
        summary: ResearchSummary,
        plan: EngineeringReport
    ) -> str:
        """Store paper analysis results in memory."""
        try:
            memory = PaperMemory(
                paper_id=paper.paper_id,
                title=paper.title,
                authors=[a.model_dump() for a in paper.authors],
                abstract=paper.abstract or "",
                research_summary=summary.model_dump(mode="json"),
                engineering_report=plan.model_dump(mode="json"),
                tags=self._extract_tags(paper.title, paper.abstract),
                confidence_score=0.95
            )

            if self.embedding_strategy and memory.abstract:
                embedding = await self.embedding_strategy.embed(memory.abstract, MemoryType.PAPER)
                memory.embedding_key = f"paper:{memory.memory_id}"

                if self.vector_store:
                    await self.vector_store.add(
                        ids=[memory.embedding_key],
                        embeddings=embedding.reshape(1, -1),
                        metadatas=[{"memory_type": MemoryType.PAPER.value, "paper_id": paper.paper_id}]
                    )

            from research_engineer.tools.memory_storage import MemoryStorageInput

            input = MemoryStorageInput(memory=memory, operation="store")
            await self.storage.execute(input)

            if self.config.log_access:
                await self.storage.log_access(memory.memory_id, "write", "MemoryAgent", "store_paper")

            if self.config.auto_detect_relationships:
                await self._detect_paper_relationships(memory)

            return memory.memory_id

        except Exception as e:
            raise ToolError(f"Failed to store paper: {e}", None, e)

    async def store_repository(self, repo: Repository, analysis: RepositorySummary) -> str:
        """Store repository analysis in memory."""
        try:
            memory = RepositoryMemory(
                repo_path=str(repo.path),
                repo_name=repo.name,
                repo_type=repo.repo_type.value if hasattr(repo.repo_type, "value") else str(repo.repo_type),
                architecture_summary=analysis.architecture_summary or "",
                key_components=analysis.key_components or [],
                training_pipeline=analysis.training_pipeline.model_dump(mode="json") if analysis.training_pipeline else {},
                config_structure=analysis.config_structure or {},
                dependencies=analysis.dependencies or [],
                tags=self._extract_tags(repo.name, analysis.architecture_summary),
                confidence_score=0.9
            )

            if self.embedding_strategy and memory.architecture_summary:
                embedding = await self.embedding_strategy.embed(memory.architecture_summary, MemoryType.REPOSITORY)
                memory.embedding_key = f"repo:{memory.memory_id}"

                if self.vector_store:
                    await self.vector_store.add(
                        ids=[memory.embedding_key],
                        embeddings=embedding.reshape(1, -1),
                        metadatas=[{"memory_type": MemoryType.REPOSITORY.value, "repo_path": str(repo.path)}]
                    )

            from research_engineer.tools.memory_storage import MemoryStorageInput

            input = MemoryStorageInput(memory=memory, operation="store")
            await self.storage.execute(input)

            if self.config.log_access:
                await self.storage.log_access(memory.memory_id, "write", "MemoryAgent", "store_repository")

            return memory.memory_id

        except Exception as e:
            raise ToolError(f"Failed to store repository: {e}", None, e)

    async def store_plan(self, plan_result: PlanResult) -> str:
        """Store experiment plan in memory."""
        try:
            memory = ExperimentPlanMemory(
                plan_id=plan_result.plan_id,
                paper_id=plan_result.paper_id,
                repo_path=plan_result.repo_path,
                compatibility_report=plan_result.compatibility.model_dump(mode="json"),
                implementation_plan=plan_result.implementation_plan.model_dump(mode="json"),
                impact_report=plan_result.impact.model_dump(mode="json"),
                experiment_matrix=plan_result.experiment_matrix.model_dump(mode="json"),
                validation_plan=plan_result.validation_plan.model_dump(mode="json"),
                risk_assessment=plan_result.risk_assessment.model_dump(mode="json"),
                compute_estimate=plan_result.compute_estimate.model_dump(mode="json"),
                result_prediction=plan_result.result_prediction.model_dump(mode="json"),
                output_dir=plan_result.output_dir,
                tags=self._extract_tags(plan_result.paper_id, plan_result.repo_path),
                confidence_score=0.85
            )

            if self.embedding_strategy:
                description = f"{plan_result.paper_id} {plan_result.repo_path}"
                embedding = await self.embedding_strategy.embed(description, MemoryType.EXPERIMENT_PLAN)
                memory.embedding_key = f"plan:{memory.memory_id}"

                if self.vector_store:
                    await self.vector_store.add(
                        ids=[memory.embedding_key],
                        embeddings=embedding.reshape(1, -1),
                        metadatas=[{"memory_type": MemoryType.EXPERIMENT_PLAN.value, "plan_id": plan_result.plan_id}]
                    )

            from research_engineer.tools.memory_storage import MemoryStorageInput

            input = MemoryStorageInput(memory=memory, operation="store")
            await self.storage.execute(input)

            if self.config.log_access:
                await self.storage.log_access(memory.memory_id, "write", "MemoryAgent", "store_plan")

            await self._link_plan_to_paper(memory)

            return memory.memory_id

        except Exception as e:
            raise ToolError(f"Failed to store plan: {e}", None, e)

    async def store_patch(
        self,
        patch: GeneratedPatch,
        review: ReviewResult,
        applied: bool,
        implementation_id: str,
        paper_id: str | None = None,
        repo_path: str = ""
    ) -> str:
        """Store implementation patch in memory."""
        try:
            memory = PatchMemory(
                patch_id=patch.patch_id,
                implementation_id=implementation_id,
                paper_id=paper_id,
                repo_path=repo_path,
                patch_content=patch.content,
                files_modified=patch.files_modified,
                change_type=patch.change_type.value if hasattr(patch.change_type, "value") else str(patch.change_type),
                test_coverage=patch.test_coverage if hasattr(patch, "test_coverage") else 0.0,
                review_result=review.model_dump(mode="json") if review else {},
                applied_successfully=applied,
                tags=self._extract_tags(patch.content[:500]),
                confidence_score=0.9 if applied else 0.5
            )

            if self.embedding_strategy and patch.content:
                embedding = await self.embedding_strategy.embed(patch.content[:2000], MemoryType.PATCH)
                memory.embedding_key = f"patch:{memory.memory_id}"

                if self.vector_store:
                    await self.vector_store.add(
                        ids=[memory.embedding_key],
                        embeddings=embedding.reshape(1, -1),
                        metadatas=[{"memory_type": MemoryType.PATCH.value, "patch_id": patch.patch_id}]
                    )

            from research_engineer.tools.memory_storage import MemoryStorageInput

            input = MemoryStorageInput(memory=memory, operation="store")
            await self.storage.execute(input)

            if self.config.log_access:
                await self.storage.log_access(memory.memory_id, "write", "MemoryAgent", "store_patch")

            return memory.memory_id

        except Exception as e:
            raise ToolError(f"Failed to store patch: {e}", None, e)

    async def store_decision(self, context: str, decision: str, rationale: str, alternatives: list[str] = None) -> str:
        """Store architecture decision in memory."""
        try:
            memory = ArchitectureDecisionMemory(
                context=context,
                decision=decision,
                rationale=rationale,
                alternatives_considered=alternatives or [],
                tags=self._extract_tags(context, decision),
                confidence_score=0.8
            )

            if self.embedding_strategy:
                embedding = await self.embedding_strategy.embed(f"{context} {decision} {rationale}", MemoryType.ARCHITECTURE_DECISION)
                memory.embedding_key = f"decision:{memory.memory_id}"

                if self.vector_store:
                    await self.vector_store.add(
                        ids=[memory.embedding_key],
                        embeddings=embedding.reshape(1, -1),
                        metadatas=[{"memory_type": MemoryType.ARCHITECTURE_DECISION.value}]
                    )

            from research_engineer.tools.memory_storage import MemoryStorageInput

            input = MemoryStorageInput(memory=memory, operation="store")
            await self.storage.execute(input)

            if self.config.log_access:
                await self.storage.log_access(memory.memory_id, "write", "MemoryAgent", "store_decision")

            return memory.memory_id

        except Exception as e:
            raise ToolError(f"Failed to store decision: {e}", None, e)

    async def store_insight(
        self,
        insight_type: str,
        domain: str,
        description: str,
        evidence: list[str] = None,
        applicability: list[str] = None
    ) -> str:
        """Store research insight in memory."""
        from research_engineer.models.memory import InsightType

        try:
            memory = ResearchInsightMemory(
                insight_type=InsightType(insight_type) if insight_type in [e.value for e in InsightType] else InsightType.PATTERN,
                domain=domain,
                description=description,
                evidence=evidence or [],
                applicability=applicability or [],
                confidence=0.85,
                tags=[domain, insight_type],
                confidence_score=0.85
            )

            if self.embedding_strategy:
                embedding = await self.embedding_strategy.embed(description, MemoryType.RESEARCH_INSIGHT)
                memory.embedding_key = f"insight:{memory.memory_id}"

                if self.vector_store:
                    await self.vector_store.add(
                        ids=[memory.embedding_key],
                        embeddings=embedding.reshape(1, -1),
                        metadatas=[{"memory_type": MemoryType.RESEARCH_INSIGHT.value, "domain": domain}]
                    )

            from research_engineer.tools.memory_storage import MemoryStorageInput

            input = MemoryStorageInput(memory=memory, operation="store")
            await self.storage.execute(input)

            if self.config.log_access:
                await self.storage.log_access(memory.memory_id, "write", "MemoryAgent", "store_insight")

            return memory.memory_id

        except Exception as e:
            raise ToolError(f"Failed to store insight: {e}", None, e)

    async def store_failure(
        self,
        context: str,
        approach: str,
        failure_mode: str,
        lessons: list[str] = None,
        error_details: str = None
    ) -> str:
        """Store failed approach in memory."""
        from research_engineer.models.memory import FailureMode

        try:
            memory = FailedApproachMemory(
                context=context,
                approach_description=approach,
                failure_mode=FailureMode(failure_mode) if failure_mode in [e.value for e in FailureMode] else FailureMode.POOR_PERFORMANCE,
                error_details=error_details,
                lessons_learned=lessons or [],
                tags=self._extract_tags(context, approach),
                confidence_score=1.0
            )

            if self.embedding_strategy:
                embedding = await self.embedding_strategy.embed(f"{context} {approach}", MemoryType.FAILED_APPROACH)
                memory.embedding_key = f"failure:{memory.memory_id}"

                if self.vector_store:
                    await self.vector_store.add(
                        ids=[memory.embedding_key],
                        embeddings=embedding.reshape(1, -1),
                        metadatas=[{"memory_type": MemoryType.FAILED_APPROACH.value}]
                    )

            from research_engineer.tools.memory_storage import MemoryStorageInput

            input = MemoryStorageInput(memory=memory, operation="store")
            await self.storage.execute(input)

            if self.config.log_access:
                await self.storage.log_access(memory.memory_id, "write", "MemoryAgent", "store_failure")

            return memory.memory_id

        except Exception as e:
            raise ToolError(f"Failed to store failure: {e}", None, e)

    async def store_success(
        self,
        context: str,
        approach: str,
        metrics: dict[str, float] = None,
        key_factors: list[str] = None
    ) -> str:
        """Store successful approach in memory."""
        try:
            memory = SuccessfulApproachMemory(
                context=context,
                approach_description=approach,
                success_metrics=metrics or {},
                key_factors=key_factors or [],
                tags=self._extract_tags(context, approach),
                confidence_score=1.0
            )

            if self.embedding_strategy:
                embedding = await self.embedding_strategy.embed(f"{context} {approach}", MemoryType.SUCCESSFUL_APPROACH)
                memory.embedding_key = f"success:{memory.memory_id}"

                if self.vector_store:
                    await self.vector_store.add(
                        ids=[memory.embedding_key],
                        embeddings=embedding.reshape(1, -1),
                        metadatas=[{"memory_type": MemoryType.SUCCESSFUL_APPROACH.value}]
                    )

            from research_engineer.tools.memory_storage import MemoryStorageInput

            input = MemoryStorageInput(memory=memory, operation="store")
            await self.storage.execute(input)

            if self.config.log_access:
                await self.storage.log_access(memory.memory_id, "write", "MemoryAgent", "store_success")

            return memory.memory_id

        except Exception as e:
            raise ToolError(f"Failed to store success: {e}", None, e)

    async def retrieve(self, memory_id: str) -> dict | None:
        """Retrieve specific memory by ID."""
        result = await self.storage.get_memory_by_id(memory_id)

        if result and self.config.log_access:
            await self.storage.log_access(memory_id, "read", "MemoryAgent", "retrieve")

        return result

    async def search(self, query: str, filters: MemoryFilters | None = None, limit: int = 10) -> list[MemoryResult]:
        """Semantic search for memories."""
        semantic_query = SemanticQuery(
            query_text=query,
            memory_types=filters.memory_types if filters else None,
            tags=filters.tags if filters else None,
            min_confidence=filters.min_confidence if filters else 0.0,
            max_results=limit
        )

        results = await self.query_processor.process(
            semantic_query,
            storage=self.storage,
            vector_store=self.vector_store,
            graph=self.graph
        )

        if self.config.log_access and results:
            for result in results:
                await self.storage.log_access(
                    result.memory["memory_id"],
                    "read",
                    "MemoryAgent",
                    f"search: {query}"
                )

        return results

    async def get_context(self, current_task: str, limit: int = 5) -> list[MemoryResult]:
        """Get relevant context for current task."""
        return await self.search(current_task, limit=limit)

    async def get_related(self, memory_id: str, max_depth: int = 2) -> list[MemoryResult]:
        """Get related memories via relationships."""
        relationships = await self.storage.get_relationships(memory_id)

        related_ids = set()
        for rel in relationships:
            if rel["source_memory_id"] == memory_id:
                related_ids.add(rel["target_memory_id"])
            else:
                related_ids.add(rel["source_memory_id"])

        results = []
        for related_id in related_ids:
            memory = await self.retrieve(related_id)
            if memory:
                results.append(
                    MemoryResult(
                        memory=memory["content_json"],
                        score=0.8,
                        match_type="related"
                    )
                )

        return results[:max_depth * 5]

    async def get_stats(self) -> MemoryStats:
        """Get memory storage statistics."""
        return await self.storage.get_stats()

    def _extract_tags(self, *texts: str) -> list[str]:
        """Extract tags from text."""
        combined = " ".join(texts).lower()

        keywords = []
        ml_terms = [
            "attention", "transformer", "diffusion", "gan", "vae",
            "optimization", "regularization", "normalization",
            "pytorch", "tensorflow", "jax",
            "training", "inference", "deployment"
        ]

        for term in ml_terms:
            if term in combined:
                keywords.append(term)

        return keywords[:10]

    async def _detect_paper_relationships(self, memory: PaperMemory):
        """Detect relationships for a paper memory using RelationshipDetector."""
        try:
            from research_engineer.tools.relationship_detector import (
                RelationshipDetectorInput,
            )

            result = await self.relationship_detector.execute(
                RelationshipDetectorInput(source_memory=memory)
            )

            for rel in result.relationships:
                try:
                    await self.storage.store_relationship(rel)
                    if self.graph is not None:
                        self.graph.add_node(rel.source_memory_id)
                        self.graph.add_node(rel.target_memory_id)
                        self.graph.add_relationship(rel)
                except Exception:
                    pass
        except Exception:
            pass

    async def _link_plan_to_paper(self, plan_memory: ExperimentPlanMemory):
        """Link plan to its paper."""
        try:
            relationship = MemoryRelationship(
                source_memory_id=plan_memory.memory_id,
                target_memory_id=plan_memory.paper_id,
                relationship_type=RelationshipType.IMPLEMENTS,
                confidence=0.9
            )

            await self.storage.store_relationship(relationship)

            if self.graph is not None:
                self.graph.add_node(plan_memory.memory_id, "experiment_plan")
                self.graph.add_node(plan_memory.paper_id, "paper")
                self.graph.add_relationship(relationship)

        except Exception:
            pass

    async def get_graph_stats(self) -> dict:
        """Get knowledge graph statistics."""
        if self.graph is None:
            return {"error": "Graph not available (networkx missing)"}
        return self.graph.get_stats().model_dump()

    async def search_hybrid(
        self,
        query: str,
        memory_types: list[MemoryType] | None = None,
        limit: int = 10,
        vector_weight: float = 0.7,
        graph_weight: float = 0.3,
    ) -> list[MemoryResult]:
        """Hybrid search combining vector similarity and graph expansion."""
        if self.graph is None:
            return await self.search(query, limit=limit)

        filters = MemoryFilters(memory_types=memory_types) if memory_types else None
        retrieval_query = RetrievalQuery(
            query_text=query,
            memory_types=memory_types,
            limit=limit,
            vector_weight=vector_weight,
            graph_weight=graph_weight,
        )

        try:
            results = await self.hybrid_strategy.retrieve(
                retrieval_query, self.storage, self.vector_store, self.graph
            )
            if self.config.log_access and results:
                for result in results:
                    memory_id = (
                        result.memory.get("memory_id")
                        if isinstance(result.memory, dict)
                        else None
                    )
                    if memory_id:
                        await self.storage.log_access(
                            memory_id, "read", "MemoryAgent", f"hybrid_search: {query}"
                        )
            return results
        except Exception:
            return await self.search(query, filters=filters, limit=limit)

    async def close(self):
        """Close resources."""
        await self.storage.close()
