"""Tools initialization and exports."""

from .arxiv import ArxivInput, ArxivOutput, ArxivTool
from .ast_analyzer import ASTAnalysisTool, ASTInput, ASTOutput
from .base import Tool, ToolError
from .base_cache import (
    CacheBase,
    CacheEntry,
    FileCache,
    RateLimitConfig,
    SimpleCache,
    SlidingWindowRateLimiter,
    TokenBucketConfig,
    TokenBucketRateLimiter,
    generate_key,
)
from .config_analyzer import ConfigAnalysisTool, ConfigInput, ConfigOutput
from .dependency_graph import DependencyGraphTool, DependencyInput, DependencyOutput
from .documentation import DocumentationInput, DocumentationOutput, DocumentationTool
from .knowledge_graph import (
    KnowledgeGraphInput,
    KnowledgeGraphOutput,
    KnowledgeGraphTool,
)

try:
    from .llm_analyzer import (
        LLMAnalysisInput,
        LLMAnalysisOutput,
        LLMAnalysisTool,
        LLMIntegrationTool,
    )
except ImportError:
    LLMAnalysisTool = None  # type: ignore[misc,assignment]
    LLMAnalysisInput = None  # type: ignore[misc,assignment]
    LLMAnalysisOutput = None  # type: ignore[misc,assignment]
    LLMIntegrationTool = None  # type: ignore[misc,assignment]
# Phase 4 - Coding Agent
# Phase 8 - Evaluation
from . import _stats
from .artifact_collector import ArtifactCollectorTool
from .code_generation import (
    CodeGenerationInput,
    CodeGenerationOutput,
    CodeGenerationTool,
)
from .compatibility import (
    CompatibilityAnalysisTool,
    CompatibilityInput,
    CompatibilityOutput,
)
from .compute_estimator import (
    ComputeEstimatorInput,
    ComputeEstimatorOutput,
    ComputeEstimatorTool,
)
from .embedding_strategy import (
    EmbeddingConfig,
    EmbeddingStrategy,
)
from .evaluation_storage import EvaluationStorageTool
from .experiment_comparison import ExperimentComparisonTool
from .experiment_design import (
    ExperimentDesignInput,
    ExperimentDesignOutput,
    ExperimentDesignTool,
)

# Phase 7 - Experiment Execution
from .experiment_runner import ExperimentRunnerTool
from .experiment_storage import ExperimentStorageTool
from .failure_detector import FailureDetectorTool
from .impact_analysis import (
    ImpactAnalysisInput,
    ImpactAnalysisOutput,
    ImpactAnalysisTool,
)
from .implementation_planner import (
    ImplementationPlannerInput,
    ImplementationPlannerOutput,
    ImplementationPlannerTool,
)
from .implementation_report import (
    ImplementationReportInput,
    ImplementationReportOutput,
    ImplementationReportTool,
)
from .literature_review import LiteratureReviewTool
# Phase 9 - Autonomous Research Loop
from .loop_storage import LoopStorageTool
from .memory_graph import (
    GraphStats,
    MemoryGraphInput,
    MemoryGraphOutput,
    MemoryKnowledgeGraph,
)
from .memory_query import (
    QueryIntent,
    QueryProcessor,
    SemanticQuery,
)

# Phase 5 - Memory
from .memory_storage import (
    MemoryQueryInput,
    MemoryQueryOutput,
    MemoryRelationshipInput,
    MemoryRelationshipOutput,
    MemoryStorageInput,
    MemoryStorageOutput,
    MemoryStorageTool,
)
from .memory_tools import (
    MemoryGraphTool,
    MemoryGraphToolInput,
    MemoryGraphToolOutput,
    MemoryQueryTool,
    MemoryQueryToolInput,
    MemoryQueryToolOutput,
    MemoryRecallTool,
    MemoryRecallToolInput,
    MemoryRecallToolOutput,
    MemoryWriteTool,
    MemoryWriteToolInput,
    MemoryWriteToolOutput,
)
from .metric_collector import MetricCollectorTool
from .migration_planner import (
    MigrationPlannerInput,
    MigrationPlannerOutput,
    MigrationPlannerTool,
)
from .monitoring import MonitoringTool
from .next_experiment import NextExperimentTool
from .paper_comparison import PaperComparisonTool
from .paper_recommendation import PaperRecommendationTool
from .paper_relationship import PaperRelationshipTool

# Phase 6 - Literature Intelligence
from .paper_search import PaperSearchTool
from .parser import PaperParserTool, ParserInput, ParserOutput
from .patch_application import (
    PatchApplicationInput,
    PatchApplicationOutput,
    PatchApplicationTool,
)
from .patch_generation import (
    PatchGenerationInput,
    PatchGenerationOutput,
    PatchGenerationTool,
)
from .pdf import PDFInput, PDFOutput, PDFTool
from .rate_limiter import AsyncRateLimiter, RateLimitStrategy
from .relationship_detector import (
    RelationshipDetector,
    RelationshipDetectorInput,
    RelationshipDetectorOutput,
)
from .relevance_scoring import RelevanceScoringTool
from .report_generator import ReportGeneratorTool
from .result_prediction import (
    ResultPredictionInput,
    ResultPredictionOutput,
    ResultPredictionTool,
)
from .retrieval_strategies import (
    STRATEGY_REGISTRY,
    DirectLookupStrategy,
    GraphTraversalStrategy,
    HybridSearchStrategy,
    RetrievalQuery,
    RetrievalStrategy,
    SemanticSearchStrategy,
    TagBasedFilterStrategy,
    TemporalQueryStrategy,
    get_strategy,
)
from .risk_assessment import (
    RiskAssessmentInput,
    RiskAssessmentOutput,
    RiskAssessmentTool,
)
from .rollback_planner import (
    RollbackPlannerInput,
    RollbackPlannerOutput,
    RollbackPlannerTool,
)
from .scanner import RepoScanInput, RepoScanOutput, RepositoryScannerTool
from .self_review import (
    SelfReviewInput,
    SelfReviewOutput,
    SelfReviewTool,
)
from .statistical_significance import StatisticalSignificanceTool
from .stopping_condition import StoppingConditionChecker
from .storage import SQLiteStorage, StorageInput, StorageTool, StorageToolAlias
from .test_generation import (
    TestGenerationInput,
    TestGenerationOutput,
    TestGenerationTool,
)
from .training_dynamics import TrainingDynamicsTool
from .training_pipeline import (
    TrainingPipelineInput,
    TrainingPipelineOutput,
    TrainingPipelineTool,
)
from .trend_analysis import TrendAnalysisTool
from .validation_planner import (
    ValidationPlannerInput,
    ValidationPlannerOutput,
    ValidationPlannerTool,
)
from .vector_store import (
    ChromaVectorStore,
    VectorSearchResult,
    VectorStore,
    VectorStoreConfig,
)

# Re-export for convenience
__all__ = [
    # Base
    "Tool",
    "ToolError",
    # Caching
    "CacheBase",
    "CacheEntry",
    "SimpleCache",
    "FileCache",
    "TokenBucketConfig",
    "TokenBucketRateLimiter",
    "RateLimitConfig",
    "SlidingWindowRateLimiter",
    "AsyncRateLimiter",
    "RateLimitStrategy",
    "generate_key",
    # Arxiv
    "ArxivTool",
    "ArxivInput",
    "ArxivOutput",
    # PDF
    "PDFTool",
    "PDFInput",
    "PDFOutput",
    # Parser
    "PaperParserTool",
    "ParserInput",
    "ParserOutput",
    # Storage
    "StorageTool",
    "StorageToolAlias",
    "StorageInput",
    "SQLiteStorage",
    # Scanner
    "RepositoryScannerTool",
    "RepoScanInput",
    "RepoScanOutput",
    # AST
    "ASTAnalysisTool",
    "ASTInput",
    "ASTOutput",
    # Dependency
    "DependencyGraphTool",
    "DependencyInput",
    "DependencyOutput",
    # Training Pipeline
    "TrainingPipelineTool",
    "TrainingPipelineInput",
    "TrainingPipelineOutput",
    # Config
    "ConfigAnalysisTool",
    "ConfigInput",
    "ConfigOutput",
    # Knowledge Graph
    "KnowledgeGraphTool",
    "KnowledgeGraphInput",
    "KnowledgeGraphOutput",
    # Documentation
    "DocumentationTool",
    "DocumentationInput",
    "DocumentationOutput",
    # LLM Analyzer
    "LLMAnalysisTool",
    "LLMAnalysisInput",
    "LLMAnalysisOutput",
    "LLMIntegrationTool",
    # Phase 3 - Experiment Planner
    "CompatibilityAnalysisTool",
    "CompatibilityInput",
    "CompatibilityOutput",
    "ImplementationPlannerTool",
    "ImplementationPlannerInput",
    "ImplementationPlannerOutput",
    "ImpactAnalysisTool",
    "ImpactAnalysisInput",
    "ImpactAnalysisOutput",
    "ExperimentDesignTool",
    "ExperimentDesignInput",
    "ExperimentDesignOutput",
    "ValidationPlannerTool",
    "ValidationPlannerInput",
    "ValidationPlannerOutput",
    "RiskAssessmentTool",
    "RiskAssessmentInput",
    "RiskAssessmentOutput",
    "ComputeEstimatorTool",
    "ComputeEstimatorInput",
    "ComputeEstimatorOutput",
    "ResultPredictionTool",
    "ResultPredictionInput",
    "ResultPredictionOutput",
    # Phase 4 - Coding Agent
    "CodeGenerationTool",
    "CodeGenerationInput",
    "CodeGenerationOutput",
    "PatchGenerationTool",
    "PatchGenerationInput",
    "PatchGenerationOutput",
    "PatchApplicationTool",
    "PatchApplicationInput",
    "PatchApplicationOutput",
    "TestGenerationTool",
    "TestGenerationInput",
    "TestGenerationOutput",
    "MigrationPlannerTool",
    "MigrationPlannerInput",
    "MigrationPlannerOutput",
    "RollbackPlannerTool",
    "RollbackPlannerInput",
    "RollbackPlannerOutput",
    "SelfReviewTool",
    "SelfReviewInput",
    "SelfReviewOutput",
    "ImplementationReportTool",
    "ImplementationReportInput",
    "ImplementationReportOutput",
    # Phase 5 - Memory
    "MemoryStorageTool",
    "MemoryStorageInput",
    "MemoryStorageOutput",
    "MemoryRelationshipInput",
    "MemoryRelationshipOutput",
    "MemoryQueryInput",
    "MemoryQueryOutput",
    "VectorStore",
    "VectorStoreConfig",
    "VectorSearchResult",
    "ChromaVectorStore",
    "EmbeddingStrategy",
    "EmbeddingConfig",
    "QueryProcessor",
    "SemanticQuery",
    "QueryIntent",
    # Phase 5 - Memory Graph & Relationships
    "MemoryKnowledgeGraph",
    "GraphStats",
    "MemoryGraphInput",
    "MemoryGraphOutput",
    "RelationshipDetector",
    "RelationshipDetectorInput",
    "RelationshipDetectorOutput",
    # Phase 5 - Retrieval Strategies
    "RetrievalQuery",
    "RetrievalStrategy",
    "DirectLookupStrategy",
    "SemanticSearchStrategy",
    "GraphTraversalStrategy",
    "TagBasedFilterStrategy",
    "TemporalQueryStrategy",
    "HybridSearchStrategy",
    "STRATEGY_REGISTRY",
    "get_strategy",
    # Phase 5 - Formal Memory Tools
    "MemoryQueryTool",
    "MemoryWriteTool",
    "MemoryGraphTool",
    "MemoryRecallTool",
    "MemoryQueryToolInput",
    "MemoryQueryToolOutput",
    "MemoryWriteToolInput",
    "MemoryWriteToolOutput",
    "MemoryGraphToolInput",
    "MemoryGraphToolOutput",
    "MemoryRecallToolInput",
    "MemoryRecallToolOutput",
    # Phase 6 - Literature Intelligence
    "PaperSearchTool",
    "PaperComparisonTool",
    "PaperRelationshipTool",
    "TrendAnalysisTool",
    "LiteratureReviewTool",
    "PaperRecommendationTool",
    "RelevanceScoringTool",
    # Phase 7 - Experiment Execution
    "ExperimentRunnerTool",
    "MonitoringTool",
    "MetricCollectorTool",
    "ArtifactCollectorTool",
    "FailureDetectorTool",
    "ExperimentStorageTool",
    # Phase 8 - Evaluation
    "ExperimentComparisonTool",
    "TrainingDynamicsTool",
    "StatisticalSignificanceTool",
    "NextExperimentTool",
    "EvaluationStorageTool",
    "_stats",
    # Phase 9 - Autonomous Research Loop
    "LoopStorageTool",
    "StoppingConditionChecker",
    "ReportGeneratorTool",
]
