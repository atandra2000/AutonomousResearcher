"""Repository agent for orchestrating repository analysis."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from research_engineer.models import (
    ArchitectureOverview,
    FileImportance,
    ImplementationTarget,
    RepositorySummary,
)
from research_engineer.llm import LLMProvider
from research_engineer.tools import (
    ASTAnalysisTool,
    ConfigAnalysisTool,
    DependencyGraphTool,
    DocumentationTool,
    KnowledgeGraphTool,
    RepositoryScannerTool,
    TrainingPipelineTool,
)


class RepositoryAnalysisResult(BaseModel):
    """Result of repository analysis."""

    repository_name: str = Field(..., description="Repository name")
    project_type: str = Field(..., description="Project type classification")
    architecture_summary: str = Field(..., description="Architecture summary")
    important_files: list = Field(default_factory=list, description="File importance rankings")
    generated_files: list = Field(default_factory=list, description="Generated markdown files")
    analysis_time_seconds: float = Field(..., description="Analysis duration")
    output_dir: str = Field(..., description="Output directory path")


class _LLMShim:
    """Minimal duck-typed adapter exposing ``analysis/confidence/entities``.

    Lets the provider-agnostic :class:`LLMResponse` flow through the
    existing ``llm_analysis`` result dict without restructuring callers.
    """

    __slots__ = ("analysis", "confidence", "entities", "recommendations")

    def __init__(
        self,
        analysis: str,
        confidence: float = 0.8,
        entities: list[dict] | None = None,
        recommendations: list[str] | None = None,
    ) -> None:
        self.analysis = analysis
        self.confidence = confidence
        self.entities = entities or []
        self.recommendations = recommendations or []


class RepositoryAgent:
    """
    Main agent for analyzing ML repositories and generating documentation.
    
    This agent orchestrates multiple tools to:
    1. Scan repository structure
    2. Analyze Python AST
    3. Build dependency graphs
    4. Extract training pipelines
    5. Parse configurations
    6. Build knowledge graphs
    7. Generate documentation
    8. Optionally run LLM-powered analysis
    9. Apply rate limiting and caching
    """

    def __init__(
        self,
        scanner: RepositoryScannerTool | None = None,
        ast_analyzer: ASTAnalysisTool | None = None,
        dependency_graph: DependencyGraphTool | None = None,
        training_pipeline: TrainingPipelineTool | None = None,
        config_analyzer: ConfigAnalysisTool | None = None,
        knowledge_graph: KnowledgeGraphTool | None = None,
        documentation: DocumentationTool | None = None,
        enable_caching: bool = True,
        cache_path: str = ".cache/repo_analysis",
        rate_limit_enabled: bool = False,
        llm_enabled: bool = False,
        llm_model: str = "llama3",
        llm: LLMProvider | None = None,
    ):
        # Canonical agent name used for LLM router resolution.
        self.agent_name: str = "RepositoryAgent"
        self.scanner = scanner or RepositoryScannerTool()
        self.ast = ast_analyzer or ASTAnalysisTool()
        self.dependencies = dependency_graph or DependencyGraphTool()
        self.training = training_pipeline or TrainingPipelineTool()
        self.config = config_analyzer or ConfigAnalysisTool()
        self.kg = knowledge_graph or KnowledgeGraphTool()
        self.docs = documentation or DocumentationTool()

        # Phase 2 LLM integration features
        self._enable_caching = enable_caching
        self._cache_path = cache_path
        self._rate_limit_enabled = rate_limit_enabled
        self._llm_enabled = llm_enabled

        # Provider-agnostic LLM layer (Phase 10). ``llm`` takes precedence;
        # otherwise the router resolves a provider for this agent when
        # ``llm_enabled`` is True. Legacy ``llm_model`` is retained only for
        # backwards compatibility with the deprecated LlamaIndex tool.
        from research_engineer.agents._llm_support import resolve_llm
        self.llm_provider = resolve_llm(self.agent_name, llm, llm_enabled=llm_enabled)

        # Initialize the legacy LLM analyzer (kept for backwards compat).
        self.llm: Any = None
        if self._llm_enabled and self.llm_provider is None:
            try:
                from research_engineer.tools.llm_analyzer import LLMAnalysisTool
                self.llm = LLMAnalysisTool(model_name=llm_model)
            except ImportError:
                self.llm = None
                print("⚠️  LLM tools not available. Install: pip install llama-index-llms-ollama")

    async def analyze(self, repo_path: str, output_dir: str = "output", enable_llm: bool | None = None) -> dict:
        """Main entry point for repository analysis."""
        import time
        from datetime import datetime
        start_time = time.time()

        # Override LLM setting if explicitly provided
        if enable_llm is not None:
           llm_enabled = enable_llm
        else:
            llm_enabled = self._llm_enabled

        # Phase 2: Rate limiting (optional)
        rate_limiter = None
        if self._rate_limit_enabled:
            from research_engineer.tools.rate_limiter import (
                RateLimitConfig,
                RateLimiter,
                RateLimitStrategy,
            )
            rate_limiter = RateLimiter(
                RateLimitConfig(
                    strategy=RateLimitStrategy.TOKEN_BUCKET,
                    rate=1.0,
                    capacity=10,
                )
            )

        # Validate input
        path = Path(repo_path)
        if not path.exists():
            raise ValueError(f"Repository path does not exist: {repo_path}")
        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {repo_path}")

        # Phase 2: Apply rate limiting if enabled
        if rate_limiter:
            await rate_limiter.wait()

        # Step 1: Scan repository structure
        from research_engineer.tools.scanner import RepoScanInput
        scan_input = RepoScanInput(
            path=str(repo_path),
            max_depth=3,
            include_patterns=['*.py', '*.yaml', '*.yml', '*.json', '*.toml', '*.md', '*.txt'],
            exclude_patterns=['__pycache__', '.git', '.venv', 'dist', 'build', '.egg-info'],
        )
        scan_result = await self.scanner.execute(scan_input)

        if not scan_result or not hasattr(scan_result, 'files_by_type'):
            print(f"⚠️  Warning: Scan returned no files in {repo_path}")

        # Step 2: Analyze AST for all Python files
        from research_engineer.tools.ast_analyzer import ASTInput
        python_files = scan_result.files_by_type.get('.py', [])
        ast_results = []
        for file_path in python_files:
            try:
                with open(file_path, encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                ast_input = ASTInput(
                    file_path=file_path,
                    content=content,
                    extract_decorators=True,
                    extract_type_hints=True,
                    extract_docstrings=True,
                )
                ast_result = await self.ast.execute(ast_input)
                ast_results.append(ast_result)
            except Exception:
                continue

        # Step 3: Build dependency graph
        from research_engineer.tools.dependency_graph import DependencyInput
        dep_input = DependencyInput(
            files=ast_results,
            project_root=str(repo_path),
        )
        dependency_result = await self.dependencies.execute(dep_input)

        # Step 4: Analyze training pipeline
        from research_engineer.tools.training_pipeline import TrainingPipelineInput, TrainingPipelineOutput
        if ast_results:
            train_input = TrainingPipelineInput(
                ast_outputs=ast_results,
                repo_path=str(repo_path),
            )
            training_result = await self.training.execute(train_input)
        else:
            training_result = TrainingPipelineOutput(
                full_pipeline=[],
                training_loop={},
            )

        # Step 5: Analyze configurations
        from research_engineer.tools.config_analyzer import ConfigInput
        config_paths = (
            scan_result.files_by_type.get('.yaml', []) +
            scan_result.files_by_type.get('.yml', []) +
            scan_result.files_by_type.get('.json', []) +
            scan_result.files_by_type.get('.toml', [])
        )
        config_result = await self.config.execute(
            ConfigInput(
                config_paths=config_paths,
            )
        )

        # Step 6: Build knowledge graph
        from research_engineer.tools.config_analyzer import ConfigOutput
        from research_engineer.tools.knowledge_graph import KnowledgeGraphInput
        from research_engineer.tools.training_pipeline import TrainingPipelineOutput

        training_pipeline = TrainingPipelineOutput(
            full_pipeline=[],
            training_loop={},
        )

        config_analysis = ConfigOutput(
            all_configs={},
            training_hyperparameters={},
            model_hyperparameters={},
            data_paths={},
            distributed_settings={},
            checkpoint_settings={},
            logging_config={},
            optimizer_config={},
            scheduler_config={},
            config_framework='unknown',
            config_sources={},
        )

        kg_input = KnowledgeGraphInput(
            repo_scan=scan_result.__dict__,
            ast_results=ast_results,
            dependencies=dependency_result,
            training_pipeline=training_pipeline,
            config_analysis=config_analysis,
        )
        kg_result = await self.kg.execute(kg_input)

        # Step 7: Detect project type
        repo_type = scan_result.repository_type if hasattr(scan_result, 'repository_type') else 'Unknown'

        # Get python files list
        python_files = scan_result.files_by_type.get('.py', [])
        config_paths = (
            scan_result.files_by_type.get('.yaml', []) +
            scan_result.files_by_type.get('.yml', []) +
            scan_result.files_by_type.get('.json', []) +
            scan_result.files_by_type.get('.toml', [])
        )

        # Step 8: Generate architecture overview
        architecture = ArchitectureOverview(
            high_level_structure=f"{repo_type} project with {len(ast_results)} modules",
            entry_points=scan_result.entry_points,
            main_modules=list(set([str(Path(f).parent) for f in python_files]))[:10],
            core_abstractions=["Model", "Dataset", "Optimizer", "Trainer"],
            critical_classes=["Model", "Trainer"],
            interfaces=["Trainable", "Configurable"],
            package_boundaries=["data/", "model/", "train/", "eval/"],
            mermaid_diagram=self._generate_mermaid_diagram(ast_results, dependency_result),
        )

        # Step 9: Generate important files ranking
        important_files = self._rank_file_importance(python_files, ast_results, dependency_result)

        # Step 10: Generate implementation targets
        implementation_targets = self._generate_implementation_targets(ast_results, training_result)

        # Step 11: Create repository summary
        from research_engineer.models.repo import ConfigurationAnalysis, KnowledgeGraph
        from research_engineer.models.ast_models import ModuleInfo
        
        ast_module_list = []
        for r in ast_results:
            module_name = Path(r.file_path).stem
            ast_module_list.append(ModuleInfo(
                name=module_name,
                path=r.file_path,
                imports=list(r.imports),
                exports=[c.name for c in r.classes] + [f.name for f in r.functions],
                line_count=r.line_count,
                has_tests='test' in r.file_path.lower(),
            ))
        
        summary = RepositorySummary(
            repository_name=path.name,
            project_type=repo_type,
            architecture_summary=architecture.high_level_structure,
            important_files=important_files if important_files else [],
            training_pipeline=str(training_result),
            knowledge_graph=KnowledgeGraph(
                nodes=[n.model_dump() for n in kg_result.nodes],
                edges=[e.model_dump() for e in kg_result.edges],
                communities=kg_result.communities,
                central_nodes=kg_result.central_nodes,
                relationships_by_type=kg_result.relationships_by_type,
            ),
            implementation_targets=implementation_targets if implementation_targets else [],
            configuration_analysis=ConfigurationAnalysis(
                config_files=config_paths,
                training_hyperparameters=config_result.training_hyperparameters,
                model_hyperparameters=config_result.model_hyperparameters,
                data_paths=config_result.data_paths,
                distributed_settings=config_result.distributed_settings,
                checkpoint_settings=config_result.checkpoint_settings,
                config_framework=config_result.config_framework,
            ),
            analysis_timestamp=datetime.now(),
            modules=ast_module_list,
        )

        # Step 12: Generate documentation
        from research_engineer.tools.documentation import DocumentationInput
        output_path = Path(output_dir) / path.name
        output_path.mkdir(parents=True, exist_ok=True)

        doc_input = DocumentationInput(
            repo_summary=summary,
            architecture=architecture.__dict__,
            training_pipeline=training_result,
            dependencies=dependency_result,
            knowledge_graph=kg_result,
            config_analysis=config_result,
            important_files=important_files,
            implementation_targets=implementation_targets,
            output_dir=str(output_path),
        )
        doc_output = await self.docs.execute(doc_input)

        # Step 13: Optional LLM analysis (Phase 2 / Phase 10)
        llm_result: Any = None
        llm_enabled_now = llm_enabled and (
            getattr(self, "llm_provider", None) is not None or
            (hasattr(self, "llm") and self.llm is not None)
        )
        if llm_enabled_now:
            code_snippet = "\n\n".join([
                f"{f.file_path}:\n{f.classes[0].source_code[:500]}"
                for f in ast_results[:5] if f.classes
            ])
            if code_snippet:
                provider = getattr(self, "llm_provider", None)
                if provider is not None:
                    # Provider-agnostic path (Phase 10).
                    from research_engineer.llm import LLMMessage, LLMRequest, LLMRole
                    try:
                        req = LLMRequest(
                            messages=[
                                LLMMessage(
                                    role=LLMRole.SYSTEM,
                                    content=(
                                        "You are a senior ML engineer analysing repository "
                                        "architecture. Summarise purpose, components, and risks."
                                    ),
                                ),
                                LLMMessage(
                                    role=LLMRole.USER,
                                    content=f"Analyse this code's architecture:\n\n{code_snippet}",
                                ),
                            ],
                            temperature=0.2,
                        )
                        resp = await provider.complete(req)
                        llm_result = _LLMShim(
                            analysis=resp.content,
                            confidence=0.8,
                            entities=[],
                            recommendations=[],
                        )
                    except Exception:
                        llm_result = None
                elif hasattr(self, "llm") and self.llm is not None:
                    # Legacy LlamaIndex path (fallback).
                    try:
                        llm_input = type(
                            "LLMAnalysisInput",
                            (),
                            {"code_snippet": code_snippet, "analysis_type": "architecture"}
                        )()
                        llm_result = await self.llm.execute(llm_input)
                    except Exception:
                        llm_result = None

        elapsed = time.time() - start_time

        generated_files = doc_output.generated_files if hasattr(doc_output, 'generated_files') else []

        result = {
            "repository_name": path.name,
            "project_type": repo_type,
            "architecture_summary": architecture.high_level_structure,
            "important_files": [f.model_dump() for f in important_files[:10]],
            "generated_files": generated_files,
            "analysis_time_seconds": round(elapsed, 2),
            "output_dir": str(output_path),
        }

        # Add optional phase 2 features
        if self._rate_limit_enabled and rate_limiter:
            result["rate_limit_stats"] = await rate_limiter.stats()

        if llm_result:
            result["llm_analysis"] = {
                "analysis": llm_result.analysis,
                "confidence": llm_result.confidence,
                "entities": llm_result.entities,
            }

        return result

    def _generate_mermaid_diagram(self, ast_results: list, dependency_result) -> str:
        """Generate Mermaid architecture diagram."""
        mermaid = ["```mermaid", "flowchart TD"]

        # Add file nodes
        for result in ast_results[:5]:
            file_path = result.file_path
            file_name = Path(file_path).name
            module_id = file_name.replace('.', '_').replace('-', '_')
            mermaid.append(f"    {module_id}[{file_name}]")

        # Add connections
        for source_file, targets in dependency_result.file_level_graph.items():
            if targets:
                src_name = Path(source_file).name.replace('.', '_').replace('-', '_')
                for target in targets[:2]:
                    tgt_name = Path(target).name.replace('.', '_').replace('-', '_')
                    mermaid.append(f"    {src_name} --> {tgt_name}")

        mermaid.append("```")
        return '\n'.join(mermaid)

    def _rank_file_importance(self, python_files: list, ast_results: list, dependency_result) -> list[FileImportance]:
        """Rank files by importance."""
        if not python_files:
            return []
            
        importance_scores = {}

        for file_path in python_files:
            score = 0
            reason = []

            if file_path in dependency_result.dependency_tree:
                score += 10
                reason.append("main module")

            if 'model' in file_path.lower():
                score += 5
                reason.append("contains model")

            if 'train' in file_path.lower():
                score += 5
                reason.append("training related")

            try:
                with open(file_path, encoding='utf-8', errors='ignore') as f:
                    lines = len(f.readlines())
                if lines > 500:
                    score += 3
                    reason.append("large file")
                elif lines > 100:
                    score += 1
                    reason.append("moderate size")
            except Exception:
                pass

            importance = "Medium"
            if score >= 15:
                importance = "Critical"
            elif score >= 10:
                importance = "High"
            elif score >= 5:
                importance = "Medium"
            else:
                importance = "Low"

            importance_scores[file_path] = {
                'score': score,
                'importance': importance,
                'reason': ', '.join(reason) if reason else 'Standard file',
            }

        files = []
        for file_path, info in sorted(importance_scores.items(), key=lambda x: x[1]['score'], reverse=True)[:20]:
            files.append(FileImportance(
                file_path=file_path,
                importance=info['importance'],
                reason=info['reason'],
                complexity="Medium",
                lines_of_code=info['score'],
                dependencies_count=1,
            ))

        return files

    def _generate_implementation_targets(self, ast_results: list, training_result) -> list:
        """Generate implementation targets for code modification."""
        targets = []

        for result in ast_results:
            for cls in result.classes:
                target = ImplementationTarget(
                    file_path=result.file_path,
                    class_name=cls.name,
                    target_type="class",
                    insertion_point=f"class {cls.name}",
                    complexity="Medium",
                    estimated_lines=10,
                )
                targets.append(target)

            for func in result.functions:
                target = ImplementationTarget(
                    file_path=result.file_path,
                    method_name=func.name,
                    target_type="function",
                    insertion_point=f"def {func.name}()",
                    complexity="Low",
                    estimated_lines=5,
                )
                targets.append(target)

        return targets[:10]
