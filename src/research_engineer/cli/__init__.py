"""CLI interface for research engine."""

import asyncio
import json
from pathlib import Path

import typer

from research_engineer.agents import (
    CodingAgent,
    EvaluationAgent,
    ExperimentAgent,
    ExperimentPlannerAgent,
    LiteratureAgent,
    MemoryAgent,
    RepositoryAgent,
    ResearchAgent,
    ResearchLoopAgent,
    TaskAgent,
)
from research_engineer.models.experiment import ExperimentStatus, ExperimentType
from research_engineer.models.memory import MemoryFilters, MemoryType
from research_engineer.models.task import TaskConfig, TaskStatus
from research_engineer.tools.storage import StorageTool

app = typer.Typer(
    name="research-engineer",
    help="Autonomous ML Research Engineer - Analyze papers and generate implementation plans"
)

# Memory sub-application
memory_app = typer.Typer(
    name="memory",
    help="Search, list, export, import, and archive research memories"
)
app.add_typer(memory_app, name="memory")

# Literature sub-application (Phase 6)
literature_app = typer.Typer(
    name="literature",
    help="Literature intelligence: search, compare, review, trends, recommendations"
)
app.add_typer(literature_app, name="literature")

# Experiment sub-application (Phase 7)
experiment_app = typer.Typer(
    name="experiment",
    help="Run, monitor, and query ML experiments"
)
app.add_typer(experiment_app, name="experiment")

# Evaluation sub-application (Phase 8)
evaluate_app = typer.Typer(
    name="evaluate",
    help="Evaluate experiments and generate research conclusions"
)
app.add_typer(evaluate_app, name="evaluate")

# Loop sub-application (Phase 9)
loop_app = typer.Typer(
    name="loop",
    help="Autonomous research loop orchestration"
)
app.add_typer(loop_app, name="loop")

# LLM sub-application (Phase 10)
llm_app = typer.Typer(
    name="llm",
    help="Inspect and configure the provider-agnostic LLM layer"
)
app.add_typer(llm_app, name="llm")

# Global instances
# Global instances
_agent: ResearchAgent | None = None
_agent_repo: RepositoryAgent | None = None
_storage: StorageTool | None = None
_planner_agent: ExperimentPlannerAgent | None = None
_coding_agent: CodingAgent | None = None
_memory_agent: MemoryAgent | None = None
_literature_agent: LiteratureAgent | None = None
_experiment_agent: ExperimentAgent | None = None
_evaluation_agent: EvaluationAgent | None = None
_loop_agent: ResearchLoopAgent | None = None
_task_agent: TaskAgent | None = None


def _get_agent() -> ResearchAgent:
    """Get or create agent instance."""
    global _agent
    if _agent is None:
        _agent = ResearchAgent()
    return _agent


def _get_repo_agent() -> RepositoryAgent:
    """Get or create repository agent instance with CLI options."""
    global _agent_repo
    if _agent_repo is None:
        _agent_repo = RepositoryAgent(
            enable_caching=False,
            rate_limit_enabled=False,
            llm_enabled=False,
        )
    return _agent_repo


def _get_storage() -> StorageTool:
    """Get or create storage instance."""
    global _storage
    if _storage is None:
        _storage = StorageTool()
    return _storage


def _get_planner_agent() -> ExperimentPlannerAgent:
    """Get or create planner agent instance."""
    global _planner_agent
    if _planner_agent is None:
        _planner_agent = ExperimentPlannerAgent()
    return _planner_agent


def _get_memory_agent() -> MemoryAgent:
    """Get or create memory agent instance."""
    global _memory_agent
    if _memory_agent is None:
        _memory_agent = MemoryAgent()
    return _memory_agent


def _get_literature_agent() -> LiteratureAgent:
    """Get or create literature agent instance."""
    global _literature_agent
    if _literature_agent is None:
        _literature_agent = LiteratureAgent(memory_agent=_get_memory_agent())
    return _literature_agent


def _get_experiment_agent() -> ExperimentAgent:
    """Get or create experiment agent instance."""
    global _experiment_agent
    if _experiment_agent is None:
        _experiment_agent = ExperimentAgent(memory_agent=_get_memory_agent())
    return _experiment_agent


def _get_evaluation_agent() -> EvaluationAgent:
    """Get or create evaluation agent instance."""
    global _evaluation_agent
    if _evaluation_agent is None:
        _evaluation_agent = EvaluationAgent(
            memory_agent=_get_memory_agent(),
            literature_agent=_get_literature_agent(),
        )
    return _evaluation_agent


def _get_loop_agent() -> ResearchLoopAgent:
    """Get or create research loop agent instance."""
    global _loop_agent
    if _loop_agent is None:
        _loop_agent = ResearchLoopAgent(
            memory_agent=_get_memory_agent(),
            literature_agent=_get_literature_agent(),
            experiment_agent=_get_experiment_agent(),
            evaluation_agent=_get_evaluation_agent(),
        )
    return _loop_agent


def _get_task_agent() -> TaskAgent:
    """Get or create the terminal-first task agent instance."""
    global _task_agent
    if _task_agent is None:
        _task_agent = TaskAgent(
            repository_agent=_get_repo_agent(),
            coding_agent=_get_coding_agent(),
        )
    return _task_agent


def _dump_json_safe(obj: object) -> str:
    """Serialize a Pydantic model to JSON with default=str (safe for datetime)."""
    import json as _json

    if hasattr(obj, "model_dump"):
        return _json.dumps(obj.model_dump(), indent=2, default=str)  # type: ignore[attr-defined]
    return _json.dumps(obj, indent=2, default=str)


@app.command()
def analyze(
    paper: str = typer.Argument(
        ...,
        help="arXiv ID (e.g., 2503.12345), arXiv URL, or PDF file path"
    ),
    output_format: str = typer.Option(
        "json",
        help="Output format: json, markdown, or console"
    ),
    output_dir: str = typer.Option(
        "output",
        help="Directory to save output files"
    )
):
    """
    Analyze an ML paper and generate implementation plan.

    Examples:
        research-engineer analyze 2503.12345
        research-engineer analyze https://arxiv.org/abs/2503.12345
        research-engineer analyze paper.pdf
    """
    agent = _get_agent()

    try:
        result = asyncio.run(agent.analyze(paper, output_dir=output_dir))

        # Console output
        if output_format == "console":
            typer.echo(f"\n✅ Analysis complete for paper: {result['title']}")
            typer.echo(f"📝 Storage record ID: {result['storage_record_id']}")
            typer.echo(f"⏱️  Analysis time: {result['analysis_time_seconds']}s")
            typer.echo(f"📁 Output saved to: {result['output_dir']}")

            # Brief summary
            summary = result['summary']
            typer.echo("\n🎯 Key contributions:")
            for i, contrib in enumerate(summary['core_contributions'][:3], 1):
                typer.echo(f"  {i}. {contrib}")

            typer.echo("\n📈 Key results:")
            for i, result_item in enumerate(summary['key_results'][:3], 1):
                typer.echo(f"  {i}. {result_item}")

        # JSON output
        elif output_format == "json":
            typer.echo(json.dumps(result, indent=2))

        # Markdown output
        elif output_format == "markdown":
            md = f"# Analysis: {result['title']}\n\n"
            md += f"**Paper ID**: {result['paper_id']}\n\n"
            md += f"**Storage Record ID**: {result['storage_record_id']}\n\n"
            md += f"**Analysis Time**: {result['analysis_time_seconds']}s\n\n"
            md += "## Executive Summary\n\n"
            md += result["summary"]["executive_summary"] + "\n\n"
            md += "## Core Contributions\n\n"
            for i, contrib in enumerate(result["summary"]["core_contributions"], 1):
                md += f"{i}. {contrib}\n"
            md += "\n## Implementation Plan\n\n"
            md += result["plan"]["step_by_step_implementation"]

            typer.echo(md)

        return 0

    except Exception as e:
        typer.echo(f"❌ Error analyzing paper: {e}", err=True)
        return 1


@app.command()
def history(
    limit: int = typer.Option(10, help="Number of papers to show"),
):
    """Show analysis history."""
    storage = _get_storage()

    try:
        papers = asyncio.run(storage.list_papers(limit=limit))

        if not papers:
            typer.echo("No papers in history.")
            return

        typer.echo("\n📊 Analysis History:")
        typer.echo("─" * 80)

        for paper in papers:
            typer.echo(f"📄 ID: {paper['paper_id']}")
            typer.echo(f"   Title: {paper['title'][:80]}...")
            typer.echo(f"   Stored: {paper['created_at']}")
            typer.echo("─" * 80)

    except Exception as e:
        typer.echo(f"❌ Error retrieving history: {e}", err=True)


@app.command()
def cache_status(
    output_format: str = typer.Option(
        "console",
        help="Output format: console, json"
    ),
    cache_path: str = typer.Option(
        ".cache/repo_analysis",
        help="Cache directory path"
    )
):
    """Show cache status and statistics."""
    # Check cache existence
    cache_dir = Path(cache_path)
    cache_exists = cache_dir.exists()

    if not cache_exists:
        typer.echo("Cache directory does not exist yet.")
        return

    # Count cache entries
    cache_files = list(cache_dir.glob("**/*.json"))

    # Calculate cache size
    total_size = sum(f.stat().st_size for f in cache_files)

    result = {
        "cache_path": str(cache_dir),
        "exists": cache_exists,
        "entry_count": len(cache_files),
        "total_size_bytes": total_size,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "last_modified": max(f.stat().st_mtime for f in cache_files) if cache_files else None,
    }

    if output_format == "json":
        typer.echo(json.dumps(result, indent=2))
    else:
        typer.echo("-cache Status-")
        typer.echo(f"Path: {result['cache_path']}")
        typer.echo(f"Entries: {result['entry_count']}")
        typer.echo(f"Size: {result['total_size_mb']} MB")
        if result['last_modified']:
            from datetime import datetime
            dt = datetime.fromtimestamp(result['last_modified'])
            typer.echo(f"Last Modified: {dt}")


@app.command()
def get(
    paper_id: str = typer.Argument(..., help="Paper ID to retrieve"),
    output_format: str = typer.Option("json", help="Output format"),
):
    """Retrieve a previously analyzed paper."""
    storage = _get_storage()

    try:
        paper = asyncio.run(storage.get_paper(paper_id))

        if not paper:
            typer.echo(f"❌ Paper {paper_id} not found in storage.", err=True)
            return 1

        if output_format == "json":
            typer.echo(json.dumps(paper, indent=2))
        elif output_format == "markdown":
            typer.echo(f"# {paper['title']}\n\n{paper['summary_json']}\n\n{paper['plan_json']}")

        return 0

    except Exception as e:
        typer.echo(f"❌ Error retrieving paper: {e}", err=True)
        return 1


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query (title or author)"),
):
    """Search for papers in storage."""
    storage = _get_storage()

    try:
        papers = asyncio.run(storage.search_papers(query))

        if not papers:
            typer.echo(f"No papers found matching '{query}'.")
            return

        typer.echo(f"\n🔍 Found {len(papers)} papers matching '{query}':")
        typer.echo("─" * 80)

        for paper in papers:
            typer.echo(f"📄 ID: {paper['paper_id']}")
            typer.echo(f"   Title: {paper['title']}")
            typer.echo(f"   Stored: {paper['created_at']}")
            typer.echo("─" * 80)

    except Exception as e:
        typer.echo(f"❌ Error searching papers: {e}", err=True)


@app.command()
def analyze_repo(
    repo_path: str = typer.Argument(
        ...,
        help="Path to repository (absolute or relative path, or '.')"
    ),
    output_dir: str = typer.Option(
        "output",
        help="Directory to save output files"
    ),
    output_format: str = typer.Option(
        "markdown",
        help="Output format: markdown (default), json, console"
    ),
    enable_caching: bool = typer.Option(
        False,
        help="Enable caching for analysis results"
    ),
    rate_limit: bool = typer.Option(
        False,
        help="Enable rate limiting for external tool calls"
    ),
    llm_enabled: bool = typer.Option(
        False,
        help="Enable LLM-powered analysis"
    ),
    llm_model: str = typer.Option(
        "llama3",
        help="LLM model name for analysis"
    ),
    cache_path: str = typer.Option(
        ".cache/repo_analysis",
        help="Cache directory path"
    ),
):
    """
    Analyze ML repository and generate documentation.

    Examples:
        research-engineer analyze-repo .
        research-engineer analyze-repo ./DeepSeek
        research-engineer analyze-repo ~/projects/llm
    """
    repo_agent = _get_repo_agent()

    try:
        result = asyncio.run(repo_agent.analyze(
            repo_path,
            output_dir=output_dir,
            enable_llm=llm_enabled,
        ))

        # Console output
        if output_format == "console":
            typer.echo(f"\n✅ Repository analysis complete: {result['repository_name']}")
            typer.echo(f"📂 Project Type: {result['project_type']}")
            typer.echo(f"⏱️  Analysis time: {result['analysis_time_seconds']}s")
            typer.echo(f"📁 Output saved to: {result['output_dir']}")

            typer.echo("\n📊 Key findings:")
            typer.echo(f"   Architecture: {result['architecture_summary'][:100]}...")

            typer.echo(f"\n📝 Important files analyzed: {len(result['important_files'])}")
            typer.echo(f"   Generated files: {len(result['generated_files'])}")

        # JSON output
        elif output_format == "json":
            typer.echo(json.dumps(result, indent=2))

        # Markdown output
        elif output_format == "markdown":
            md = f"# Repository Analysis: {result['repository_name']}\n\n"
            md += f"**Project Type**: {result['project_type']}\n\n"
            md += f"**Architecture Summary**: {result['architecture_summary']}\n\n"
            md += f"**Analysis Time**: {result['analysis_time_seconds']}s\n\n"
            md += "## Generated Documentation\n\n"
            for f in result['generated_files']:
                md += f"- {f}\n"

            typer.echo(md)

        return 0

    except Exception as e:
        typer.echo(f"❌ Error analyzing repository: {e}", err=True)
        return 1


@app.command()
def plan(
    paper: str = typer.Argument(
        ...,
        help="arXiv ID (e.g., 2503.12345), arXiv URL, or PDF file path"
    ),
    repo: str = typer.Argument(
        ...,
        help="Path to target repository"
    ),
    output_dir: str = typer.Option(
        "output",
        help="Directory to save output files"
    ),
    output_format: str = typer.Option(
        "markdown",
        help="Output format: json, markdown, or console"
    ),
):
    """
    Plan experiment integration of a paper into a repository.

    Generates compatibility analysis, implementation plan,
    experiment matrix, validation strategy, risk assessment,
    cost estimation, and expected results.

    Examples:
        research-engineer plan 2503.12345 ./gpt
        research-engineer plan flashattention.pdf ./DeepSeek
        research-engineer plan https://arxiv.org/abs/2503.12345 ./training_framework
    """
    planner = _get_planner_agent()

    try:
        result = asyncio.run(planner.plan(paper, repo, output_dir=output_dir))

        if output_format == "json":
            typer.echo(json.dumps(result.model_dump(), indent=2, default=str))
        elif output_format == "markdown":
            typer.echo(result.engineering_report_md)
        elif output_format == "console":
            typer.echo(f"\n✅ Planning complete for paper: {result.paper_id}")
            typer.echo(f"📂 Repository: {result.repo_path}")
            typer.echo(f"⏱️  Planning time: {result.planning_time_seconds}s")
            typer.echo(f"📁 Output saved to: {output_dir}/plans/")
            typer.echo("\n📄 Generated files:")
            for f in result.generated_files:
                typer.echo(f"   - {f}")
            comp = result.compatibility_report
            typer.echo(f"\n🔗 Compatibility: {comp.get('overall_compatibility', 'Unknown')}")
            risk = result.risk_assessment
            typer.echo(f"⚠️  Risk Level: {risk.get('overall_risk_level', 'Unknown')}")
            compute = result.compute_estimate
            typer.echo(f"💰 Est. Cloud Cost: ${compute.get('approximate_cloud_cost_usd', 0):.2f}")
            typer.echo(f"🖥️  Est. GPU Hours: {compute.get('total_gpu_hours', 0)}")

        return 0

    except Exception as e:
        typer.echo(f"❌ Error planning experiment: {e}", err=True)
        return 1


def _get_coding_agent() -> CodingAgent:
    """Get or create coding agent instance."""
    global _coding_agent
    if _coding_agent is None:
        _coding_agent = CodingAgent()
    return _coding_agent


@app.command()
def implement(
    paper: str | None = typer.Option(
        None,
        "--paper",
        help="arXiv ID, URL, or PDF file path for context"
    ),
    repo: str = typer.Argument(
        ".",
        help="Path to target repository"
    ),
    task: str | None = typer.Option(
        None,
        "--task",
        help="Task description (what to implement)"
    ),
    plan: str | None = typer.Option(
        None,
        "--plan",
        help="Path to implementation plan file"
    ),
    output_dir: str = typer.Option(
        "output",
        help="Directory to save output files"
    ),
    output_format: str = typer.Option(
        "markdown",
        help="Output format: json, markdown, or console"
    ),
    dry_run: bool = typer.Option(
        True,
        help="If True, only generate patches without applying"
    ),
):
    """
    Implement code changes based on plans or tasks.

    Generates patches, tests, and implementation reports.
    Does NOT directly modify code by default (patch-first philosophy).

    Examples:
        research-engineer implement --paper mla.pdf --repo ./repo
        research-engineer implement --task "Add Grouped Query Attention"
        research-engineer implement --plan implementation_plan.md
    """
    coding_agent = _get_coding_agent()

    try:
        # Validate inputs
        if not task and not plan and not paper:
            typer.echo("❌ Error: Must provide --task, --plan, or --paper", err=True)
            return 1

        # Load implementation plan if provided
        implementation_plan = None
        if plan:
            plan_path = Path(plan)
            if not plan_path.exists():
                typer.echo(f"❌ Error: Plan file not found: {plan}", err=True)
                return 1
            typer.echo(f"⚠️  Loading plan from: {plan}")

        # Run implementation
        result = asyncio.run(coding_agent.implement(
            task_description=task or f"Implement {paper or 'plan'}",
            repo_path=repo,
            paper_input=paper,
            implementation_plan=implementation_plan,
            output_dir=output_dir,
        ))

        # Console output
        if output_format == "console":
            typer.echo(f"\n✅ Implementation complete: {result.implementation_id}")
            typer.echo(f"📂 Repository: {result.repo_path}")
            typer.echo(f"📝 Task: {result.task_description}")
            typer.echo(f"📊 Patches generated: {result.patches_generated}")
            typer.echo(f"🧪 Tests generated: {result.tests_generated}")
            typer.echo(f"🔍 Review status: {result.review_status}")
            typer.echo(f"⏱️  Implementation time: {result.implementation_time_seconds}s")
            typer.echo(f"📁 Output saved to: {result.output_dir}/")
            typer.echo("\n📄 Generated files:")
            for f in result.generated_files:
                typer.echo(f"   - {f}")

        # JSON output
        elif output_format == "json":
            typer.echo(result.model_dump_json(indent=2))

        # Markdown output
        elif output_format == "markdown":
            typer.echo(f"# Implementation: {result.implementation_id}\n\n")
            typer.echo(f"**Task**: {result.task_description}\n\n")
            typer.echo(f"**Repository**: {result.repo_path}\n\n")
            typer.echo(f"**Status**: {result.status}\n\n")
            typer.echo(f"**Patches Generated**: {result.patches_generated}\n\n")
            typer.echo(f"**Tests Generated**: {result.tests_generated}\n\n")
            typer.echo(f"**Review Status**: {result.review_status}\n\n")
            typer.echo(f"**Implementation Time**: {result.implementation_time_seconds}s\n\n")
            typer.echo("## Generated Files\n\n")
            for f in result.generated_files:
                typer.echo(f"- {f}\n")

        return 0

    except Exception as e:
        typer.echo(f"❌ Error implementing: {e}", err=True)
        return 1


@memory_app.command("search")
def memory_search(
    query: str = typer.Argument(..., help="Search query"),
    memory_type: str | None = typer.Option(
        None,
        "--type",
        help="Filter by memory type (paper, repository, plan, patch, decision, insight, failure, success)",
    ),
    limit: int = typer.Option(10, "--limit", help="Maximum results to return"),
    strategy: str = typer.Option(
        "semantic_search",
        "--strategy",
        help="Retrieval strategy: semantic_search, direct_lookup, graph_traversal, tag_filter, temporal_query, hybrid_search",
    ),
    output_format: str = typer.Option("console", "--format", help="Output format: console, json"),
):
    """Search research memories using pluggable retrieval strategies.

    Examples:
        research-engineer memory search "attention optimization"
        research-engineer memory search "transformer" --type paper --limit 20
        research-engineer memory search "attention" --strategy hybrid_search
    """
    memory_agent = _get_memory_agent()

    filters = None
    if memory_type:
        try:
            mem_type = MemoryType(memory_type)
            filters = MemoryFilters(memory_types=[mem_type])
        except ValueError:
            typer.echo(f"❌ Invalid memory type: {memory_type}", err=True)
            raise typer.Exit(code=1)

    try:
        if strategy == "hybrid_search":
            mem_types = [MemoryType(memory_type)] if memory_type else None
            results = asyncio.run(
                memory_agent.search_hybrid(query, memory_types=mem_types, limit=limit)
            )
        else:
            results = asyncio.run(memory_agent.search(query, filters=filters, limit=limit))

        if output_format == "json":
            typer.echo(json.dumps([r.model_dump() for r in results], indent=2))
        else:
            typer.echo(f"\n🔍 Search results for '{query}' (strategy: {strategy}):\n")
            for i, result in enumerate(results, 1):
                typer.echo(f"{i}. [{result.match_type}] Score: {result.score:.2f}")
                mem_data = result.memory
                if isinstance(mem_data, dict):
                    if "title" in mem_data:
                        typer.echo(f"   Title: {mem_data['title']}")
                    elif "description" in mem_data:
                        typer.echo(f"   Description: {str(mem_data['description'])[:100]}...")
                typer.echo(f"   Type: {mem_data.get('memory_type', 'unknown') if isinstance(mem_data, dict) else 'unknown'}")
                typer.echo(f"   ID: {mem_data.get('memory_id', 'unknown') if isinstance(mem_data, dict) else 'unknown'}\n")

        return 0
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        return 1


@memory_app.command("list")
def memory_list(
    memory_type: str | None = typer.Option(
        None, "--type", help="Filter by memory type"
    ),
    limit: int = typer.Option(20, "--limit", help="Maximum results to return"),
    offset: int = typer.Option(0, "--offset", help="Pagination offset"),
    include_archived: bool = typer.Option(False, "--include-archived", help="Include archived"),
    output_format: str = typer.Option("console", "--format", help="Output format: console, json"),
):
    """List memories with optional type filter and pagination.

    Examples:
        research-engineer memory list
        research-engineer memory list --type paper --limit 50
        research-engineer memory list --include-archived --format json
    """
    from research_engineer.models.memory import MemoryFilters as MemFilters
    from research_engineer.tools.memory_storage import (
        MemoryQueryInput,
        MemoryStorageTool,
    )

    storage = MemoryStorageTool()
    filters = MemFilters(exclude_archived=not include_archived)
    if memory_type:
        try:
            mem_type = MemoryType(memory_type)
            filters = MemFilters(memory_types=[mem_type], exclude_archived=not include_archived)
        except ValueError:
            typer.echo(f"❌ Invalid memory type: {memory_type}", err=True)
            raise typer.Exit(code=1)

    try:
        output = asyncio.run(storage.execute(MemoryQueryInput(filters=filters, limit=limit, offset=offset)))

        if output_format == "json":
            typer.echo(json.dumps(output.memories, indent=2, default=str))
        else:
            typer.echo(f"\n📋 Memories (total: {output.total}, showing: {len(output.memories)})\n")
            for i, mem in enumerate(output.memories, 1):
                mem_id = mem.get("memory_id", "unknown")
                mtype = mem.get("memory_type", "unknown")
                content = mem.get("content_json", {})
                title = content.get("title") or content.get("repo_name") or content.get("plan_id") or content.get("description", "")[:60]
                archived = " [archived]" if mem.get("is_archived") else ""
                typer.echo(f"{i}. {mtype}{archived} | {mem_id} | {title}")
            if not output.memories:
                typer.echo("No memories found.")

        return 0
    except Exception as e:
        typer.echo(f"❌ Error listing memories: {e}", err=True)
        return 1


@memory_app.command("stats")
def memory_stats(
    output_format: str = typer.Option("console", "--format", help="Output format: console, json"),
):
    """Show memory storage statistics."""
    memory_agent = _get_memory_agent()
    try:
        result = asyncio.run(memory_agent.get_stats())
        if output_format == "json":
            typer.echo(result.model_dump_json(indent=2))
        else:
            typer.echo("\n📊 Memory Statistics\n")
            typer.echo(f"Total memories: {result.total_memories}")
            typer.echo(f"Total relationships: {result.total_relationships}")
            typer.echo(f"Archived: {result.archived_count}")
            typer.echo(f"Average confidence: {result.avg_confidence:.2f}")
            typer.echo(f"Storage size: {result.storage_size_mb:.2f} MB\n")
            typer.echo("Memories by type:")
            for mem_type, count in result.memories_by_type.items():
                typer.echo(f"  {mem_type}: {count}")
        return 0
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        return 1


@memory_app.command("related")
def memory_related(
    memory_id: str = typer.Argument(..., help="Memory ID to find related memories for"),
    max_depth: int = typer.Option(2, "--depth", help="Max graph traversal depth"),
    output_format: str = typer.Option("console", "--format", help="Output format: console, json"),
):
    """Show memories related to a given memory ID via graph traversal.

    Examples:
        research-engineer memory related mem_123
        research-engineer memory related mem_123 --depth 3
    """
    memory_agent = _get_memory_agent()
    try:
        results = asyncio.run(memory_agent.get_related(memory_id, max_depth=max_depth))
        if output_format == "json":
            typer.echo(json.dumps([r.model_dump() for r in results], indent=2))
        else:
            typer.echo(f"\n📚 Related memories for {memory_id}:\n")
            for i, result in enumerate(results, 1):
                typer.echo(f"{i}. [{result.match_type}] Score: {result.score:.2f}")
                typer.echo(f"   Type: {result.memory.get('memory_type', 'unknown') if isinstance(result.memory, dict) else 'unknown'}")
                typer.echo(f"   ID: {result.memory.get('memory_id', 'unknown') if isinstance(result.memory, dict) else 'unknown'}\n")
            if not results:
                typer.echo("No related memories found.")
        return 0
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        return 1


@memory_app.command("graph")
def memory_graph(
    operation: str = typer.Option("stats", "--op", help="Operation: stats, traverse, neighbors"),
    memory_id: str | None = typer.Option(None, "--memory-id", help="Memory ID for neighbors/traverse"),
    max_depth: int = typer.Option(2, "--depth", help="Max traversal depth"),
    output_format: str = typer.Option("console", "--format", help="Output format: console, json"),
):
    """Query the memory knowledge graph.

    Examples:
        research-engineer memory graph --op stats
        research-engineer memory graph --op neighbors --memory-id mem_123
        research-engineer memory graph --op traverse --memory-id mem_123 --depth 3
    """
    memory_agent = _get_memory_agent()

    if operation != "stats" and not memory_id:
        typer.echo("❌ --memory-id required for neighbors/traverse operations", err=True)
        raise typer.Exit(code=1)

    try:
        if operation == "stats":
            stats = asyncio.run(memory_agent.get_graph_stats())
            if output_format == "json":
                typer.echo(json.dumps(stats, indent=2, default=str))
            else:
                typer.echo("\n🕸️  Memory Graph Statistics\n")
                typer.echo(f"Nodes: {stats.get('node_count', 0)}")
                typer.echo(f"Edges: {stats.get('edge_count', 0)}")
                typer.echo(f"Density: {stats.get('density', 0)}")
                typer.echo(f"Connected components: {stats.get('connected_components', 0)}")
                typer.echo(f"Most central: {', '.join(stats.get('most_central', [])[:5])}")
                typer.echo("\nRelationship counts:")
                for rtype, count in stats.get("relationship_counts", {}).items():
                    typer.echo(f"  {rtype}: {count}")
            return 0

        if operation == "neighbors" and memory_agent.graph is not None:
            neighbors = memory_agent.graph.get_neighbors(memory_id)
            if output_format == "json":
                typer.echo(json.dumps({"neighbors": neighbors}))
            else:
                typer.echo(f"\nNeighbors of {memory_id}: {', '.join(neighbors) if neighbors else 'none'}")
            return 0

        if operation == "traverse" and memory_agent.graph is not None:
            nodes = memory_agent.graph.traverse(memory_id, max_depth=max_depth)
            if output_format == "json":
                typer.echo(json.dumps({"reachable": nodes}))
            else:
                typer.echo(f"\nReachable from {memory_id} (depth {max_depth}): {', '.join(nodes) if nodes else 'none'}")
            return 0

        typer.echo(f"❌ Unknown operation or graph unavailable: {operation}", err=True)
        return 1
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        return 1


@memory_app.command("export")
def memory_export(
    output: str = typer.Option("memories_export.json", "--output", help="Output file path"),
    memory_type: str | None = typer.Option(None, "--type", help="Filter by memory type"),
    limit: int = typer.Option(1000, "--limit", help="Max memories to export"),
    include_relationships: bool = typer.Option(True, "--include-relationships", help="Include relationships"),
):
    """Export memories (and optionally relationships) to a JSON file.

    Examples:
        research-engineer memory export --output memories.json
        research-engineer memory export --type paper --output papers.json
    """
    from research_engineer.models.memory import MemoryFilters as MemFilters
    from research_engineer.tools.memory_storage import (
        MemoryQueryInput,
        MemoryStorageTool,
    )

    storage = MemoryStorageTool()
    filters = MemFilters()
    if memory_type:
        try:
            filters = MemFilters(memory_types=[MemoryType(memory_type)])
        except ValueError:
            typer.echo(f"❌ Invalid memory type: {memory_type}", err=True)
            raise typer.Exit(code=1)

    try:
        out_obj = asyncio.run(storage.execute(MemoryQueryInput(filters=filters, limit=limit)))
        memories = out_obj.memories

        rels: list[dict] = []
        if include_relationships:
            for mem in memories:
                mid = mem.get("memory_id")
                if mid:
                    try:
                        mid_rels = asyncio.run(storage.get_relationships(mid))
                        rels.extend(mid_rels)
                    except Exception:
                        pass

        export_data = {
            "memories": memories,
            "relationships": rels,
            "exported_at": str(__import__("datetime").datetime.now().isoformat()),
            "count": len(memories),
        }

        Path(output).write_text(json.dumps(export_data, indent=2, default=str))
        typer.echo(f"✅ Exported {len(memories)} memories and {len(rels)} relationships to {output}")
        return 0
    except Exception as e:
        typer.echo(f"❌ Error exporting: {e}", err=True)
        return 1


def _import_memory_row(storage, mem: dict, skip_existing: bool) -> tuple[bool, bool]:
    """Import a single memory row. Returns (imported, skipped)."""
    import sqlite3

    mem_id = mem.get("memory_id")
    if not mem_id:
        return False, False
    if skip_existing and asyncio.run(storage.get_memory_by_id(mem_id)):
        return False, True

    try:
        conn = sqlite3.connect(storage.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO memories (memory_id, memory_type, content_json, embedding_key, tags, confidence_score, accessed_count, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                mem_id,
                mem.get("memory_type", "paper"),
                json.dumps(mem.get("content_json", {}), default=str),
                mem.get("embedding_key"),
                json.dumps(mem.get("tags", [])),
                mem.get("confidence_score", 1.0),
                mem.get("accessed_count", 0),
                mem.get("created_at"),
            ),
        )
        conn.commit()
        conn.close()
        return True, False
    except Exception:
        return False, False


def _import_relationship_row(storage, rel: dict) -> bool:
    """Import a single relationship row. Returns True if imported."""
    try:
        from research_engineer.models.memory import MemoryRelationship, RelationshipType

        rtype_str = rel.get("relationship_type", "similar_to")
        try:
            rtype = RelationshipType(rtype_str)
        except ValueError:
            rtype = RelationshipType.SIMILAR_TO
        rel_obj = MemoryRelationship(
            source_memory_id=rel.get("source_memory_id", ""),
            target_memory_id=rel.get("target_memory_id", ""),
            relationship_type=rtype,
            confidence=rel.get("confidence", 1.0),
        )
        asyncio.run(storage.store_relationship(rel_obj))
        return True
    except Exception:
        return False


@memory_app.command("import")
def memory_import(
    file: str = typer.Argument(..., help="JSON file to import"),
    skip_existing: bool = typer.Option(True, "--skip-existing", help="Skip if memory already exists"),
):
    """Import memories and relationships from a JSON export file.

    Examples:
        research-engineer memory import memories_export.json
        research-engineer memory import memories.json --skip-existing false
    """
    path = Path(file)
    if not path.exists():
        typer.echo(f"❌ File not found: {file}", err=True)
        raise typer.Exit(code=1)

    try:
        data = json.loads(path.read_text())
        memories = data.get("memories", [])
        relationships = data.get("relationships", [])

        from research_engineer.tools.memory_storage import MemoryStorageTool
        storage = MemoryStorageTool()
        imported = 0
        skipped = 0

        for mem in memories:
            did_import, did_skip = _import_memory_row(storage, mem, skip_existing)
            if did_import:
                imported += 1
            elif did_skip:
                skipped += 1

        rel_imported = sum(1 for rel in relationships if _import_relationship_row(storage, rel))

        typer.echo(f"✅ Imported {imported} memories ({skipped} skipped) and {rel_imported} relationships")
        return 0
    except Exception as e:
        typer.echo(f"❌ Error importing: {e}", err=True)
        return 1


@memory_app.command("archive")
def memory_archive(
    older_than_days: int = typer.Option(90, "--older-than", help="Archive memories older than N days"),
    memory_type: str | None = typer.Option(None, "--type", help="Only archive memories of this type"),
    dry_run: bool = typer.Option(True, "--dry-run", help="Preview without making changes"),
):
    """Archive memories older than a given number of days.

    Examples:
        research-engineer memory archive --older-than 90 --dry-run
        research-engineer memory archive --older-than 180 --dry-run false
        research-engineer memory archive --type failed_approach --older-than 30
    """
    try:
        import sqlite3
        from datetime import datetime, timedelta

        from research_engineer.tools.memory_storage import MemoryStorageTool

        storage = MemoryStorageTool()
        cutoff = (datetime.now() - timedelta(days=older_than_days)).isoformat()

        conn = sqlite3.connect(storage.db_path)
        cursor = conn.cursor()
        query = "SELECT memory_id, memory_type, created_at FROM memories WHERE is_archived = 0 AND created_at < ?"
        params: list = [cutoff]
        if memory_type:
            query += " AND memory_type = ?"
            params.append(memory_type)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        candidate_ids = [r[0] for r in rows]

        if dry_run:
            typer.echo(f"🔍 Dry run: would archive {len(candidate_ids)} memories older than {older_than_days} days")
            for mid in candidate_ids[:10]:
                typer.echo(f"   - {mid}")
            if len(candidate_ids) > 10:
                typer.echo(f"   ... and {len(candidate_ids) - 10} more")
            conn.close()
            return 0

        archived = 0
        for mid in candidate_ids:
            cursor.execute("UPDATE memories SET is_archived = 1, updated_at = CURRENT_TIMESTAMP WHERE memory_id = ?", (mid,))
            if cursor.rowcount > 0:
                archived += 1
        conn.commit()
        conn.close()

        typer.echo(f"✅ Archived {archived} memories older than {older_than_days} days")
        return 0
    except Exception as e:
        typer.echo(f"❌ Error archiving: {e}", err=True)
        return 1


# ---------------------------------------------------------------------------
# Phase 12 - Repository Memory CLI Commands
# ---------------------------------------------------------------------------


@memory_app.command("build")
def memory_build(
    repo: str = typer.Option(
        ".", "--repo", help="Repository path to index"
    ),
    output_format: str = typer.Option(
        "console", "--format", help="Output: console, json"
    ),
):
    """Build a repository memory index (full index).

    Indexes files, modules, classes, functions, methods, imports, configs,
    and builds a symbol graph + vector index for hybrid retrieval.

    Examples:
        research-engineer memory build --repo ./my_repo
        research-engineer memory build --repo . --format json
    """
    from research_engineer.memory import RepositoryMemory

    try:
        mem = RepositoryMemory(repo)
        stats = mem.build()
        if output_format == "json":
            typer.echo(stats.model_dump_json(indent=2))
        else:
            typer.echo(f"\n🏗️  Repository memory built for: {repo}")
            typer.echo(f"   Files indexed:    {stats.total_files}")
            typer.echo(f"   Symbols indexed:  {stats.total_symbols}")
            typer.echo(f"   Code chunks:      {stats.total_chunks}")
            typer.echo(f"   Graph edges:      {stats.total_edges}")
            typer.echo(f"   Index time:       {stats.index_time_seconds}s")
            typer.echo(f"   Symbols by kind:")
            for kind, count in sorted(stats.symbols_by_kind.items()):
                typer.echo(f"     {kind}: {count}")
        return 0
    except Exception as e:
        typer.echo(f"❌ Error building memory: {e}", err=True)
        return 1


@memory_app.command("refresh")
def memory_refresh(
    repo: str = typer.Option(
        ".", "--repo", help="Repository path to refresh"
    ),
    output_format: str = typer.Option(
        "console", "--format", help="Output: console, json"
    ),
):
    """Incrementally refresh the repository memory index.

    Only re-indexes files whose content hash has changed since the last
    build. Use this after editing files to keep the index up to date.

    Examples:
        research-engineer memory refresh --repo ./my_repo
    """
    from research_engineer.memory import RepositoryMemory

    try:
        mem = RepositoryMemory(repo)
        if not mem.store.has_index(str(Path(repo).resolve())):
            typer.echo("⚠️  No existing index. Run `memory build` first.")
            return 1
        stats, changed = mem.refresh()
        if output_format == "json":
            typer.echo(
                json.dumps(
                    {
                        "stats": stats.model_dump(),
                        "changed_files": changed,
                    },
                    indent=2,
                    default=str,
                )
            )
        else:
            typer.echo(f"\n🔄 Repository memory refreshed for: {repo}")
            typer.echo(f"   Changed files: {len(changed)}")
            for f in changed[:10]:
                typer.echo(f"     - {f}")
            if len(changed) > 10:
                typer.echo(f"     ... and {len(changed) - 10} more")
            typer.echo(f"   Total symbols: {stats.total_symbols}")
            typer.echo(f"   Total chunks:  {stats.total_chunks}")
        return 0
    except Exception as e:
        typer.echo(f"❌ Error refreshing memory: {e}", err=True)
        return 1


@memory_app.command("stats")
def memory_stats_repo(
    repo: str = typer.Option(
        ".", "--repo", help="Repository path"
    ),
    output_format: str = typer.Option(
        "console", "--format", help="Output: console, json"
    ),
):
    """Show repository memory index statistics.

    Examples:
        research-engineer memory stats --repo ./my_repo
    """
    from research_engineer.memory import RepositoryMemory

    try:
        mem = RepositoryMemory(repo, auto_load=True)
        stats = mem.stats()
        if stats is None:
            typer.echo("⚠️  No index found. Run `memory build` first.")
            return 1
        if output_format == "json":
            typer.echo(stats.model_dump_json(indent=2))
        else:
            typer.echo(f"\n📊 Repository Memory Stats: {repo}")
            typer.echo(f"   Files:     {stats.total_files}")
            typer.echo(f"   Symbols:   {stats.total_symbols}")
            typer.echo(f"   Chunks:    {stats.total_chunks}")
            typer.echo(f"   Edges:     {stats.total_edges}")
            typer.echo(f"   Indexed:   {stats.indexed_at}")
            typer.echo(f"   By kind:")
            for kind, count in sorted(stats.symbols_by_kind.items()):
                typer.echo(f"     {kind}: {count}")
        return 0
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        return 1


@memory_app.command("query")
def memory_query(
    query: str = typer.Argument(..., help="Natural-language query"),
    repo: str = typer.Option(
        ".", "--repo", help="Repository path"
    ),
    limit: int = typer.Option(10, "--limit", help="Max results"),
    output_format: str = typer.Option(
        "console", "--format", help="Output: console, json"
    ),
):
    """Query repository memory for relevant code + context.

    Uses hybrid retrieval (semantic + graph + metadata) to find the most
    relevant code chunks, symbols, dependencies, and tests.

    Examples:
        research-engineer memory query "EMA checkpoint support" --repo ./my_repo
        research-engineer memory query "training loop" --limit 5 --format json
    """
    from research_engineer.memory import RepositoryMemory

    try:
        mem = RepositoryMemory(repo, auto_load=True)
        if not mem.store.has_index(str(Path(repo).resolve())):
            typer.echo("⚠️  No index found. Run `memory build` first.")
            return 1
        results = mem.query(query, limit=limit)
        if output_format == "json":
            typer.echo(
                json.dumps([r.model_dump() for r in results], indent=2, default=str)
            )
        else:
            typer.echo(f"\n🔍 Query: '{query}' ({len(results)} results)\n")
            for i, r in enumerate(results, 1):
                sym = r.symbol
                typer.echo(
                    f"{i}. [{r.combined_score:.3f}] "
                    f"{sym.kind.value if sym else 'unknown'}:"
                    f"{sym.qualified_name if sym else r.chunk.chunk_id}"
                )
                typer.echo(
                    f"   📄 {r.chunk.file_path}:{r.chunk.line_start}-{r.chunk.line_end}"
                )
                typer.echo(
                    f"   📊 sem={r.semantic_score:.2f} "
                    f"graph={r.graph_score:.2f} "
                    f"meta={r.metadata_score:.2f}"
                )
                if r.related_symbols:
                    typer.echo(
                        f"   🔗 related: {', '.join(s.name for s in r.related_symbols[:5])}"
                    )
                if sym and sym.docstring:
                    typer.echo(f"   📝 {sym.docstring.splitlines()[0][:80]}")
                typer.echo()
        return 0
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        return 1


@memory_app.command("symbol-graph")
def memory_graph_cmd(
    symbol: str = typer.Argument(..., help="Symbol name to explore"),
    repo: str = typer.Option(
        ".", "--repo", help="Repository path"
    ),
    output_format: str = typer.Option(
        "console", "--format", help="Output: console, json"
    ),
):
    """Show the symbol graph neighborhood for a symbol.

    Displays dependencies, dependents, callers, callees, related symbols,
    and associated tests.

    Examples:
        research-engineer memory graph "Trainer" --repo ./my_repo
        research-engineer memory graph "save_checkpoint" --format json
    """
    from research_engineer.memory import RepositoryMemory

    try:
        mem = RepositoryMemory(repo, auto_load=True)
        if not mem.store.has_index(str(Path(repo).resolve())):
            typer.echo("⚠️  No index found. Run `memory build` first.")
            return 1
        result = mem.graph(symbol)
        if output_format == "json":
            typer.echo(json.dumps(result, indent=2, default=str))
        else:
            if not result.get("found"):
                typer.echo(f"❌ Symbol not found: {symbol}")
                return 1
            sym = result["symbol"]
            typer.echo(f"\n🕸️  Symbol: {sym['qualified_name']}")
            typer.echo(f"   Kind: {sym['kind']}")
            typer.echo(f"   File: {sym['file_path']}:{sym['line_start']}")
            if result.get("dependencies"):
                typer.echo(f"\n   ⬇️  Dependencies ({len(result['dependencies'])}):")
                for d in result["dependencies"][:10]:
                    typer.echo(f"      - {d['qualified_name']} ({d['file_path']})")
            if result.get("dependents"):
                typer.echo(f"\n   ⬆️  Dependents ({len(result['dependents'])}):")
                for d in result["dependents"][:10]:
                    typer.echo(f"      - {d['qualified_name']} ({d['file_path']})")
            if result.get("callers"):
                typer.echo(f"\n   📞 Callers ({len(result['callers'])}):")
                for c in result["callers"][:10]:
                    typer.echo(f"      - {c['qualified_name']}")
            if result.get("callees"):
                typer.echo(f"\n   📞 Callees ({len(result['callees'])}):")
                for c in result["callees"][:10]:
                    typer.echo(f"      - {c['qualified_name']}")
            if result.get("tests"):
                typer.echo(f"\n   🧪 Tests ({len(result['tests'])}):")
                for t in result["tests"][:10]:
                    typer.echo(f"      - {t['file_path']}")
            if result.get("related"):
                typer.echo(f"\n   🔗 Related ({len(result['related'])}):")
                for r in result["related"][:10]:
                    typer.echo(f"      - {r['qualified_name']}")
        return 0
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        return 1


async def main():
    """Main entry point."""
    pass


# ---------------------------------------------------------------------------
# Phase 6 - Literature Intelligence CLI Commands
# ---------------------------------------------------------------------------


@literature_app.command("search")
def literature_search(
    query: str = typer.Argument(..., help="Search query"),
    sources: str = typer.Option(
        "local,arxiv",
        "--sources",
        help="Comma-separated sources: local, arxiv, semantic_scholar",
    ),
    max_results: int = typer.Option(20, "--max-results", help="Max results per source"),
    output_format: str = typer.Option("console", "--format", help="Output: console, json"),
):
    """Search papers across multiple sources.

    Examples:
        research-engineer literature search "attention optimization"
        research-engineer literature search "moe" --sources arxiv,semantic_scholar
    """
    from research_engineer.models.literature import SearchSource

    source_list = []
    for s in sources.split(","):
        s = s.strip().lower()
        try:
            source_list.append(SearchSource(s))
        except ValueError:
            typer.echo(f"⚠️  Unknown source: {s}", err=True)

    agent = _get_literature_agent()
    try:
        result = asyncio.run(agent.search_papers(query, max_results=max_results, sources=source_list))
        if output_format == "json":
            typer.echo(result.model_dump_json(indent=2))
        else:
            typer.echo(f"\n🔍 Found {result.total_found} papers for '{query}'")
            typer.echo(f"   Sources: {', '.join(result.sources_searched)}")
            typer.echo(f"   Time: {result.search_time_seconds}s\n")
            for i, p in enumerate(result.papers[:20], 1):
                year_str = f" [{p.year}]" if p.year else ""
                cite_str = f" (cites: {p.citation_count})" if p.citation_count else ""
                typer.echo(f"{i}. [{p.source.value}]{year_str}{cite_str} {p.title[:80]}")
                typer.echo(f"   ID: {p.paper_id}")
        return 0
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        return 1


@literature_app.command("compare")
def literature_compare(
    papers: str = typer.Argument(..., help="Comma-separated paper IDs or paths"),
    output_format: str = typer.Option("console", "--format", help="Output: console, json"),
):
    """Compare papers across multiple dimensions.

    Examples:
        research-engineer literature compare 2503.12345,2401.00001
    """
    from research_engineer.models.literature import PaperSummary

    paper_ids = [p.strip() for p in papers.split(",") if p.strip()]
    if len(paper_ids) < 2:
        typer.echo("❌ Need at least 2 paper IDs to compare", err=True)
        return 1

    summaries: list[PaperSummary] = []
    for pid in paper_ids:
        summaries.append(PaperSummary(paper_id=pid, title=f"Paper {pid}", abstract=""))

    agent = _get_literature_agent()
    try:
        result = asyncio.run(agent.compare_papers(summaries))
        if output_format == "json":
            typer.echo(result.model_dump_json(indent=2))
        else:
            typer.echo(f"\n📊 Comparison of {len(summaries)} papers\n")
            typer.echo("Similarities:")
            for sim in result.similarities[:5]:
                typer.echo(f"  {sim.paper_a} <-> {sim.paper_b}: {sim.similarity_score:.2f}")
            typer.echo("\nRanking:")
            for rank in result.ranking:
                typer.echo(f"  {rank.rank}. {rank.paper_id} (score: {rank.score:.2f})")
        return 0
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        return 1


@literature_app.command("review")
def literature_review(
    topic: str = typer.Argument(..., help="Review topic"),
    max_papers: int = typer.Option(15, "--max-papers", help="Max papers to include"),
    depth: str = typer.Option("standard", "--depth", help="brief, standard, comprehensive"),
    output_dir: str = typer.Option("output/literature", "--output-dir", help="Output directory"),
    output_format: str = typer.Option("console", "--format", help="Output: console, json, markdown"),
):
    """Generate a structured literature review.

    Examples:
        research-engineer literature review "flash attention"
        research-engineer literature review "mixture of experts" --depth comprehensive
    """
    from research_engineer.models.literature import ReviewDepth

    try:
        review_depth = ReviewDepth(depth)
    except ValueError:
        typer.echo(f"❌ Invalid depth: {depth}. Use: brief, standard, comprehensive", err=True)
        return 1

    agent = _get_literature_agent()
    try:
        search_result = asyncio.run(agent.search_papers(topic, max_results=max_papers))
        papers = agent._to_summaries(search_result)

        if not papers:
            typer.echo(f"No papers found for '{topic}'")
            return 0

        result = asyncio.run(agent.generate_review(topic, papers, depth=review_depth))

        if output_format == "json":
            typer.echo(result.model_dump_json(indent=2))
        elif output_format == "markdown":
            typer.echo(result.markdown)
        else:
            typer.echo(f"\n📚 Literature Review: {topic}\n")
            typer.echo(f"Papers analyzed: {result.review.papers_analyzed}")
            typer.echo(f"Sections: {len(result.review.sections)}")
            typer.echo(f"Key findings: {len(result.review.key_findings)}")
            typer.echo(f"Research gaps: {len(result.review.research_gaps)}")
            typer.echo(f"\nExecutive Summary:\n{result.review.executive_summary[:300]}...")
        return 0
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        return 1


@literature_app.command("relationships")
def literature_relationships(
    papers: str = typer.Argument(..., help="Comma-separated paper IDs"),
    output_format: str = typer.Option("console", "--format", help="Output: console, json"),
):
    """Detect relationships between papers.

    Examples:
        research-engineer literature relationships 2503.12345,2401.00001,2305.12345
    """
    from research_engineer.models.literature import PaperSummary

    paper_ids = [p.strip() for p in papers.split(",") if p.strip()]
    summaries = [PaperSummary(paper_id=pid, title=f"Paper {pid}", abstract="") for pid in paper_ids]

    agent = _get_literature_agent()
    try:
        result = asyncio.run(agent.detect_relationships(summaries))
        if output_format == "json":
            typer.echo(result.model_dump_json(indent=2))
        else:
            typer.echo(f"\n🔗 Detected {len(result.relationships)} relationships\n")
            for rel in result.relationships[:20]:
                typer.echo(f"  {rel.source_paper_id} --[{rel.relationship_type.value}]--> {rel.target_paper_id} (conf: {rel.confidence:.2f})")
            typer.echo(f"\nSummary: {result.summary}")
        return 0
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        return 1


@literature_app.command("trends")
def literature_trends(
    topic: str = typer.Argument(..., help="Topic to analyze trends for"),
    years: int = typer.Option(5, "--years", help="Time window in years"),
    max_papers: int = typer.Option(30, "--max-papers", help="Max papers to search"),
    output_format: str = typer.Option("console", "--format", help="Output: console, json"),
):
    """Analyze research trends for a topic.

    Examples:
        research-engineer literature trends "mixture of experts" --years 5
    """
    agent = _get_literature_agent()
    try:
        search_result = asyncio.run(agent.search_papers(topic, max_results=max_papers))
        papers = agent._to_summaries(search_result)

        if not papers:
            typer.echo(f"No papers found for '{topic}'")
            return 0

        from research_engineer.models.literature import TrendAnalysisInput

        result = asyncio.run(agent.trend.execute(
            TrendAnalysisInput(papers=papers, time_window_years=years)
        ))

        if output_format == "json":
            typer.echo(result.model_dump_json(indent=2))
        else:
            typer.echo(f"\n📈 Trend Analysis: {topic}\n")
            typer.echo(result.trend_summary)
            typer.echo(f"\nTrends ({len(result.trends)}):")
            for t in result.trends[:10]:
                typer.echo(f"  {t.topic}: {t.direction.value} ({t.growth_rate:.1f}%/yr)")
            if result.emerging_topics:
                typer.echo(f"\nEmerging: {', '.join(e.topic for e in result.emerging_topics[:5])}")
            if result.hot_topics:
                typer.echo(f"Hot: {', '.join(h.topic for h in result.hot_topics[:5])}")
        return 0
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        return 1


@literature_app.command("recommend")
def literature_recommend(
    topic: str = typer.Argument(..., help="Topic for recommendations"),
    max: int = typer.Option(10, "--max", help="Max recommendations"),
    max_papers: int = typer.Option(30, "--max-papers", help="Max papers to search"),
    output_format: str = typer.Option("console", "--format", help="Output: console, json"),
):
    """Recommend papers worth implementing.

    Examples:
        research-engineer literature recommend "attention" --max 10
    """
    agent = _get_literature_agent()
    try:
        search_result = asyncio.run(agent.search_papers(topic, max_results=max_papers))
        papers = agent._to_summaries(search_result)

        if not papers:
            typer.echo(f"No papers found for '{topic}'")
            return 0

        result = asyncio.run(agent.recommend_papers(papers, max_recommendations=max))

        if output_format == "json":
            typer.echo(result.model_dump_json(indent=2))
        else:
            typer.echo(f"\n🏆 Paper Recommendations: {topic}\n")
            typer.echo(result.ranking_rationale)
            typer.echo()
            for rec in result.recommendations:
                typer.echo(f"{rec.rank}. {rec.title[:70]}")
                typer.echo(f"   Score: {rec.overall_score:.2f} (impact={rec.impact_score:.2f}, novelty={rec.novelty_score:.2f})")
        return 0
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        return 1


@literature_app.command("relevance")
def literature_relevance(
    paper: str = typer.Argument(..., help="Paper ID or arXiv ID"),
    repo: str = typer.Argument(..., help="Repository path"),
    output_format: str = typer.Option("console", "--format", help="Output: console, json"),
):
    """Score paper relevance to a repository.

    Examples:
        research-engineer literature relevance 2503.12345 ./my_repo
    """
    agent = _get_literature_agent()
    try:
        repo_summary = asyncio.run(agent._load_repo_summary(repo))
        if not repo_summary:
            typer.echo(f"❌ Could not analyze repository: {repo}", err=True)
            return 1

        from research_engineer.models.literature import PaperSummary

        paper_summary = PaperSummary(paper_id=paper, title=f"Paper {paper}", abstract="")
        result = asyncio.run(agent.score_relevance(paper_summary, repo_summary))

        if not result:
            typer.echo("❌ Could not compute relevance score", err=True)
            return 1

        if output_format == "json":
            typer.echo(result.model_dump_json(indent=2))
        else:
            typer.echo(f"\n🎯 Relevance Score: {result.score.paper_id} -> {result.score.repo_path}")
            typer.echo(f"Overall: {result.score.overall_score:.2f} ({result.score.relevance_level.value})\n")
            for dim in result.dimension_scores:
                typer.echo(f"  {dim.dimension}: {dim.score:.2f} - {dim.reasoning}")
            typer.echo("\nRecommendations:")
            for rec in result.recommendations:
                typer.echo(f"  - {rec}")
        return 0
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        return 1


@literature_app.command("discover")
def literature_discover(
    topic: str = typer.Argument(..., help="Research topic to explore"),
    repo: str | None = typer.Option(None, "--repo", help="Repository path for relevance scoring"),
    max_papers: int = typer.Option(20, "--max-papers", help="Max papers to search"),
    output_dir: str = typer.Option("output/literature", "--output-dir", help="Output directory"),
    output_format: str = typer.Option("console", "--format", help="Output: console, json"),
):
    """Full literature intelligence workflow.

    Searches, compares, detects relationships, analyzes trends,
    generates review, recommends papers, and optionally scores relevance.

    Examples:
        research-engineer literature discover "flash attention"
        research-engineer literature discover "moe" --repo ./my_repo
    """
    agent = _get_literature_agent()
    try:
        result = asyncio.run(
            agent.discover(topic, repo_path=repo, output_dir=output_dir, max_papers=max_papers)
        )

        if output_format == "json":
            typer.echo(result.model_dump_json(indent=2, default=str))
        else:
            typer.echo(f"\n🚀 Literature Discovery: {topic}\n")
            typer.echo(f"Time: {result.processing_time_seconds}s")
            typer.echo(f"Memory IDs: {len(result.memory_ids)}")
            if result.search_results:
                typer.echo(f"Papers found: {result.search_results.total_found}")
            if result.relationships:
                typer.echo(f"Relationships: {len(result.relationships.relationships)}")
            if result.trends:
                typer.echo(f"Trends: {len(result.trends.trends)}")
            if result.review:
                typer.echo(f"Review sections: {len(result.review.review.sections)}")
            if result.recommendations:
                typer.echo(f"Recommendations: {len(result.recommendations.recommendations)}")
            if result.relevance:
                typer.echo(f"Relevance: {result.relevance.score.overall_score:.2f} ({result.relevance.score.relevance_level.value})")
            typer.echo(f"\nFiles generated: {len(result.generated_files)}")
            for f in result.generated_files:
                typer.echo(f"  - {f}")
        return 0
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        return 1


# ---------------------------------------------------------------------------
# Phase 7 - Experiment Execution CLI Commands
# ---------------------------------------------------------------------------


@experiment_app.command("run")
def experiment_run(
    command: str = typer.Option(
        ...,
        "--command",
        help="Command to execute (e.g., 'python train.py --config config.yaml')",
    ),
    repo: str = typer.Option(
        ...,
        "--repo",
        help="Repository working directory path",
    ),
    type: str = typer.Option(
        "training",
        "--type",
        help="Experiment type: training, evaluation, validation",
    ),
    paper: str | None = typer.Option(
        None, "--paper", help="Associated paper ID"
    ),
    plan: str | None = typer.Option(
        None, "--plan", help="Associated plan ID"
    ),
    patch: str | None = typer.Option(
        None, "--patch", help="Associated patch ID"
    ),
    implementation: str | None = typer.Option(
        None, "--implementation", help="Associated implementation ID"
    ),
    timeout: int = typer.Option(
        3600, "--timeout", help="Timeout in seconds"
    ),
    output_dir: str = typer.Option(
        "output/experiments", "--output-dir", help="Output directory"
    ),
    dry_run: bool = typer.Option(
        True, "--dry-run", help="Dry run (do not execute command)"
    ),
    output_format: str = typer.Option(
        "console", "--format", help="Output: console, json"
    ),
):
    """Run an ML experiment (training, evaluation, or validation).

    By default, runs in dry-run mode (prints command without executing).
    Use --no-dry-run to execute the command.

    Examples:
        research-engineer experiment run --command "python train.py" --repo ./my_repo
        research-engineer experiment run --command "python eval.py" --repo . --type evaluation --no-dry-run
        research-engineer experiment run --command "pytest" --repo . --type validation --paper 2503.12345
    """
    try:
        exp_type = ExperimentType(type)
    except ValueError:
        typer.echo(
            f"❌ Invalid type: {type}. Use: training, evaluation, validation",
            err=True,
        )
        return 1

    agent = _get_experiment_agent()
    try:
        result = asyncio.run(
            agent.run(
                command=command,
                repo_path=repo,
                paper_id=paper,
                plan_id=plan,
                patch_id=patch,
                implementation_id=implementation,
                experiment_type=exp_type,
                timeout_seconds=timeout,
                dry_run=dry_run,
                output_dir=output_dir,
            )
        )

        if output_format == "json":
            typer.echo(result.model_dump_json(indent=2))
        else:
            typer.echo(f"\n🧪 Experiment: {result.experiment_id}")
            typer.echo(f"   Repository: {result.repo_path}")
            if result.run:
                typer.echo(f"   Status: {result.run.status.value}")
                typer.echo(f"   Exit code: {result.run.exit_code}")
                typer.echo(f"   Duration: {result.run.duration_seconds}s")
            if result.metrics:
                typer.echo(
                    f"   Metrics: {len(result.metrics.summary_metrics)} collected"
                )
            if result.artifacts:
                typer.echo(f"   Artifacts: {len(result.artifacts.artifacts)}")
            if result.failure and result.failure.detected_failure:
                typer.echo(
                    f"   Failure: {result.failure.failure_mode} "
                    f"({result.failure.severity.value})"
                )
            if result.memory_ids:
                typer.echo(f"   Memory IDs: {len(result.memory_ids)}")
            typer.echo(f"   Time: {result.processing_time_seconds}s")
            if result.generated_files:
                typer.echo("\n   Files generated:")
                for f in result.generated_files:
                    typer.echo(f"     - {f}")
        return 0
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        return 1


@experiment_app.command("monitor")
def experiment_monitor(
    experiment_id: str = typer.Argument(..., help="Experiment ID to query"),
    output_format: str = typer.Option(
        "console", "--format", help="Output: console, json"
    ),
):
    """Show monitoring summary for an experiment.

    Examples:
        research-engineer experiment monitor exp_abc123
    """
    agent = _get_experiment_agent()
    try:
        record = asyncio.run(agent.get_experiment(experiment_id))
        if not record:
            typer.echo(f"❌ Experiment not found: {experiment_id}", err=True)
            return 1

        if output_format == "json":
            typer.echo(record.model_dump_json(indent=2))
        else:
            typer.echo(f"\n📊 Experiment: {record.experiment_id}")
            typer.echo(f"   Status: {record.status.value}")
            typer.echo(f"   Type: {record.experiment_type.value}")
            typer.echo(f"   Repo: {record.repo_path}")
            typer.echo(f"   Duration: {record.duration_seconds}s")
            typer.echo(f"   Exit code: {record.exit_code}")
            if record.metrics:
                typer.echo(f"   Metrics: {len(record.metrics)}")
            if record.failure_mode:
                typer.echo(
                    f"   Failure: {record.failure_mode} "
                    f"({record.failure_severity.value})"
                )
        return 0
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        return 1


@experiment_app.command("list")
def experiment_list(
    status: str | None = typer.Option(
        None, "--status", help="Filter by status"
    ),
    type: str | None = typer.Option(
        None, "--type", help="Filter by experiment type"
    ),
    paper: str | None = typer.Option(
        None, "--paper", help="Filter by paper ID"
    ),
    repo: str | None = typer.Option(
        None, "--repo", help="Filter by repository path"
    ),
    limit: int = typer.Option(20, "--limit", help="Max results"),
    offset: int = typer.Option(0, "--offset", help="Pagination offset"),
    output_format: str = typer.Option(
        "console", "--format", help="Output: console, json"
    ),
):
    """List experiments with optional filters.

    Examples:
        research-engineer experiment list
        research-engineer experiment list --status completed --limit 50
        research-engineer experiment list --paper 2503.12345
    """
    agent = _get_experiment_agent()
    try:
        status_filter = None
        if status:
            try:
                status_filter = ExperimentStatus(status)
            except ValueError:
                typer.echo(f"❌ Invalid status: {status}", err=True)
                return 1

        type_filter = None
        if type:
            try:
                type_filter = ExperimentType(type)
            except ValueError:
                typer.echo(f"❌ Invalid type: {type}", err=True)
                return 1

        output = asyncio.run(
            agent.list_experiments(
                paper_id=paper,
                repo_path=repo,
                status=status_filter,
                experiment_type=type_filter,
                limit=limit,
                offset=offset,
            )
        )

        if output_format == "json":
            typer.echo(output.model_dump_json(indent=2))
        else:
            typer.echo(
                f"\n📋 Experiments (total: {output.total}, showing: {len(output.experiments)})\n"
            )
            for i, exp in enumerate(output.experiments, 1):
                typer.echo(
                    f"{i}. {exp.status.value} | {exp.experiment_type.value} | "
                    f"{exp.experiment_id} | {exp.repo_path}"
                )
            if not output.experiments:
                typer.echo("No experiments found.")
        return 0
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        return 1


@experiment_app.command("get")
def experiment_get(
    experiment_id: str = typer.Argument(..., help="Experiment ID"),
    output_format: str = typer.Option(
        "console", "--format", help="Output: console, json"
    ),
):
    """Get experiment details by ID.

    Examples:
        research-engineer experiment get exp_abc123
    """
    agent = _get_experiment_agent()
    try:
        record = asyncio.run(agent.get_experiment(experiment_id))
        if not record:
            typer.echo(f"❌ Experiment not found: {experiment_id}", err=True)
            return 1

        if output_format == "json":
            typer.echo(record.model_dump_json(indent=2))
        else:
            typer.echo(f"\n🧪 Experiment: {record.experiment_id}")
            typer.echo(f"   Type: {record.experiment_type.value}")
            typer.echo(f"   Status: {record.status.value}")
            typer.echo(f"   Repo: {record.repo_path}")
            typer.echo(f"   Paper: {record.paper_id or 'N/A'}")
            typer.echo(f"   Plan: {record.plan_id or 'N/A'}")
            typer.echo(f"   Patch: {record.patch_id or 'N/A'}")
            typer.echo(f"   Start: {record.start_time}")
            typer.echo(f"   End: {record.end_time}")
            typer.echo(f"   Duration: {record.duration_seconds}s")
            typer.echo(f"   Exit code: {record.exit_code}")
            typer.echo(f"   Command: {' '.join(record.command)}")
            if record.metrics:
                typer.echo(f"   Metrics: {record.metrics}")
            if record.failure_mode:
                typer.echo(f"   Failure: {record.failure_mode}")
                typer.echo(f"   Root cause: {record.root_cause}")
            if record.memory_id:
                typer.echo(f"   Memory ID: {record.memory_id}")
        return 0
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        return 1


@experiment_app.command("search")
def experiment_search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(20, "--limit", help="Max results"),
    output_format: str = typer.Option(
        "console", "--format", help="Output: console, json"
    ),
):
    """Search experiment history by text.

    Examples:
        research-engineer experiment search "attention"
        research-engineer experiment search "OOM" --limit 10
    """
    agent = _get_experiment_agent()
    try:
        output = asyncio.run(agent.search_experiments(query, limit=limit))

        if output_format == "json":
            typer.echo(output.model_dump_json(indent=2))
        else:
            typer.echo(
                f"\n🔍 Search results for '{query}' "
                f"(total: {output.total})\n"
            )
            for i, exp in enumerate(output.experiments, 1):
                typer.echo(
                    f"{i}. {exp.status.value} | {exp.experiment_id} | "
                    f"{exp.repo_path}"
                )
            if not output.experiments:
                typer.echo("No experiments found.")
        return 0
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        return 1


@experiment_app.command("cancel")
def experiment_cancel(
    experiment_id: str = typer.Argument(..., help="Experiment ID to cancel"),
):
    """Cancel a running experiment.

    Examples:
        research-engineer experiment cancel exp_abc123
    """
    agent = _get_experiment_agent()
    try:
        cancelled = asyncio.run(agent.cancel_experiment(experiment_id))
        if cancelled:
            typer.echo(f"✅ Cancelled experiment: {experiment_id}")
        else:
            typer.echo(
                f"⚠️  Experiment not running or not found: {experiment_id}",
                err=True,
            )
        return 0
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        return 1


@experiment_app.command("history")
def experiment_history(
    paper: str = typer.Argument(..., help="Paper ID"),
    limit: int = typer.Option(20, "--limit", help="Max results"),
    output_format: str = typer.Option(
        "console", "--format", help="Output: console, json"
    ),
):
    """Show experiment history for a paper.

    Examples:
        research-engineer experiment history 2503.12345
        research-engineer experiment history 2503.12345 --limit 50
    """
    agent = _get_experiment_agent()
    try:
        output = asyncio.run(agent.history(paper, limit=limit))

        if output_format == "json":
            typer.echo(output.model_dump_json(indent=2))
        else:
            typer.echo(
                f"\n📜 History for paper {paper} "
                f"(total: {output.total})\n"
            )
            for i, exp in enumerate(output.experiments, 1):
                typer.echo(
                    f"{i}. {exp.status.value} | {exp.experiment_type.value} | "
                    f"{exp.experiment_id} | {exp.start_time}"
                )
            if not output.experiments:
                typer.echo("No experiments found for this paper.")
        return 0
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        return 1


# ---------------------------------------------------------------------------
# Phase 8 - Evaluation CLI Commands
# ---------------------------------------------------------------------------


def _load_experiment_records(ids: list[str]) -> list:
    """Load experiment records by ID (skips missing)."""
    exp_agent = _get_experiment_agent()
    records = []
    for eid in ids:
        r = asyncio.run(exp_agent.get_experiment(eid))
        if r:
            records.append(r)
    return records


@evaluate_app.command("run")
def evaluate_run(
    experiment: str = typer.Argument(
        ..., help="Experiment ID to evaluate (single-run dynamics)"
    ),
    paper: str | None = typer.Option(
        None, "--paper", help="Associated paper ID"
    ),
    output_dir: str = typer.Option(
        "output/evaluations", "--output-dir", help="Output directory"
    ),
    output_format: str = typer.Option(
        "console", "--format", help="Output: console, json"
    ),
):
    """Evaluate a single experiment (training dynamics analysis).

    Examples:
        research-engineer evaluate run exp_abc123
        research-engineer evaluate run exp_abc123 --paper 2503.12345
    """
    agent = _get_evaluation_agent()
    try:
        record = asyncio.run(
            _get_experiment_agent().get_experiment(experiment)
        )
        if not record:
            typer.echo(
                f"❌ Experiment not found: {experiment}", err=True
            )
            return 1
        result = asyncio.run(
            agent.evaluate_single(
                record, paper_id=paper, output_dir=output_dir
            )
        )
        if output_format == "json":
            typer.echo(result.model_dump_json(indent=2))
        else:
            typer.echo(f"\n📊 Evaluation: {result.evaluation_id}")
            typer.echo(f"   Experiment: {experiment}")
            if result.dynamics:
                dyn = result.dynamics[0]
                typer.echo(f"   Summary: {dyn.summary}")
                typer.echo(f"   Stability: {dyn.stability_score:.2f}")
            if result.next_experiments:
                typer.echo(
                    f"   Recommendations: "
                    f"{len(result.next_experiments.experiment_recommendations)}"
                )
            typer.echo(f"   Time: {result.processing_time_seconds}s")
        return 0
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        return 1


@evaluate_app.command("compare")
def evaluate_compare(
    experiments: str = typer.Argument(
        ..., help="Comma-separated experiment IDs"
    ),
    metric: str | None = typer.Option(
        None, "--metric", help="Primary metric for winner"
    ),
    higher_is_better: bool = typer.Option(
        False, "--higher-is-better", help="Higher metric is better"
    ),
    output_dir: str = typer.Option(
        "output/evaluations", "--output-dir", help="Output directory"
    ),
    output_format: str = typer.Option(
        "console", "--format", help="Output: console, json"
    ),
):
    """Compare experiments across metrics.

    Examples:
        research-engineer evaluate compare exp_a,exp_b --metric loss
        research-engineer evaluate compare exp_a,exp_b,exp_c --metric accuracy --higher-is-better
    """
    ids = [e.strip() for e in experiments.split(",") if e.strip()]
    if len(ids) < 2:
        typer.echo("❌ Need at least 2 experiment IDs to compare", err=True)
        return 1
    exp_agent = _get_experiment_agent()
    agent = _get_evaluation_agent()
    try:
        records = []
        for eid in ids:
            r = asyncio.run(exp_agent.get_experiment(eid))
            if r:
                records.append(r)
        if len(records) < 2:
            typer.echo(
                "❌ Could not load enough experiments to compare",
                err=True,
            )
            return 1
        result = asyncio.run(
            agent.analyze(
                records,
                primary_metric=metric,
                higher_is_better=higher_is_better,
                output_dir=output_dir,
            )
        )
        if output_format == "json":
            typer.echo(result.model_dump_json(indent=2))
        else:
            typer.echo(f"\n📊 Comparison: {result.evaluation_id}")
            if result.comparison:
                comp = result.comparison
                typer.echo(f"   Compared: {comp.experiments_compared}")
                typer.echo(
                    f"   Winner: {comp.winner_experiment_id or 'N/A'}"
                )
                for d in comp.metric_deltas[:5]:
                    typer.echo(
                        f"   {d.metric}: best={d.best_experiment_id} "
                        f"({d.best_value:.4f})"
                    )
            typer.echo(f"   Time: {result.processing_time_seconds}s")
        return 0
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        return 1


@evaluate_app.command("analyze")
def evaluate_analyze(
    experiments: str = typer.Argument(
        ..., help="Comma-separated experiment IDs"
    ),
    paper: str | None = typer.Option(
        None, "--paper", help="Associated paper ID"
    ),
    repo: str | None = typer.Option(
        None, "--repo", help="Repository path"
    ),
    metric: str | None = typer.Option(
        None, "--metric", help="Primary metric"
    ),
    higher_is_better: bool = typer.Option(
        False, "--higher-is-better", help="Higher metric is better"
    ),
    output_dir: str = typer.Option(
        "output/evaluations", "--output-dir", help="Output directory"
    ),
    output_format: str = typer.Option(
        "console", "--format", help="Output: console, json"
    ),
):
    """Full evaluation workflow (compare + dynamics + stats + next).

    Examples:
        research-engineer evaluate analyze exp_a,exp_b --metric loss --paper 2503.12345
    """
    ids = [e.strip() for e in experiments.split(",") if e.strip()]
    if not ids:
        typer.echo("❌ Need at least 1 experiment ID", err=True)
        return 1
    agent = _get_evaluation_agent()
    try:
        records = _load_experiment_records(ids)
        if not records:
            typer.echo("❌ No experiments found", err=True)
            return 1
        result = asyncio.run(
            agent.analyze(
                records,
                paper_id=paper,
                repo_path=repo,
                primary_metric=metric,
                higher_is_better=higher_is_better,
                output_dir=output_dir,
            )
        )
        if output_format == "json":
            typer.echo(result.model_dump_json(indent=2))
        else:
            _print_analyze_console(result)
        return 0
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        return 1


def _print_analyze_console(result) -> None:
    """Print the analyze workflow result in console format."""
    typer.echo(f"\n📊 Evaluation: {result.evaluation_id}")
    typer.echo(f"   Experiments: {len(result.experiment_ids)}")
    if result.comparison:
        typer.echo(
            f"   Winner: "
            f"{result.comparison.winner_experiment_id or 'N/A'}"
        )
    if result.dynamics:
        typer.echo(f"   Dynamics: {len(result.dynamics)} analyzed")
    if result.significance:
        typer.echo(
            f"   Significance: "
            f"{result.significance.pairwise_count} pairs"
        )
    if result.next_experiments:
        typer.echo(
            f"   Recommendations: "
            f"{len(result.next_experiments.experiment_recommendations)}"
        )
    if result.memory_ids:
        typer.echo(f"   Memory IDs: {len(result.memory_ids)}")
    typer.echo(f"   Time: {result.processing_time_seconds}s")
    if result.generated_files:
        typer.echo("\n   Files generated:")
        for f in result.generated_files:
            typer.echo(f"     - {f}")


@evaluate_app.command("dynamics")
def evaluate_dynamics(
    experiment: str = typer.Argument(
        ..., help="Experiment ID"
    ),
    output_format: str = typer.Option(
        "console", "--format", help="Output: console, json"
    ),
):
    """Show training dynamics for an experiment.

    Examples:
        research-engineer evaluate dynamics exp_abc123
    """
    agent = _get_evaluation_agent()
    try:
        record = asyncio.run(
            _get_experiment_agent().get_experiment(experiment)
        )
        if not record:
            typer.echo(
                f"❌ Experiment not found: {experiment}", err=True
            )
            return 1
        dyn = asyncio.run(agent.dynamics_analysis(record))
        if output_format == "json":
            typer.echo(dyn.model_dump_json(indent=2))
        else:
            typer.echo(f"\n📈 Dynamics: {experiment}")
            typer.echo(f"   Summary: {dyn.summary}")
            typer.echo(f"   Stability: {dyn.stability_score:.2f}")
            for p in dyn.patterns:
                status = "DETECTED" if p.detected else "no"
                typer.echo(
                    f"   - {p.pattern_type.value} ({status}, "
                    f"conf={p.confidence:.2f})"
                )
        return 0
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        return 1


@evaluate_app.command("significance")
def evaluate_significance(
    experiments: str = typer.Argument(
        ..., help="Comma-separated experiment IDs"
    ),
    metric: str = typer.Option(
        ..., "--metric", help="Metric to compare"
    ),
    higher_is_better: bool = typer.Option(
        False, "--higher-is-better", help="Higher metric is better"
    ),
    output_format: str = typer.Option(
        "console", "--format", help="Output: console, json"
    ),
):
    """Show statistical significance between experiments.

    Examples:
        research-engineer evaluate significance exp_a,exp_b --metric accuracy --higher-is-better
    """
    ids = [e.strip() for e in experiments.split(",") if e.strip()]
    if len(ids) < 2:
        typer.echo(
            "❌ Need at least 2 experiment IDs for significance", err=True
        )
        return 1
    exp_agent = _get_experiment_agent()
    agent = _get_evaluation_agent()
    try:
        records = []
        for eid in ids:
            r = asyncio.run(exp_agent.get_experiment(eid))
            if r:
                records.append(r)
        if len(records) < 2:
            typer.echo("❌ Could not load enough experiments", err=True)
            return 1
        out = asyncio.run(
            agent.significance_test(
                records, metric=metric, higher_is_better=higher_is_better
            )
        )
        if output_format == "json":
            typer.echo(out.model_dump_json(indent=2))
        else:
            typer.echo(f"\n📐 Significance on '{metric}'\n")
            typer.echo(f"Verdict: {out.overall_verdict}")
            typer.echo(f"Best: {out.best_experiment_id or 'N/A'}")
            for r in out.results:
                typer.echo(
                    f"  {r.comparison}: p={r.p_value:.4f} "
                    f"({'sig' if r.significant else 'ns'}, "
                    f"{r.effect_size_label})"
                )
            if out.insufficient_data_pairs:
                typer.echo(
                    f"  Insufficient data: "
                    f"{', '.join(out.insufficient_data_pairs)}"
                )
        return 0
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        return 1


@evaluate_app.command("next")
def evaluate_next(
    experiments: str = typer.Argument(
        ..., help="Comma-separated experiment IDs"
    ),
    max: int = typer.Option(5, "--max", help="Max recommendations"),
    output_format: str = typer.Option(
        "console", "--format", help="Output: console, json"
    ),
):
    """Recommend next experiments to run.

    Examples:
        research-engineer evaluate next exp_a,exp_b --max 5
    """
    ids = [e.strip() for e in experiments.split(",") if e.strip()]
    exp_agent = _get_experiment_agent()
    agent = _get_evaluation_agent()
    try:
        records = []
        for eid in ids:
            r = asyncio.run(exp_agent.get_experiment(eid))
            if r:
                records.append(r)
        out = asyncio.run(agent.next_experiments(records))
        if output_format == "json":
            typer.echo(out.model_dump_json(indent=2))
        else:
            typer.echo("\n🎯 Next Experiments\n")
            typer.echo(out.overall_strategy)
            typer.echo()
            for rec in out.experiment_recommendations:
                typer.echo(
                    f"{rec.rank}. [{rec.priority.value}] {rec.title}"
                )
                typer.echo(f"   {rec.rationale}")
            if out.paper_suggestions:
                typer.echo("\nPapers:")
                for ps in out.paper_suggestions:
                    typer.echo(f"  - {ps.paper_id}: {ps.title}")
        return 0
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        return 1


@evaluate_app.command("list")
def evaluate_list(
    paper: str | None = typer.Option(
        None, "--paper", help="Filter by paper ID"
    ),
    repo: str | None = typer.Option(
        None, "--repo", help="Filter by repository path"
    ),
    experiment: str | None = typer.Option(
        None, "--experiment", help="Filter by experiment ID"
    ),
    limit: int = typer.Option(20, "--limit", help="Max results"),
    offset: int = typer.Option(0, "--offset", help="Pagination offset"),
    output_format: str = typer.Option(
        "console", "--format", help="Output: console, json"
    ),
):
    """List evaluations with optional filters.

    Examples:
        research-engineer evaluate list
        research-engineer evaluate list --paper 2503.12345 --limit 50
    """
    agent = _get_evaluation_agent()
    try:
        out = asyncio.run(
            agent.list_evaluations(
                paper_id=paper,
                repo_path=repo,
                experiment_id=experiment,
                limit=limit,
                offset=offset,
            )
        )
        if output_format == "json":
            typer.echo(out.model_dump_json(indent=2))
        else:
            typer.echo(
                f"\n📋 Evaluations (total: {out.total}, "
                f"showing: {len(out.evaluations)})\n"
            )
            for i, ev in enumerate(out.evaluations, 1):
                typer.echo(
                    f"{i}. {ev.evaluation_id} | "
                    f"experiments={len(ev.experiment_ids)} | "
                    f"{ev.paper_id or 'N/A'}"
                )
            if not out.evaluations:
                typer.echo("No evaluations found.")
        return 0
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        return 1


@evaluate_app.command("get")
def evaluate_get(
    evaluation_id: str = typer.Argument(..., help="Evaluation ID"),
    output_format: str = typer.Option(
        "console", "--format", help="Output: console, json"
    ),
):
    """Get evaluation details by ID.

    Examples:
        research-engineer evaluate get eval_abc123
    """
    agent = _get_evaluation_agent()
    try:
        record = asyncio.run(agent.get_evaluation(evaluation_id))
        if not record:
            typer.echo(
                f"❌ Evaluation not found: {evaluation_id}", err=True
            )
            return 1
        if output_format == "json":
            typer.echo(record.model_dump_json(indent=2))
        else:
            typer.echo(f"\n📊 Evaluation: {record.evaluation_id}")
            typer.echo(
                f"   Experiments: {', '.join(record.experiment_ids)}"
            )
            typer.echo(f"   Paper: {record.paper_id or 'N/A'}")
            typer.echo(f"   Repo: {record.repo_path or 'N/A'}")
            typer.echo(f"   Summary: {record.summary}")
            if record.conclusions:
                typer.echo("   Conclusions:")
                for c in record.conclusions:
                    typer.echo(f"     - {c}")
            if record.memory_ids:
                typer.echo(
                    f"   Memory IDs: {', '.join(record.memory_ids)}"
                )
        return 0
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        return 1


@evaluate_app.command("search")
def evaluate_search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(20, "--limit", help="Max results"),
    output_format: str = typer.Option(
        "console", "--format", help="Output: console, json"
    ),
):
    """Search evaluation history by text.

    Examples:
        research-engineer evaluate search "overfitting"
        research-engineer evaluate search "winner" --limit 10
    """
    agent = _get_evaluation_agent()
    try:
        out = asyncio.run(agent.search_evaluations(query, limit=limit))
        if output_format == "json":
            typer.echo(out.model_dump_json(indent=2))
        else:
            typer.echo(
                f"\n🔍 Search results for '{query}' "
                f"(total: {out.total})\n"
            )
            for i, ev in enumerate(out.evaluations, 1):
                typer.echo(
                    f"{i}. {ev.evaluation_id} | {ev.summary[:60]}"
                )
            if not out.evaluations:
                typer.echo("No evaluations found.")
        return 0
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        return 1


# ---------------------------------------------------------------------------
# Phase 9 - Autonomous Research Loop
# ---------------------------------------------------------------------------


@loop_app.command("run")
def loop_run(
    goal: str = typer.Argument(..., help="High-level research goal"),
    repo: str = typer.Option(..., "--repo", help="Repository path"),
    max_iterations: int = typer.Option(
        5, "--max-iterations", help="Maximum iterations"
    ),
    target_metric: str | None = typer.Option(
        None, "--target-metric", help="Metric name to optimize"
    ),
    target_value: float | None = typer.Option(
        None, "--target-value", help="Target metric value"
    ),
    higher_is_better: bool = typer.Option(
        False, "--higher-is-better", help="Higher metric is better"
    ),
    budget_hours: float | None = typer.Option(
        None, "--budget-hours", help="GPU-hour budget"
    ),
    budget_cost: float | None = typer.Option(
        None, "--budget-cost", help="USD budget"
    ),
    approval: bool = typer.Option(
        False, "--approval", help="Enable human-approval mode"
    ),
    dry_run: bool = typer.Option(
        True, "--dry-run", help="Dry-run experiments (no execution)"
    ),
    skip_literature: bool = typer.Option(
        True, "--skip-literature",
        help="Skip literature after first iteration",
    ),
    output_dir: str = typer.Option(
        "output/loops", "--output-dir", help="Output directory"
    ),
    output_format: str = typer.Option(
        "console", "--format", help="Output: console, json"
    ),
):
    """Start an autonomous research loop.

    Examples:
        research-engineer loop run "Improve training stability" --repo ./my_repo
        research-engineer loop run "Reduce loss" --repo ./repo --max-iterations 3 --dry-run
        research-engineer loop run "Boost accuracy" --repo ./repo --target-metric accuracy --target-value 0.95 --higher-is-better
    """
    from research_engineer.models.loop import LoopConfig

    agent = _get_loop_agent()
    config = LoopConfig(
        goal=goal,
        repo_path=repo,
        max_iterations=max_iterations,
        target_metric_name=target_metric,
        target_metric_value=target_value,
        higher_is_better=higher_is_better,
        budget_hours=budget_hours,
        budget_cost=budget_cost,
        approval_mode=approval,
        dry_run=dry_run,
        skip_literature_after_first=skip_literature,
        output_dir=output_dir,
    )
    try:
        result = asyncio.run(agent.run(goal, repo, config=config))
        if output_format == "json":
            typer.echo(result.model_dump_json(indent=2))
        else:
            typer.echo(f"\n🔄 Research Loop: {result.loop_id}")
            typer.echo(f"   Goal: {result.goal}")
            typer.echo(f"   Status: {result.status.value}")
            typer.echo(f"   Iterations: {result.iteration_count}")
            if result.best_metric_value is not None:
                typer.echo(
                    f"   Best metric: {result.best_metric_value:.6f}"
                )
            if result.stopping_condition:
                typer.echo(
                    f"   Stopped: {result.stopping_condition.value}"
                )
            if result.memory_ids:
                typer.echo(f"   Memory IDs: {len(result.memory_ids)}")
            typer.echo(f"   Time: {result.processing_time_seconds}s")
            if result.generated_files:
                typer.echo("\n   Files generated:")
                for f in result.generated_files:
                    typer.echo(f"     - {f}")
        return 0
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        return 1


@loop_app.command("list")
def loop_list(
    status: str | None = typer.Option(
        None, "--status", help="Filter by status"
    ),
    limit: int = typer.Option(20, "--limit", help="Max results"),
    offset: int = typer.Option(0, "--offset", help="Pagination offset"),
    output_format: str = typer.Option(
        "console", "--format", help="Output: console, json"
    ),
):
    """List research loops.

    Examples:
        research-engineer loop list
        research-engineer loop list --status stopped --limit 10
    """
    from research_engineer.models.loop import LoopStatus

    agent = _get_loop_agent()
    status_enum: LoopStatus | None = None
    if status:
        try:
            status_enum = LoopStatus(status)
        except ValueError:
            typer.echo(f"❌ Invalid status: {status}", err=True)
            return 1
    try:
        out = asyncio.run(
            agent.list_loops(status=status_enum, limit=limit, offset=offset)
        )
        if output_format == "json":
            typer.echo(out.model_dump_json(indent=2))
        else:
            typer.echo(
                f"\n🔄 Loops (total: {out.total}, "
                f"showing: {len(out.loops)})\n"
            )
            for i, lp in enumerate(out.loops, 1):
                typer.echo(
                    f"{i}. {lp.loop_id} | {lp.status.value} | "
                    f"iters={lp.iteration_count} | {lp.goal[:50]}"
                )
            if not out.loops:
                typer.echo("No loops found.")
        return 0
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        return 1


@loop_app.command("get")
def loop_get(
    loop_id: str = typer.Argument(..., help="Loop ID"),
    output_format: str = typer.Option(
        "console", "--format", help="Output: console, json"
    ),
):
    """Get loop details by ID.

    Examples:
        research-engineer loop get loop_abc123
    """
    agent = _get_loop_agent()
    try:
        record = asyncio.run(agent.get_loop(loop_id))
        if not record:
            typer.echo(f"❌ Loop not found: {loop_id}", err=True)
            return 1
        if output_format == "json":
            typer.echo(record.model_dump_json(indent=2))
        else:
            typer.echo(f"\n🔄 Loop: {record.loop_id}")
            typer.echo(f"   Goal: {record.goal}")
            typer.echo(f"   Status: {record.status.value}")
            typer.echo(f"   Iterations: {record.iteration_count}")
            if record.best_metric_value is not None:
                typer.echo(
                    f"   Best metric: {record.best_metric_value:.6f}"
                )
            if record.stopping_condition:
                typer.echo(
                    f"   Stopped: {record.stopping_condition.value}"
                )
            if record.memory_ids:
                typer.echo(
                    f"   Memory IDs: {', '.join(record.memory_ids)}"
                )
        return 0
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        return 1


@loop_app.command("iterations")
def loop_iterations(
    loop_id: str = typer.Argument(..., help="Loop ID"),
    limit: int = typer.Option(50, "--limit", help="Max results"),
    output_format: str = typer.Option(
        "console", "--format", help="Output: console, json"
    ),
):
    """List iterations of a loop.

    Examples:
        research-engineer loop iterations loop_abc123
        research-engineer loop iterations loop_abc123 --limit 10
    """
    agent = _get_loop_agent()
    try:
        out = asyncio.run(agent.get_iterations(loop_id, limit=limit))
        if output_format == "json":
            typer.echo(out.model_dump_json(indent=2))
        else:
            typer.echo(
                f"\n📋 Iterations (total: {out.total}, "
                f"showing: {len(out.iterations)})\n"
            )
            for it in out.iterations:
                metric = (
                    f"{it.primary_metric_value:.4f}"
                    if it.primary_metric_value is not None
                    else "N/A"
                )
                typer.echo(
                    f"  #{it.iteration_number} {it.iteration_id} | "
                    f"{it.phase.value} | {it.status.value} | "
                    f"metric={metric}"
                )
            if not out.iterations:
                typer.echo("No iterations found.")
        return 0
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        return 1


@loop_app.command("iteration")
def loop_iteration(
    iteration_id: str = typer.Argument(..., help="Iteration ID"),
    output_format: str = typer.Option(
        "console", "--format", help="Output: console, json"
    ),
):
    """Get iteration details by ID.

    Examples:
        research-engineer loop iteration iter_abc123
    """
    agent = _get_loop_agent()
    try:
        record = asyncio.run(agent.get_iteration(iteration_id))
        if not record:
            typer.echo(
                f"❌ Iteration not found: {iteration_id}", err=True
            )
            return 1
        if output_format == "json":
            typer.echo(record.model_dump_json(indent=2))
        else:
            typer.echo(f"\n📋 Iteration: {record.iteration_id}")
            typer.echo(f"   Loop: {record.loop_id}")
            typer.echo(f"   Number: {record.iteration_number}")
            typer.echo(f"   Phase: {record.phase.value}")
            typer.echo(f"   Status: {record.status.value}")
            if record.paper_id:
                typer.echo(f"   Paper: {record.paper_id}")
            if record.plan_id:
                typer.echo(f"   Plan: {record.plan_id}")
            if record.experiment_id:
                typer.echo(f"   Experiment: {record.experiment_id}")
            if record.evaluation_id:
                typer.echo(f"   Evaluation: {record.evaluation_id}")
            if record.primary_metric_value is not None:
                typer.echo(
                    f"   Metric: {record.primary_metric_value:.6f}"
                )
            if record.error:
                typer.echo(f"   Error: {record.error}")
        return 0
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        return 1


@loop_app.command("search")
def loop_search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(20, "--limit", help="Max results"),
    output_format: str = typer.Option(
        "console", "--format", help="Output: console, json"
    ),
):
    """Search loop history by text.

    Examples:
        research-engineer loop search "training stability"
        research-engineer loop search "overfitting" --limit 10
    """
    agent = _get_loop_agent()
    try:
        out = asyncio.run(agent.search_loops(query, limit=limit))
        if output_format == "json":
            typer.echo(out.model_dump_json(indent=2))
        else:
            typer.echo(
                f"\n🔍 Search results for '{query}' "
                f"(total: {out.total})\n"
            )
            for i, lp in enumerate(out.loops, 1):
                typer.echo(
                    f"{i}. {lp.loop_id} | {lp.goal[:60]}"
                )
            if not out.loops:
                typer.echo("No loops found.")
        return 0
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        return 1


@loop_app.command("report")
def loop_report(
    loop_id: str = typer.Argument(..., help="Loop ID"),
    output_dir: str | None = typer.Option(
        None, "--output-dir", help="Output directory override"
    ),
    output_format: str = typer.Option(
        "console", "--format", help="Output: console, json"
    ),
):
    """Generate a research report for a loop.

    Examples:
        research-engineer loop report loop_abc123
        research-engineer loop report loop_abc123 --output-dir ./reports
    """
    agent = _get_loop_agent()
    try:
        report = asyncio.run(
            agent.generate_report(loop_id, output_dir=output_dir)
        )
        if output_format == "json":
            typer.echo(report.model_dump_json(indent=2))
        else:
            typer.echo(f"\n📄 Report generated for {loop_id}")
            typer.echo(f"   Report: {report.report_path}")
            typer.echo(f"   JSON: {report.json_path}")
            typer.echo(f"   Summary: {report.summary}")
        return 0
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        return 1


# ---------------------------------------------------------------------------
# Phase 11 - Terminal-first autonomous coding agent (`task` command)
# ---------------------------------------------------------------------------


@app.command()
def task(
    goal: str = typer.Argument(
        ...,
        help='Natural-language coding goal, e.g. "Add Grouped Query Attention"',
    ),
    repo: str = typer.Option(
        ".",
        "--repo",
        help="Path to target repository",
    ),
    paper: str | None = typer.Option(
        None,
        "--paper",
        help="Optional paper ID/URL/PDF for research-grounded tasks",
    ),
    run_tests: bool = typer.Option(
        False,
        "--run-tests",
        help="Execute the test suite after patching",
    ),
    test_command: str = typer.Option(
        "uv run pytest",
        "--test-command",
        help="Test command to run when --run-tests is set",
    ),
    dry_run: bool = typer.Option(
        True,
        "--dry-run",
        help="Generate patches without applying them",
    ),
    stream: bool = typer.Option(
        True,
        "--stream/--no-stream",
        help="Stream LLM tokens to stdout during planning",
    ),
    output_dir: str = typer.Option(
        "output/tasks",
        "--output-dir",
        help="Directory to save task artifacts",
    ),
    output_format: str = typer.Option(
        "console",
        "--format",
        help="Output: console, json, markdown",
    ),
    delegate: bool = typer.Option(
        False,
        "--delegate",
        help="Use multi-agent delegation pipeline (Phase 13): "
        "research → architect → code → review → test → repair",
    ),
    max_repairs: int = typer.Option(
        2,
        "--max-repairs",
        help="Max review/test repair iterations in delegation mode",
    ),
):
    """Run a terminal-first autonomous coding turn.

    Analyzes the repository, creates an implementation plan, generates a
    patch, shows the diff, and optionally executes tests.

    With --delegate, uses the multi-agent delegation pipeline: each phase
    is handled by a specialized agent (Architect, Coder, Reviewer, Tester)
    with automatic repair loops on review/test failures.

    Examples:
        research-engineer task "Add RMSNorm to the model"
        research-engineer task "Refactor data loader" --repo ./my_repo
        research-engineer task "Fix gradient clipping" --run-tests
        research-engineer task "Implement LoRA" --paper 2503.12345 --repo .
        research-engineer task "Add EMA checkpoint support" --delegate --max-repairs 3
    """
    agent = _get_task_agent()
    config = TaskConfig(
        goal=goal,
        repo_path=repo,
        paper_input=paper,
        run_tests=run_tests,
        test_command=test_command,
        dry_run=dry_run,
        stream=stream,
        output_dir=output_dir,
        delegate=delegate,
        max_repair_iterations=max_repairs,
    )
    try:
        result = asyncio.run(agent.run(goal, repo, config=config))

        if output_format == "json":
            typer.echo(result.model_dump_json(indent=2))
        elif output_format == "markdown":
            typer.echo(f"# Task: {result.task_id}\n")
            typer.echo(f"**Goal**: {result.goal}\n")
            typer.echo(f"**Status**: {result.status.value}\n")
            typer.echo(f"**Patches**: {result.patches_generated}\n")
            typer.echo(f"**Time**: {result.processing_time_seconds}s\n")
            typer.echo("## Steps\n")
            for s in result.steps:
                typer.echo(
                    f"- {s.step_type.value}: {s.status.value} "
                    f"({s.duration_seconds}s) {s.summary}"
                )
            if result.diff:
                typer.echo("\n## Diff\n\n```diff\n")
                typer.echo(result.diff)
                typer.echo("\n```")
            if result.test_exit_code is not None:
                typer.echo(
                    f"\n## Tests (exit {result.test_exit_code})\n\n"
                    f"```\n{result.test_stdout}\n```"
                )
        else:
            typer.echo(f"\n🎯 Task: {result.task_id}")
            typer.echo(f"   Goal: {result.goal}")
            typer.echo(f"   Repo: {result.repo_path}")
            typer.echo(f"   Status: {result.status.value}")
            if result.delegated:
                typer.echo(
                    f"   🤖 Delegated: {result.repair_iterations} repair iteration(s)"
                )
            typer.echo(f"   Patches: {result.patches_generated}")
            typer.echo(f"   Time: {result.processing_time_seconds}s")
            typer.echo("\n📋 Steps:")
            for s in result.steps:
                mark = "✅" if s.status == TaskStatus.COMPLETED else "❌"
                typer.echo(
                    f"   {mark} {s.step_type.value}: {s.summary}"
                )
            if result.review_issues:
                typer.echo(f"\n🔍 Review issues ({len(result.review_issues)}):")
                for issue in result.review_issues[:5]:
                    typer.echo(f"   - {issue}")
            if result.test_failures:
                typer.echo(f"\n🧪 Test failures ({len(result.test_failures)}):")
                for fail in result.test_failures[:5]:
                    typer.echo(f"   - {fail}")
            if result.diff:
                typer.echo("\n📝 Diff:")
                typer.echo(result.diff[:4000])
            if result.test_exit_code is not None:
                typer.echo(
                    f"\n🧪 Tests: exit={result.test_exit_code}"
                )
                if result.test_stdout:
                    typer.echo(result.test_stdout[:2000])
            if result.generated_files:
                typer.echo("\n📁 Files:")
                for f in result.generated_files:
                    typer.echo(f"   - {f}")
            if result.error:
                typer.echo(f"\n❌ Error: {result.error}", err=True)
        return 0
    except Exception as e:
        typer.echo(f"❌ Error running task: {e}", err=True)
        return 1


        return 0
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        return 1


# ---------------------------------------------------------------------------
# Phase 10: Provider-agnostic LLM layer CLI
# ---------------------------------------------------------------------------


@llm_app.command("status")
def llm_status(
    output_format: str = typer.Option(
        "console", "--format", help="Output: console, json"
    ),
):
    """Show the active LLM provider/model configuration per agent."""
    from research_engineer.llm import get_factory, get_router

    factory = get_factory()
    router = get_router(factory)
    agents = router.KNOWN_AGENTS
    rows = []
    for name in agents:
        prov_name = router.provider_name_for(name)
        model = router.model_for(name)
        rows.append({"agent": name, "provider": prov_name, "model": model})
    config = factory.config
    payload = {
        "default_provider": config.get("default_provider"),
        "default_model": config.get("default_model"),
        "configured_providers": list((config.get("providers") or {}).keys()),
        "agent_routing": rows,
    }
    if output_format == "json":
        typer.echo(json.dumps(payload, indent=2, default=str))
    else:
        typer.echo("LLM layer status")
        typer.echo(f"  default provider: {payload['default_provider']}")
        typer.echo(f"  default model:    {payload['default_model']}")
        typer.echo(f"  providers:        {', '.join(payload['configured_providers']) or '(none)'}")
        typer.echo("\nPer-agent routing:")
        for r in rows:
            typer.echo(f"  {r['agent']:<24} -> {r['provider']} / {r['model']}")
    return 0


@llm_app.command("config")
def llm_config(
    config_path: str = typer.Option(
        "", "--config", help="Path to llm_config.yaml to load"
    ),
):
    """Dump the resolved LLM configuration as JSON."""
    from research_engineer.llm import load_config, reset_factory

    cfg = load_config(config_path or None) if config_path else None
    if cfg is None:
        reset_factory()
        from research_engineer.llm import get_factory
        cfg = get_factory().config
    typer.echo(json.dumps(cfg, indent=2, default=str))
    return 0


# ---------------------------------------------------------------------------
# Phase 15 - Autonomous Research Workflows
# ---------------------------------------------------------------------------


def _get_research_orchestrator():
    """Get or create the research orchestrator instance."""
    from research_engineer.agents import ResearchOrchestrator

    return ResearchOrchestrator(
        literature_agent=_get_literature_agent(),
    )


@app.command()
def research(
    goal: str = typer.Argument(
        ...,
        help='Research goal, e.g. "Design a more efficient diffusion transformer"',
    ),
    repo: str = typer.Option(
        ".",
        "--repo",
        help="Repository path for experiment execution",
    ),
    max_papers: int = typer.Option(
        20,
        "--max-papers",
        help="Max papers to discover in literature review",
    ),
    max_hypotheses: int = typer.Option(
        5,
        "--max-hypotheses",
        help="Max hypotheses to generate",
    ),
    dry_run: bool = typer.Option(
        True,
        "--dry-run",
        help="Dry-run experiments (don't execute commands)",
    ),
    timeout: int = typer.Option(
        3600,
        "--timeout",
        help="Experiment timeout in seconds",
    ),
    output_dir: str = typer.Option(
        "output/research",
        "--output-dir",
        help="Directory to save research artifacts",
    ),
    output_format: str = typer.Option(
        "console",
        "--format",
        help="Output: console, json, markdown",
    ),
):
    """Run an autonomous research workflow.

    Transforms a research goal into a complete research workflow:
    literature review, knowledge synthesis, hypothesis generation,
    experiment planning, experiment execution, result analysis, and
    report generation.

    Examples:
        research-engineer research "Design a more efficient diffusion transformer"
        research-engineer research "Improve attention efficiency" --max-papers 30
        research-engineer research "Novel loss function" --no-dry-run --repo ./my_repo
    """
    from research_engineer.agents import ResearchConfig

    orchestrator = _get_research_orchestrator()
    config = ResearchConfig(
        max_papers=max_papers,
        max_hypotheses=max_hypotheses,
        dry_run_experiments=dry_run,
        experiment_timeout=timeout,
        output_dir=output_dir,
    )
    try:
        result = asyncio.run(orchestrator.run(goal, repo, config=config))

        if output_format == "json":
            typer.echo(result.model_dump_json(indent=2, default=str))
        elif output_format == "markdown":
            if result.final_report:
                typer.echo(result.final_report)
            else:
                typer.echo(f"# Research: {result.research_goal}\n\nNo report generated.")
        else:
            typer.echo(f"\n🔬 Research Workflow: {result.workflow_id}")
            typer.echo(f"   Goal: {result.research_goal}")
            typer.echo(f"   Status: {result.status.value}")
            typer.echo(f"   Papers found: {result.papers_found}")
            typer.echo(f"   Hypotheses: {result.hypotheses_generated}")
            typer.echo(f"   Experiments: {result.experiments_run}")
            typer.echo(f"   Time: {result.processing_time_seconds}s")
            typer.echo("\n📋 Stages:")
            for s in result.stages:
                mark = "✅" if s.status.value == "completed" else "⏭️" if s.status.value == "skipped" else "❌"
                typer.echo(
                    f"   {mark} {s.stage_type.value}: {s.summary} ({s.duration_seconds}s)"
                )
            if result.report_path:
                typer.echo(f"\n📄 Report: {result.report_path}")
            if result.final_report:
                typer.echo(f"\n📝 Report preview ({len(result.final_report)} chars):")
                typer.echo(result.final_report[:1000])
            if result.error:
                typer.echo(f"\n❌ Error: {result.error}", err=True)
        return 0
    except Exception as e:
        typer.echo(f"❌ Error running research: {e}", err=True)
        return 1


if __name__ == "__main__":
    app()
