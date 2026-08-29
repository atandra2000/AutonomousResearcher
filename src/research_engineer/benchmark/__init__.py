"""Benchmark machinery — P1/P2 graded suites over the production runtime path.

Public surface::

    from research_engineer.benchmark import (
        AgentFactoryRegistry, DEFAULT_AGENT_KIND,   # agent factories
        BenchmarkRunner, BenchmarkReport,           # P1 runner + report
        build_default_safety_chain,                 # E3+E5 enforcement
    )

Benchmarks execute cases directly through the E1 ``AgentRuntime`` behind
the E3 gateway + E5 safety chain — the same path autonomous CLI runs use —
and grade the persisted run payloads with the E4 grader machinery.
"""

from research_engineer.benchmark.agents import (
    DEFAULT_AGENT_KIND,
    AgentFactoryRegistry,
)
from research_engineer.benchmark.benchmark import load_benchmark_suite
from research_engineer.benchmark.benchmark_runner import (
    FAILURE_TAXONOMY,
    BenchmarkReport,
    BenchmarkRunner,
    CaseOutcome,
    classify_failure,
    render_markdown,
)
from research_engineer.benchmark.safety import build_default_safety_chain

__all__ = [
    "DEFAULT_AGENT_KIND",
    "FAILURE_TAXONOMY",
    "AgentFactoryRegistry",
    "BenchmarkReport",
    "BenchmarkRunner",
    "CaseOutcome",
    "build_default_safety_chain",
    "classify_failure",
    "load_benchmark_suite",
    "render_markdown",
]

