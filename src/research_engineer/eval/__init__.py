"""E4 - Agent Evaluation Harness.

Execute predefined agent tasks through the real
:class:`~research_engineer.runtime.runtime.AgentRuntime` and quantitatively
measure success, quality, efficiency, tool usage, cost, latency, failure
recovery, human interventions, and termination reasons.

Typical use::

    from research_engineer.eval import EvalRunner, load_suite
    from research_engineer.eval.scripted import ScriptedAgentFactory

    suite = load_suite("evals/sample_suite.yaml")
    runner = EvalRunner(ScriptedAgentFactory(), label="v2.1")
    report = await runner.run_suite(suite)
"""

from research_engineer.eval.graders import (
    DETERMINISTIC_GRADERS,
    BudgetGrader,
    CompositeGrader,
    Grader,
    GradingRequest,
    LLMPromptGrader,
    MaxStepsGrader,
    OutputContainsGrader,
    OutputEqualsGrader,
    OutputJSONFieldGrader,
    OutputRegexGrader,
    RecoveryGrader,
    TerminationGrader,
    ToolUsageGrader,
    build_grader,
)
from research_engineer.eval.metrics import (
    CaseDiff,
    RegressionComparison,
    aggregate,
    build_report,
    compare_reports,
    load_report,
    save_report,
)
from research_engineer.eval.models import (
    CaseMetrics,
    EvalMetric,
    EvalReport,
    EvalResult,
    EvalStatus,
    EvalSuite,
    EvalTask,
    GraderResult,
    SuccessCriterion,
    SuiteMetrics,
)
from research_engineer.eval.runner import (
    AgentFactory,
    EvalRunner,
    load_suite,
    save_suite,
)
from research_engineer.eval.scripted import (
    ScriptedAgentFactory,
    resolve_agent_factory,
)

__all__ = [
    "DETERMINISTIC_GRADERS",
    "AgentFactory",
    "BudgetGrader",
    "CaseDiff",
    "CaseMetrics",
    "CompositeGrader",
    "EvalMetric",
    "EvalReport",
    "EvalResult",
    "EvalRunner",
    "EvalStatus",
    "EvalSuite",
    "EvalTask",
    "Grader",
    "GraderResult",
    "GradingRequest",
    "LLMPromptGrader",
    "MaxStepsGrader",
    "OutputContainsGrader",
    "OutputEqualsGrader",
    "OutputJSONFieldGrader",
    "OutputRegexGrader",
    "RecoveryGrader",
    "RegressionComparison",
    "ScriptedAgentFactory",
    "SuccessCriterion",
    "SuiteMetrics",
    "TerminationGrader",
    "ToolUsageGrader",
    "aggregate",
    "build_grader",
    "build_report",
    "compare_reports",
    "load_report",
    "load_suite",
    "resolve_agent_factory",
    "save_report",
    "save_suite",
]
