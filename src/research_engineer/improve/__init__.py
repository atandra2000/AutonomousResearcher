"""E8 - Continuous Agent Improvement.

Controlled loop converting agent failures and E4 evaluation results into
measurable improvement candidates::

    Runs/Telemetry -> Failure & Pattern Mining -> Improvement Candidate
        -> E4 Evaluation Suite -> Baseline vs Candidate -> Regression Gate
            -> reject | recommend | approve/canary

Safety boundaries (mandatory): no autonomous production self-modification;
candidates are data/configuration artifacts only; hard safety metrics reject
on any regression; E3/E5 remain authoritative; promotion requires explicit
approval and is always reversible; every candidate is reproducible from its
recorded configuration hashes.
"""

from research_engineer.improve.gate import evaluate_gate, snapshot_from_report
from research_engineer.improve.mining import (
    FailurePattern,
    MiningConfig,
    extract_features,
    mine_patterns,
    mine_report,
)
from research_engineer.improve.models import (
    HARD_SAFETY_METRICS,
    Baseline,
    CandidateComponent,
    CandidateEvaluation,
    CandidateStatus,
    GateVerdict,
    ImprovementCandidate,
    ImprovementProposal,
    MetricSnapshot,
    PatternKind,
    PromotionAction,
    PromotionDecision,
    RegressionGate,
    build_default_gate,
    canonical_json,
    content_hash,
)
from research_engineer.improve.pipeline import (
    LIFECYCLE_TRANSITIONS,
    ApprovalRequiredError,
    EvalFn,
    IllegalTransitionError,
    ImprovementPipeline,
    ImprovementStore,
    transition_allowed,
)
from research_engineer.improve.proposals import (
    InvalidProposalError,
    ProposalSource,
    RuleBasedProposalSource,
    apply_candidate_changes,
    build_candidate,
    candidate_eval_runner_builder,
    validate_changes,
)

__all__ = [
    "ApprovalRequiredError",
    "Baseline",
    "CandidateComponent",
    "CandidateEvaluation",
    "CandidateStatus",
    "EvalFn",
    "FailurePattern",
    "GateVerdict",
    "HARD_SAFETY_METRICS",
    "IllegalTransitionError",
    "ImprovementCandidate",
    "ImprovementPipeline",
    "ImprovementProposal",
    "ImprovementStore",
    "InvalidProposalError",
    "LIFECYCLE_TRANSITIONS",
    "MetricSnapshot",
    "MiningConfig",
    "PatternKind",
    "PromotionAction",
    "PromotionDecision",
    "ProposalSource",
    "RegressionGate",
    "RuleBasedProposalSource",
    "apply_candidate_changes",
    "build_candidate",
    "build_default_gate",
    "candidate_eval_runner_builder",
    "canonical_json",
    "content_hash",
    "evaluate_gate",
    "extract_features",
    "mine_patterns",
    "mine_report",
    "snapshot_from_report",
    "transition_allowed",
    "validate_changes",
]
