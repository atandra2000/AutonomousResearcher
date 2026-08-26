"""E5 - Autonomy & Safety Controls.

A deterministic policy/control layer for the E1
:class:`~research_engineer.runtime.runtime.AgentRuntime`. It detects
unsafe, wasteful, or non-progressing autonomous behaviour and returns
controlled decisions (CONTINUE / REPLAN / PAUSE_FOR_APPROVAL / TERMINATE)
with machine-readable reasons.

Hard constraint: the controls are deterministic and authoritative. An LLM
evaluator may provide supplementary reasoning (:attr:`ControlDecision.advisory_note`)
but can never override hard limits, deny policies, sandbox restrictions,
or mandatory approval requirements.

Public surface::

    from research_engineer.safety import (
        AutonomyPolicy,
        SafetyPolicy,
        RuleBasedSafetyPolicy,
        SafetyController,
        ControlAction,
        ControlDecision,
        ControlTrigger,
        ProgressSignal,
        RiskAssessment,
        SafetyState,
        PauseRequest,
        CallbackPauseGate,
    )
"""

from research_engineer.safety.approval import (
    CallbackPauseGate,
    PauseApprovalCallback,
    PauseApprovalGate,
    PauseRequest,
)
from research_engineer.safety.controller import (
    SAFETY_STATE_METADATA_KEY,
    SafetyController,
)
from research_engineer.safety.detectors import (
    DuplicateToolCallDetector,
    LoopDetector,
    normalize_args,
    step_signature,
    tool_call_key,
)
from research_engineer.safety.models import (
    ControlAction,
    ControlDecision,
    ControlTrigger,
    DecisionRecord,
    ProgressSignal,
    RiskAssessment,
    RiskLevel,
    SafetyState,
    ToolCallRecord,
    continue_decision,
    terminate_decision,
)
from research_engineer.safety.policies import (
    AutonomyPolicy,
    RuleBasedSafetyPolicy,
    SafetyPolicy,
)

__all__ = [
    "SAFETY_STATE_METADATA_KEY",
    "AutonomyPolicy",
    "CallbackPauseGate",
    "ControlAction",
    "ControlDecision",
    "ControlTrigger",
    "DecisionRecord",
    "DuplicateToolCallDetector",
    "LoopDetector",
    "PauseApprovalCallback",
    "PauseApprovalGate",
    "PauseRequest",
    "ProgressSignal",
    "RiskAssessment",
    "RiskLevel",
    "RuleBasedSafetyPolicy",
    "SafetyController",
    "SafetyPolicy",
    "SafetyState",
    "ToolCallRecord",
    "continue_decision",
    "normalize_args",
    "step_signature",
    "terminate_decision",
    "tool_call_key",
]
