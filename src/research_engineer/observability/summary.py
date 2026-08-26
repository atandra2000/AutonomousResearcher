"""E6 - Run summary models and the event-driven run aggregator.

The :class:`RunAggregator` is a best-effort :class:`EventSink` that consumes
events already flowing through any :class:`EventBus` and maintains an
in-memory index keyed by ``run_id`` (falling back to ``execution_id``) so a
single run can be reconstructed — steps, LLM calls, tool calls, safety
decisions, approvals, checkpoints, and termination — without manually
parsing raw events.

Concurrent runs are isolated by construction: each ingested event updates
exactly one run's record, selected by its correlation identifiers.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LLMCallSummary(BaseModel):
    """One LLM call observed for a run."""

    ts: str | None = None
    model: str = ""
    provider: str = ""
    prompt_hash: str | None = None
    finish_reason: str | None = None
    attempt: int = 1
    latency_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    trace_id: str | None = None
    step_id: str | None = None


class ToolCallSummary(BaseModel):
    """One gateway-mediated tool invocation."""

    call_id: str | None = None
    tool_name: str = ""
    agent_name: str = ""
    status: str | None = None
    failure_kind: str | None = None
    duration_seconds: float = 0.0
    error: str | None = None
    ts: str | None = None
    trace_id: str | None = None
    step_id: str | None = None


class SafetyDecisionSummary(BaseModel):
    """One deterministic safety/autonomy decision."""

    step: int = 0
    action: str = ""
    trigger: str = ""
    reason_code: str | None = None
    mandatory: bool = False
    warnings: list[str] = Field(default_factory=list)
    ts: str | None = None
    trace_id: str | None = None


class CheckpointSummary(BaseModel):
    """A successful or failed checkpoint write."""

    step: int = 0
    failed: bool = False
    error: str | None = None
    ts: str | None = None


class ApprovalSummary(BaseModel):
    """A human-intervention / approval touchpoint."""

    kind: str = ""  # e.g. "tool_approval_requested"
    tool_name: str | None = None
    risk: str | None = None
    ts: str | None = None


class StepRecord(BaseModel):
    """Runtime lifecycle entry for one loop step."""

    step: int
    step_id: str | None = None
    score: float | None = None
    duration_seconds: float | None = None
    error: str | None = None
    ts: str | None = None


class RunSummary(BaseModel):
    """Full reconstruction of one autonomous run from its event stream."""

    run_id: str = ""
    execution_id: str = ""
    goal: str | None = None
    state: str | None = None
    termination: str | None = None
    termination_reason: str | None = None
    best_score: float | None = None
    started_ts: str | None = None
    ended_ts: str | None = None
    duration_seconds: float | None = None
    cost_usd: float = 0.0
    total_tokens: int = 0
    step_count: int = 0
    llm_call_count: int = 0
    tool_call_count: int = 0
    tool_failure_count: int = 0
    safety_decision_count: int = 0
    human_intervention_count: int = 0
    checkpoint_count: int = 0
    checkpoint_failure_count: int = 0
    resume_count: int = 0
    error_count: int = 0
    steps: list[StepRecord] = Field(default_factory=list)
    llm_calls: list[LLMCallSummary] = Field(default_factory=list)
    tool_calls: list[ToolCallSummary] = Field(default_factory=list)
    safety_decisions: list[SafetyDecisionSummary] = Field(default_factory=list)
    checkpoints: list[CheckpointSummary] = Field(default_factory=list)
    approvals: list[ApprovalSummary] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    trace_ids: set[str] = Field(default_factory=set)

    def to_json(self) -> str:
        """Serialize the summary."""
        return str(self.model_dump_json())


def _str(event: dict[str, Any], key: str) -> str:
    value = event.get(key)
    return value if isinstance(value, str) else ""


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def _num_or_none(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


class RunAggregator:
    """Best-effort sink/index that reconstructs runs from raw events."""

    def __init__(self) -> None:
        self._runs: dict[str, RunSummary] = {}

    # -- EventSink protocol -------------------------------------------------
    def emit(self, event: dict[str, Any]) -> None:
        try:
            self._ingest(dict(event))
        except Exception:  # noqa: BLE001 - observability is best-effort
            pass

    def close(self) -> None:
        return None

    def __repr__(self) -> str:
        return f"<RunAggregator runs={len(self._runs)}>"

    # -- Query API -----------------------------------------------------------
    def get_summary(self, run_id: str) -> RunSummary | None:
        """Return the summary for ``run_id`` (or ``None`` if unknown)."""
        return self._runs.get(run_id)

    def all_run_ids(self) -> list[str]:
        """Every run currently indexed."""
        return sorted(self._runs)

    def summaries(self) -> list[RunSummary]:
        """All indexed run summaries."""
        return [self._runs[k] for k in sorted(self._runs)]

    # -- Internals -----------------------------------------------------------
    def _summary_for(
        self,
        *,
        run_id: str = "",
        execution_id: str = "",
        goal: Any = None,
        state: Any = None,
        trace_id: str | None = None,
    ) -> tuple[str, RunSummary]:
        key = run_id or execution_id or "unknown"
        summary = self._runs.get(key)
        if summary is None:
            summary = RunSummary(run_id=run_id, execution_id=execution_id)
            self._runs[key] = summary
        if run_id and not summary.run_id:
            summary.run_id = run_id
        if execution_id and not summary.execution_id:
            summary.execution_id = execution_id
        if isinstance(goal, str) and goal:
            summary.goal = goal
        if isinstance(state, str) and state:
            summary.state = state
        if trace_id:
            summary.trace_ids.add(trace_id)
        return key, summary

    def _ingest(self, event: dict[str, Any]) -> None:
        kind = event.get("kind")
        sub = event.get("event")
        ts = event.get("ts") if isinstance(event.get("ts"), str) else None

        _, s = self._summary_for(
            run_id=_str(event, "run_id"),
            execution_id=_str(event, "execution_id"),
            goal=event.get("goal"),
            state=event.get("state"),
            trace_id=_str(event, "trace_id") or None,
        )

        if kind == "agent_runtime":
            self._ingest_runtime(sub, event, s, ts)
        elif kind == "llm_call":
            self._ingest_llm(event, s, ts)
        elif kind == "tool_gateway":
            self._ingest_gateway(sub, event, s, ts)



    def _ingest_runtime(  # noqa: C901 - explicit per-event handling
        self, sub: Any, event: dict[str, Any], s: RunSummary, ts: str | None
    ) -> None:
        if sub == "start":
            s.started_ts = ts
        elif sub == "end":
            s.ended_ts = ts
            dur = _num_or_none(event.get("duration_seconds"))
            if dur is not None:
                s.duration_seconds = dur
        elif sub == "step":
            step_no = _int_or_none(event.get("step_number"))
            if step_no is None:
                raw = event.get("step")
                step_no = raw if isinstance(raw, int) else None
            if step_no is None:
                return
            s.step_count = max(s.step_count, step_no)
            err = event.get("error")
            s.steps.append(
                StepRecord(
                    step=step_no,
                    step_id=_str(event, "step_id") or None,
                    score=_num_or_none(event.get("score")),
                    duration_seconds=_num_or_none(event.get("duration_seconds")),
                    error=getattr(err, "message", None)
                    or (err if isinstance(err, str) else None),
                    ts=ts,
                )
            )
            best = _num_or_none(event.get("best_score"))
            if best is not None:
                s.best_score = best
        elif sub == "evaluation":
            score = _num_or_none(event.get("score"))
            if score is not None and (s.best_score is None or score > s.best_score):
                s.best_score = score
        elif sub == "error":
            err = event.get("error")
            s.error_count += 1
            text = getattr(err, "message", None) or (
                err if isinstance(err, str) else str(err)
            )
            s.errors.append(str(text)[:500])
        elif sub == "terminate":
            term = event.get("termination")
            term_value = getattr(term, "value", None)
            if isinstance(term_value, str):
                s.termination = term_value
            elif _str(event, "termination"):
                s.termination = _str(event, "termination")
            reason = event.get("reason")
            if isinstance(reason, str) and reason:
                s.termination_reason = reason
            elif isinstance(term, str):
                s.termination_reason = term
        elif sub == "resume":
            s.resume_count += 1
        elif sub == "checkpoint":
            s.checkpoint_count += 1
            s.checkpoints.append(
                CheckpointSummary(
                    step=_int_or_none(event.get("step")) or 0, failed=False, ts=ts
                )
            )
        elif sub == "checkpoint_failed":
            s.checkpoint_failure_count += 1
            s.checkpoints.append(
                CheckpointSummary(failed=True, error=_str(event, "error"), ts=ts)
            )
        elif sub == "safety_decision":
            s.safety_decision_count += 1
            warnings = event.get("warnings")
            s.safety_decisions.append(
                SafetyDecisionSummary(
                    step=_int_or_none(event.get("step")) or 0,
                    action=_str(event, "action"),
                    trigger=_str(event, "trigger"),
                    reason_code=_str(event, "reason_code") or None,
                    mandatory=bool(event.get("mandatory", False)),
                    warnings=[str(w) for w in warnings]
                    if isinstance(warnings, list)
                    else [],
                    ts=ts,
                    trace_id=_str(event, "trace_id") or None,
                )
            )

    def _ingest_llm(self, event: dict[str, Any], s: RunSummary, ts: str | None) -> None:
        tokens = event.get("tokens") or {}
        pt = tokens.get("prompt") if isinstance(tokens, dict) else None
        ct = tokens.get("completion") if isinstance(tokens, dict) else None
        tt = tokens.get("total") if isinstance(tokens, dict) else None
        try:
            cost = float(event.get("cost_usd") or 0.0)
        except (TypeError, ValueError):
            cost = 0.0
        try:
            latency = float(event.get("latency_seconds") or 0.0)
        except (TypeError, ValueError):
            latency = 0.0
        s.llm_calls.append(
            LLMCallSummary(
                ts=ts,
                model=_str(event, "model"),
                provider=_str(event, "provider"),
                prompt_hash=_str(event, "prompt_hash") or None,
                finish_reason=_str(event, "finish_reason") or None,
                attempt=int(event.get("attempt") or 1),
                latency_seconds=latency,
                prompt_tokens=int(pt or 0),
                completion_tokens=int(ct or 0),
                total_tokens=int(tt or 0),
                cost_usd=cost,
                trace_id=_str(event, "trace_id") or None,
                step_id=_str(event, "step_id") or None,
            )
        )
        s.llm_call_count += 1
        s.total_tokens += int(tt or ((pt or 0) + (ct or 0)))
        s.cost_usd = round(s.cost_usd + cost, 9)

    def _ingest_gateway(
        self, sub: Any, event: dict[str, Any], s: RunSummary, ts: str | None
    ) -> None:
        if sub == "tool_approval_requested":
            s.human_intervention_count += 1
            s.approvals.append(
                ApprovalSummary(
                    kind="tool_approval_requested",
                    tool_name=_str(event, "tool_name") or None,
                    risk=_str(event, "risk") or None,
                    ts=ts,
                )
            )
        elif sub == "tool_call_end":
            s.tool_call_count += 1
            status = _str(event, "status") or None
            if status and status != "success":
                s.tool_failure_count += 1
            s.tool_calls.append(
                ToolCallSummary(
                    call_id=_str(event, "call_id") or None,
                    tool_name=_str(event, "tool_name"),
                    agent_name=_str(event, "agent_name"),
                    status=status,
                    failure_kind=_str(event, "failure_kind") or None,
                    duration_seconds=float(event.get("duration_seconds") or 0.0),
                    error=_str(event, "error") or None,
                    ts=ts,
                    trace_id=_str(event, "trace_id") or None,
                    step_id=_str(event, "step_id") or None,
                )
            )


__all__ = [
    "ApprovalSummary",
    "CheckpointSummary",
    "LLMCallSummary",
    "RunAggregator",
    "RunSummary",
    "SafetyDecisionSummary",
    "StepRecord",
    "ToolCallSummary",
]
