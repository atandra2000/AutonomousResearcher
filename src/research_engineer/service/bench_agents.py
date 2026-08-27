"""P1 - Deterministic benchmark agents exercising the production chains.

These adapters give the benchmark *real* execution paths through the full
stack - E1 runtime loop, E2 checkpointing, E3 ToolGateway policy/approval/
sandbox dispatch, E5 SafetyController controls, E6 telemetry - without
requiring LLM credentials, so runs stay reproducible and bounded:

* ``bench_tool``   - performs genuine gateway-dispatched filesystem writes
                     into the artifact workspace (real tool-use chain).
* ``bench_flaky``  - raises simulated transient failures first, then
                     succeeds (recovery/error-classification path).
* ``bench_loop``   - repeats an identical step forever so the E5 loop
                     detector escalates through REPLAN to a controlled
                     TERMINATE (deterministic safety intervention).
* ``bench_risky``  - calls an approval-gated HIGH-risk tool that is denied
                     when ``enforce_approval`` is set (approval chain).

Sandbox tools registered here follow the standard :class:`Tool` ABC and are
the *only* tools registered on the service gateway beyond operator-added
ones, keeping default-deny meaningful.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from research_engineer.gateway.models import RiskLevel
from research_engineer.runtime.adapters import AgentAdapter
from research_engineer.service.agents import AgentFactoryRegistry, AgentFactoryReturn
from research_engineer.tools.base import Tool, ToolError

# Registry keys; the E7 default kind remains ``planning_checklist``.
KIND_BENCH_TOOL = "bench_tool"
KIND_BENCH_FLAKY = "bench_flaky"
KIND_BENCH_LOOP = "bench_loop"
KIND_BENCH_RISKY = "bench_risky"

#: Default transient failures injected per flaky benchmark run.
DEFAULT_FLAKY_FAILURES = 2


# ---------------------------------------------------------------------------
# Sandbox tools (real implementations, workspace-confined)
# ---------------------------------------------------------------------------


class NoteWriteInput(BaseModel):
    """Input for the sandbox research-note writer."""

    name: str = Field(
        ..., min_length=1, max_length=64,
        pattern=r"^[A-Za-z0-9_.-]+$", description="Note file stem",
    )
    content: str = Field(..., max_length=20_000, description="Note body")


class NoteWriteOutput(BaseModel):
    path: str
    bytes_written: int


class NoteWriteTool(Tool[NoteWriteInput, NoteWriteOutput]):
    """Write a research note under ``<workspace>/notes/``."""

    def __init__(self, workspace: Path) -> None:
        self._workspace = Path(workspace)

    async def execute(self, input: NoteWriteInput) -> NoteWriteOutput:
        notes_dir = self._workspace / "notes"
        notes_dir.mkdir(parents=True, exist_ok=True)
        target = notes_dir / f"{input.name}.txt"
        target.write_text(input.content, encoding="utf-8")
        return NoteWriteOutput(
            path=str(target), bytes_written=len(input.content.encode())
        )


class EmptyInput(BaseModel):
    pass


class FileListModel(BaseModel):
    files: list[str]


class ListNotesTool(Tool[EmptyInput, FileListModel]):
    """List research notes in the workspace (read-only)."""

    def __init__(self, workspace: Path) -> None:
        self._workspace = Path(workspace)

    async def execute(self, input: EmptyInput) -> FileListModel:
        notes_dir = self._workspace / "notes"
        if not notes_dir.is_dir():
            return FileListModel(files=[])
        return FileListModel(
            files=sorted(p.name for p in notes_dir.iterdir() if p.is_file())
        )


class ProbeInput(BaseModel):
    payload: str = Field(default="", max_length=2000)


class ProbeOutput(BaseModel):
    digest: str


class EchoProbeTool(Tool[ProbeInput, ProbeOutput]):
    """Stateless probe returning a content digest (loop-detection fodder)."""

    async def execute(self, input: ProbeInput) -> ProbeOutput:
        return ProbeOutput(
            digest=hashlib.sha256(input.payload.encode()).hexdigest()[:16]
        )


class ExternalProbeTool(Tool[ProbeInput, ProbeOutput]):
    """HIGH-risk external probe; never executes when approvals enforced.

    Registered ``requires_approval=True``; an enforcing autonomous stack
    denies it before this body can ever run.
    """

    async def execute(self, input: ProbeInput) -> ProbeOutput:
        raise ToolError(
            "external_probe must be denied by approval policy "
            "in autonomous production",
            input,
        )


def register_sandbox_tools(gateway: Any, *, workspace: Path | str) -> None:
    """Register the deterministic sandbox set on ``gateway``.

    ``workspace`` confines the filesystem tools; production callers pass
    the artifact volume so agent writes stay inside approved roots.
    """
    ws = Path(workspace)
    register = gateway.register_tool
    register(NoteWriteTool(ws), name="research_note_write",
             risk_level=RiskLevel.MEDIUM,
             description="Write workspace research note")
    register(ListNotesTool(ws), name="research_note_list",
             risk_level=RiskLevel.LOW,
             description="List workspace research notes")
    register(EchoProbeTool(), name="echo_probe", risk_level=RiskLevel.LOW,
             description="Deterministic probe for loop/duplicate detectors")
    register(
        ExternalProbeTool(),
        name="external_probe",
        risk_level=RiskLevel.HIGH,
        requires_approval=True,
        description="High-risk external probe (approval-gated)",
    )


# ---------------------------------------------------------------------------
# Runtime-aware adapter base
# ---------------------------------------------------------------------------


class RuntimeAwareAdapter(AgentAdapter):
    """Adapter that receives the executing ``AgentRuntime`` for tool calls.

    The worker injects the runtime right after construction so the actor
    can route every tool invocation through
    :meth:`AgentRuntime.call_tool`, i.e. the full gateway+safety chain.
    """

    @staticmethod
    async def _unused_invoke(**_: Any) -> None:
        """Subclasses override ``actor``; this stub only satisfies init."""
        return None

    def __init__(self, agent_name: str) -> None:
        super().__init__(agent_name=agent_name, invoke=self._unused_invoke)
        self._runtime: Any | None = None

    def attach_runtime(self, runtime: Any) -> None:
        self._runtime = runtime

    async def _call(self, tool_name: str, payload: Any) -> Any:
        if self._runtime is None:
            raise RuntimeError(
                f"{type(self).__name__} used without an attached runtime; "
                "tool calls must go through AgentRuntime.call_tool"
            )
        return await self._runtime.call_tool(
            tool_name, payload, agent_name=self.agent_name
        )

    async def actor(self, ctx: Any, plan: Any) -> Any:
        # Base.AgentAdapter.actor ignores subclass overrides unless we
        # redefine invoke; simpler to override actor directly.
        return await self._act(ctx)

    async def _act(self, ctx: Any) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError


# ---------------------------------------------------------------------------
# bench_tool - genuine gated tool-use
# ---------------------------------------------------------------------------


class BenchToolAdapter(RuntimeAwareAdapter):
    """Decompose the goal into <=4 checklist items written as real notes."""

    def __init__(self) -> None:
        super().__init__(agent_name="bench_tool_research")
        self._written = 0

    async def planner(self, ctx: Any) -> Any:
        return {"checklist": self.checklist(ctx.goal)}

    @staticmethod
    def checklist(goal: str) -> list[str]:
        from research_engineer.service.agents import _sentence_chunks

        items = [c for c in _sentence_chunks(goal) if len(c) > 8][:4]
        return items or ["summarize objective"]

    async def observer(self, ctx: Any, action: Any) -> Any:
        return action

    async def evaluator(self, ctx: Any, observation: Any) -> Any:
        obs = observation or {}
        total = int(obs.get("total_items", 1))
        written = int(obs.get("notes_written", 0))
        if written < total:
            return {
                "done": False,
                "score": min(1.0, written / total),
            }
        # ``output`` envelope: AgentRuntime._final_output unwraps this as the
        # run's graded result payload.
        return {
            "done": True,
            "score": 1.0,
            "output": {
                "summary": (
                    f"Completed research plan: wrote {written} of "
                    f"{total} planned notes to the workspace."
                ),
                "notes_written": written,
                "total_items": total,
            },
        }

    async def _act(self, ctx: Any) -> Any:
        items = self.checklist(ctx.goal)
        index = min(ctx.current_step, max(0, len(items) - 1))
        write_result = await self._call(
            "research_note_write",
            NoteWriteInput(
                name=f"note_{ctx.execution_id[:8]}_{index}",
                content=f"[{index + 1}/{len(items)}] {items[index]}",
            ),
        )
        status = str(getattr(write_result, "status", ""))
        if status != "success":
            raise RuntimeError(
                f"gateway rejected research_note_write ({status}): "
                f"{getattr(write_result, 'error', '')}"
            )
        self._written += 1
        listing = await self._call("research_note_list", EmptyInput())
        return {
            "written": True,
            "item_index": index,
            "total_items": len(items),
            "notes_written": self._written,
            "note_path": getattr(write_result.output, "path", None),
            "known_notes": list(
                getattr(getattr(listing, "output", None), "files", [])
            ),
        }


# ---------------------------------------------------------------------------
# bench_flaky - recovery from transient failures
# ---------------------------------------------------------------------------


class BenchFlakyAdapter(BenchToolAdapter):
    """Same flow as bench_tool but the first writes fail transiently.

    Simulated ``ConnectionError`` s are classified recoverable by the E1
    classifier, so retries flow through the normal recovery path; after
    the injected failures are spent, writes succeed via the real gateway.
    """

    def __init__(self, fail_calls: int = DEFAULT_FLAKY_FAILURES) -> None:
        super().__init__()
        self.agent_name = "bench_flaky_research"
        self._remaining = max(0, fail_calls)

    async def _act(self, ctx: Any) -> Any:
        if self._remaining > 0:
            self._remaining -= 1
            raise ConnectionError(
                "simulated transient infrastructure failure "
                "(benchmark recovery scenario)"
            )
        return await super()._act(ctx)


# ---------------------------------------------------------------------------
# bench_loop - deterministic E5 loop-detector escalation
# ---------------------------------------------------------------------------


class BenchLoopAdapter(RuntimeAwareAdapter):
    """Repeat one identical echo-probe step until safety controls stop it.

    Every step issues the byte-identical tool call, so duplicate-call and
    loop detectors both fire; the replayed REPLAN guidance is deliberately
    ignored, driving ``max_replans`` exhaustion and a controlled
    ``safety_terminated`` ending.
    """

    def __init__(self, payload: str = "identical-benchmark-step") -> None:
        super().__init__(agent_name="bench_loop_probe")
        self._payload = payload

    async def planner(self, ctx: Any) -> Any:
        return {"step_plan": "repeat probe"}

    async def observer(self, ctx: Any, action: Any) -> Any:
        return action

    async def evaluator(self, ctx: Any, observation: Any) -> Any:
        # Strictly rising micro-scores keep the E1 stagnation guard quiet so
        # the E5 loop detector's REPLAN->replan-limit escalation is what
        # deterministically terminates this run.
        return {"done": False,
                "score": 0.5 + 0.02 * min(ctx.current_step + 1, 20)}

    async def _act(self, ctx: Any) -> Any:
        result = await self._call(
            "echo_probe", ProbeInput(payload=self._payload)
        )
        output = getattr(result, "output", None)
        digest = getattr(output, "digest", "") or ""
        return {"digest": str(digest)}


# ---------------------------------------------------------------------------
# bench_risky - approval-gated HIGH-risk denial
# ---------------------------------------------------------------------------


class BenchRiskyAdapter(RuntimeAwareAdapter):
    """Call an approval-gated tool every step; enforcing stacks deny it."""

    def __init__(self) -> None:
        super().__init__(agent_name="bench_risky_external")
        self._denials = 0

    async def planner(self, ctx: Any) -> Any:
        return {"step_plan": "consult external source"}

    async def observer(self, ctx: Any, action: Any) -> Any:
        return action

    async def evaluator(self, ctx: Any, observation: Any) -> Any:
        # A well-behaved agent stops once blanket denial is evident instead
        # of hammering an approval gate.
        obs = observation or {}
        if int(obs.get("denial_count", 0)) >= 2:
            return {
                "done": True,
                "score": 0.25,
                "output": {
                    "blocked": (
                        "external access denied by approval policy after "
                        f"{obs.get('denial_count')} attempts"
                    ),
                    "gateway_status": obs.get("gateway_status"),
                },
            }
        return {"done": False, "score": 0.5}

    async def _act(self, ctx: Any) -> Any:
        result = await self._call(
            "external_probe", ProbeInput(payload=ctx.goal[:256])
        )
        status = str(getattr(result, "status", ""))
        self._denials += 0 if status == "success" else 1
        return {
            "probe_ok": status == "success",
            "denied": status != "success",
            "gateway_status": status,
            "denial_count": self._denials,
        }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

async def _bench_tool_factory(_overrides: dict[str, Any]) -> AgentFactoryReturn:
    from research_engineer.runtime.models import AgentPolicy

    return BenchToolAdapter(), AgentPolicy()


async def _bench_flaky_factory(
    overrides: dict[str, Any],
) -> AgentFactoryReturn:
    from research_engineer.runtime.models import AgentPolicy

    failures = int(
        overrides.get("bench_fail_calls", DEFAULT_FLAKY_FAILURES)
    )
    return BenchFlakyAdapter(fail_calls=failures), AgentPolicy()


async def _bench_loop_factory(_overrides: dict[str, Any]) -> AgentFactoryReturn:
    from research_engineer.runtime.models import AgentPolicy

    return BenchLoopAdapter(), AgentPolicy()


async def _bench_risky_factory(_overrides: dict[str, Any]) -> AgentFactoryReturn:
    from research_engineer.runtime.models import AgentPolicy

    return BenchRiskyAdapter(), AgentPolicy()


def register_benchmark_kinds(registry: AgentFactoryRegistry) -> None:
    """Register the four benchmark agent kinds on an E7 factory registry."""
    registry.register(KIND_BENCH_TOOL, _bench_tool_factory)
    registry.register(KIND_BENCH_FLAKY, _bench_flaky_factory)
    registry.register(KIND_BENCH_LOOP, _bench_loop_factory)
    registry.register(KIND_BENCH_RISKY, _bench_risky_factory)


__all__ = [
    "DEFAULT_FLAKY_FAILURES",
    "KIND_BENCH_LOOP",
    "KIND_BENCH_RISKY",
    "KIND_BENCH_TOOL",
    "KIND_BENCH_FLAKY",
    "register_benchmark_kinds",
    "register_sandbox_tools",
]
