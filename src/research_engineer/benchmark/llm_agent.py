"""P2 - LLM-backed benchmark agent for real-agent research runs.

The deterministic P1 tiers (``bench_tool`` / ``bench_flaky`` / ``bench_loop``
/ ``bench_risky``) exercise the production chains without credentials. This
module wires a *real* LLM through the identical production path:

    API -> PostgreSQL/SQLite -> Worker -> AgentRuntime -> ToolGateway ->
    SafetyController -> checkpointing -> evaluation -> E6 telemetry -> E8

``KIND_LLM_REACT`` builds an adapter where each runtime step performs one
model call via the Phase-10 provider abstraction
(:class:`~research_engineer.llm.base.LLMProvider.complete_with_tools`,
through :class:`~research_engineer.llm.router.ModelRouter` bindings so
retry/backoff/cost-stamping apply). Every tool request the model makes is
routed through ``AgentRuntime.call_tool``, i.e. the full E3 gateway + E5
safety chain - the model can never bypass policy, approval, or sandbox.

Fail-closed properties preserved:

* No provider configured/reachable -> the step raises and the run fails
  with an ERROR termination (never fabricated results).
* Credential values come exclusively from environment expansion inside
  ``llm_config.yaml`` / provider defaults; nothing here reads or logs keys.
* Without an attached runtime the adapter refuses tool dispatch (same
  contract as every ``RuntimeAwareAdapter``).

Configuration is supplied per case through scalar ``agent_overrides``
(forwarded verbatim as ``budget_overrides``):

=========================== ==========================================
Key                         Meaning
=========================== ==========================================
``llm_provider``            ProviderFactory name (config-selected)
``llm_model``               Model id override (per-suite experiment arm)
``llm_temperature``         Sampling temperature (default 0.2)
``llm_max_tokens_per_call`` Completion cap per model call (default 1024)
``max_steps``               Runtime step (= model call) budget
``max_tool_calls``          Gateway-dispatched tool-call budget
``max_tokens``              Cumulative token budget (runtime enforced)
``max_cost_usd``            Cumulative USD cost budget (runtime enforced)
=========================== ==========================================

Resume caveat (documented at-least-once boundary, mirroring the worker's):
after a mid-flight crash the rebuilt adapter restarts its conversation from
the goal; checkpointed context (tokens/cost/tool log/steps) is preserved and
never repeated by the runtime, but uncommitted conversation turns replay.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, ValidationError

from research_engineer.benchmark.agents import AgentFactoryRegistry, AgentFactoryReturn
from research_engineer.benchmark.bench_agents import (
    EmptyInput,
    NoteWriteInput,
    ProbeInput,
    RuntimeAwareAdapter,
)
from research_engineer.llm.base import (
    LLMMessage,
    LLMRequest,
    LLMRole,
    ToolCall,
    ToolDefinition,
    ToolResult,
)

logger = logging.getLogger(__name__)

#: Registry key for the LLM-backed research agent kind.
KIND_LLM_REACT = "llm_react"

#: Marker terminating the ReAct conversation.
FINAL_MARKER = "FINAL_ANSWER:"

#: Router agent name used to bind the benchmark's provider/model.
ROUTER_AGENT_NAME = "BenchmarkLLM"

_WRITE_TOOL = "research_note_write"

_SYSTEM_PROMPT = """\
You are the autonomous research agent of the Autonomous ML Research \
Engineer platform. You execute ONE bounded research task per run using the \
provided tools. Respond concisely; think before you act.

Working rules:
1. Reason ONLY from the information contained in the task description. Do \
not invent citations, numbers, or experimental results you were not given.
2. Record your deliverables with the research_note_write tool as you go \
(one note per required deliverable), using short descriptive file names.
3. Respect your budgets: once every required deliverable exists, stop \
making further tool calls.
4. When every required deliverable has been written, finish your reply with \
a line starting exactly with:
""" + FINAL_MARKER + """
followed by your final structured answer. Include every required section \
header from the task inside that final answer."""

#: Optional planning-strategy addenda selected per benchmark arm through the
#: scalar ``llm_strategy`` agent override. Each variant appends exactly one
#: paragraph to the shared system prompt so a strategy change is a single,
#: auditable delta versus the frozen baseline prompt. Unknown strategy names
#: fail closed (the factory raises instead of silently running base prompt).
STRATEGY_PROMPTS: dict[str, str] = {
    # B: global plan-first finishing discipline (targets the observed
    # single-step zero-tool-call failure mode where deliverables were never
    # written before the model declared FINAL_ANSWER).
    "plan_first": (
        "\n5. Finish discipline (mandatory): first enumerate internally which "
        "deliverables the task requires, then produce them with "
        "research_note_write, and only afterwards emit your FINAL_ANSWER. A "
        "final answer that omits a required deliverable or a required section "
        "header counts as failure even if the prose is excellent."
    ),
    # E: implementation-task-specific strategy (scoped to implementation-
    # category cases by the experiment configuration).
    "impl_focus": (
        "\n5. Implementation-task method (mandatory): (a) state the exact "
        "interface/signature with explicit tensor shapes and verify they are "
        "dimensionally consistent; (b) order concrete build steps; (c) list "
        "genuine edge cases with why they break naive implementations; "
        "(d) specify unit tests including a numeric equivalence check. Write "
        "this plan via research_note_write BEFORE emitting FINAL_ANSWER, and "
        "repeat every required header verbatim inside FINAL_ANSWER."
    ),
}


def system_prompt(strategy: str | None = None) -> str:
    """Compose the agent system prompt for an optional strategy name."""
    if not strategy:
        return _SYSTEM_PROMPT
    suffix = STRATEGY_PROMPTS.get(strategy)
    if suffix is None:
        raise ValueError(f"unknown llm_strategy {strategy!r}")
    return _SYSTEM_PROMPT + suffix


# ---------------------------------------------------------------------------
# Tool surface exposed to the model (gateway-registered sandbox tools only)
# ---------------------------------------------------------------------------


class _ToolSpec(BaseModel):
    """One model-visible tool bound to its gateway input model."""

    input_model: type[BaseModel]


def _tool_spec(
    name: str, description: str, model: type[BaseModel],
) -> tuple[str, ToolDefinition, _ToolSpec]:
    definition = ToolDefinition(
        name=name,
        description=description,
        parameters=model.model_json_schema(),
    )
    return name, definition, _ToolSpec(input_model=model)


_TOOLS: tuple[tuple[str, ToolDefinition, _ToolSpec], ...] = (
    _tool_spec(
        "research_note_write",
        "Write a research note into the workspace sandbox "
        "(creates notes/<name>.txt). `name` is a short filename stem using "
        "letters, digits, dots, dashes, or underscores.",
        NoteWriteInput,
    ),
    _tool_spec(
        "research_note_list",
        "List the research notes currently in the workspace.",
        EmptyInput,
    ),
    _tool_spec(
        "echo_probe",
        "Low-risk stateless probe; returns a digest of its payload.",
        ProbeInput,
    ),
)

_TOOL_MAP: dict[str, _ToolSpec] = {n: s for n, _, s in _TOOLS}


def tool_definitions() -> list[ToolDefinition]:
    """Model-visible tool definitions (mirror of the gateway registrations)."""
    return [definition for _, definition, _ in _TOOLS]



def _sanitize_content(result: Any, *, limit: int = 4000) -> str:
    """Best-effort compact serialization of a tool output for the model."""
    output = getattr(result, "output", None)
    if output is None:
        return ""
    try:
        rendered = json.dumps(output, default=str)
    except (TypeError, ValueError):
        rendered = str(output)
    return rendered[:limit]


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class LLMReActAdapter(RuntimeAwareAdapter):
    """Real-LLM research adapter driven one model call per runtime step.

    The runtime owns budgets, checkpointing, error classification, and the
    safety chain; this adapter contributes reasoning (the provider), the
    conversation state, and gateway-dispatched tool execution.
    """

    def __init__(
        self,
        provider: Any,
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens_per_call: int = 1024,
        strategy: str | None = None,
        agent_name: str = "benchmark_llm",
    ) -> None:
        super().__init__(agent_name=agent_name)
        # Fail closed on unknown strategy names (P3 experiment arms set this
        # through the ``llm_strategy`` override; a typo must not silently
        # execute the baseline prompt under a candidate label).
        self.strategy = str(strategy) if strategy else None
        self.system_prompt_text = system_prompt(self.strategy)
        self.provider = provider
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens_per_call = int(max_tokens_per_call)
        # Per-execution conversation state (rebuilt after crash-resume).
        self._conversations: dict[str, list[LLMMessage]] = {}
        self._notes_written = 0
        self._tool_calls_ok = 0
        self._tool_calls_failed = 0
        self._model_calls = 0
        self._final_output: dict[str, Any] | None = None

    # -- internals --------------------------------------------------------

    def _messages(self, execution_id: str, goal: str) -> list[LLMMessage]:
        conversation = self._conversations.get(execution_id)
        if conversation is None:
            conversation = [
                LLMMessage(
                    role=LLMRole.SYSTEM, content=self.system_prompt_text
                ),
                LLMMessage(role=LLMRole.USER, content=goal),
            ]
            self._conversations[execution_id] = conversation
        return conversation

    async def _complete(self, messages: list[LLMMessage]) -> Any:
        """One retried model call with the suite's sampling parameters."""
        request = LLMRequest(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens_per_call,
        )
        # Wrapped providers (e.g. the router's bound provider) retry
        # transient transport/API errors internally and stamp USD cost.
        return await self.provider.complete_with_tools(
            request, tool_definitions()
        )

    async def _execute_tool(self, call: ToolCall) -> ToolResult:
        """Dispatch one model-requested tool through the gateway chain."""
        spec = _TOOL_MAP.get(call.name)
        if spec is None:
            self._tool_calls_failed += 1
            return ToolResult(
                tool_call_id=call.id, name=call.name,
                content=(f"error: unknown tool {call.name!r}; available: "
                         f"{sorted(_TOOL_MAP)}"),
                is_error=True,
            )
        try:
            payload = spec.input_model.model_validate(call.arguments or {})
        except ValidationError as exc:
            self._tool_calls_failed += 1
            return ToolResult(
                tool_call_id=call.id, name=call.name,
                content=f"error: invalid arguments "
                        f"({exc.error_count()} validation problem(s))",
                is_error=True,
            )
        result = await self._call(call.name, payload)
        status = str(getattr(result, "status", ""))
        if status == "success":
            self._tool_calls_ok += 1
            if call.name == _WRITE_TOOL:
                self._notes_written += 1
        else:
            self._tool_calls_failed += 1
        error = getattr(result, "error", "")
        content = (
            _sanitize_content(result)
            if status == "success" else f"error: {error}"[:1200]
        )
        return ToolResult(
            tool_call_id=call.id, name=call.name,
            content=content, is_error=status != "success",
        )

    def _resolved_model(self, response: Any) -> str:
        reported = str(getattr(response, "model", "") or "")
        return reported or (self.model or "")

    # -- runtime phases ----------------------------------------------------

    async def planner(self, ctx: Any) -> Any:
        return {"step_plan": "one model call with gateway tool access"}

    async def observer(self, ctx: Any, action: Any) -> Any:
        return action

    async def actor(self, ctx: Any, plan: Any) -> Any:
        messages = self._messages(
            str(getattr(ctx, "execution_id", "exec")), str(ctx.goal)
        )
        response = await self._complete(messages)
        self._model_calls += 1

        usage = getattr(response, "usage", None)
        ctx.tokens += int(getattr(usage, "total_tokens", 0) or 0)
        cost_delta = float(getattr(usage, "cost_usd", 0.0) or 0.0)
        ctx.cost_usd = round(ctx.cost_usd + cost_delta, 6)

        content = str(getattr(response, "content", "") or "")
        # Reasoning models often emit tool calls with no visible text;
        # empty content would fail provider request validation on the
        # next turn, so record an explicit placeholder instead.
        if not content.strip():
            content = "(tool call issued; no textual content)"
        tool_calls = list(getattr(response, "tool_calls", None) or [])
        messages.append(
            LLMMessage(
                role=LLMRole.ASSISTANT,
                content=content,
                tool_calls=tool_calls or None,
            )
        )
        resolved_model = self._resolved_model(response)

        finished = FINAL_MARKER in content or not tool_calls
        if not finished:
            for call in tool_calls:
                result = await self._execute_tool(call)
                messages.append(
                    LLMMessage(
                        role=LLMRole.TOOL,
                        content=result.content,
                        tool_call_id=result.tool_call_id,
                        name=result.name,
                    )
                )
            return {
                "finished": False,
                "model": resolved_model,
                "model_calls": self._model_calls,
                "notes_written": self._notes_written,
                "pending_tool_results": len(tool_calls),
            }

        final_answer = content.replace(FINAL_MARKER, "").strip()
        self._final_output = {
            "llm_agent": True,
            "model": resolved_model,
            "model_calls": self._model_calls,
            "notes_written": self._notes_written,
            "tool_calls_ok": self._tool_calls_ok,
            "tool_calls_failed": self._tool_calls_failed,
            "tokens_reported": int(getattr(usage, "total_tokens", 0) or 0),
            "cost_reported_usd": float(
                getattr(usage, "cost_usd", 0.0) or 0.0
            ),
            "final_answer": final_answer[:20_000],
        }
        return {"finished": True, **self._final_output}

    async def evaluator(self, ctx: Any, observation: Any) -> Any:
        obs = observation if isinstance(observation, dict) else {}
        if obs.get("finished"):
            output = {k: v for k, v in obs.items() if k != "finished"}
            if self._final_output is not None:
                output = {**self._final_output}
            return {"done": True, "score": 1.0, "output": output}
        # Strictly rising micro-scores keep the E1 stagnation guard quiet;
        # honest progress lives in token/tool-count metrics.
        step = int(getattr(ctx, "current_step", 0)) + 1
        return {"done": False, "score": min(0.95, 0.1 + 0.05 * step)}


# ---------------------------------------------------------------------------
# Factory + registration
# ---------------------------------------------------------------------------


def resolve_llm_provider(
    overrides: dict[str, Any], injected: Any = None,
) -> tuple[Any, str | None]:
    """Resolve ``(provider, explicit_model)`` from overrides/configuration.

    Resolution order:
      1. Injected provider (test/ops dependency injection) wins.
      2. ``llm_provider`` override naming a configured provider; the
         configured default model is bound (retry + cost stamping).
      3. The router binding for ``BenchmarkLLM`` from
         ``llm_config.yaml`` / env fallbacks.

    Credentials are never accepted as overrides; providers obtain them from
    environment-expanded configuration only.
    """
    if injected is not None:
        return injected, None
    from research_engineer.llm.factory import get_factory
    from research_engineer.llm.router import _BoundProvider, get_router

    factory = get_factory()
    router = get_router(factory)
    name = overrides.get("llm_provider")
    model_override = overrides.get("llm_model")
    if not name and not model_override:
        # Fully config-driven: bound provider adds retry/backoff and USD
        # cost stamping on every call.
        return router.for_agent(ROUTER_AGENT_NAME), None
    base = factory.get_provider(str(name) if name else None)
    bound = _BoundProvider(base, str(model_override) if model_override else (
        router.model_for(ROUTER_AGENT_NAME)
    ))
    if model_override:
        return bound, str(model_override)
    return bound, None


def _llm_react_factory(injected_provider: Any = None) -> Any:
    async def factory(overrides: dict[str, Any]) -> AgentFactoryReturn:
        from research_engineer.runtime.models import AgentPolicy

        provider, explicit_model = resolve_llm_provider(
            overrides, injected=injected_provider
        )
        adapter = LLMReActAdapter(
            provider,
            model=explicit_model,
            temperature=float(overrides.get("llm_temperature", 0.2)),
            # Reasoning-style models spend visible-budget on internal CoT;
            # the resilience layer escalates once on truncation.
            max_tokens_per_call=int(
                overrides.get("llm_max_tokens_per_call", 2048)
            ),
            strategy=(
                str(overrides["llm_strategy"])
                if overrides.get("llm_strategy") else None
            ),
        )
        return adapter, AgentPolicy()

    return factory


def register_llm_agent_kinds(
    registry: AgentFactoryRegistry, *, provider: Any = None,
) -> None:
    """Register the ``llm_react`` agent kind on the E7 factory registry."""
    registry.register(KIND_LLM_REACT, _llm_react_factory(provider))


__all__ = [
    "FINAL_MARKER",
    "KIND_LLM_REACT",
    "LLMReActAdapter",
    "ROUTER_AGENT_NAME",
    "STRATEGY_PROMPTS",
    "register_llm_agent_kinds",
    "resolve_llm_provider",
    "system_prompt",
    "tool_definitions",
]

