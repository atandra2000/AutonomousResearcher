"""Model routing for agents.

The :class:`ModelRouter` resolves which provider *and* model an agent
should use, then returns a configured :class:`~research_engineer.llm.base.LLMProvider`
instance primed with that model.

Agents never call ``complete(model=...)`` directly; they ask the router
for a provider, then call :meth:`LLMProvider.complete` on the returned
instance. The router bakes the chosen model into a thin wrapper so that
agents can simply call ``provider.complete(request)`` and the right model
is selected automatically.
"""

from __future__ import annotations

from typing import Any

from research_engineer.llm.base import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
    ToolDefinition,
)
from research_engineer.llm.cost import PricingTable
from research_engineer.llm.factory import ProviderFactory, get_factory


class _BoundProvider(LLMProvider):
    """Wraps a concrete provider, pinning ``model`` on every request."""

    name = "bound"

    def __init__(self, delegate: LLMProvider, model: str | None) -> None:
        self._delegate = delegate
        self._model = model
        # Surface the underlying provider identity for introspection.
        self.name = delegate.name
        self.default_model = model or delegate.default_model

    async def complete(self, request: LLMRequest) -> LLMResponse:
        if request.model is None and self._model is not None:
            request = request.model_copy(update={"model": self._model})
        from research_engineer.llm.resilience import complete_with_retry

        return await complete_with_retry(
            self._delegate,
            request,
            agent_name=f"{self._delegate.name}/{self._model or 'default'}",
            on_complete=self._stamp_cost_record_and_emit,
        )

    async def complete_with_tools(
        self,
        request: LLMRequest,
        tools: list[ToolDefinition],
    ) -> LLMResponse:
        if request.model is None and self._model is not None:
            request = request.model_copy(update={"model": self._model})
        from research_engineer.llm.resilience import complete_with_retry

        return await complete_with_retry(
            self._delegate,
            request,
            agent_name=f"{self._delegate.name}/{self._model or 'default'}",
            tools=tools,
            on_complete=self._stamp_cost_record_and_emit,
        )

    def _stamp_cost_record_and_emit(
        self, request: LLMRequest, response: LLMResponse, latency_seconds: float
    ) -> None:
        """Stamp USD cost, record into the UsageTracker, and emit an event.

        Combines D1 (cost accounting) and D2 (observability) into a single
        best-effort hook invoked once per completed call.
        """
        from research_engineer.llm.cost import compute_usage_cost, get_usage_tracker
        from research_engineer.observability import get_event_bus

        model = response.model or self._model or self._delegate.default_model
        # Stamp cost in place so callers see it on the returned response.
        response.usage = compute_usage_cost(
            response.usage, model, self._pricing_table()
        )
        label = f"{self._delegate.name}/{self._model or 'default'}"
        get_usage_tracker().record(label, response.usage)
        # Emit a structured event for observability (best-effort).
        try:
            get_event_bus().emit_llm_call(
                agent_name=label,
                request=request,
                response=response,
                latency_seconds=latency_seconds,
            )
        except Exception:
            pass

    def _pricing_table(self) -> PricingTable | None:
        """Resolve the pricing table from the factory (best-effort)."""
        try:
            from research_engineer.llm.factory import get_factory

            return get_factory().pricing_table
        except Exception:
            return None

    async def stream(self, request: LLMRequest) -> Any:
        if request.model is None and self._model is not None:
            request = request.model_copy(update={"model": self._model})
        # ``delegate.stream`` may be an async generator; iterate generically.
        async for chunk in self._delegate.stream(request):  # type: ignore[attr-defined]
            yield chunk

    async def stream_response(self, request: LLMRequest) -> LLMResponse:
        """Stream a completion and return the assembled, cost-stamped response.

        Uses :func:`~research_engineer.llm.streaming.stream_with_retry` so
        streamed calls get the same retry/backoff, cost accounting (D1), and
        observability (D2) treatment as non-streamed completions.
        """
        if request.model is None and self._model is not None:
            request = request.model_copy(update={"model": self._model})
        from research_engineer.llm.streaming import stream_with_retry

        return await stream_with_retry(
            self._delegate,
            request,
            agent_name=f"{self._delegate.name}/{self._model or 'default'}",
            on_complete=self._stamp_cost_record_and_emit,
        )


    async def validate(self, request: LLMRequest) -> bool:
        return await self._delegate.validate(request)

    @property
    def models(self) -> list[str]:
        return self._delegate.models

    def __repr__(self) -> str:
        return f"<BoundProvider delegate={self._delegate!r} model={self._model!r}>"


class ModelRouter:
    """Resolves a provider/model binding for each agent.

    Usage::

        router = ModelRouter(factory)
        provider = router.for_agent("CodingAgent")
        response = await provider.complete(request)
    """

    #: Canonical list of agent names recognised by the platform.
    KNOWN_AGENTS: tuple[str, ...] = (
        "ResearchAgent",
        "RepositoryAgent",
        "ExperimentPlannerAgent",
        "CodingAgent",
        "MemoryAgent",
        "LiteratureAgent",
        "ExperimentAgent",
        "EvaluationAgent",
        "ResearchLoopAgent",
        "TaskAgent",
        "ArchitectAgent",
        "ReviewerAgent",
        "TestAgent",
        "FailureAnalyzer",
        "RepairStrategist",
        "LiteratureDiscoveryAgent",
        "KnowledgeSynthesisAgent",
        "HypothesisGeneratorAgent",
        "ResearchExperimentPlannerAgent",
        "ExperimentExecutorAgent",
        "ResultAnalyzerAgent",
        "ReportGeneratorAgent",
        "ResearchOrchestrator",
    )

    def __init__(self, factory: ProviderFactory | None = None) -> None:
        self._factory = factory or get_factory()
        self._cache: dict[str, LLMProvider] = {}

    @property
    def factory(self) -> ProviderFactory:
        return self._factory

    def for_agent(self, agent_name: str) -> LLMProvider:
        """Return a model-bound provider for ``agent_name``."""
        if agent_name in self._cache:
            return self._cache[agent_name]
        spec = self._factory.get_spec(agent_name)
        provider = self._factory.get_provider(spec.provider_name)
        bound = _BoundProvider(provider, spec.model)
        self._cache[agent_name] = bound
        return bound

    async def for_agent_with_failover(self, agent_name: str) -> LLMProvider:
        """Return a model-bound provider, preferring healthy providers.

        Probes the configured providers via the factory's health check and
        binds the first healthy provider (in config order). If no provider
        reports healthy, falls back to the agent's configured provider so
        the request can still be attempted. The result is cached per agent.
        """
        if agent_name in self._cache:
            return self._cache[agent_name]
        spec = self._factory.get_spec(agent_name)
        order = await self._factory.healthy_provider_order()
        # ``order`` lists healthy providers first (config order), then
        # unhealthy ones. Prefer the first healthy provider; if none are
        # healthy, fall back to the agent's configured provider.
        chosen = order[0] if order else spec.provider_name
        provider = self._factory.get_provider(chosen)
        bound = _BoundProvider(provider, spec.model)
        self._cache[agent_name] = bound
        return bound

    async def health_check(self, agent_name: str) -> bool:
        """Probe the health of the provider bound to ``agent_name``."""
        spec = self._factory.get_spec(agent_name)
        return await self._factory.health_check(spec.provider_name)

    def model_for(self, agent_name: str) -> str | None:
        """Return the configured model id for ``agent_name`` (or None)."""
        return self._factory.get_spec(agent_name).model

    def provider_name_for(self, agent_name: str) -> str | None:
        """Return the configured provider name for ``agent_name``."""
        return self._factory.get_spec(agent_name).provider_name

    def reconfigure(self, factory: ProviderFactory) -> None:
        """Swap the underlying factory and drop cached bindings."""
        self._factory = factory
        self._cache.clear()

    def __repr__(self) -> str:
        return f"<ModelRouter factory={self._factory!r}>"


# ---------------------------------------------------------------------------
# Process-wide router accessor
# ---------------------------------------------------------------------------

_router: ModelRouter | None = None


def get_router(factory: ProviderFactory | None = None) -> ModelRouter:
    """Return the process-wide :class:`ModelRouter`.

    On first call the router wraps the singleton factory obtained from
    :func:`~research_engineer.llm.factory.get_factory`. Pass ``factory``
    to bind an explicit factory (used by tests).
    """
    global _router
    if _router is None or factory is not None:
        _router = ModelRouter(factory or get_factory())
    return _router


def reset_router() -> None:
    """Drop the cached router (tests use this)."""
    global _router
    _router = None
