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

from research_engineer.llm.base import LLMProvider, LLMRequest, LLMResponse
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
        return await self._delegate.complete(request)

    async def stream(self, request: LLMRequest) -> Any:
        if request.model is None and self._model is not None:
            request = request.model_copy(update={"model": self._model})
        # ``delegate.stream`` may be an async generator; iterate generically.
        async for chunk in self._delegate.stream(request):  # type: ignore[attr-defined]
            yield chunk

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
