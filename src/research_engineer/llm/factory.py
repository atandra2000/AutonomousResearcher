"""LLM provider factory and configuration loader.

Reads a YAML configuration file (default: ``llm_config.yaml`` at the
package root or a path supplied via the ``RE_LLM_CONFIG`` environment
variable) describing:

1. Named providers (``providers.<name>``) with their connection settings.
2. Per-agent model assignments (``agents.<agent_name>``) selecting which
   provider *and* which model an agent should use.

A singleton :class:`ProviderFactory` builds and caches the provider
instances. The :func:`get_factory` accessor lazily constructs the factory
using the resolved config path, while :func:`reset_factory` clears the
cache (useful in tests).

Example ``llm_config.yaml``::

    default_provider: ollama
    default_model: llama3
    providers:
      ollama:
        type: ollama
        base_url: https://api.olama.cloud
        api_key: ${OLLAMA_API_KEY}
        default_model: llama3
        timeout: 60
    agents:
      ResearchAgent:
        provider: ollama
        model: llama3
      CodingAgent:
        provider: ollama
        model: qwen2.5-coder
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from research_engineer.llm.base import LLMProvider, ProviderError
from research_engineer.llm.ollama_provider import OllamaCloudProvider

#: Mapping of provider ``type`` strings -> provider classes.
_PROVIDER_REGISTRY: dict[str, type[LLMProvider]] = {
    "ollama": OllamaCloudProvider,
}


_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _expand_env(value: Any) -> Any:
    """Expand ``${VAR}`` placeholders using the process environment."""
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


class AgentModelSpec:
    """Resolved provider/model binding for a single agent."""

    __slots__ = ("agent_name", "provider_name", "model")

    def __init__(self, agent_name: str, provider_name: str, model: str | None) -> None:
        self.agent_name = agent_name
        self.provider_name = provider_name
        self.model = model

    def __repr__(self) -> str:
        return (
            f"AgentModelSpec(agent={self.agent_name!r}, "
            f"provider={self.provider_name!r}, model={self.model!r})"
        )


class ProviderFactory:
    """Builds and caches LLM provider instances from a config dict."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config: dict[str, Any] = config or {}
        self._providers: dict[str, LLMProvider] = {}
        self._specs: dict[str, AgentModelSpec] = {}
        self._initialized = False

    # ------------------------------------------------------------------
    # Construction / config
    # ------------------------------------------------------------------

    @property
    def config(self) -> dict[str, Any]:
        return self._config

    def initialize(self) -> None:
        """Parse the config dict eagerly into providers and agent specs."""
        if self._initialized:
            return
        self._initialized = True

        providers_cfg = _expand_env(self._config.get("providers", {})) or {}
        agents_cfg = _expand_env(self._config.get("agents", {})) or {}

        default_provider = self._config.get("default_provider")
        default_model = self._config.get("default_model")

        # Lazy-build each referenced provider.
        for name, pcfg in providers_cfg.items():
            if not isinstance(pcfg, dict):
                continue
            ptype = pcfg.pop("type", None) or name
            cls = _PROVIDER_REGISTRY.get(ptype)
            if cls is None:
                raise ProviderError(
                    f"Unknown provider type {ptype!r} for provider {name!r}; "
                    f"registered types: {sorted(_PROVIDER_REGISTRY)}"
                )
            # Drop empty-string kwargs (so defaults kick in).
            kwargs = {k: v for k, v in pcfg.items() if v not in (None, "")}
            # Coerce numeric timeout if provided as a string.
            if "timeout" in kwargs and isinstance(kwargs["timeout"], str):
                try:
                    kwargs["timeout"] = float(kwargs["timeout"])
                except ValueError:
                    pass
            self._providers[name] = cls(**kwargs)

        # Resolve per-agent specs.
        for agent_name, acfg in agents_cfg.items():
            if not isinstance(acfg, dict):
                continue
            prov = acfg.get("provider") or default_provider or _first(self._providers) or ""
            model = acfg.get("model") or default_model
            self._specs[agent_name] = AgentModelSpec(agent_name, str(prov), model)

        # Ensure default provider exists even when config omits it.
        if not self._providers and default_provider is None:
            # Fall back to a stock Ollama provider configured from env.
            self._providers["ollama"] = OllamaCloudProvider()

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def get_provider(self, name: str | None = None) -> LLMProvider:
        """Return a (cached) provider by name, falling back to the default."""
        if not self._initialized:
            self.initialize()
        if name and name in self._providers:
            return self._providers[name]
        default = self._config.get("default_provider")
        if default and default in self._providers:
            return self._providers[default]
        if self._providers:
            return next(iter(self._providers.values()))
        # Last-resort default (env-configured Ollama).
        prov = OllamaCloudProvider()
        self._providers["ollama"] = prov
        return prov

    def get_spec(self, agent_name: str) -> AgentModelSpec:
        """Return the resolved provider/model spec for ``agent_name``."""
        if not self._initialized:
            self.initialize()
        if agent_name in self._specs:
            return self._specs[agent_name]
        # Fall back to defaults.
        default_provider = self._config.get("default_provider") or _first(self._providers) or ""
        default_model = self._config.get("default_model")
        spec = AgentModelSpec(agent_name, str(default_provider), default_model)
        self._specs[agent_name] = spec
        return spec

    def reset(self) -> None:
        """Discard built providers and specs (used by tests)."""
        import inspect

        for prov in self._providers.values():
            close = getattr(prov, "aclose", None)
            if callable(close):
                try:
                    if inspect.iscoroutinefunction(close):
                        # Best-effort: schedule the coroutine if a loop is
                        # running; otherwise skip (caller should ``await``
                        # ``aclose`` directly in async tests).
                        continue
                    close()
                except Exception:
                    pass
        self._providers.clear()
        self._specs.clear()
        self._initialized = False


# ---------------------------------------------------------------------------
# Config loading + singleton
# ---------------------------------------------------------------------------


def _default_config_path() -> Path:
    """Resolve the default config-file search path.

    Order:
      1. ``$RE_LLM_CONFIG`` env var (if set).
      2. ``llm_config.yaml`` next to the package root.
      3. ``llm_config.yaml`` in the current working directory.
    """
    env_path = os.environ.get("RE_LLM_CONFIG")
    if env_path:
        return Path(env_path)
    pkg_root = Path(__file__).resolve().parent.parent.parent.parent
    candidates = [pkg_root / "llm_config.yaml", Path.cwd() / "llm_config.yaml"]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config file into a plain dict (empty if missing)."""
    p = Path(path) if path else _default_config_path()
    if not p.exists():
        return {}
    import yaml  # type: ignore[import-untyped]  # local import: PyYAML

    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"LLM config root must be a mapping, got {type(data).__name__}")
    return data


_factory: ProviderFactory | None = None


def get_factory(config: dict[str, Any] | None = None) -> ProviderFactory:
    """Return the process-wide :class:`ProviderFactory` singleton.

    On first call the factory is constructed from ``config`` if supplied,
    otherwise from the resolved config file. Subsequent calls return the
    cached instance unless :func:`reset_factory` is called.
    """
    global _factory
    if _factory is None:
        cfg = config if config is not None else load_config()
        _factory = ProviderFactory(cfg)
    return _factory


def reset_factory() -> None:
    """Drop and re-create the singleton factory (tests use this)."""
    global _factory
    if _factory is not None:
        _factory.reset()
    _factory = None


def register_provider_type(type_name: str, cls: type[LLMProvider]) -> None:
    """Register an additional provider implementation by ``type`` name."""
    _PROVIDER_REGISTRY[type_name] = cls


def _first(d: dict[str, Any]) -> str | None:
    return next(iter(d), None)
