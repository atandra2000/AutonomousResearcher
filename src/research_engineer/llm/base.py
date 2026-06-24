"""Provider-agnostic LLM abstraction.

Defines the core ``LLMProvider`` interface plus the request/response data
models that every concrete provider must consume and produce.

All agents in the platform talk to models exclusively through the
``LLMProvider`` ABC defined here; no agent should ever instantiate a
specific provider or speak a vendor-specific protocol directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class LLMRole(StrEnum):
    """Standard chat-message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class LLMMessage(BaseModel):
    """A single chat-completion message."""

    role: LLMRole = Field(..., description="Message role")
    content: str = Field(..., description="Message content")
    name: str | None = Field(default=None, description="Optional author name")
    tool_call_id: str | None = Field(
        default=None,
        description="Optional tool call id (for tool-role messages)",
    )


class LLMRequest(BaseModel):
    """A provider-agnostic completion request."""

    messages: list[LLMMessage] = Field(..., description="Conversation turns")
    model: str | None = Field(
        default=None,
        description="Override model id; falls back to provider default",
    )
    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Maximum tokens to generate",
    )
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Nucleus sampling")
    stop: list[str] | None = Field(
        default=None,
        description="Sequences that stop generation",
    )
    stream: bool = Field(default=False, description="Request streaming output")
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific passthrough options",
    )


class LLMUsage(BaseModel):
    """Token usage accounting."""

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class LLMResponse(BaseModel):
    """A provider-agnostic completion response."""

    content: str = Field(..., description="Generated text")
    model: str = Field(..., description="Model that produced this response")
    provider: str = Field(..., description="Provider name that produced this")
    usage: LLMUsage = Field(default_factory=LLMUsage)
    finish_reason: str | None = Field(default=None, description="Why generation stopped")
    raw: dict[str, Any] = Field(
        default_factory=dict,
        description="Raw provider payload (opaque passthrough)",
    )


class ProviderError(RuntimeError):
    """Raised when an LLM provider fails to fulfil a request."""

    def __init__(self, message: str, *, provider: str | None = None, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.provider = provider
        self.cause = cause


class LLMProvider(ABC):
    """Abstract base class for all LLM providers.

    Concrete providers (Ollama Cloud, OpenAI, Anthropic, ...) implement
    :meth:`complete` (and optionally :meth:`stream`) against their native
    HTTP API, returning provider-agnostic :class:`LLMResponse` objects.
    """

    #: Short, stable identifier for this provider (e.g. ``"ollama"``).
    name: str = "base"

    #: Default model id used when a request omits ``model``.
    default_model: str = ""

    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Generate a completion for ``request``."""

    async def stream(self, request: LLMRequest) -> Any:
        """Yield streamed completion chunks.

        Providers MAY override this. The default implementation raises
        :class:`NotImplementedError` to signal that streaming is unsupported.
        """
        raise NotImplementedError(f"{self.name} does not support streaming")

    async def validate(self, request: LLMRequest) -> bool:
        """Lightweight request validation."""
        return bool(request.messages) and all(bool(m.content) for m in request.messages)

    @property
    def models(self) -> list[str]:
        """Optional list of model ids served by this provider."""
        return []

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} model={self.default_model!r}>"
