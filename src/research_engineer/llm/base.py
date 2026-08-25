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


class ToolDefinition(BaseModel):
    """A tool the model may call during a completion.

    Mirrors the OpenAI ``tools`` array entry: ``type`` is always ``"function"``
    and ``function`` carries the name, description, and a JSON-Schema
    ``parameters`` object describing the expected arguments.
    """

    name: str = Field(..., description="Unique tool name")
    description: str = Field(default="", description="Human-readable tool description")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON-Schema describing the tool's arguments",
    )

    def to_openai(self) -> dict[str, Any]:
        """Return the OpenAI-compatible ``tools`` entry for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolCall(BaseModel):
    """A single tool invocation requested by the model."""

    id: str = Field(..., description="Tool call id (echoed back in tool results)")
    name: str = Field(..., description="Name of the tool to invoke")
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments for the tool (parsed from JSON)",
    )


class ToolResult(BaseModel):
    """The outcome of executing a tool call, fed back to the model."""

    tool_call_id: str = Field(..., description="Id of the originating tool call")
    name: str = Field(..., description="Name of the tool that was executed")
    content: str = Field(..., description="Serialized tool output")
    is_error: bool = Field(default=False, description="True when the tool failed")


class LLMMessage(BaseModel):
    """A single chat-completion message."""

    role: LLMRole = Field(..., description="Message role")
    content: str = Field(..., description="Message content")
    name: str | None = Field(default=None, description="Optional author name")
    tool_call_id: str | None = Field(
        default=None,
        description="Optional tool call id (for tool-role messages)",
    )
    tool_calls: list[ToolCall] | None = Field(
        default=None,
        description="Tool calls requested by the assistant (assistant-role only)",
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
    tool_calls: list[ToolCall] | None = Field(
        default=None,
        description="Tool calls requested by the model (when tool calling is used)",
    )
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

    async def complete_with_tools(
        self,
        request: LLMRequest,
        tools: list[ToolDefinition],
    ) -> LLMResponse:
        """Generate a completion that may request tool calls.

        Providers MAY override this to expose native function-calling. The
        default implementation raises :class:`NotImplementedError` so callers
        can detect that tool calling is unsupported and fall back to plain
        :meth:`complete`.
        """
        raise NotImplementedError(
            f"{self.name} does not support tool calling"
        )

    async def stream(self, request: LLMRequest) -> Any:
        """Yield streamed completion chunks.

        Providers MAY override this. The default implementation raises
        :class:`NotImplementedError` to signal that streaming is unsupported.
        """
        raise NotImplementedError(f"{self.name} does not support streaming")

    async def validate(self, request: LLMRequest) -> bool:
        """Lightweight request validation."""
        return bool(request.messages) and all(bool(m.content) for m in request.messages)

    async def health(self) -> bool:
        """Probe whether this provider is currently reachable/healthy.

        Providers MAY override this to perform a lightweight liveness check
        (e.g. a ``/v1/models`` or ``/v1/messages`` ping). The default
        implementation returns ``True`` so that providers without a health
        probe are always considered available; the router's failover logic
        relies on this to decide whether to fall back to a secondary provider.
        """
        return True

    @property
    def models(self) -> list[str]:
        """Optional list of model ids served by this provider."""
        return []

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} model={self.default_model!r}>"
