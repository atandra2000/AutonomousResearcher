"""ReAct (Reason + Act) agent loop.

A reusable loop that lets any agent iterate: call the model → execute the
requested tools via an injected executor → feed the results back as ``tool``-role
messages → repeat until the model produces a final answer, stops requesting
tools, or the step budget is exhausted.

The loop is provider-agnostic: it talks only to the :class:`LLMProvider` ABC and
the injected ``executor`` callable, so it works with any provider that implements
:meth:`LLMProvider.complete_with_tools`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from enum import StrEnum

from pydantic import BaseModel, Field

from research_engineer.llm.base import (
    LLMMessage,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMRole,
    ToolCall,
    ToolDefinition,
    ToolResult,
)


class ReActTermination(StrEnum):
    """Why a ReAct loop stopped."""

    FINAL_ANSWER = "final_answer"
    NO_TOOL_CALLS = "no_tool_calls"
    MAX_STEPS = "max_steps"


class ReActStep(BaseModel):
    """A single model-call iteration of the loop."""

    step: int = Field(..., description="One-based step number")
    messages: list[LLMMessage] = Field(
        ..., description="Conversation snapshot after this step's tool execution"
    )
    tool_calls: list[ToolCall] = Field(
        ..., description="Tool calls requested by the model this step"
    )
    tool_results: list[ToolResult] = Field(
        ..., description="Results of executing the requested tool calls"
    )
    response: LLMResponse = Field(..., description="The model response for this step")


class ReActLoopConfig(BaseModel):
    """Configuration for :func:`run_react_loop`."""

    max_steps: int = Field(
        default=8, ge=1, description="Maximum model calls before stopping"
    )
    max_tool_calls_per_step: int = Field(
        default=4, ge=1, description="Max tools executed per step"
    )
    stop_on_final: bool = Field(
        default=True, description="Stop when the final marker appears"
    )
    final_marker: str = Field(
        default="FINAL_ANSWER:", description="Marker signalling a final answer"
    )
    on_step: Callable[[ReActStep], None] | None = Field(
        default=None, description="Observability hook invoked after each step"
    )
    stream: bool = Field(
        default=False,
        description="Stream each model call and assemble the response via "
        "``stream_response`` when the provider supports it",
    )


class ReActResult(BaseModel):
    """The outcome of a ReAct loop run."""

    response: LLMResponse = Field(..., description="Final model response")
    steps: list[ReActStep] = Field(
        ..., description="Full step trace (for observability)"
    )
    messages: list[LLMMessage] = Field(..., description="Final conversation")
    termination: ReActTermination = Field(..., description="Why the loop stopped")
    num_steps: int = Field(..., ge=0, description="Number of model calls made")


async def run_react_loop(
    provider: LLMProvider,
    messages: list[LLMMessage],
    tools: list[ToolDefinition],
    executor: Callable[[ToolCall], Awaitable[ToolResult]],
    config: ReActLoopConfig | None = None,
) -> ReActResult:
    """Run a ReAct loop until a final answer, no tool calls, or max steps.

    Each iteration calls ``provider.complete_with_tools`` with the current
    conversation. If the model requests tool calls, they are executed via
    ``executor`` (capped at ``max_tool_calls_per_step``) and the results are
    appended as ``tool``-role messages before the next call.

    The loop stops when:
    - ``final_marker`` appears in the response content (if ``stop_on_final``),
    - the model requests no tool calls, or
    - ``max_steps`` model calls have been made.

    Exceptions raised by ``executor`` or ``provider`` propagate to the caller.
    """
    cfg = config or ReActLoopConfig()
    conversation = list(messages)
    steps: list[ReActStep] = []
    response: LLMResponse | None = None

    for step in range(1, cfg.max_steps + 1):
        request = LLMRequest(messages=conversation)
        if cfg.stream:
            # Stream the call and assemble the full response. Falls back to
            # ``complete_with_tools`` when the provider lacks a streaming path.
            streamer = getattr(provider, "stream_response", None)
            if streamer is not None:
                response = await streamer(request)
            else:
                response = await provider.complete_with_tools(request, tools)
        else:
            response = await provider.complete_with_tools(request, tools)
        tool_calls = response.tool_calls or []

        # Append the assistant message (with any tool calls) to the conversation.
        conversation.append(
            LLMMessage(
                role=LLMRole.ASSISTANT,
                content=response.content,
                tool_calls=tool_calls or None,
            )
        )

        if cfg.stop_on_final and cfg.final_marker in response.content:
            return _finish(
                response, steps, conversation, ReActTermination.FINAL_ANSWER, step
            )

        if not tool_calls:
            return _finish(
                response, steps, conversation, ReActTermination.NO_TOOL_CALLS, step
            )

        executed = tool_calls[: cfg.max_tool_calls_per_step]
        tool_results: list[ToolResult] = []
        for call in executed:
            result = await executor(call)
            tool_results.append(result)
            conversation.append(
                LLMMessage(
                    role=LLMRole.TOOL,
                    content=result.content,
                    tool_call_id=result.tool_call_id,
                    name=result.name,
                )
            )

        steps.append(
            ReActStep(
                step=step,
                messages=list(conversation),
                tool_calls=executed,
                tool_results=tool_results,
                response=response,
            )
        )
        if cfg.on_step is not None:
            cfg.on_step(steps[-1])

    assert response is not None  # max_steps >= 1 guarantees at least one call
    return _finish(
        response, steps, conversation, ReActTermination.MAX_STEPS, cfg.max_steps
    )


def _finish(
    response: LLMResponse,
    steps: list[ReActStep],
    conversation: list[LLMMessage],
    termination: ReActTermination,
    num_steps: int,
) -> ReActResult:
    """Build a :class:`ReActResult` from the loop's final state."""
    return ReActResult(
        response=response,
        steps=steps,
        messages=conversation,
        termination=termination,
        num_steps=num_steps,
    )
