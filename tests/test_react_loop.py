"""Tests for the ReAct agent loop (A2)."""

from __future__ import annotations

import pytest

from research_engineer.llm import (
    LLMMessage,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMRole,
    ReActLoopConfig,
    ReActResult,
    ReActStep,
    ReActTermination,
    ToolCall,
    ToolDefinition,
    ToolResult,
    run_react_loop,
)


class _ScriptedProvider(LLMProvider):
    """Provider that returns a scripted sequence of responses."""

    name = "scripted"

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[LLMRequest] = []

    async def complete_with_tools(
        self, request: LLMRequest, tools: list[ToolDefinition]
    ) -> LLMResponse:
        self.calls.append(request)
        if not self.responses:
            raise AssertionError("No more scripted responses")
        return self.responses.pop(0)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        # Not used by the loop, but required by the abstract base class.
        raise AssertionError("complete() should not be called by the ReAct loop")


def _resp(content: str, tool_calls: list[ToolCall] | None = None) -> LLMResponse:
    return LLMResponse(
        content=content,
        model="fake-model",
        provider="scripted",
        tool_calls=tool_calls,
    )


def _tool_call(name: str, call_id: str = "c1") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments={})


async def _echo_executor(call: ToolCall) -> ToolResult:
    return ToolResult(
        tool_call_id=call.id, name=call.name, content=f"result:{call.name}"
    )


def _messages() -> list[LLMMessage]:
    return [LLMMessage(role=LLMRole.USER, content="do the thing")]


TOOLS = [ToolDefinition(name="f", description="", parameters={"type": "object"})]


class TestReActLoop:
    @pytest.mark.asyncio
    async def test_multi_step_tool_roundtrip(self) -> None:
        provider = _ScriptedProvider(
            [
                _resp("", [_tool_call("f", "c1")]),
                _resp("FINAL_ANSWER: done"),
            ]
        )
        result = await run_react_loop(provider, _messages(), TOOLS, _echo_executor)
        assert result.termination == ReActTermination.FINAL_ANSWER
        assert result.num_steps == 2
        assert result.response.content == "FINAL_ANSWER: done"
        # Tool result was fed back as a tool-role message.
        tool_msgs = [m for m in result.messages if m.role == LLMRole.TOOL]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].content == "result:f"
        assert tool_msgs[0].tool_call_id == "c1"
        # The assistant message carried the tool call.
        assistant = [m for m in result.messages if m.role == LLMRole.ASSISTANT]
        assert assistant[0].tool_calls is not None
        assert assistant[0].tool_calls[0].name == "f"
        # Step trace recorded.
        assert len(result.steps) == 1
        assert result.steps[0].tool_calls[0].name == "f"
        assert result.steps[0].tool_results[0].content == "result:f"

    @pytest.mark.asyncio
    async def test_final_answer_termination_single_step(self) -> None:
        provider = _ScriptedProvider([_resp("FINAL_ANSWER: yes")])
        result = await run_react_loop(provider, _messages(), TOOLS, _echo_executor)
        assert result.termination == ReActTermination.FINAL_ANSWER
        assert result.num_steps == 1
        assert result.steps == []

    @pytest.mark.asyncio
    async def test_no_tool_calls_termination(self) -> None:
        provider = _ScriptedProvider([_resp("plain answer")])
        result = await run_react_loop(provider, _messages(), TOOLS, _echo_executor)
        assert result.termination == ReActTermination.NO_TOOL_CALLS
        assert result.num_steps == 1
        assert result.response.content == "plain answer"

    @pytest.mark.asyncio
    async def test_max_steps_exhaustion(self) -> None:
        # Always requests a tool call; never terminates on its own.
        provider = _ScriptedProvider(
            [_resp("", [_tool_call("f", f"c{i}")]) for i in range(3)]
        )
        config = ReActLoopConfig(max_steps=3)
        result = await run_react_loop(
            provider, _messages(), TOOLS, _echo_executor, config
        )
        assert result.termination == ReActTermination.MAX_STEPS
        assert result.num_steps == 3
        assert len(result.steps) == 3

    @pytest.mark.asyncio
    async def test_executor_error_propagates(self) -> None:
        async def boom(call: ToolCall) -> ToolResult:
            raise RuntimeError("tool exploded")

        provider = _ScriptedProvider([_resp("", [_tool_call("f")])])
        with pytest.raises(RuntimeError, match="tool exploded"):
            await run_react_loop(provider, _messages(), TOOLS, boom)

    @pytest.mark.asyncio
    async def test_on_step_hook_invoked(self) -> None:
        seen: list[ReActStep] = []
        provider = _ScriptedProvider(
            [
                _resp("", [_tool_call("f", "c1")]),
                _resp("", [_tool_call("f", "c2")]),
                _resp("FINAL_ANSWER: done"),
            ]
        )
        config = ReActLoopConfig(on_step=seen.append)
        result = await run_react_loop(
            provider, _messages(), TOOLS, _echo_executor, config
        )
        assert len(seen) == 2
        assert seen[0].step == 1
        assert seen[1].step == 2
        assert result.termination == ReActTermination.FINAL_ANSWER

    @pytest.mark.asyncio
    async def test_max_tool_calls_per_step_cap(self) -> None:
        provider = _ScriptedProvider(
            [
                _resp(
                    "",
                    [
                        _tool_call("f", "c1"),
                        _tool_call("f", "c2"),
                        _tool_call("f", "c3"),
                    ],
                ),
                _resp("FINAL_ANSWER: done"),
            ]
        )
        config = ReActLoopConfig(max_tool_calls_per_step=2)
        result = await run_react_loop(
            provider, _messages(), TOOLS, _echo_executor, config
        )
        # Only the first two tool calls were executed.
        assert len(result.steps[0].tool_calls) == 2
        assert result.steps[0].tool_calls[0].id == "c1"
        assert result.steps[0].tool_calls[1].id == "c2"
        tool_msgs = [m for m in result.messages if m.role == LLMRole.TOOL]
        assert len(tool_msgs) == 2

    @pytest.mark.asyncio
    async def test_stop_on_final_false_continues(self) -> None:
        # Content contains the marker but stop_on_final is False and a tool
        # call is requested → the loop continues instead of stopping.
        provider = _ScriptedProvider(
            [
                _resp("FINAL_ANSWER: not final yet", [_tool_call("f", "c1")]),
                _resp("plain answer"),
            ]
        )
        config = ReActLoopConfig(stop_on_final=False)
        result = await run_react_loop(
            provider, _messages(), TOOLS, _echo_executor, config
        )
        assert result.num_steps == 2
        assert result.termination == ReActTermination.NO_TOOL_CALLS

    def test_react_loop_config_defaults(self) -> None:
        cfg = ReActLoopConfig()
        assert cfg.max_steps == 8
        assert cfg.max_tool_calls_per_step == 4
        assert cfg.stop_on_final is True
        assert cfg.final_marker == "FINAL_ANSWER:"
        assert cfg.on_step is None

    def test_react_termination_values(self) -> None:
        assert ReActTermination.FINAL_ANSWER.value == "final_answer"
        assert ReActTermination.NO_TOOL_CALLS.value == "no_tool_calls"
        assert ReActTermination.MAX_STEPS.value == "max_steps"

    def test_react_result_roundtrip(self) -> None:
        resp = _resp("FINAL_ANSWER: done")
        result = ReActResult(
            response=resp,
            steps=[],
            messages=_messages(),
            termination=ReActTermination.FINAL_ANSWER,
            num_steps=1,
        )
        d = result.model_dump()
        assert d["termination"] == "final_answer"
        assert d["num_steps"] == 1

