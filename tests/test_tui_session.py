"""Tests for the interactive chat session (TUI)."""

from __future__ import annotations

import asyncio
from io import StringIO
from types import SimpleNamespace

from rich.console import Console

from research_engineer.tui import ChatSession


def _make_console(llm: object | None = None) -> tuple[ChatSession, StringIO]:
    buf = StringIO()
    console = Console(file=buf, width=120, legacy_windows=False)
    # use_router=False keeps tests offline: provider resolution happens
    # only when an explicit llm is supplied.
    return ChatSession(console=console, llm=llm, use_router=False), buf


def test_unknown_command_reports_error() -> None:
    session, buf = _make_console()
    keep = asyncio.run(session.handle_line("/frobnicate"))
    assert keep is True
    assert "Unknown command" in buf.getvalue()
    assert "frobnicate" in buf.getvalue()


def test_exit_command_ends_session() -> None:
    session, _ = _make_console()
    assert asyncio.run(session.handle_line("/exit")) is False
    assert asyncio.run(session.handle_line("quit")) is False


def test_help_lists_commands() -> None:
    session, buf = _make_console()
    keep = asyncio.run(session.handle_line("/help"))
    assert keep is True
    text = buf.getvalue()
    for cmd in ("/research", "/analyze", "/llm", "/status", "/exit"):
        assert cmd in text


def test_blank_line_is_ignored() -> None:
    session, buf = _make_console()
    assert asyncio.run(session.handle_line("   ")) is True
    assert buf.getvalue() == ""


def test_freeform_without_llm_prints_hint() -> None:
    session, buf = _make_console()
    keep = asyncio.run(session.handle_line("What is attention?"))
    assert keep is True
    assert "No LLM provider configured" in buf.getvalue()


def test_llm_receives_prompt_and_renders_response() -> None:
    from research_engineer.llm.base import LLMResponse

    class _FakeProvider:
        provider_name = "fake"

        async def complete(self, request):
            return LLMResponse(
                content="Attention is all you need.",
                model="fake-model",
                provider="fake",
            )

    session, buf = _make_console(llm=_FakeProvider())
    keep = asyncio.run(session.handle_line("/llm explain attention"))
    assert keep is True
    text = buf.getvalue()
    assert "Attention is all you need." in text
    assert "fake-model" in text
    assert session.turns == [("llm", "explain attention")]


def test_llm_error_is_contained() -> None:
    class _BrokenProvider:
        async def complete(self, request):
            raise RuntimeError("provider down")

    session, buf = _make_console(llm=_BrokenProvider())
    keep = asyncio.run(session.handle_line("hello?"))
    assert keep is True
    assert "LLM error" in buf.getvalue()
    assert "provider down" in buf.getvalue()


def test_research_command_renders_stages_and_report(monkeypatch) -> None:
    class _FakeOrchestrator:
        async def run(self, goal, repo, config=None):
            stages = [
                SimpleNamespace(
                    stage_type=SimpleNamespace(value="literature_discovery"),
                    status=SimpleNamespace(value="completed"),
                    duration_seconds=0.1,
                ),
                SimpleNamespace(
                    stage_type=SimpleNamespace(value="report_generation"),
                    status=SimpleNamespace(value="completed"),
                    duration_seconds=0.2,
                ),
            ]
            return SimpleNamespace(
                stages=stages,
                papers_found=3,
                hypotheses_generated=2,
                experiments_run=2,
                report_path="output/research/wf/report.md",
                final_report="# Findings\nBetter attention.",
            )

    monkeypatch.setattr(
        "research_engineer.agents.ResearchOrchestrator",
        _FakeOrchestrator,
    )
    session, buf = _make_console()
    keep = asyncio.run(session.handle_line("/research efficient attention"))
    assert keep is True
    text = buf.getvalue()
    assert "literature_discovery" in text
    assert "output/research/wf/report.md" in text
    assert "Better attention." in text
    assert session.turns == [("research", "efficient attention")]


def test_research_without_goal_prints_usage() -> None:
    session, buf = _make_console()
    keep = asyncio.run(session.handle_line("/research"))
    assert keep is True
    assert "Usage" in buf.getvalue()


def test_analyze_renders_summary(monkeypatch) -> None:
    class _FakeAgent:
        async def analyze(self, paper, output_dir="output"):
            return {
                "title": "FlashAttention",
                "summary": {
                    "core_contributions": ["IO-aware attention"],
                    "executive_summary": "Faster exact attention.",
                },
                "output_dir": "output/analyze",
            }

    monkeypatch.setattr("research_engineer.agents.ResearchAgent", _FakeAgent)
    session, buf = _make_console()
    keep = asyncio.run(session.handle_line("/analyze 2503.12345"))
    assert keep is True
    text = buf.getvalue()
    assert "FlashAttention" in text
    assert "IO-aware attention" in text
    assert session.turns == [("analyze", "2503.12345")]


def test_cli_registers_chat_command() -> None:
    from typer.testing import CliRunner

    from research_engineer.cli import app

    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "chat" in result.output
