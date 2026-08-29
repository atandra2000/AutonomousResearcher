"""Tests for the interactive chat session (TUI)."""

from __future__ import annotations

import asyncio
import stat
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

from rich.console import Console

from research_engineer.tui import ChatSession


def _make_console(
    llm: object | None = None,
    *,
    task_agent: object | None = None,
    repo_path: str | Path = ".",
    persist: bool = False,
    state_dir: str | Path = "output/sessions",
    session_id: str | None = None,
) -> tuple[ChatSession, StringIO]:
    buf = StringIO()
    console = Console(file=buf, width=120, legacy_windows=False)
    # use_router=False keeps tests offline: provider resolution happens
    # only when an explicit llm is supplied.
    return (
        ChatSession(
            console=console,
            llm=llm,
            task_agent=task_agent,
            repo_path=repo_path,
            persist=persist,
            state_dir=state_dir,
            session_id=session_id,
            use_router=False,
        ),
        buf,
    )


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
    for cmd in (
        "/task",
        "/repo",
        "/research",
        "/analyze",
        "/llm",
        "/clear",
        "/status",
        "/exit",
    ):
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


def test_freeform_turns_preserve_conversation_context() -> None:
    from research_engineer.llm.base import LLMResponse, LLMRole

    class _FakeProvider:
        provider_name = "fake"

        def __init__(self) -> None:
            self.requests = []

        async def complete(self, request):
            self.requests.append(request)
            return LLMResponse(
                content=f"answer {len(self.requests)}",
                model="fake-model",
                provider="fake",
            )

    provider = _FakeProvider()
    session, _ = _make_console(llm=provider)
    asyncio.run(session.handle_line("first question"))
    asyncio.run(session.handle_line("follow up"))

    messages = provider.requests[1].messages
    assert [message.role for message in messages] == [
        LLMRole.USER,
        LLMRole.ASSISTANT,
        LLMRole.USER,
    ]
    assert [message.content for message in messages] == [
        "first question",
        "answer 1",
        "follow up",
    ]
    assert session.turns == [
        ("llm", "first question"),
        ("llm", "follow up"),
    ]


def test_clear_resets_conversation_and_turns() -> None:
    session, buf = _make_console()
    session.turns.append(("llm", "old question"))
    session._conversation.extend(
        [("user", "old question"), ("assistant", "old answer")]
    )

    assert asyncio.run(session.handle_line("/clear")) is True
    assert session.turns == []
    assert session._conversation == []
    assert "cleared" in buf.getvalue().lower()


def test_session_state_roundtrip(tmp_path: Path) -> None:
    from research_engineer.llm.base import LLMResponse

    class _FakeProvider:
        provider_name = "fake"

        async def complete(self, request):
            return LLMResponse(
                content="persisted answer",
                model="fake-model",
                provider="fake",
            )

    repo = tmp_path / "repo"
    repo.mkdir()
    state_dir = tmp_path / "sessions"
    session, _ = _make_console(
        llm=_FakeProvider(),
        repo_path=repo,
        persist=True,
        state_dir=state_dir,
        session_id="session_test",
    )
    asyncio.run(session.handle_line("remember this"))

    restored = ChatSession.from_saved(
        "session_test",
        llm=_FakeProvider(),
        state_dir=state_dir,
        use_router=False,
    )

    assert restored.session_id == "session_test"
    assert restored.repo_path == repo.resolve()
    assert restored.turns == [("llm", "remember this")]
    assert restored._conversation == [
        ("user", "remember this"),
        ("assistant", "persisted answer"),
    ]
    state_file = state_dir / "session_test.json"
    assert stat.S_IMODE(state_file.stat().st_mode) == 0o600
    assert not list(state_dir.glob("*.tmp"))


def test_resume_rejects_unsafe_session_id(tmp_path: Path) -> None:
    try:
        ChatSession.from_saved("../outside", state_dir=tmp_path)
    except ValueError as exc:
        assert "session id" in str(exc).lower()
    else:
        raise AssertionError("unsafe session id was accepted")


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


def test_repo_command_changes_workspace(tmp_path: Path) -> None:
    target = tmp_path / "repo with spaces"
    target.mkdir()
    session, buf = _make_console(repo_path=tmp_path)

    assert asyncio.run(session.handle_line(f'/repo "{target}"')) is True
    assert session.repo_path == target.resolve()
    assert str(target.resolve()) in buf.getvalue().replace("\n", "")


def test_repo_command_rejects_non_directory(tmp_path: Path) -> None:
    target = tmp_path / "not-a-repo"
    target.write_text("file")
    session, buf = _make_console(repo_path=tmp_path)

    assert asyncio.run(session.handle_line(f"/repo {target}")) is True
    assert session.repo_path == tmp_path.resolve()
    assert "not a directory" in buf.getvalue().lower()


def test_task_runs_against_workspace_in_safe_mode(tmp_path: Path) -> None:
    class _FakeTaskAgent:
        def __init__(self) -> None:
            self.calls = []

        async def run(self, goal, repo_path, config=None, stream_sink=None):
            self.calls.append((goal, repo_path, config, stream_sink))
            return SimpleNamespace(
                task_id="task_test",
                status=SimpleNamespace(value="completed"),
                steps=[
                    SimpleNamespace(
                        step_type=SimpleNamespace(value="plan"),
                        status=SimpleNamespace(value="completed"),
                        summary="Plan ready",
                    )
                ],
                patches_generated=1,
                processing_time_seconds=0.2,
                delegated=False,
                repair_iterations=0,
                diff="+new code",
                test_exit_code=None,
                test_stdout="",
                test_stderr="",
                generated_files=["output/tasks/task_test/plan.md"],
                review_issues=[],
                test_failures=[],
                error=None,
            )

    agent = _FakeTaskAgent()
    session, buf = _make_console(task_agent=agent, repo_path=tmp_path)

    assert asyncio.run(session.handle_line("/task add RMSNorm")) is True

    goal, repo_path, config, stream_sink = agent.calls[0]
    assert goal == "add RMSNorm"
    assert repo_path == str(tmp_path.resolve())
    assert config.dry_run is True
    assert config.run_tests is False
    assert config.delegate is False
    assert stream_sink is not None
    assert "task_test" in buf.getvalue()
    assert "dry-run" in buf.getvalue().lower()
    assert session.turns == [("task", "add RMSNorm")]


def test_task_explicit_flags_enable_apply_tests_and_delegation(
    tmp_path: Path,
) -> None:
    class _FakeTaskAgent:
        def __init__(self) -> None:
            self.config = None

        async def run(self, goal, repo_path, config=None, stream_sink=None):
            self.config = config
            return SimpleNamespace(
                task_id="task_flags",
                status=SimpleNamespace(value="completed"),
                steps=[],
                patches_generated=0,
                processing_time_seconds=0.1,
                delegated=True,
                repair_iterations=1,
                diff="",
                test_exit_code=0,
                test_stdout="1 passed",
                test_stderr="",
                generated_files=[],
                review_issues=[],
                test_failures=[],
                error=None,
            )

    agent = _FakeTaskAgent()
    session, _ = _make_console(task_agent=agent, repo_path=tmp_path)
    line = "/task --apply --tests --delegate implement GQA"

    assert asyncio.run(session.handle_line(line)) is True
    assert agent.config.dry_run is False
    assert agent.config.run_tests is True
    assert agent.config.delegate is True


def test_task_rejects_unknown_option(tmp_path: Path) -> None:
    session, buf = _make_console(task_agent=object(), repo_path=tmp_path)
    assert asyncio.run(session.handle_line("/task --force do it")) is True
    assert "unknown option" in buf.getvalue().lower()


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

    chat_help = CliRunner().invoke(app, ["chat", "--help"])
    assert chat_help.exit_code == 0
    assert "--repo" in chat_help.output
    assert "--resume" in chat_help.output
