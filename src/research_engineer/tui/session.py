"""Codex-style interactive terminal session.

``research-engineer chat`` opens a project-aware REPL where slash commands
run the platform's coding and research agents and free-form text is routed
to the configured LLM provider with conversation context.
"""

from __future__ import annotations

import asyncio
import re
import shlex
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ValidationError
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

BANNER = """\
[bold cyan]research-engineer[/bold cyan] — autonomous ML research agent
Type a research question, or a command. [dim]([/dim]/help[dim] for the list, /exit to quit)[/dim]"""

MAX_REPORT_PREVIEW = 800
MAX_CONVERSATION_MESSAGES = 20
SESSION_SCHEMA_VERSION = 1
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


class SessionState(BaseModel):
    """Persisted interactive-session state."""

    schema_version: int = SESSION_SCHEMA_VERSION
    session_id: str
    repo_path: str
    turns: list[tuple[str, str]]
    conversation: list[tuple[str, str]]


class ChatSession:
    """Interactive REPL over the research platform's agents.

    Args:
        console: Optional pre-configured rich console (tests inject a
            file-backed console to capture output).
        llm: Optional explicit :class:`~research_engineer.llm.base.LLMProvider`
            (overrides router resolution for ``/llm`` and free-form turns).
    """

    def __init__(
        self,
        console: Console | None = None,
        llm: Any | None = None,
        *,
        task_agent: Any | None = None,
        repo_path: str | Path = ".",
        persist: bool = True,
        state_dir: str | Path = "output/sessions",
        session_id: str | None = None,
        use_router: bool = True,
    ) -> None:
        self.console = console or Console()
        self._explicit_llm = llm
        self._llm: Any | None = llm
        self._use_router = use_router
        self._llm_checked = llm is not None or not use_router
        self._task_agent = task_agent
        self.repo_path = Path(repo_path).expanduser().resolve()
        if not self.repo_path.is_dir():
            raise ValueError(f"Repository is not a directory: {self.repo_path}")
        self._persist = persist
        self.state_dir = Path(state_dir).expanduser().resolve()
        self.session_id = session_id or f"session_{uuid4().hex[:12]}"
        self._validate_session_id(self.session_id)
        self.turns: list[tuple[str, str]] = []
        self._conversation: list[tuple[str, str]] = []

    @classmethod
    def from_saved(
        cls,
        session_id: str,
        *,
        console: Console | None = None,
        llm: Any | None = None,
        task_agent: Any | None = None,
        state_dir: str | Path = "output/sessions",
        use_router: bool = True,
    ) -> ChatSession:
        """Restore a session by ID from its typed local state file."""
        cls._validate_session_id(session_id)
        directory = Path(state_dir).expanduser().resolve()
        path = directory / f"{session_id}.json"
        try:
            state = SessionState.model_validate_json(path.read_text())
        except FileNotFoundError as exc:
            raise ValueError(f"Session not found: {session_id}") from exc
        except (OSError, ValidationError) as exc:
            raise ValueError(f"Cannot load session {session_id}: {exc}") from exc
        if state.schema_version != SESSION_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported session schema {state.schema_version}; "
                f"expected {SESSION_SCHEMA_VERSION}"
            )
        session = cls(
            console=console,
            llm=llm,
            task_agent=task_agent,
            repo_path=state.repo_path,
            persist=True,
            state_dir=directory,
            session_id=state.session_id,
            use_router=use_router,
        )
        session.turns = state.turns
        session._conversation = state.conversation[-MAX_CONVERSATION_MESSAGES:]
        return session

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        if not SESSION_ID_PATTERN.fullmatch(session_id):
            raise ValueError(
                "Invalid session ID; use only letters, numbers, '_' and '-'"
            )

    def _save_state(self) -> None:
        if not self._persist:
            return
        self.state_dir.mkdir(parents=True, exist_ok=True)
        path = self.state_dir / f"{self.session_id}.json"
        temporary = path.with_suffix(".json.tmp")
        state = SessionState(
            session_id=self.session_id,
            repo_path=str(self.repo_path),
            turns=self.turns,
            conversation=self._conversation,
        )
        # ponytail: one writer per session; add a lock if concurrent resume
        # becomes a supported workflow.
        temporary.write_text(state.model_dump_json(indent=2))
        temporary.chmod(0o600)
        temporary.replace(path)

    def _persist_state(self) -> None:
        try:
            self._save_state()
        except OSError as exc:
            self.console.print(f"[yellow]Session state was not saved:[/yellow] {exc}")

    # ------------------------------------------------------------------
    # Input loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Run the interactive loop until ``/exit`` or EOF."""
        self._print_banner()
        prompt_session = self._make_prompt_session()
        while True:
            try:
                if prompt_session is not None:
                    line = await prompt_session.prompt_async(
                        "research-engineer> "
                    )
                else:
                    line = await asyncio.to_thread(input, "research-engineer> ")
            except (EOFError, KeyboardInterrupt):
                self.console.print("\n[dim]Session ended.[/dim]")
                return
            keep_going = await self.handle_line(line)
            if not keep_going:
                self.console.print("[dim]Goodbye — happy researching.[/dim]")
                return

    @staticmethod
    def _make_prompt_session() -> Any | None:
        """Prefer prompt_toolkit (history/editing); fall back to input()."""
        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.history import InMemoryHistory

            return PromptSession(history=InMemoryHistory())
        except Exception:  # pragma: no cover - depends on env
            return None

    def _print_banner(self) -> None:
        self.console.print(Panel(BANNER, border_style="cyan"))
        self.console.print(f"[dim]Workspace: {self.repo_path}[/dim]")
        self.console.print(f"[dim]Session: {self.session_id}[/dim]")

    # ------------------------------------------------------------------
    # Command dispatch
    # ------------------------------------------------------------------

    async def handle_line(self, line: str) -> bool:
        """Process one input line; return ``False`` to end the session."""
        stripped = line.strip()
        if not stripped:
            return True
        if stripped in {"/exit", "/quit", "exit", "quit"}:
            return False
        if stripped in {"/help", "help", "?"}:
            self._show_help()
            return True
        if stripped.startswith("/"):
            command, _, rest = stripped[1:].partition(" ")
            handler = getattr(self, f"_cmd_{command}", None)
            if handler is None:
                self.console.print(
                    f"[red]Unknown command[/red] /{command} — try /help"
                )
                return True
            await handler(rest.strip())
            self._persist_state()
            return True
        await self._ask_llm(stripped)
        self._persist_state()
        return True

    def _show_help(self) -> None:
        table = Table(show_header=False, box=None, pad_edge=False)
        table.add_column(style="bold cyan", no_wrap=True)
        table.add_column()
        table.add_row(
            "/task <goal>",
            "Run a coding turn (--apply, --tests, --delegate)",
        )
        table.add_row("/repo [path]", "Show or change the active repository")
        table.add_row("/research <goal>", "Run the full autonomous "
                      "research workflow (LangGraph engine)")
        table.add_row("/analyze <paper>", "Analyze an arXiv ID/URL/PDF")
        table.add_row("/llm <prompt>", "Ask the configured LLM directly")
        table.add_row("/clear", "Clear conversation context and turn history")
        table.add_row("/status", "Show session and LLM status")
        table.add_row("/help", "Show this help")
        table.add_row("/exit", "Leave the session")
        self.console.print(table)
        self.console.print(
            "[dim]/task is dry-run by default; use --apply to write changes. "
            "Free-form lines continue the LLM conversation.[/dim]"
        )

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    async def _cmd_repo(self, path: str) -> None:
        if not path:
            self.console.print(f"[bold]Workspace:[/bold] {self.repo_path}")
            return
        try:
            parts = shlex.split(path)
        except ValueError as exc:
            self.console.print(f"[red]Invalid repository path:[/red] {exc}")
            return
        if len(parts) != 1:
            self.console.print("[yellow]Usage:[/yellow] /repo <path>")
            return
        candidate = Path(parts[0]).expanduser().resolve()
        if not candidate.is_dir():
            self.console.print(
                f"[red]Repository is not a directory:[/red] {candidate}"
            )
            return
        self.repo_path = candidate
        self.console.print(f"[green]Workspace changed:[/green] {candidate}")

    async def _cmd_task(self, raw_args: str) -> None:
        try:
            args = shlex.split(raw_args)
        except ValueError as exc:
            self.console.print(f"[red]Invalid task command:[/red] {exc}")
            return

        flags = {arg for arg in args if arg.startswith("--")}
        supported = {"--apply", "--tests", "--delegate"}
        unknown = flags - supported
        if unknown:
            self.console.print(
                f"[red]Unknown option:[/red] {', '.join(sorted(unknown))}"
            )
            return
        goal = " ".join(arg for arg in args if arg not in supported).strip()
        if not goal:
            self.console.print(
                "[yellow]Usage:[/yellow] /task [--apply] [--tests] "
                "[--delegate] <coding goal>"
            )
            return

        from research_engineer.models.task import TaskConfig

        if self._task_agent is None:
            try:
                from research_engineer.agents import TaskAgent

                self._task_agent = TaskAgent(llm=self._ensure_llm())
            except Exception as exc:
                self.console.print(f"[red]Task agent unavailable:[/red] {exc}")
                return

        dry_run = "--apply" not in flags
        run_tests = "--tests" in flags
        delegate = "--delegate" in flags
        mode = "dry-run" if dry_run else "apply"
        config = TaskConfig(
            goal=goal,
            repo_path=str(self.repo_path),
            dry_run=dry_run,
            run_tests=run_tests,
            delegate=delegate,
        )
        self.console.print(
            f"[cyan]▸ Coding task ({mode}):[/cyan] {goal}\n"
            f"[dim]Workspace: {self.repo_path}[/dim]"
        )
        try:
            result = await self._task_agent.run(
                goal,
                str(self.repo_path),
                config=config,
                stream_sink=self.console.file,
            )
        except Exception as exc:
            self.console.print(f"[red]Task failed:[/red] {exc}")
            return

        self.turns.append(("task", goal))
        table = Table(title=f"Task {result.task_id}", box=None)
        table.add_column("Step")
        table.add_column("Status")
        table.add_column("Summary")
        for step in result.steps:
            status = step.status.value
            style = "green" if status == "completed" else "red"
            table.add_row(
                step.step_type.value,
                f"[{style}]{status}[/{style}]",
                step.summary,
            )
        self.console.print(table)
        self.console.print(
            f"Status: [bold]{result.status.value}[/bold] · "
            f"Mode: {mode} · Patches: {result.patches_generated} · "
            f"Time: {result.processing_time_seconds}s"
        )
        if result.delegated:
            self.console.print(
                f"Delegated · repair iterations: {result.repair_iterations}"
            )
        if result.diff:
            self.console.print(
                Panel(result.diff[:4000], title="Diff", border_style="cyan")
            )
        if result.test_exit_code is not None:
            test_style = "green" if result.test_exit_code == 0 else "red"
            self.console.print(
                f"[{test_style}]Tests exited {result.test_exit_code}"
                f"[/{test_style}]"
            )
            if result.test_stdout:
                self.console.print(result.test_stdout[:2000])
        if result.generated_files:
            self.console.print(
                "[bold]Artifacts:[/bold] " + ", ".join(result.generated_files)
            )
        if result.error:
            self.console.print(f"[red]Error:[/red] {result.error}")

    async def _cmd_clear(self, _: str) -> None:
        self.turns.clear()
        self._conversation.clear()
        self.console.print("[green]Conversation cleared.[/green]")

    async def _cmd_research(self, goal: str) -> None:
        if not goal:
            self.console.print(
                "[yellow]Usage:[/yellow] /research <research goal>"
            )
            return
        from research_engineer.agents import (
            ResearchConfig,
            ResearchOrchestrator,
        )

        orchestrator = ResearchOrchestrator()
        config = ResearchConfig()
        self.console.print(
            f"[cyan]▸ Running research workflow:[/cyan] {goal}"
        )
        try:
            result = await orchestrator.run(
                goal, str(self.repo_path), config=config
            )
        except Exception as exc:
            self.console.print(
                f"[red]Research workflow failed:[/red] {exc}"
            )
            return
        self.turns.append(("research", goal))
        table = Table(title="Stages", box=None)
        table.add_column("Stage")
        table.add_column("Status")
        table.add_column("Duration (s)", justify="right")
        for stage in result.stages:
            status = stage.status.value
            style = (
                "green" if status == "completed"
                else "yellow" if status == "skipped"
                else "red"
            )
            table.add_row(
                stage.stage_type.value,
                f"[{style}]{status}[/{style}]",
                str(stage.duration_seconds or ""),
            )
        self.console.print(table)
        self.console.print(
            f"Papers: {result.papers_found} · "
            f"Hypotheses: {result.hypotheses_generated} · "
            f"Experiments: {result.experiments_run}"
        )
        if result.report_path:
            self.console.print(f"[bold]Report:[/bold] {result.report_path}")
        if result.final_report:
            preview = result.final_report[:MAX_REPORT_PREVIEW]
            self.console.print(
                Panel(
                    Markdown(preview),
                    title="Report preview",
                    border_style="cyan",
                )
            )


    async def _cmd_analyze(self, paper: str) -> None:
        if not paper:
            self.console.print(
                "[yellow]Usage:[/yellow] /analyze <arxiv_id | url | pdf>"
            )
            return
        from research_engineer.agents import ResearchAgent

        self.console.print(f"[cyan]▸ Analyzing:[/cyan] {paper}")
        try:
            result = await ResearchAgent().analyze(paper)
        except Exception as exc:
            self.console.print(f"[red]Analysis failed:[/red] {exc}")
            return
        self.turns.append(("analyze", paper))
        payload = result if isinstance(result, dict) else {}
        summary = payload.get("summary", {})
        lines = [f"[bold]{payload.get('title', paper)}[/bold]", ""]
        for i, contrib in enumerate(
            (summary.get("core_contributions") or [])[:3], 1
        ):
            lines.append(f"{i}. {contrib}")
        if summary.get("executive_summary"):
            lines += ["", str(summary["executive_summary"])]
        self.console.print(
            Panel(
                Markdown("\n".join(lines)),
                title="Analysis",
                border_style="cyan",
            )
        )
        if payload.get("output_dir"):
            self.console.print(
                f"[dim]Full output: {payload['output_dir']}[/dim]"
            )


    async def _cmd_llm(self, prompt: str) -> None:
        if not prompt:
            self.console.print("[yellow]Usage:[/yellow] /llm <prompt>")
            return
        await self._ask_llm(prompt)

    async def _cmd_status(self, _: str) -> None:
        provider = self._ensure_llm()
        if provider is not None:
            name = getattr(provider, "provider_name", None)
            provider_desc = str(name) if name else type(provider).__name__
        else:
            provider_desc = (
                "[yellow]unavailable[/yellow] (configure llm_config.yaml "
                "or set OLLAMA_API_KEY — rule-based agents still work)"
            )
        table = Table(show_header=False, box=None, pad_edge=False)
        table.add_column(style="bold cyan", no_wrap=True)
        table.add_column()
        table.add_row("Session", self.session_id)
        table.add_row("Workspace", str(self.repo_path))
        table.add_row("LLM provider", provider_desc)
        table.add_row("Turns this session", str(len(self.turns)))
        table.add_row("Task writes", "dry-run unless --apply is explicit")
        self.console.print(table)

    # ------------------------------------------------------------------
    # LLM plumbing
    # ------------------------------------------------------------------

    def _ensure_llm(self) -> Any | None:
        """Resolve the LLM provider once; ``None`` when unavailable."""
        if not self._llm_checked:
            self._llm_checked = True
            if self._llm is None and self._use_router:
                try:
                    from research_engineer.agents._llm_support import (
                        resolve_llm,
                    )

                    self._llm = resolve_llm("ChatSession", None)
                except Exception:  # pragma: no cover - config dependent
                    self._llm = None
        return self._llm

    async def _ask_llm(self, prompt: str) -> None:
        provider = self._ensure_llm()
        if provider is None:
            self.console.print(
                "[yellow]No LLM provider configured.[/yellow] Autonomous "
                "commands (/research, /analyze) work without one; to chat, "
                "configure llm_config.yaml or set OLLAMA_API_KEY."
            )
            return
        try:
            from research_engineer.llm.base import (
                LLMMessage,
                LLMRequest,
                LLMRole,
            )

            history = self._conversation[-MAX_CONVERSATION_MESSAGES:]
            messages = [
                LLMMessage(role=LLMRole(role), content=content)
                for role, content in history
            ]
            messages.append(LLMMessage(role=LLMRole.USER, content=prompt))
            response = await provider.complete(LLMRequest(messages=messages))
        except Exception as exc:
            self.console.print(f"[red]LLM error:[/red] {exc}")
            return
        self.turns.append(("llm", prompt))
        self._conversation.extend(
            [(LLMRole.USER.value, prompt), (LLMRole.ASSISTANT.value, response.content)]
        )
        self._conversation = self._conversation[-MAX_CONVERSATION_MESSAGES:]
        self.console.print(
            Panel(
                Markdown(response.content),
                title=f"{response.provider} · {response.model}",
                border_style="cyan",
            )
        )


__all__ = ["ChatSession"]
