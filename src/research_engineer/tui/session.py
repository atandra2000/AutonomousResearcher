"""Codex-style interactive terminal session.

``research-engineer chat`` opens a REPL where each line is either a slash
command (``/research``, ``/analyze``, ``/llm``, ``/status``, ``/help``)
or free-form text routed to the configured LLM provider. Results render
as rich panels/tables — the same agents the one-shot CLI commands use,
in a conversational loop.
"""

from __future__ import annotations

import asyncio
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

BANNER = """\
[bold cyan]research-engineer[/bold cyan] — autonomous ML research agent
Type a research question, or a command. [dim]([/dim]/help[dim] for the list, /exit to quit)[/dim]"""

MAX_REPORT_PREVIEW = 800


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
        use_router: bool = True,
    ) -> None:
        self.console = console or Console()
        self._explicit_llm = llm
        self._llm: Any | None = llm
        self._use_router = use_router
        self._llm_checked = llm is not None or not use_router
        self.turns: list[tuple[str, str]] = []

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
            return True
        await self._ask_llm(stripped)
        return True

    def _show_help(self) -> None:
        table = Table(show_header=False, box=None, pad_edge=False)
        table.add_column(style="bold cyan", no_wrap=True)
        table.add_column()
        table.add_row("/research <goal>", "Run the full autonomous "
                      "research workflow (LangGraph engine)")
        table.add_row("/analyze <paper>", "Analyze an arXiv ID/URL/PDF")
        table.add_row("/llm <prompt>", "Ask the configured LLM directly")
        table.add_row("/status", "Show session and LLM status")
        table.add_row("/help", "Show this help")
        table.add_row("/exit", "Leave the session")
        self.console.print(table)
        self.console.print(
            "[dim]Free-form lines go straight to the LLM.[/dim]"
        )


    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

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
            result = await orchestrator.run(goal, ".", config=config)
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
        await self._ask_llm(prompt, record=True)

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
        table.add_row("LLM provider", provider_desc)
        table.add_row("Turns this session", str(len(self.turns)))
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

    async def _ask_llm(self, prompt: str, *, record: bool = False) -> None:
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

            response = await provider.complete(
                LLMRequest(
                    messages=[
                        LLMMessage(role=LLMRole.USER, content=prompt),
                    ],
                )
            )
        except Exception as exc:
            self.console.print(f"[red]LLM error:[/red] {exc}")
            return
        if record:
            self.turns.append(("llm", prompt))
        self.console.print(
            Panel(
                Markdown(response.content),
                title=f"{response.provider} · {response.model}",
                border_style="cyan",
            )
        )


__all__ = ["ChatSession"]
