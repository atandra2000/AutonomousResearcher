"""Phase 13 - Reviewer agent.

Specialized agent that reviews generated code changes (diff/patches)
and provides structured feedback: approved or changes-requested, with
specific issues listed. Uses the reasoning model for LLM-based review
and falls back to a heuristic checker when no provider is configured.

Interface: ``async execute(ctx: SharedTaskContext) -> dict``
"""

from __future__ import annotations

import re
from typing import Any

from research_engineer.agents._llm_support import resolve_llm
from research_engineer.agents._streaming import stream_complete
from research_engineer.llm import LLMMessage, LLMProvider, LLMRequest, LLMRole
from research_engineer.models.delegation import (
    Feedback,
    FeedbackType,
    SharedTaskContext,
)


class ReviewerAgent:
    """Reviews generated code changes and provides structured feedback.

    Reads ``ctx.diff`` from the shared context, reviews it for
    common issues (syntax, missing imports, style), and writes
    ``ctx.review_feedback`` and ``ctx.review_issues``. Returns a
    :class:`Feedback` dict indicating approval or changes-requested.
    """

    def __init__(self, llm: LLMProvider | None = None) -> None:
        self.agent_name: str = "ReviewerAgent"
        self.llm_provider = resolve_llm(self.agent_name, llm)

    async def execute(
        self, ctx: SharedTaskContext, **kwargs: Any
    ) -> dict[str, Any]:
        """Review the current diff and write feedback to ``ctx``."""
        stream_sink = kwargs.get("stream_sink")
        feedback = await self._review(ctx, stream_sink)
        ctx.review_feedback = feedback.summary
        ctx.review_issues = feedback.issues
        return {
            "summary": feedback.summary,
            "feedback": feedback.model_dump(),
            "approved": feedback.approved,
        }

    async def _review(
        self,
        ctx: SharedTaskContext,
        stream_sink: Any | None = None,
    ) -> Feedback:
        """Review the diff and return structured feedback."""
        if not ctx.diff:
            return Feedback(
                feedback_type=FeedbackType.APPROVED,
                approved=True,
                summary="No diff to review; skipping.",
            )
        provider = self.llm_provider
        if provider is None:
            return self._heuristic_review(ctx)
        system = (
            "You are a strict code reviewer. Review the following diff "
            "for correctness, style, and potential issues. Respond with "
            "APPROVED or CHANGES_REQUESTED on the first line, followed "
            "by a list of specific issues (if any)."
        )
        user = (
            f"Goal: {ctx.goal}\n\n"
            f"## Implementation Plan\n{ctx.implementation_plan}\n\n"
            f"## Diff\n```diff\n{ctx.diff[:8000]}\n```"
        )
        request = LLMRequest(
            messages=[
                LLMMessage(role=LLMRole.SYSTEM, content=system),
                LLMMessage(role=LLMRole.USER, content=user),
            ],
            temperature=0.2,
            max_tokens=512,
        )
        if stream_sink is not None:
            resp = await stream_complete(provider, request, sink=stream_sink)
            if resp is not None:
                return self._parse_llm_review(resp.content)
        try:
            resp = await provider.complete(request)
            return self._parse_llm_review(resp.content)
        except Exception:
            return self._heuristic_review(ctx)

    @staticmethod
    def _parse_llm_review(content: str) -> Feedback:
        """Parse LLM review output into structured feedback."""
        lines = content.strip().splitlines()
        first = lines[0].upper() if lines else ""
        approved = "APPROVED" in first
        issues = [
            line.lstrip("- ").strip()
            for line in lines[1:]
            if line.strip() and not line.strip().upper().startswith("APPROVED")
        ]
        return Feedback(
            feedback_type=(
                FeedbackType.APPROVED
                if approved
                else FeedbackType.CHANGES_REQUESTED
            ),
            approved=approved,
            issues=issues[:10],
            summary=content[:300],
            repair_needed=not approved,
        )

    @staticmethod
    def _heuristic_review(ctx: SharedTaskContext) -> Feedback:
        """Rule-based code review when no LLM is available."""
        issues: list[str] = []
        diff = ctx.diff
        if not diff:
            return Feedback(
                feedback_type=FeedbackType.APPROVED,
                approved=True,
                summary="No diff to review.",
            )
        # Check for common issues.
        added_lines = [
            line[1:] for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")
        ]
        added_text = "\n".join(added_lines)
        # Bare except.
        if re.search(r"^\s*except\s*:", added_text, re.MULTILINE):
            issues.append("Bare 'except:' clause found; use specific exceptions.")
        # Print in production code.
        if re.search(r"^\s*print\s*\(", added_text, re.MULTILINE):
            issues.append("Debug 'print()' found; consider using logging.")
        # TODO/FIXME.
        if re.search(r"\b(TODO|FIXME)\b", added_text):
            issues.append("TODO/FIXME found in diff; resolve before merging.")
        # Missing newline at end.
        if added_text and not added_text.endswith("\n"):
            issues.append("File may be missing trailing newline.")
        approved = len(issues) == 0
        return Feedback(
            feedback_type=(
                FeedbackType.APPROVED
                if approved
                else FeedbackType.CHANGES_REQUESTED
            ),
            approved=approved,
            issues=issues,
            summary=(
                "Heuristic review: no issues found."
                if approved
                else f"Heuristic review: {len(issues)} issue(s) found."
            ),
            repair_needed=not approved,
        )


__all__ = ["ReviewerAgent"]
