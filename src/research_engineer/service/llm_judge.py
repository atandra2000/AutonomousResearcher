"""P2 - LLM-as-judge wiring for research-quality grading.

Implements the E4 :class:`~research_engineer.eval.graders.LLMPromptGrader`
pluggable scoring contract using any Phase-10 provider. The judge is an
*explicit opt-in*: deterministic graders remain the default, and every
criterion graded through this module is clearly distinguishable by its
``llm_quality`` grader name.

Integrity contract:

* The rubric is fixed in the suite definition and sent verbatim to the
  judge; the judge never receives hidden ground truth and never uses the
  candidate output *as* ground truth - it only scores it against the rubric.
* Judge failures fail closed (score 0.0) but are marked explicitly as
  ``JUDGE_ERROR`` (``error_kind="judge_error"`` on the grader result), so
  evaluator malfunctions are distinguishable from — and never counted as —
  a genuine zero score against the agent. No score is ever fabricated.
* Responses are parsed strictly (``SCORE: <float>``) and clamped to [0, 1].
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from typing import Any

from research_engineer.llm.base import LLMMessage, LLMRequest, LLMRole

_JUDGE_SYSTEM = """\
You are a strict, independent evaluation judge for ML research outputs. \
Score the candidate output ONLY against the given rubric. Be skeptical; \
reward evidence-grounded, structured, quantified reasoning and penalize \
vagueness, unsupported claims, and missing required sections.

Reply with EXACTLY one line in this format and nothing else:
SCORE: <number between 0.0 and 1.0>
"""

_SCORE_RE = re.compile(r"SCORE\s*[:=]?\s*(\d+(?:\.\d+)?)")


async def _judge_call(provider: Any, prompt: str, rubric: str,
                      ) -> tuple[float, str]:
    """One judged scoring call; returns ``(score, raw_content)``."""
    request = LLMRequest(
        messages=[
            LLMMessage(role=LLMRole.SYSTEM, content=_JUDGE_SYSTEM),
            LLMMessage(
                role=LLMRole.USER,
                content=f"RUBRIC:\n{rubric}\n\n"
                        f"CANDIDATE OUTPUT:\n{prompt}",
            ),
        ],
        temperature=0.0,
        # Reasoning-style judges spend visible budget on internal CoT
        # before emitting the SCORE line; 2048 avoids the flash-class
        # reasoning models exhausting the window pre-SCORE (observed
        # truncated/unparseable replies at 1024).
        max_tokens=2048,
    )
    response = await provider.complete(request)
    content = str(getattr(response, "content", "") or "")
    match = _SCORE_RE.search(content)
    if match is None:
        return 0.0, f"unparseable judge reply: {content[:200]}"
    return min(1.0, max(0.0, float(match.group(1)))), content


def make_judge_score_fn(
    provider: Any,
) -> Callable[[str, str], Awaitable[float]]:
    """Build an ``LLMPromptGrader``-compatible ``(output, rubric) -> score``."""

    async def score_fn(candidate_output: str, rubric: str) -> float:
        if not rubric:
            # Fail closed: no rubric means nothing to grade against.
            return 0.0
        # Judge failures propagate to LLMPromptGrader, which catches them
        # and records ``scoring function failed: <reason>`` in the criterion
        # detail - fail closed *with* an auditable cause instead of a
        # silently swallowed error indistinguishable from a true 0.0.
        score, raw = await _judge_call(provider, candidate_output[:12_000], rubric)
        if not _last_ok(raw):
            raise ValueError(f"unparseable judge reply: {raw[:200]}")
        return score

    return score_fn


def _last_ok(raw: str) -> bool:
    """True when the reply carried a parseable marker line."""
    return _SCORE_RE.search(raw) is not None


def build_llm_quality_grader() -> Any:
    """Construct the ``llm_quality`` grader bound to the config provider."""
    from research_engineer.eval.graders import LLMPromptGrader

    provider = resolve_judge_provider()
    return LLMPromptGrader(make_judge_score_fn(provider))


def resolve_judge_provider() -> Any:
    """Resolve the judge's provider via the router (agent ``EvaluationAgent``)."""
    from research_engineer.llm.router import get_router

    return get_router().for_agent("EvaluationAgent")


__all__ = [
    "build_llm_quality_grader",
    "make_judge_score_fn",
    "resolve_judge_provider",
]
