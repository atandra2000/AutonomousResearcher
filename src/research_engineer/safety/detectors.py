"""E5 - Deterministic safety detectors.

Pure, side-effect-free detection primitives used by the rule-based
safety policy:

* :func:`step_signature` — stable hash of a step's plan/action/observation/
  evaluation outputs (drives loop detection).
* :func:`tool_call_key` — stable hash of tool name + normalized arguments
  (drives duplicate-call detection).
* :class:`LoopDetector` — repeated-signature and cyclic behaviour.
* :class:`DuplicateToolCallDetector` — identical tool + args repeated
  without useful (different) results.

All detectors are deterministic and LLM-free by construction.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from research_engineer.runtime.models import AgentStep
from research_engineer.safety.models import SafetyState


def _stable_hash(payload: Any) -> str:
    """Stable SHA-256 of any JSON-serializable-ish payload."""
    try:
        text = json.dumps(payload, sort_keys=True, default=repr)
    except Exception:  # noqa: BLE001 - fall back to repr for exotic objects
        text = repr(payload)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def step_signature(step: AgentStep) -> str:
    """Stable signature of one runtime step's observable behaviour.

    Plan, action, observation, and evaluation are all included so that a
    step whose evaluation changes (e.g. ``done`` flips) gets a new
    signature — this avoids false-positive loop detection on agents whose
    observation is stable but who make real progress via evaluation.
    """
    return _stable_hash(
        {
            "plan": step.plan,
            "action": step.action,
            "observation": step.observation,
            "evaluation": step.evaluation,
        }
    )


def normalize_args(tool_input: Any) -> Any:
    """Normalize a tool input to hashable JSON-serializable form.

    Pydantic models are dumped via ``model_dump(mode="json")`` when
    available; dicts are used as-is; anything else falls back to ``repr``.
    """
    if hasattr(tool_input, "model_dump"):
        try:
            return tool_input.model_dump(mode="json")
        except Exception:  # noqa: BLE001 - best effort normalization
            pass
    return tool_input


def tool_call_key(tool_name: str, tool_input: Any) -> str:
    """Stable key for a tool invocation: name + hashed arguments."""
    return f"{tool_name}:{_stable_hash(normalize_args(tool_input))}"


class LoopDetector:
    """Detect repeated signatures and short cycles in step history.

    Two detections:

    * **Repeated state** — the trailing ``threshold`` signatures are all
      identical.
    * **Cycle** — within the last ``window`` signatures there exists a
      repeating pattern of length ≤ ``max_cycle_len`` occurring at least
      ``min_repeats`` times back-to-back at the end of the window.
    """

    def __init__(
        self,
        *,
        threshold: int = 3,
        window: int = 12,
        max_cycle_len: int = 2,
    ) -> None:
        self.threshold = max(2, threshold)
        self.window = max(window, self.threshold * (max_cycle_len + 1))
        self.max_cycle_len = max(1, max_cycle_len)

    def detect(self, state: SafetyState) -> tuple[bool, int]:
        """Return ``(detected, cycle_length)``.

        Detects cycles whose final ``length``-sized block repeats back-to-
        back enough times to cover ``threshold`` steps (e.g. threshold=3:
        three identical steps -> cycle length 1; ABAB -> cycle length 2).
        """
        sigs = state.signatures[-self.window :]
        n = len(sigs)
        if n < self.threshold:
            return False, 0

        for length in range(1, self.max_cycle_len + 1):
            if n - length < 0:
                break
            block = sigs[-length:]
            repeats = 0
            cursor = n - length
            while cursor >= 0 and sigs[cursor : cursor + length] == block:
                repeats += 1
                cursor -= length
            required_repeats = -(-self.threshold // length)  # ceil division
            if repeats >= required_repeats and repeats * length >= self.threshold:
                return True, length
        return False, 0

    @staticmethod
    def _is_repeated(sigs: list[str], length: int, repeats: int) -> bool:
        """True when the final ``length*repeats`` entries form ``repeats``
        consecutive copies of the same length-``length`` pattern."""
        end = sigs[-(length * repeats) :]
        block = end[:length]
        return all(end[i : i + length] == block for i in range(length, len(end), length))


class DuplicateToolCallDetector:
    """Detect identical tool + arguments calls repeated without progress.

    A duplicate violation occurs when the same ``tool_call_key`` occurs
    more than ``max_identical`` times inside the recorded window *and*
    (when ``require_same_result``) produced an identical result each time
    — i.e. calling the tool again changed nothing useful.
    """

    def __init__(
        self,
        *,
        max_identical: int = 2,
        window: int = 10,
        require_same_result: bool = True,
    ) -> None:
        self.max_identical = max(1, max_identical)
        self.window = max(window, self.max_identical + 1)
        self.require_same_result = require_same_result

    def detect(self, state: SafetyState) -> tuple[bool, str]:
        """Return ``(detected, tool_call_key)``."""
        recent = state.tool_calls[-self.window :]
        counts: dict[str, list[str]] = {}
        for record in recent:
            entry = f"{record.tool_name}:{record.args_hash}"
            counts.setdefault(entry, []).append(record.result_signature)
        for entry, results in counts.items():
            if len(results) <= self.max_identical:
                continue
            if self.require_same_result and len(set(results)) != 1:
                continue
            return True, entry.rsplit(":", 1)[0]
        return False, ""


__all__ = [
    "LoopDetector",
    "DuplicateToolCallDetector",
    "normalize_args",
    "step_signature",
    "tool_call_key",
]
