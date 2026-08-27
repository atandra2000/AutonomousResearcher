"""P3 - Agent capability optimization experiment orchestration.

Composes the *existing* P2/E4/E5/E6/E8 infrastructure into controlled,
single-variable benchmark arms. No new evaluation framework is introduced:

* Arms are versioned YAML declarations under ``configs/p3/arms/`` that
  mutate exactly one configuration surface of the frozen P2 baseline
  (model id, prompt strategy, reasoning budget, stagnation policy).
* :func:`derive_suite` materializes a derived suite snapshot per arm from
  the untouched ``evals/research_benchmark/v2/suite.yaml`` so the original
  P2 benchmark is preserved bit-for-bit.
* :func:`freeze_baseline` records every reproducibility input of the P2
  final run and refuses to proceed when the live environment does not
  reproduce it.
* Arms execute through ``research_engineer.service.p2_benchmark.run_p2``,
  i.e. the full E7 production path (API -> store -> worker -> AgentRuntime
  -> ToolGateway -> SafetyController -> checkpointing -> evaluation -> E6).

Statistical helpers (:func:`wilson_interval`, :func:`mean_ci`) power the
aggregate comparison; results are never extrapolated beyond measured runs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal

import yaml  # type: ignore[import-untyped]  # PyYAML
from pydantic import BaseModel, Field, model_validator

from research_engineer.service.benchmark import (
    DEFAULT_SUITE_V2_PATH,
    load_benchmark_suite,
    validate_benchmark_suite,
)
from research_engineer.service.p2_benchmark import (
    DEFAULT_VARIANCE_CASES,
    config_fingerprint,
    sha256_text,
)

#: Root holding the versioned experiment arm declarations.
DEFAULT_ARMS_DIR = Path("configs/p3/arms")

#: Directory of the authoritative, frozen P2 baseline report.
P2_BASELINE_REPORT = Path("artifacts/p2_final_run/p2_report.json")

#: Variance subset: the P2 default set plus every case that failed
#: objectively or produced an unparseable-judge event in the frozen P2 run,
#: so each known failure gets n >= 2 attempts in EVERY arm.
P3_VARIANCE_CASES: tuple[str, ...] = tuple(sorted(
    set(DEFAULT_VARIANCE_CASES) | {
        "design_02_data_scaling_study",
        "debug_03_oom_step1200",
        "abl_02_next_component_choice",
        "e2e_03_build_decision_memorandum",
        "litdisc_01_kv_cache_survey",
    }
))

#: Scalar override keys an experiment arm may set through ``agent_overrides``.
#: Anything else is rejected at load time so a typo can never silently widen
#: an arm's blast radius (safety/policy allowlists stay outside this surface).
ALLOWED_OVERRIDE_KEYS: frozenset[str] = frozenset({
    "llm_model",                # A: agent model id (judge binding unaffected)
    "llm_strategy",             # B/E: system-prompt strategy variant
    "llm_max_tokens_per_call",  # C: per-call completion cap
    "max_tokens",               # C: cumulative token budget
    "stagnation_window",        # D: steps without progress before NO_PROGRESS
    "progress_threshold",       # D companion knob (unused by current arms)
})


# ---------------------------------------------------------------------------
# Arm configuration models
# ---------------------------------------------------------------------------


class CaseSelector(BaseModel):
    """Which suite cases an arm mutation applies to (exactly one mode)."""

    all_cases: bool = False
    categories: list[str] = Field(default_factory=list)
    case_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _one_mode(self) -> CaseSelector:
        modes = [self.all_cases, bool(self.categories), bool(self.case_ids)]
        if sum(modes) != 1:
            raise ValueError(
                "selector must set exactly one of "
                "all_cases / categories / case_ids"
            )
        return self


class ArmMutation(BaseModel):
    """The single controlled change an arm applies to the base suite."""

    kind: Literal["none", "set_agent_overrides"]
    select: CaseSelector | None = None
    overrides: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate(self) -> ArmMutation:
        if self.kind == "none":
            if self.select is not None or self.overrides:
                raise ValueError(
                    "mutation kind 'none' takes no select/overrides"
                )
            return self
        if self.select is None or not self.overrides:
            raise ValueError(
                "set_agent_overrides requires select + non-empty overrides"
            )
        unknown = sorted(set(self.overrides) - ALLOWED_OVERRIDE_KEYS)
        if unknown:
            raise ValueError(
                f"override keys outside the experiment surface: {unknown}"
            )
        return self

    def matches(self, category: str, case_id: str) -> bool:
        """True when this mutation applies to ``case_id``."""
        if self.kind != "set_agent_overrides":
            return False
        assert self.select is not None  # validated above
        sel = self.select
        if sel.all_cases:
            return True
        if sel.categories and category in sel.categories:
            return True
        return case_id in sel.case_ids


class ExperimentArm(BaseModel):
    """One versioned, single-variable experiment configuration."""

    arm_id: str = Field(..., pattern=r"^[a-z0-9_]+$")
    variable: str
    description: str
    base_suite: str = str(DEFAULT_SUITE_V2_PATH)
    mutation: ArmMutation

    @property
    def is_baseline(self) -> bool:
        return self.mutation.kind == "none"


def load_arm(path: str | Path) -> ExperimentArm:
    """Load and validate one experiment arm declaration."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ExperimentArm.model_validate(raw)


def load_all_arms(
    arms_dir: str | Path = DEFAULT_ARMS_DIR,
) -> list[ExperimentArm]:
    """Load every declared arm; the baseline arm must be present."""
    directory = Path(arms_dir)
    arms = [load_arm(p) for p in sorted(directory.glob("*.yaml"))]
    ids = [a.arm_id for a in arms]
    if "baseline_repro" not in ids:
        raise ValueError("baseline_repro arm declaration missing")
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate arm_id in {directory}")
    return arms


# ---------------------------------------------------------------------------
# Derived suite materialization
# ---------------------------------------------------------------------------


def _expected_match_count(arm: ExperimentArm, total_cases: int) -> int:
    sel = arm.mutation.select
    if arm.mutation.kind == "none" or sel is None:
        return 0
    if sel.all_cases:
        return total_cases
    if sel.categories:
        return -1  # validated by count > 0 below; category sizes vary
    return len(sel.case_ids)


def derive_suite(arm: ExperimentArm, out_dir: str | Path) -> Path:
    """Write the arm's derived suite snapshot; returns its path.

    The snapshot is a plain-JSON YAML document (JSON is valid YAML so the
    standard loader reads it); a ``kind: none`` arm copies its base suite
    byte-for-byte. Derivation is deterministic: identical inputs produce
    byte-identical snapshots. The derived suite re-validates against the
    full benchmark contract and every selector-matched case must have
    actually changed (fail closed on selector typos).
    """
    base_path = Path(arm.base_suite)
    if arm.mutation.kind == "none":
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "derived_suite.yaml"
        path.write_bytes(base_path.read_bytes())
        return path
    suite = load_benchmark_suite(base_path)
    problems = validate_benchmark_suite(suite)
    if problems:
        raise ValueError(f"base suite invalid: {problems}")

    changed = 0
    for case in suite.cases:
        meta = dict(case.metadata or {})
        category = str(meta.get("category", ""))
        if not arm.mutation.matches(category, case.case_id):
            continue
        overrides = dict(meta.get("agent_overrides") or {})
        before = json.dumps(overrides, sort_keys=True)
        overrides.update(arm.mutation.overrides)
        after = json.dumps(overrides, sort_keys=True)
        if before != after:
            changed += 1
            meta["agent_overrides"] = overrides
            case.metadata = meta

    expected = _expected_match_count(arm, len(suite.cases))
    ambiguous = expected == 0 or (expected > 0 and changed != expected)
    if changed == 0 or ambiguous:
        raise ValueError(
            f"selector matched {changed} cases "
            f"(expected {expected}); refusing to write an "
            "ambiguous or empty snapshot"
        )

    problems = validate_benchmark_suite(suite)
    if problems:
        raise ValueError(f"derived suite invalid: {problems}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "derived_suite.yaml"
    payload = suite.model_dump(mode="json")
    path.write_text(
        json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Baseline freeze / integrity
# ---------------------------------------------------------------------------


def _graders_digest(suite: Any) -> str:
    items = sorted(
        json.dumps(
            [c.grader, c.weight, c.required, c.config],
            sort_keys=True, default=str,
        )
        for case in suite.cases
        for c in case.criteria
    )
    return sha256_text(json.dumps(items))


def _safety_digest() -> str:
    from research_engineer.safety.policies import AutonomyPolicy

    return sha256_text(AutonomyPolicy().model_dump_json())


def freeze_baseline(
    out_dir: str | Path = "artifacts/p3/baseline",
    *,
    p2_report_path: str | Path = P2_BASELINE_REPORT,
    repeats: int = 2,
    variance_repeats: int = 3,
) -> Path:
    """Record the immutable P2 baseline manifest; verify reproduction.

    Fails loudly when the live LLM configuration no longer reproduces the
    authoritative P2 final-run fingerprints (provider/model/config hash):
    nothing may be attributed as a delta over a drifting baseline.
    """
    suite = load_benchmark_suite(DEFAULT_SUITE_V2_PATH)
    live = config_fingerprint()
    p2_report = json.loads(Path(p2_report_path).read_text(encoding="utf-8"))
    frozen_cfg = p2_report.get("configuration", {})
    mismatches = [
        f"{k}: frozen={frozen_cfg.get(k)!r} live={live.get(k)!r}"
        for k in ("provider", "model", "judge_provider", "judge_model",
                  "llm_config_sha256")
        if str(live.get(k)) != str(frozen_cfg.get(k))
    ]
    if mismatches:
        raise RuntimeError(
            "baseline does not reproduce the frozen P2 configuration: "
            + "; ".join(mismatches)
        )
    import os

    llm_config_path = Path(os.environ.get("RE_LLM_CONFIG", "llm_config.yaml"))
    first_suite = (p2_report.get("suites") or [{}])[0]
    manifest = {
        "experiment": "p3_capability_optimization",
        "frozen_from_report": str(p2_report_path),
        "provider": live.get("provider"),
        "model": live.get("model"),
        "judge_provider": live.get("judge_provider"),
        "judge_model": live.get("judge_model"),
        "llm_config_sha256": live.get("llm_config_sha256"),
        "llm_config_file_sha256": (
            sha256_text(llm_config_path.read_text(encoding="utf-8"))
            if llm_config_path.exists() else None
        ),
        "suite_fingerprint": {
            "path": str(DEFAULT_SUITE_V2_PATH),
            "suite_id": first_suite.get("suite_id"),
            "case_count": first_suite.get("case_count"),
            "content_sha256": first_suite.get("content_sha256"),
        },
        "suite_file_sha256": sha256_text(
            Path(str(DEFAULT_SUITE_V2_PATH)).read_text(encoding="utf-8")
        ),
        "grader_config_digest": _graders_digest(suite),
        "safety_policy_digest": _safety_digest(),
        "benchmark_seed": 0,
        "full_suite_repeats": repeats,
        "variance_subset_repeats": variance_repeats,
        "p2_baseline_metrics": {
            k: v for k, v in (p2_report.get("headline") or {}).items()
            if isinstance(v, (int, float))
        },
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "baseline_manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Statistics (computed only from measured outcomes)
# ---------------------------------------------------------------------------


def wilson_interval(
    successes: int, total: int, z: float = 1.96,
) -> tuple[float, float]:
    """Wilson score interval (95% default) for a binomial proportion."""
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    spread = (
        z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
        / denom
    )
    return (max(0.0, centre - spread), min(1.0, centre + spread))


_T95: dict[int, float] = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
    7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 15: 2.145, 20: 2.093,
    25: 2.064, 30: 2.045, 40: 2.023, 60: 2.001, 120: 1.980,
}


def mean_ci(values: list[float]) -> tuple[float, float] | None:
    """95% t-based CI ``(lo, hi)`` of the mean; ``None`` when empty."""
    n = len(values)
    if n == 0:
        return None
    m = sum(values) / n
    if n == 1:
        return (m, m)
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    se = math.sqrt(var / n)
    t = _T95[min(_T95, key=lambda k: abs(k - (n - 1)))]
    return (m - t * se, m + t * se)


__all__ = [
    "ALLOWED_OVERRIDE_KEYS",
    "ArmMutation",
    "CaseSelector",
    "DEFAULT_ARMS_DIR",
    "ExperimentArm",
    "P2_BASELINE_REPORT",
    "P3_VARIANCE_CASES",
    "derive_suite",
    "freeze_baseline",
    "load_all_arms",
    "load_arm",
    "mean_ci",
    "wilson_interval",
]
