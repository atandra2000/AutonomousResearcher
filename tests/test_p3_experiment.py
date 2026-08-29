"""P3 - experiment orchestration, baseline freeze, and statistics.

Covers the versioned arm declarations (single-variable guard), deterministic
suite derivation with byte-identity for the baseline arm, fail-closed
selector matching, statistics math, the ``llm_strategy`` prompt hook
(incl. unknown-strategy fail-closed), and the stagnation-window policy
override plumbing.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest

from research_engineer.benchmark.agents import AgentFactoryRegistry
from research_engineer.benchmark.benchmark import (
    DEFAULT_SUITE_V2_PATH,
    load_benchmark_suite,
)
from research_engineer.benchmark.llm_agent import (
    STRATEGY_PROMPTS,
    LLMReActAdapter,
    system_prompt,
)
from research_engineer.benchmark.p3_experiment import (
    ArmMutation,
    CaseSelector,
    derive_suite,
    load_all_arms,
    mean_ci,
    wilson_interval,
)

# ---------------------------------------------------------------------------
# Versioned arm declarations / single-variable discipline
# ---------------------------------------------------------------------------


def test_all_declared_arms_load_and_validate() -> None:
    arms = load_all_arms()
    assert len(arms) >= 8  # baseline + A1-A3 + B + C + D + E
    ids = [a.arm_id for a in arms]
    assert ids[0] == "baseline_repro"
    assert len(ids) == len(set(ids))
    for arm in arms:
        assert arm.variable in {
            "none", "model", "planning_strategy", "token_budget",
            "stagnation_policy", "implementation_strategy",
        }


def test_mutation_rejects_unknown_override_keys() -> None:
    """A typo must never silently widen an arm's blast radius."""
    with pytest.raises(ValueError):
        ArmMutation(
            kind="set_agent_overrides",
            select=CaseSelector(all_cases=True),
            overrides={"llm_api_key": "leak"},
        )


def test_selector_requires_exactly_one_mode() -> None:
    with pytest.raises(ValueError):
        CaseSelector(all_cases=True, case_ids=["x"])
    with pytest.raises(ValueError):
        CaseSelector()


# ---------------------------------------------------------------------------
# Derivation: determinism + single-variable deltas
# ---------------------------------------------------------------------------


def _overrides_by_case(path: Path) -> dict[str, dict]:
    suite = load_benchmark_suite(path)
    return {
        c.case_id: dict(
            dict(c.metadata or {}).get("agent_overrides") or {}
        )
        for c in suite.cases
    }


@pytest.mark.parametrize("arm_idx", range(8))
def test_derive_suite_deterministic_and_valid(arm_idx: int) -> None:
    arm = load_all_arms()[arm_idx]
    with tempfile.TemporaryDirectory() as td:
        first = derive_suite(arm, td)
        payload_first = first.read_bytes()
        second = derive_suite(arm, td)
        assert second.read_bytes() == payload_first
        derived = load_benchmark_suite(first)
        assert len(derived.cases) == 20


def test_baseline_arm_snapshot_is_byte_identical() -> None:
    arms = [a for a in load_all_arms() if a.arm_id == "baseline_repro"]
    assert len(arms) == 1
    with tempfile.TemporaryDirectory() as td:
        path = derive_suite(arms[0], td)
        assert path.read_bytes() == Path(DEFAULT_SUITE_V2_PATH).read_bytes()


def _arm_dir(arm_id: str) -> tuple[Any, Path]:
    return tempfile.TemporaryDirectory(), Path(arm_id)  # pragma: no cover


ArmsCache: dict[str, Any] = {}


def _arm_by_id(arm_id: str) -> Any:
    arms = {a.arm_id: a for a in load_all_arms()}
    return arms[arm_id]


def test_model_arm_changes_only_llm_model() -> None:
    base = _overrides_by_case(Path(DEFAULT_SUITE_V2_PATH))
    with tempfile.TemporaryDirectory() as td:
        got = _overrides_by_case(
            derive_suite(_arm_by_id("exp_a1_kimi_k27_code"), td)
        )
    assert set(got) == set(base)
    for cid, before in base.items():
        expected = {**before, "llm_model": "kimi-k2.7-code"}
        assert got[cid] == expected, cid


def test_impl_strategy_arm_touches_only_implementation_cases() -> None:
    base = _overrides_by_case(Path(DEFAULT_SUITE_V2_PATH))
    with tempfile.TemporaryDirectory() as td:
        got = _overrides_by_case(
            derive_suite(_arm_by_id("exp_e_impl_strategy"), td)
        )
    changed = [c for c in base if got[c] != base[c]]
    assert sorted(changed) == [
        "impl_01_attention_module_plan", "impl_02_patch_plan_memory_leak",
    ]
    for cid in changed:
        assert got[cid]["llm_strategy"] == "impl_focus"


def test_strategy_arm_appends_single_prompt_variable() -> None:
    base = _overrides_by_case(Path(DEFAULT_SUITE_V2_PATH))
    with tempfile.TemporaryDirectory() as td:
        got = _overrides_by_case(
            derive_suite(_arm_by_id("exp_b_plan_first_strategy"), td)
        )
    assert sorted(c for c in base if got[c] != base[c]) == sorted(base)
    for cid in base:
        assert set(got[cid]) - set(base[cid]) == {"llm_strategy"}, cid


def test_budget_arm_changes_only_budget_scalars() -> None:
    base = _overrides_by_case(Path(DEFAULT_SUITE_V2_PATH))
    with tempfile.TemporaryDirectory() as td:
        got = _overrides_by_case(
            derive_suite(_arm_by_id("exp_c_reasoning_budget"), td)
        )
    for cid in base:
        delta = set(got[cid]) - set(base[cid])
        assert delta == {"llm_max_tokens_per_call", "max_tokens"}, cid
        assert got[cid]["llm_max_tokens_per_call"] == 4096
        assert got[cid]["max_tokens"] == 48000


def test_stagnation_arm_changes_only_policy_window() -> None:
    base = _overrides_by_case(Path(DEFAULT_SUITE_V2_PATH))
    with tempfile.TemporaryDirectory() as td:
        got = _overrides_by_case(
            derive_suite(_arm_by_id("exp_d_stagnation_window6"), td)
        )
    for cid in base:
        assert got[cid] == {**base[cid], "stagnation_window": 6}, cid


# ---------------------------------------------------------------------------
# Strategy prompt hook (fail closed on unknown names)
# ---------------------------------------------------------------------------


def test_system_prompt_base_is_unchanged_without_strategy() -> None:
    assert system_prompt(None) == system_prompt("")
    assert "autonomous research agent" in system_prompt(None)


@pytest.mark.parametrize("name", ["plan_first", "impl_focus"])
def test_known_strategies_extend_the_shared_prompt(name: str) -> None:
    extended = system_prompt(name)
    assert extended.startswith(system_prompt(None))
    assert STRATEGY_PROMPTS[name] in extended


def test_unknown_strategy_fails_closed() -> None:
    with pytest.raises(ValueError):
        system_prompt("nonexistent_strategy")
    with pytest.raises(ValueError):
        LLMReActAdapter(object(), strategy="typo_arm")


# ---------------------------------------------------------------------------
# Policy plumbing (stagnation window reaches AgentPolicy)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_stagnation_window_override_plumbs_into_policy() -> None:
    registry = AgentFactoryRegistry()
    adapter, policy = await registry.build(
        "planning_checklist", {"stagnation_window": 6}
    )
    assert adapter is not None
    assert policy.stagnation_window == 6
    _, default_policy = await registry.build("planning_checklist", {})
    assert default_policy.stagnation_window == 3


# ---------------------------------------------------------------------------
# Statistics math
# ---------------------------------------------------------------------------


def test_wilson_interval_known_values() -> None:
    lo, hi = wilson_interval(22, 28)
    assert abs(lo - 0.6046) < 5e-3
    assert abs(hi - 0.8979) < 5e-3
    assert wilson_interval(0, 0) == (0.0, 0.0)
    lo, hi = wilson_interval(28, 28)
    assert hi <= 1.0 and 0.87 < lo < 0.90


def test_mean_ci_bounds() -> None:
    assert mean_ci([]) is None
    single = mean_ci([0.7])
    assert single == (0.7, 0.7)
    ci = mean_ci([0.8, 0.9, 0.85])
    assert ci is not None
    lo, hi = ci
    assert lo < 0.85 < hi
