"""P1 §2 tests — benchmark suite loading, validation, and contract."""

from __future__ import annotations

import copy
from collections.abc import Callable

import pytest

from research_engineer.eval.models import EvalSuite
from research_engineer.eval.runner import load_suite
from research_engineer.service.benchmark import (
    BENCHMARK_CATEGORIES,
    DEFAULT_SUITE_PATH,
    load_benchmark_suite,
    validate_benchmark_suite,
)


@pytest.fixture(scope="module")
def suite() -> EvalSuite:
    return load_benchmark_suite()


class TestBenchmarkContract:
    def test_v1_suite_loads_from_packaged_path(self) -> None:
        suite = load_suite(DEFAULT_SUITE_PATH)
        assert suite.suite_id == "research_benchmark_v1"
        assert suite.version == "1"

    def test_case_count_and_categories(self, suite: EvalSuite) -> None:
        assert len(suite.cases) == 40
        categories = {c.metadata["category"] for c in suite.cases}
        assert categories == set(BENCHMARK_CATEGORIES)
        per_category = {
            cat: sum(
                1 for c in suite.cases if c.metadata["category"] == cat
            )
            for cat in categories
        }
        # 8 categories x exactly 5 cases each (P1 §3 deliverable).
        assert set(per_category.values()) == {5}

    def test_unique_revisions(self, suite: EvalSuite) -> None:
        ids = [c.case_id for c in suite.cases]
        assert len(ids) == len(set(ids))
        # Every case pins a revision so cross-version compares are stable.
        assert all(c.revision for c in suite.cases)

    def test_known_kinds_and_modes(self, suite: EvalSuite) -> None:
        from research_engineer.service.benchmark import (
            BENCHMARK_AGENT_KINDS,
            BENCHMARK_MODES,
        )

        for case in suite.cases:
            assert case.metadata["agent_kind"] in BENCHMARK_AGENT_KINDS
            assert case.metadata["mode"] in BENCHMARK_MODES

    def test_every_budget_is_finite(self, suite: EvalSuite) -> None:
        for case in suite.cases:
            assert case.budget.max_steps is not None
            assert case.budget.max_steps >= 1

    def _mutated(self, suite: EvalSuite,
                  mutate: Callable[[dict], object]) -> EvalSuite:
        data = copy.deepcopy(suite.model_dump(mode="json"))
        mutate(data)
        return EvalSuite.model_validate(data)

    def test_unknown_grader_rejected(self, suite: EvalSuite) -> None:
        broken = self._mutated(
            suite,
            lambda d: d["cases"][0]["criteria"].append(
                {"grader": "vibes", "weight": 1.0}
            ),
        )
        problems = validate_benchmark_suite(broken)
        assert any("unknown grader 'vibes'" in p for p in problems)

    def test_unknown_kind_rejected(self, suite: EvalSuite) -> None:
        broken = self._mutated(
            suite,
            lambda d: d["cases"][0]["metadata"].update(agent_kind="ghost"),
        )
        problems = validate_benchmark_suite(broken)
        assert any("unknown agent_kind" in p for p in problems)

    def test_missing_category_coverage_rejected(self, suite: EvalSuite) -> None:
        # Re-label every case of one category so coverage genuinely drops.
        broken = self._mutated(
            suite,
            lambda d: [
                c["metadata"].update(category="politics")
                if c["metadata"]["category"] == "ablations"
                else None
                for c in d["cases"]
            ],
        )
        problems = validate_benchmark_suite(broken)
        assert any("unknown category 'politics'" in p for p in problems)
        assert any(
            "categories not covered: ['ablations']" in p for p in problems
        )

    def test_unbounded_budget_rejected(self, suite: EvalSuite) -> None:
        broken = self._mutated(
            suite,
            lambda d: d["cases"][0].update(budget={"max_steps": None}),
        )
        problems = validate_benchmark_suite(broken)
        assert any("must be finite" in p for p in problems)

    def test_max_steps_mismatch_rejected(self, suite: EvalSuite) -> None:
        broken = self._mutated(
            suite,
            lambda d: d["cases"][0].update(budget={"max_steps": 99}),
        )
        problems = validate_benchmark_suite(broken)
        assert any("must match budget" in p for p in problems)

    def test_no_required_criteria_rejected(self, suite: EvalSuite) -> None:
        broken = self._mutated(
            suite,
            lambda d: [
                c.update({"criteria": [
                    dict(cr, required=False) for cr in c["criteria"]
                ]})
                for c in d["cases"]
            ],
        )
        problems = validate_benchmark_suite(broken)
        assert sum("at least one required criterion" in p for p in problems) \
            >= 1

    def test_artifact_expectation_shape_rejected(self, suite: EvalSuite) -> None:
        broken = self._mutated(
            suite,
            lambda d: d["cases"][0]["metadata"].update(
                expected_artifacts=[{"type": "research_notes"}]
            ),
        )
        problems = validate_benchmark_suite(broken)
        assert any("type+min_count" in p for p in problems)

    def test_expected_note_counts_match_checklist_size(
        self, suite: EvalSuite
    ) -> None:
        """notes_written criteria equal the sandbox artifacts promised."""
        from research_engineer.eval.graders import build_grader

        for case in suite.cases:
            artifacts = case.metadata.get("expected_artifacts") or []
            json_fields = {
                int(c.config.get("expected", -1))
                for c in case.criteria
                if c.grader == "output_json_field"
                and c.config.get("field") == "notes_written"
            }
            min_counts = {int(a["min_count"]) for a in artifacts}
            # A complete run writes exactly notes_written notes; the JSON
            # criterion and the artifact minimum must tell one story.
            for expected in json_fields:
                assert not min_counts or max(min_counts) <= expected, (
                    f"{case.case_id}: artifacts {min_counts} vs "
                    f"criterion {expected}"
                )
                assert build_grader("output_json_field", {
                    "field": "notes_written", "expected": expected
                }) is not None
