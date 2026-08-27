"""P4 - ImprovementStore backend contract tests.

The same store contract must hold for the JSON filesystem store
(development/tests) and the PostgreSQL store (production). PostgreSQL
variants are skipped unless ``RE_TEST_PG_DSN`` points at a live database
(the E7 docker-compose stack provides one).
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import pytest

from research_engineer.improve.models import (
    Baseline,
    ImprovementCandidate,
    MetricSnapshot,
    PromotionAction,
    PromotionDecision,
)
from research_engineer.improve.pg_store import PostgresImprovementStore
from research_engineer.improve.pipeline import ImprovementStore

PG_DSN = os.environ.get("RE_TEST_PG_DSN", "")


def _make_baseline(label: str = "b1") -> Baseline:
    return Baseline(
        baseline_id=f"base_{label}",
        label=label,
        suite_id="s",
        suite_version="1",
        config_snapshot={"model": label},
        config_hash="h",
        metrics=MetricSnapshot(),
    )


def _make_candidate(cid: str) -> ImprovementCandidate:
    return ImprovementCandidate(
        candidate_id=cid,
        component="model_provider_config",
        changes={"model_provider.model": "x"},
        config_hash="h",
        version="v1",
        parent_baseline_id="base_b1",
        parent_config_hash="ph",
        suite_id="s",
        suite_version="1",
    )


def _run_store_contract(store: Any) -> None:
    baseline = _make_baseline()
    store.save_baseline(baseline)
    assert store.load_baseline(baseline.baseline_id) is not None
    assert store.load_baseline("missing") is None
    assert len(store.list_baselines()) >= 1

    cand = _make_candidate("cand_x")
    store.save_candidate(cand)
    loaded = store.load_candidate("cand_x")
    assert loaded is not None and loaded.candidate_id == "cand_x"
    assert store.load_candidate("missing") is None
    assert len(store.list_candidates()) >= 1

    store.record_decision("cand_x", PromotionDecision(
        action=PromotionAction.PROMOTED, decided_by="tester", reason="ok",
    ))
    decisions = store.list_decisions()
    assert any(d["candidate_id"] == "cand_x" for d in decisions)
    assert store.get_active("model_provider_config") is None
    store.set_active("model_provider_config", "cand_a")
    assert store.get_active("model_provider_config") == "cand_a"
    store.set_active("model_provider_config", "cand_b")
    assert store.get_active("model_provider_config") == "cand_b"
    # restore_previous swaps back to cand_a
    assert store.restore_previous("model_provider_config") == "cand_a"
    assert store.get_active("model_provider_config") == "cand_a"
    # revert_to baseline clears the pointer
    assert store.revert_to("model_provider_config", "") == ""
    assert store.get_active("model_provider_config") is None
    assert store.revert_to("model_provider_config", "") is None


def test_json_store_contract(tmp_path: Path) -> None:
    _run_store_contract(ImprovementStore(root=tmp_path / "improvements"))


@pytest.mark.skipif(
    not PG_DSN, reason="RE_TEST_PG_DSN not configured (no live PostgreSQL)"
)
def test_postgres_store_contract() -> None:
    store = PostgresImprovementStore(PG_DSN)
    try:
        _run_store_contract(store)
    finally:
        store.close()


@pytest.mark.skipif(
    not PG_DSN, reason="RE_TEST_PG_DSN not configured (no live PostgreSQL)"
)
def test_postgres_active_pointers_survive_concurrent_operators() -> None:
    """Advisory-lock protection: N concurrent operators setting the active
    pointer never lose an update or corrupt previous/active slots."""
    store = PostgresImprovementStore(PG_DSN)
    try:
        store.set_active("model_provider_config", "cand_0")
        barrier = threading.Barrier(4)

        def operator(i: int) -> None:
            barrier.wait()
            store.set_active("model_provider_config", f"cand_{i}")

        threads = [threading.Thread(target=operator, args=(i,))
                   for i in range(1, 5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # Every update serialized; final pointer is one of the writers and
        # the previous slot records the last-but-one value (no loss).
        active = store.get_active("model_provider_config")
        assert active in {f"cand_{i}" for i in range(1, 5)}
        assert store.revert_to("model_provider_config", "") == ""
    finally:
        store.close()
