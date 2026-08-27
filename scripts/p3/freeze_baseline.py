#!/usr/bin/env python
"""P3 Phase 1 - freeze the P2 baseline and verify it reproduces.

Records the immutable baseline manifest (provider/model, config hashes,
suite fingerprint + raw-file hash, grader/safety digests, seed/repeat
plan) into artifacts/p3/baseline/baseline_manifest.json. Exits non-zero
when the live environment does NOT reproduce the frozen configuration -
in that case no experiment may be attributed against this baseline.

    uv run python scripts/p3/freeze_baseline.py
"""

from __future__ import annotations

import json

from research_engineer.service.p3_experiment import freeze_baseline


def main() -> int:
    path = freeze_baseline()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    print(f"[p3] baseline frozen: {path}")
    for key in ("provider", "model", "judge_provider", "judge_model",
                "llm_config_sha256", "benchmark_seed",
                "full_suite_repeats", "variance_subset_repeats"):
        print(f"  {key}: {manifest.get(key)}")
    print(
        "  suite content_sha256: "
        f"{manifest['suite_fingerprint'].get('content_sha256')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
