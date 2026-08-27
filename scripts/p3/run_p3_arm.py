#!/usr/bin/env python
"""P3 - execute one experiment arm through the real production path.

Reproducible single-arm runner: derives the arm's suite snapshot (recorded
under artifacts/p3/<arm>/derived_suite.yaml for auditability), then runs the
unchanged P2 flow (full v2 suite -> variance repeats -> optional P1
deterministic regression) writing p2_report.json/.md plus raw tier reports.

    uv run python scripts/p3/run_p3_arm.py --arm baseline_repro
    uv run python scripts/p3/run_p3_arm.py --arm exp_a1_kimi_k27_code \
        --repeats 3

The judge is always resolved from the frozen llm_config.yaml routing
(EvaluationAgent) so evaluator quality stays constant across arms.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from research_engineer.service.p2_benchmark import config_fingerprint
from research_engineer.service.p3_experiment import (
    DEFAULT_ARMS_DIR,
    P3_VARIANCE_CASES,
    derive_suite,
    load_arm,
)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-root", default="artifacts/p3")
    args = parser.parse_args()

    arm_path = Path(DEFAULT_ARMS_DIR) / f"{args.arm}.yaml"
    arm = load_arm(arm_path)
    out_dir = Path(args.output_root) / arm.arm_id
    suite_path = derive_suite(arm, out_dir)

    print(f"[p3] arm={arm.arm_id} variable={arm.variable}")
    print(f"[p3] derived suite: {suite_path}")
    meta = {
        "arm_id": arm.arm_id,
        "variable": arm.variable,
        "description": arm.description,
        "mutation": arm.mutation.model_dump(mode="json"),
        "derived_suite_sha256": hashlib.sha256(
            suite_path.read_bytes()
        ).hexdigest(),
        "variance_cases": list(P3_VARIANCE_CASES),
        "repeats": args.repeats,
        "config": config_fingerprint(),
    }
    (out_dir / "arm_manifest.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    report = await _run(arm, suite_path, out_dir, args.repeats)
    print(
        f"[p3] report: {out_dir / 'p2_report.json'} "
        f"verdict={report.verdict}"
    )
    return 0


async def _run(arm: Any, suite_path: Path, out_dir: Path,
               repeats: int) -> Any:
    from research_engineer.service.p2_benchmark import run_p2

    return await run_p2(
        out_dir,
        suite_path=suite_path,
        p1_regression=arm.is_baseline,
        repeats=repeats,
        variance_cases=P3_VARIANCE_CASES,
        label=f"p3-{arm.arm_id}",
    )


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
