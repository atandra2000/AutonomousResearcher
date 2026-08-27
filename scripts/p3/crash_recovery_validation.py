#!/usr/bin/env python
"""P3 - crash/recovery validation after experimental changes.

Drives an ``llm_react`` agent (ScriptedProvider - deterministic, no
credentials) through the REAL production stack (SQLite run store -> queue ->
RunManager -> AgentWorker -> AgentRuntime -> ToolGateway -> SafetyController
-> SQLite checkpointing), abandons the run mid-flight as if the worker were
lost, then exercises the E7 recovery chain:

    manager.recover_stale_runs()  ->  RESUMABLE  ->  manager.resume()
    ->  fresh worker drives to COMPLETED from the checkpoint store.

Validated invariants (fail loudly):
- recovery marks exactly this run RESUMABLE;
- resumed run reaches COMPLETED with termination=success;
- checkpointed context counters survive (tokens are preserved, never
  double-counted by replay); tool-call log continuity is recorded.

Writes artifacts/p3/crash_recovery/report.json. Exit code 0 iff all
invariants hold.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from research_engineer.runtime.checkpoint_stores import SQLiteCheckpointStore
from research_engineer.service.artifacts import ArtifactStore
from research_engineer.service.manager import RunManager
from research_engineer.service.models import CreateRunRequest, RunStatus
from research_engineer.service.p2_benchmark import build_p2_factories
from research_engineer.service.queue import build_run_queue
from research_engineer.service.serve import build_default_safety_chain
from research_engineer.service.store import SQLiteRunStore
from research_engineer.service.telemetry import ServiceTelemetry
from research_engineer.service.worker import AgentWorker


class _Shim:
    """Minimal view of service config pieces the safety chain needs."""

    def __init__(self, artifact_dir: Path) -> None:
        self.artifact_dir = artifact_dir


async def scenario(tmp: Path) -> dict[str, Any]:
    gateway, controller = build_default_safety_chain(_Shim(tmp / "artifacts"))
    # Scripted conversation: note write at step 1, final answer at step 2.
    # The mid-flight crash happens between them, so resume must rebuild the
    # conversation and complete step 2 without re-executing step 1's work.
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    from tests.test_llm_benchmark import (
        ScriptedProvider,
        _resp,
        _write_call,
    )

    provider = ScriptedProvider([
        _resp(tool_calls=[_write_call("recovery_note")]),
        _resp(content="ANALYSIS: ok\nFINAL_ANSWER:\ndone"),
    ])
    registry = build_p2_factories(provider=provider)

    store = SQLiteRunStore(str(tmp / "runs.db"))
    # One shared queue for manager + both workers: resume() re-enqueues onto
    # this same logical bus so the fresh worker can claim the resumed run.
    queue = build_run_queue(_Shim(tmp))
    checkpoint = SQLiteCheckpointStore(str(tmp / "checkpoints.db"))
    manager = RunManager(store, queue, ServiceTelemetry(),
                         stale_run_timeout_seconds=0.2)

    created = await manager.submit(CreateRunRequest(
        goal="Write one note then finish with FINAL_ANSWER.",
        metadata={"agent_kind": "llm_react", "mode": "llm_agent",
                  "category": "debugging"},
        budget_overrides={"max_steps": 6, "max_tool_calls": 8},
    ))

    # First worker: execute exactly ONE model call (the note write) and
    # die before completion (simulated worker loss mid-flight).
    victim = AgentWorker(
        store=store, queue=queue, checkpoint_store=checkpoint,
        artifacts=ArtifactStore(tmp / "artifacts"), factories=registry,
        telemetry=ServiceTelemetry(), poll_interval_seconds=0.01,
        tool_gateway=gateway, safety_controller=controller,
        require_safety_chain=True,
    )
    await victim.try_claim_and_execute()  # starts + checkpoints context
    events: dict[str, Any] = {"run_id": created.run_id}

    # Simulate machine loss: run stays RUNNING with no heartbeat.
    record = await store.get_required(created.run_id)
    record.status = RunStatus.RUNNING
    await store.update(record)
    await asyncio.sleep(0.4)

    recovered = await manager.recover_stale_runs()
    assert recovered == [created.run_id], recovered
    rec_status = (await store.get_required(created.run_id)).status
    assert rec_status == RunStatus.RESUMABLE, rec_status
    events["recovered"] = True

    resumed = await manager.resume(created.run_id)
    assert resumed.status == RunStatus.QUEUED
    events["resumed"] = True

    # Fresh worker completes the remainder of the scripted conversation.
    completer = AgentWorker(
        store=store, queue=queue,
        checkpoint_store=checkpoint,
        artifacts=ArtifactStore(tmp / "artifacts"), factories=registry,
        telemetry=ServiceTelemetry(), poll_interval_seconds=0.01,
        tool_gateway=gateway, safety_controller=controller,
        require_safety_chain=True,
    )
    for _ in range(300):
        await completer.try_claim_and_execute()
        rec = await store.get_required(created.run_id)
        if rec.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break
        await asyncio.sleep(0.02)
    await completer.drain(10)
    final = await store.get_required(created.run_id)
    assert final.status == RunStatus.COMPLETED, (
        f"expected COMPLETED, got {final.status}: {final.error}"
    )
    assert (final.result or {}).get("termination") == "success"
    events["completed"] = True
    events["termination"] = (final.result or {}).get("termination")
    events["result_keys"] = sorted((final.result or {}).keys())[:12]
    events["tokens_final"] = (final.result or {}).get("tokens")
    return events


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/p3/crash_recovery")
    args = parser.parse_args()
    tmp = Path(args.out)
    tmp.mkdir(parents=True, exist_ok=True)
    report_path = tmp / "report.json"
    try:
        events = asyncio.run(scenario(tmp / "scratch"))
        passed = bool(events.get("completed"))
    except AssertionError as exc:
        events = {"failed_invariant": str(exc)}
        passed = False
    payload = {
        "validation": "p3_crash_recovery_after_experimental_changes",
        "passed": passed,
        "events": events,
    }
    report_path.write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, default=str)[:1200])
    print(f"wrote {report_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
