#!/usr/bin/env python3
"""P1 §10 - production pilot: real API + simulated worker crash + resume.

Exercises the actual E7 deployment topology end to end. Cross-process runs
require the Postgres-backed queue (InMemoryQueue is process-local by
design), so the primary mode drives the ``deploy/docker-compose.yml``
stack:

1. Verifies the stack's readiness endpoint and fail-closed enforcement.
2. Submits a benchmark case through the authenticated HTTP API.
3. Once the run is mid-flight, SIGKILLs the *worker container* - simulating
   a crashed machine. Compose restarts it per its restart policy.
4. The fresh worker must take over the stale run from the checkpoint store
   (``claim_count`` >= 2) and drive it to completion.

Mode ``local`` instead spawns api+worker subprocesses against an externally
provided Postgres DSN (no Docker involved).

Writes ``pilot_report.json`` into its scratch dir. This is a crash/HA probe,
not a grading run - see BenchmarkRunner for suite metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COMPOSE_FILE = REPO / "deploy" / "docker-compose.yml"

DEFAULT_CASE_GOAL = (
    "Pilot incident replay. "
    "Load the agent-kind registry entry point. "
    "Publish the integration status digest."
)

READY_TIMEOUT_S = 60.0
TERMINAL_TIMEOUT_S = 240.0


def _http(method: str, url: str, *, body: dict | None = None,
          token: str = "") -> tuple[int, dict]:
    req = urllib.request.Request(url, method=method)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    data = None
    if body is not None:
        req.add_header("Content-Type", "application/json")
        data = json.dumps(body).encode()
    try:
        with urllib.request.urlopen(req, data=data, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, {"error": exc.read()[:300].decode(
            "utf-8", "replace")}


def _await_ready(api_url: str) -> dict:
    deadline = time.monotonic() + READY_TIMEOUT_S
    last: dict = {}
    while time.monotonic() < deadline:
        try:
            _, last = _http("GET", f"{api_url}/ready")
            if last.get("ready"):
                return last
        except Exception as exc:  # connection refused during startup
            last = {"error": str(exc)}
        time.sleep(0.5)
    raise RuntimeError(f"API never became ready: {last}")


def _submit(api_url: str, token: str, case_id: str | None) -> dict:
    body: dict = {
        "goal": DEFAULT_CASE_GOAL,
        "metadata": {"agent_kind": "bench_flaky",
                     "benchmark_case_id": case_id or "pilot_inline"},
        "budget_overrides": {"bench_fail_calls": 3},
    }
    status, created = _http("POST", f"{api_url}/runs", body=body,
                            token=token)
    if status != 202:
        raise RuntimeError(f"submit failed ({status}): {created}")
    return created


def _get_record(api_url: str, token: str, run_id: str) -> dict:
    status, rec = _http("GET", f"{api_url}/runs/{run_id}", token=token)
    if status != 200:
        raise RuntimeError(f"GET /runs/{run_id} failed ({status}): {rec}")
    return rec


def _await_terminal(record: Callable[[], dict]) -> tuple[dict, int]:
    """Poll until terminal; returns (record, max observed claim_count)."""
    claim_max = 0
    final: dict = {}
    last: dict = {}
    dl = time.monotonic() + TERMINAL_TIMEOUT_S
    while time.monotonic() < dl:
        last = record()
        claim_max = max(claim_max, int(last.get("claim_count", 0)))
        if last["status"] in ("completed", "failed", "cancelled"):
            final = last
            break
        time.sleep(0.25)
    assert final.get("status"), (
        f"run never reached terminal state; last={last}"
    )
    return final, claim_max


def _crash_worker() -> str:
    """SIGKILL the compose worker container; compose restarts it."""
    ps = subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE),
         "ps", "-q", "worker"],
        capture_output=True, text=True,
        cwd=str(COMPOSE_FILE.parent), check=True)
    cid = ps.stdout.strip().splitlines()[0]
    subprocess.run(["docker", "kill", "--signal=SIGKILL", cid],
                   check=True, capture_output=True)
    return f"container:{cid[:12]}"


def _spawn_local_worker(env: dict[str, str], procs: list) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-m", "research_engineer.service.serve", "worker"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    procs.append(proc)
    time.sleep(1.0)
    return proc


def _local_env(args: argparse.Namespace, artifacts_dir: Path) -> dict[str, str]:
    """Env shared by every locally spawned process (one backend)."""
    env = dict(os.environ)
    env["RE_POSTGRES_DSN"] = args.postgres_dsn
    # The DSN selects Postgres-backed store/queue/checkpoints, so no path
    # overrides - both processes must share one backend.
    for stale in ("RE_SERVICE_DB_PATH", "RE_CHECKPOINT_DB"):
        env.pop(stale, None)
    env["RE_STALE_RUN_TIMEOUT_SECONDS"] = "4"
    # Generous per-step throttle so the SIGKILL lands reliably mid-flight
    # (bench adapters honor this via AgentFactoryRegistry injection).
    env["RE_STEP_DELAY_SECONDS"] = "1"
    # P1 §1: the pilot is a production probe - mirror the compose stack and
    # run the worker fail-closed with the full E3+E5 safety chain.
    env["RE_SERVICE_ENFORCE_SAFETY"] = "1"
    # Both processes must share one artifact workspace for sandbox tools.
    env["RE_SERVICE_ARTIFACT_DIR"] = str(artifacts_dir)
    return env


def _spawn_local_api(env: dict[str, str], procs: list) -> None:
    procs.append(subprocess.Popen(
        [sys.executable, "-m", "research_engineer.service.serve", "api"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))


def _crash_and_replace(
        args: argparse.Namespace, procs: list[subprocess.Popen],
        local_env: dict[str, str] | None) -> str:
    """Kill the active worker; docker mode relies on compose restart."""
    if args.mode == "docker":
        return _crash_worker()
    # Local: SIGKILL worker #1; a fresh one takes over after the lease.
    os.kill(procs[-1].pid, signal.SIGKILL)
    procs[-1].wait()
    victim = f"pid:{procs[-1].pid}"
    assert local_env is not None
    _spawn_local_worker(local_env, procs)
    time.sleep(4.0)  # stale-lease expiry margin
    return victim


def _await_running(record: Callable[[], dict]) -> None:
    """Block until the run is claimed and executing on a worker.

    Raises if the run terminalizes first: with a correctly throttled worker
    the mid-flight window always exists, so this indicates a config problem
    rather than something to paper over.
    """
    while True:
        record_now = record()
        status_now = str(record_now.get("status"))
        if status_now == "running":
            return
        if status_now in ("completed", "failed", "cancelled"):
            raise RuntimeError(
                f"run reached terminal state {status_now!r} before a "
                "mid-flight window existed; cannot exercise crash "
                "recovery. Check RE_STEP_DELAY_SECONDS on workers."
            )
        time.sleep(0.05)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("docker", "local"),
                        default="docker")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-token",
                        default=os.environ.get("RE_SERVICE_API_TOKEN", ""))
    parser.add_argument("--postgres-dsn",
                        help="local mode only: shared cross-process queue "
                             "DSN; the queue needs Postgres when more than "
                             "one process participates")
    parser.add_argument("--case-id", default=None,
                        help="benchmark case id recorded in metadata")
    args = parser.parse_args()
    if args.mode == "local" and not args.postgres_dsn:
        parser.error("local mode requires --postgres-dsn")

    base = Path(tempfile.mkdtemp(prefix="re_pilot_"))
    report: dict = {"scratch_dir": str(base), "mode": args.mode,
                    "case_id": args.case_id}
    timeline: list[dict] = []
    procs: list[subprocess.Popen] = []
    try:
        local_env: dict[str, str] | None = None
        if args.mode == "local":
            local_env = _local_env(args, base / "artifacts")
            _spawn_local_api(local_env, procs)
            _spawn_local_worker(local_env, procs)
        else:
            report["enforce_safety_env"] = True  # compose pins this to "1"

        ready = _await_ready(args.api_url)
        timeline.append({"event": "ready", "checks": ready.get("checks")})

        created = _submit(args.api_url, args.api_token, args.case_id)
        run_id = created["run_id"]
        report["run_id"] = run_id
        timeline.append({"event": "submitted", "status":
                         str(created.get("status"))})

        def record() -> dict:
            return _get_record(args.api_url, args.api_token, run_id)

        _await_running(record)
        timeline.append({"event": "running_on_first_worker"})

        victim = _crash_and_replace(args, procs, local_env)
        timeline.append({"event": "worker_crashed", "victim": victim})

        final, claim_max = _await_terminal(record)
        report.update({
            "final_status": final["status"],
            "claim_count_observed_max": claim_max,
            "termination_reason": final.get("termination_reason"),
            "timeline": timeline,
        })

        status, result = _http("GET",
                               f"{args.api_url}/runs/{run_id}/result",
                               token=args.api_token)
        payload = result or {}
        report["termination"] = payload.get("termination")
        report["steps"] = payload.get("steps")
        report["tokens"] = payload.get("tokens")
        report["tool_calls"] = payload.get("tool_calls")
        report["recoverable_errors"] = payload.get("recoverable_errors")
        report["fatal_errors"] = payload.get("fatal_errors")

        ok = (
            report["final_status"] == "completed"
            and claim_max >= 2
            and report["termination"] == "success"
            and int(report.get("recoverable_errors") or 0) > 0
        )
        report["verdict"] = "PASS" if ok else "FAIL"
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    out = base / "pilot_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\npilot report: {out}")
    return 0 if report.get("verdict") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
