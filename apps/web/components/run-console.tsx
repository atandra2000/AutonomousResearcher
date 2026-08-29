"use client";

import { type FormEvent, useEffect, useState } from "react";

type RunStatus = {
  run_id: string;
  status: string;
  created_at?: string;
  updated_at?: string;
  error?: string | null;
  has_checkpoint?: boolean;
  claim_count?: number;
};

export function RunConsole({ operator }: { operator: string }) {
  const [goal, setGoal] = useState("");
  const [run, setRun] = useState<RunStatus | null>(null);
  const [status, setStatus] = useState("Ready to submit a dry-run research goal.");
  const [submitting, setSubmitting] = useState(false);
  const [acting, setActing] = useState(false);
  const [result, setResult] = useState<unknown>(null);

  useEffect(() => {
    if (!run || ["completed", "failed", "cancelled"].includes(run.status)) return;
    const timer = window.setInterval(async () => {
      const response = await fetch(`/api/runs/${encodeURIComponent(run.run_id)}`, { cache: "no-store" });
      if (!response.ok) return;
      const next = (await response.json()) as RunStatus;
      setRun(next);
      setStatus(`Run ${next.run_id} is ${next.status}.`);
    }, 2_000);
    return () => window.clearInterval(timer);
  }, [run?.run_id, run?.status]);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmedGoal = goal.trim();
    if (!trimmedGoal) {
      setStatus("Enter a research goal before submitting.");
      return;
    }
    setSubmitting(true);
    setStatus("Submitting run…");
    try {
      const response = await fetch("/api/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ goal: trimmedGoal })
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail ?? "Run submission failed.");
      setRun(payload as RunStatus);
      setStatus(`Run ${payload.run_id} queued.`);
      setGoal("");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Run submission failed.");
    } finally {
      setSubmitting(false);
    }
  }

  async function action(name: "cancel" | "resume" | "result") {
    if (!run) return;
    setActing(true);
    setStatus(`${name === "result" ? "Loading result" : `${name[0].toUpperCase()}${name.slice(1)}ing run`}…`);
    try {
      const response = await fetch(`/api/runs/${encodeURIComponent(run.run_id)}/${name}`, {
        method: name === "result" ? "GET" : "POST",
        cache: "no-store"
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail ?? `Run ${name} failed.`);
      if (name === "result") {
        setResult(payload);
        setStatus(`Result for ${run.run_id} loaded.`);
      } else {
        setRun((current) => current ? { ...current, ...payload } : current);
        setStatus(`Run ${run.run_id} ${payload.status}.`);
      }
    } catch (error) {
      setStatus(error instanceof Error ? error.message : `Run ${name} failed.`);
    } finally {
      setActing(false);
    }
  }

  return (
    <main>
      <p className="eyebrow">Autonomous ML Research Engineer</p>
      <h1>Run operations console</h1>
      <p className="operator">Authenticated operator: {operator}</p>
      <div className="grid">
        <section className="panel" aria-labelledby="submit-title">
          <h2 id="submit-title">Submit research workflow</h2>
          <form onSubmit={submit}>
            <label htmlFor="goal">Research goal
              <textarea id="goal" value={goal} onChange={(event) => setGoal(event.target.value)} maxLength={10_000} required />
            </label>
            <button type="submit" disabled={submitting}>{submitting ? "Submitting…" : "Queue run"}</button>
          </form>
          <p className={status.includes("failed") ? "status error" : "status"} aria-live="polite">{status}</p>
        </section>
        <section className="panel" aria-labelledby="run-title">
          <h2 id="run-title">Latest run</h2>
          {run ? (
            <dl>
              <dt>Run</dt><dd>{run.run_id}</dd>
              <dt>Status</dt><dd>{run.status}</dd>
              <dt>Checkpoint</dt><dd>{run.has_checkpoint ? "available" : "not yet created"}</dd>
              <dt>Claims</dt><dd>{run.claim_count ?? "not yet claimed"}</dd>
              <dt>Error</dt><dd>{run.error ?? "—"}</dd>
            </dl>
          ) : <p className="operator">No run selected.</p>}
          {run && (
            <p className="actions">
              <button type="button" onClick={() => action("cancel")} disabled={acting || run.status !== "queued"}>Cancel</button>
              <button type="button" onClick={() => action("resume")} disabled={acting || !["failed", "cancelled"].includes(run.status)}>Resume</button>
              <button type="button" onClick={() => action("result")} disabled={acting || run.status !== "completed"}>View result</button>
            </p>
          )}
          {result !== null && <pre aria-label="Run result">{JSON.stringify(result, null, 2)}</pre>}
        </section>
      </div>
    </main>
  );
}
