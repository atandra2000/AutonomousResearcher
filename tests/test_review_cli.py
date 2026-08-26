"""Tests for D3 - Human-in-the-loop UX (rich ApprovalRequest + review CLI)."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from research_engineer.agents.research_loop_agent import format_approval_prompt
from research_engineer.cli import app
from research_engineer.models.loop import ApprovalGate, ApprovalRequest

runner = CliRunner()


class TestApprovalRequestContextFields:
    """D3 extended ApprovalRequest with rich context fields."""

    def test_defaults_are_optional(self):
        r = ApprovalRequest(loop_id="l1", iteration_number=1, gate=ApprovalGate.PLAN)
        assert r.plan_diff is None
        assert r.expected_cost_usd is None
        assert r.expected_gpu_hours is None
        assert r.risk_summary is None
        assert r.risk_level is None
        assert r.metric_snapshot == {}
        assert r.model_name is None

    def test_populated_fields_round_trip(self):
        r = ApprovalRequest(
            loop_id="l1", iteration_number=2, gate=ApprovalGate.IMPLEMENTATION,
            summary="patch", plan_diff="@@ -1 +1 @@",
            expected_cost_usd=0.05, expected_gpu_hours=0.5,
            risk_summary="may break tests", risk_level="medium",
            metric_snapshot={"loss": 0.03}, model_name="kimi-k2.7-code",
        )
        d = r.model_dump()
        assert d["plan_diff"] == "@@ -1 +1 @@"
        assert d["expected_cost_usd"] == 0.05
        assert d["risk_level"] == "medium"
        assert d["metric_snapshot"] == {"loss": 0.03}
        assert d["model_name"] == "kimi-k2.7-code"

    def test_negative_cost_rejected(self):
        with pytest.raises(Exception):
            ApprovalRequest(
                loop_id="l1", iteration_number=1, gate=ApprovalGate.PLAN,
                expected_cost_usd=-1.0,
            )


class TestFormatApprovalPrompt:
    def test_minimal_request(self):
        r = ApprovalRequest(loop_id="l1", iteration_number=1, gate=ApprovalGate.PLAN)
        out = format_approval_prompt(r)
        assert "Approval Gate: plan" in out
        assert "Loop: l1" in out
        assert "Options:" in out

    def test_includes_context_when_present(self):
        r = ApprovalRequest(
            loop_id="l1", iteration_number=1, gate=ApprovalGate.PLAN,
            summary="add EMA", expected_cost_usd=0.0123, expected_gpu_hours=1.5,
            risk_level="high", risk_summary="might OOM",
            metric_snapshot={"loss": 0.05, "acc": 0.9},
            model_name="glm-5.2:cloud", plan_diff="@@ -1 +1 @@\n+ema",
        )
        out = format_approval_prompt(r)
        assert "add EMA" in out
        assert "Model: glm-5.2:cloud" in out
        assert "Expected cost: $0.0123" in out
        assert "Expected GPU-hours: 1.500" in out
        assert "high" in out
        assert "might OOM" in out
        assert "loss=0.05" in out
        assert "Plan diff" in out

    def test_diff_truncation(self):
        long_diff = "x" * 3000
        r = ApprovalRequest(
            loop_id="l1", iteration_number=1, gate=ApprovalGate.PLAN, plan_diff=long_diff,
        )
        out = format_approval_prompt(r)
        assert "truncated" in out
class TestReviewPlanCommand:
    def test_help(self):
        result = runner.invoke(app, ["review", "plan", "--help"])
        assert result.exit_code == 0

    def test_auto_approve_with_yes(self):
        result = runner.invoke(
            app, ["review", "plan", "--summary", "my plan", "--cost-usd", "0.1", "--yes"],
        )
        assert result.exit_code == 0
        assert "Auto-approved" in result.output

    def test_json_output(self):
        result = runner.invoke(
            app, ["review", "plan", "--summary", "json plan", "--format", "json"],
        )
        assert result.exit_code == 0
        assert "plan_diff" in result.output

    def test_reads_diff_file(self, tmp_path: Path):
        diff_file = tmp_path / "plan.diff"
        diff_file.write_text("@@ -1 +1 @@\n+new line")
        result = runner.invoke(
            app, ["review", "plan", "--diff", str(diff_file), "--yes"],
        )
        assert result.exit_code == 0
        assert "new line" in result.output

    def test_metric_snapshot_parsed(self):
        result = runner.invoke(
            app, ["review", "plan", "--metric", "loss=0.05,acc=0.9", "--format", "json"],
        )
        assert result.exit_code == 0
        assert "loss" in result.output
        assert "acc" in result.output


class TestReviewPatchCommand:
    def test_help(self):
        result = runner.invoke(app, ["review", "patch", "--help"])
        assert result.exit_code == 0

    def test_reads_patch_file(self, tmp_path: Path):
        patch = tmp_path / "c.patch"
        patch.write_text("--- a\n+++ b\n@@ -1 +1 @@\n-x\n+y")
        result = runner.invoke(
            app, ["review", "patch", "--diff", str(patch), "--yes"],
        )
        assert result.exit_code == 0
        assert "y" in result.output

    def test_missing_diff_file(self):
        result = runner.invoke(
            app, ["review", "patch", "--diff", "/nonexistent/path.patch", "--yes"],
        )
        assert result.exit_code == 1
        assert "Could not read patch file" in result.output


class TestReviewExperimentCommand:
    def test_help(self):
        result = runner.invoke(app, ["review", "experiment", "--help"])
        assert result.exit_code == 0

    def test_auto_approve_with_cost(self):
        result = runner.invoke(
            app, ["review", "experiment", "--cost-usd", "0.2", "--gpu-hours", "1.0", "--yes"],
        )
        assert result.exit_code == 0
        assert "Auto-approved" in result.output
        assert "Expected cost: $0.2000" in result.output

    def test_json_output(self):
        result = runner.invoke(
            app, ["review", "experiment", "--metric", "loss=0.01", "--format", "json"],
        )
        assert result.exit_code == 0
        assert "loss" in result.output
