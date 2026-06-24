"""Tests for ExperimentAgent (Phase 7)."""

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from research_engineer.agents.experiment_agent import ExperimentAgent
from research_engineer.models.experiment import (
    ExperimentConfig,
    ExperimentQueryInput,
    ExperimentStatus,
    ExperimentType,
)


@pytest.fixture
def mock_memory_agent():
    agent = MagicMock()
    agent.storage = AsyncMock()
    agent.storage.store_relationship = AsyncMock(return_value=MagicMock(relationship_id="rel_1"))
    agent.graph = MagicMock()
    agent.graph.add_node = MagicMock()
    agent.graph.add_relationship = MagicMock()
    agent.store_insight = AsyncMock(return_value="mem_insight")
    agent.store_success = AsyncMock(return_value="mem_success")
    agent.store_failure = AsyncMock(return_value="mem_failure")
    return agent


@pytest.fixture
def experiment_agent(mock_memory_agent, tmp_path):
    config = ExperimentConfig(
        default_timeout_seconds=10,
        dry_run_default=True,
        output_dir=str(tmp_path / "experiments"),
        store_results=True,
        update_graph=True,
    )
    return ExperimentAgent(
        memory_agent=mock_memory_agent,
        config=config,
    )


class TestExperimentAgentInit:
    def test_defaults(self):
        agent = ExperimentAgent()
        assert agent.memory is None
        assert agent.runner is not None
        assert agent.monitoring is not None
        assert agent.metrics is not None
        assert agent.artifacts is not None
        assert agent.failure is not None
        assert agent.storage is not None

    def test_config(self):
        agent = ExperimentAgent()
        assert agent.config.dry_run_default is True


class TestRunWorkflow:
    @pytest.mark.asyncio
    async def test_dry_run(self, experiment_agent, tmp_path):
        result = await experiment_agent.run(
            command="python -c print(1)",
            repo_path=str(tmp_path),
            dry_run=True,
        )
        assert result.experiment_id.startswith("exp_")
        assert result.run is not None
        assert result.run.status == ExperimentStatus.PENDING

    @pytest.mark.asyncio
    async def test_real_run_success(self, experiment_agent, tmp_path):
        result = await experiment_agent.run(
            command=[sys.executable, "-c", "print('hello')"],
            repo_path=str(tmp_path),
            dry_run=False,
            timeout_seconds=10,
        )
        assert result.run is not None
        assert result.run.status == ExperimentStatus.COMPLETED
        assert result.metrics is not None
        assert result.failure is not None

    @pytest.mark.asyncio
    async def test_real_run_failure(self, experiment_agent, tmp_path):
        result = await experiment_agent.run(
            command=[sys.executable, "-c", "import sys; sys.exit(1)"],
            repo_path=str(tmp_path),
            dry_run=False,
            timeout_seconds=10,
        )
        assert result.run.status == ExperimentStatus.FAILED
        assert result.failure is not None
        assert result.failure.detected_failure is True

    @pytest.mark.asyncio
    async def test_stores_in_memory_success(
        self, experiment_agent, mock_memory_agent, tmp_path
    ):
        await experiment_agent.run(
            command=[sys.executable, "-c", "print(1)"],
            repo_path=str(tmp_path),
            dry_run=False,
            timeout_seconds=10,
        )
        assert mock_memory_agent.store_success.called

    @pytest.mark.asyncio
    async def test_stores_in_memory_failure(
        self, experiment_agent, mock_memory_agent, tmp_path
    ):
        await experiment_agent.run(
            command=[sys.executable, "-c", "raise RuntimeError('OOM')"],
            repo_path=str(tmp_path),
            dry_run=False,
            timeout_seconds=10,
        )
        assert mock_memory_agent.store_failure.called

    @pytest.mark.asyncio
    async def test_graph_updated_on_success(
        self, experiment_agent, mock_memory_agent, tmp_path
    ):
        await experiment_agent.run(
            command=[sys.executable, "-c", "print(1)"],
            repo_path=str(tmp_path),
            dry_run=False,
            timeout_seconds=10,
        )
        assert mock_memory_agent.graph.add_node.called
        assert mock_memory_agent.graph.add_relationship.called

    @pytest.mark.asyncio
    async def test_output_files_generated(self, experiment_agent, tmp_path):
        result = await experiment_agent.run(
            command=[sys.executable, "-c", "print(1)"],
            repo_path=str(tmp_path),
            dry_run=False,
            timeout_seconds=10,
        )
        assert len(result.generated_files) > 0
        for f in result.generated_files:
            from pathlib import Path

            assert Path(f).exists()

    @pytest.mark.asyncio
    async def test_with_paper_id(self, experiment_agent, tmp_path):
        result = await experiment_agent.run(
            command=[sys.executable, "-c", "print(1)"],
            repo_path=str(tmp_path),
            paper_id="2503.12345",
            plan_id="plan1",
            patch_id="patch1",
            dry_run=False,
            timeout_seconds=10,
        )
        assert result.paper_id == "2503.12345"
        assert result.plan_id == "plan1"
        assert result.patch_id == "patch1"

    @pytest.mark.asyncio
    async def test_record_stored(self, experiment_agent, tmp_path):
        result = await experiment_agent.run(
            command=[sys.executable, "-c", "print(1)"],
            repo_path=str(tmp_path),
            dry_run=False,
            timeout_seconds=10,
        )
        assert result.record is not None
        assert result.record.status == ExperimentStatus.COMPLETED


class TestConvenienceMethods:
    @pytest.mark.asyncio
    async def test_run_training(self, experiment_agent, tmp_path):
        result = await experiment_agent.run_training(
            command=[sys.executable, "-c", "print(1)"],
            repo_path=str(tmp_path),
            dry_run=False,
            timeout_seconds=10,
        )
        assert result.run.experiment_type == ExperimentType.TRAINING

    @pytest.mark.asyncio
    async def test_run_evaluation(self, experiment_agent, tmp_path):
        result = await experiment_agent.run_evaluation(
            command=[sys.executable, "-c", "print(1)"],
            repo_path=str(tmp_path),
            dry_run=False,
            timeout_seconds=10,
        )
        assert result.run.experiment_type == ExperimentType.EVALUATION

    @pytest.mark.asyncio
    async def test_run_validation(self, experiment_agent, tmp_path):
        result = await experiment_agent.run_validation(
            command=[sys.executable, "-c", "print(1)"],
            repo_path=str(tmp_path),
            dry_run=False,
            timeout_seconds=10,
        )
        assert result.run.experiment_type == ExperimentType.VALIDATION


class TestQueryMethods:
    @pytest.mark.asyncio
    async def test_list_experiments(self, experiment_agent, tmp_path):
        await experiment_agent.run(
            command=[sys.executable, "-c", "print(1)"],
            repo_path=str(tmp_path),
            dry_run=False,
            timeout_seconds=10,
        )
        out = await experiment_agent.list_experiments(
            repo_path=str(tmp_path)
        )
        assert out.total >= 1

    @pytest.mark.asyncio
    async def test_get_experiment(self, experiment_agent, tmp_path):
        result = await experiment_agent.run(
            command=[sys.executable, "-c", "print(1)"],
            repo_path=str(tmp_path),
            dry_run=False,
            timeout_seconds=10,
        )
        record = await experiment_agent.get_experiment(result.experiment_id)
        assert record is not None
        assert record.experiment_id == result.experiment_id

    @pytest.mark.asyncio
    async def test_search_experiments(self, experiment_agent, tmp_path):
        await experiment_agent.run(
            command=[sys.executable, "-c", "print(1)"],
            repo_path=str(tmp_path),
            dry_run=False,
            timeout_seconds=10,
        )
        out = await experiment_agent.search_experiments("python")
        assert out.total >= 1

    @pytest.mark.asyncio
    async def test_history(self, experiment_agent, tmp_path):
        await experiment_agent.run(
            command=[sys.executable, "-c", "print(1)"],
            repo_path=str(tmp_path),
            paper_id="2503.12345",
            dry_run=False,
            timeout_seconds=10,
        )
        out = await experiment_agent.history("2503.12345")
        assert out.total >= 1

    @pytest.mark.asyncio
    async def test_cancel_nonexistent(self, experiment_agent):
        cancelled = await experiment_agent.cancel_experiment("nonexistent")
        assert cancelled is False


class TestParseCommand:
    def test_parse_string(self):
        cmd = ExperimentAgent._parse_command("python train.py --lr 0.001")
        assert cmd == ["python", "train.py", "--lr", "0.001"]

    def test_parse_list(self):
        cmd = ExperimentAgent._parse_command(["python", "train.py"])
        assert cmd == ["python", "train.py"]


class TestIsSuccess:
    @pytest.mark.asyncio
    async def test_result_is_success(self, experiment_agent, tmp_path):
        result = await experiment_agent.run(
            command=[sys.executable, "-c", "print(1)"],
            repo_path=str(tmp_path),
            dry_run=False,
            timeout_seconds=10,
        )
        assert result.is_success() is True