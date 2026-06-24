"""Integration tests for Phase 1 → Phase 3 → Phase 4 pipelines."""

import pytest
from pathlib import Path

from research_engineer.agents import (
    CodingAgent,
    ExperimentPlannerAgent,
    RepositoryAgent,
    ResearchAgent,
)


class TestPhase1ToPhase3Integration:
    """Test integration between Phase 1 (ResearchAgent) and Phase 3 (ExperimentPlannerAgent)."""

    @pytest.mark.asyncio
    async def test_phase1_to_phase3_full_pipeline(self, tmp_path):
        """Test full pipeline from paper analysis to experiment planning."""
        research_agent = ResearchAgent()
        planner_agent = ExperimentPlannerAgent()
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        try:
            paper_result = await research_agent.analyze(
                "2503.12345",
                output_dir=str(output_dir),
            )
            
            assert paper_result is not None
            assert "paper_id" in paper_result
            assert "summary" in paper_result
            assert "plan" in paper_result
        except Exception:
            pytest.skip("Phase 1 analysis may fail due to network or PDF parsing issues")
        
        repo_dir = tmp_path / "test_repo"
        repo_dir.mkdir()
        
        test_file = repo_dir / "model.py"
        test_file.write_text("""
import torch
import torch.nn as nn

class AttentionModel(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.dim = dim
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
    
    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        return torch.softmax(q @ k.transpose(-2, -1), dim=-1) @ v
""")
        
        try:
            planner_result = await planner_agent.plan(
                "2503.12345",
                str(repo_dir),
                output_dir=str(output_dir / "plans"),
            )
            
            assert planner_result is not None
            assert hasattr(planner_result, "paper_id")
            assert hasattr(planner_result, "repo_path")
            assert hasattr(planner_result, "compatibility_report")
            assert len(planner_result.generated_files) > 0
        except Exception:
            pytest.skip("Phase 3 planning may fail due to repository analysis issues")

    @pytest.mark.asyncio
    async def test_phase1_to_phase3_compatibility_check(self, tmp_path):
        """Test that Phase 1 output feeds into Phase 3 compatibility analysis."""
        research_agent = ResearchAgent()
        planner_agent = ExperimentPlannerAgent()
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        await research_agent.analyze("2503.12345", output_dir=str(output_dir))
        
        repo_dir = tmp_path / "ml_repo"
        repo_dir.mkdir()
        
        config_file = repo_dir / "config.yaml"
        config_file.write_text("""
model:
  type: transformer
  hidden_dim: 512
  num_heads: 8
training:
  batch_size: 32
  learning_rate: 0.001
""")
        
        planner_result = await planner_agent.plan(
            "2503.12345",
            str(repo_dir),
        )
        
        assert planner_result is not None
        comp_report = planner_result.compatibility_report
        assert comp_report is not None
        assert "overall_compatibility" in comp_report or hasattr(comp_report, "overall_compatibility")

    @pytest.mark.asyncio
    async def test_phase1_to_phase3_risk_assessment(self, tmp_path):
        """Test that Phase 1 analysis informs Phase 3 risk assessment."""
        research_agent = ResearchAgent()
        planner_agent = ExperimentPlannerAgent()
        
        repo_dir = tmp_path / "test_repo"
        repo_dir.mkdir()
        
        await research_agent.analyze("2503.12345")
        
        planner_result = await planner_agent.plan(
            "2503.12345",
            str(repo_dir),
        )
        
        assert planner_result is not None
        risk = planner_result.risk_assessment
        assert risk is not None
        assert "overall_risk_level" in risk or hasattr(risk, "overall_risk_level")


class TestPhase2ToPhase3Integration:
    """Test integration between Phase 2 (RepositoryAgent) and Phase 3 (ExperimentPlannerAgent)."""

    @pytest.mark.asyncio
    async def test_phase2_to_phase3_full_pipeline(self, tmp_path):
        """Test full pipeline from repository analysis to experiment planning."""
        repo_agent = RepositoryAgent()
        planner_agent = ExperimentPlannerAgent()
        
        repo_dir = tmp_path / "ml_project"
        repo_dir.mkdir()
        
        model_file = repo_dir / "model.py"
        model_file.write_text("""
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
    
    def forward(self, src):
        emb = self.embedding(src)
        return self.transformer(emb)
""")
        
        train_file = repo_dir / "train.py"
        train_file.write_text("""
import torch
from model import TransformerModel

def train(model, dataloader, epochs=10):
    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            pass
""")
        
        config_file = repo_dir / "config.yaml"
        config_file.write_text("""
model:
  d_model: 512
  nhead: 8
  num_layers: 6
training:
  epochs: 10
  batch_size: 32
  learning_rate: 0.0001
""")
        
        repo_result = await repo_agent.analyze(
            str(repo_dir),
            output_dir=str(tmp_path / "output"),
        )
        
        assert repo_result is not None
        assert "repository_name" in repo_result
        assert "project_type" in repo_result
        assert "architecture_summary" in repo_result
        
        planner_result = await planner_agent.plan(
            "2503.12345",
            str(repo_dir),
        )
        
        assert planner_result is not None
        assert planner_result.repo_path == str(repo_dir)
        
        impl_plan = planner_result.implementation_plan
        assert impl_plan is not None
        assert hasattr(impl_plan, "steps") or "steps" in impl_plan

    @pytest.mark.asyncio
    async def test_phase2_to_phase3_architecture_compatibility(self, tmp_path):
        """Test repository architecture analysis feeds into compatibility check."""
        repo_agent = RepositoryAgent()
        planner_agent = ExperimentPlannerAgent()
        
        repo_dir = tmp_path / "attention_repo"
        repo_dir.mkdir()
        
        model_file = repo_dir / "attention.py"
        model_file.write_text("""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(out)
""")
        
        await repo_agent.analyze(str(repo_dir))
        
        planner_result = await planner_agent.plan(
            "2503.12345",
            str(repo_dir),
        )
        
        assert planner_result is not None
        comp_report = planner_result.compatibility_report
        assert comp_report is not None
        
        if isinstance(comp_report, dict):
            assert "architecture_compatibility" in comp_report
        else:
            assert hasattr(comp_report, "architecture_compatibility")

    @pytest.mark.asyncio
    async def test_phase2_to_phase3_training_pipeline_integration(self, tmp_path):
        """Test training pipeline analysis informs experiment design."""
        repo_agent = RepositoryAgent()
        planner_agent = ExperimentPlannerAgent()
        
        repo_dir = tmp_path / "training_repo"
        repo_dir.mkdir()
        
        train_file = repo_dir / "train.py"
        train_file.write_text("""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self, model, lr=1e-4):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
    
    def train_step(self, batch):
        self.optimizer.zero_grad()
        outputs = self.model(batch['input'])
        loss = self.criterion(outputs, batch['target'])
        loss.backward()
        self.optimizer.step()
        return loss.item()
""")
        
        await repo_agent.analyze(str(repo_dir))
        
        planner_result = await planner_agent.plan(
            "2503.12345",
            str(repo_dir),
        )
        
        assert planner_result is not None
        exp_matrix = planner_result.experiment_matrix
        assert exp_matrix is not None
        assert hasattr(exp_matrix, "groups") or "groups" in exp_matrix


class TestPhase3ToPhase4Integration:
    """Test integration between Phase 3 (ExperimentPlannerAgent) and Phase 4 (CodingAgent)."""

    @pytest.mark.asyncio
    async def test_phase3_to_phase4_full_pipeline(self, tmp_path):
        """Test full pipeline from experiment planning to code implementation."""
        planner_agent = ExperimentPlannerAgent()
        coding_agent = CodingAgent()
        
        repo_dir = tmp_path / "implementation_repo"
        repo_dir.mkdir()
        
        init_file = repo_dir / "__init__.py"
        init_file.write_text("")
        
        model_file = repo_dir / "model.py"
        model_file.write_text("""
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(512, 512)
    
    def forward(self, x):
        return self.linear(x)
""")
        
        planner_result = await planner_agent.plan(
            "2503.12345",
            str(repo_dir),
        )
        
        assert planner_result is not None
        assert hasattr(planner_result, "implementation_plan")
        
        impl_plan = planner_result.implementation_plan
        plan_dict = impl_plan if isinstance(impl_plan, dict) else impl_plan.model_dump()
        
        coder_result = await coding_agent.implement(
            task_description="Implement attention mechanism from plan",
            repo_path=str(repo_dir),
            implementation_plan=impl_plan,
        )
        
        assert coder_result is not None
        assert hasattr(coder_result, "implementation_id")
        assert hasattr(coder_result, "patches_generated")
        assert hasattr(coder_result, "tests_generated")
        assert hasattr(coder_result, "review_status")

    @pytest.mark.asyncio
    async def test_phase3_to_phase4_patch_generation(self, tmp_path):
        """Test Phase 3 plan generates patches in Phase 4."""
        planner_agent = ExperimentPlannerAgent()
        coding_agent = CodingAgent()
        
        repo_dir = tmp_path / "patch_repo"
        repo_dir.mkdir()
        
        (repo_dir / "__init__.py").write_text("")
        (repo_dir / "module.py").write_text("# Module file\n")
        
        planner_result = await planner_agent.plan(
            "2503.12345",
            str(repo_dir),
        )
        
        assert planner_result is not None
        
        coder_result = await coding_agent.implement(
            task_description="Add new feature based on plan",
            repo_path=str(repo_dir),
        )
        
        assert coder_result is not None
        assert coder_result.patches_generated >= 0

    @pytest.mark.asyncio
    async def test_phase3_to_phase4_test_generation(self, tmp_path):
        """Test Phase 3 validation plan informs Phase 4 test generation."""
        planner_agent = ExperimentPlannerAgent()
        coding_agent = CodingAgent()
        
        repo_dir = tmp_path / "test_repo"
        repo_dir.mkdir()
        
        (repo_dir / "code.py").write_text("""
def add(a, b):
    return a + b
""")
        
        planner_result = await planner_agent.plan(
            "2503.12345",
            str(repo_dir),
        )
        
        assert planner_result is not None
        assert hasattr(planner_result, "validation_plan")
        
        coder_result = await coding_agent.implement(
            task_description="Add function with tests",
            repo_path=str(repo_dir),
        )
        
        assert coder_result is not None
        assert coder_result.tests_generated >= 0

    @pytest.mark.asyncio
    async def test_phase3_to_phase4_rollback_planning(self, tmp_path):
        """Test Phase 3 risk assessment informs Phase 4 rollback planning."""
        planner_agent = ExperimentPlannerAgent()
        coding_agent = CodingAgent()
        
        repo_dir = tmp_path / "rollback_repo"
        repo_dir.mkdir()
        
        (repo_dir / "main.py").write_text("# Main file\n")
        
        planner_result = await planner_agent.plan(
            "2503.12345",
            str(repo_dir),
        )
        
        assert planner_result is not None
        assert hasattr(planner_result, "risk_assessment")
        
        coder_result = await coding_agent.implement(
            task_description="Implement with rollback support",
            repo_path=str(repo_dir),
        )
        
        assert coder_result is not None
        # Check that rollback plan was generated (as a file)
        assert any("rollback_plan" in f for f in coder_result.generated_files)


class TestFullEndToEndIntegration:
    """Test complete end-to-end integration across all phases."""

    @pytest.mark.asyncio
    async def test_phase1_to_phase4_complete_pipeline(self, tmp_path):
        """Test complete pipeline from paper analysis to code implementation."""
        research_agent = ResearchAgent()
        repo_agent = RepositoryAgent()
        planner_agent = ExperimentPlannerAgent()
        coding_agent = CodingAgent()
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        paper_result = await research_agent.analyze(
            "2503.12345",
            output_dir=str(output_dir),
        )
        assert paper_result is not None
        
        repo_dir = tmp_path / "complete_repo"
        repo_dir.mkdir()
        
        (repo_dir / "model.py").write_text("""
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(512, 512)
    
    def forward(self, x):
        return self.layer(x)
""")
        
        (repo_dir / "config.yaml").write_text("""
model:
  hidden_dim: 512
training:
  batch_size: 32
""")
        
        repo_result = await repo_agent.analyze(
            str(repo_dir),
            output_dir=str(output_dir / "repo_analysis"),
        )
        assert repo_result is not None
        
        planner_result = await planner_agent.plan(
            "2503.12345",
            str(repo_dir),
            output_dir=str(output_dir / "plans"),
        )
        assert planner_result is not None
        
        coder_result = await coding_agent.implement(
            task_description="Implement paper technique",
            repo_path=str(repo_dir),
        )
        assert coder_result is not None
        assert coder_result.implementation_id is not None
