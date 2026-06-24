"""Implementation Planner Tool for Phase 3 - Experiment Planner.

Generates step-by-step implementation plans for integrating
a research paper's technique into a target repository. Identifies
files, classes, and interfaces requiring modification.
"""

from datetime import datetime

from pydantic import BaseModel, Field

from research_engineer.models.planner import (
    DifficultyLevel,
    ImplementationPlan,
    ImplementationStep,
    ImplementationTarget,
    RiskLevel,
)
from research_engineer.models.repo import RepositorySummary
from research_engineer.models.summary import ResearchSummary
from research_engineer.tools.base import Tool, ToolError


class ImplementationPlannerInput(BaseModel):
    """Input for implementation planning."""

    paper_id: str = Field(..., description="Paper ID")
    repo_path: str = Field(..., description="Repository path")
    summary: ResearchSummary = Field(..., description="Paper summary")
    repo_summary: RepositorySummary = Field(
        ..., description="Repository summary"
    )
    compatibility_level: str = Field(
        default="Medium", description="Overall compatibility level"
    )


class ImplementationPlannerOutput(BaseModel):
    """Output from implementation planning."""

    plan: ImplementationPlan = Field(..., description="Implementation plan")


_PAPER_TECHNIQUE_KEYWORDS: dict[str, list[str]] = {
    "attention": [
        "attention", "flash attention", "ring attention",
        "multi-head", "cross-attention", "self-attention",
        "gqa", "grouped query", "mla", "multi-head latent",
    ],
    "optimizer": [
        "optimizer", "adam", "adamw", "sgd", "lion",
        "schedule-free", "learning rate",
    ],
    "normalization": [
        "normalization", "layer norm", "batch norm", "rmsnorm",
        "group norm",
    ],
    "activation": [
        "activation", "relu", "gelu", "swiglu", "silu",
        "glu", "mish",
    ],
    "architectural": [
        "moe", "mixture of experts", "transformer", "encoder",
        "decoder", "residual", "skip connection",
    ],
    "training_technique": [
        "distillation", "pre-training", "fine-tuning", "rlhf",
        "dpo", "ppo", "curriculum",
    ],
    "data": [
        "data augmentation", "tokenization", "dataset",
        "preprocessing", "dataloader",
    ],
}


class ImplementationPlannerTool(Tool[ImplementationPlannerInput, ImplementationPlannerOutput]):
    """Generate implementation plans for paper-repo integration."""

    def __init__(self) -> None:
        pass

    async def validate(self, input: ImplementationPlannerInput) -> bool:
        return bool(input.paper_id and input.repo_path and input.summary and input.repo_summary)

    async def execute(self, input: ImplementationPlannerInput) -> ImplementationPlannerOutput:
        try:
            targets = self._identify_targets(input)
            steps = self._generate_steps(input, targets)
            total_effort = self._estimate_total_effort(steps)

            plan = ImplementationPlan(
                paper_id=input.paper_id,
                repo_path=input.repo_path,
                timestamp=datetime.now(),
                steps=steps,
                targets=targets,
                total_estimated_effort=total_effort,
                prerequisite_knowledge=self._identify_prerequisites(input),
                success_criteria=self._define_success_criteria(input),
            )

            return ImplementationPlannerOutput(plan=plan)

        except Exception as e:
            raise ToolError(f"Implementation planning failed: {e}", input, e)

    def _detect_technique_categories(self, summary: ResearchSummary) -> list[str]:
        arch_text = summary.model_architecture.lower()
        train_text = summary.training_methodology.lower()
        combined = arch_text + " " + train_text

        categories = []
        for category, keywords in _PAPER_TECHNIQUE_KEYWORDS.items():
            if any(kw in combined for kw in keywords):
                categories.append(category)
        return categories if categories else ["architectural"]

    def _identify_targets(self, input: ImplementationPlannerInput) -> list[ImplementationTarget]:  # noqa: C901
        targets: list[ImplementationTarget] = []
        repo = input.repo_summary
        categories = self._detect_technique_categories(input.summary)
        important_files = repo.important_files[:10]
        impl_targets = repo.implementation_targets[:10]

        for category in categories:
            if category == "attention":
                targets.extend(self._identify_attention_targets(repo, important_files))
            elif category == "optimizer":
                targets.extend(self._identify_optimizer_targets(repo, important_files))
            elif category == "normalization":
                targets.extend(self._identify_norm_targets(repo, important_files))
            elif category == "architectural":
                targets.extend(self._identify_architecture_targets(repo, important_files))
            elif category == "training_technique":
                targets.extend(self._identify_training_targets(repo, important_files))
            elif category == "data":
                targets.extend(self._identify_data_targets(repo, important_files))
            elif category == "activation":
                targets.extend(self._identify_activation_targets(repo, important_files))

        for tgt in impl_targets[:5]:
            existing_paths = {t.file_path for t in targets}
            if tgt.file_path not in existing_paths:
                targets.append(
ImplementationTarget(
                    file_path=tgt.file_path,
                    class_name=tgt.class_name,
                    method_name=tgt.method_name,
                    target_type="integration",
                    modification_type="modify",
                    description=f"Modify {tgt.class_name or tgt.method_name or tgt.file_path}",
                    estimated_lines=tgt.estimated_lines if hasattr(tgt, "estimated_lines") else 30,
                    complexity=DifficultyLevel.MODERATE,
                ))

        return targets[:15]

    def _identify_attention_targets(
        self, repo: RepositorySummary, important_files: list
    ) -> list[ImplementationTarget]:
        targets: list[ImplementationTarget] = []
        attn_file = None
        config_file = None
        model_file = None

        for f in important_files:
            path_lower = f.file_path.lower() if hasattr(f, "file_path") else str(f).lower()
            if "attention" in path_lower:
                attn_file = f.file_path if hasattr(f, "file_path") else str(f)
            elif "config" in path_lower:
                config_file = f.file_path if hasattr(f, "file_path") else str(f)
            elif "model" in path_lower:
                model_file = f.file_path if hasattr(f, "file_path") else str(f)

        targets.append(
ImplementationTarget(
            file_path=attn_file or "models/attention.py",
            class_name="Attention",
            method_name=None,
            target_type="attention",
            modification_type="modify",
            description="Modify attention mechanism (add new variant)",
            estimated_lines=80,
            complexity=DifficultyLevel.HARD,
        ))
        targets.append(
ImplementationTarget(
            file_path=config_file or "configs/model_config.yaml",
            class_name=None, method_name=None,
            target_type="config",
            modification_type="extend",
            description="Add configuration for new attention variant",
            estimated_lines=20,
            complexity=DifficultyLevel.EASY,
        ))
        if model_file:
            targets.append(
ImplementationTarget(
                file_path=model_file,
                class_name="Model",
                method_name=None,
                target_type="model",
                modification_type="modify",
                description="Update model to use new attention variant",
                estimated_lines=30,
                complexity=DifficultyLevel.MODERATE,
            ))

        return targets

    def _identify_optimizer_targets(
        self, repo: RepositorySummary, important_files: list
    ) -> list[ImplementationTarget]:
        targets: list[ImplementationTarget] = []
        targets.append(
ImplementationTarget(
            file_path="optim/optimizer.py",
            class_name=None, method_name=None,
            target_type="optimizer",
            modification_type="add",
            description="Add new optimizer implementation",
            estimated_lines=100,
            complexity=DifficultyLevel.HARD,
        ))
        targets.append(
ImplementationTarget(
            file_path="configs/train_config.yaml",
            class_name=None, method_name=None,
            target_type="config",
            modification_type="extend",
            description="Add optimizer configuration parameters",
            estimated_lines=15,
            complexity=DifficultyLevel.EASY,
        ))
        targets.append(
ImplementationTarget(
            file_path="trainer.py",
            class_name=None, method_name=None,
            target_type="training",
            modification_type="modify",
            description="Update training loop to use new optimizer",
            estimated_lines=25,
            complexity=DifficultyLevel.MODERATE,
        ))
        return targets

    def _identify_norm_targets(
        self, repo: RepositorySummary, important_files: list
    ) -> list[ImplementationTarget]:
        return [

ImplementationTarget(
                file_path="models/layers.py",
            class_name=None, method_name=None,
                target_type="normalization",
                modification_type="add",
                description="Add new normalization layer",
                estimated_lines=40,
                complexity=DifficultyLevel.MODERATE,
            ),

ImplementationTarget(
                file_path="configs/model_config.yaml",
            class_name=None, method_name=None,
                target_type="config",
                modification_type="extend",
                description="Add normalization configuration",
                estimated_lines=10,
                complexity=DifficultyLevel.EASY,
            ),
        ]

    def _identify_architecture_targets(
        self, repo: RepositorySummary, important_files: list
    ) -> list[ImplementationTarget]:
        targets: list[ImplementationTarget] = []
        targets.append(
ImplementationTarget(
            file_path="models/model.py",
            class_name=None, method_name=None,
            target_type="model",
            modification_type="modify",
            description="Update model architecture",
            estimated_lines=150,
            complexity=DifficultyLevel.HARD,
        ))
        targets.append(
ImplementationTarget(
            file_path="configs/model_config.yaml",
            class_name=None, method_name=None,
            target_type="config",
            modification_type="extend",
            description="Add model configuration parameters",
            estimated_lines=25,
            complexity=DifficultyLevel.EASY,
        ))
        targets.append(
ImplementationTarget(
            file_path="trainer.py",
            class_name=None, method_name=None,
            target_type="training",
            modification_type="modify",
            description="Update training loop for architectural changes",
            estimated_lines=35,
            complexity=DifficultyLevel.MODERATE,
        ))
        return targets

    def _identify_training_targets(
        self, repo: RepositorySummary, important_files: list
    ) -> list[ImplementationTarget]:
        return [

ImplementationTarget(
                file_path="trainer.py",
            class_name=None, method_name=None,
                target_type="training",
                modification_type="modify",
                description="Update training loop for new technique",
                estimated_lines=60,
                complexity=DifficultyLevel.HARD,
            ),

ImplementationTarget(
                file_path="configs/train_config.yaml",
            class_name=None, method_name=None,
                target_type="config",
                modification_type="extend",
                description="Add training hyperparameters",
                estimated_lines=15,
                complexity=DifficultyLevel.EASY,
            ),
        ]

    def _identify_data_targets(
        self, repo: RepositorySummary, important_files: list
    ) -> list[ImplementationTarget]:
        return [

ImplementationTarget(
                file_path="data/dataset.py",
            class_name=None, method_name=None,
                target_type="dataset",
                modification_type="modify",
                description="Update data pipeline",
                estimated_lines=50,
                complexity=DifficultyLevel.MODERATE,
            ),

ImplementationTarget(
                file_path="data/preprocessing.py",
            class_name=None, method_name=None,
                target_type="preprocessing",
                modification_type="add",
                description="Add preprocessing logic",
                estimated_lines=40,
                complexity=DifficultyLevel.MODERATE,
            ),
        ]

    def _identify_activation_targets(
        self, repo: RepositorySummary, important_files: list
    ) -> list[ImplementationTarget]:
        return [

ImplementationTarget(
                file_path="models/activations.py",
            class_name=None, method_name=None,
                target_type="activation",
                modification_type="add",
                description="Add new activation function",
                estimated_lines=25,
                complexity=DifficultyLevel.EASY,
            ),

ImplementationTarget(
                file_path="models/model.py",
            class_name=None, method_name=None,
                target_type="model",
                modification_type="modify",
                description="Replace activation in model",
                estimated_lines=15,
                complexity=DifficultyLevel.EASY,
            ),
        ]

    def _generate_steps(
        self, input: ImplementationPlannerInput, targets: list[ImplementationTarget]
    ) -> list[ImplementationStep]:
        steps: list[ImplementationStep] = []

        steps.append(ImplementationStep(
            step_number=1,
            title="Understand existing codebase",
            description="Read and understand the relevant modules, classes, and interfaces in the target repository before making changes.",
            targets=[],
            difficulty=DifficultyLevel.EASY,
            dependencies=[],
            estimated_effort="0.5-1 day",
            risk_level=RiskLevel.LOW,
            validation_criteria=["Can explain module structure", "Identified all relevant files"],
        ))

        steps.append(ImplementationStep(
            step_number=2,
            title="Set up development environment",
            description="Create a branch, install dependencies, verify existing tests pass, and set up experiment tracking.",
            targets=[],
            difficulty=DifficultyLevel.TRIVIAL,
            dependencies=[1],
            estimated_effort="0.5 day",
            risk_level=RiskLevel.LOW,
            validation_criteria=["Branch created", "Tests pass", "Environment verified"],
        ))

        step_num = 3
        config_targets = [t for t in targets if t.target_type == "config"]
        if config_targets:
            steps.append(ImplementationStep(
                step_number=step_num,
                title="Update configuration files",
                description="Add configuration parameters for the new technique. This is typically low-risk and should be done first.",
                targets=config_targets,
                difficulty=DifficultyLevel.EASY,
                dependencies=[2],
                estimated_effort="0.5-1 day",
                risk_level=RiskLevel.LOW,
                validation_criteria=["Config loads without errors", "Default values are reasonable"],
            ))
            step_num += 1

        core_targets = [t for t in targets if t.target_type in ("attention", "optimizer", "normalization", "activation")]
        if core_targets:
            steps.append(ImplementationStep(
                step_number=step_num,
                title="Implement core technique",
                description="Implement the core technical contribution from the paper. This is the highest-risk step and should be tested thoroughly.",
                targets=core_targets,
                difficulty=DifficultyLevel.HARD,
                dependencies=[step_num - 1],
                estimated_effort="2-5 days",
                risk_level=RiskLevel.HIGH,
                validation_criteria=["Forward pass produces correct shapes", "Numerical results match paper", "Unit tests pass"],
            ))
            step_num += 1

        model_targets = [t for t in targets if t.target_type == "model"]
        if model_targets:
            steps.append(ImplementationStep(
                step_number=step_num,
                title="Integrate into model",
                description="Update model class to use the new technique. May require interface changes.",
                targets=model_targets,
                difficulty=DifficultyLevel.MODERATE,
                dependencies=[step_num - 1],
                estimated_effort="1-2 days",
                risk_level=RiskLevel.MEDIUM,
                validation_criteria=["Model instantiates correctly", "Forward/backward pass works", "No shape errors"],
            ))
            step_num += 1

        training_targets = [t for t in targets if t.target_type == "training"]
        if training_targets:
            steps.append(ImplementationStep(
                step_number=step_num,
                title="Update training loop",
                description="Modify training loop to accommodate the new technique (new optimizer, loss, scheduling, etc.).",
                targets=training_targets,
                difficulty=DifficultyLevel.MODERATE,
                dependencies=[step_num - 1],
                estimated_effort="1-2 days",
                risk_level=RiskLevel.MEDIUM,
                validation_criteria=["Training runs without errors", "Loss decreases", "No NaN/Inf"],
            ))
            step_num += 1

        steps.append(ImplementationStep(
            step_number=step_num,
            title="Write tests and validation",
            description="Write comprehensive tests: unit tests for the core technique, integration tests for model-level, and regression tests for baseline comparison.",
            targets=[],
            difficulty=DifficultyLevel.MODERATE,
            dependencies=[step_num - 1],
            estimated_effort="1-2 days",
            risk_level=RiskLevel.LOW,
            validation_criteria=["All unit tests pass", "Integration tests pass", "Baseline matches"],
        ))
        step_num += 1

        steps.append(ImplementationStep(
            step_number=step_num,
            title="Run baseline and validation experiments",
            description="Run baseline experiments to verify no regression, then run initial experiments with the new technique.",
            targets=[],
            difficulty=DifficultyLevel.MODERATE,
            dependencies=[step_num - 1],
            estimated_effort="1-3 days",
            risk_level=RiskLevel.MEDIUM,
            validation_criteria=["Baseline metrics match", "New technique trains successfully", "Preliminary results show expected trends"],
        ))

        return steps

    def _estimate_total_effort(self, steps: list[ImplementationStep]) -> str:
        high_count = sum(1 for s in steps if s.difficulty == DifficultyLevel.HARD)
        moderate_count = sum(1 for s in steps if s.difficulty == DifficultyLevel.MODERATE)

        if high_count >= 2:
            return "2-3 weeks for an experienced ML engineer"
        elif high_count >= 1:
            return "1-2 weeks for an experienced ML engineer"
        elif moderate_count >= 3:
            return "1-2 weeks for an experienced ML engineer"
        else:
            return "3-5 days for an experienced ML engineer"

    def _identify_prerequisites(self, input: ImplementationPlannerInput) -> list[str]:
        categories = self._detect_technique_categories(input.summary)
        prereqs = ["PyTorch fundamentals", "Model training pipeline"]

        if "attention" in categories:
            prereqs.extend(["Attention mechanisms", "CUDA kernel basics (for FlashAttention-type work)"])
        if "optimizer" in categories:
            prereqs.extend(["Optimization theory", "Learning rate scheduling"])
        if "distributed" in categories:
            prereqs.extend(["Distributed training (DDP/FSDP)", "NCCL communication"])
        if "moe" in input.summary.model_architecture.lower() or "mixture" in input.summary.model_architecture.lower():
            prereqs.extend(["Mixture of Experts theory", "Routing mechanisms"])
        return prereqs

    def _define_success_criteria(self, input: ImplementationPlannerInput) -> list[str]:
        return [
            "New technique integrates without breaking existing functionality",
            "All existing tests pass with new changes",
            "Baseline metrics are reproduced within acceptable tolerance",
            "New technique shows expected behavior in simple tests",
            "Training completes without NaN/Inf errors",
            "Memory usage stays within GPU limits",
        ]
