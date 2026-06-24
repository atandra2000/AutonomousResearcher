"""Tests for Phase 3 planner models."""


from research_engineer.models.planner import (
    CompatibilityDimension,
    CompatibilityLevel,
    CompatibilityReport,
    ComputeEstimate,
    ConfidenceLevel,
    DifficultyLevel,
    Experiment,
    ExperimentType,
    FailureMode,
    ImpactDimension,
    ImplementationPlan,
    ImplementationStep,
    ImplementationTarget,
    MetricDefinition,
    ResultPrediction,
    RiskItem,
    RiskLevel,
    ScenarioOutcome,
    ValidationTestType,
)


class TestCompatibilityLevel:
    def test_values(self):
        assert CompatibilityLevel.LOW.value == "Low"
        assert CompatibilityLevel.MEDIUM.value == "Medium"
        assert CompatibilityLevel.HIGH.value == "High"


class TestRiskLevel:
    def test_values(self):
        assert RiskLevel.LOW.value == "Low"
        assert RiskLevel.MEDIUM.value == "Medium"
        assert RiskLevel.HIGH.value == "High"


class TestConfidenceLevel:
    def test_values(self):
        assert ConfidenceLevel.LOW.value == "Low"
        assert ConfidenceLevel.MEDIUM.value == "Medium"
        assert ConfidenceLevel.HIGH.value == "High"


class TestDifficultyLevel:
    def test_values(self):
        assert DifficultyLevel.TRIVIAL.value == "Trivial"
        assert DifficultyLevel.EASY.value == "Easy"
        assert DifficultyLevel.MODERATE.value == "Moderate"
        assert DifficultyLevel.HARD.value == "Hard"
        assert DifficultyLevel.VERY_HARD.value == "Very Hard"


class TestExperimentType:
    def test_values(self):
        assert ExperimentType.BASELINE.value == "baseline"
        assert ExperimentType.MINIMUM_VIABLE.value == "minimum_viable"
        assert ExperimentType.ABLATION.value == "ablation"
        assert ExperimentType.STRESS.value == "stress"
        assert ExperimentType.SCALING.value == "scaling"


class TestValidationTestType:
    def test_values(self):
        assert ValidationTestType.UNIT.value == "unit"
        assert ValidationTestType.INTEGRATION.value == "integration"
        assert ValidationTestType.NUMERICAL_EQUIVALENCE.value == "numerical_equivalence"
        assert ValidationTestType.REGRESSION.value == "regression"
        assert ValidationTestType.PERFORMANCE.value == "performance"
        assert ValidationTestType.CHECKPOINT_COMPAT.value == "checkpoint_compatibility"


class TestCompatibilityDimension:
    def test_creation(self):
        dim = CompatibilityDimension(
            dimension="Architecture",
            level=CompatibilityLevel.HIGH,
            reasoning="Good compatibility",
            evidence=["Test evidence"],
            blockers=[],
        )
        assert dim.dimension == "Architecture"
        assert dim.level == CompatibilityLevel.HIGH
        assert dim.reasoning == "Good compatibility"

    def test_defaults(self):
        dim = CompatibilityDimension(
            dimension="Training",
            level=CompatibilityLevel.MEDIUM,
            reasoning="Reasonable fit",
        )
        assert dim.evidence == []
        assert dim.blockers == []


class TestCompatibilityReport:
    def test_creation(self):
        dim = CompatibilityDimension(
            dimension="Test",
            level=CompatibilityLevel.HIGH,
            reasoning="Works",
        )
        report = CompatibilityReport(
            paper_id="2503.12345",
            repo_path="/test/repo",
            architecture_compatibility=dim,
            training_compatibility=dim,
            inference_compatibility=dim,
            config_compatibility=dim,
            checkpoint_compatibility=dim,
            distributed_compatibility=dim,
            evaluation_compatibility=dim,
            overall_compatibility=CompatibilityLevel.MEDIUM,
            overall_reasoning="Mixed compatibility",
        )
        assert report.paper_id == "2503.12345"
        assert report.overall_compatibility == CompatibilityLevel.MEDIUM


class TestImplementationTarget:
    def test_creation(self):
        target = ImplementationTarget(
            file_path="models/attention.py",
            target_type="attention",
            modification_type="modify",
            description="Add new attention variant",
        )
        assert target.file_path == "models/attention.py"
        assert target.target_type == "attention"
        assert target.modification_type == "modify"
        assert target.complexity == DifficultyLevel.MODERATE

    def test_with_class(self):
        target = ImplementationTarget(
            file_path="models/model.py",
            class_name="GPTModel",
            method_name="forward",
            target_type="model",
            modification_type="extend",
            description="Update forward pass",
            estimated_lines=30,
            complexity=DifficultyLevel.HARD,
        )
        assert target.class_name == "GPTModel"
        assert target.method_name == "forward"


class TestImplementationStep:
    def test_creation(self):
        step = ImplementationStep(
            step_number=1,
            title="Setup",
            description="Set up development environment",
        )
        assert step.step_number == 1
        assert step.title == "Setup"
        assert step.difficulty == DifficultyLevel.MODERATE
        assert step.dependencies == []

    def test_full_step(self):
        step = ImplementationStep(
            step_number=2,
            title="Implement attention",
            description="Add new attention mechanism",
            difficulty=DifficultyLevel.HARD,
            dependencies=[1],
            estimated_effort="3-5 days",
            risk_level=RiskLevel.HIGH,
            validation_criteria=["Forward pass works", "No NaN values"],
        )
        assert step.risk_level == RiskLevel.HIGH
        assert len(step.validation_criteria) == 2


class TestImplementationPlan:
    def test_creation(self):
        step = ImplementationStep(
            step_number=1,
            title="Setup",
            description="Initial setup",
        )
        target = ImplementationTarget(
            file_path="test.py",
            target_type="test",
            modification_type="add",
            description="Add test file",
        )
        plan = ImplementationPlan(
            paper_id="2503.12345",
            repo_path="/test",
            steps=[step],
            targets=[target],
        )
        assert plan.paper_id == "2503.12345"
        assert len(plan.steps) == 1
        assert len(plan.targets) == 1


class TestImpactDimension:
    def test_creation(self):
        impact = ImpactDimension(
            dimension="Memory Usage",
            current_estimate="Standard footprint",
            projected_estimate="Reduced memory",
            change_direction="decrease",
            confidence=ConfidenceLevel.HIGH,
            reasoning="Efficient attention reduces memory",
        )
        assert impact.change_direction == "decrease"


class TestMetricDefinition:
    def test_creation(self):
        metric = MetricDefinition(
            name="loss",
            description="Training loss",
            unit="nats",
            why_important="Primary convergence indicator",
        )
        assert metric.name == "loss"
        assert metric.target_value is None


class TestExperiment:
    def test_creation(self):
        exp = Experiment(
            experiment_id="exp_001",
            experiment_type=ExperimentType.BASELINE,
            title="Baseline",
            description="Run baseline",
            hypothesis="Baseline produces expected metrics",
        )
        assert exp.experiment_id == "exp_001"
        assert exp.experiment_type == ExperimentType.BASELINE


class TestRiskItem:
    def test_creation(self):
        risk = RiskItem(
            risk_id="RISK-001",
            category="implementation",
            description="Implementation may fail",
            level=RiskLevel.HIGH,
            probability=ConfidenceLevel.MEDIUM,
            impact="Cannot reproduce results",
            mitigation="Write tests first",
        )
        assert risk.risk_id == "RISK-001"
        assert risk.level == RiskLevel.HIGH


class TestComputeEstimate:
    def test_creation(self):
        estimate = ComputeEstimate(
            paper_id="2503.12345",
            repo_path="/test",
            gpu_type="A100",
            gpu_count_per_experiment=2,
            total_experiments=5,
            total_gpu_hours=50.0,
        )
        assert estimate.gpu_type == "A100"
        assert estimate.total_gpu_hours == 50.0


class TestScenarioOutcome:
    def test_creation(self):
        outcome = ScenarioOutcome(
            scenario="best_case",
            description="Everything works perfectly",
            metrics={"quality": "SOTA"},
            probability=ConfidenceLevel.LOW,
        )
        assert outcome.scenario == "best_case"


class TestFailureMode:
    def test_creation(self):
        fm = FailureMode(
            failure_id="FAIL-001",
            description="Training diverges",
            trigger="Bad learning rate",
            detection="Loss monitoring",
            recovery="Lower LR, check gradients",
        )
        assert fm.failure_id == "FAIL-001"


class TestResultPrediction:
    def test_creation(self):
        best = ScenarioOutcome(
            scenario="best_case",
            description="Perfect.",
            metrics={"quality": "Great"},
        )
        likely = ScenarioOutcome(
            scenario="likely_case",
            description="Good enough.",
            metrics={"quality": "OK"},
        )
        worst = ScenarioOutcome(
            scenario="worst_case",
            description="Bad.",
            metrics={"quality": "Poor"},
        )
        pred = ResultPrediction(
            paper_id="2503.12345",
            repo_path="/test",
            best_case=best,
            likely_case=likely,
            worst_case=worst,
        )
        assert pred.paper_id == "2503.12345"
        assert pred.overall_confidence == ConfidenceLevel.MEDIUM
