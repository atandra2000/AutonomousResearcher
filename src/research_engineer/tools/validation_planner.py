"""Validation Planner Tool for Phase 3 - Experiment Planner.

Creates validation strategies with unit tests, integration tests,
numerical equivalence tests, regression tests, performance benchmarks,
and checkpoint compatibility tests.
"""

from datetime import datetime

from pydantic import BaseModel, Field

from research_engineer.models.planner import (
    RiskLevel,
    TestCase,
    TestSuite,
    ValidationPlan,
    ValidationTestType,
)
from research_engineer.models.repo import RepositorySummary
from research_engineer.models.summary import ResearchSummary
from research_engineer.tools.base import Tool, ToolError


class ValidationPlannerInput(BaseModel):
    """Input for validation planning."""

    paper_id: str = Field(..., description="Paper ID")
    repo_path: str = Field(..., description="Repository path")
    summary: ResearchSummary = Field(..., description="Paper summary")
    repo_summary: RepositorySummary = Field(..., description="Repository summary")


class ValidationPlannerOutput(BaseModel):
    """Output from validation planning."""

    plan: ValidationPlan = Field(..., description="Validation plan")


class ValidationPlannerTool(Tool[ValidationPlannerInput, ValidationPlannerOutput]):
    """Generate validation strategy for paper-repo integration."""

    def __init__(self) -> None:
        pass

    async def validate(self, input: ValidationPlannerInput) -> bool:
        return bool(input.paper_id and input.repo_path and input.summary)

    async def execute(self, input: ValidationPlannerInput) -> ValidationPlannerOutput:
        try:
            suites = self._create_test_suites(input)
            total = sum(len(s.test_cases) for s in suites)

            plan = ValidationPlan(
                paper_id=input.paper_id,
                repo_path=input.repo_path,
                timestamp=datetime.now(),
                test_suites=suites,
                total_test_cases=total,
                validation_approach=self._describe_validation_approach(input),
                acceptance_criteria=self._define_acceptance_criteria(input),
            )

            return ValidationPlannerOutput(plan=plan)

        except Exception as e:
            raise ToolError(f"Validation planning failed: {e}", input, e)

    def _create_test_suites(self, input: ValidationPlannerInput) -> list[TestSuite]:
        suites: list[TestSuite] = []

        suites.append(self._create_unit_tests(input))
        suites.append(self._create_integration_tests(input))
        suites.append(self._create_numerical_equivalence_tests(input))
        suites.append(self._create_regression_tests(input))
        suites.append(self._create_performance_benchmarks(input))
        suites.append(self._create_checkpoint_tests(input))

        return suites

    def _create_unit_tests(self, input: ValidationPlannerInput) -> TestSuite:
        arch = input.summary.model_architecture.lower()
        cases: list[TestCase] = [
            TestCase(
                test_name="test_forward_pass_shape",
                test_type=ValidationTestType.UNIT,
                description="Verify forward pass produces correct output shape",
                test_file="tests/test_new_technique.py",
                assertion="Output tensor shape matches expected shape",
                priority=RiskLevel.HIGH,
            ),
            TestCase(
                test_name="test_backward_pass_gradients",
                test_type=ValidationTestType.UNIT,
                description="Verify backward pass computes correct gradients",
                test_file="tests/test_new_technique.py",
                assertion="All parameters receive gradients",
                priority=RiskLevel.HIGH,
            ),
            TestCase(
                test_name="test_initialization",
                test_type=ValidationTestType.UNIT,
                description="Verify module initializes with correct defaults",
                test_file="tests/test_new_technique.py",
                assertion="Module initializes without errors and all parameters exist",
                priority=RiskLevel.MEDIUM,
            ),
        ]

        if "attention" in arch:
            cases.extend([
                TestCase(
                    test_name="test_attention_mask",
                    test_type=ValidationTestType.UNIT,
                    description="Verify attention mask is applied correctly",
                    test_file="tests/test_attention.py",
                    assertion="Masked positions have zero/neg-inf attention weights",
                    priority=RiskLevel.HIGH,
                ),
                TestCase(
                    test_name="test_attention_causal",
                    test_type=ValidationTestType.UNIT,
                    description="Verify causal attention prevents lookahead",
                    test_file="tests/test_attention.py",
                    assertion="Future positions have zero attention weight",
                    priority=RiskLevel.HIGH,
                ),
            ])

        if "moe" in arch or "mixture" in arch:
            cases.extend([
                TestCase(
                    test_name="test_expert_routing",
                    test_type=ValidationTestType.UNIT,
                    description="Verify expert routing selects correct experts",
                    test_file="tests/test_moe.py",
                    assertion="Router outputs valid expert indices and weights",
                    priority=RiskLevel.HIGH,
                ),
                TestCase(
                    test_name="test_expert_balance",
                    test_type=ValidationTestType.UNIT,
                    description="Verify load balancing loss functions correctly",
                    test_file="tests/test_moe.py",
                    assertion="Load balancing loss produces gradient and pushes toward balance",
                    priority=RiskLevel.MEDIUM,
                ),
            ])

        return TestSuite(
            suite_name="Unit Tests",
            test_type=ValidationTestType.UNIT,
            description="Verify individual components work correctly in isolation",
            test_cases=cases,
        )

    def _create_integration_tests(self, input: ValidationPlannerInput) -> TestSuite:
        return TestSuite(
            suite_name="Integration Tests",
            test_type=ValidationTestType.INTEGRATION,
            description="Verify components work together correctly",
            test_cases=[
                TestCase(
                    test_name="test_model_instantiation",
                    test_type=ValidationTestType.INTEGRATION,
                    description="Verify model with new technique instantiates correctly",
                    test_file="tests/test_integration.py",
                    assertion="Model creates without errors with new technique config",
                    priority=RiskLevel.HIGH,
                ),
                TestCase(
                    test_name="test_training_step",
                    test_type=ValidationTestType.INTEGRATION,
                    description="Verify a single training step completes",
                    test_file="tests/test_integration.py",
                    assertion="Training step runs, loss is finite, gradients exist",
                    priority=RiskLevel.HIGH,
                ),
                TestCase(
                    test_name="test_inference_step",
                    test_type=ValidationTestType.INTEGRATION,
                    description="Verify inference produces valid output",
                    test_file="tests/test_integration.py",
                    assertion="Inference output is finite and has correct shapes",
                    priority=RiskLevel.HIGH,
                ),
                TestCase(
                    test_name="test_checkpoint_save_load",
                    test_type=ValidationTestType.INTEGRATION,
                    description="Verify checkpoint save and load works",
                    test_file="tests/test_integration.py",
                    assertion="Model weights match after save/load cycle",
                    priority=RiskLevel.MEDIUM,
                ),
                TestCase(
                    test_name="test_config_loading",
                    test_type=ValidationTestType.INTEGRATION,
                    description="Verify configuration loads correctly",
                    test_file="tests/test_integration.py",
                    assertion="All config parameters are parsed correctly",
                    priority=RiskLevel.MEDIUM,
                ),
            ],
        )

    def _create_numerical_equivalence_tests(self, input: ValidationPlannerInput) -> TestSuite:
        return TestSuite(
            suite_name="Numerical Equivalence Tests",
            test_type=ValidationTestType.NUMERICAL_EQUIVALENCE,
            description="Verify numerical correctness against reference implementations",
            test_cases=[
                TestCase(
                    test_name="test_output_numerical_equivalence",
                    test_type=ValidationTestType.NUMERICAL_EQUIVALENCE,
                    description="Verify output matches reference implementation within tolerance",
                    test_file="tests/test_numerical.py",
                    assertion="Max absolute difference < 1e-5 for deterministic operations",
                    priority=RiskLevel.HIGH,
                ),
                TestCase(
                    test_name="test_gradient_numerical_equivalence",
                    test_type=ValidationTestType.NUMERICAL_EQUIVALENCE,
                    description="Verify gradients match reference implementation",
                    test_file="tests/test_numerical.py",
                    assertion="Gradient difference < 1e-4 for all parameters",
                    priority=RiskLevel.HIGH,
                ),
                TestCase(
                    test_name="test_determinism",
                    test_type=ValidationTestType.NUMERICAL_EQUIVALENCE,
                    description="Verify deterministic behavior with same seed",
                    test_file="tests/test_numerical.py",
                    assertion="Same seed produces identical results across runs",
                    priority=RiskLevel.MEDIUM,
                ),
            ],
        )

    def _create_regression_tests(self, input: ValidationPlannerInput) -> TestSuite:
        return TestSuite(
            suite_name="Regression Tests",
            test_type=ValidationTestType.REGRESSION,
            description="Verify new changes don't break existing functionality",
            test_cases=[
                TestCase(
                    test_name="test_baseline_matches",
                    test_type=ValidationTestType.REGRESSION,
                    description="Verify baseline model produces same metrics as before changes",
                    test_file="tests/test_regression.py",
                    assertion="Baseline loss/perplexity within 0.5% of reference",
                    priority=RiskLevel.HIGH,
                ),
                TestCase(
                    test_name="test_existing_ops_unchanged",
                    test_type=ValidationTestType.REGRESSION,
                    description="Verify existing operations are unchanged",
                    test_file="tests/test_regression.py",
                    assertion="Known test cases produce same outputs",
                    priority=RiskLevel.HIGH,
                ),
                TestCase(
                    test_name="test_config_backward_compatible",
                    test_type=ValidationTestType.REGRESSION,
                    description="Verify old configs still work",
                    test_file="tests/test_regression.py",
                    assertion="Old config files load and train without errors",
                    priority=RiskLevel.MEDIUM,
                ),
            ],
        )

    def _create_performance_benchmarks(self, input: ValidationPlannerInput) -> TestSuite:
        return TestSuite(
            suite_name="Performance Benchmarks",
            test_type=ValidationTestType.PERFORMANCE,
            description="Benchmark performance of new vs old implementation",
            test_cases=[
                TestCase(
                    test_name="bench_forward_pass_latency",
                    test_type=ValidationTestType.PERFORMANCE,
                    description="Measure forward pass latency",
                    test_file="tests/bench_performance.py",
                    assertion="Forward pass latency is within 2x of baseline",
                    priority=RiskLevel.MEDIUM,
                ),
                TestCase(
                    test_name="bench_memory_usage",
                    test_type=ValidationTestType.PERFORMANCE,
                    description="Measure peak GPU memory usage",
                    test_file="tests/bench_performance.py",
                    assertion="Peak memory within GPU limits and reasonable vs baseline",
                    priority=RiskLevel.MEDIUM,
                ),
                TestCase(
                    test_name="bench_throughput",
                    test_type=ValidationTestType.PERFORMANCE,
                    description="Measure training throughput (tokens/sec)",
                    test_file="tests/bench_performance.py",
                    assertion="Throughput is within 2x of baseline",
                    priority=RiskLevel.MEDIUM,
                ),
                TestCase(
                    test_name="bench_convergence_speed",
                    test_type=ValidationTestType.PERFORMANCE,
                    description="Measure steps to converge to target loss",
                    test_file="tests/bench_performance.py",
                    assertion="Convergence speed is comparable to baseline",
                    priority=RiskLevel.LOW,
                ),
            ],
        )

    def _create_checkpoint_tests(self, input: ValidationPlannerInput) -> TestSuite:
        return TestSuite(
            suite_name="Checkpoint Compatibility Tests",
            test_type=ValidationTestType.CHECKPOINT_COMPAT,
            description="Verify checkpoint compatibility for migration",
            test_cases=[
                TestCase(
                    test_name="test_checkpoint_load_old",
                    test_type=ValidationTestType.CHECKPOINT_COMPAT,
                    description="Verify old checkpoints can be loaded",
                    test_file="tests/test_checkpoint_compat.py",
                    assertion="Old checkpoint loads without errors",
                    priority=RiskLevel.HIGH,
                ),
                TestCase(
                    test_name="test_checkpoint_save_load_roundtrip",
                    test_type=ValidationTestType.CHECKPOINT_COMPAT,
                    description="Verify save/load roundtrip preserves weights",
                    test_file="tests/test_checkpoint_compat.py",
                    assertion="Weights match after save/load",
                    priority=RiskLevel.HIGH,
                ),
                TestCase(
                    test_name="test_checkpoint_migration",
                    test_type=ValidationTestType.CHECKPOINT_COMPAT,
                    description="Verify checkpoint migration from old to new format",
                    test_file="tests/test_checkpoint_compat.py",
                    assertion="Migrated checkpoint produces valid model state",
                    priority=RiskLevel.MEDIUM,
                ),
            ],
        )

    def _describe_validation_approach(self, input: ValidationPlannerInput) -> str:
        return (
            "Validate implementation in stages: (1) Unit tests verify individual components, "
            "(2) Integration tests verify end-to-end functionality, "
            "(3) Numerical equivalence tests verify correctness against reference, "
            "(4) Regression tests ensure no functionality breakage, "
            "(5) Performance benchmarks establish efficiency baselines, "
            "(6) Checkpoint compatibility tests verify migration works."
        )

    def _define_acceptance_criteria(self, input: ValidationPlannerInput) -> list[str]:
        return [
            "All unit tests pass",
            "All integration tests pass",
            "Numerical equivalence within tolerance",
            "No regression in baseline metrics",
            "Memory usage within GPU limits",
            "Checkpoint save/load works correctly",
            "Training converges without NaN/Inf",
            "Forward pass latency within 2x of baseline",
        ]
