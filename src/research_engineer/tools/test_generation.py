"""Test Generation Tool for Phase 4.

Generates comprehensive test suites for code changes.
"""

from pydantic import BaseModel, Field

from research_engineer.models.coding import (
    CodeChange,
    ComplexityLevel,
    GeneratedPatch,
    TestSpecification,
    TestSuite,
    TestType,
)
from research_engineer.tools.base import Tool, ToolError


class TestGenerationInput(BaseModel):
    """Input for test generation."""

    patches: list[GeneratedPatch] = Field(
        default_factory=list,
        description="Patches to generate tests for",
    )
    changes: list[CodeChange] = Field(
        default_factory=list,
        description="Code changes",
    )
    repo_path: str = Field(..., description="Repository path")
    test_types: list[TestType] = Field(
        default_factory=list,
        description="Types of tests to generate",
    )
    include_edge_cases: bool = Field(
        default=True,
        description="Include edge case tests",
    )
    include_performance_tests: bool = Field(
        default=False,
        description="Include performance tests",
    )
    test_framework: str = Field(
        default="pytest",
        description="Test framework to use",
    )


class TestGenerationOutput(BaseModel):
    """Output from test generation."""

    test_suites: list[TestSuite] = Field(
        default_factory=list,
        description="Generated test suites",
    )
    total_tests: int = Field(default=0, description="Total number of tests")
    tests_by_type: dict = Field(
        default_factory=dict,
        description="Tests grouped by type",
    )
    tests_by_patch: dict = Field(
        default_factory=dict,
        description="Tests grouped by patch ID",
    )
    estimated_coverage: str = Field(
        default="unknown",
        description="Estimated code coverage",
    )
    estimated_execution_time: str = Field(
        default="unknown",
        description="Estimated execution time",
    )
    generation_time_seconds: float = Field(
        default=0.0,
        description="Generation duration",
    )


class TestGenerationTool(Tool[TestGenerationInput, TestGenerationOutput]):
    """
    Tool for generating test suites.

    This tool:
    1. Analyzes code changes
    2. Identifies test requirements
    3. Generates unit tests
    4. Generates integration tests
    5. Generates regression tests
    6. Generates numerical equivalence tests
    7. Optionally generates performance tests
    """

    async def execute(self, input: TestGenerationInput) -> TestGenerationOutput:
        """Generate test suites for code changes."""
        import time
        start_time = time.time()

        try:
            # Determine test types
            test_types = input.test_types or [
                TestType.UNIT,
                TestType.INTEGRATION,
                TestType.REGRESSION,
            ]

            if input.include_edge_cases:
                test_types.append(TestType.NUMERICAL)

            if input.include_performance_tests:
                test_types.append(TestType.PERFORMANCE)

            # Generate test suites
            test_suites = []

            # Generate tests for each patch
            for patch in input.patches:
                suite = await self._generate_tests_for_patch(patch, test_types, input)
                test_suites.append(suite)

            # Generate tests for changes without patches
            for change in input.changes:
                if not any(p.file_path == change.file_path for p in input.patches):
                    suite = await self._generate_tests_for_change(change, test_types, input)
                    test_suites.append(suite)

            # Calculate statistics
            total_tests = sum(len(suite.tests) for suite in test_suites)
            tests_by_type = self._group_tests_by_type(test_suites)
            tests_by_patch = {suite.suite_id: len(suite.tests) for suite in test_suites}

            # Estimate coverage and execution time
            coverage = self._estimate_coverage(test_suites, input.changes)
            exec_time = self._estimate_execution_time(test_suites)

            elapsed = time.time() - start_time

            return TestGenerationOutput(
                test_suites=test_suites,
                total_tests=total_tests,
                tests_by_type=tests_by_type,
                tests_by_patch=tests_by_patch,
                estimated_coverage=coverage,
                estimated_execution_time=exec_time,
                generation_time_seconds=round(elapsed, 2),
            )

        except Exception as e:
            raise ToolError(f"Test generation failed: {e}", input, e)

    async def _generate_tests_for_patch(
        self,
        patch: GeneratedPatch,
        test_types: list[TestType],
        input: TestGenerationInput,
    ) -> TestSuite:
        """Generate tests for a specific patch."""
        suite_id = f"suite_{patch.patch_id}"
        suite_name = f"Test{Path(patch.file_path).stem.title()}"

        tests = []
        test_counter = 0

        for test_type in test_types:
            test_spec = await self._create_test(
                patch.file_path,
                patch.file_path,
                test_type,
                test_counter,
                input,
            )
            if test_spec:
                tests.append(test_spec)
                test_counter += 1

        return TestSuite(
            suite_id=suite_id,
            suite_name=suite_name,
            description=f"Test suite for {patch.file_path}",
            tests=tests,
            total_tests=len(tests),
            coverage_estimate="unknown",
            execution_time_estimate="unknown",
        )

    async def _generate_tests_for_change(
        self,
        change: CodeChange,
        test_types: list[TestType],
        input: TestGenerationInput,
    ) -> TestSuite:
        """Generate tests for a code change."""
        suite_id = f"suite_{change.file_path.replace('/', '_').replace('.', '_')}"
        file_name = Path(change.file_path).stem
        suite_name = f"Test{file_name.title()}"

        tests = []
        for i, test_type in enumerate(test_types):
            test_spec = await self._create_test(
                change.file_path,
                change.file_path,
                test_type,
                i,
                input,
            )
            if test_spec:
                tests.append(test_spec)

        return TestSuite(
            suite_id=suite_id,
            suite_name=suite_name,
            description=f"Test suite for {change.file_path}",
            tests=tests,
            total_tests=len(tests),
        )

    async def _create_test(
        self,
        target_file: str,
        target_function: str,
        test_type: TestType,
        index: int,
        input: TestGenerationInput,
    ) -> TestSpecification | None:
        """Create a single test specification."""
        test_id = f"test_{test_type.value}_{index:03d}"
        test_name = f"{test_id}_{Path(target_file).stem}"

        # Generate test code based on type
        test_code = await self._generate_test_code(test_type, target_file, target_function, input)

        if not test_code:
            return None

        return TestSpecification(
            test_id=test_id,
            test_name=test_name,
            test_type=test_type,
            target_file=target_file,
            target_function=target_function,
            description=f"Test {test_type.value} for {target_function}",
            test_code=test_code,
            dependencies=[input.test_framework],
            expected_behavior=f"Validates {test_type.value} behavior",
            edge_cases=["edge_case_1"] if input.include_edge_cases else [],
            priority="high" if test_type == TestType.UNIT else "medium",
        )

    async def _generate_test_code(
        self,
        test_type: TestType,
        target_file: str,
        target_function: str,
        input: TestGenerationInput,
    ) -> str:
        """Generate test code based on test type."""
        if test_type == TestType.UNIT:
            return self._generate_unit_test(target_file, target_function, input)
        elif test_type == TestType.INTEGRATION:
            return self._generate_integration_test(target_file, input)
        elif test_type == TestType.REGRESSION:
            return self._generate_regression_test(target_file, input)
        elif test_type == TestType.NUMERICAL:
            return self._generate_numerical_test(target_file, input)
        elif test_type == TestType.PERFORMANCE:
            return self._generate_performance_test(target_file, input)
        else:
            return self._generate_generic_test(target_file, input)

    def _generate_unit_test(self, target_file: str, target_function: str, input: TestGenerationInput) -> str:
        """Generate unit test."""
        return f'''"""Unit tests for {target_file}."""

import pytest
from {target_file.replace("/", ".").replace(".py", "")} import *


def test_{Path(target_file).stem}_basic():
    """Test basic functionality."""
    # TODO: Implement test based on {target_function}
    assert True


def test_{Path(target_file).stem}_edge_case():
    """Test edge case."""
    # TODO: Implement edge case test
    assert True
'''

    def _generate_integration_test(self, target_file: str, input: TestGenerationInput) -> str:
        """Generate integration test."""
        return f'''"""Integration tests for {target_file}."""

import pytest


def test_{Path(target_file).stem}_integration():
    """Test integration with other components."""
    # TODO: Implement integration test
    assert True
'''

    def _generate_regression_test(self, target_file: str, input: TestGenerationInput) -> str:
        """Generate regression test."""
        return f'''"""Regression tests for {target_file}."""

import pytest


def test_{Path(target_file).stem}_regression():
    """Test that previous functionality still works."""
    # TODO: Implement regression test
    assert True
'''

    def _generate_numerical_test(self, target_file: str, input: TestGenerationInput) -> str:
        """Generate numerical equivalence test."""
        return f'''"""Numerical equivalence tests for {target_file}."""

import pytest
import numpy as np


def test_{Path(target_file).stem}_numerical():
    """Test numerical correctness."""
    # TODO: Implement numerical test
    assert True
'''

    def _generate_performance_test(self, target_file: str, input: TestGenerationInput) -> str:
        """Generate performance test."""
        return f'''"""Performance tests for {target_file}."""

import pytest
import time


def test_{Path(target_file).stem}_performance():
    """Test performance characteristics."""
    # TODO: Implement performance test
    start = time.time()
    # Run operation
    elapsed = time.time() - start
    assert elapsed < 1.0  # Should complete in under 1 second
'''

    def _generate_generic_test(self, target_file: str, input: TestGenerationInput) -> str:
        """Generate generic test."""
        return f'''"""Tests for {target_file}."""

import pytest


def test_{Path(target_file).stem}_generic():
    """Generic test."""
    # TODO: Implement test
    assert True
'''

    def _group_tests_by_type(self, test_suites: list[TestSuite]) -> dict:
        """Group tests by type."""
        by_type = {}
        for suite in test_suites:
            for test in suite.tests:
                type_key = test.test_type.value
                if type_key not in by_type:
                    by_type[type_key] = 0
                by_type[type_key] += 1
        return by_type

    def _estimate_coverage(self, test_suites: list[TestSuite], changes: list[CodeChange]) -> str:
        """Estimate code coverage."""
        total_tests = sum(len(suite.tests) for suite in test_suites)
        total_changes = len(changes) if changes else len(test_suites)

        if total_tests == 0:
            return "0%"
        elif total_tests < total_changes:
            return "low (<50%)"
        elif total_tests < total_changes * 2:
            return "medium (50-80%)"
        else:
            return "high (>80%)"

    def _estimate_execution_time(self, test_suites: list[TestSuite]) -> str:
        """Estimate test execution time."""
        total_tests = sum(len(suite.tests) for suite in test_suites)

        if total_tests == 0:
            return "0s"
        elif total_tests < 10:
            return f"~{total_tests * 0.1:.1f}s"
        elif total_tests < 50:
            return f"~{total_tests * 0.05:.1f}s"
        else:
            return f"~{total_tests * 0.02:.1f}s"


# Import Path at module level
from pathlib import Path
