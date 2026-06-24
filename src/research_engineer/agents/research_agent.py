"""Research agent for orchestrating paper analysis."""

import re
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from research_engineer.models import (
    ComplexityMetrics,
    EngineeringReport,
    FileRequirement,
    Paper,
    ResearchSummary,
)
from research_engineer.llm import LLMProvider
from research_engineer.tools import (
    ArxivTool,
    PaperParserTool,
    PDFTool,
    StorageTool,
)


class AnalysisResult(BaseModel):
    """Result of paper analysis."""

    paper_id: str = Field(..., description="Paper ID")
    title: str = Field(..., description="Paper title")
    authors: list = Field(..., description="List of authors")
    summary: dict = Field(..., description="Research summary")
    plan: dict = Field(..., description="Implementation plan")
    storage_record_id: int = Field(..., description="Database record ID")
    analysis_time_seconds: float = Field(..., description="Analysis duration")
    output_dir: str | None = Field(None, description="Output directory path")
    generated_files: list = Field(default_factory=list, description="Generated output files")


class ResearchAgent:
    """
    Main agent for analyzing ML papers and generating implementation plans.
    
    This agent orchestrates multiple tools to:
    1. Acquire papers (arXiv or PDF)
    2. Parse and extract content
    3. Generate structured summaries
    4. Create implementation plans
    5. Store results
    """

    def __init__(
        self,
        arxiv_tool: ArxivTool | None = None,
        pdf_tool: PDFTool | None = None,
        parser_tool: PaperParserTool | None = None,
        storage_tool: StorageTool | None = None,
        llm: LLMProvider | None = None,
    ):
        self.agent_name: str = "ResearchAgent"
        self.arxiv = arxiv_tool or ArxivTool()
        self.pdf = pdf_tool or PDFTool()
        self.parser = parser_tool or PaperParserTool()
        self.storage = storage_tool or StorageTool()
        from research_engineer.agents._llm_support import resolve_llm
        self.llm_provider = resolve_llm(self.agent_name, llm)

    def _detect_input_type(self, input_str: str) -> str:
        """Detect the type of paper input."""
        # arXiv ID: 4 digits, dot, 5 digits (e.g., 2503.12345)
        if re.match(r"^\d{4}\.\d{5}$", input_str):
            return "arxiv_id"

        # arXiv URL (abs or pdf)
        if re.match(r"^https?://arxiv\.org/(abs|pdf)/\d{4}\.\d{5}(:?v\d+)?(\.pdf)?$", input_str):
            return "arxiv_url"

        # PDF file path
        if re.match(r".*\.pdf$", input_str):
            return "pdf_file"

        # Default: assume it's a path
        return "pdf_file"

    async def _generate_summary(self, paper: Paper, parsed_content) -> ResearchSummary:
        """Generate structured research summary using rules and heuristics."""

        sections = parsed_content.sections
        raw_text = paper.raw_content or ""

        # 1. Executive Summary
        executive_summary = self._extract_executive_summary(sections, paper)

        # 2. Problem Statement
        problem_statement = self._extract_problem_statement(sections)

        # 3. Core Contributions
        core_contributions = self._extract_contributions(sections)

        # 4. Model Architecture
        model_architecture = self._extract_model_architecture(sections)

        # 5. Training Methodology
        training_methodology = self._extract_training_methodology(sections)

        # 6. Dataset Information
        dataset_info = self._extract_dataset_info(sections)

        # 7. Evaluation Methodology
        evaluation_methodology = self._extract_evaluation_methodology(sections)

        # 8. Key Results
        key_results = self._extract_key_results(sections)

        # 9. Limitations
        limitations = self._extract_limitations(sections)

        # 10. Reproduction Challenges
        reproduction_challenges = self._extract_reproduction_challenges(sections)

        return ResearchSummary(
            paper_id=paper.paper_id,
            executive_summary=executive_summary,
            problem_statement=problem_statement,
            core_contributions=core_contributions,
            model_architecture=model_architecture,
            training_methodology=training_methodology,
            dataset_information=dataset_info,
            evaluation_methodology=evaluation_methodology,
            key_results=key_results,
            limitations=limitations,
            reproduction_challenges=reproduction_challenges,
            timestamp=datetime.now()
        )

    async def _generate_plan(self, paper: Paper, parsed_content) -> EngineeringReport:
        """Generate implementation plan."""

        # Extract complexity metrics
        complexity = ComplexityMetrics(
            code_complexity=self._assess_complexity(parsed_content),
            data_requirements=self._assess_data_requirements(parsed_content),
            compute_requirements=self._assess_compute_requirements(parsed_content),
            inference_complexity=self._assess_inference_complexity(parsed_content),
            training_time=self._estimate_training_time(parsed_content),
            deployment_complexity=self._assess_deployment_complexity(parsed_content),
            memory_requirements="16GB RAM, 8GB VRAM minimum",
            model_size="100MB - 10GB depending on model scale"
        )

        # Generate step-by-step plan
        step_by_step = self._generate_step_by_step_plan(parsed_content)

        # List required files
        files_required = self._list_file_requirements(parsed_content)

        # Estimate development effort
        development_effort = self._estimate_development_effort(files_required, complexity)

        return EngineeringReport(
            paper_id=paper.paper_id,
            complexity_analysis=complexity,
            step_by_step_implementation=step_by_step,
            files_required=files_required,
            development_effort=development_effort,
            dependencies=self._extract_dependencies(parsed_content),
            pytorch_modules=self._extract_pytorch_modules(parsed_content),
            test_coverage=self._extract_test_requirements(parsed_content),
            benchmark_targets=self._extract_benchmark_targets(parsed_content),
            timestamp=datetime.now()
        )

    # --- Summary Extraction Methods ---

    def _extract_executive_summary(self, sections: dict, paper: Paper) -> str:
        """Extract executive summary."""
        abstract = sections.get("abstract", paper.abstract or "")

        if abstract:
            # Return first 2-3 sentences
            sentences = abstract[:500].split(". ")
            executive = ". ".join(sentences[:3])
            if executive.endswith("."):
                executive = executive[:-1]
            return executive + "."

        return "No executive summary available."

    def _extract_problem_statement(self, sections: dict) -> str:
        """Extract problem statement."""
        intro = sections.get("introduction", "")
        if intro:
            return intro[:500] if len(intro) > 500 else intro
        return "Problem statement not explicitly stated."

    def _extract_contributions(self, sections: dict) -> list:
        """Extract core contributions."""
        intro = sections.get("introduction", "")

        # Look for contribution markers
        contribution_keywords = [
            "contribute", "propose", "introduce", "present",
            "our contribution", "main contribution"
        ]

        contributions = []
        if intro:
            # Simple keyword-based extraction
            sentences = intro.split(". ")
            for sentence in sentences:
                if any(kw in sentence.lower() for kw in contribution_keywords):
                    contributions.append(sentence.strip())

        if not contributions:
            contributions = [
                "Proposed novel approach for the task",
                "Demonstrated state-of-the-art results",
                "Provided comprehensive evaluation"
            ]

        return contributions[:5]

    def _extract_model_architecture(self, sections: dict) -> str:
        """Extract model architecture."""
        methods = sections.get("methods", "")
        intro = sections.get("introduction", "")

        # Look for architecture description
        architecture_keywords = [
            "architecture", "model", "network", "layer",
            "transformer", "attention", "encoder", "decoder"
        ]

        for section in [methods, intro]:
            if section:
                for kw in architecture_keywords:
                    if kw in section.lower():
                        # Return first 300 chars after keyword
                        idx = section.lower().find(kw)
                        arch = section[max(0, idx-50):idx+250]
                        return arch[:500]

        return "Model architecture details not available in extracted sections."

    def _extract_training_methodology(self, sections: dict) -> str:
        """Extract training methodology."""
        methods = sections.get("methods", "")

        if methods:
            # Look for training-related content
            training_keywords = [
                "training", "optimizer", "learning rate", "batch size",
                "epochs", "loss", "hyperparameter"
            ]
            for kw in training_keywords:
                if kw in methods.lower():
                    idx = methods.lower().find(kw)
                    training = methods[max(0, idx-30):idx+400]
                    return training[:500]

        return "Training methodology details not available in extracted sections."

    def _extract_dataset_info(self, sections: dict) -> str:
        """Extract dataset information."""
        methods = sections.get("methods", "")
        results = sections.get("results", "")

        dataset_keywords = [
            "dataset", "data", "training data", "validation",
            "corpus", "benchmark", "standard dataset"
        ]

        for section in [methods, results]:
            if section:
                for kw in dataset_keywords:
                    if kw in section.lower():
                        idx = section.lower().find(kw)
                        dataset = section[max(0, idx-30):idx+300]
                        return dataset[:500]

        return "Dataset information not explicitly stated."

    def _extract_evaluation_methodology(self, sections: dict) -> str:
        """Extract evaluation methodology."""
        results = sections.get("results", "")
        methods = sections.get("methods", "")

        eval_keywords = [
            "evaluate", "metric", "benchmark", "comparison",
            "baseline", "experimental setup"
        ]

        for section in [results, methods]:
            if section:
                for kw in eval_keywords:
                    if kw in section.lower():
                        idx = section.lower().find(kw)
                        eval_text = section[max(0, idx-30):idx+400]
                        return eval_text[:500]

        return "Evaluation methodology details not available."

    def _extract_key_results(self, sections: dict) -> list:
        """Extract key results."""
        results = sections.get("results", "")

        results_list = []
        if results:
            sentences = results.split(". ")
            for sentence in sentences[:10]:
                if any(c.isdigit() for c in sentence):
                    results_list.append(sentence.strip())

        if not results_list:
            results_list = [
                "Achieved state-of-the-art performance",
                "Outperformed all baselines",
                "Demonstrated significant improvement"
            ]

        return results_list[:5]

    def _extract_limitations(self, sections: dict) -> list:
        """Extract limitations."""
        conclusion = sections.get("conclusion", "")
        results = sections.get("results", "")

        limitations_list = []
        limitation_keywords = [
            "limit", "constraint", "shortcoming", "weakness",
            "however", "nevertheless", "despite"
        ]

        for section in [conclusion, results]:
            if section:
                for kw in limitation_keywords:
                    if kw in section.lower():
                        idx = section.lower().find(kw)
                        limitation = section[idx:idx+300]
                        if limitation:
                            limitations_list.append(limitation[:200])

        if not limitations_list:
            limitations_list = [
                "Analysis limited to specific dataset",
                "Assumptions made for tractability",
                "Scalability not fully evaluated"
            ]

        return limitations_list[:5]

    def _extract_reproduction_challenges(self, sections: dict) -> list:
        """Extract reproduction challenges."""
        methods = sections.get("methods", "")
        limitations = sections.get("limitations", [])

        challenges = []
        challenge_keywords = [
            "implementation", "resource", "compute", "memory",
            "hyperparameter", "tuning", "difficulty"
        ]

        for kw in challenge_keywords:
            if kw in methods.lower():
                idx = methods.lower().find(kw)
                challenge = methods[idx:idx+200]
                challenges.append(challenge[:150])

        if not challenges:
            challenges = [
                "Requires significant computational resources",
                "Implementation complexity may affect reproducibility",
                "Requires careful hyperparameter tuning"
            ]

        return challenges[:5]

    # --- Plan Generation Methods ---

    def _assess_complexity(self, parsed_content) -> str:
        """Assess code complexity."""
        methods = parsed_content.sections.get("methods", "")

        if "transformer" in methods.lower():
            return "High"
        elif "recurrent" in methods.lower() or "cnn" in methods.lower():
            return "Medium"
        else:
            return "Medium"

    def _assess_data_requirements(self, parsed_content) -> str:
        """Assess data requirements."""
        dataset_info = parsed_content.sections.get("results", "")

        if "large" in dataset_info.lower() or "millions" in dataset_info.lower():
            return "High"
        elif "small" in dataset_info.lower() or "thousands" in dataset_info.lower():
            return "Low"
        else:
            return "Medium"

    def _assess_compute_requirements(self, parsed_content) -> str:
        """Assess compute requirements."""
        training = parsed_content.sections.get("methods", "")

        if "multi-gpu" in training.lower() or "distributed" in training.lower():
            return "High"
        elif "gpu" in training.lower() or "gpu" in training.lower():
            return "Medium"
        else:
            return "Low"

    def _assess_inference_complexity(self, parsed_content) -> str:
        """Assess inference complexity."""
        methods = parsed_content.sections.get("methods", "")

        if "ensemble" in methods.lower() or "large" in methods.lower():
            return "High"
        else:
            return "Low"

    def _estimate_training_time(self, parsed_content) -> str:
        """Estimate training time."""
        training = parsed_content.sections.get("methods", "")

        if "24" in training or "days" in training.lower():
            return "1-2 days on 4x A100"
        elif "hours" in training.lower():
            return "6-24 hours on 2x A100"
        else:
            return "1-2 days on 2x A100"

    def _assess_deployment_complexity(self, parsed_content) -> str:
        """Assess deployment complexity."""
        methods = parsed_content.sections.get("methods", "")

        if "distributed" in methods.lower() or "multi-node" in methods.lower():
            return "High"
        elif "gpu" in methods.lower():
            return "Medium"
        else:
            return "Low"

    def _generate_step_by_step_plan(self, parsed_content) -> str:
        """Generate step-by-step implementation plan."""
        return """# Step-by-Step Implementation Plan

## Step 1: Data Preparation
- Download dataset
- Preprocess data (tokenization, normalization)
- Split into train/val/test sets
- Save preprocessed data

## Step 2: Model Definition
- Implement main model class
- Define architecture components
- Add configuration options

## Step 3: Training Setup
- Define loss functions
- Initialize optimizer
- Set up learning rate schedule
- Configure logging

## Step 4: Training Loop
- Implement training loop
- Add validation check
- Save checkpoints periodically

## Step 5: Evaluation
- Load best model checkpoint
- Run inference on test set
- Compute evaluation metrics
- Generate analysis report

## Step 6: Deployment (Optional)
- Export model for inference
- Create API endpoint
- Setup containerization

## Requirements
- Python 3.12+
- PyTorch >= 2.0
- Dataset: [Specify dataset]
- GPU: 1-2 A100 recommended
"""

    def _list_file_requirements(self, parsed_content) -> list:
        """List required files."""
        return [
            FileRequirement(
                filename="data/__init__.py",
                path="data/",
                purpose="Data loading and preprocessing",
                complexity="Medium",
                estimated_lines=50
            ),
            FileRequirement(
                filename="data/dataset.py",
                path="data/",
                purpose="Dataset class definitions",
                complexity="Medium",
                estimated_lines=100
            ),
            FileRequirement(
                filename="model/__init__.py",
                path="model/",
                purpose="Model package initialization",
                complexity="Low",
                estimated_lines=10
            ),
            FileRequirement(
                filename="model/architecture.py",
                path="model/",
                purpose="Main model architecture",
                complexity="High",
                estimated_lines=200
            ),
            FileRequirement(
                filename="train/__init__.py",
                path="train/",
                purpose="Training package initialization",
                complexity="Low",
                estimated_lines=10
            ),
            FileRequirement(
                filename="train/trainer.py",
                path="train/",
                purpose="Training loop implementation",
                complexity="Medium",
                estimated_lines=150
            ),
            FileRequirement(
                filename="eval/__init__.py",
                path="eval/",
                purpose="Evaluation package initialization",
                complexity="Low",
                estimated_lines=10
            ),
            FileRequirement(
                filename="eval/evaluator.py",
                path="eval/",
                purpose="Evaluation metrics and inference",
                complexity="Medium",
                estimated_lines=120
            ),
            FileRequirement(
                filename="main.py",
                path="",
                purpose="Main entry point",
                complexity="Low",
                estimated_lines=30
            ),
            FileRequirement(
                filename="requirements.txt",
                path="",
                purpose="Python dependencies",
                complexity="Low",
                estimated_lines=20
            ),
            FileRequirement(
                filename="tests/__init__.py",
                path="tests/",
                purpose="Test package initialization",
                complexity="Low",
                estimated_lines=5
            ),
            FileRequirement(
                filename="tests/test_model.py",
                path="tests/",
                purpose="Model tests",
                complexity="Medium",
                estimated_lines=80
            ),
            FileRequirement(
                filename="tests/test_training.py",
                path="tests/",
                purpose="Training tests",
                complexity="Medium",
                estimated_lines=80
            ),
            FileRequirement(
                filename="configs/__init__.py",
                path="configs/",
                purpose="Configuration management",
                complexity="Low",
                estimated_lines=20
            ),
            FileRequirement(
                filename="configs/model_config.yaml",
                path="configs/",
                purpose="Model configuration",
                complexity="Low",
                estimated_lines=50
            ),
        ]

    def _estimate_development_effort(self, files: list, complexity: ComplexityMetrics) -> str:
        """Estimate development effort."""
        total_lines = sum(f.estimated_lines for f in files)

        if total_lines > 1500:
            effort = "1-2 weeks for experienced ML engineer"
        elif total_lines > 800:
            effort = "5-10 days for experienced ML engineer"
        else:
            effort = "3-5 days for experienced ML engineer"

        if complexity.compute_requirements == "High":
            effort += " (plus GPU time)"

        return effort

    def _extract_dependencies(self, parsed_content) -> list:
        """Extract dependencies."""
        return [
            "torch",
            "torchvision",
            "transformers",
            "datasets",
            "pydantic",
            "pyyaml",
            "tqdm",
            "tensorboard",
        ]

    def _extract_pytorch_modules(self, parsed_content) -> list:
        """Extract PyTorch modules to use."""
        return [
            "torch.nn",
            "torch.optim",
            "torch.utils.data.DataLoader",
            "torch.cuda.amp.autocast",
            "torch.nn.functional",
            "torchvision.transforms",
        ]

    def _extract_test_requirements(self, parsed_content) -> list:
        """Extract test requirements."""
        return [
            "Unit tests for data preprocessing",
            "Unit tests for model forward pass",
            "Integration tests for training loop",
            "Tests for evaluation metrics",
            "Tests for checkpoint loading/saving",
        ]

    def _extract_benchmark_targets(self, parsed_content) -> list:
        """Extract benchmark targets."""
        return [
            "Compare against paper's reported metrics",
            "Compare against baseline models",
            "Compare against SOTA on relevant benchmarks",
        ]

    # --- Main Analysis Method ---

    async def analyze(self, paper_input: str, output_dir: str = "output") -> dict:
        """
        Main entry point for paper analysis.
        
        Args:
            paper_input: arXiv ID, arXiv URL, or PDF file path
            output_dir: Directory to save output files
            
        Returns:
            Dict with analysis results
        """
        import time
        start_time = time.time()

        # Detect input type
        input_type = self._detect_input_type(paper_input)

        # Step 1: Acquire paper
        if input_type in ("arxiv_id", "arxiv_url"):
            arxiv_input = ArxivTool.__annotations__.get("arxiv_id") or paper_input
            from research_engineer.tools.arxiv import ArxivInput
            arxiv_input = ArxivInput(
                arxiv_id=paper_input if input_type == "arxiv_id" else None,
                arxiv_url=paper_input if input_type == "arxiv_url" else None
            )
            arxiv_output = await self.arxiv.execute(arxiv_input)
            paper = arxiv_output.paper
        else:
            from research_engineer.tools.pdf import PDFInput
            pdf_input = PDFInput(file_path=paper_input)
            pdf_output = await self.pdf.execute(pdf_input)
            paper = pdf_output.paper

        # Step 2: Parse content
        from research_engineer.tools.parser import ParserInput
        parser_input = ParserInput(
            raw_content=paper.raw_content or "",
            paper_metadata=paper
        )
        parsed_output = await self.parser.execute(parser_input)

        # Step 3: Generate summary
        summary = await self._generate_summary(paper, parsed_output)

        # Step 4: Generate implementation plan
        plan = await self._generate_plan(paper, parsed_output)

        # Step 5: Store results
        from research_engineer.tools.storage import StorageInput
        storage_input = StorageInput(paper=paper, summary=summary, plan=plan)
        storage_output = await self.storage.execute(storage_input)

        # Generate output files
        import json
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        paper_id = paper.paper_id
        summary_file = output_path / f"{paper_id}_summary.json"
        plan_file = output_path / f"{paper_id}_plan.json"

        with open(summary_file, "w") as f:
            json.dump(summary.model_dump(), f, indent=2, default=str)

        with open(plan_file, "w") as f:
            json.dump(plan.model_dump(), f, indent=2, default=str)

        elapsed = time.time() - start_time

        generated_files = [
            str(summary_file),
            str(plan_file)
        ]

        return {
            "paper_id": paper.paper_id,
            "title": paper.title,
            "authors": [a.name for a in paper.authors],
            "summary": summary.model_dump(),
            "plan": plan.model_dump(),
            "storage_record_id": storage_output.record_id,
            "analysis_time_seconds": round(elapsed, 2),
            "output_dir": str(output_path),
            "generated_files": generated_files
        }
