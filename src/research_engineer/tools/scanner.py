"""Repository scanner tool for Phase 2 - Repository Understanding Agent."""

import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from research_engineer.tools.base import Tool, ToolError


class RepoScanInput(BaseModel):
    """Input for repository scanner."""

    path: str = Field(..., description="Path to repository")
    max_depth: int = Field(default=3, description="Maximum directory depth to scan")
    include_patterns: list[str] = Field(
        default_factory=lambda: ["*.py", "*.yaml", "*.yml", "*.json", "*.toml", "*.md", "*.txt"],
        description="File patterns to include"
    )
    exclude_patterns: list[str] = Field(
        default_factory=lambda: ["__pycache__", ".git", ".venv", "dist", "build", ".egg-info"],
        description="Directory/file patterns to exclude"
    )


class RepoScanOutput(BaseModel):
    """Output from repository scanner."""

    root_path: str = Field(..., description="Root path of repository")
    repository_name: str = Field(..., description="Detected repository name")
    total_files: int = Field(..., description="Total number of files scanned")
    files_by_type: dict[str, list[str]] = Field(default_factory=dict, description="Files grouped by type")
    entry_points: list[str] = Field(default_factory=list, description="Detected entry points")
    project_structure: dict[str, Any] = Field(default_factory=dict, description="Recursive directory tree")
    language_distribution: dict[str, float] = Field(default_factory=dict, description="Language percentages")
    detected_dependencies: list[str] = Field(default_factory=list, description="Detected Python dependencies")
    detected_frameworks: list[str] = Field(default_factory=list, description="Detected ML frameworks")
    repository_type: str = Field(default="Unknown", description="Detected repository type")
    total_lines: int = Field(default=0, description="Total lines of code")


class RepositoryScannerTool(Tool[RepoScanInput, RepoScanOutput]):
    """Scan directory structure and analyze repository contents."""

    def __init__(self):
        self._exclude_patterns: list[re.Pattern] = []

    def _compile_exclude_patterns(self, patterns: list[str]) -> list[re.Pattern]:
        """Compile exclude patterns to regex."""
        compiled = []
        for pattern in patterns:
            compiled.append(re.compile(re.escape(pattern)))
        return compiled

    def _should_exclude(self, path: Path, exclude_patterns: list[re.Pattern]) -> bool:
        """Check if path should be excluded."""
        path_str = str(path)
        for pattern in exclude_patterns:
            if pattern.search(path_str):
                return True
        return False

    def _count_lines(self, file_path: Path) -> int:
        """Count lines in a file."""
        try:
            with open(file_path, encoding='utf-8', errors='ignore') as f:
                return len(f.readlines())
        except Exception:
            return 0

    def _get_file_extension(self, file_path: Path) -> str:
        """Get file extension."""
        return file_path.suffix

    def _is_python_file(self, file_path: Path) -> bool:
        """Check if file is a Python file."""
        return file_path.suffix == '.py'

    def _is_config_file(self, file_path: Path) -> bool:
        """Check if file is a configuration file."""
        config_extensions = {'.yaml', '.yml', '.json', '.toml'}
        return file_path.suffix in config_extensions

    def _is_markdown_file(self, file_path: Path) -> bool:
        """Check if file is a Markdown file."""
        return file_path.suffix == '.md'

    async def _find_entry_points(self, python_files: list[str]) -> list[str]:
        """Find likely entry points in Python files."""
        entry_points = []

        for file_path in python_files:
            file_lower = file_path.lower()

            # Check for common entry point names
            if any(name in file_lower for name in ['main', 'train', 'cli', 'app', 'run', 'entry']):
                entry_points.append(file_path)
                continue

            # Check for __main__ block
            try:
                with open(file_path, encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if 'if __name__' in content and '__main__' in content:
                        entry_points.append(file_path)
            except Exception:
                pass

        return list(set(entry_points))[:10]  # Limit to 10 entry points

    async def _detect_dependencies(self, python_files: list[str]) -> list[str]:
        """Detect Python dependencies from imports."""
        dependencies = set()

        # Common ML frameworks and libraries
        frameworks = {
            'torch': 'PyTorch',
            'tensorflow': 'TensorFlow',
            'jax': 'JAX',
            'flax': 'Flax',
            'transformers': 'Transformers',
            'datasets': 'Datasets',
            ' Accelerate': 'Accelerate',
            'peft': 'PEFT',
            'bitsandbytes': 'BitsAndBytes',
        }

        for file_path in python_files:
            try:
                with open(file_path, encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    for dep, framework in frameworks.items():
                        if f'import {dep}' in content or f'from {dep}' in content:
                            dependencies.add(framework)
            except Exception:
               pass

        return list(dependencies)

    async def _detect_frameworks(self, python_files: list[str]) -> list[str]:
        """Detect ML frameworks from code content."""
        frameworks = set()

        framework_patterns = {
            'PyTorch': ['torch.nn', 'torch.optim', 'torch.utils.data', 'F.relu', 'nn.Module'],
            'TensorFlow': ['tf.', 'keras', 'tf.nn'],
            'JAX': ['jax.', 'jax.numpy'],
            'Transformers': ['transformers.', 'AutoModel', 'AutoTokenizer'],
            'Datasets': ['datasets.', 'load_dataset', 'Dataset'],
            'Accelerate': ['accelerate', 'Accelerator'],
            'PEFT': ['peft.', 'PeftModel'],
            'BitsAndBytes': ['bitsandbytes', '4bit', '8bit'],
            'Lightning': ['pytorch_lightning', 'LightningModule'],
            'Ray': ['ray.train', 'ray.tune'],
        }

        for file_path in python_files:
            try:
                with open(file_path, encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    for framework, patterns in framework_patterns.items():
                        for pattern in patterns:
                            if pattern in content:
                                frameworks.add(framework)
                                break
            except Exception:
                pass

        return list(frameworks)

    async def _detect_repository_type(self, python_files: list[str], frameworks: list[str]) -> str:
        """Detect type of repository based on code analysis."""
        types = {
            'LLMTrainingFramework': ['transformers', 'AutoModel', 'AutoTokenizer', 'TrainingArguments'],
            'CVTrainingPipeline': ['torchvision', 'nn.Conv2d', 'nn.MaxPool2d', 'transforms'],
            'InferenceServer': ['FastAPI', 'Flask', 'uvicorn', 'gradio', 'streamlit'],
            'RAGSystem': ['RAG', 'Retriever', 'VectorStore', 'FAISS', 'Chroma'],
            'AgentFramework': ['Agent', 'Tool', 'ReAct', 'Chain'],
            'FineTuningSystem': ['peft', 'LoraConfig', 'peft_models'],
            'EvaluationFramework': ['eval', 'evaluate', 'metrics', 'benchmark'],
            'MultimodalSystem': ['multimodal', 'Vision', 'CLIP', 'Blip'],
            'DiffusionModel': ['diffusion', 'UNet', 'Denoise', 'StableDiffusion'],
            'ReinforcementLearning': ['RL', 'DQN', 'PPO', 'Agent'],
        }

        # Read content from all Python files
        combined_content = ""
        for file_path in python_files:
            try:
                with open(file_path, encoding='utf-8', errors='ignore') as f:
                    combined_content += f"\n\n--- {file_path} ---\n"
                    combined_content += f.read()
            except Exception:
                pass

        # Check for type indicators
        for repo_type, patterns in types.items():
            for pattern in patterns:
                if pattern in combined_content:
                    return repo_type

        return 'Unknown'

    async def _scan_directory_recursive(
        self,
        path: Path,
        depth: int,
        max_depth: int,
        include_patterns: list[str],
        exclude_patterns: list[re.Pattern],
        structure: dict[str, Any],
        files_by_type: dict[str, list[str]]
    ) -> None:
        """Recursively scan directory structure."""
        if depth > max_depth:
            return

        try:
            entries = list(path.iterdir())
        except PermissionError:
            return

        # Separate directories and files
        directories = [e for e in entries if e.is_dir() and not self._should_exclude(e, exclude_patterns)]
        files = [e for e in entries if e.is_file() and not self._should_exclude(e, exclude_patterns)]

        # Process files
        for file_path in files:
            extension = self._get_file_extension(file_path)
            # Check extension against include patterns (simple string comparison)
            for pattern in include_patterns:
                # pattern like "*.py" -> match ".py"
                if pattern.startswith("*") and extension == pattern[1:]:
                    files_by_type.setdefault(extension, []).append(str(file_path))

        # Process directories
        for dir_path in directories:
            dir_name = dir_path.name
            if dir_name not in structure:
                structure[dir_name] = {}
            await self._scan_directory_recursive(
                dir_path, depth + 1, max_depth, include_patterns, exclude_patterns,
                structure.get(dir_name, {}), files_by_type
            )

    async def execute(self, input: RepoScanInput) -> RepoScanOutput:
        """Execute repository scan."""
        try:
            path = Path(input.path)

            if not path.exists():
                raise ToolError(f"Path does not exist: {input.path}", input)

            if not path.is_dir():
                raise ToolError(f"Path is not a directory: {input.path}", input)

            # Compile exclude patterns
            exclude_patterns = self._compile_exclude_patterns(input.exclude_patterns)

            # Scan structure
            files_by_type: dict[str, list[str]] = {}
            structure: dict[str, Any] = {}

            await self._scan_directory_recursive(
                path, 0, input.max_depth,
                input.include_patterns, exclude_patterns,
                structure, files_by_type
            )

            # Collect all Python files
            python_files = files_by_type.get('.py', [])

            # Find entry points
            entry_points = await self._find_entry_points(python_files)

            # Detect dependencies
            dependencies = await self._detect_dependencies(python_files)

            # Detect frameworks
            frameworks = await self._detect_frameworks(python_files)

            # Detect repository type
            repo_type = await self._detect_repository_type(python_files, frameworks)

            # Count total lines
            total_lines = sum(self._count_lines(Path(f)) for f in python_files)

            # Calculate language distribution
            total_files = sum(len(files) for files in files_by_type.values())
            language_distribution = {}
            for ext, files in files_by_type.items():
                language_distribution[ext] = len(files) / total_files * 100 if total_files > 0 else 0

            return RepoScanOutput(
                root_path=str(path),
                repository_name=path.name,
                total_files=total_files,
                files_by_type=files_by_type,
                entry_points=entry_points,
                project_structure=structure,
                language_distribution=language_distribution,
                detected_dependencies=dependencies,
                detected_frameworks=frameworks,
                repository_type=repo_type,
                total_lines=total_lines
            )

        except Exception as e:
            raise ToolError(f"Failed to scan repository: {e}", input, e)

    async def close(self):
        """Close resources."""
        pass
