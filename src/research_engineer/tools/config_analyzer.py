"""Configuration Analysis Tool for Phase 2 - Repository Understanding Agent."""

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from research_engineer.tools.base import Tool, ToolError


class TrainingStage(BaseModel):
    """Represents a stage in the training pipeline."""

    order: int = Field(..., description="Order in pipeline")
    stage_name: str = Field(..., description="Stage name")
    file_path: str = Field(..., description="File path")
    line_range: list[int] = Field(default_factory=list, description="Line range [start, end]")
    description: str = Field(default="", description="Stage description")
    key_variables: list[str] = Field(default_factory=list, description="Key variables in stage")


class DatasetInfo(BaseModel):
    """Represents dataset configuration."""

    name: str = Field(default="", description="Dataset name")
    class_path: str = Field(default="", description="Dataset class path")
    batch_size: int = Field(default=1, description="Batch size")
    num_workers: int = Field(default=0, description="Number of workers")
    transforms: list[str] = Field(default_factory=list, description="Transforms applied")
    train_split: str = Field(default="", description="Train split path")
    val_split: str = Field(default="", description="Validation split path")
    test_split: str = Field(default="", description="Test split path")
    dataset_size: int = Field(default=0, description="Total dataset size")


class OptimizerConfig(BaseModel):
    """Represents optimizer configuration."""

    name: str = Field(default="", description="Optimizer name")
    learning_rate: float = Field(default=0.001, description="Learning rate")
    weight_decay: float | None = Field(default=None, description="Weight decay")
    betas: list[float] = Field(default_factory=lambda: [0.9, 0.999], description="Beta parameters")
    parameters: list[str] = Field(default_factory=list, description="Optimized parameters")


class TrainingLoopInfo(BaseModel):
    """Represents training loop configuration."""

    epochs: int = Field(default=1, description="Number of epochs")
    batch_size: int = Field(default=1, description="Batch size")
    forward_pass: str = Field(default="", description="Forward pass description")
    loss_computation: str = Field(default="", description="Loss computation")
    backward_pass: str = Field(default="", description="Backward pass")
    gradient_accumulation: int | None = Field(default=None, description="Gradient accumulation steps")


class ValidationLoopInfo(BaseModel):
    """Represents validation loop configuration."""

    frequency: str = Field(default="", description="Validation frequency")
    metrics: list[str] = Field(default_factory=list, description="Validation metrics")
    checkpoint_on: str = Field(default="", description="Metric for checkpointing")
    best_value: float = Field(default=float('-inf'), description="Best validation metric value")


class CheckpointingInfo(BaseModel):
    """Represents checkpointing configuration."""

    enabled: bool = Field(default=False, description="Whether checkpointing is enabled")
    frequency: str = Field(default="", description="Checkpoint frequency")
    save_dir: str = Field(default="", description="Save directory")
    save_best: bool = Field(default=True, description="Save best model")
    load_from: str | None = Field(default=None, description="Checkpoint to load from")


class DistributedInfo(BaseModel):
    """Represents distributed training configuration."""

    enabled: bool = Field(default=False, description="Whether distributed training is enabled")
    backend: str = Field(default="", description="Distributed backend (nccl, gloo, etc.)")
    world_size: int = Field(default=1, description="World size")
    rank: int = Field(default=0, description="Process rank")
    local_rank: int = Field(default=0, description="Local rank")


class MetricInfo(BaseModel):
    """Represents a training/validation metric."""

    name: str = Field(default="", description="Metric name")
    type: str = Field(default="", description="Metric type")
    mode: str = Field(default="min", description="Optimization mode (min/max)")
    log: bool = Field(default=True, description="Whether to log metric")


class ConfigInput(BaseModel):
    """Input for config analyzer tool."""

    config_paths: list[str] = Field(default_factory=list, description="Paths to config files")
    detect_framework: bool = Field(default=True, description="Detect config framework")
    include_python_configs: bool = Field(default=True, description="Parse Python config files")


class ConfigOutput(BaseModel):
    """Output from config analyzer tool."""

    all_configs: dict[str, dict[str, Any]] = Field(default_factory=dict, description="All configs by name")
    training_hyperparameters: dict[str, Any] = Field(default_factory=dict, description="Training hyperparams")
    model_hyperparameters: dict[str, Any] = Field(default_factory=dict, description="Model hyperparams")
    data_paths: dict[str, str] = Field(default_factory=dict, description="Data file paths")
    distributed_settings: dict[str, Any] = Field(default_factory=dict, description="Distributed settings")
    checkpoint_settings: dict[str, Any] = Field(default_factory=dict, description="Checkpoint settings")
    logging_config: dict[str, Any] = Field(default_factory=dict, description="Logging configuration")
    optimizer_config: dict[str, Any] = Field(default_factory=dict, description="Optimizer configuration")
    scheduler_config: dict[str, Any] = Field(default_factory=dict, description="Scheduler configuration")
    config_framework: str = Field(default="unknown", description="Config framework used")
    config_sources: dict[str, str] = Field(default_factory=dict, description="Config name to file mapping")
    analysis_timestamp: datetime = Field(default_factory=datetime.now, description="Analysis timestamp")


class ConfigAnalysisTool(Tool[ConfigInput, ConfigOutput]):
    """Parse configuration files (YAML, JSON, TOML, Python configs)."""

    def __init__(self):
        self._yaml_cache: dict[str, dict] = {}
        self._json_cache: dict[str, dict] = {}
        self._toml_cache: dict[str, dict] = {}

    def _is_yaml_file(self, file_path: str) -> bool:
        """Check if file is YAML."""
        return file_path.endswith(('.yaml', '.yml'))

    def _is_json_file(self, file_path: str) -> bool:
        """Check if file is JSON."""
        return file_path.endswith('.json')

    def _is_toml_file(self, file_path: str) -> bool:
        """Check if file is TOML."""
        return file_path.endswith('.toml')

    def _is_python_config(self, file_path: str) -> bool:
        """Check if file is a Python config (simple attribute assignment)."""
        return file_path.endswith('.py')

    def _parse_yaml(self, content: str, file_path: str) -> dict[str, Any]:
        """Parse YAML content."""
        import yaml
        try:
            return yaml.safe_load(content) or {}
        except Exception:
            return {}

    def _parse_json(self, content: str, file_path: str) -> dict[str, Any]:
        """Parse JSON content."""
        import json
        try:
            return json.loads(content)
        except Exception:
            return {}

    def _parse_toml(self, content: str, file_path: str) -> dict[str, Any]:
        """Parse TOML content."""
        import tomli
        try:
            return tomli.loads(content)
        except Exception:
            # Try toml if tomli not available
            try:
                import toml as toml_legacy
                return toml_legacy.loads(content)
            except Exception:
                return {}

    def _extract_config_name(self, file_path: str) -> str:
        """Extract config name from file path."""
        return Path(file_path).stem

    def _detect_framework(self, configs: dict[str, dict]) -> str:
        """Detect config framework used."""
        # Look for common framework indicators
        framework_indicators = {
            'Hydra': ['_target_', 'hydra', 'cfg'],
            'argparse': ['--', 'parser.add_argument', 'argparse'],
            'PyTorch Lightning': ['lightning', 'LightningModule'],
            'Transformers': ['TrainingArguments', 'HfArgumentParser'],
            'Custom': ['config', 'settings', 'args'],
        }

        # Check all configs for indicators
        for config_name, config in configs.items():
            for framework, indicators in framework_indicators.items():
                for indicator in indicators:
                    if isinstance(config, dict):
                        # Check keys and values
                        for key in config.keys():
                            if indicator in key:
                                return framework
                        # Check nested values
                        for value in config.values():
                            if isinstance(value, str) and indicator in value:
                                return framework

        return 'unknown'

    def _extract_training_hyperparams(self, config: dict[str, Any]) -> dict[str, Any]:
        """Extract training hyperparameters from config."""
        hp = {}
        training_keywords = [
            'epochs', 'batch_size', 'lr', 'learning_rate',
            'optimizer', 'loss', 'metrics', 'train'
        ]

        for key, value in config.items():
            key_lower = key.lower()
            if any(kw in key_lower for kw in training_keywords):
                hp[key] = value

        # Also check nested dicts
        for key, value in config.items():
            if isinstance(value, dict):
                nested = self._extract_training_hyperparams(value)
                for k, v in nested.items():
                    hp[f"{key}.{k}"] = v

        return hp

    def _extract_model_hyperparams(self, config: dict[str, Any]) -> dict[str, Any]:
        """Extract model hyperparameters from config."""
        hp = {}
        model_keywords = [
            'model', 'arch', 'layer', 'hidden', 'vocab',
            'embedding', 'dropout', 'attention'
        ]

        for key, value in config.items():
            key_lower = key.lower()
            if any(kw in key_lower for kw in model_keywords):
                hp[key] = value

        # Also check nested dicts
        for key, value in config.items():
            if isinstance(value, dict):
                nested = self._extract_model_hyperparams(value)
                for k, v in nested.items():
                    hp[f"{key}.{k}"] = v

        return hp

    def _extract_data_paths(self, config: dict[str, Any]) -> dict[str, str]:
        """Extract data paths from config."""
        paths = {}
        path_keywords = [
            'data', 'path', 'dir', 'file', 'dataset',
            'train', 'val', 'test', 'load'
        ]

        for key, value in config.items():
            key_lower = key.lower()
            if isinstance(value, str) and any(kw in key_lower for kw in path_keywords):
                if '://' in value or value.startswith('/') or value.startswith('./') or value.startswith('../'):
                    paths[key] = value
                elif any(kw in key_lower for kw in ['path', 'dir', 'file']):
                    paths[key] = value

        # Also check nested dicts
        for key, value in config.items():
            if isinstance(value, dict):
                nested = self._extract_data_paths(value)
                for k, v in nested.items():
                    paths[f"{key}.{k}"] = v

        return paths

    def _extract_distributed_settings(self, config: dict[str, Any]) -> dict[str, Any]:
        """Extract distributed training settings."""
        settings = {}
        dist_keywords = [
            'distributed', 'ddp', 'dp', 'gpu', 'device',
            'world_size', 'rank', 'backend', 'local_rank'
        ]

        for key, value in config.items():
            key_lower = key.lower()
            if any(kw in key_lower for kw in dist_keywords):
                settings[key] = value

        # Also check nested dicts
        for key, value in config.items():
            if isinstance(value, dict):
                nested = self._extract_distributed_settings(value)
                for k, v in nested.items():
                    settings[f"{key}.{k}"] = v

        return settings

    def _extract_checkpoint_settings(self, config: dict[str, Any]) -> dict[str, Any]:
        """Extract checkpoint settings."""
        settings = {}
        ckpt_keywords = [
            'checkpoint', 'save', 'load', 'resume', 'period',
            'every', 'interval', 'latest', 'best'
        ]

        for key, value in config.items():
            key_lower = key.lower()
            if any(kw in key_lower for kw in ckpt_keywords):
                settings[key] = value

        # Also check nested dicts
        for key, value in config.items():
            if isinstance(value, dict):
                nested = self._extract_checkpoint_settings(value)
                for k, v in nested.items():
                    settings[f"{key}.{k}"] = v

        return settings

    def _extract_optimizer_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Extract optimizer configuration."""
        config_dict = {}
        opt_keywords = [
            'optimizer', 'optimizer_type', 'opt_type',
            'Adam', 'SGD', 'RMSprop', 'learning_rate', 'lr',
            'weight_decay', 'betas', 'momentum'
        ]

        for key, value in config.items():
            key_lower = key.lower()
            if any(kw in key_lower for kw in opt_keywords):
                config_dict[key] = value

        # Also check nested dicts
        for key, value in config.items():
            if isinstance(value, dict):
                nested = self._extract_optimizer_config(value)
                for k, v in nested.items():
                    config_dict[f"{key}.{k}"] = v

        return config_dict

    def _extract_scheduler_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Extract scheduler configuration."""
        config_dict = {}
        sched_keywords = [
            'scheduler', 'schedule', 'lr_scheduler',
            'step', 'epoch', 'warmup', 'decay',
            'gamma', 'milestones', 'poly'
        ]

        for key, value in config.items():
            key_lower = key.lower()
            if any(kw in key_lower for kw in sched_keywords):
                config_dict[key] = value

        # Also check nested dicts
        for key, value in config.items():
            if isinstance(value, dict):
                nested = self._extract_scheduler_config(value)
                for k, v in nested.items():
                    config_dict[f"{key}.{k}"] = v

        return config_dict

    def _extract_logging_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Extract logging configuration."""
        config_dict = {}
        log_keywords = [
            'log', 'logger', 'verbose', 'print',
            'every', 'interval', 'step'
        ]

        for key, value in config.items():
            key_lower = key.lower()
            if any(kw in key_lower for kw in log_keywords):
                config_dict[key] = value

        # Also check nested dicts
        for key, value in config.items():
            if isinstance(value, dict):
                nested = self._extract_logging_config(value)
                for k, v in nested.items():
                    config_dict[f"{key}.{k}"] = v

        return config_dict

    async def execute(self, input: ConfigInput) -> ConfigOutput:
        """Execute configuration analysis."""
        try:
            all_configs: dict[str, dict[str, Any]] = {}
            config_sources: dict[str, str] = {}

            # Process each config file
            for config_path in input.config_paths:
                try:
                    path = Path(config_path)
                    if not path.exists():
                        continue

                    content = path.read_text()
                    config_name = self._extract_config_name(config_path)

                    # Parse based on file extension
                    if self._is_yaml_file(config_path):
                        parsed = self._parse_yaml(content, config_path)
                    elif self._is_json_file(config_path):
                        parsed = self._parse_json(content, config_path)
                    elif self._is_toml_file(config_path):
                        parsed = self._parse_toml(content, config_path)
                    elif input.include_python_configs and self._is_python_config(config_path):
                        parsed = self._parse_python_config(content, config_path)
                    else:
                        parsed = {}

                    if parsed:
                        all_configs[config_name] = parsed
                        config_sources[config_name] = config_path
                except Exception:
                    continue

            if not all_configs:
                # Return empty output if no configs found
                return ConfigOutput(
                    all_configs={},
                    training_hyperparameters={},
                    model_hyperparameters={},
                    data_paths={},
                    distributed_settings={},
                    checkpoint_settings={},
                    logging_config={},
                    optimizer_config={},
                    scheduler_config={},
                    config_framework='unknown',
                    config_sources={},
                )

            # Detect framework
            framework = self._detect_framework(all_configs)

            # Extract training hyperparameters from all configs
            training_hypers = {}
            for config_name, config in all_configs.items():
                training_hypers.update(self._extract_training_hyperparams(config))

            # Extract model hyperparameters
            model_hypers = {}
            for config_name, config in all_configs.items():
                model_hypers.update(self._extract_model_hyperparams(config))

            # Extract data paths
            data_paths = {}
            for config_name, config in all_configs.items():
                data_paths.update(self._extract_data_paths(config))

            # Extract distributed settings
            dist_settings = {}
            for config_name, config in all_configs.items():
                dist_settings.update(self._extract_distributed_settings(config))

            # Extract checkpoint settings
            ckpt_settings = {}
            for config_name, config in all_configs.items():
                ckpt_settings.update(self._extract_checkpoint_settings(config))

            # Extract optimizer config
            opt_config = {}
            for config_name, config in all_configs.items():
                opt_config.update(self._extract_optimizer_config(config))

            # Extract scheduler config
            sched_config = {}
            for config_name, config in all_configs.items():
                sched_config.update(self._extract_scheduler_config(config))

            # Extract logging config
            log_config = {}
            for config_name, config in all_configs.items():
                log_config.update(self._extract_logging_config(config))

            return ConfigOutput(
                all_configs=all_configs,
                training_hyperparameters=training_hypers,
                model_hyperparameters=model_hypers,
                data_paths=data_paths,
                distributed_settings=dist_settings,
                checkpoint_settings=ckpt_settings,
                logging_config=log_config,
                optimizer_config=opt_config,
                scheduler_config=sched_config,
                config_framework=framework,
                config_sources=config_sources,
            )

        except Exception as e:
            raise ToolError(f"Failed to analyze configs: {e}", input, e)

    async def close(self):
        """Close resources."""
        pass

    def _parse_python_config(self, content: str, file_path: str) -> dict[str, Any]:
        """Parse Python config file (simple attribute assignment)."""
        config = {}

        # Match pattern: variable_name = value
        pattern = r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+)'

        for line in content.split('\n'):
            match = re.match(pattern, line.strip())
            if match:
                var_name = match.group(1)
                value_str = match.group(2).strip()

                # Try to parse the value
                try:
                    # Try numeric
                    if value_str.replace('.', '').replace('-', '').isdigit():
                        if '.' in value_str:
                            config[var_name] = float(value_str)
                        else:
                            config[var_name] = int(value_str)
                    # Try boolean
                    elif value_str.lower() == 'true':
                        config[var_name] = True
                    elif value_str.lower() == 'false':
                        config[var_name] = False
                    # Try string (quoted)
                    elif (value_str.startswith('"') and value_str.endswith('"')) or \
                         (value_str.startswith("'") and value_str.endswith("'")):
                        config[var_name] = value_str[1:-1]
                    else:
                        # Keep as string
                        config[var_name] = value_str
                except Exception:
                    config[var_name] = value_str

        return config
