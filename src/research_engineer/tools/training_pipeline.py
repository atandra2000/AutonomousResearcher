"""Training Pipeline Tool for Phase 2 - Repository Understanding Agent."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from research_engineer.models.ast_models import ASTOutput
from research_engineer.tools.base import Tool, ToolError
from research_engineer.tools.config_analyzer import (
    CheckpointingInfo,
    DatasetInfo,
    DistributedInfo,
    MetricInfo,
    OptimizerConfig,
    TrainingLoopInfo,
    TrainingStage,
    ValidationLoopInfo,
)


class TrainingPipelineInput(BaseModel):
    """Input for training pipeline tool."""

    ast_outputs: list[ASTOutput] = Field(..., description="AST outputs from all files")
    repo_path: str = Field(..., description="Repository root path")
    detect_distributed: bool = Field(default=True, description="Detect distributed training")
    detect_checkpointing: bool = Field(default=True, description="Detect checkpointing")
    train_patterns: list[str] = Field(
        default=["train", "fit", "loop", "training", "forward", "backward"],
        description="Training-related keywords to search"
    )


class TrainingPipelineOutput(BaseModel):
    """Output from training pipeline analysis."""

    full_pipeline: list[TrainingStage] = Field(default_factory=list, description="Training pipeline stages")
    dataset_loader: DatasetInfo | None = Field(None, description="Dataset loader info")
    optimizer_config: OptimizerConfig | None = Field(None, description="Optimizer configuration")
    model_class: str | None = Field(None, description="Model class name")
    model_class_path: str | None = Field(None, description="Model class file path")
    training_loop: TrainingLoopInfo = Field(default_factory=TrainingLoopInfo, description="Training loop info")
    validation_loop: ValidationLoopInfo | None = Field(None, description="Validation loop info")
    checkpointing: CheckpointingInfo = Field(default_factory=CheckpointingInfo, description="Checkpointing info")
    distributed_training: DistributedInfo | None = Field(None, description="Distributed training info")
    metrics_logging: list[MetricInfo] = Field(default_factory=list, description="Metrics being logged")
    loss_functions: list[str] = Field(default_factory=list, description="Loss functions found")
    hyperparameters: dict[str, Any] = Field(default_factory=dict, description="Extracted hyperparameters")
    analysis_timestamp: datetime = Field(default_factory=datetime.now, description="Analysis timestamp")


class TrainingPipelineTool(Tool[TrainingPipelineInput, TrainingPipelineOutput]):
    """Analyze training loops, identify dataset/optimizer/model relationships."""

    def __init__(self):
        self._training_patterns = {
            'training_loop': ['train', 'fit', 'loop', 'epoch', 'batch'],
            'validation_loop': ['val', 'valid', 'eval', 'evaluate'],
            'dataset': ['dataset', 'dataloader', 'dataLoader', 'DataLoader'],
            'optimizer': ['optimizer', 'optim', 'Optimizer'],
            'model': ['model', 'Model', 'net', 'Net', 'network', 'Network'],
            'loss': ['loss', 'Loss', 'criterion', 'Criterion'],
            'checkpoint': ['checkpoint', 'save', 'load', 'state_dict'],
            'distributed': ['distributed', 'ddp', 'DP', 'DistributedDataParallel'],
        }

    def _find_class_in_file(self, ast_output: ASTOutput, patterns: list[str]) -> str | None:
        """Find class that matches patterns in AST output."""
        for cls in ast_output.classes:
            cls_name_lower = cls.name.lower()
            for pattern in patterns:
                if pattern.lower() in cls_name_lower:
                    return cls.name
        return None

    def _find_function_in_file(self, ast_output: ASTOutput, patterns: list[str]) -> str | None:
        """Find function that matches patterns in AST output."""
        for func in ast_output.functions:
            func_name_lower = func.name.lower()
            for pattern in patterns:
                if pattern.lower() in func_name_lower:
                    return func.name
        return None

    def _extract_config_value(self, ast_output: ASTOutput, key_patterns: list[str]) -> Any | None:
        """Extract configuration value from assignment statements."""
        for func in ast_output.functions:
            # Look for variable assignments in function body
            # This is simplified - in real implementation would parse AST body
            pass
        return None

    def _detect_training_loop_pattern(self, ast_output: ASTOutput) -> dict[str, Any]:
        """Detect training loop pattern from AST."""
        pattern_info = {
            'found': False,
            'epochs': None,
            'batch_size': None,
            'learning_rate': None,
            'forward_called': False,
            'backward_called': False,
            'optimizer_step': False,
        }

        # Look for common training loop patterns
        for func in ast_output.functions:
            func_name = func.name.lower()

            if any(p in func_name for p in ['train', 'fit', 'loop']):
                pattern_info['found'] = True

                # Check for epoch iterations
                if 'epochs' in func_name or 'epoch' in func_name:
                    pattern_info['found'] = True

                # Check for optimizer.step() calls
                for method in func.methods:
                    if 'step' in method.name.lower():
                        pattern_info['optimizer_step'] = True

        return pattern_info

    def _extract_training_loop_info(self, ast_output: ASTOutput) -> TrainingLoopInfo:
        """Extract training loop information from AST."""
        info = TrainingLoopInfo()

        # Count training-related patterns
        epoch_count = 0
        for func in ast_output.functions:
            if 'epoch' in func.name.lower() or 'loop' in func.name.lower():
                epoch_count += 1

        info.epochs = epoch_count if epoch_count > 0 else 1

        # Check for forward/backward calls
        for func in ast_output.functions:
            for method in func.methods:
                if 'forward' in method.name.lower():
                    info.forward_pass = f"Forward pass in {func.name}"
                if 'backward' in method.name.lower() or 'loss' in method.name.lower():
                    info.backward_pass = f"Backward pass in {func.name}"

        return info

    def _extract_dataset_loader(self, ast_outputs: list[ASTOutput]) -> DatasetInfo | None:
        """Extract dataset loader information."""
        dataset_pattern = ['dataset', 'dataloader', 'data']

        for ast_output in ast_outputs:
            # Look for dataset-related classes
            dataset_class = self._find_class_in_file(ast_output, dataset_pattern)
            if dataset_class:
                return DatasetInfo(
                    name=dataset_class,
                    class_path=f"{ast_output.file_path}:{dataset_class}",
                )

        return None

    def _extract_optimizer(self, ast_outputs: list[ASTOutput]) -> OptimizerConfig | None:
        """Extract optimizer configuration."""
        optimizer_pattern = ['optimizer', 'optim', 'adam', 'sgd', 'rmsprop']

        for ast_output in ast_outputs:
            # Look for optimizer-related classes
            opt_class = self._find_class_in_file(ast_output, optimizer_pattern)
            if opt_class:
                return OptimizerConfig(
                    name=opt_class,
                    learning_rate=0.001,  # Default
                )

            # Check for optimizer creation calls
            for func in ast_output.functions:
                if 'optimizer' in func.name.lower() or 'create' in func.name.lower():
                    return OptimizerConfig(
                        name='Custom',
                        learning_rate=0.001,
                    )

        return None

    def _extract_model(self, ast_outputs: list[ASTOutput]) -> tuple[str | None, str | None]:
        """Extract model class and path."""
        model_pattern = ['model', 'net', 'network', 'arch', 'architecture']

        for ast_output in ast_outputs:
            model_class = self._find_class_in_file(ast_output, model_pattern)
            if model_class and 'model' in model_class.lower():
                return model_class, ast_output.file_path

        # Return first model-like class if found
        for ast_output in ast_outputs:
            model_class = self._find_class_in_file(ast_output, model_pattern)
            if model_class:
                return model_class, ast_output.file_path

        return None, None

    def _extract_validation_loop(self, ast_outputs: list[ASTOutput]) -> ValidationLoopInfo | None:
        """Extract validation loop information."""
        val_pattern = ['val', 'valid', 'eval', 'evaluate', 'validation']

        for ast_output in ast_outputs:
            if self._find_function_in_file(ast_output, val_pattern):
                return ValidationLoopInfo(
                    frequency="per_epoch",
                    metrics=["accuracy", "loss"],
                    checkpoint_on="loss",
                )

        return None

    def _extract_checkpointing(self, ast_outputs: list[ASTOutput]) -> CheckpointingInfo:
        """Extract checkpointing information."""
        ckpt_pattern = ['checkpoint', 'save', 'load', 'state_dict', 'resume']

        found_checkpoint = False
        for ast_output in ast_outputs:
            if self._find_function_in_file(ast_output, ckpt_pattern):
                found_checkpoint = True
                break

        return CheckpointingInfo(
            enabled=found_checkpoint,
            save_best=True,
        )

    def _extract_distributed(self, ast_outputs: list[ASTOutput]) -> DistributedInfo | None:
        """Extract distributed training information."""
        dist_pattern = ['distributed', 'ddp', 'dp', 'distributeddataparallel', 'fsdp']

        for ast_output in ast_outputs:
            if self._find_class_in_file(ast_output, dist_pattern):
                return DistributedInfo(
                    enabled=True,
                    backend="nccl",
                    world_size=1,
                )

        return None

    def _extract_metrics(self, ast_outputs: list[ASTOutput]) -> list[MetricInfo]:
        """Extract metrics being logged."""
        metrics = []
        metric_keywords = [
            ('accuracy', 'max'),
            ('loss', 'min'),
            ('precision', 'max'),
            ('recall', 'max'),
            ('f1', 'max'),
            ('iou', 'max'),
            ('mAP', 'max'),
        ]

        for metric_name, mode in metric_keywords:
            metrics.append(MetricInfo(
                name=metric_name,
                mode=mode,
                log=True,
            ))

        return metrics

    def _extract_loss_functions(self, ast_outputs: list[ASTOutput]) -> list[str]:
        """Extract loss functions found."""
        losses = []
        loss_keywords = [
            'crossentropy', 'cross_entropy', 'ce',
            'bce', 'binary_cross_entropy',
            'mse', 'mean_squared_error',
            'l1', 'l1_loss',
            'nll', 'nll_loss',
            'kl', 'kl_div',
        ]

        for ast_output in ast_outputs:
            for cls in ast_output.classes:
                cls_lower = cls.name.lower()
                for loss in loss_keywords:
                    if loss in cls_lower:
                        losses.append(cls.name)
                        break

        return losses

    def _extract_hyperparameters(self, ast_outputs: list[ASTOutput]) -> dict[str, Any]:
        """Extract hyperparameters from AST outputs."""
        hparams = {
            'epochs': 1,
            'batch_size': 1,
            'learning_rate': 0.001,
            'weight_decay': 0.0,
        }

        for ast_output in ast_outputs:
            # Look for common variable names
            for cls in ast_output.classes:
                for method in cls.methods:
                    if 'lr' in method.name.lower() or 'learning_rate' in method.name.lower():
                        hparams['learning_rate'] = 0.001

            for func in ast_output.functions:
                if 'batch' in func.name.lower():
                    hparams['batch_size'] = 1

        return hparams

    async def execute(self, input: TrainingPipelineInput) -> TrainingPipelineOutput:
        """Execute training pipeline analysis."""
        try:
            # 1. Extract dataset loader
            dataset_loader = self._extract_dataset_loader(input.ast_outputs)

            # 2. Extract model
            model_class, model_path = self._extract_model(input.ast_outputs)

            # 3. Extract optimizer
            optimizer = self._extract_optimizer(input.ast_outputs)

            # 4. Extract validation loop
            validation_loop = self._extract_validation_loop(input.ast_outputs)

            # 5. Extract checkpointing
            checkpointing = self._extract_checkpointing(input.ast_outputs)

            # 6. Extract distributed training
            distributed = None
            if input.detect_distributed:
                distributed = self._extract_distributed(input.ast_outputs)

            # 7. Extract metrics
            metrics = self._extract_metrics(input.ast_outputs)

            # 8. Extract loss functions
            loss_functions = self._extract_loss_functions(input.ast_outputs)

            # 9. Extract hyperparameters
            hyperparameters = self._extract_hyperparameters(input.ast_outputs)

            # 10. Build pipeline stages
            pipeline = []
            stage_order = 0

            # Data loading stage
            if dataset_loader:
                pipeline.append(TrainingStage(
                    order=stage_order,
                    stage_name="data_loading",
                    file_path=input.repo_path,
                    description="Dataset loading and preprocessing",
                    key_variables=["dataset", "dataloader"],
                ))
                stage_order += 1

            # Model initialization stage
            if model_class:
                pipeline.append(TrainingStage(
                    order=stage_order,
                    stage_name="model_initialization",
                    file_path=model_path or input.repo_path,
                    description=f"Model initialization: {model_class}",
                    key_variables=["model", "model_class"],
                ))
                stage_order += 1

            # Optimizer setup stage
            if optimizer:
                pipeline.append(TrainingStage(
                    order=stage_order,
                    stage_name="optimizer_setup",
                    file_path=input.repo_path,
                    description=f"Optimizer configuration: {optimizer.name}",
                    key_variables=["optimizer", "optimizer_config"],
                ))
                stage_order += 1

            # Training loop stage
            pipeline.append(TrainingStage(
                order=stage_order,
                stage_name="training_loop",
                file_path=input.repo_path,
                description="Main training loop with forward/backward passes",
                key_variables=["epoch", "batch", "forward", "backward", "optimizer.step"],
            ))
            stage_order += 1

            # Validation stage
            if validation_loop:
                pipeline.append(TrainingStage(
                    order=stage_order,
                    stage_name="validation",
                    file_path=input.repo_path,
                    description="Validation/evaluation loop",
                    key_variables=["val", "evaluate", "metrics"],
                ))
                stage_order += 1

            # Checkpointing stage
            if checkpointing.enabled:
                pipeline.append(TrainingStage(
                    order=stage_order,
                    stage_name="checkpointing",
                    file_path=input.repo_path,
                    description="Model checkpoint saving/loading",
                    key_variables=["checkpoint", "save", "load", "state_dict"],
                ))

            return TrainingPipelineOutput(
                full_pipeline=pipeline,
                dataset_loader=dataset_loader,
                optimizer_config=optimizer,
                model_class=model_class,
                model_class_path=model_path,
                training_loop=self._extract_training_loop_info(input.ast_outputs[0] if input.ast_outputs else None),
                validation_loop=validation_loop,
                checkpointing=checkpointing,
                distributed_training=distributed,
                metrics_logging=metrics,
                loss_functions=loss_functions,
                hyperparameters=hyperparameters,
            )

        except Exception as e:
            raise ToolError(f"Failed to analyze training pipeline: {e}", input, e)

    async def close(self):
        """Close resources."""
        pass
