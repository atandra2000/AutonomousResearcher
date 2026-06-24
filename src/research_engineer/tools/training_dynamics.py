"""Training Dynamics Tool for Phase 8.

Analyzes metric time-series to detect overfitting, underfitting,
convergence/plateau, instability, and divergence. Rule-based, no LLM.
"""

from __future__ import annotations

import math

from research_engineer.models.evaluation import (
    DynamicsPattern,
    DynamicsPatternType,
    TrainingDynamicsInput,
    TrainingDynamicsOutput,
)
from research_engineer.models.experiment import (
    MetricSeries,
)
from research_engineer.tools.base import Tool, ToolError


class TrainingDynamicsTool(Tool[TrainingDynamicsInput, TrainingDynamicsOutput]):
    """Detect training dynamics patterns from metric series."""

    def __init__(self) -> None:
        pass

    async def validate(self, input: TrainingDynamicsInput) -> bool:
        return input.experiment is not None

    async def execute(
        self, input: TrainingDynamicsInput
    ) -> TrainingDynamicsOutput:
        try:
            exp = input.experiment
            series = input.metric_series or exp.metric_series
            train_series = self._find_series(series, input.train_metric)
            eval_series = self._find_series(series, input.eval_metric)

            final_train = self._final_value(train_series, exp, input.train_metric)
            final_eval = self._final_value(eval_series, exp, input.eval_metric)
            best_eval, best_eval_step = self._best_eval(
                eval_series, exp, input.eval_metric
            )
            gap = self._gap(final_train, final_eval)

            patterns: list[DynamicsPattern] = []
            patterns.append(
                self._check_overfitting(
                    train_series, eval_series, gap, final_train, input
                )
            )
            patterns.append(
                self._check_underfitting(
                    train_series, eval_series, final_train, input
                )
            )
            patterns.append(
                self._check_convergence(train_series, input)
            )
            patterns.append(
                self._check_instability(train_series, input)
            )
            patterns.append(
                self._check_divergence(train_series, input)
            )
            patterns.append(self._check_healthy(patterns, self._is_decreasing(train_series)))

            convergence_step = self._convergence_step(train_series, input)
            stability = self._stability_score(train_series)

            detected = [p for p in patterns if p.detected]
            if detected:
                top = max(detected, key=lambda p: p.confidence)
                summary = (
                    f"Primary pattern: {top.pattern_type.value} "
                    f"(confidence {top.confidence:.2f}). "
                    f"{top.recommendation}"
                )
            else:
                summary = "No notable dynamics patterns detected."

            return TrainingDynamicsOutput(
                experiment_id=exp.experiment_id,
                patterns=patterns,
                convergence_step=convergence_step,
                final_train_metric=final_train,
                final_eval_metric=final_eval,
                best_eval_metric=best_eval,
                best_eval_step=best_eval_step,
                train_eval_gap=gap,
                stability_score=stability,
                summary=summary,
            )
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"Dynamics analysis failed: {e}", input, e)

    @staticmethod
    def _find_series(
        series: list[MetricSeries], name: str
    ) -> MetricSeries | None:
        for s in series:
            if s.name == name:
                return s
        lower = name.lower()
        for s in series:
            if s.name.lower() == lower:
                return s
        return None

    @staticmethod
    def _final_value(
        s: MetricSeries | None,
        exp: object,
        metric: str,
    ) -> float | None:
        if s and s.values:
            return s.final_value if s.final_value is not None else s.values[-1]
        metrics = getattr(exp, "metrics", {}) or {}
        if metric in metrics:
            return float(metrics[metric])
        for key, val in metrics.items():
            if key.lower() == metric.lower():
                return float(val)
        return None

    @staticmethod
    def _best_eval(
        s: MetricSeries | None,
        exp: object,
        metric: str,
    ) -> tuple[float | None, int | None]:
        if s and s.values:
            return s.best_value, s.best_step
        metrics = getattr(exp, "metrics", {}) or {}
        if metric in metrics:
            return float(metrics[metric]), None
        return None, None

    @staticmethod
    def _gap(train: float | None, eval: float | None) -> float | None:
        if train is None or eval is None:
            return None
        return eval - train

    def _check_overfitting(
        self,
        train: MetricSeries | None,
        eval_s: MetricSeries | None,
        gap: float | None,
        final_train: float | None,
        input: TrainingDynamicsInput,
    ) -> DynamicsPattern:
        if gap is None or final_train is None:
            return DynamicsPattern(
                pattern_type=DynamicsPatternType.OVERFITTING,
                detected=False,
                evidence=["missing train/eval metrics"],
                recommendation="",
            )
        eps = 1e-9
        gap_ratio = abs(gap) / max(abs(final_train), eps)
        eval_increasing = self._is_increasing(eval_s)
        train_decreasing = self._is_decreasing(train)
        detected = (
            gap_ratio > input.overfit_threshold
            and (eval_increasing or train_decreasing)
        )
        confidence = min(gap_ratio, 1.0)
        evidence = [
            f"train_eval_gap={gap:.4f}",
            f"gap_ratio={gap_ratio:.3f}",
            f"eval_increasing={eval_increasing}",
            f"train_decreasing={train_decreasing}",
        ]
        rec = (
            "Add regularization (dropout/weight decay), "
            "data augmentation, or early stopping."
            if detected
            else ""
        )
        return DynamicsPattern(
            pattern_type=DynamicsPatternType.OVERFITTING,
            detected=detected,
            confidence=confidence if detected else 0.0,
            evidence=evidence,
            recommendation=rec,
        )

    def _check_underfitting(
        self,
        train: MetricSeries | None,
        eval_s: MetricSeries | None,
        final_train: float | None,
        input: TrainingDynamicsInput,
    ) -> DynamicsPattern:
        if final_train is None or not train or not train.values:
            return DynamicsPattern(
                pattern_type=DynamicsPatternType.UNDERFITTING,
                detected=False,
                evidence=["missing train metric"],
                recommendation="",
            )
        first = train.values[0]
        best = self._best_value(train)
        eps = 1e-9
        train_decreasing = self._is_decreasing(train)
        # Underfitting: loss barely decreased from start and not improving.
        barely_decreased = (
            first > 0
            and abs(final_train) > 0.8 * abs(first)
        )
        not_improving = not train_decreasing
        detected = bool(barely_decreased and not_improving)
        confidence = min(abs(final_train) / max(abs(first or 1.0), eps), 1.0)
        evidence = [
            f"final_train={final_train:.4f}",
            f"first_train={first:.4f}",
            f"best_train={best if best is not None else 'n/a'}",
            f"train_decreasing={train_decreasing}",
            f"barely_decreased={barely_decreased}",
        ]
        rec = (
            "Increase model capacity, train longer, or raise learning rate."
            if detected
            else ""
        )
        return DynamicsPattern(
            pattern_type=DynamicsPatternType.UNDERFITTING,
            detected=detected,
            confidence=confidence if detected else 0.0,
            evidence=evidence,
            recommendation=rec,
        )

    def _check_convergence(
        self, train: MetricSeries | None, input: TrainingDynamicsInput
    ) -> DynamicsPattern:
        if not train or len(train.values) < input.convergence_window:
            return DynamicsPattern(
                pattern_type=DynamicsPatternType.CONVERGENCE,
                detected=False,
                evidence=["insufficient data for plateau check"],
                recommendation="",
            )
        window = train.values[-input.convergence_window:]
        slope = self._slope(window)
        mean_val = sum(abs(v) for v in window) / len(window)
        eps = 1e-9
        detected = abs(slope) < input.convergence_tolerance * max(
            mean_val, eps
        )
        evidence = [
            f"slope={slope:.6f}",
            f"window_mean_abs={mean_val:.4f}",
            f"window_size={len(window)}",
        ]
        rec = (
            "Training has plateaued; try LR decay/cosine schedule or "
            "increase capacity if underfit."
            if detected
            else ""
        )
        return DynamicsPattern(
            pattern_type=DynamicsPatternType.CONVERGENCE,
            detected=detected,
            confidence=0.8 if detected else 0.0,
            evidence=evidence,
            recommendation=rec,
        )

    def _check_instability(
        self, train: MetricSeries | None, input: TrainingDynamicsInput
    ) -> DynamicsPattern:
        if not train or len(train.values) < 3:
            return DynamicsPattern(
                pattern_type=DynamicsPatternType.INSTABILITY,
                detected=False,
                evidence=["insufficient data for instability check"],
                recommendation="",
            )
        values = train.values
        mean_v = sum(values) / len(values)
        eps = 1e-9
        std_v = math.sqrt(
            sum((v - mean_v) ** 2 for v in values) / max(len(values) - 1, 1)
        )
        cv = std_v / max(abs(mean_v), eps)
        swings = self._has_large_swings(values)
        detected = cv > input.instability_threshold or swings
        evidence = [
            f"coeff_of_variation={cv:.3f}",
            f"mean={mean_v:.4f}",
            f"std={std_v:.4f}",
            f"large_swings={swings}",
        ]
        rec = (
            "Reduce batch size variance, use gradient accumulation, "
            "or lower the learning rate."
            if detected
            else ""
        )
        return DynamicsPattern(
            pattern_type=DynamicsPatternType.INSTABILITY,
            detected=detected,
            confidence=min(cv, 1.0) if detected else 0.0,
            evidence=evidence,
            recommendation=rec,
        )

    def _check_divergence(
        self, train: MetricSeries | None, input: TrainingDynamicsInput
    ) -> DynamicsPattern:
        if not train or len(train.values) < 4:
            return DynamicsPattern(
                pattern_type=DynamicsPatternType.DIVERGENCE,
                detected=False,
                evidence=["insufficient data for divergence check"],
                recommendation="",
            )
        values = train.values
        first = values[0]
        last = values[-1]
        eps = 1e-9
        increasing = last > first * 1.5 and first > 0
        detected = bool(increasing or last > abs(first) * 1.5)
        evidence = [
            f"first={first:.4f}",
            f"last={last:.4f}",
            f"ratio={last / max(abs(first), eps):.3f}",
        ]
        rec = (
            "Lower the learning rate, add gradient clipping, "
            "and check the LR schedule."
            if detected
            else ""
        )
        return DynamicsPattern(
            pattern_type=DynamicsPatternType.DIVERGENCE,
            detected=detected,
            confidence=0.9 if detected else 0.0,
            evidence=evidence,
            recommendation=rec,
        )

    @staticmethod
    def _check_healthy(
        patterns: list[DynamicsPattern],
        train_decreasing: bool,
    ) -> DynamicsPattern:
        bad = {
            DynamicsPatternType.OVERFITTING,
            DynamicsPatternType.UNDERFITTING,
            DynamicsPatternType.INSTABILITY,
            DynamicsPatternType.DIVERGENCE,
        }
        any_bad = any(
            p.detected and p.pattern_type in bad for p in patterns
        )
        converged = any(
            p.detected and p.pattern_type == DynamicsPatternType.CONVERGENCE
            for p in patterns
        )
        detected = (not any_bad) and (converged or train_decreasing)
        return DynamicsPattern(
            pattern_type=DynamicsPatternType.HEALTHY,
            detected=detected,
            confidence=0.8 if detected else 0.0,
            evidence=(
                ["no over/underfit, stable, decreasing loss"]
                if detected
                else []
            ),
            recommendation=(
                "Healthy run; consider scaling up or moving to next "
                "paper technique."
                if detected
                else ""
            ),
        )

    @staticmethod
    def _is_increasing(s: MetricSeries | None) -> bool:
        if not s or len(s.values) < 4:
            return False
        first_half = s.values[: len(s.values) // 2]
        second_half = s.values[len(s.values) // 2:]
        if not first_half or not second_half:
            return False
        avg1 = sum(first_half) / len(first_half)
        avg2 = sum(second_half) / len(second_half)
        return avg2 > avg1 * 1.05

    @staticmethod
    def _is_decreasing(s: MetricSeries | None) -> bool:
        if not s or len(s.values) < 4:
            return False
        first_half = s.values[: len(s.values) // 2]
        second_half = s.values[len(s.values) // 2:]
        if not first_half or not second_half:
            return False
        avg1 = sum(first_half) / len(first_half)
        avg2 = sum(second_half) / len(second_half)
        return avg2 < avg1 * 0.95

    @staticmethod
    def _best_value(s: MetricSeries | None) -> float | None:
        if not s or not s.values:
            return None
        return min(s.values)

    @staticmethod
    def _slope(window: list[float]) -> float:
        n = len(window)
        if n < 2:
            return 0.0
        xs = list(range(n))
        mean_x = sum(xs) / n
        mean_y = sum(window) / n
        num = sum(
            (x - mean_x) * (y - mean_y)
            for x, y in zip(xs, window)
        )
        denom = sum((x - mean_x) ** 2 for x in xs)
        if denom == 0:
            return 0.0
        return num / denom

    @staticmethod
    def _has_large_swings(values: list[float]) -> bool:
        if len(values) < 4:
            return False
        rng = max(values) - min(values)
        if rng == 0:
            return False
        swings = 0
        for i in range(1, len(values)):
            delta = abs(values[i] - values[i - 1])
            if delta > 0.5 * rng:
                swings += 1
        return swings >= 2

    def _convergence_step(
        self, train: MetricSeries | None, input: TrainingDynamicsInput
    ) -> int | None:
        if not train or len(train.values) < input.convergence_window:
            return None
        steps = train.steps
        if not steps or len(steps) < len(train.values):
            return None
        window = input.convergence_window
        return steps[len(steps) - window]

    @staticmethod
    def _stability_score(train: MetricSeries | None) -> float:
        if not train or len(train.values) < 3:
            return 1.0
        values = train.values
        mean_v = sum(values) / len(values)
        eps = 1e-9
        std_v = math.sqrt(
            sum((v - mean_v) ** 2 for v in values)
            / max(len(values) - 1, 1)
        )
        cv = std_v / max(abs(mean_v), eps)
        return max(0.0, min(1.0, 1.0 / (1.0 + cv)))
