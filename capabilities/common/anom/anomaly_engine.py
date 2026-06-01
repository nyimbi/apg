"""Deterministic anomaly scoring helpers for ANOM."""

from __future__ import annotations

from statistics import mean, pstdev
from typing import Any

from .models import BaselineProfile, Observation


MINIMUM_STDEV = 0.0001
SENSITIVITY_THRESHOLDS = {
	"low": 3.5,
	"medium": 2.5,
	"high": 1.8,
}


class AnomalyDetectionEngine:
	"""Build baselines, score observations, and summarize signal quality."""

	def build_baseline(
		self,
		baseline_id: str,
		tenant_id: str,
		source_id: str,
		metric: str,
		values: list[float] | tuple[float, ...],
		sensitivity: str = "medium",
		status: str = "active",
		decision: str = "allow",
		matched_rules: tuple[str, ...] = (),
		review_reasons: tuple[str, ...] = (),
	) -> BaselineProfile:
		center = mean(values)
		spread = max(pstdev(values), MINIMUM_STDEV)
		return BaselineProfile(
			id=baseline_id,
			tenant_id=tenant_id,
			source_id=source_id,
			metric=metric,
			mean=float(center),
			stdev=float(spread),
			history_points=len(values),
			sensitivity=sensitivity,
			status=status,
			decision=decision,
			matched_rules=matched_rules,
			review_reasons=review_reasons,
		)

	def score_observation(
		self,
		baseline: BaselineProfile,
		observation: Observation,
	) -> dict[str, Any]:
		score = abs(observation.value - baseline.mean) / baseline.stdev
		threshold = SENSITIVITY_THRESHOLDS.get(baseline.sensitivity, SENSITIVITY_THRESHOLDS["medium"])
		if score >= max(threshold + 1.5, 4.0):
			severity = "critical"
		elif score >= max(threshold + 0.75, 3.0):
			severity = "high"
		elif score >= threshold:
			severity = "medium"
		else:
			severity = "normal"
		return {
			"score": round(score, 4),
			"severity": severity,
			"anomalous": severity != "normal",
			"threshold": threshold,
			"root_cause_hints": self.root_cause_hints(baseline, observation, score),
		}

	def root_cause_hints(
		self,
		baseline: BaselineProfile,
		observation: Observation,
		score: float,
	) -> tuple[str, ...]:
		direction = "above" if observation.value > baseline.mean else "below"
		hints = [
			f"{observation.metric} is {direction} baseline mean",
			f"score {score:.2f} against {baseline.sensitivity} sensitivity",
		]
		if observation.context.get("deployment"):
			hints.append("recent deployment present in observation context")
		if observation.context.get("region"):
			hints.append(f"region={observation.context['region']}")
		return tuple(hints)

	def summarize_signals(self, signals: list[dict[str, Any]]) -> dict[str, Any]:
		by_severity = {"critical": 0, "high": 0, "medium": 0, "normal": 0}
		for signal in signals:
			severity = str(signal.get("severity") or "normal")
			by_severity[severity] = by_severity.get(severity, 0) + 1
		return {
			"signal_count": len(signals),
			"by_severity": by_severity,
			"critical_or_high_count": by_severity.get("critical", 0) + by_severity.get("high", 0),
		}

	def false_positive_rate(self, feedback: list[dict[str, Any]]) -> float:
		if not feedback:
			return 0.0
		false_positive_count = sum(1 for item in feedback if item.get("label") == "false_positive")
		return false_positive_count / len(feedback)
