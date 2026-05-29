"""Service layer for APG Anomaly Detection."""

from __future__ import annotations

from typing import Any

from .anomaly_engine import AnomalyDetectionEngine
from .capability_contract import evaluate_capability_rules, get_capability_contract
from .models import (
	AnomalySignal,
	AnomalyAuditEvent,
	BaselineProfile,
	DetectionFeedback,
	Investigation,
	MonitoringSource,
	Observation,
)


class AnomService:
	"""Monitoring source registry, baseline manager, detector, and investigation queue."""

	def __init__(self) -> None:
		self._sources: dict[tuple[str, str], MonitoringSource] = {}
		self._baselines: dict[tuple[str, str], BaselineProfile] = {}
		self._observations: dict[tuple[str, str], Observation] = {}
		self._signals: dict[tuple[str, str], AnomalySignal] = {}
		self._investigations: dict[tuple[str, str], Investigation] = {}
		self._feedback: dict[tuple[str, str], DetectionFeedback] = {}
		self._events: list[AnomalyAuditEvent] = []
		self._engine = AnomalyDetectionEngine()

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_source(
		self,
		source_id: str,
		tenant_id: str,
		name: str,
		kind: str = "metric",
		owner: str = "operations",
		labels: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		self._enforce_tenant(tenant_id)
		source = MonitoringSource(
			id=source_id,
			tenant_id=tenant_id,
			name=name,
			kind=kind,
			owner=owner,
			labels=dict(labels or {}),
		)
		self._sources[self._tenant_key(tenant_id, source_id)] = source
		self._record_event(
			tenant_id=tenant_id,
			event_type="monitoring_source_registered",
			subject_id=source_id,
			message=f"Registered monitoring source {name}.",
			evidence={"kind": kind, "owner": owner},
		)
		return source.to_dict()

	def list_sources(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		sources = list(self._sources.values())
		if tenant_id is not None:
			sources = [item for item in sources if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(sources, key=lambda item: item.id)]

	def create_baseline(
		self,
		baseline_id: str,
		tenant_id: str,
		source_id: str,
		metric: str,
		values: list[float] | tuple[float, ...],
		sensitivity: str = "medium",
	) -> dict[str, Any]:
		self._enforce_baseline_policy(tenant_id, len(values))
		source = self._get_source(tenant_id, source_id)
		baseline = self._engine.build_baseline(
			baseline_id=baseline_id,
			tenant_id=tenant_id,
			source_id=source_id,
			metric=metric,
			values=values,
			sensitivity=sensitivity,
		)
		self._baselines[self._tenant_key(tenant_id, baseline_id)] = baseline
		self._record_event(
			tenant_id=tenant_id,
			event_type="baseline_created",
			subject_id=baseline_id,
			message=f"Created baseline {baseline_id}.",
			evidence={"source_id": source.id, "metric": metric, "history_points": len(values)},
		)
		return baseline.to_dict()

	def list_baselines(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		baselines = list(self._baselines.values())
		if tenant_id is not None:
			baselines = [item for item in baselines if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(baselines, key=lambda item: item.id)]

	def reset_baseline(
		self,
		baseline_id: str,
		values: list[float] | tuple[float, ...],
		approval_recorded: bool,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		if not tenant_id:
			self._enforce_tenant("")
		baseline = self._resolve_baseline(baseline_id, tenant_id)
		self._enforce_reset_policy(baseline.tenant_id, approval_recorded)
		updated = self.create_baseline(
			baseline_id=baseline_id,
			tenant_id=baseline.tenant_id,
			source_id=baseline.source_id,
			metric=baseline.metric,
			values=values,
			sensitivity=baseline.sensitivity,
		)
		self._record_event(
			tenant_id=baseline.tenant_id,
			event_type="baseline_reset",
			subject_id=baseline_id,
			message=f"Reset baseline {baseline_id}.",
			evidence={"approval_recorded": approval_recorded, "history_points": len(values)},
		)
		return updated

	def detect(
		self,
		detection_id: str,
		tenant_id: str,
		source_id: str,
		baseline_id: str,
		metric: str,
		value: float,
		timestamp: str | None = None,
		context: dict[str, Any] | None = None,
		owner: str | None = None,
	) -> dict[str, Any]:
		self._enforce_detection_policy(tenant_id, self._tenant_key(tenant_id, source_id) in self._sources)
		source = self._get_source(tenant_id, source_id)
		baseline = self._get_baseline(tenant_id, baseline_id)
		observation = Observation(
			id=f"obs:{detection_id}",
			tenant_id=tenant_id,
			source_id=source_id,
			metric=metric,
			value=float(value),
			timestamp=timestamp,
			context=dict(context or {}),
		)
		self._observations[self._tenant_key(tenant_id, observation.id)] = observation
		scored = self._engine.score_observation(baseline, observation)
		severity = str(scored["severity"])
		self._enforce_signal_policy(tenant_id, severity, bool(owner))
		signal = AnomalySignal(
			id=detection_id,
			tenant_id=tenant_id,
			source_id=source_id,
			baseline_id=baseline_id,
			observation_id=observation.id,
			score=float(scored["score"]),
			severity=severity,
			status="open" if scored["anomalous"] else "normal",
			root_cause_hints=tuple(scored["root_cause_hints"]),
		)
		self._signals[self._tenant_key(tenant_id, detection_id)] = signal
		self._record_event(
			tenant_id=tenant_id,
			event_type="signal_detected",
			subject_id=detection_id,
			message=f"Detected {severity} anomaly signal {detection_id}.",
			evidence={
				"source_id": source.id,
				"baseline_id": baseline.id,
				"metric": metric,
				"score": scored["score"],
				"owner_assigned": bool(owner),
			},
		)
		if owner and scored["anomalous"]:
			self.open_investigation(
				investigation_id=f"investigate:{detection_id}",
				tenant_id=tenant_id,
				signal_id=detection_id,
				owner=owner,
			)
		return {**signal.to_dict(), "observation": observation.to_dict(), "threshold": scored["threshold"]}

	def list_signals(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		signals = list(self._signals.values())
		if tenant_id is not None:
			signals = [item for item in signals if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(signals, key=lambda item: item.id)]

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""Compatibility alias exposing anomaly signals as ANOM records."""
		records = list(self._signals.values())
		if tenant_id is not None:
			records = [record for record in records if record.tenant_id == tenant_id]
		return [record.to_dict() for record in sorted(records, key=lambda item: item.id)]

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		"""Compatibility helper that records a manual anomaly signal."""
		metadata = dict(metadata or {})
		self._enforce_signal_policy(tenant_id, str(metadata.get("severity") or "medium"), bool(metadata.get("owner")))
		signal = AnomalySignal(
			id=record_id,
			tenant_id=tenant_id,
			source_id=str(metadata.get("source_id") or "manual"),
			baseline_id=str(metadata.get("baseline_id") or "manual"),
			observation_id=str(metadata.get("observation_id") or "manual"),
			score=float(metadata.get("score", 0.0)),
			severity=str(metadata.get("severity") or "medium"),
			status=status,
			root_cause_hints=tuple(metadata.get("root_cause_hints") or ()),
		)
		self._signals[self._tenant_key(tenant_id, record_id)] = signal
		self._record_event(
			tenant_id=tenant_id,
			event_type="manual_signal_recorded",
			subject_id=record_id,
			message=f"Recorded manual anomaly signal {record_id}.",
			evidence={"severity": signal.severity, "status": status},
		)
		return signal.to_dict()

	def open_investigation(
		self,
		investigation_id: str,
		tenant_id: str,
		signal_id: str,
		owner: str,
	) -> dict[str, Any]:
		self._enforce_tenant(tenant_id)
		if not owner:
			raise ValueError("investigation owner is required")
		signal = self._get_signal(tenant_id, signal_id)
		investigation = Investigation(
			id=investigation_id,
			tenant_id=tenant_id,
			signal_id=signal_id,
			owner=owner,
		)
		self._investigations[self._tenant_key(tenant_id, investigation_id)] = investigation
		self._record_event(
			tenant_id=tenant_id,
			event_type="investigation_opened",
			subject_id=investigation_id,
			message=f"Opened investigation {investigation_id}.",
			evidence={"signal_id": signal.id, "owner": owner},
		)
		return investigation.to_dict()

	def close_investigation(
		self,
		investigation_id: str,
		resolution: str,
		tenant_id: str | None = None,
		closed_by: str | None = None,
		resolution_evidence: list[str] | tuple[str, ...] | None = None,
	) -> dict[str, Any]:
		if not tenant_id:
			self._enforce_tenant("")
		investigation = self._resolve_investigation(investigation_id, tenant_id)
		self._enforce_tenant(tenant_id)
		if not resolution:
			raise ValueError("investigation resolution is required")
		if not closed_by:
			raise ValueError("investigation closer is required")
		evidence = tuple(resolution_evidence or ())
		if not evidence:
			raise ValueError("investigation resolution evidence is required")
		updated = Investigation(
			id=investigation.id,
			tenant_id=investigation.tenant_id,
			signal_id=investigation.signal_id,
			owner=investigation.owner,
			status="closed",
			resolution=resolution,
			closed_by=closed_by,
			resolution_evidence=evidence,
		)
		self._investigations[self._tenant_key(investigation.tenant_id, investigation_id)] = updated
		self._record_event(
			tenant_id=investigation.tenant_id,
			event_type="investigation_closed",
			subject_id=investigation_id,
			message=f"Closed investigation {investigation_id}.",
			evidence={"closed_by": closed_by, "resolution": resolution, "evidence_count": len(evidence)},
		)
		return updated.to_dict()

	def list_investigations(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		investigations = list(self._investigations.values())
		if tenant_id is not None:
			investigations = [item for item in investigations if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(investigations, key=lambda item: item.id)]

	def record_feedback(
		self,
		feedback_id: str,
		tenant_id: str,
		signal_id: str,
		label: str,
		reviewer: str,
		notes: str = "",
		tuning_review_recorded: bool = False,
	) -> dict[str, Any]:
		signal = self._get_signal(tenant_id, signal_id)
		if not reviewer:
			raise ValueError("feedback reviewer is required")
		projected_feedback = [item.to_dict() for item in self._feedback.values() if item.tenant_id == tenant_id]
		projected_feedback.append({"label": label})
		false_positive_rate = self._engine.false_positive_rate(projected_feedback)
		self._enforce_feedback_policy(tenant_id, false_positive_rate, tuning_review_recorded)
		feedback = DetectionFeedback(
			id=feedback_id,
			tenant_id=tenant_id,
			signal_id=signal_id,
			label=label,
			reviewer=reviewer,
			notes=notes,
		)
		self._feedback[self._tenant_key(tenant_id, feedback_id)] = feedback
		self._record_event(
			tenant_id=tenant_id,
			event_type="feedback_recorded",
			subject_id=feedback_id,
			message=f"Recorded {label} feedback for {signal.id}.",
			evidence={"signal_id": signal.id, "reviewer": reviewer, "false_positive_rate": false_positive_rate},
		)
		return feedback.to_dict()

	def list_feedback(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		feedback = list(self._feedback.values())
		if tenant_id is not None:
			feedback = [item for item in feedback if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(feedback, key=lambda item: item.id)]

	def signal_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		signals = self.list_signals(tenant_id)
		feedback = self.list_feedback(tenant_id)
		return {
			"tenant_id": tenant_id,
			"source_count": len(self.list_sources(tenant_id)),
			"baseline_count": len(self.list_baselines(tenant_id)),
			"signal_count": len(signals),
			"investigation_count": len(self.list_investigations(tenant_id)),
			"feedback_count": len(feedback),
			"false_positive_rate": self._engine.false_positive_rate(feedback),
			**self._engine.summarize_signals(signals),
		}

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		events = list(self._events)
		if tenant_id is not None:
			events = [item for item in events if item.tenant_id == tenant_id]
		return [item.to_dict() for item in events]

	def _tenant_key(self, tenant_id: str, record_id: str) -> tuple[str, str]:
		return (tenant_id, record_id)

	def _get_source(self, tenant_id: str, source_id: str) -> MonitoringSource:
		source = self._sources.get(self._tenant_key(tenant_id, source_id))
		if source is None:
			raise KeyError(f"unknown monitoring source for tenant: {source_id}")
		return source

	def _get_baseline(self, tenant_id: str, baseline_id: str) -> BaselineProfile:
		baseline = self._baselines.get(self._tenant_key(tenant_id, baseline_id))
		if baseline is None:
			raise KeyError(f"unknown baseline for tenant: {baseline_id}")
		return baseline

	def _get_signal(self, tenant_id: str, signal_id: str) -> AnomalySignal:
		signal = self._signals.get(self._tenant_key(tenant_id, signal_id))
		if signal is None:
			raise KeyError(f"unknown anomaly signal: {signal_id}")
		return signal

	def _resolve_baseline(self, baseline_id: str, tenant_id: str | None) -> BaselineProfile:
		if tenant_id is not None:
			return self._get_baseline(tenant_id, baseline_id)
		matches = [item for item in self._baselines.values() if item.id == baseline_id]
		if not matches:
			raise KeyError(f"unknown baseline: {baseline_id}")
		if len(matches) > 1:
			raise ValueError("tenant_id is required for duplicate baseline IDs")
		return matches[0]

	def _resolve_investigation(self, investigation_id: str, tenant_id: str | None) -> Investigation:
		if tenant_id is not None:
			investigation = self._investigations.get(self._tenant_key(tenant_id, investigation_id))
			if investigation is None:
				raise KeyError(f"unknown investigation for tenant: {investigation_id}")
			return investigation
		matches = [item for item in self._investigations.values() if item.id == investigation_id]
		if not matches:
			raise KeyError(f"unknown investigation: {investigation_id}")
		if len(matches) > 1:
			raise ValueError("tenant_id is required for duplicate investigation IDs")
		return matches[0]

	def _record_event(
		self,
		tenant_id: str,
		event_type: str,
		subject_id: str,
		message: str,
		evidence: dict[str, Any] | None = None,
	) -> None:
		self._events.append(
			AnomalyAuditEvent(
				id=f"event:{len(self._events) + 1}",
				tenant_id=tenant_id,
				event_type=event_type,
				subject_id=subject_id,
				message=message,
				evidence=dict(evidence or {}),
			)
		)

	def _enforce_tenant(self, tenant_id: str) -> None:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
		})
		_raise_if_blocked(result)

	def _enforce_detection_policy(self, tenant_id: str, source_present: bool) -> None:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "detect",
			"monitoring_source_present": source_present,
		})
		_raise_if_blocked(result)

	def _enforce_baseline_policy(self, tenant_id: str, history_points: int) -> None:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "create_baseline",
			"history_points": history_points,
		})
		_raise_if_blocked(result)

	def _enforce_signal_policy(self, tenant_id: str, severity: str, owner_assigned: bool) -> None:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"severity": severity,
			"owner_assigned": owner_assigned,
		})
		_raise_if_blocked(result)

	def _enforce_reset_policy(self, tenant_id: str, approval_recorded: bool) -> None:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "reset_baseline",
			"approval_recorded": approval_recorded,
		})
		_raise_if_blocked(result)

	def _enforce_feedback_policy(
		self,
		tenant_id: str,
		false_positive_rate: float,
		tuning_review_recorded: bool,
	) -> None:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"false_positive_rate": false_positive_rate,
			"tuning_review_recorded": tuning_review_recorded,
		})
		_raise_if_blocked(result)


def _raise_if_blocked(result: dict[str, Any]) -> None:
	if result["decision"] == "allow":
		return
	reasons = ", ".join(action.get("reason", "anomaly_policy_blocked") for action in result["actions"])
	if result["decision"] == "require_review":
		raise PermissionError(reasons or "anomaly_review_required")
	raise PermissionError(reasons or "anomaly_policy_blocked")
