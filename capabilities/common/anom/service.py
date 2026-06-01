"""Service layer for APG Anomaly Detection."""

from __future__ import annotations

from typing import Any

from .anomaly_engine import AnomalyDetectionEngine
from .capability_contract import evaluate_capability_rules, get_capability_contract
from .models import (
	AnomLifecycleBatchRecord,
	AnomalyAgentRecord,
	AnomalyAuditEvent,
	AnomalySignal,
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
		self._agents: dict[tuple[str, str], AnomalyAgentRecord] = {}
		self._lifecycle_batches: dict[tuple[str, str], AnomLifecycleBatchRecord] = {}
		self._events: list[AnomalyAuditEvent] = []
		self._engine = AnomalyDetectionEngine()
		contract = get_capability_contract()
		self._agent_runtimes = set(contract["agents"]["supported_runtimes"])
		self._agent_roles = set(contract["agents"]["supported_roles"])
		self._privileged_agent_roles = set(contract["agents"]["privileged_roles"])
		self._lifecycle_operations = set(contract["streaming"]["required_operations"])

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_source(
		self,
		source_id: str,
		tenant_id: str,
		name: str,
		kind: str = "",
		owner: str = "",
		labels: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		self._enforce_tenant(tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_source",
			"source_name_present": bool(name),
			"source_owner_present": bool(owner),
			"source_kind_present": bool(kind),
			"source_kind_known": kind in self.describe(tenant_id)["configuration"]["sources"]["allowed_kinds"],
		})
		_raise_if_denied(result)
		status = "pending_review" if result["decision"] == "require_review" else "active"
		source = MonitoringSource(
			id=source_id,
			tenant_id=tenant_id,
			name=name,
			kind=kind,
			owner=owner,
			labels=dict(labels or {}),
			status=status,
			decision=result["decision"],
			matched_rules=tuple(result["matched_rules"]),
			review_reasons=_review_reasons(result),
		)
		self._sources[self._tenant_key(tenant_id, source_id)] = source
		self._record_event(
			tenant_id=tenant_id,
			event_type="monitoring_source_registered",
			subject_id=source_id,
			message=f"Registered monitoring source {name}.",
			evidence={
				"kind": kind,
				"owner": owner,
				"decision": result["decision"],
				"matched_rules": list(result["matched_rules"]),
				"reasons": list(_review_reasons(result)),
			},
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
		sensitivity: str = "",
	) -> dict[str, Any]:
		source_present = self._tenant_key(tenant_id, source_id) in self._sources
		result = self._enforce_baseline_policy(
			tenant_id=tenant_id,
			source_present=source_present,
			metric_present=bool(metric),
			history_points=len(values),
			sensitivity=sensitivity,
		)
		source = self._get_source(tenant_id, source_id)
		status = "pending_review" if result["decision"] == "require_review" else "active"
		baseline = self._engine.build_baseline(
			baseline_id=baseline_id,
			tenant_id=tenant_id,
			source_id=source_id,
			metric=metric,
			values=values,
			sensitivity=sensitivity,
			status=status,
			decision=result["decision"],
			matched_rules=tuple(result["matched_rules"]),
			review_reasons=_review_reasons(result),
		)
		self._baselines[self._tenant_key(tenant_id, baseline_id)] = baseline
		self._record_event(
			tenant_id=tenant_id,
			event_type="baseline_created",
			subject_id=baseline_id,
			message=f"Created baseline {baseline_id}.",
			evidence={
				"source_id": source.id,
				"metric": metric,
				"history_points": len(values),
				"decision": result["decision"],
				"matched_rules": list(result["matched_rules"]),
				"reasons": list(_review_reasons(result)),
			},
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
		triage_recorded: bool = False,
	) -> dict[str, Any]:
		value_present = value is not None
		self._enforce_detection_policy(
			tenant_id=tenant_id,
			source_present=self._tenant_key(tenant_id, source_id) in self._sources,
			baseline_present=self._tenant_key(tenant_id, baseline_id) in self._baselines,
			metric_present=bool(metric),
			value_present=value_present,
		)
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
		scored = self._engine.score_observation(baseline, observation)
		severity = str(scored["severity"])
		result = self._enforce_signal_policy(tenant_id, severity, bool(owner), triage_recorded=bool(triage_recorded))
		self._observations[self._tenant_key(tenant_id, observation.id)] = observation
		if result["decision"] == "require_review":
			status = "pending_review"
		else:
			status = "open" if scored["anomalous"] else "normal"
		signal = AnomalySignal(
			id=detection_id,
			tenant_id=tenant_id,
			source_id=source_id,
			baseline_id=baseline_id,
			observation_id=observation.id,
			score=float(scored["score"]),
			severity=severity,
			status=status,
			root_cause_hints=tuple(scored["root_cause_hints"]),
			decision=result["decision"],
			matched_rules=tuple(result["matched_rules"]),
			review_reasons=_review_reasons(result),
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
				"triage_recorded": bool(triage_recorded),
				"decision": result["decision"],
				"matched_rules": list(result["matched_rules"]),
				"reasons": list(_review_reasons(result)),
			},
		)
		if owner and scored["anomalous"] and status != "pending_review":
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
		result = self._enforce_signal_policy(
			tenant_id,
			str(metadata.get("severity") or "medium"),
			bool(metadata.get("owner")),
			triage_recorded=bool(metadata.get("triage_recorded", False)),
		)
		record_status = "pending_review" if result["decision"] == "require_review" else status
		signal = AnomalySignal(
			id=record_id,
			tenant_id=tenant_id,
			source_id=str(metadata.get("source_id") or "manual"),
			baseline_id=str(metadata.get("baseline_id") or "manual"),
			observation_id=str(metadata.get("observation_id") or "manual"),
			score=float(metadata.get("score", 0.0)),
			severity=str(metadata.get("severity") or "medium"),
			status=record_status,
			root_cause_hints=tuple(metadata.get("root_cause_hints") or ()),
			decision=result["decision"],
			matched_rules=tuple(result["matched_rules"]),
			review_reasons=_review_reasons(result),
		)
		self._signals[self._tenant_key(tenant_id, record_id)] = signal
		self._record_event(
			tenant_id=tenant_id,
			event_type="manual_signal_recorded",
			subject_id=record_id,
			message=f"Recorded manual anomaly signal {record_id}.",
			evidence={
				"severity": signal.severity,
				"status": record_status,
				"decision": result["decision"],
				"matched_rules": list(result["matched_rules"]),
				"reasons": list(_review_reasons(result)),
			},
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
		signal_present = self._tenant_key(tenant_id, signal_id) in self._signals
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "open_investigation",
			"signal_present": signal_present,
			"owner_assigned": bool(owner),
		})
		_raise_if_blocked(result)
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
		evidence = tuple(resolution_evidence or ())
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "close_investigation",
			"resolution_present": bool(resolution),
			"closed_by_present": bool(closed_by),
			"resolution_evidence_present": bool(evidence),
		})
		_raise_if_blocked(result)
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
		signal_present = self._tenant_key(tenant_id, signal_id) in self._signals
		projected_feedback = [item.to_dict() for item in self._feedback.values() if item.tenant_id == tenant_id]
		projected_feedback.append({"label": label})
		false_positive_rate = self._engine.false_positive_rate(projected_feedback)
		result = self._enforce_feedback_policy(
			tenant_id=tenant_id,
			signal_present=signal_present,
			label=label,
			reviewer=reviewer,
			false_positive_rate=false_positive_rate,
			tuning_review_recorded=tuning_review_recorded,
		)
		signal = self._get_signal(tenant_id, signal_id)
		feedback = DetectionFeedback(
			id=feedback_id,
			tenant_id=tenant_id,
			signal_id=signal_id,
			label=label,
			reviewer=reviewer,
			notes=notes,
			status="pending_review" if result["decision"] == "require_review" else "recorded",
			decision=result["decision"],
			matched_rules=tuple(result["matched_rules"]),
			review_reasons=_review_reasons(result),
		)
		self._feedback[self._tenant_key(tenant_id, feedback_id)] = feedback
		self._record_event(
			tenant_id=tenant_id,
			event_type="feedback_recorded",
			subject_id=feedback_id,
			message=f"Recorded {label} feedback for {signal.id}.",
			evidence={
				"signal_id": signal.id,
				"reviewer": reviewer,
				"false_positive_rate": false_positive_rate,
				"decision": result["decision"],
				"matched_rules": list(result["matched_rules"]),
				"reasons": list(_review_reasons(result)),
			},
		)
		return feedback.to_dict()

	def list_feedback(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		feedback = list(self._feedback.values())
		if tenant_id is not None:
			feedback = [item for item in feedback if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(feedback, key=lambda item: item.id)]

	def register_anomaly_agent(
		self,
		agent_id: str,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
		owner: str,
		purpose: str,
		contribution_disclosed: bool = True,
		human_approval_required: bool = False,
	) -> dict[str, Any]:
		runtime_value = _normalize_token(runtime)
		role_value = _normalize_token(role)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_anomaly_agent",
			"agent_runtime_supported": runtime_value in self._agent_runtimes,
			"agent_role_supported": role_value in self._agent_roles,
			"scope_present": bool(str(scope or "").strip()),
			"owner_present": bool(str(owner or "").strip()),
			"purpose_present": bool(str(purpose or "").strip()),
			"contribution_disclosed": bool(contribution_disclosed),
			"privileged_role": role_value in self._privileged_agent_roles,
			"human_approval_required": bool(human_approval_required),
		})
		_raise_if_denied(result)
		if not name:
			raise ValueError("anomaly_agent_name_required")
		status = "pending_review" if result["decision"] == "require_review" else "active"
		agent = AnomalyAgentRecord(
			id=agent_id,
			tenant_id=tenant_id,
			name=name,
			runtime=runtime_value,
			role=role_value,
			scope=str(scope).strip(),
			owner=str(owner).strip(),
			purpose=str(purpose).strip(),
			contribution_disclosed=bool(contribution_disclosed),
			human_approval_required=bool(human_approval_required),
			status=status,
		)
		self._agents[self._tenant_key(tenant_id, agent.id)] = agent
		self._record_event(
			tenant_id=tenant_id,
			event_type="anomaly_agent_registered",
			subject_id=agent.id,
			message=f"Registered anomaly agent {name}.",
			evidence={
				"runtime": runtime_value,
				"role": role_value,
				"decision": result["decision"],
				"reasons": list(_reasons(result)),
			},
		)
		return agent.to_dict()

	def validate_anom_lifecycle_batch(
		self,
		tenant_id: str,
		event_stream: str,
		mutation_count: int,
		operation: str = "anomaly_agent_batch",
		batch_id: str | None = None,
	) -> dict[str, Any]:
		mutation_count = int(mutation_count)
		if mutation_count <= 0:
			raise ValueError("anom_lifecycle_batch_empty")
		stream_value = _normalize_token(event_stream)
		operation_value = _normalize_token(operation)
		if operation_value not in self._lifecycle_operations:
			raise ValueError(f"unsupported_anom_lifecycle_operation:{operation_value}")
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "validate_anom_lifecycle_batch",
			"event_stream": stream_value,
		})
		accepted = result["decision"] == "allow"
		batch = AnomLifecycleBatchRecord(
			id=batch_id or f"anombatch:{len(self._lifecycle_batches) + 1:06d}",
			tenant_id=tenant_id,
			event_stream=stream_value,
			mutation_count=mutation_count,
			operation=operation_value,
			accepted=accepted,
			decision=result["decision"],
			matched_rules=tuple(result["matched_rules"]),
			status="accepted" if accepted else "denied",
		)
		self._lifecycle_batches[self._tenant_key(tenant_id, batch.id)] = batch
		self._record_event(
			tenant_id=tenant_id,
			event_type=f"anom_lifecycle_batch_{batch.status}",
			subject_id=batch.id,
			message=f"Validated ANOM lifecycle batch {batch.id}.",
			evidence={
				"event_stream": stream_value,
				"operation": operation_value,
				"decision": result["decision"],
				"reasons": list(_reasons(result)),
			},
		)
		if not accepted:
			_raise_if_denied(result)
		return batch.to_dict()

	def list_anomaly_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		agents = list(self._agents.values())
		if tenant_id is not None:
			agents = [item for item in agents if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(agents, key=lambda item: item.id)]

	def list_lifecycle_batches(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		batches = list(self._lifecycle_batches.values())
		if tenant_id is not None:
			batches = [item for item in batches if item.tenant_id == tenant_id]
		return [item.to_dict() for item in sorted(batches, key=lambda item: item.id)]

	def signal_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		signals = self.list_signals(tenant_id)
		feedback = self.list_feedback(tenant_id)
		sources = self.list_sources(tenant_id)
		baselines = self.list_baselines(tenant_id)
		return {
			"tenant_id": tenant_id,
			"source_count": len(sources),
			"baseline_count": len(baselines),
			"signal_count": len(signals),
			"investigation_count": len(self.list_investigations(tenant_id)),
			"feedback_count": len(feedback),
			"pending_source_review_count": len([item for item in sources if item["status"] == "pending_review"]),
			"pending_baseline_review_count": len([item for item in baselines if item["status"] == "pending_review"]),
			"pending_signal_review_count": len([item for item in signals if item["status"] == "pending_review"]),
			"pending_feedback_review_count": len([item for item in feedback if item["status"] == "pending_review"]),
			"anomaly_agent_count": len(self.list_anomaly_agents(tenant_id)),
			"pending_agent_review_count": len([item for item in self.list_anomaly_agents(tenant_id) if item["status"] == "pending_review"]),
			"lifecycle_batch_count": len(self.list_lifecycle_batches(tenant_id)),
			"denied_lifecycle_batch_count": len([item for item in self.list_lifecycle_batches(tenant_id) if item["status"] == "denied"]),
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

	def _enforce_detection_policy(
		self,
		tenant_id: str,
		source_present: bool,
		baseline_present: bool,
		metric_present: bool,
		value_present: bool,
	) -> None:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "detect",
			"monitoring_source_present": source_present,
			"baseline_present": baseline_present,
			"metric_present": metric_present,
			"value_present": value_present,
		})
		_raise_if_blocked(result)

	def _enforce_baseline_policy(
		self,
		tenant_id: str,
		source_present: bool,
		metric_present: bool,
		history_points: int,
		sensitivity: str,
	) -> dict[str, Any]:
		allowed = self.describe(tenant_id)["configuration"]["detection"]["allowed_sensitivities"]
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "create_baseline",
			"monitoring_source_present": source_present,
			"metric_present": metric_present,
			"history_points": history_points,
			"sensitivity_present": bool(sensitivity),
			"sensitivity_known": sensitivity in allowed,
		})
		_raise_if_denied(result)
		return result

	def _enforce_signal_policy(
		self,
		tenant_id: str,
		severity: str,
		owner_assigned: bool,
		triage_recorded: bool,
	) -> dict[str, Any]:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "detect",
			"severity": severity,
			"owner_assigned": owner_assigned,
			"triage_recorded": triage_recorded,
		})
		_raise_if_denied(result)
		return result

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
		signal_present: bool,
		label: str,
		reviewer: str,
		false_positive_rate: float,
		tuning_review_recorded: bool,
	) -> dict[str, Any]:
		allowed = self.describe(tenant_id)["configuration"]["feedback"]["allowed_labels"]
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "record_feedback",
			"signal_present": signal_present,
			"reviewer_present": bool(reviewer),
			"label_present": bool(label),
			"label_known": label in allowed,
			"false_positive_rate": false_positive_rate,
			"tuning_review_recorded": tuning_review_recorded,
		})
		_raise_if_denied(result)
		return result


def _raise_if_blocked(result: dict[str, Any]) -> None:
	if result["decision"] == "allow":
		return
	reasons = ", ".join(_reasons(result))
	if result["decision"] == "require_review":
		raise PermissionError(reasons or "anomaly_review_required")
	raise PermissionError(reasons or "anomaly_policy_blocked")


def _raise_if_denied(result: dict[str, Any]) -> None:
	if result["decision"] == "deny":
		raise PermissionError(", ".join(_reasons(result)) or "anomaly_policy_blocked")


def _review_reasons(result: dict[str, Any]) -> tuple[str, ...]:
	return tuple(
		action.get("reason", "anomaly_review_required")
		for action in result["actions"]
		if action.get("decision") == "require_review"
	)


def _reasons(result: dict[str, Any]) -> tuple[str, ...]:
	return tuple(action.get("reason", "anomaly_policy_blocked") for action in result["actions"])


def _normalize_token(value: str) -> str:
	return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
