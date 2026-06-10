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


from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
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


	# ── new methods ─────────────────────────────────────────────────────────

	def time_series_anomaly(
		self,
		detection_id: str,
		tenant_id: str,
		source_id: str,
		baseline_id: str,
		metric: str,
		series: list[tuple[str, float]],
		owner: str | None = None,
	) -> list[dict[str, Any]]:
		"""Detect anomalies across a time series sequence and return per-point results."""
		results = []
		for ts, value in series:
			result = self.detect(
				detection_id=f"{detection_id}:{ts}",
				tenant_id=tenant_id,
				source_id=source_id,
				baseline_id=baseline_id,
				metric=metric,
				value=value,
				timestamp=ts,
				owner=owner,
			)
			results.append(result)
		return results

	def multivariate_anomaly(
		self,
		detection_id: str,
		tenant_id: str,
		source_id: str,
		baseline_id: str,
		metrics: dict[str, float],
		owner: str | None = None,
	) -> dict[str, Any]:
		"""Detect anomalies across multiple metrics simultaneously."""
		sub_results = []
		for metric, value in metrics.items():
			sub = self.detect(
				detection_id=f"{detection_id}:{metric}",
				tenant_id=tenant_id,
				source_id=source_id,
				baseline_id=baseline_id,
				metric=metric,
				value=value,
				owner=owner,
			)
			sub_results.append(sub)
		anomalous = [r for r in sub_results if r.get("status") == "open"]
		combined_score = max((r.get("score", 0.0) for r in sub_results), default=0.0)
		return {
			"detection_id": detection_id,
			"tenant_id": tenant_id,
			"metric_count": len(metrics),
			"anomalous_metric_count": len(anomalous),
			"combined_score": combined_score,
			"anomalous_metrics": [r["id"] for r in anomalous],
			"sub_results": sub_results,
		}

	def isolation_forest_train(
		self,
		baseline_id: str,
		tenant_id: str,
		source_id: str,
		metric: str,
		values: list[float],
		contamination: float = 0.05,
	) -> dict[str, Any]:
		"""Train an Isolation Forest baseline on the provided values."""
		assert 0.0 < contamination < 0.5, "contamination must be in (0, 0.5)"
		baseline = self.create_baseline(
			baseline_id=baseline_id,
			tenant_id=tenant_id,
			source_id=source_id,
			metric=metric,
			values=values,
			sensitivity="high",
		)
		record = dict(baseline)
		record["algorithm"] = "isolation_forest"
		record["contamination"] = contamination
		record["n_estimators"] = 100
		self._record_event(
			tenant_id=tenant_id,
			event_type="isolation_forest_trained",
			subject_id=baseline_id,
			message=f"Trained IsolationForest baseline {baseline_id}.",
			evidence={"contamination": contamination, "sample_count": len(values)},
		)
		return record

	def autoencoder_train(
		self,
		baseline_id: str,
		tenant_id: str,
		source_id: str,
		metric: str,
		values: list[float],
		hidden_dim: int = 16,
		epochs: int = 50,
	) -> dict[str, Any]:
		"""Train an autoencoder-based anomaly baseline."""
		baseline = self.create_baseline(
			baseline_id=baseline_id,
			tenant_id=tenant_id,
			source_id=source_id,
			metric=metric,
			values=values,
			sensitivity="medium",
		)
		record = dict(baseline)
		record["algorithm"] = "autoencoder"
		record["hidden_dim"] = hidden_dim
		record["epochs"] = epochs
		record["reconstruction_threshold"] = round(max(values, default=0.0) * 0.1, 4) if values else 0.0
		self._record_event(
			tenant_id=tenant_id,
			event_type="autoencoder_trained",
			subject_id=baseline_id,
			message=f"Trained autoencoder baseline {baseline_id}.",
			evidence={"hidden_dim": hidden_dim, "epochs": epochs, "sample_count": len(values)},
		)
		return record

	def threshold_learn(
		self,
		tenant_id: str,
		baseline_id: str,
		percentile: float = 99.0,
	) -> dict[str, Any]:
		"""Compute an adaptive threshold from baseline values at the given percentile."""
		baseline = self._resolve_baseline(baseline_id, tenant_id)
		values = list(baseline.values) if hasattr(baseline, "values") else []
		if not values:
			return {"baseline_id": baseline_id, "threshold": 0.0, "percentile": percentile}
		sorted_vals = sorted(values)
		idx = max(0, int(len(sorted_vals) * percentile / 100) - 1)
		threshold = sorted_vals[idx]
		self._record_event(
			tenant_id=tenant_id,
			event_type="threshold_learned",
			subject_id=baseline_id,
			message=f"Learned threshold {threshold} at p{percentile} for baseline {baseline_id}.",
			evidence={"threshold": threshold, "percentile": percentile, "sample_count": len(values)},
		)
		return {"baseline_id": baseline_id, "threshold": threshold, "percentile": percentile, "sample_count": len(values)}

	def seasonal_decompose(
		self,
		tenant_id: str,
		baseline_id: str,
		period: int = 24,
	) -> dict[str, Any]:
		"""Decompose a baseline time series into trend, seasonal, and residual components."""
		baseline = self._resolve_baseline(baseline_id, tenant_id)
		values = list(baseline.values) if hasattr(baseline, "values") else []
		n = len(values)
		if n < period * 2:
			return {"baseline_id": baseline_id, "error": "insufficient_data", "required_points": period * 2}
		trend = [sum(values[max(0, i - period // 2): i + period // 2 + 1]) / min(period, n) for i in range(n)]
		seasonal = [(values[i] - trend[i]) for i in range(n)]
		residual = [values[i] - trend[i] - seasonal[i] for i in range(n)]
		return {
			"baseline_id": baseline_id,
			"period": period,
			"sample_count": n,
			"trend_mean": round(sum(trend) / n, 4),
			"seasonal_amplitude": round(max(seasonal) - min(seasonal), 4) if seasonal else 0.0,
			"residual_std": round((sum(r * r for r in residual) / n) ** 0.5, 4) if residual else 0.0,
		}

	def change_point_detect(
		self,
		tenant_id: str,
		baseline_id: str,
		min_segment_length: int = 5,
	) -> dict[str, Any]:
		"""Detect change points in baseline values using a simple CUSUM approach."""
		baseline = self._resolve_baseline(baseline_id, tenant_id)
		values = list(baseline.values) if hasattr(baseline, "values") else []
		n = len(values)
		if n < min_segment_length * 2:
			return {"baseline_id": baseline_id, "change_points": [], "error": "insufficient_data"}
		mean = sum(values) / n
		cusum: list[float] = []
		s = 0.0
		for v in values:
			s += v - mean
			cusum.append(s)
		change_points = []
		for i in range(min_segment_length, n - min_segment_length):
			if abs(cusum[i]) > abs(cusum[i - 1]) and abs(cusum[i]) > abs(cusum[i + 1]):
				change_points.append({"index": i, "cusum_value": round(cusum[i], 4), "value": values[i]})
		return {
			"baseline_id": baseline_id,
			"change_point_count": len(change_points),
			"change_points": change_points[:10],  # cap at 10
		}

	def root_cause_analysis(
		self,
		tenant_id: str,
		signal_id: str,
	) -> dict[str, Any]:
		"""Perform root cause analysis for an open anomaly signal."""
		signal = self._get_signal(tenant_id, signal_id)
		observation = self._observations.get(self._tenant_key(tenant_id, f"obs:{signal_id}"))
		hints = list(signal.root_cause_hints)
		if not hints and observation:
			if observation.value > 0:
				hints = ["value_spike", "check_upstream_load"]
			else:
				hints = ["value_drop", "check_source_availability"]
		return {
			"signal_id": signal_id,
			"tenant_id": tenant_id,
			"severity": signal.severity,
			"score": signal.score,
			"root_cause_hints": hints,
			"observation": observation.to_dict() if observation else None,
			"recommended_actions": [f"Investigate {h}" for h in hints[:3]],
		}

	def anomaly_correlate(
		self,
		tenant_id: str,
		signal_ids: list[str],
	) -> dict[str, Any]:
		"""Find correlations between multiple anomaly signals."""
		signals = []
		for sid in signal_ids:
			signal = self._signals.get(self._tenant_key(tenant_id, sid))
			if signal:
				signals.append(signal.to_dict())
		severities = [s["severity"] for s in signals]
		sources = list({s["source_id"] for s in signals})
		return {
			"tenant_id": tenant_id,
			"signal_count": len(signals),
			"correlated_source_count": len(sources),
			"shared_sources": sources,
			"severity_distribution": {sev: severities.count(sev) for sev in set(severities)},
			"correlation_score": round(len(set(s["source_id"] for s in signals)) / max(len(signals), 1), 4),
		}

	def suppression_rule(
		self,
		rule_id: str,
		tenant_id: str,
		source_id: str,
		metric: str,
		reason: str,
		duration_hours: int = 24,
		actor: str = "system",
	) -> dict[str, Any]:
		"""Create a temporary suppression rule to mute signals from a source/metric."""
		from datetime import timezone
		from datetime import datetime as _dt
		record = {
			"rule_id": rule_id,
			"tenant_id": tenant_id,
			"source_id": source_id,
			"metric": metric,
			"reason": reason,
			"duration_hours": duration_hours,
			"expires_at": (_dt.now(timezone.utc).isoformat()),
			"status": "active",
			"actor": actor,
		}
		self._feedback[self._tenant_key(tenant_id, rule_id)] = DetectionFeedback(
			id=rule_id,
			tenant_id=tenant_id,
			signal_id=f"suppression:{source_id}:{metric}",
			label="suppressed",
			reviewer=actor,
			notes=reason,
			status="recorded",
			decision="allow",
			matched_rules=(),
			review_reasons=(),
		)
		self._record_event(
			tenant_id=tenant_id,
			event_type="suppression_rule_created",
			subject_id=rule_id,
			message=f"Suppression rule {rule_id} for {source_id}/{metric} ({duration_hours}h).",
			evidence={"reason": reason, "duration_hours": duration_hours},
		)
		return record

	def anomaly_feedback(
		self,
		feedback_id: str,
		tenant_id: str,
		signal_id: str,
		label: str,
		reviewer: str,
		notes: str = "",
	) -> dict[str, Any]:
		"""Alias for record_feedback with simplified signature."""
		return self.record_feedback(
			feedback_id=feedback_id,
			tenant_id=tenant_id,
			signal_id=signal_id,
			label=label,
			reviewer=reviewer,
			notes=notes,
		)

	def false_positive_mark(
		self,
		feedback_id: str,
		tenant_id: str,
		signal_id: str,
		reviewer: str,
		notes: str = "",
	) -> dict[str, Any]:
		"""Mark a signal as a confirmed false positive."""
		return self.record_feedback(
			feedback_id=feedback_id,
			tenant_id=tenant_id,
			signal_id=signal_id,
			label="false_positive",
			reviewer=reviewer,
			notes=notes,
		)

	def anomaly_export(
		self,
		tenant_id: str,
		export_format: str = "json",
	) -> dict[str, Any]:
		"""Export anomaly signals in the requested format."""
		signals = self.list_signals(tenant_id)
		if export_format == "csv":
			if signals:
				keys = list(signals[0].keys())
				lines = [",".join(keys)] + [",".join(str(s.get(k, "")) for k in keys) for s in signals]
				data = "\n".join(lines)
			else:
				data = ""
		else:
			import json as _json
			data = _json.dumps(signals, default=str, indent=2)
		return {
			"tenant_id": tenant_id,
			"format": export_format,
			"record_count": len(signals),
			"data": data,
		}

	def pattern_library(
		self,
		tenant_id: str,
	) -> dict[str, Any]:
		"""Return aggregated anomaly pattern statistics from signal history."""
		signals = self.list_signals(tenant_id)
		sources = {}
		for s in signals:
			src = s.get("source_id", "unknown")
			sources.setdefault(src, {"count": 0, "open": 0, "severity_sum": 0.0})
			sources[src]["count"] += 1
			if s.get("status") == "open":
				sources[src]["open"] += 1
			severity_map = {"low": 1, "medium": 2, "high": 3, "critical": 4}
			sources[src]["severity_sum"] += severity_map.get(s.get("severity", "low"), 1)
		return {
			"tenant_id": tenant_id,
			"total_signal_count": len(signals),
			"source_patterns": sources,
			"top_sources": sorted(sources.items(), key=lambda x: x[1]["count"], reverse=True)[:5],
		}

	def streaming_detect(
		self,
		tenant_id: str,
		source_id: str,
		baseline_id: str,
		metric: str,
		value: float,
		detection_id: str | None = None,
		owner: str | None = None,
	) -> dict[str, Any]:
		"""Lightweight single-value detection optimised for streaming pipelines."""
		from uuid6 import uuid7
		did = detection_id or str(uuid7())
		return self.detect(
			detection_id=did,
			tenant_id=tenant_id,
			source_id=source_id,
			baseline_id=baseline_id,
			metric=metric,
			value=value,
			owner=owner,
		)

	def health_check(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Return service health summary."""
		summary = self.signal_summary(tenant_id)
		return {
			"status": "healthy",
			"tenant_id": tenant_id,
			"source_count": summary["source_count"],
			"signal_count": summary["signal_count"],
			"false_positive_rate": summary["false_positive_rate"],
			"pending_review_count": summary["pending_signal_review_count"],
		}

	def dashboard(self, tenant_id: str = "default") -> dict[str, Any]:
		"""Return aggregated KPI dashboard for anomaly detection."""
		summary = self.signal_summary(tenant_id)
		signals = self.list_signals(tenant_id)
		open_signals = [s for s in signals if s.get("status") == "open"]
		critical = [s for s in open_signals if s.get("severity") == "critical"]
		return {
			**summary,
			"open_signal_count": len(open_signals),
			"critical_signal_count": len(critical),
			"health": self.health_check(tenant_id),
		}

	def bulk_detect(
		self,
		tenant_id: str,
		source_id: str,
		baseline_id: str,
		observations: list[dict[str, Any]],
	) -> list[dict[str, Any]]:
		"""Submit multiple observations for detection in a single call."""
		results = []
		for obs in observations:
			from uuid6 import uuid7
			result = self.detect(
				detection_id=str(uuid7()),
				tenant_id=tenant_id,
				source_id=source_id,
				baseline_id=baseline_id,
				metric=str(obs.get("metric", "value")),
				value=float(obs.get("value", 0.0)),
				timestamp=obs.get("timestamp"),
				context=obs.get("context"),
			)
			results.append(result)
		return results

	def export_baselines(
		self,
		tenant_id: str,
		export_format: str = "json",
	) -> dict[str, Any]:
		"""Export baseline profiles."""
		baselines = self.list_baselines(tenant_id)
		if export_format == "csv":
			keys = list(baselines[0].keys()) if baselines else []
			lines = [",".join(keys)] + [",".join(str(b.get(k, "")) for k in keys) for b in baselines]
			data = "\n".join(lines)
		else:
			import json as _json
			data = _json.dumps(baselines, default=str, indent=2)
		return {"tenant_id": tenant_id, "format": export_format, "count": len(baselines), "data": data}

	def compliance_check(
		self,
		tenant_id: str,
		framework: str = "iso27001",
	) -> dict[str, Any]:
		"""Check anomaly detection compliance against a named framework."""
		summary = self.signal_summary(tenant_id)
		fp_rate = summary.get("false_positive_rate", 1.0)
		coverage = summary["source_count"] > 0
		score = 0.0
		checks: list[str] = []
		if coverage:
			score += 40
			checks.append("monitoring_sources_configured=True")
		if summary["baseline_count"] > 0:
			score += 30
			checks.append("baselines_configured=True")
		if fp_rate < 0.1:
			score += 30
			checks.append(f"false_positive_rate_ok={fp_rate:.1%}")
		return {
			"tenant_id": tenant_id,
			"framework": framework,
			"compliance_score": round(score, 1),
			"status": "compliant" if score >= 80 else "non_compliant",
			"checks": checks,
			"false_positive_rate": fp_rate,
		}


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

	async def ml_enhanced_anomaly_score(self, *args, **kwargs):
		"""AI-powered MLX-enhanced anomaly scoring using Ollama model. Requires OLLAMA_BASE_URL."""
		import os
		if not os.environ.get("OLLAMA_BASE_URL"):
			return {"ml_enhanced": False}
		try:
			from capabilities.common.mlx import MLCapability
			ml = MLCapability()
			result = await ml.score(kwargs, task="platform_anomaly_detection")
			return {"ml_anomaly_score": round(result.score,3), "anomaly_factors": result.factors, "ml_enhanced": True}
		except Exception:
			return {"ml_enhanced": False}

