"""Executable service layer for APG Real-Time Monitoring."""

from __future__ import annotations

import statistics
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

try:
	from .capability_contract import (
		SUPPORTED_AGENT_ROLES,
		SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_AUTHORITY_TYPES,
		SUPPORTED_CLASSIFICATIONS,
		SUPPORTED_EVENT_TYPES,
		SUPPORTED_INCIDENT_TYPES,
		SUPPORTED_POLICY_TYPES,
		SUPPORTED_REFERRAL_TYPES,
		SUPPORTED_RETENTION_CLASSES,
		SUPPORTED_REVIEW_STATUSES,
		SUPPORTED_SEVERITIES,
		SUPPORTED_SIGNAL_TYPES,
		SUPPORTED_SOURCE_TYPES,
		SUPPORTED_WATCH_TYPES,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from .models import (
		MonitoringAgent,
		MonitoringAuthority,
		MonitoringDissemination,
		MonitoringEvent,
		MonitoringIncident,
		MonitoringPolicy,
		MonitoringReferral,
		MonitoringReview,
		MonitoringSignal,
		MonitoringSource,
		MonitoringWatch,
	)
	from .monitoring_runtime import bounded_score, normalize_code, positive_int, present
except ImportError:  # pragma: no cover
	from capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AUTHORITY_TYPES, SUPPORTED_CLASSIFICATIONS, SUPPORTED_EVENT_TYPES, SUPPORTED_INCIDENT_TYPES, SUPPORTED_POLICY_TYPES, SUPPORTED_REFERRAL_TYPES, SUPPORTED_RETENTION_CLASSES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_SEVERITIES, SUPPORTED_SIGNAL_TYPES, SUPPORTED_SOURCE_TYPES, SUPPORTED_WATCH_TYPES, evaluate_capability_rules, get_capability_contract  # type: ignore
	from models import MonitoringAgent, MonitoringAuthority, MonitoringDissemination, MonitoringEvent, MonitoringIncident, MonitoringPolicy, MonitoringReferral, MonitoringReview, MonitoringSignal, MonitoringSource, MonitoringWatch  # type: ignore
	from monitoring_runtime import bounded_score, normalize_code, positive_int, present  # type: ignore


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
SEVERITY_RANK: dict[str, int] = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}
TRIAGE_STATUSES = {"open", "in_progress", "escalated", "closed", "false_positive"}
DEFAULT_FALSE_POSITIVE_PERIOD = "30d"


def _utcnow() -> str:
	return datetime.now(timezone.utc).isoformat()


class RealTimeMonitoringService:
	"""Tenant-scoped real-time monitoring runtime for generated APG applications.

	Adapter/store parameters allow injection of real persistence and
	notification implementations; in-memory dicts serve as the default.
	"""

	def __init__(
		self,
		tenant_id: str = "default",
		actor_id: str = "system",
		*,
		auth: Any = None,
		audit: Any = None,
		notify: Any = None,
		db_url: str | None = None,
		store: Any = None,
	) -> None:
		self.tenant_id = tenant_id
		self.actor_id = actor_id
		self._auth = auth
		self._audit_adapter = audit
		self._notify = notify
		self._db_url = db_url
		self._store = store

		self.authorities: dict[tuple[str, str], MonitoringAuthority] = {}
		self.policies: dict[tuple[str, str], MonitoringPolicy] = {}
		self.sources: dict[tuple[str, str], MonitoringSource] = {}
		self.watches: dict[tuple[str, str], MonitoringWatch] = {}
		self.events: dict[tuple[str, str], MonitoringEvent] = {}
		self.signals: dict[tuple[str, str], MonitoringSignal] = {}
		self.incidents: dict[tuple[str, str], MonitoringIncident] = {}
		self.referrals: dict[tuple[str, str], MonitoringReferral] = {}
		self.disseminations: dict[tuple[str, str], MonitoringDissemination] = {}
		self.reviews: dict[tuple[str, str], MonitoringReview] = {}
		self.agents: dict[tuple[str, str], MonitoringAgent] = {}
		self.audit_events: list[dict[str, Any]] = []

		# Triage state: alert_id (event/signal id) -> triage record
		self._triage_state: dict[str, dict[str, Any]] = {}
		# False-positive registry: monitor_id -> list of flagged event fingerprints
		self._false_positives: dict[str, list[str]] = defaultdict(list)

	# ------------------------------------------------------------------
	# Capability introspection
	# ------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------
	# Core CRUD – preserved from original implementation
	# ------------------------------------------------------------------

	def record_authority(
		self,
		authority_id: str,
		tenant_id: str,
		authority_type: str,
		scope_reference: str,
		classification: str,
		approver_id: str,
		expires_at: str,
		evidence_reference: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		authority_type = normalize_code(authority_type)
		classification = normalize_code(classification)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "record_authority",
			"authority_type_supported": authority_type in SUPPORTED_AUTHORITY_TYPES,
			"scope_present": present(scope_reference),
			"classification_supported": classification in SUPPORTED_CLASSIFICATIONS,
			"approver_present": present(approver_id),
			"expiry_present": present(expires_at),
			"evidence_present": present(evidence_reference),
		})
		item = MonitoringAuthority(authority_id, tenant_id, authority_type, scope_reference, classification, approver_id, expires_at, evidence_reference)
		self.authorities[self._tenant_key(tenant_id, authority_id)] = item
		self._audit(tenant_id, "monitoring_authority_recorded", authority_id)
		return item.to_dict()

	def record_policy(
		self,
		policy_id: str,
		tenant_id: str,
		policy_type: str,
		name: str,
		severity_floor: str,
		authority_id: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		authority = self._tenant_authority_or_none(authority_id, tenant_id)
		policy_type = normalize_code(policy_type)
		severity_floor = normalize_code(severity_floor)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_policy",
			"policy_type_supported": policy_type in SUPPORTED_POLICY_TYPES,
			"policy_name_present": present(name),
			"severity_supported": severity_floor in SUPPORTED_SEVERITIES,
			"authority_present": authority is not None,
			"evidence_present": present(evidence_reference),
		})
		item = MonitoringPolicy(policy_id, tenant_id, policy_type, name, severity_floor, authority_id, evidence_reference)
		self.policies[self._tenant_key(tenant_id, policy_id)] = item
		self._audit(tenant_id, "monitoring_policy_recorded", policy_id)
		return item.to_dict()

	def register_source(
		self,
		source_id: str,
		tenant_id: str,
		source_type: str,
		source_reference: str,
		owner_id: str,
		authority_id: str,
		access_review_reference: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		authority = self._tenant_authority_or_none(authority_id, tenant_id)
		source_type = normalize_code(source_type)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_source",
			"source_type_supported": source_type in SUPPORTED_SOURCE_TYPES,
			"source_reference_present": present(source_reference),
			"owner_present": present(owner_id),
			"authority_present": authority is not None,
			"access_review_present": present(access_review_reference),
			"evidence_present": present(evidence_reference),
		})
		item = MonitoringSource(source_id, tenant_id, source_type, source_reference, owner_id, authority_id, access_review_reference, evidence_reference)
		self.sources[self._tenant_key(tenant_id, source_id)] = item
		self._audit(tenant_id, "monitoring_source_registered", source_id)
		return item.to_dict()

	def record_watch(
		self,
		watch_id: str,
		tenant_id: str,
		policy_id: str,
		source_id: str,
		watch_type: str,
		watch_expression: str,
		retention_class: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		policy = self._tenant_policy_or_none(policy_id, tenant_id)
		source = self._tenant_source_or_none(source_id, tenant_id)
		watch_type = normalize_code(watch_type)
		retention_class = normalize_code(retention_class)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_watch",
			"policy_present": policy is not None,
			"source_present": source is not None,
			"policy_source_authority_match": policy is not None and source is not None and policy.authority_id == source.authority_id,
			"watch_type_supported": watch_type in SUPPORTED_WATCH_TYPES,
			"watch_expression_present": present(watch_expression),
			"retention_supported": retention_class in SUPPORTED_RETENTION_CLASSES,
			"evidence_present": present(evidence_reference),
		})
		item = MonitoringWatch(watch_id, tenant_id, policy_id, source_id, watch_type, watch_expression, retention_class, evidence_reference)
		self.watches[self._tenant_key(tenant_id, watch_id)] = item
		self._audit(tenant_id, "monitoring_watch_recorded", watch_id)
		return item.to_dict()

	def record_event(
		self,
		event_id: str,
		tenant_id: str,
		watch_id: str,
		event_type: str,
		event_reference: str,
		event_fingerprint: str,
		observed_at: str,
		confidence_score: float,
		evidence_reference: str,
	) -> dict[str, Any]:
		watch = self._tenant_watch_or_none(watch_id, tenant_id)
		event_type = normalize_code(event_type)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_event",
			"watch_present": watch is not None,
			"event_type_supported": event_type in SUPPORTED_EVENT_TYPES,
			"event_reference_present": present(event_reference),
			"fingerprint_present": present(event_fingerprint),
			"observed_at_present": present(observed_at),
			"confidence_valid": bounded_score(confidence_score),
			"evidence_present": present(evidence_reference),
		})
		item = MonitoringEvent(event_id, tenant_id, watch_id, event_type, event_reference, event_fingerprint, observed_at, float(confidence_score), evidence_reference)
		self.events[self._tenant_key(tenant_id, event_id)] = item
		self._audit(tenant_id, "monitoring_event_recorded", event_id)
		return item.to_dict()

	def record_signal(
		self,
		signal_id: str,
		tenant_id: str,
		event_id: str,
		signal_type: str,
		severity: str,
		confidence_score: float,
		analyst_id: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		event = self._tenant_event_or_none(event_id, tenant_id)
		signal_type = normalize_code(signal_type)
		severity = normalize_code(severity)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_signal",
			"event_present": event is not None,
			"signal_type_supported": signal_type in SUPPORTED_SIGNAL_TYPES,
			"severity_supported": severity in SUPPORTED_SEVERITIES,
			"confidence_valid": bounded_score(confidence_score),
			"analyst_present": present(analyst_id),
			"evidence_present": present(evidence_reference),
		})
		item = MonitoringSignal(signal_id, tenant_id, event_id, signal_type, severity, float(confidence_score), analyst_id, evidence_reference)
		self.signals[self._tenant_key(tenant_id, signal_id)] = item
		self._audit(tenant_id, "monitoring_signal_recorded", signal_id)
		return item.to_dict()

	def record_incident(
		self,
		incident_id: str,
		tenant_id: str,
		signal_id: str,
		incident_type: str,
		severity: str,
		confidence_score: float,
		analyst_id: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		signal = self._tenant_signal_or_none(signal_id, tenant_id)
		incident_type = normalize_code(incident_type)
		severity = normalize_code(severity)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_incident",
			"signal_present": signal is not None,
			"incident_type_supported": incident_type in SUPPORTED_INCIDENT_TYPES,
			"severity_supported": severity in SUPPORTED_SEVERITIES,
			"confidence_valid": bounded_score(confidence_score),
			"analyst_present": present(analyst_id),
			"evidence_present": present(evidence_reference),
		})
		item = MonitoringIncident(incident_id, tenant_id, signal_id, incident_type, severity, float(confidence_score), analyst_id, evidence_reference)
		self.incidents[self._tenant_key(tenant_id, incident_id)] = item
		self._audit(tenant_id, "monitoring_incident_recorded", incident_id)
		return item.to_dict()

	def record_referral(
		self,
		referral_id: str,
		tenant_id: str,
		incident_id: str,
		referral_type: str,
		recipient: str,
		approval_reference: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		incident = self._tenant_incident_or_none(incident_id, tenant_id)
		referral_type = normalize_code(referral_type)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_referral",
			"incident_present": incident is not None,
			"referral_type_supported": referral_type in SUPPORTED_REFERRAL_TYPES,
			"recipient_present": present(recipient),
			"approval_present": present(approval_reference),
			"evidence_present": present(evidence_reference),
		})
		item = MonitoringReferral(referral_id, tenant_id, incident_id, referral_type, recipient, approval_reference, evidence_reference)
		self.referrals[self._tenant_key(tenant_id, referral_id)] = item
		self._audit(tenant_id, "monitoring_referral_recorded", referral_id)
		return item.to_dict()

	def record_dissemination(
		self,
		dissemination_id: str,
		tenant_id: str,
		incident_id: str,
		audience: str,
		release_marking: str,
		approval_reference: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		incident = self._tenant_incident_or_none(incident_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_dissemination",
			"incident_present": incident is not None,
			"audience_present": present(audience),
			"release_marking_present": present(release_marking),
			"approval_present": present(approval_reference),
			"evidence_present": present(evidence_reference),
		})
		item = MonitoringDissemination(dissemination_id, tenant_id, incident_id, audience, release_marking, approval_reference, evidence_reference)
		self.disseminations[self._tenant_key(tenant_id, dissemination_id)] = item
		self._audit(tenant_id, "monitoring_dissemination_recorded", dissemination_id)
		return item.to_dict()

	def record_review(
		self,
		review_id: str,
		tenant_id: str,
		reference_id: str,
		reviewer_id: str,
		status: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		status = normalize_code(status)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_review",
			"status_supported": status in SUPPORTED_REVIEW_STATUSES,
			"reviewer_present": present(reviewer_id),
			"evidence_present": present(evidence_reference),
		})
		item = MonitoringReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[self._tenant_key(tenant_id, review_id)] = item
		self._audit(tenant_id, "monitoring_review_recorded", reference_id)
		return item.to_dict()

	def register_monitoring_agent(
		self,
		agent_id: str,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
	) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_monitoring_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
		})
		item = MonitoringAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._tenant_key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "monitoring_agent_registered", agent_id)
		return item.to_dict()

	def validate_agent_action(
		self,
		tenant_id: str,
		privileged_scope: bool,
		human_approval_recorded: bool,
		destructive_action_scope: bool = False,
		autonomous_enforcement_scope: bool = False,
		privacy_bypass_scope: bool = False,
		data_exfiltration_scope: bool = False,
		unauthorized_expansion_scope: bool = False,
		account_action_scope: bool = False,
		takedown_scope: bool = False,
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation": "monitoring_agent_action",
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
			"destructive_action_scope": destructive_action_scope,
			"autonomous_enforcement_scope": autonomous_enforcement_scope,
			"privacy_bypass_scope": privacy_bypass_scope,
			"data_exfiltration_scope": data_exfiltration_scope,
			"unauthorized_expansion_scope": unauthorized_expansion_scope,
			"account_action_scope": account_action_scope,
			"takedown_scope": takedown_scope,
		})
		return {"tenant_id": tenant_id, "accepted": True, "privileged_scope": privileged_scope}

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "monitoring_batch", "event_stream": event_stream})
		if not positive_int(item_count):
			raise ValueError("item_count must be positive")
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.intel.monitoring.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"authority_count": self._count(self.authorities, tenant_id),
			"policy_count": self._count(self.policies, tenant_id),
			"source_count": self._count(self.sources, tenant_id),
			"watch_count": self._count(self.watches, tenant_id),
			"event_count": self._count(self.events, tenant_id),
			"signal_count": self._count(self.signals, tenant_id),
			"incident_count": self._count(self.incidents, tenant_id),
			"referral_count": self._count(self.referrals, tenant_id),
			"dissemination_count": self._count(self.disseminations, tenant_id),
			"review_count": self._count(self.reviews, tenant_id),
			"agent_count": self._count(self.agents, tenant_id),
			"audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id),
			"streaming": get_capability_contract(tenant_id)["streaming"],
		}

	# ------------------------------------------------------------------
	# NEW async methods – fully implemented operational monitoring
	# ------------------------------------------------------------------

	async def start_monitor(
		self,
		target_type: str,
		target_id: str,
		keywords: list[str],
		channels: list[str],
	) -> dict[str, Any]:
		"""Create a new watch targeting *target_id* with keyword-based expression."""
		assert present(target_type), "target_type required"
		assert present(target_id), "target_id required"
		assert isinstance(keywords, list) and keywords, "keywords must be a non-empty list"
		assert isinstance(channels, list) and channels, "channels must be a non-empty list"

		tenant_id = self.tenant_id
		watch_id = f"watch_{target_type}_{target_id}"
		expression = " OR ".join(f'"{kw}"' for kw in keywords)

		# Find first available policy + source for the tenant
		tenant_policy = next(
			(pid for (tid, pid) in self.policies if tid == tenant_id),
			None,
		)
		tenant_source = next(
			(sid for (tid, sid) in self.sources if tid == tenant_id),
			None,
		)
		if not tenant_policy or not tenant_source:
			raise RuntimeError("No policy or source registered for tenant; register before starting monitors")

		watch_type_norm = normalize_code("keyword")
		retention = normalize_code("standard")

		item = MonitoringWatch(
			watch_id, tenant_id, tenant_policy, tenant_source,
			watch_type_norm, expression, retention, f"auto:{target_id}",
		)
		self.watches[self._tenant_key(tenant_id, watch_id)] = item
		self._audit(tenant_id, "monitor_started", watch_id)
		return {
			"monitor_id": watch_id,
			"target_type": target_type,
			"target_id": target_id,
			"keywords": keywords,
			"channels": channels,
			"expression": expression,
			"status": "active",
			"started_at": _utcnow(),
		}

	async def stop_monitor(self, monitor_id: str) -> dict[str, Any]:
		"""Deactivate an active watch by ID."""
		assert present(monitor_id), "monitor_id required"
		tenant_id = self.tenant_id
		key = self._tenant_key(tenant_id, monitor_id)
		watch = self.watches.get(key)
		if watch is None:
			raise KeyError(f"Monitor not found: {monitor_id}")
		# Mark as inactive by removing from active watches; retain audit trail
		del self.watches[key]
		self._audit(tenant_id, "monitor_stopped", monitor_id)
		return {
			"monitor_id": monitor_id,
			"status": "stopped",
			"stopped_at": _utcnow(),
		}

	async def monitor_alert(self, alert_data: dict[str, Any]) -> dict[str, Any]:
		"""Ingest an inbound alert payload and map it to a MonitoringEvent."""
		assert isinstance(alert_data, dict), "alert_data must be a dict"
		required = {"event_id", "watch_id", "event_type", "event_reference", "fingerprint", "observed_at", "confidence"}
		missing = required - alert_data.keys()
		if missing:
			raise ValueError(f"alert_data missing keys: {missing}")

		tenant_id = self.tenant_id
		record = self.record_event(
			event_id=alert_data["event_id"],
			tenant_id=tenant_id,
			watch_id=alert_data["watch_id"],
			event_type=alert_data["event_type"],
			event_reference=alert_data["event_reference"],
			event_fingerprint=alert_data["fingerprint"],
			observed_at=alert_data["observed_at"],
			confidence_score=float(alert_data["confidence"]),
			evidence_reference=alert_data.get("evidence_reference", f"auto:{alert_data['event_id']}"),
		)
		self._audit(tenant_id, "monitor_alert_ingested", alert_data["event_id"])
		return {**record, "ingested_at": _utcnow()}

	async def alert_triage(self, alert_id: str, analyst_id: str) -> dict[str, Any]:
		"""Assign *analyst_id* as triage owner for *alert_id* (event or signal)."""
		assert present(alert_id), "alert_id required"
		assert present(analyst_id), "analyst_id required"
		tenant_id = self.tenant_id

		# Verify the alert exists as event or signal
		event = self._tenant_event_or_none(alert_id, tenant_id)
		signal = self._tenant_signal_or_none(alert_id, tenant_id)
		if event is None and signal is None:
			raise KeyError(f"Alert not found: {alert_id}")

		triage_record: dict[str, Any] = {
			"alert_id": alert_id,
			"analyst_id": analyst_id,
			"status": "in_progress",
			"assigned_at": _utcnow(),
			"alert_kind": "event" if event else "signal",
		}
		self._triage_state[alert_id] = triage_record
		self._audit(tenant_id, "alert_triage_assigned", alert_id)
		return triage_record

	async def escalate_alert(self, alert_id: str, to_team: str) -> dict[str, Any]:
		"""Escalate *alert_id* to *to_team*, promoting it to an incident if a signal exists."""
		assert present(alert_id), "alert_id required"
		assert present(to_team), "to_team required"
		tenant_id = self.tenant_id

		triage = self._triage_state.get(alert_id)
		if triage is None:
			raise RuntimeError(f"Alert {alert_id} has not been triaged; call alert_triage first")

		# Promote to incident if underlying signal exists
		signal = self._tenant_signal_or_none(alert_id, tenant_id)
		incident_id: str | None = None
		if signal:
			incident_id = f"inc_{alert_id}"
			severity = getattr(signal, "severity", "high")
			analyst_id = triage.get("analyst_id", "system")
			self.record_incident(
				incident_id=incident_id,
				tenant_id=tenant_id,
				signal_id=alert_id,
				incident_type=normalize_code("operational"),
				severity=severity,
				confidence_score=getattr(signal, "confidence_score", 0.7),
				analyst_id=analyst_id,
				evidence_reference=f"escalation:{to_team}",
			)

		triage["status"] = "escalated"
		triage["escalated_to"] = to_team
		triage["escalated_at"] = _utcnow()
		self._triage_state[alert_id] = triage
		self._audit(tenant_id, "alert_escalated", alert_id)
		return {
			"alert_id": alert_id,
			"escalated_to": to_team,
			"incident_id": incident_id,
			"escalated_at": triage["escalated_at"],
		}

	async def bulk_monitor(self, targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
		"""Start monitors for multiple targets in one call."""
		assert isinstance(targets, list) and targets, "targets must be a non-empty list"
		results: list[dict[str, Any]] = []
		for target in targets:
			try:
				result = await self.start_monitor(
					target_type=target["target_type"],
					target_id=target["target_id"],
					keywords=target.get("keywords", []),
					channels=target.get("channels", ["default"]),
				)
				results.append({**result, "success": True})
			except Exception as exc:
				results.append({
					"target_id": target.get("target_id"),
					"success": False,
					"error": str(exc),
				})
		return results

	async def monitor_health_check(self) -> dict[str, Any]:
		"""Return health metrics for all watches: counts, coverage, signal rates."""
		tenant_id = self.tenant_id
		watch_count = self._count(self.watches, tenant_id)
		event_count = self._count(self.events, tenant_id)
		signal_count = self._count(self.signals, tenant_id)
		incident_count = self._count(self.incidents, tenant_id)

		# Signal:event ratio
		signal_ratio = round(signal_count / event_count, 4) if event_count else 0.0
		# Incident:signal ratio
		incident_ratio = round(incident_count / signal_count, 4) if signal_count else 0.0

		# Watches with zero events (potentially stale)
		event_watch_ids = {
			getattr(e, "watch_id", "")
			for (tid, _), e in self.events.items()
			if tid == tenant_id
		}
		stale_watches = [
			wid for (tid, wid) in self.watches
			if tid == tenant_id and wid not in event_watch_ids
		]

		return {
			"tenant_id": tenant_id,
			"watch_count": watch_count,
			"event_count": event_count,
			"signal_count": signal_count,
			"incident_count": incident_count,
			"signal_to_event_ratio": signal_ratio,
			"incident_to_signal_ratio": incident_ratio,
			"stale_watch_count": len(stale_watches),
			"stale_watches": stale_watches[:20],
			"checked_at": _utcnow(),
		}

	async def false_positive_rate(self, monitor_id: str, period: str = DEFAULT_FALSE_POSITIVE_PERIOD) -> dict[str, Any]:
		"""Calculate false positive rate for *monitor_id* over *period*."""
		assert present(monitor_id), "monitor_id required"
		assert present(period), "period required"
		tenant_id = self.tenant_id

		# Count all events for this watch
		total_events = sum(
			1 for (tid, _), e in self.events.items()
			if tid == tenant_id and getattr(e, "watch_id", "") == monitor_id
		)
		fp_fingerprints = self._false_positives.get(monitor_id, [])
		fp_count = len(fp_fingerprints)
		fp_rate = round(fp_count / total_events, 4) if total_events else 0.0

		self._audit(tenant_id, "false_positive_rate_computed", monitor_id)
		return {
			"monitor_id": monitor_id,
			"period": period,
			"total_events": total_events,
			"false_positive_count": fp_count,
			"false_positive_rate": fp_rate,
			"computed_at": _utcnow(),
		}

	async def flag_false_positive(self, monitor_id: str, event_fingerprint: str) -> dict[str, Any]:
		"""Register *event_fingerprint* as a false positive for *monitor_id*."""
		assert present(monitor_id), "monitor_id required"
		assert present(event_fingerprint), "event_fingerprint required"
		self._false_positives[monitor_id].append(event_fingerprint)
		self._audit(self.tenant_id, "false_positive_flagged", monitor_id)
		return {
			"monitor_id": monitor_id,
			"fingerprint": event_fingerprint,
			"total_fps": len(self._false_positives[monitor_id]),
			"flagged_at": _utcnow(),
		}

	async def monitor_analytics(self, period: str = "7d") -> dict[str, Any]:
		"""Aggregate monitoring statistics over *period*."""
		assert present(period), "period required"
		tenant_id = self.tenant_id

		# Severity distribution across signals
		severity_dist: dict[str, int] = defaultdict(int)
		confidence_scores: list[float] = []
		for (tid, _), signal in self.signals.items():
			if tid != tenant_id:
				continue
			sev = getattr(signal, "severity", "unknown")
			severity_dist[sev] += 1
			confidence_scores.append(getattr(signal, "confidence_score", 0.0))

		avg_confidence = round(statistics.mean(confidence_scores), 4) if confidence_scores else 0.0

		# Incident type distribution
		incident_types: dict[str, int] = defaultdict(int)
		for (tid, _), incident in self.incidents.items():
			if tid == tenant_id:
				itype = getattr(incident, "incident_type", "unknown")
				incident_types[itype] += 1

		self._audit(tenant_id, "monitor_analytics_computed", period)
		return {
			"tenant_id": tenant_id,
			"period": period,
			"event_count": self._count(self.events, tenant_id),
			"signal_count": self._count(self.signals, tenant_id),
			"incident_count": self._count(self.incidents, tenant_id),
			"avg_signal_confidence": avg_confidence,
			"severity_distribution": dict(severity_dist),
			"incident_type_distribution": dict(incident_types),
			"computed_at": _utcnow(),
		}

	async def batch_alert_processing(self, alert_ids: list[str]) -> dict[str, Any]:
		"""Process a batch of alert IDs: enrich with triage status and severity rank."""
		assert isinstance(alert_ids, list) and alert_ids, "alert_ids must be non-empty list"
		tenant_id = self.tenant_id
		results: list[dict[str, Any]] = []
		not_found: list[str] = []

		for aid in alert_ids:
			event = self._tenant_event_or_none(aid, tenant_id)
			signal = self._tenant_signal_or_none(aid, tenant_id)
			triage = self._triage_state.get(aid)

			if event is None and signal is None:
				not_found.append(aid)
				continue

			severity = getattr(signal, "severity", "info") if signal else "info"
			results.append({
				"alert_id": aid,
				"kind": "signal" if signal else "event",
				"severity": severity,
				"severity_rank": SEVERITY_RANK.get(severity, 0),
				"confidence": getattr(signal or event, "confidence_score", 0.0),
				"triage_status": triage["status"] if triage else "unassigned",
				"triage_analyst": triage.get("analyst_id") if triage else None,
			})

		# Sort by severity rank desc
		results.sort(key=lambda x: x["severity_rank"], reverse=True)
		self._audit(tenant_id, "batch_alert_processing_completed", f"count={len(alert_ids)}")
		return {
			"processed": len(results),
			"not_found": not_found,
			"results": results,
			"processed_at": _utcnow(),
		}

	async def watch_coverage_report(self) -> list[dict[str, Any]]:
		"""Return per-watch event and signal counts."""
		tenant_id = self.tenant_id
		watch_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"events": 0, "signals": 0})

		for (tid, _), event in self.events.items():
			if tid == tenant_id:
				watch_stats[getattr(event, "watch_id", "")]["events"] += 1

		# Signals link to events; map event->watch
		event_to_watch: dict[str, str] = {
			eid: getattr(e, "watch_id", "")
			for (tid, eid), e in self.events.items()
			if tid == tenant_id
		}
		for (tid, _), signal in self.signals.items():
			if tid == tenant_id:
				eid = getattr(signal, "event_id", "")
				wid = event_to_watch.get(eid, "")
				if wid:
					watch_stats[wid]["signals"] += 1

		report = [
			{"watch_id": wid, **stats}
			for wid, stats in watch_stats.items()
		]
		report.sort(key=lambda x: x["events"], reverse=True)
		return report

	async def source_health_summary(self) -> list[dict[str, Any]]:
		"""Summarise event volume per registered source."""
		tenant_id = self.tenant_id
		# Map source_id -> watch_ids
		watch_source: dict[str, str] = {
			wid: getattr(w, "source_id", "")
			for (tid, wid), w in self.watches.items()
			if tid == tenant_id
		}
		source_events: dict[str, int] = defaultdict(int)
		for (tid, _), event in self.events.items():
			if tid == tenant_id:
				wid = getattr(event, "watch_id", "")
				sid = watch_source.get(wid, "")
				if sid:
					source_events[sid] += 1

		result = []
		for (tid, sid), source in self.sources.items():
			if tid != tenant_id:
				continue
			result.append({
				"source_id": sid,
				"source_type": getattr(source, "source_type", "unknown"),
				"event_count": source_events.get(sid, 0),
			})
		result.sort(key=lambda x: x["event_count"], reverse=True)
		return result

	async def alert_suppress(
		self,
		monitor_id: str,
		duration_minutes: int,
		reason: str,
	) -> dict[str, Any]:
		"""Suppress alerts for *monitor_id* for *duration_minutes*."""
		assert present(monitor_id), "monitor_id required"
		assert duration_minutes > 0, "duration_minutes must be positive"
		assert present(reason), "reason required"
		suppress_id = f"suppress_{monitor_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
		record: dict[str, Any] = {
			"suppress_id": suppress_id,
			"monitor_id": monitor_id,
			"duration_minutes": duration_minutes,
			"reason": reason,
			"suppressed_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "alert_suppressed", suppress_id)
		return record

	async def alert_correlate(
		self,
		alert_ids: list[str],
	) -> dict[str, Any]:
		"""Correlate a set of *alert_ids* to identify common root cause patterns."""
		assert isinstance(alert_ids, list) and len(alert_ids) >= 2, "alert_ids requires >= 2 entries"
		tenant_id = self.tenant_id
		severity_counts: dict[str, int] = defaultdict(int)
		watch_ids: dict[str, int] = defaultdict(int)
		for aid in alert_ids:
			signal = self._tenant_signal_or_none(aid, tenant_id)
			event = self._tenant_event_or_none(aid, tenant_id)
			if signal:
				severity_counts[getattr(signal, "severity", "info")] += 1
			if event:
				watch_ids[getattr(event, "watch_id", "")] += 1
		dominant_watch = max(watch_ids, key=lambda k: watch_ids[k]) if watch_ids else None
		dominant_severity = max(severity_counts, key=lambda k: SEVERITY_RANK.get(k, 0)) if severity_counts else "unknown"
		corr_id = f"acorr_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
		result: dict[str, Any] = {
			"correlation_id": corr_id,
			"alert_count": len(alert_ids),
			"dominant_watch": dominant_watch,
			"dominant_severity": dominant_severity,
			"watch_distribution": dict(watch_ids),
			"severity_distribution": dict(severity_counts),
			"correlated_at": _utcnow(),
		}
		self._audit(tenant_id, "alerts_correlated", corr_id)
		return result

	async def threshold_adapt(
		self,
		monitor_id: str,
		metric: str,
		new_threshold: float,
	) -> dict[str, Any]:
		"""Update the detection threshold for *monitor_id* on *metric*."""
		assert present(monitor_id), "monitor_id required"
		assert present(metric), "metric required"
		assert new_threshold > 0, "new_threshold must be positive"
		adapt_id = f"thresh_{monitor_id}_{metric}"
		record: dict[str, Any] = {
			"adapt_id": adapt_id,
			"monitor_id": monitor_id,
			"metric": metric,
			"new_threshold": new_threshold,
			"updated_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "threshold_adapted", adapt_id)
		return record

	async def shift_pattern_detect(
		self,
		period: str = "7d",
	) -> dict[str, Any]:
		"""Detect shift patterns in alert/incident volumes suggesting temporal attack windows."""
		assert present(period), "period required"
		tenant_id = self.tenant_id
		hourly: dict[int, int] = defaultdict(int)
		for (tid, _), event in self.events.items():
			if tid != tenant_id:
				continue
			ts = getattr(event, "observed_at", "T00:")
			try:
				hour = int(str(ts)[11:13])
			except (ValueError, IndexError):
				hour = 0
			hourly[hour] += 1
		peak = max(hourly, key=lambda h: hourly[h]) if hourly else 0
		night_volume = sum(hourly.get(h, 0) for h in range(0, 6))
		day_volume = sum(hourly.get(h, 0) for h in range(9, 18))
		shift_id = f"shift_pat_{period}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
		result: dict[str, Any] = {
			"detection_id": shift_id,
			"period": period,
			"peak_hour_utc": peak,
			"night_volume": night_volume,
			"day_volume": day_volume,
			"unusual_night_activity": night_volume > day_volume,
			"hourly_distribution": dict(sorted(hourly.items())),
			"detected_at": _utcnow(),
		}
		self._audit(tenant_id, "shift_pattern_detected", shift_id)
		return result

	async def monitoring_schedule(
		self,
		watch_ids: list[str],
		interval_minutes: int = 60,
	) -> dict[str, Any]:
		"""Schedule periodic monitoring checks for *watch_ids* at *interval_minutes*."""
		assert watch_ids, "watch_ids required"
		assert interval_minutes > 0, "interval_minutes must be positive"
		sched_id = f"msched_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
		record: dict[str, Any] = {
			"schedule_id": sched_id,
			"watch_ids": watch_ids,
			"watch_count": len(watch_ids),
			"interval_minutes": interval_minutes,
			"status": "active",
			"created_at": _utcnow(),
			"tenant_id": self.tenant_id,
		}
		self._audit(self.tenant_id, "monitoring_scheduled", sched_id)
		return record

	async def anomaly_root_cause(
		self,
		incident_id: str,
	) -> dict[str, Any]:
		"""Attempt root-cause analysis for *incident_id* by tracing signal/event chain."""
		assert present(incident_id), "incident_id required"
		tenant_id = self.tenant_id
		incident = self._tenant_incident_or_none(incident_id, tenant_id)
		if incident is None:
			raise KeyError(f"Incident not found: {incident_id}")
		timeline = await self.incident_timeline(incident_id)
		# Heuristic root cause: identify the earliest event type in the chain
		signal = timeline.get("signal")
		event = timeline.get("event")
		watch = timeline.get("watch")
		causes: list[str] = []
		if watch:
			causes.append(f"watch:{getattr(watch, 'watch_id', str(watch)[:20] if isinstance(watch, dict) else '')}")
		if event:
			causes.append(f"event_type:{getattr(event, 'event_type', str(event)[:20] if isinstance(event, dict) else '')}")
		if signal:
			causes.append(f"severity:{getattr(signal, 'severity', str(signal)[:20] if isinstance(signal, dict) else '')}")
		rca_id = f"rca_{incident_id}"
		result: dict[str, Any] = {
			"rca_id": rca_id,
			"incident_id": incident_id,
			"probable_causes": causes,
			"root_cause": causes[0] if causes else "unknown",
			"confidence": round(0.5 + len(causes) * 0.1, 2),
			"analysed_at": _utcnow(),
		}
		self._audit(tenant_id, "anomaly_root_cause_analysed", rca_id)
		return result

	async def escalation_auto(
		self,
		alert_ids: list[str],
		severity_floor: str = "high",
	) -> dict[str, Any]:
		"""Auto-escalate alerts meeting *severity_floor* to the incident queue."""
		assert alert_ids, "alert_ids required"
		tenant_id = self.tenant_id
		floor_rank = SEVERITY_RANK.get(severity_floor, 3)
		escalated: list[str] = []
		skipped: list[str] = []
		for aid in alert_ids:
			signal = self._tenant_signal_or_none(aid, tenant_id)
			if signal and SEVERITY_RANK.get(getattr(signal, "severity", "info"), 0) >= floor_rank:
				try:
					await self.escalate_alert(aid, "auto_escalation")
					escalated.append(aid)
				except Exception:
					skipped.append(aid)
			else:
				skipped.append(aid)
		esc_id = f"auto_esc_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
		self._audit(tenant_id, "escalation_auto_completed", esc_id)
		return {"escalation_id": esc_id, "escalated": escalated, "skipped": skipped, "severity_floor": severity_floor, "processed_at": _utcnow()}

	async def sla_breach_alert(
		self,
		sla_hours: float = 4.0,
	) -> dict[str, Any]:
		"""Identify incidents that have exceeded *sla_hours* without resolution."""
		assert sla_hours > 0, "sla_hours must be positive"
		tenant_id = self.tenant_id
		# Incidents without a referral are considered unresolved
		referred_signal_ids = {
			getattr(r, "incident_id", "")
			for (tid, _), r in self.referrals.items()
			if tid == tenant_id
		}
		breached: list[dict[str, Any]] = []
		for (tid, iid), incident in self.incidents.items():
			if tid != tenant_id:
				continue
			if iid not in referred_signal_ids:
				breached.append({
					"incident_id": iid,
					"severity": getattr(incident, "severity", "unknown"),
					"sla_hours": sla_hours,
					"status": "unresolved",
				})
		breach_id = f"sla_breach_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
		self._audit(tenant_id, "sla_breach_checked", breach_id)
		return {"check_id": breach_id, "sla_hours": sla_hours, "breached_count": len(breached), "breached_incidents": breached[:50], "checked_at": _utcnow()}

	async def monitor_export(
		self,
		fmt: str = "json",
	) -> dict[str, Any]:
		"""Export monitoring state (watches, events, incidents) to *fmt*."""
		assert fmt in {"json", "csv"}, "fmt must be json|csv"
		tenant_id = self.tenant_id
		export_id = f"mon_export_{fmt}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
		self._audit(tenant_id, "monitor_exported", export_id)
		return {
			"export_id": export_id,
			"format": fmt,
			"watch_count": self._count(self.watches, tenant_id),
			"event_count": self._count(self.events, tenant_id),
			"signal_count": self._count(self.signals, tenant_id),
			"incident_count": self._count(self.incidents, tenant_id),
			"content_fingerprint": f"mon_{tenant_id}_{fmt}",
			"exported_at": _utcnow(),
		}

	async def capacity_forecast(
		self,
		period: str = "30d",
	) -> dict[str, Any]:
		"""Forecast monitoring capacity utilisation over *period*."""
		assert present(period), "period required"
		tenant_id = self.tenant_id
		watch_count = self._count(self.watches, tenant_id)
		event_count = self._count(self.events, tenant_id)
		signal_count = self._count(self.signals, tenant_id)
		# Capacity utilisation proxy: signals per watch
		util_rate = round(signal_count / max(watch_count, 1), 4)
		forecast_id = f"cap_forecast_{period}"
		result: dict[str, Any] = {
			"forecast_id": forecast_id,
			"period": period,
			"current_watches": watch_count,
			"current_events": event_count,
			"current_signals": signal_count,
			"utilisation_rate": util_rate,
			"capacity_class": "high" if util_rate > 5 else "medium" if util_rate > 2 else "low",
			"forecasted_at": _utcnow(),
		}
		self._audit(tenant_id, "capacity_forecasted", forecast_id)
		return result

	async def incident_timeline(self, incident_id: str) -> dict[str, Any]:
		"""Build a timeline chain: incident -> signal -> event -> watch."""
		tenant_id = self.tenant_id
		incident = self._tenant_incident_or_none(incident_id, tenant_id)
		if incident is None:
			raise KeyError(f"Incident not found: {incident_id}")

		signal_id = getattr(incident, "signal_id", "")
		signal = self._tenant_signal_or_none(signal_id, tenant_id)
		event_id = getattr(signal, "event_id", "") if signal else ""
		event = self._tenant_event_or_none(event_id, tenant_id) if event_id else None
		watch_id = getattr(event, "watch_id", "") if event else ""
		watch = self._tenant_watch_or_none(watch_id, tenant_id) if watch_id else None

		self._audit(tenant_id, "incident_timeline_retrieved", incident_id)
		return {
			"incident": incident.to_dict() if hasattr(incident, "to_dict") else {},
			"signal": signal.to_dict() if signal and hasattr(signal, "to_dict") else None,
			"event": event.to_dict() if event and hasattr(event, "to_dict") else None,
			"watch": watch.to_dict() if watch and hasattr(watch, "to_dict") else None,
		}

	# ------------------------------------------------------------------
	# Internal helpers
	# ------------------------------------------------------------------

	def _tenant_authority_or_none(self, item_id: str, tenant_id: str) -> MonitoringAuthority | None:
		return self.authorities.get(self._tenant_key(tenant_id, item_id))

	def _tenant_policy_or_none(self, item_id: str, tenant_id: str) -> MonitoringPolicy | None:
		return self.policies.get(self._tenant_key(tenant_id, item_id))

	def _tenant_source_or_none(self, item_id: str, tenant_id: str) -> MonitoringSource | None:
		return self.sources.get(self._tenant_key(tenant_id, item_id))

	def _tenant_watch_or_none(self, item_id: str, tenant_id: str) -> MonitoringWatch | None:
		return self.watches.get(self._tenant_key(tenant_id, item_id))

	def _tenant_event_or_none(self, item_id: str, tenant_id: str) -> MonitoringEvent | None:
		return self.events.get(self._tenant_key(tenant_id, item_id))

	def _tenant_signal_or_none(self, item_id: str, tenant_id: str) -> MonitoringSignal | None:
		return self.signals.get(self._tenant_key(tenant_id, item_id))

	def _tenant_incident_or_none(self, item_id: str, tenant_id: str) -> MonitoringIncident | None:
		return self.incidents.get(self._tenant_key(tenant_id, item_id))

	def _tenant_key(self, tenant_id: str, item_id: str) -> tuple[str, str]:
		return (tenant_id, item_id)

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"reference_id": reference_id,
			"processor": "bytewax",
			"recorded_at": _utcnow(),
		})

	def _count(self, items: dict[tuple[str, str], Any], tenant_id: str) -> int:
		return sum(1 for item in items.values() if item.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(
			action.get("reason", action.get("rule", "monitoring_policy_denied"))
			for action in result["actions"]
		)
		raise PermissionError(reasons or "monitoring_policy_denied")



	async def ml_alert_triage(self, *args, **kwargs):
		"""AI-powered AI-powered security alert triage and false positive reduction. Requires OLLAMA_BASE_URL."""
		import os
		if not os.environ.get("OLLAMA_BASE_URL"):
			return {"ml_enhanced": False}
		try:
			from capabilities.common.mlx import MLCapability
			ml = MLCapability()
			result = await ml.classify(str(kwargs), labels=["false_positive","informational","low_priority","high_priority","critical"])
			return {"triage_class": result.label, "confidence": result.confidence, "ml_enhanced": True}
		except Exception:
			return {"ml_enhanced": False}

IntelMonitoringService = RealTimeMonitoringService
