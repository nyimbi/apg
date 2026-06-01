"""Executable service layer for APG Real-Time Monitoring."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AUTHORITY_TYPES, SUPPORTED_CLASSIFICATIONS, SUPPORTED_EVENT_TYPES, SUPPORTED_INCIDENT_TYPES, SUPPORTED_POLICY_TYPES, SUPPORTED_REFERRAL_TYPES, SUPPORTED_RETENTION_CLASSES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_SEVERITIES, SUPPORTED_SIGNAL_TYPES, SUPPORTED_SOURCE_TYPES, SUPPORTED_WATCH_TYPES, evaluate_capability_rules, get_capability_contract
	from .models import MonitoringAgent, MonitoringAuthority, MonitoringDissemination, MonitoringEvent, MonitoringIncident, MonitoringPolicy, MonitoringReferral, MonitoringReview, MonitoringSignal, MonitoringSource, MonitoringWatch
	from .monitoring_runtime import bounded_score, normalize_code, positive_int, present
except ImportError:  # pragma: no cover
	from capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_AUTHORITY_TYPES, SUPPORTED_CLASSIFICATIONS, SUPPORTED_EVENT_TYPES, SUPPORTED_INCIDENT_TYPES, SUPPORTED_POLICY_TYPES, SUPPORTED_REFERRAL_TYPES, SUPPORTED_RETENTION_CLASSES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_SEVERITIES, SUPPORTED_SIGNAL_TYPES, SUPPORTED_SOURCE_TYPES, SUPPORTED_WATCH_TYPES, evaluate_capability_rules, get_capability_contract  # type: ignore
	from models import MonitoringAgent, MonitoringAuthority, MonitoringDissemination, MonitoringEvent, MonitoringIncident, MonitoringPolicy, MonitoringReferral, MonitoringReview, MonitoringSignal, MonitoringSource, MonitoringWatch  # type: ignore
	from monitoring_runtime import bounded_score, normalize_code, positive_int, present  # type: ignore


class RealTimeMonitoringService:
	"""Tenant-scoped real-time monitoring runtime for generated APG applications."""

	def __init__(self) -> None:
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

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def record_authority(self, authority_id: str, tenant_id: str, authority_type: str, scope_reference: str, classification: str, approver_id: str, expires_at: str, evidence_reference: str, policy_attached: bool = True) -> dict[str, Any]:
		authority_type = normalize_code(authority_type)
		classification = normalize_code(classification)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "record_authority", "authority_type_supported": authority_type in SUPPORTED_AUTHORITY_TYPES, "scope_present": present(scope_reference), "classification_supported": classification in SUPPORTED_CLASSIFICATIONS, "approver_present": present(approver_id), "expiry_present": present(expires_at), "evidence_present": present(evidence_reference)})
		item = MonitoringAuthority(authority_id, tenant_id, authority_type, scope_reference, classification, approver_id, expires_at, evidence_reference)
		self.authorities[self._tenant_key(tenant_id, authority_id)] = item
		self._audit(tenant_id, "monitoring_authority_recorded", authority_id)
		return item.to_dict()

	def record_policy(self, policy_id: str, tenant_id: str, policy_type: str, name: str, severity_floor: str, authority_id: str, evidence_reference: str) -> dict[str, Any]:
		authority = self._tenant_authority_or_none(authority_id, tenant_id)
		policy_type = normalize_code(policy_type)
		severity_floor = normalize_code(severity_floor)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_policy", "policy_type_supported": policy_type in SUPPORTED_POLICY_TYPES, "policy_name_present": present(name), "severity_supported": severity_floor in SUPPORTED_SEVERITIES, "authority_present": authority is not None, "evidence_present": present(evidence_reference)})
		item = MonitoringPolicy(policy_id, tenant_id, policy_type, name, severity_floor, authority_id, evidence_reference)
		self.policies[self._tenant_key(tenant_id, policy_id)] = item
		self._audit(tenant_id, "monitoring_policy_recorded", policy_id)
		return item.to_dict()

	def register_source(self, source_id: str, tenant_id: str, source_type: str, source_reference: str, owner_id: str, authority_id: str, access_review_reference: str, evidence_reference: str) -> dict[str, Any]:
		authority = self._tenant_authority_or_none(authority_id, tenant_id)
		source_type = normalize_code(source_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_source", "source_type_supported": source_type in SUPPORTED_SOURCE_TYPES, "source_reference_present": present(source_reference), "owner_present": present(owner_id), "authority_present": authority is not None, "access_review_present": present(access_review_reference), "evidence_present": present(evidence_reference)})
		item = MonitoringSource(source_id, tenant_id, source_type, source_reference, owner_id, authority_id, access_review_reference, evidence_reference)
		self.sources[self._tenant_key(tenant_id, source_id)] = item
		self._audit(tenant_id, "monitoring_source_registered", source_id)
		return item.to_dict()

	def record_watch(self, watch_id: str, tenant_id: str, policy_id: str, source_id: str, watch_type: str, watch_expression: str, retention_class: str, evidence_reference: str) -> dict[str, Any]:
		policy = self._tenant_policy_or_none(policy_id, tenant_id)
		source = self._tenant_source_or_none(source_id, tenant_id)
		watch_type = normalize_code(watch_type)
		retention_class = normalize_code(retention_class)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_watch", "policy_present": policy is not None, "source_present": source is not None, "policy_source_authority_match": policy is not None and source is not None and policy.authority_id == source.authority_id, "watch_type_supported": watch_type in SUPPORTED_WATCH_TYPES, "watch_expression_present": present(watch_expression), "retention_supported": retention_class in SUPPORTED_RETENTION_CLASSES, "evidence_present": present(evidence_reference)})
		item = MonitoringWatch(watch_id, tenant_id, policy_id, source_id, watch_type, watch_expression, retention_class, evidence_reference)
		self.watches[self._tenant_key(tenant_id, watch_id)] = item
		self._audit(tenant_id, "monitoring_watch_recorded", watch_id)
		return item.to_dict()

	def record_event(self, event_id: str, tenant_id: str, watch_id: str, event_type: str, event_reference: str, event_fingerprint: str, observed_at: str, confidence_score: float, evidence_reference: str) -> dict[str, Any]:
		watch = self._tenant_watch_or_none(watch_id, tenant_id)
		event_type = normalize_code(event_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_event", "watch_present": watch is not None, "event_type_supported": event_type in SUPPORTED_EVENT_TYPES, "event_reference_present": present(event_reference), "fingerprint_present": present(event_fingerprint), "observed_at_present": present(observed_at), "confidence_valid": bounded_score(confidence_score), "evidence_present": present(evidence_reference)})
		item = MonitoringEvent(event_id, tenant_id, watch_id, event_type, event_reference, event_fingerprint, observed_at, float(confidence_score), evidence_reference)
		self.events[self._tenant_key(tenant_id, event_id)] = item
		self._audit(tenant_id, "monitoring_event_recorded", event_id)
		return item.to_dict()

	def record_signal(self, signal_id: str, tenant_id: str, event_id: str, signal_type: str, severity: str, confidence_score: float, analyst_id: str, evidence_reference: str) -> dict[str, Any]:
		event = self._tenant_event_or_none(event_id, tenant_id)
		signal_type = normalize_code(signal_type)
		severity = normalize_code(severity)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_signal", "event_present": event is not None, "signal_type_supported": signal_type in SUPPORTED_SIGNAL_TYPES, "severity_supported": severity in SUPPORTED_SEVERITIES, "confidence_valid": bounded_score(confidence_score), "analyst_present": present(analyst_id), "evidence_present": present(evidence_reference)})
		item = MonitoringSignal(signal_id, tenant_id, event_id, signal_type, severity, float(confidence_score), analyst_id, evidence_reference)
		self.signals[self._tenant_key(tenant_id, signal_id)] = item
		self._audit(tenant_id, "monitoring_signal_recorded", signal_id)
		return item.to_dict()

	def record_incident(self, incident_id: str, tenant_id: str, signal_id: str, incident_type: str, severity: str, confidence_score: float, analyst_id: str, evidence_reference: str) -> dict[str, Any]:
		signal = self._tenant_signal_or_none(signal_id, tenant_id)
		incident_type = normalize_code(incident_type)
		severity = normalize_code(severity)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_incident", "signal_present": signal is not None, "incident_type_supported": incident_type in SUPPORTED_INCIDENT_TYPES, "severity_supported": severity in SUPPORTED_SEVERITIES, "confidence_valid": bounded_score(confidence_score), "analyst_present": present(analyst_id), "evidence_present": present(evidence_reference)})
		item = MonitoringIncident(incident_id, tenant_id, signal_id, incident_type, severity, float(confidence_score), analyst_id, evidence_reference)
		self.incidents[self._tenant_key(tenant_id, incident_id)] = item
		self._audit(tenant_id, "monitoring_incident_recorded", incident_id)
		return item.to_dict()

	def record_referral(self, referral_id: str, tenant_id: str, incident_id: str, referral_type: str, recipient: str, approval_reference: str, evidence_reference: str) -> dict[str, Any]:
		incident = self._tenant_incident_or_none(incident_id, tenant_id)
		referral_type = normalize_code(referral_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_referral", "incident_present": incident is not None, "referral_type_supported": referral_type in SUPPORTED_REFERRAL_TYPES, "recipient_present": present(recipient), "approval_present": present(approval_reference), "evidence_present": present(evidence_reference)})
		item = MonitoringReferral(referral_id, tenant_id, incident_id, referral_type, recipient, approval_reference, evidence_reference)
		self.referrals[self._tenant_key(tenant_id, referral_id)] = item
		self._audit(tenant_id, "monitoring_referral_recorded", referral_id)
		return item.to_dict()

	def record_dissemination(self, dissemination_id: str, tenant_id: str, incident_id: str, audience: str, release_marking: str, approval_reference: str, evidence_reference: str) -> dict[str, Any]:
		incident = self._tenant_incident_or_none(incident_id, tenant_id)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_dissemination", "incident_present": incident is not None, "audience_present": present(audience), "release_marking_present": present(release_marking), "approval_present": present(approval_reference), "evidence_present": present(evidence_reference)})
		item = MonitoringDissemination(dissemination_id, tenant_id, incident_id, audience, release_marking, approval_reference, evidence_reference)
		self.disseminations[self._tenant_key(tenant_id, dissemination_id)] = item
		self._audit(tenant_id, "monitoring_dissemination_recorded", dissemination_id)
		return item.to_dict()

	def record_review(self, review_id: str, tenant_id: str, reference_id: str, reviewer_id: str, status: str, evidence_reference: str) -> dict[str, Any]:
		status = normalize_code(status)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_review", "status_supported": status in SUPPORTED_REVIEW_STATUSES, "reviewer_present": present(reviewer_id), "evidence_present": present(evidence_reference)})
		item = MonitoringReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[self._tenant_key(tenant_id, review_id)] = item
		self._audit(tenant_id, "monitoring_review_recorded", reference_id)
		return item.to_dict()

	def register_monitoring_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_monitoring_agent", "agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES, "agent_role_supported": role in SUPPORTED_AGENT_ROLES})
		item = MonitoringAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._tenant_key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "monitoring_agent_registered", agent_id)
		return item.to_dict()

	def validate_agent_action(self, tenant_id: str, privileged_scope: bool, human_approval_recorded: bool, destructive_action_scope: bool = False, autonomous_enforcement_scope: bool = False, privacy_bypass_scope: bool = False, data_exfiltration_scope: bool = False, unauthorized_expansion_scope: bool = False, account_action_scope: bool = False, takedown_scope: bool = False) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "monitoring_agent_action", "privileged_scope": privileged_scope, "human_approval_recorded": human_approval_recorded, "destructive_action_scope": destructive_action_scope, "autonomous_enforcement_scope": autonomous_enforcement_scope, "privacy_bypass_scope": privacy_bypass_scope, "data_exfiltration_scope": data_exfiltration_scope, "unauthorized_expansion_scope": unauthorized_expansion_scope, "account_action_scope": account_action_scope, "takedown_scope": takedown_scope})
		return {"tenant_id": tenant_id, "accepted": True, "privileged_scope": privileged_scope}

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "monitoring_batch", "event_stream": event_stream})
		if not positive_int(item_count):
			raise ValueError("item_count must be positive")
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.intel.monitoring.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {"tenant_id": tenant_id, "authority_count": self._count(self.authorities, tenant_id), "policy_count": self._count(self.policies, tenant_id), "source_count": self._count(self.sources, tenant_id), "watch_count": self._count(self.watches, tenant_id), "event_count": self._count(self.events, tenant_id), "signal_count": self._count(self.signals, tenant_id), "incident_count": self._count(self.incidents, tenant_id), "referral_count": self._count(self.referrals, tenant_id), "dissemination_count": self._count(self.disseminations, tenant_id), "review_count": self._count(self.reviews, tenant_id), "agent_count": self._count(self.agents, tenant_id), "audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id), "streaming": get_capability_contract(tenant_id)["streaming"]}

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
		self.audit_events.append({"tenant_id": tenant_id, "event_type": event_type, "reference_id": reference_id, "processor": "bytewax"})

	def _count(self, items: dict[tuple[str, str], Any], tenant_id: str) -> int:
		return sum(1 for item in items.values() if item.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", action.get("rule", "monitoring_policy_denied")) for action in result["actions"])
		raise PermissionError(reasons or "monitoring_policy_denied")


IntelMonitoringService = RealTimeMonitoringService
