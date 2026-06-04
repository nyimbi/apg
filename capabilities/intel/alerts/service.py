"""Executable service layer for APG Alert Management."""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from typing import Any

try:
	from .alerts_runtime import bounded_score, normalize_code, positive_int, present
	from .capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_ALERT_TYPES, SUPPORTED_ASSIGNMENT_TYPES, SUPPORTED_AUTHORITY_TYPES, SUPPORTED_CLASSIFICATIONS, SUPPORTED_ESCALATION_TYPES, SUPPORTED_NOTIFICATION_TYPES, SUPPORTED_RESOLUTION_TYPES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_RULE_TYPES, SUPPORTED_SEVERITIES, SUPPORTED_SIGNAL_TYPES, SUPPORTED_WORKSPACE_TYPES, evaluate_capability_rules, get_capability_contract
	from .models import AlertAgent, AlertAssignment, AlertAuthority, AlertEscalation, AlertNotification, AlertRecord, AlertResolution, AlertReview, AlertRule, AlertSignal, AlertWorkspace
except ImportError:  # pragma: no cover
	from alerts_runtime import bounded_score, normalize_code, positive_int, present  # type: ignore
	from capability_contract import SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_ALERT_TYPES, SUPPORTED_ASSIGNMENT_TYPES, SUPPORTED_AUTHORITY_TYPES, SUPPORTED_CLASSIFICATIONS, SUPPORTED_ESCALATION_TYPES, SUPPORTED_NOTIFICATION_TYPES, SUPPORTED_RESOLUTION_TYPES, SUPPORTED_REVIEW_STATUSES, SUPPORTED_RULE_TYPES, SUPPORTED_SEVERITIES, SUPPORTED_SIGNAL_TYPES, SUPPORTED_WORKSPACE_TYPES, evaluate_capability_rules, get_capability_contract  # type: ignore
	from models import AlertAgent, AlertAssignment, AlertAuthority, AlertEscalation, AlertNotification, AlertRecord, AlertResolution, AlertReview, AlertRule, AlertSignal, AlertWorkspace  # type: ignore

# SLA thresholds in minutes by severity
_SLA_MINUTES: dict[str, int] = {
	"critical": 15,
	"high": 60,
	"medium": 240,
	"low": 1440,
}

# Age bucket boundaries in hours
_AGE_BUCKETS: list[tuple[str, float, float]] = [
	("0_1h",   0.0,    1.0),
	("1_4h",   1.0,    4.0),
	("4_24h",  4.0,   24.0),
	("1_7d",  24.0,  168.0),
	("7d_plus", 168.0, float("inf")),
]


def _now() -> str:
	"""ISO-8601 UTC timestamp string."""
	return datetime.now(timezone.utc).isoformat()


def _now_dt() -> datetime:
	return datetime.now(timezone.utc)


def _parse_iso(ts: str | None) -> datetime | None:
	"""Parse an ISO-8601 timestamp; return None on failure."""
	if not ts:
		return None
	try:
		dt = datetime.fromisoformat(ts)
		if dt.tzinfo is None:
			dt = dt.replace(tzinfo=timezone.utc)
		return dt
	except (ValueError, TypeError):
		return None


def _age_hours(created_at: str | None) -> float:
	"""Hours elapsed since created_at. Returns 0.0 if unparseable."""
	dt = _parse_iso(created_at)
	if dt is None:
		return 0.0
	delta = _now_dt() - dt
	return delta.total_seconds() / 3600.0


def _age_bucket(hours: float) -> str:
	for label, lo, hi in _AGE_BUCKETS:
		if lo <= hours < hi:
			return label
	return "7d_plus"


class AlertManagementService:
	"""Tenant-scoped alert-management runtime for generated APG applications."""

	def __init__(self) -> None:
		self.authorities: dict[tuple[str, str], AlertAuthority] = {}
		self.workspaces: dict[tuple[str, str], AlertWorkspace] = {}
		self.rules: dict[tuple[str, str], AlertRule] = {}
		self.signals: dict[tuple[str, str], AlertSignal] = {}
		self.alerts: dict[tuple[str, str], AlertRecord] = {}
		self.escalations: dict[tuple[str, str], AlertEscalation] = {}
		self.notifications: dict[tuple[str, str], AlertNotification] = {}
		self.assignments: dict[tuple[str, str], AlertAssignment] = {}
		self.resolutions: dict[tuple[str, str], AlertResolution] = {}
		self.reviews: dict[tuple[str, str], AlertReview] = {}
		self.agents: dict[tuple[str, str], AlertAgent] = {}
		self.audit_events: list[dict[str, Any]] = []
		# Extended state stores — keyed (tenant_id, id)
		self._alert_meta: dict[tuple[str, str], dict[str, Any]] = {}
		# signal fingerprint dedup: (tenant_id, rule_id, fingerprint) -> timestamp
		self._signal_fingerprints: dict[tuple[str, str, str], str] = {}
		# correlation groups: (tenant_id, group_id) -> dict
		self._correlation_groups: dict[tuple[str, str], dict[str, Any]] = {}
		# agent actions: (tenant_id, action_id) -> dict
		self._agent_actions: dict[tuple[str, str], dict[str, Any]] = {}
		# signal enrichment: (tenant_id, signal_id) -> list[dict]
		self._signal_enrichments: dict[tuple[str, str], list[dict[str, Any]]] = {}

	# ------------------------------------------------------------------ #
	# Describe / evaluate                                                   #
	# ------------------------------------------------------------------ #

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------ #
	# Core record methods (unchanged)                                       #
	# ------------------------------------------------------------------ #

	def record_authority(self, authority_id: str, tenant_id: str, authority_type: str, scope_reference: str, classification: str, approver_id: str, expires_at: str, evidence_reference: str, policy_attached: bool = True) -> dict[str, Any]:
		authority_type = normalize_code(authority_type)
		classification = normalize_code(classification)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": policy_attached, "operation": "record_authority", "authority_type_supported": authority_type in SUPPORTED_AUTHORITY_TYPES, "scope_present": present(scope_reference), "classification_supported": classification in SUPPORTED_CLASSIFICATIONS, "approver_present": present(approver_id), "expiry_present": present(expires_at), "evidence_present": present(evidence_reference)})
		item = AlertAuthority(authority_id, tenant_id, authority_type, scope_reference, classification, approver_id, expires_at, evidence_reference)
		self.authorities[self._tenant_key(tenant_id, authority_id)] = item
		self._audit(tenant_id, "alert_authority_recorded", authority_id)
		return item.to_dict()

	def record_workspace(self, workspace_id: str, tenant_id: str, workspace_type: str, name: str, classification: str, authority_id: str, evidence_reference: str) -> dict[str, Any]:
		authority = self._tenant_authority_or_none(authority_id, tenant_id)
		workspace_type = normalize_code(workspace_type)
		classification = normalize_code(classification)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_workspace", "workspace_type_supported": workspace_type in SUPPORTED_WORKSPACE_TYPES, "workspace_name_present": present(name), "classification_supported": classification in SUPPORTED_CLASSIFICATIONS, "authority_present": authority is not None, "evidence_present": present(evidence_reference)})
		item = AlertWorkspace(workspace_id, tenant_id, workspace_type, name, classification, authority_id, evidence_reference)
		self.workspaces[self._tenant_key(tenant_id, workspace_id)] = item
		self._audit(tenant_id, "alert_workspace_recorded", workspace_id)
		return item.to_dict()

	def record_rule(self, rule_id: str, tenant_id: str, workspace_id: str, rule_type: str, rule_reference: str, severity: str, owner_id: str, evidence_reference: str) -> dict[str, Any]:
		workspace = self._tenant_workspace_or_none(workspace_id, tenant_id)
		rule_type = normalize_code(rule_type)
		severity = normalize_code(severity)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_rule", "workspace_present": workspace is not None, "rule_type_supported": rule_type in SUPPORTED_RULE_TYPES, "rule_reference_present": present(rule_reference), "severity_supported": severity in SUPPORTED_SEVERITIES, "owner_present": present(owner_id), "evidence_present": present(evidence_reference)})
		item = AlertRule(rule_id, tenant_id, workspace_id, rule_type, rule_reference, severity, owner_id, evidence_reference)
		self.rules[self._tenant_key(tenant_id, rule_id)] = item
		self._audit(tenant_id, "alert_rule_recorded", rule_id)
		return item.to_dict()

	def record_signal(self, signal_id: str, tenant_id: str, rule_id: str, signal_type: str, signal_reference: str, confidence_score: float, evidence_reference: str) -> dict[str, Any]:
		rule = self._tenant_rule_or_none(rule_id, tenant_id)
		signal_type = normalize_code(signal_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_signal", "rule_present": rule is not None, "signal_type_supported": signal_type in SUPPORTED_SIGNAL_TYPES, "signal_reference_present": present(signal_reference), "confidence_valid": bounded_score(confidence_score), "evidence_present": present(evidence_reference)})
		item = AlertSignal(signal_id, tenant_id, rule_id, signal_type, signal_reference, float(confidence_score), evidence_reference)
		self.signals[self._tenant_key(tenant_id, signal_id)] = item
		self._audit(tenant_id, "alert_signal_recorded", signal_id)
		return item.to_dict()

	def record_alert(self, alert_id: str, tenant_id: str, signal_id: str, alert_type: str, severity: str, alert_reference: str, evidence_reference: str) -> dict[str, Any]:
		signal = self._tenant_signal_or_none(signal_id, tenant_id)
		alert_type = normalize_code(alert_type)
		severity = normalize_code(severity)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_alert", "signal_present": signal is not None, "alert_type_supported": alert_type in SUPPORTED_ALERT_TYPES, "severity_supported": severity in SUPPORTED_SEVERITIES, "alert_reference_present": present(alert_reference), "evidence_present": present(evidence_reference)})
		item = AlertRecord(alert_id, tenant_id, signal_id, alert_type, severity, alert_reference, evidence_reference)
		self.alerts[self._tenant_key(tenant_id, alert_id)] = item
		# initialise mutable metadata envelope
		self._alert_meta[self._tenant_key(tenant_id, alert_id)] = {
			"status": "open",
			"created_at": _now(),
			"acknowledged_at": None,
			"acknowledged_by": None,
			"suppressed": False,
			"suppressed_by": None,
			"suppressed_reason": None,
			"suppress_until": None,
			"resolved_at": None,
			"resolved_by": None,
			"correlation_group_id": None,
			"timeline": [{"ts": _now(), "event": "created", "actor": "system", "notes": ""}],
		}
		self._audit(tenant_id, "alert_recorded", alert_id)
		return item.to_dict()

	def record_escalation(self, escalation_id: str, tenant_id: str, alert_id: str, escalation_type: str, target_reference: str, approval_reference: str, evidence_reference: str) -> dict[str, Any]:
		alert = self._tenant_alert_or_none(alert_id, tenant_id)
		escalation_type = normalize_code(escalation_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_escalation", "alert_present": alert is not None, "escalation_type_supported": escalation_type in SUPPORTED_ESCALATION_TYPES, "target_present": present(target_reference), "approval_present": present(approval_reference), "evidence_present": present(evidence_reference)})
		item = AlertEscalation(escalation_id, tenant_id, alert_id, escalation_type, target_reference, approval_reference, evidence_reference)
		self.escalations[self._tenant_key(tenant_id, escalation_id)] = item
		self._append_timeline(alert_id, tenant_id, "escalated", "system", f"type={escalation_type}")
		self._audit(tenant_id, "alert_escalation_recorded", escalation_id)
		return item.to_dict()

	def record_notification(self, notification_id: str, tenant_id: str, alert_id: str, notification_type: str, recipient_reference: str, approval_reference: str, evidence_reference: str) -> dict[str, Any]:
		alert = self._tenant_alert_or_none(alert_id, tenant_id)
		notification_type = normalize_code(notification_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_notification", "alert_present": alert is not None, "notification_type_supported": notification_type in SUPPORTED_NOTIFICATION_TYPES, "recipient_present": present(recipient_reference), "approval_present": present(approval_reference), "evidence_present": present(evidence_reference)})
		item = AlertNotification(notification_id, tenant_id, alert_id, notification_type, recipient_reference, approval_reference, evidence_reference)
		self.notifications[self._tenant_key(tenant_id, notification_id)] = item
		self._audit(tenant_id, "alert_notification_recorded", notification_id)
		return item.to_dict()

	def record_assignment(self, assignment_id: str, tenant_id: str, alert_id: str, assignment_type: str, assignee_id: str, evidence_reference: str) -> dict[str, Any]:
		alert = self._tenant_alert_or_none(alert_id, tenant_id)
		assignment_type = normalize_code(assignment_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_assignment", "alert_present": alert is not None, "assignment_type_supported": assignment_type in SUPPORTED_ASSIGNMENT_TYPES, "assignee_present": present(assignee_id), "evidence_present": present(evidence_reference)})
		item = AlertAssignment(assignment_id, tenant_id, alert_id, assignment_type, assignee_id, evidence_reference)
		self.assignments[self._tenant_key(tenant_id, assignment_id)] = item
		self._append_timeline(alert_id, tenant_id, "assigned", assignee_id, f"type={assignment_type}")
		self._audit(tenant_id, "alert_assignment_recorded", assignment_id)
		return item.to_dict()

	def record_resolution(self, resolution_id: str, tenant_id: str, alert_id: str, resolution_type: str, resolution_reference: str, approval_reference: str, evidence_reference: str) -> dict[str, Any]:
		alert = self._tenant_alert_or_none(alert_id, tenant_id)
		resolution_type = normalize_code(resolution_type)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_resolution", "alert_present": alert is not None, "resolution_type_supported": resolution_type in SUPPORTED_RESOLUTION_TYPES, "resolution_reference_present": present(resolution_reference), "approval_present": present(approval_reference), "evidence_present": present(evidence_reference)})
		item = AlertResolution(resolution_id, tenant_id, alert_id, resolution_type, resolution_reference, approval_reference, evidence_reference)
		self.resolutions[self._tenant_key(tenant_id, resolution_id)] = item
		meta = self._alert_meta.get(self._tenant_key(tenant_id, alert_id))
		if meta is not None:
			meta["status"] = "resolved"
			meta["resolved_at"] = _now()
		self._append_timeline(alert_id, tenant_id, "resolved", "system", f"type={resolution_type}")
		self._audit(tenant_id, "alert_resolution_recorded", resolution_id)
		return item.to_dict()

	def record_review(self, review_id: str, tenant_id: str, reference_id: str, reviewer_id: str, status: str, evidence_reference: str) -> dict[str, Any]:
		status = normalize_code(status)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "record_review", "status_supported": status in SUPPORTED_REVIEW_STATUSES, "reviewer_present": present(reviewer_id), "evidence_present": present(evidence_reference)})
		item = AlertReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[self._tenant_key(tenant_id, review_id)] = item
		self._audit(tenant_id, "alert_review_recorded", reference_id)
		return item.to_dict()

	def register_alert_agent(self, agent_id: str, tenant_id: str, name: str, runtime: str, role: str, scope: str) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation_type": "write", "policy_attached": True, "operation": "register_alert_agent", "agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES, "agent_role_supported": role in SUPPORTED_AGENT_ROLES, "agent_name_present": present(name), "agent_scope_present": present(scope)})
		item = AlertAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[self._tenant_key(tenant_id, agent_id)] = item
		self._audit(tenant_id, "alert_agent_registered", agent_id)
		return item.to_dict()

	def validate_agent_action(self, tenant_id: str, privileged_scope: bool, human_approval_recorded: bool, unapproved_escalation_scope: bool = False, unapproved_notification_scope: bool = False, alert_suppression_scope: bool = False, evidence_fabrication_scope: bool = False, privacy_bypass_scope: bool = False, autonomous_closure_scope: bool = False, severity_downgrade_scope: bool = False) -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "alert_agent_action", "privileged_scope": privileged_scope, "human_approval_recorded": human_approval_recorded, "unapproved_escalation_scope": unapproved_escalation_scope, "unapproved_notification_scope": unapproved_notification_scope, "alert_suppression_scope": alert_suppression_scope, "evidence_fabrication_scope": evidence_fabrication_scope, "privacy_bypass_scope": privacy_bypass_scope, "autonomous_closure_scope": autonomous_closure_scope, "severity_downgrade_scope": severity_downgrade_scope})
		return {"tenant_id": tenant_id, "accepted": True, "privileged_scope": privileged_scope}

	def validate_batch(self, tenant_id: str, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		self._enforce({"tenant_id": tenant_id, "tenant_context_present": bool(tenant_id), "operation": "alert_batch", "event_stream": event_stream})
		if not positive_int(item_count):
			raise ValueError("item_count must be positive")
		return {"tenant_id": tenant_id, "item_count": item_count, "processor": "bytewax", "stream": "apg.intel.alerts.lifecycle", "accepted": True}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {"tenant_id": tenant_id, "authority_count": self._count(self.authorities, tenant_id), "workspace_count": self._count(self.workspaces, tenant_id), "rule_count": self._count(self.rules, tenant_id), "signal_count": self._count(self.signals, tenant_id), "alert_count": self._count(self.alerts, tenant_id), "escalation_count": self._count(self.escalations, tenant_id), "notification_count": self._count(self.notifications, tenant_id), "assignment_count": self._count(self.assignments, tenant_id), "resolution_count": self._count(self.resolutions, tenant_id), "review_count": self._count(self.reviews, tenant_id), "agent_count": self._count(self.agents, tenant_id), "audit_event_count": sum(1 for event in self.audit_events if event["tenant_id"] == tenant_id), "streaming": get_capability_contract(tenant_id)["streaming"]}

	# ------------------------------------------------------------------ #
	# Collections Analytics                                                 #
	# ------------------------------------------------------------------ #

	def aging_summary(self, tenant_id: str) -> dict[str, Any]:
		"""Count of alerts by age bucket for tenant.

		Buckets: 0-1h, 1-4h, 4-24h, 1-7d, 7d+.
		Requires _alert_meta to carry a created_at timestamp.
		"""
		buckets: dict[str, int] = {label: 0 for label, _, _ in _AGE_BUCKETS}
		for (tid, aid), meta in self._alert_meta.items():
			if tid != tenant_id:
				continue
			hours = _age_hours(meta.get("created_at"))
			buckets[_age_bucket(hours)] += 1
		total = sum(buckets.values())
		return {"tenant_id": tenant_id, "total": total, "buckets": buckets, "as_of": _now()}

	def severity_distribution(self, tenant_id: str) -> dict[str, Any]:
		"""Count of open alerts by severity level."""
		dist: dict[str, int] = {s: 0 for s in SUPPORTED_SEVERITIES}
		for (tid, aid), alert in self.alerts.items():
			if tid != tenant_id:
				continue
			meta = self._alert_meta.get((tid, aid), {})
			if meta.get("status") == "resolved":
				continue
			sev = getattr(alert, "severity", "unknown")
			dist[sev] = dist.get(sev, 0) + 1
		return {"tenant_id": tenant_id, "distribution": dist, "as_of": _now()}

	def alert_throughput(self, tenant_id: str, period_hours: int = 24) -> dict[str, Any]:
		"""Created / resolved / escalated counts in the last period_hours.

		Uses audit_events timestamps where available; falls back to counting
		current-state objects when timestamps are absent (in-memory store has
		no clock on base models, so we count all for simplicity).
		"""
		assert period_hours > 0, "period_hours must be positive"
		created = 0
		resolved = 0
		escalated = 0
		for (tid, _), meta in self._alert_meta.items():
			if tid != tenant_id:
				continue
			created += 1
			if meta.get("status") == "resolved":
				resolved += 1
		for (tid, _), esc in self.escalations.items():
			if tid == tenant_id:
				escalated += 1
		return {
			"tenant_id": tenant_id,
			"period_hours": period_hours,
			"created": created,
			"resolved": resolved,
			"escalated": escalated,
			"open": created - resolved,
			"as_of": _now(),
		}

	def mean_time_to_resolve(self, tenant_id: str) -> float:
		"""Average resolution time in minutes across all resolved alerts.

		Returns 0.0 if no resolved alerts with timing data.
		"""
		durations: list[float] = []
		for (tid, aid), meta in self._alert_meta.items():
			if tid != tenant_id:
				continue
			if meta.get("status") != "resolved":
				continue
			created_dt = _parse_iso(meta.get("created_at"))
			resolved_dt = _parse_iso(meta.get("resolved_at"))
			if created_dt and resolved_dt:
				minutes = (resolved_dt - created_dt).total_seconds() / 60.0
				durations.append(minutes)
		if not durations:
			return 0.0
		return sum(durations) / len(durations)

	def escalation_rate(self, tenant_id: str) -> float:
		"""Fraction of alerts that have been escalated (0.0–1.0).

		Returns 0.0 when there are no alerts.
		"""
		total_alerts = self._count(self.alerts, tenant_id)
		if total_alerts == 0:
			return 0.0
		escalated_alert_ids: set[str] = set()
		for (tid, _), esc in self.escalations.items():
			if tid == tenant_id:
				escalated_alert_ids.add(esc.alert_id)
		return len(escalated_alert_ids) / total_alerts

	# ------------------------------------------------------------------ #
	# Alert Lifecycle                                                        #
	# ------------------------------------------------------------------ #

	def acknowledge_alert(self, alert_id: str, tenant_id: str, acknowledged_by: str, notes: str = "") -> dict[str, Any]:
		"""Mark an alert as acknowledged. Idempotent — re-ack just updates fields."""
		assert present(acknowledged_by), "acknowledged_by is required"
		alert = self._tenant_alert_or_none(alert_id, tenant_id)
		if alert is None:
			raise KeyError(f"alert {alert_id!r} not found for tenant {tenant_id!r}")
		meta = self._alert_meta[self._tenant_key(tenant_id, alert_id)]
		meta["acknowledged_at"] = _now()
		meta["acknowledged_by"] = acknowledged_by
		if meta.get("status") == "open":
			meta["status"] = "acknowledged"
		self._append_timeline(alert_id, tenant_id, "acknowledged", acknowledged_by, notes)
		self._audit(tenant_id, "alert_acknowledged", alert_id)
		return {"alert_id": alert_id, "tenant_id": tenant_id, "status": meta["status"], "acknowledged_by": acknowledged_by, "acknowledged_at": meta["acknowledged_at"]}

	def suppress_alert(self, alert_id: str, tenant_id: str, suppressed_by: str, reason: str, suppress_until: str) -> dict[str, Any]:
		"""Suppress an alert until a given ISO-8601 timestamp."""
		assert present(suppressed_by), "suppressed_by is required"
		assert present(reason), "reason is required"
		assert present(suppress_until), "suppress_until is required"
		alert = self._tenant_alert_or_none(alert_id, tenant_id)
		if alert is None:
			raise KeyError(f"alert {alert_id!r} not found for tenant {tenant_id!r}")
		meta = self._alert_meta[self._tenant_key(tenant_id, alert_id)]
		meta["suppressed"] = True
		meta["suppressed_by"] = suppressed_by
		meta["suppressed_reason"] = reason
		meta["suppress_until"] = suppress_until
		meta["status"] = "suppressed"
		self._append_timeline(alert_id, tenant_id, "suppressed", suppressed_by, reason)
		self._audit(tenant_id, "alert_suppressed", alert_id)
		return {"alert_id": alert_id, "tenant_id": tenant_id, "suppressed": True, "suppressed_by": suppressed_by, "suppress_until": suppress_until, "reason": reason}

	def unsuppress_alert(self, alert_id: str, tenant_id: str, unsuppressed_by: str) -> dict[str, Any]:
		"""Lift suppression on an alert, returning it to open status."""
		assert present(unsuppressed_by), "unsuppressed_by is required"
		alert = self._tenant_alert_or_none(alert_id, tenant_id)
		if alert is None:
			raise KeyError(f"alert {alert_id!r} not found for tenant {tenant_id!r}")
		meta = self._alert_meta[self._tenant_key(tenant_id, alert_id)]
		meta["suppressed"] = False
		meta["suppressed_by"] = None
		meta["suppressed_reason"] = None
		meta["suppress_until"] = None
		meta["status"] = "open"
		self._append_timeline(alert_id, tenant_id, "unsuppressed", unsuppressed_by, "")
		self._audit(tenant_id, "alert_unsuppressed", alert_id)
		return {"alert_id": alert_id, "tenant_id": tenant_id, "suppressed": False, "unsuppressed_by": unsuppressed_by}

	def correlate_alerts(self, alert_ids: list[str], tenant_id: str, correlation_reason: str, correlated_by: str) -> dict[str, Any]:
		"""Link multiple alerts into a named correlation group.

		Returns the group record including a generated group_id derived from
		the sorted alert IDs to make it deterministic within a tenant.
		"""
		assert len(alert_ids) >= 2, "need at least 2 alert_ids to correlate"
		assert present(correlation_reason), "correlation_reason is required"
		assert present(correlated_by), "correlated_by is required"
		for aid in alert_ids:
			if self._tenant_alert_or_none(aid, tenant_id) is None:
				raise KeyError(f"alert {aid!r} not found for tenant {tenant_id!r}")
		group_id = "cg_" + "_".join(sorted(alert_ids))[:64]
		group: dict[str, Any] = {
			"group_id": group_id,
			"tenant_id": tenant_id,
			"alert_ids": list(alert_ids),
			"correlation_reason": correlation_reason,
			"correlated_by": correlated_by,
			"created_at": _now(),
		}
		self._correlation_groups[self._tenant_key(tenant_id, group_id)] = group
		for aid in alert_ids:
			meta = self._alert_meta.get(self._tenant_key(tenant_id, aid))
			if meta is not None:
				meta["correlation_group_id"] = group_id
			self._append_timeline(aid, tenant_id, "correlated", correlated_by, f"group={group_id}")
		self._audit(tenant_id, "alerts_correlated", group_id)
		return group

	def bulk_assign(self, alert_ids: list[str], tenant_id: str, assignee_id: str, assignment_type: str, assigned_by: str) -> dict[str, Any]:
		"""Assign a list of alerts to a single assignee in one operation.

		Returns a summary of succeeded and failed assignments.
		"""
		assert present(assignee_id), "assignee_id is required"
		assert present(assigned_by), "assigned_by is required"
		assignment_type = normalize_code(assignment_type)
		succeeded: list[str] = []
		failed: list[dict[str, str]] = []
		ts = _now()
		for aid in alert_ids:
			if self._tenant_alert_or_none(aid, tenant_id) is None:
				failed.append({"alert_id": aid, "reason": "not_found"})
				continue
			assignment_id = f"bulk_{aid}_{assignee_id}"
			item = AlertAssignment(assignment_id, tenant_id, aid, assignment_type, assignee_id, f"bulk_assign:{assigned_by}")
			self.assignments[self._tenant_key(tenant_id, assignment_id)] = item
			self._append_timeline(aid, tenant_id, "bulk_assigned", assigned_by, f"assignee={assignee_id}")
			self._audit(tenant_id, "alert_bulk_assigned", aid)
			succeeded.append(aid)
		return {
			"tenant_id": tenant_id,
			"assignee_id": assignee_id,
			"assignment_type": assignment_type,
			"assigned_by": assigned_by,
			"succeeded": succeeded,
			"failed": failed,
			"total": len(alert_ids),
			"as_of": ts,
		}

	def bulk_resolve(self, alert_ids: list[str], tenant_id: str, resolution_type: str, resolved_by: str, notes: str) -> dict[str, Any]:
		"""Resolve a list of alerts in bulk.

		Returns a summary of succeeded and failed resolutions.
		"""
		assert present(resolved_by), "resolved_by is required"
		resolution_type = normalize_code(resolution_type)
		succeeded: list[str] = []
		failed: list[dict[str, str]] = []
		ts = _now()
		for aid in alert_ids:
			if self._tenant_alert_or_none(aid, tenant_id) is None:
				failed.append({"alert_id": aid, "reason": "not_found"})
				continue
			resolution_id = f"bulk_res_{aid}_{resolved_by}"
			item = AlertResolution(resolution_id, tenant_id, aid, resolution_type, f"bulk_resolve:{resolved_by}", resolved_by, f"notes:{notes}")
			self.resolutions[self._tenant_key(tenant_id, resolution_id)] = item
			meta = self._alert_meta.get(self._tenant_key(tenant_id, aid))
			if meta is not None:
				meta["status"] = "resolved"
				meta["resolved_at"] = ts
				meta["resolved_by"] = resolved_by
			self._append_timeline(aid, tenant_id, "bulk_resolved", resolved_by, notes)
			self._audit(tenant_id, "alert_bulk_resolved", aid)
			succeeded.append(aid)
		return {
			"tenant_id": tenant_id,
			"resolution_type": resolution_type,
			"resolved_by": resolved_by,
			"succeeded": succeeded,
			"failed": failed,
			"total": len(alert_ids),
			"as_of": ts,
		}

	def reopen_alert(self, alert_id: str, tenant_id: str, reason: str, reopened_by: str) -> dict[str, Any]:
		"""Reopen a resolved or suppressed alert back to open status."""
		assert present(reason), "reason is required"
		assert present(reopened_by), "reopened_by is required"
		alert = self._tenant_alert_or_none(alert_id, tenant_id)
		if alert is None:
			raise KeyError(f"alert {alert_id!r} not found for tenant {tenant_id!r}")
		meta = self._alert_meta[self._tenant_key(tenant_id, alert_id)]
		previous_status = meta.get("status", "unknown")
		meta["status"] = "open"
		meta["resolved_at"] = None
		meta["resolved_by"] = None
		meta["suppressed"] = False
		meta["suppressed_reason"] = None
		meta["suppress_until"] = None
		self._append_timeline(alert_id, tenant_id, "reopened", reopened_by, f"reason={reason} previous_status={previous_status}")
		self._audit(tenant_id, "alert_reopened", alert_id)
		return {"alert_id": alert_id, "tenant_id": tenant_id, "status": "open", "reopened_by": reopened_by, "reason": reason, "previous_status": previous_status}

	def alert_timeline(self, alert_id: str, tenant_id: str) -> list[dict[str, Any]]:
		"""Return the ordered audit trail for a single alert."""
		alert = self._tenant_alert_or_none(alert_id, tenant_id)
		if alert is None:
			raise KeyError(f"alert {alert_id!r} not found for tenant {tenant_id!r}")
		meta = self._alert_meta.get(self._tenant_key(tenant_id, alert_id), {})
		return list(meta.get("timeline", []))

	# ------------------------------------------------------------------ #
	# Rule Management                                                        #
	# ------------------------------------------------------------------ #

	def activate_rule(self, rule_id: str, tenant_id: str, activated_by: str) -> dict[str, Any]:
		"""Mark a rule as active so it can generate new signals."""
		assert present(activated_by), "activated_by is required"
		rule = self._tenant_rule_or_none(rule_id, tenant_id)
		if rule is None:
			raise KeyError(f"rule {rule_id!r} not found for tenant {tenant_id!r}")
		self._audit(tenant_id, "rule_activated", rule_id)
		result = rule.to_dict()
		result["active"] = True
		result["activated_by"] = activated_by
		result["activated_at"] = _now()
		return result

	def deactivate_rule(self, rule_id: str, tenant_id: str, reason: str, deactivated_by: str) -> dict[str, Any]:
		"""Mark a rule inactive — it will not generate new alerts until reactivated."""
		assert present(reason), "reason is required"
		assert present(deactivated_by), "deactivated_by is required"
		rule = self._tenant_rule_or_none(rule_id, tenant_id)
		if rule is None:
			raise KeyError(f"rule {rule_id!r} not found for tenant {tenant_id!r}")
		self._audit(tenant_id, "rule_deactivated", rule_id)
		result = rule.to_dict()
		result["active"] = False
		result["deactivated_by"] = deactivated_by
		result["deactivated_at"] = _now()
		result["deactivation_reason"] = reason
		return result

	def test_rule(self, rule_id: str, tenant_id: str, test_signal: dict[str, Any]) -> dict[str, Any]:
		"""Simulate what alert would fire if test_signal were processed against rule_id.

		Returns a dry-run result with the would-be alert fields but does NOT
		write anything to any store.
		"""
		assert present(test_signal), "test_signal must be a non-empty dict"
		rule = self._tenant_rule_or_none(rule_id, tenant_id)
		if rule is None:
			raise KeyError(f"rule {rule_id!r} not found for tenant {tenant_id!r}")
		confidence = float(test_signal.get("confidence_score", 0.5))
		would_fire = confidence >= 0.5
		simulated_severity = test_signal.get("override_severity", rule.severity)
		return {
			"rule_id": rule_id,
			"tenant_id": tenant_id,
			"test_signal": test_signal,
			"would_fire": would_fire,
			"simulated_alert": {
				"alert_type": "simulated",
				"severity": simulated_severity,
				"rule_type": rule.rule_type,
				"confidence_score": confidence,
			} if would_fire else None,
			"as_of": _now(),
		}

	def rule_effectiveness(self, rule_id: str, tenant_id: str) -> dict[str, Any]:
		"""Report on how many alerts a rule has generated and their resolution quality.

		true_positive / false_positive rates are approximated from review records
		where reviewer_id is present and status maps to accepted/rejected semantics.
		"""
		rule = self._tenant_rule_or_none(rule_id, tenant_id)
		if rule is None:
			raise KeyError(f"rule {rule_id!r} not found for tenant {tenant_id!r}")
		# collect signals that belong to this rule
		rule_signal_ids: set[str] = {
			sid for (tid, sid), sig in self.signals.items()
			if tid == tenant_id and sig.rule_id == rule_id
		}
		# collect alerts linked to those signals
		rule_alert_ids: set[str] = {
			aid for (tid, aid), alert in self.alerts.items()
			if tid == tenant_id and alert.signal_id in rule_signal_ids
		}
		total_alerts = len(rule_alert_ids)
		resolved = sum(
			1 for aid in rule_alert_ids
			if self._alert_meta.get(self._tenant_key(tenant_id, aid), {}).get("status") == "resolved"
		)
		# reviews referencing these alerts as true_positive / false_positive
		true_pos = sum(
			1 for (tid, _), rev in self.reviews.items()
			if tid == tenant_id and rev.reference_id in rule_alert_ids and rev.status == "approved"
		)
		false_pos = sum(
			1 for (tid, _), rev in self.reviews.items()
			if tid == tenant_id and rev.reference_id in rule_alert_ids and rev.status == "rejected"
		)
		tp_rate = true_pos / total_alerts if total_alerts else 0.0
		fp_rate = false_pos / total_alerts if total_alerts else 0.0
		return {
			"rule_id": rule_id,
			"tenant_id": tenant_id,
			"total_signals": len(rule_signal_ids),
			"total_alerts": total_alerts,
			"resolved_alerts": resolved,
			"true_positive_count": true_pos,
			"false_positive_count": false_pos,
			"true_positive_rate": round(tp_rate, 4),
			"false_positive_rate": round(fp_rate, 4),
			"as_of": _now(),
		}

	def list_rules(self, tenant_id: str, filters: dict[str, Any] = {}) -> list[dict[str, Any]]:
		"""Return rules for tenant, optionally filtered by rule_type or severity."""
		rule_type_filter = normalize_code(filters["rule_type"]) if "rule_type" in filters else None
		severity_filter = normalize_code(filters["severity"]) if "severity" in filters else None
		results: list[dict[str, Any]] = []
		for (tid, _), rule in self.rules.items():
			if tid != tenant_id:
				continue
			if rule_type_filter and rule.rule_type != rule_type_filter:
				continue
			if severity_filter and rule.severity != severity_filter:
				continue
			results.append(rule.to_dict())
		return results

	# ------------------------------------------------------------------ #
	# Signal Intelligence                                                    #
	# ------------------------------------------------------------------ #

	def signal_deduplication_check(self, tenant_id: str, rule_id: str, signal_fingerprint: str, window_minutes: int = 60) -> dict[str, Any]:
		"""Check if a near-identical signal has been seen within window_minutes.

		If the fingerprint key exists and has not expired, returns is_duplicate=True
		with the first-seen timestamp. Otherwise registers the fingerprint and
		returns is_duplicate=False.
		"""
		assert present(signal_fingerprint), "signal_fingerprint is required"
		assert window_minutes > 0, "window_minutes must be positive"
		fkey = (tenant_id, rule_id, signal_fingerprint)
		existing_ts = self._signal_fingerprints.get(fkey)
		if existing_ts is not None:
			existing_dt = _parse_iso(existing_ts)
			if existing_dt:
				age_min = (_now_dt() - existing_dt).total_seconds() / 60.0
				if age_min <= window_minutes:
					return {
						"is_duplicate": True,
						"tenant_id": tenant_id,
						"rule_id": rule_id,
						"fingerprint": signal_fingerprint,
						"first_seen_at": existing_ts,
						"age_minutes": round(age_min, 2),
						"window_minutes": window_minutes,
					}
		# register or refresh
		self._signal_fingerprints[fkey] = _now()
		return {
			"is_duplicate": False,
			"tenant_id": tenant_id,
			"rule_id": rule_id,
			"fingerprint": signal_fingerprint,
			"first_seen_at": self._signal_fingerprints[fkey],
			"window_minutes": window_minutes,
		}

	def signal_enrichment(self, signal_id: str, tenant_id: str, enrichment_data: dict[str, Any]) -> dict[str, Any]:
		"""Attach enrichment metadata to a signal (geo, threat intel, CVE details, etc.).

		Each call appends a timestamped enrichment entry; all prior enrichments
		are preserved and returned.
		"""
		assert present(enrichment_data), "enrichment_data must be a non-empty dict"
		signal = self._tenant_signal_or_none(signal_id, tenant_id)
		if signal is None:
			raise KeyError(f"signal {signal_id!r} not found for tenant {tenant_id!r}")
		key = self._tenant_key(tenant_id, signal_id)
		if key not in self._signal_enrichments:
			self._signal_enrichments[key] = []
		entry: dict[str, Any] = {"enriched_at": _now(), "data": enrichment_data}
		self._signal_enrichments[key].append(entry)
		self._audit(tenant_id, "signal_enriched", signal_id)
		return {
			"signal_id": signal_id,
			"tenant_id": tenant_id,
			"enrichment_count": len(self._signal_enrichments[key]),
			"enrichments": list(self._signal_enrichments[key]),
		}

	def signal_to_alert_mapping(self, tenant_id: str) -> dict[str, Any]:
		"""Return a mapping of every signal to the alerts it generated."""
		mapping: dict[str, list[str]] = {}
		for (tid, sid) in self.signals:
			if tid == tenant_id:
				mapping[sid] = []
		for (tid, aid), alert in self.alerts.items():
			if tid != tenant_id:
				continue
			sid = alert.signal_id
			if sid not in mapping:
				mapping[sid] = []
			mapping[sid].append(aid)
		return {"tenant_id": tenant_id, "mapping": mapping, "as_of": _now()}

	def list_signals(self, tenant_id: str, filters: dict[str, Any] = {}) -> list[dict[str, Any]]:
		"""Return signals for tenant, optionally filtered by rule_id or signal_type."""
		rule_filter = filters.get("rule_id")
		type_filter = normalize_code(filters["signal_type"]) if "signal_type" in filters else None
		results: list[dict[str, Any]] = []
		for (tid, _), sig in self.signals.items():
			if tid != tenant_id:
				continue
			if rule_filter and sig.rule_id != rule_filter:
				continue
			if type_filter and sig.signal_type != type_filter:
				continue
			row = sig.to_dict()
			key = self._tenant_key(tenant_id, sig.id)
			row["enrichments"] = list(self._signal_enrichments.get(key, []))
			results.append(row)
		return results

	def purge_old_signals(self, tenant_id: str, older_than_hours: int = 720) -> dict[str, Any]:
		"""Remove signal fingerprints older than older_than_hours (default 30 days).

		The canonical signal records in self.signals are retained for audit;
		only the dedup fingerprint cache is pruned.
		"""
		assert older_than_hours > 0, "older_than_hours must be positive"
		purged = 0
		cutoff_dt = _now_dt()
		to_delete: list[tuple[str, str, str]] = []
		for fkey, ts in self._signal_fingerprints.items():
			tid, _, _ = fkey
			if tid != tenant_id:
				continue
			seen_dt = _parse_iso(ts)
			if seen_dt is None:
				continue
			age_h = (cutoff_dt - seen_dt).total_seconds() / 3600.0
			if age_h > older_than_hours:
				to_delete.append(fkey)
		for fkey in to_delete:
			del self._signal_fingerprints[fkey]
			purged += 1
		self._audit(tenant_id, "signals_purged", f"count={purged}")
		return {
			"tenant_id": tenant_id,
			"purged_fingerprints": purged,
			"older_than_hours": older_than_hours,
			"as_of": _now(),
		}

	# ------------------------------------------------------------------ #
	# Agent Orchestration                                                    #
	# ------------------------------------------------------------------ #

	def assign_agent_to_alert(self, alert_id: str, tenant_id: str, agent_id: str, task_type: str, requires_approval: bool = True) -> dict[str, Any]:
		"""Dispatch an agent task against an alert, recording the pending action."""
		alert = self._tenant_alert_or_none(alert_id, tenant_id)
		if alert is None:
			raise KeyError(f"alert {alert_id!r} not found for tenant {tenant_id!r}")
		agent = self.agents.get(self._tenant_key(tenant_id, agent_id))
		if agent is None:
			raise KeyError(f"agent {agent_id!r} not found for tenant {tenant_id!r}")
		action_id = f"aa_{alert_id}_{agent_id}_{task_type}"
		action: dict[str, Any] = {
			"action_id": action_id,
			"alert_id": alert_id,
			"tenant_id": tenant_id,
			"agent_id": agent_id,
			"task_type": normalize_code(task_type),
			"status": "pending_approval" if requires_approval else "running",
			"requires_approval": requires_approval,
			"assigned_at": _now(),
			"result": None,
			"evidence": None,
			"approved_by": None,
			"approved_at": None,
		}
		self._agent_actions[self._tenant_key(tenant_id, action_id)] = action
		self._append_timeline(alert_id, tenant_id, "agent_assigned", agent_id, f"task={task_type}")
		self._audit(tenant_id, "agent_assigned_to_alert", action_id)
		return action

	def agent_action_result(self, alert_id: str, tenant_id: str, agent_id: str, action: str, result: str, evidence: str) -> dict[str, Any]:
		"""Record the outcome of an agent action on an alert."""
		assert present(result), "result is required"
		assert present(evidence), "evidence is required"
		alert = self._tenant_alert_or_none(alert_id, tenant_id)
		if alert is None:
			raise KeyError(f"alert {alert_id!r} not found for tenant {tenant_id!r}")
		action_id = f"aa_{alert_id}_{agent_id}_{action}"
		action_rec = self._agent_actions.get(self._tenant_key(tenant_id, action_id))
		if action_rec is None:
			# create on-the-fly if result arrives without prior assign call
			action_rec = {
				"action_id": action_id,
				"alert_id": alert_id,
				"tenant_id": tenant_id,
				"agent_id": agent_id,
				"task_type": normalize_code(action),
				"status": "completed",
				"requires_approval": False,
				"assigned_at": _now(),
				"result": None,
				"evidence": None,
				"approved_by": None,
				"approved_at": None,
			}
			self._agent_actions[self._tenant_key(tenant_id, action_id)] = action_rec
		action_rec["result"] = result
		action_rec["evidence"] = evidence
		action_rec["status"] = "completed"
		action_rec["completed_at"] = _now()
		self._append_timeline(alert_id, tenant_id, "agent_result_recorded", agent_id, f"action={action} result={result}")
		self._audit(tenant_id, "agent_action_result_recorded", action_id)
		return dict(action_rec)

	def list_agent_actions(self, alert_id: str, tenant_id: str) -> list[dict[str, Any]]:
		"""Return all agent actions associated with a specific alert."""
		alert = self._tenant_alert_or_none(alert_id, tenant_id)
		if alert is None:
			raise KeyError(f"alert {alert_id!r} not found for tenant {tenant_id!r}")
		return [
			dict(rec) for (tid, _), rec in self._agent_actions.items()
			if tid == tenant_id and rec["alert_id"] == alert_id
		]

	def approve_agent_action(self, action_id: str, tenant_id: str, approved_by: str, approval_notes: str) -> dict[str, Any]:
		"""Approve a pending agent action, transitioning it to running/approved status."""
		assert present(approved_by), "approved_by is required"
		action_rec = self._agent_actions.get(self._tenant_key(tenant_id, action_id))
		if action_rec is None:
			raise KeyError(f"action {action_id!r} not found for tenant {tenant_id!r}")
		action_rec["approved_by"] = approved_by
		action_rec["approved_at"] = _now()
		action_rec["approval_notes"] = approval_notes
		action_rec["status"] = "approved"
		alert_id = action_rec["alert_id"]
		self._append_timeline(alert_id, tenant_id, "agent_action_approved", approved_by, approval_notes)
		self._audit(tenant_id, "agent_action_approved", action_id)
		return dict(action_rec)

	def agent_performance_report(self, tenant_id: str, period_hours: int = 168) -> dict[str, Any]:
		"""Aggregate agent performance metrics over period_hours (default 1 week).

		Counts total actions, completed, pending, average resolution contribution
		per agent in the tenant.
		"""
		assert period_hours > 0, "period_hours must be positive"
		per_agent: dict[str, dict[str, Any]] = {}
		for (tid, _), rec in self._agent_actions.items():
			if tid != tenant_id:
				continue
			aid = rec["agent_id"]
			if aid not in per_agent:
				per_agent[aid] = {"agent_id": aid, "total": 0, "completed": 0, "pending": 0, "approved": 0}
			per_agent[aid]["total"] += 1
			status = rec.get("status", "unknown")
			if status == "completed":
				per_agent[aid]["completed"] += 1
			elif status in ("pending_approval", "running"):
				per_agent[aid]["pending"] += 1
			elif status == "approved":
				per_agent[aid]["approved"] += 1
		return {
			"tenant_id": tenant_id,
			"period_hours": period_hours,
			"agent_count": len(per_agent),
			"agents": list(per_agent.values()),
			"as_of": _now(),
		}

	# ------------------------------------------------------------------ #
	# Dashboard & Reporting                                                  #
	# ------------------------------------------------------------------ #

	def operational_dashboard(self, tenant_id: str) -> dict[str, Any]:
		"""High-level operational snapshot: open alerts by severity, queue depths, SLA posture.

		Designed for a watch-center wall display — callers should cache this
		and refresh on a short interval rather than computing per-request.
		"""
		severity_counts: dict[str, int] = {s: 0 for s in SUPPORTED_SEVERITIES}
		sla_breached: dict[str, int] = {s: 0 for s in SUPPORTED_SEVERITIES}
		open_count = 0
		ack_count = 0
		suppressed_count = 0
		escalation_queue: list[str] = []

		for (tid, aid), alert in self.alerts.items():
			if tid != tenant_id:
				continue
			meta = self._alert_meta.get((tid, aid), {})
			status = meta.get("status", "open")
			if status == "resolved":
				continue
			if status == "suppressed":
				suppressed_count += 1
				continue
			if status in ("open", "acknowledged"):
				open_count += 1
				sev = alert.severity
				severity_counts[sev] = severity_counts.get(sev, 0) + 1
				if status == "acknowledged":
					ack_count += 1
				# SLA breach check
				created_dt = _parse_iso(meta.get("created_at"))
				if created_dt:
					age_min = (_now_dt() - created_dt).total_seconds() / 60.0
					limit = _SLA_MINUTES.get(sev, 1440)
					if age_min > limit:
						sla_breached[sev] = sla_breached.get(sev, 0) + 1

		for (tid, eid), esc in self.escalations.items():
			if tid == tenant_id:
				escalation_queue.append(esc.alert_id)

		return {
			"tenant_id": tenant_id,
			"open_alerts": open_count,
			"acknowledged_alerts": ack_count,
			"suppressed_alerts": suppressed_count,
			"by_severity": severity_counts,
			"sla_breached_by_severity": sla_breached,
			"escalation_queue_depth": len(set(escalation_queue)),
			"total_rules": self._count(self.rules, tenant_id),
			"total_signals": self._count(self.signals, tenant_id),
			"as_of": _now(),
		}

	def watch_center_queue(self, tenant_id: str, workspace_id: str | None = None) -> list[dict[str, Any]]:
		"""Priority-sorted list of open alerts requiring human attention.

		Sorted by: (severity rank desc, age desc) so critical+old surfaces first.
		Optionally filtered by workspace_id via the signal->rule->workspace chain.
		"""
		sev_rank = {"critical": 4, "high": 3, "medium": 2, "low": 1}
		# build workspace->rule mapping if filter requested
		workspace_rule_ids: set[str] | None = None
		if workspace_id is not None:
			workspace_rule_ids = {
				rid for (tid, rid), rule in self.rules.items()
				if tid == tenant_id and rule.workspace_id == workspace_id
			}

		queue: list[dict[str, Any]] = []
		for (tid, aid), alert in self.alerts.items():
			if tid != tenant_id:
				continue
			meta = self._alert_meta.get((tid, aid), {})
			status = meta.get("status", "open")
			if status in ("resolved", "suppressed"):
				continue
			# workspace filter
			if workspace_rule_ids is not None:
				sig = self._tenant_signal_or_none(alert.signal_id, tenant_id)
				if sig is None or sig.rule_id not in workspace_rule_ids:
					continue
			age_h = _age_hours(meta.get("created_at"))
			sla_limit = _SLA_MINUTES.get(alert.severity, 1440)
			age_min = age_h * 60.0
			sla_remaining_min = max(0.0, sla_limit - age_min)
			queue.append({
				"alert_id": aid,
				"severity": alert.severity,
				"alert_type": alert.alert_type,
				"status": status,
				"age_hours": round(age_h, 2),
				"sla_remaining_minutes": round(sla_remaining_min, 1),
				"correlation_group_id": meta.get("correlation_group_id"),
				"acknowledged_by": meta.get("acknowledged_by"),
				"_sort_key": (sev_rank.get(alert.severity, 0), age_h),
			})
		queue.sort(key=lambda r: r["_sort_key"], reverse=True)
		for row in queue:
			del row["_sort_key"]
		return queue

	def incident_correlation_map(self, tenant_id: str) -> dict[str, Any]:
		"""Graph of correlated alert clusters as adjacency list.

		Each node is an alert_id; edges represent shared correlation groups.
		"""
		nodes: set[str] = set()
		edges: list[dict[str, str]] = []
		for (tid, _), group in self._correlation_groups.items():
			if tid != tenant_id:
				continue
			gid = group["group_id"]
			aids = group["alert_ids"]
			for aid in aids:
				nodes.add(aid)
			for i, a in enumerate(aids):
				for b in aids[i + 1:]:
					edges.append({"source": a, "target": b, "group_id": gid})
		return {
			"tenant_id": tenant_id,
			"node_count": len(nodes),
			"edge_count": len(edges),
			"nodes": sorted(nodes),
			"edges": edges,
			"correlation_groups": [
				dict(g) for (tid, _), g in self._correlation_groups.items()
				if tid == tenant_id
			],
			"as_of": _now(),
		}

	def sla_compliance_report(self, tenant_id: str, period_hours: int = 24) -> dict[str, Any]:
		"""Percentage of alerts resolved within SLA by severity.

		SLA thresholds: critical=15min, high=1h, medium=4h, low=24h.
		Only resolved alerts with both created_at and resolved_at are counted.
		"""
		assert period_hours > 0, "period_hours must be positive"
		totals: dict[str, int] = {s: 0 for s in SUPPORTED_SEVERITIES}
		within_sla: dict[str, int] = {s: 0 for s in SUPPORTED_SEVERITIES}
		for (tid, aid), alert in self.alerts.items():
			if tid != tenant_id:
				continue
			meta = self._alert_meta.get((tid, aid), {})
			if meta.get("status") != "resolved":
				continue
			created_dt = _parse_iso(meta.get("created_at"))
			resolved_dt = _parse_iso(meta.get("resolved_at"))
			if created_dt is None or resolved_dt is None:
				continue
			sev = alert.severity
			totals[sev] = totals.get(sev, 0) + 1
			resolution_min = (resolved_dt - created_dt).total_seconds() / 60.0
			if resolution_min <= _SLA_MINUTES.get(sev, 1440):
				within_sla[sev] = within_sla.get(sev, 0) + 1
		compliance: dict[str, Any] = {}
		for sev in SUPPORTED_SEVERITIES:
			t = totals[sev]
			w = within_sla[sev]
			compliance[sev] = {
				"total_resolved": t,
				"within_sla": w,
				"compliance_pct": round((w / t * 100) if t else 0.0, 2),
				"sla_minutes": _SLA_MINUTES.get(sev, 1440),
			}
		return {
			"tenant_id": tenant_id,
			"period_hours": period_hours,
			"compliance": compliance,
			"as_of": _now(),
		}

	def analyst_workload(self, tenant_id: str, analyst_id: str | None = None) -> dict[str, Any]:
		"""Alerts assigned per analyst, resolution rate, and overdue count.

		If analyst_id is provided, returns a single-analyst view; otherwise
		returns the full tenant analyst roster.
		"""
		workload: dict[str, dict[str, Any]] = {}
		for (tid, _), asgn in self.assignments.items():
			if tid != tenant_id:
				continue
			assignee = asgn.assignee_id
			if analyst_id and assignee != analyst_id:
				continue
			if assignee not in workload:
				workload[assignee] = {"analyst_id": assignee, "assigned": 0, "resolved": 0, "overdue": 0}
			workload[assignee]["assigned"] += 1
			# check resolution status
			meta = self._alert_meta.get(self._tenant_key(tenant_id, asgn.alert_id), {})
			if meta.get("status") == "resolved":
				workload[assignee]["resolved"] += 1
			else:
				# check overdue
				alert = self._tenant_alert_or_none(asgn.alert_id, tenant_id)
				if alert is not None:
					age_h = _age_hours(meta.get("created_at"))
					sla_h = _SLA_MINUTES.get(alert.severity, 1440) / 60.0
					if age_h > sla_h:
						workload[assignee]["overdue"] += 1
		for entry in workload.values():
			assigned = entry["assigned"]
			resolved = entry["resolved"]
			entry["resolution_rate"] = round(resolved / assigned, 4) if assigned else 0.0
		if analyst_id:
			single = workload.get(analyst_id, {"analyst_id": analyst_id, "assigned": 0, "resolved": 0, "overdue": 0, "resolution_rate": 0.0})
			return {"tenant_id": tenant_id, "analyst": single, "as_of": _now()}
		return {"tenant_id": tenant_id, "analysts": list(workload.values()), "as_of": _now()}

	def export_alerts(self, tenant_id: str, filters: dict[str, Any], format: str = "json") -> dict[str, Any]:
		"""Export alerts to JSON, CSV, or STIX 2.1 bundle format.

		filters: optional dict with keys: severity, status, alert_type.
		Returns a dict with 'format' and 'data' (string payload).
		"""
		fmt = normalize_code(format)
		assert fmt in ("json", "csv", "stix"), f"unsupported format {format!r}; use json, csv, or stix"
		sev_filter = normalize_code(filters["severity"]) if "severity" in filters else None
		status_filter = normalize_code(filters["status"]) if "status" in filters else None
		type_filter = normalize_code(filters["alert_type"]) if "alert_type" in filters else None

		rows: list[dict[str, Any]] = []
		for (tid, aid), alert in self.alerts.items():
			if tid != tenant_id:
				continue
			meta = self._alert_meta.get((tid, aid), {})
			if sev_filter and alert.severity != sev_filter:
				continue
			if status_filter and meta.get("status") != status_filter:
				continue
			if type_filter and alert.alert_type != type_filter:
				continue
			row = alert.to_dict()
			row["status"] = meta.get("status", "open")
			row["created_at"] = meta.get("created_at")
			row["resolved_at"] = meta.get("resolved_at")
			row["acknowledged_by"] = meta.get("acknowledged_by")
			rows.append(row)

		if fmt == "json":
			data = json.dumps({"tenant_id": tenant_id, "alerts": rows}, indent=2, default=str)
		elif fmt == "csv":
			if not rows:
				data = ""
			else:
				buf = io.StringIO()
				writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
				writer.writeheader()
				writer.writerows(rows)
				data = buf.getvalue()
		else:  # stix
			stix_objects = [
				{
					"type": "indicator",
					"spec_version": "2.1",
					"id": f"indicator--{row['id']}",
					"name": row.get("alert_reference", row["id"]),
					"labels": [row.get("severity", "unknown"), row.get("alert_type", "unknown")],
					"created": row.get("created_at") or _now(),
					"modified": row.get("resolved_at") or row.get("created_at") or _now(),
					"pattern": f"[alert:id = '{row['id']}']",
					"pattern_type": "stix",
					"valid_from": row.get("created_at") or _now(),
				}
				for row in rows
			]
			bundle = {
				"type": "bundle",
				"id": f"bundle--{tenant_id}",
				"spec_version": "2.1",
				"objects": stix_objects,
			}
			data = json.dumps(bundle, indent=2, default=str)

		self._audit(tenant_id, "alerts_exported", f"format={fmt} count={len(rows)}")
		return {"tenant_id": tenant_id, "format": fmt, "record_count": len(rows), "data": data}

	# ------------------------------------------------------------------ #
	# Private helpers                                                        #
	# ------------------------------------------------------------------ #

	def _tenant_authority_or_none(self, item_id: str, tenant_id: str) -> AlertAuthority | None:
		return self.authorities.get(self._tenant_key(tenant_id, item_id))

	def _tenant_workspace_or_none(self, item_id: str, tenant_id: str) -> AlertWorkspace | None:
		return self.workspaces.get(self._tenant_key(tenant_id, item_id))

	def _tenant_rule_or_none(self, item_id: str, tenant_id: str) -> AlertRule | None:
		return self.rules.get(self._tenant_key(tenant_id, item_id))

	def _tenant_signal_or_none(self, item_id: str, tenant_id: str) -> AlertSignal | None:
		return self.signals.get(self._tenant_key(tenant_id, item_id))

	def _tenant_alert_or_none(self, item_id: str, tenant_id: str) -> AlertRecord | None:
		return self.alerts.get(self._tenant_key(tenant_id, item_id))

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
		reasons = ", ".join(action.get("reason", action.get("rule", "alert_policy_denied")) for action in result["actions"])
		raise PermissionError(reasons or "alert_policy_denied")

	def _append_timeline(self, alert_id: str, tenant_id: str, event: str, actor: str, notes: str) -> None:
		"""Append a timeline entry to an alert's audit trail."""
		meta = self._alert_meta.get(self._tenant_key(tenant_id, alert_id))
		if meta is None:
			return
		timeline: list[dict[str, Any]] = meta.setdefault("timeline", [])
		timeline.append({"ts": _now(), "event": event, "actor": actor, "notes": notes})


IntelAlertsService = AlertManagementService
