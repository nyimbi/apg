"""Deterministic domain rules for Monitoring and Observability.

Every business rule from the capability contract is implemented here as a
callable Python function. Rules raise RuleViolation on constraint breach.
assert_* functions guard entry points; calculate_* functions derive values.

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any


# ─── Exception ────────────────────────────────────────────────────────────────

class RuleViolation(Exception):
	"""Raised when a deterministic business rule is violated."""

	def __init__(self, rule_name: str, reason: str, required_action: str = "") -> None:
		self.rule_name = rule_name
		self.reason = reason
		self.required_action = required_action
		super().__init__(f"Rule '{rule_name}' violated: {reason}")


# ─── Tenant isolation ──────────────────────────────────────────────────────────

def assert_tenant_context(context: dict[str, Any]) -> None:
	"""All operations require a tenant context."""
	if not context.get("tenant_id"):
		raise RuleViolation(
			"tenant_context_required",
			"tenant_id is required",
			"attach_tenant_context",
		)


def assert_write_policy(context: dict[str, Any]) -> None:
	"""Write operations require an attached policy."""
	if context.get("operation_type") == "write" and not context.get("policy_attached"):
		raise RuleViolation(
			"write_requires_policy",
			"write operations require an attached policy",
			"attach_policy",
		)


def assert_no_cross_tenant_access(actor_tenant: str, resource_tenant: str) -> None:
	"""Cross-tenant access is always denied."""
	if actor_tenant != resource_tenant:
		raise RuleViolation(
			"cross_tenant_access_denied",
			"cross-tenant access is not permitted",
			"use_own_tenant_resources",
		)


# ─── Signal / metric ingestion ────────────────────────────────────────────────

def assert_source_registered(source: Any | None, source_id: str) -> None:
	"""Metric and log ingestion require a registered source."""
	if source is None:
		raise RuleViolation(
			"signal_requires_registered_source",
			f"source '{source_id}' is not registered",
			"register_signal_source",
		)


def assert_source_active(source: Any, source_id: str) -> None:
	"""Disabled sources cannot emit telemetry."""
	status = getattr(source, "status", None) or (source.get("status") if isinstance(source, dict) else None)
	if status == "disabled":
		raise RuleViolation(
			"disabled_source_blocks_ingestion",
			f"source '{source_id}' is disabled",
			"reactivate_or_select_source",
		)


def assert_metric_source_present(source_id: str | None) -> None:
	"""Metrics require an explicit source identifier."""
	if not source_id or not str(source_id).strip():
		raise RuleViolation(
			"metric_ingestion_requires_source",
			"metric ingestion requires an explicit source identifier",
			"attach_metric_source",
		)


def assert_trace_has_trace_id(signal_type: str, trace_id: str | None) -> None:
	"""Trace signals require a trace_id."""
	if signal_type == "trace" and not trace_id:
		raise RuleViolation(
			"trace_requires_trace_id",
			"trace ingestion requires a trace identifier",
			"attach_trace_id",
		)


def assert_trace_has_service_name(signal_type: str, service_name: str | None) -> None:
	"""Trace signals require a service_name."""
	if signal_type == "trace" and not service_name:
		raise RuleViolation(
			"trace_requires_service_name",
			"trace ingestion requires service name evidence",
			"attach_service_name",
		)


def assert_no_pii_in_logs(signal_type: str, contains_pii: bool, pii_redacted: bool) -> None:
	"""Logs containing PII must be redacted before ingestion."""
	if signal_type == "log" and contains_pii and not pii_redacted:
		raise RuleViolation(
			"pii_logs_blocked",
			"logs containing PII must be redacted",
			"redact_or_drop_log_event",
		)


def assert_cardinality_within_limit(
	cardinality: int,
	limit: int = 10_000,
	exception_recorded: bool = False,
) -> None:
	"""High-cardinality metrics require a recorded exception."""
	if cardinality > limit and not exception_recorded:
		raise RuleViolation(
			"high_cardinality_metric_requires_review",
			f"metric cardinality {cardinality} exceeds limit {limit}",
			"record_cardinality_exception",
		)


# ─── Alert rules ──────────────────────────────────────────────────────────────

def assert_critical_alert_has_route(severity: str, notification_route: str | None) -> None:
	"""Critical alerts require an escalation route."""
	if severity == "critical" and not notification_route:
		raise RuleViolation(
			"critical_alert_requires_route",
			"critical alerts require an escalation route",
			"configure_alert_route",
		)


def assert_critical_alert_has_owner(severity: str, owner: str | None) -> None:
	"""Critical alerts require an assigned owner."""
	if severity == "critical" and not owner:
		raise RuleViolation(
			"critical_alert_requires_owner",
			"critical alerts require an assigned owner",
			"assign_alert_owner",
		)


def assert_alert_rule_has_metric(metric_name: str | None) -> None:
	"""Alert rules must reference a metric."""
	if not metric_name or not str(metric_name).strip():
		raise RuleViolation(
			"alert_rule_requires_metric",
			"alert rules must reference a metric name",
			"attach_metric_name",
		)


def assert_threshold_operator_valid(operator: str) -> None:
	"""Threshold operator must be a recognised comparison."""
	valid = {"gt", "lt", "gte", "lte", "eq", "ne"}
	if operator not in valid:
		raise RuleViolation(
			"invalid_threshold_operator",
			f"operator '{operator}' is not one of {sorted(valid)}",
			"select_valid_threshold_operator",
		)


# ─── Incident rules ───────────────────────────────────────────────────────────

def assert_critical_incident_has_owner(severity: str, owner: str | None) -> None:
	"""Critical incidents require an assigned owner."""
	if severity == "critical" and not owner:
		raise RuleViolation(
			"critical_incident_requires_owner",
			"critical incidents require an assigned owner",
			"assign_incident_owner",
		)


def assert_incident_not_closed(status: str, incident_id: str) -> None:
	"""Operations on closed incidents are not permitted."""
	if status in ("resolved", "closed", "denied"):
		raise RuleViolation(
			"incident_already_closed",
			f"incident '{incident_id}' is already {status}",
			"open_new_incident",
		)


# ─── SLO rules ────────────────────────────────────────────────────────────────

def assert_slo_has_alert_route(notification_route: str | None) -> None:
	"""SLO definitions require an alert route."""
	if not notification_route:
		raise RuleViolation(
			"slo_requires_alert_route",
			"SLO definitions require an alert route",
			"configure_slo_alert_route",
		)


def assert_slo_objective_valid(objective_percent: float) -> None:
	"""SLO objective must be in the range (0, 100]."""
	if not (0.0 < objective_percent <= 100.0):
		raise RuleViolation(
			"slo_objective_out_of_range",
			f"SLO objective {objective_percent}% is outside (0, 100]",
			"set_valid_slo_objective",
		)


def assert_slo_window_positive(window_days: int) -> None:
	"""SLO window must be at least 1 day."""
	if window_days < 1:
		raise RuleViolation(
			"slo_window_too_short",
			f"SLO window {window_days}d must be at least 1 day",
			"increase_slo_window",
		)


# ─── Remediation rules ────────────────────────────────────────────────────────

def assert_production_remediation_has_runbook(
	environment: str,
	runbook_approved: bool,
) -> None:
	"""Production remediations require an approved runbook."""
	if environment == "production" and not runbook_approved:
		raise RuleViolation(
			"production_remediation_requires_runbook",
			"production remediation requires an approved runbook",
			"approve_remediation_runbook",
		)


def assert_independent_reviewer(reviewer: str, requester: str) -> None:
	"""Remediation approvals require an independent reviewer."""
	if reviewer == requester:
		raise RuleViolation(
			"remediation_review_requires_independent_reviewer",
			"reviewer cannot be the same as the requester",
			"assign_independent_reviewer",
		)


def assert_review_notes_present(notes: str | None) -> None:
	"""Reviews require written notes."""
	if not notes or not str(notes).strip():
		raise RuleViolation(
			"review_notes_required",
			"review notes are required",
			"attach_review_notes",
		)


# ─── Health check rules ───────────────────────────────────────────────────────

def assert_health_check_interval_sane(interval_seconds: int) -> None:
	"""Health check interval must be at least 5 seconds."""
	if interval_seconds < 5:
		raise RuleViolation(
			"health_check_interval_too_short",
			f"health check interval {interval_seconds}s must be at least 5s",
			"increase_check_interval",
		)


def assert_health_check_timeout_lt_interval(
	timeout_seconds: int,
	interval_seconds: int,
) -> None:
	"""Health check timeout must be less than the interval."""
	if timeout_seconds >= interval_seconds:
		raise RuleViolation(
			"health_check_timeout_exceeds_interval",
			f"timeout {timeout_seconds}s must be less than interval {interval_seconds}s",
			"reduce_check_timeout",
		)


# ─── Monitoring agent rules ───────────────────────────────────────────────────

SUPPORTED_RUNTIMES = {"codex", "claude_code", "opencode", "pi"}
SUPPORTED_ROLES = {
	"slo_reviewer",
	"alert_reviewer",
	"incident_reviewer",
	"anomaly_triage",
	"metric_quality_reviewer",
	"trace_correlation_reviewer",
	"dashboard_reviewer",
}
PRIVILEGED_ROLES = {
	"slo_reviewer",
	"alert_reviewer",
	"incident_reviewer",
	"anomaly_triage",
}


def assert_agent_runtime_supported(runtime: str) -> None:
	"""Monitoring agents must use a supported runtime."""
	if runtime not in SUPPORTED_RUNTIMES:
		raise RuleViolation(
			"monitoring_agent_runtime_supported",
			f"runtime '{runtime}' is not supported; use one of {sorted(SUPPORTED_RUNTIMES)}",
			"select_supported_agent_runtime",
		)


def assert_agent_role_supported(role: str) -> None:
	"""Monitoring agents must use a supported observability role."""
	if role not in SUPPORTED_ROLES:
		raise RuleViolation(
			"monitoring_agent_role_supported",
			f"role '{role}' is not supported; use one of {sorted(SUPPORTED_ROLES)}",
			"select_supported_agent_role",
		)


def assert_agent_contribution_disclosed(contribution_disclosed: bool) -> None:
	"""Agents must disclose machine contribution."""
	if not contribution_disclosed:
		raise RuleViolation(
			"monitoring_agent_requires_contribution_disclosure",
			"agents must disclose machine contribution",
			"enable_agent_contribution_disclosure",
		)


def assert_privileged_agent_has_human_approval(
	role: str,
	human_approval_required: bool,
) -> None:
	"""Privileged agent roles must require human approval."""
	if role in PRIVILEGED_ROLES and not human_approval_required:
		raise RuleViolation(
			"monitoring_agent_privileged_role_requires_human_approval",
			f"privileged role '{role}' requires human_approval_required=True",
			"require_human_approval_for_agent",
		)


# ─── Streaming / Bytewax rules ────────────────────────────────────────────────

def assert_bytewax_stream(event_stream: str) -> None:
	"""MONI lifecycle batches must use Bytewax."""
	if event_stream != "bytewax":
		raise RuleViolation(
			"bytewax_monitoring_stream_required",
			f"event_stream '{event_stream}' must be 'bytewax'",
			"route_batch_through_bytewax",
		)


# ─── Retention rules ──────────────────────────────────────────────────────────

RETENTION_LIMITS_DAYS: dict[str, int] = {
	"metrics": 90,
	"logs": 30,
	"traces": 14,
	"compliance_evidence": 2555,
}


def assert_retention_within_limit(
	signal_type: str,
	retention_days: int,
	exception_recorded: bool = False,
) -> None:
	"""Retention beyond tenant limits requires a recorded exception."""
	limit = RETENTION_LIMITS_DAYS.get(signal_type, 90)
	if retention_days > limit and not exception_recorded:
		raise RuleViolation(
			"retention_above_limit_requires_review",
			f"{signal_type} retention {retention_days}d exceeds limit {limit}d",
			"record_retention_exception",
		)


# ─── Anomaly detection rules ──────────────────────────────────────────────────

def assert_anomaly_sensitivity_valid(sensitivity: float) -> None:
	"""Anomaly sensitivity must be in [0.0, 1.0]."""
	if not (0.0 <= sensitivity <= 1.0):
		raise RuleViolation(
			"invalid_anomaly_sensitivity",
			f"sensitivity {sensitivity} must be in [0.0, 1.0]",
			"set_valid_sensitivity",
		)


def assert_baseline_sufficient(sample_count: int, minimum: int = 10) -> None:
	"""Anomaly detection requires a sufficient baseline sample."""
	if sample_count < minimum:
		raise RuleViolation(
			"insufficient_baseline_samples",
			f"only {sample_count} samples available; need at least {minimum}",
			"collect_more_baseline_data",
		)


# ─── Query / time-range rules ─────────────────────────────────────────────────

def assert_query_time_range_valid(start_time: datetime, end_time: datetime) -> None:
	"""Query end_time must be after start_time and within 30 days."""
	if end_time <= start_time:
		raise RuleViolation(
			"invalid_query_time_range",
			"end_time must be after start_time",
			"correct_time_range",
		)
	max_duration = timedelta(days=30)
	if (end_time - start_time) > max_duration:
		raise RuleViolation(
			"query_time_range_too_wide",
			"query time range cannot exceed 30 days",
			"narrow_time_range",
		)


__all__ = [
	"RuleViolation",
	# tenant
	"assert_tenant_context",
	"assert_write_policy",
	"assert_no_cross_tenant_access",
	# signal/metric
	"assert_source_registered",
	"assert_source_active",
	"assert_metric_source_present",
	"assert_trace_has_trace_id",
	"assert_trace_has_service_name",
	"assert_no_pii_in_logs",
	"assert_cardinality_within_limit",
	# alert
	"assert_critical_alert_has_route",
	"assert_critical_alert_has_owner",
	"assert_alert_rule_has_metric",
	"assert_threshold_operator_valid",
	# incident
	"assert_critical_incident_has_owner",
	"assert_incident_not_closed",
	# slo
	"assert_slo_has_alert_route",
	"assert_slo_objective_valid",
	"assert_slo_window_positive",
	# remediation
	"assert_production_remediation_has_runbook",
	"assert_independent_reviewer",
	"assert_review_notes_present",
	# health check
	"assert_health_check_interval_sane",
	"assert_health_check_timeout_lt_interval",
	# agent
	"assert_agent_runtime_supported",
	"assert_agent_role_supported",
	"assert_agent_contribution_disclosed",
	"assert_privileged_agent_has_human_approval",
	"SUPPORTED_RUNTIMES",
	"SUPPORTED_ROLES",
	"PRIVILEGED_ROLES",
	# streaming
	"assert_bytewax_stream",
	# retention
	"assert_retention_within_limit",
	"RETENTION_LIMITS_DAYS",
	# anomaly
	"assert_anomaly_sensitivity_valid",
	"assert_baseline_sufficient",
	# query
	"assert_query_time_range_valid",
]
