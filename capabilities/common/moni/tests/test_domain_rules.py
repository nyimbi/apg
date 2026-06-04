"""CI tests for MONI domain rules.

Covers every assert_* function and the RuleViolation exception contract.
No mocks — all rules are pure functions.

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""
from __future__ import annotations

import pytest
from dataclasses import dataclass

from capabilities.common.moni.domain.rules import (
	RuleViolation,
	assert_tenant_context,
	assert_write_policy,
	assert_no_cross_tenant_access,
	assert_source_registered,
	assert_source_active,
	assert_metric_source_present,
	assert_trace_has_trace_id,
	assert_trace_has_service_name,
	assert_no_pii_in_logs,
	assert_cardinality_within_limit,
	assert_critical_alert_has_route,
	assert_critical_alert_has_owner,
	assert_alert_rule_has_metric,
	assert_threshold_operator_valid,
	assert_critical_incident_has_owner,
	assert_incident_not_closed,
	assert_slo_has_alert_route,
	assert_slo_objective_valid,
	assert_slo_window_positive,
	assert_production_remediation_has_runbook,
	assert_independent_reviewer,
	assert_review_notes_present,
	assert_health_check_interval_sane,
	assert_health_check_timeout_lt_interval,
	assert_agent_runtime_supported,
	assert_agent_role_supported,
	assert_agent_contribution_disclosed,
	assert_privileged_agent_has_human_approval,
	assert_bytewax_stream,
	assert_retention_within_limit,
	assert_anomaly_sensitivity_valid,
	assert_baseline_sufficient,
	assert_query_time_range_valid,
	SUPPORTED_RUNTIMES,
	SUPPORTED_ROLES,
	PRIVILEGED_ROLES,
	RETENTION_LIMITS_DAYS,
)
from datetime import datetime, timedelta


# ─── RuleViolation contract ───────────────────────────────────────────────────

def test_rule_violation_carries_metadata():
	exc = RuleViolation("my_rule", "bad thing happened", "fix_it")
	assert exc.rule_name == "my_rule"
	assert exc.reason == "bad thing happened"
	assert exc.required_action == "fix_it"
	assert "my_rule" in str(exc)
	assert "bad thing happened" in str(exc)


def test_rule_violation_optional_required_action():
	exc = RuleViolation("r", "reason")
	assert exc.required_action == ""


# ─── Tenant isolation ──────────────────────────────────────────────────────────

def test_assert_tenant_context_passes_with_tenant_id():
	assert_tenant_context({"tenant_id": "org1"})


def test_assert_tenant_context_raises_without_tenant_id():
	with pytest.raises(RuleViolation) as exc_info:
		assert_tenant_context({})
	assert exc_info.value.rule_name == "tenant_context_required"


def test_assert_tenant_context_raises_on_empty_string():
	with pytest.raises(RuleViolation):
		assert_tenant_context({"tenant_id": ""})


def test_assert_no_cross_tenant_access_same_tenant_ok():
	assert_no_cross_tenant_access("t1", "t1")


def test_assert_no_cross_tenant_access_different_tenants_raises():
	with pytest.raises(RuleViolation) as exc_info:
		assert_no_cross_tenant_access("t1", "t2")
	assert exc_info.value.rule_name == "cross_tenant_access_denied"


def test_assert_write_policy_read_skips_check():
	assert_write_policy({"operation_type": "read", "policy_attached": False})


def test_assert_write_policy_write_without_policy_raises():
	with pytest.raises(RuleViolation) as exc_info:
		assert_write_policy({"operation_type": "write", "policy_attached": False})
	assert exc_info.value.rule_name == "write_requires_policy"


def test_assert_write_policy_write_with_policy_ok():
	assert_write_policy({"operation_type": "write", "policy_attached": True})


# ─── Signal / metric ingestion ────────────────────────────────────────────────

@dataclass
class FakeSource:
	status: str = "active"

def test_assert_source_registered_passes_with_source():
	assert_source_registered(FakeSource(), "src1")


def test_assert_source_registered_raises_on_none():
	with pytest.raises(RuleViolation) as exc_info:
		assert_source_registered(None, "missing-src")
	assert exc_info.value.rule_name == "signal_requires_registered_source"


def test_assert_source_active_passes_for_active():
	assert_source_active(FakeSource(status="active"), "s1")


def test_assert_source_active_raises_for_disabled():
	with pytest.raises(RuleViolation) as exc_info:
		assert_source_active(FakeSource(status="disabled"), "s1")
	assert exc_info.value.rule_name == "disabled_source_blocks_ingestion"


def test_assert_source_active_works_with_dict():
	with pytest.raises(RuleViolation):
		assert_source_active({"status": "disabled"}, "s1")


def test_assert_metric_source_present_ok():
	assert_metric_source_present("my-source")


def test_assert_metric_source_present_raises_on_empty():
	with pytest.raises(RuleViolation) as exc_info:
		assert_metric_source_present("")
	assert exc_info.value.rule_name == "metric_ingestion_requires_source"


def test_assert_metric_source_present_raises_on_none():
	with pytest.raises(RuleViolation):
		assert_metric_source_present(None)


def test_assert_trace_has_trace_id_non_trace_skips():
	assert_trace_has_trace_id("metric", None)


def test_assert_trace_has_trace_id_trace_with_id_ok():
	assert_trace_has_trace_id("trace", "trace-123")


def test_assert_trace_has_trace_id_trace_missing_raises():
	with pytest.raises(RuleViolation) as exc_info:
		assert_trace_has_trace_id("trace", None)
	assert exc_info.value.rule_name == "trace_requires_trace_id"


def test_assert_trace_has_service_name_ok():
	assert_trace_has_service_name("trace", "orders")


def test_assert_trace_has_service_name_missing_raises():
	with pytest.raises(RuleViolation) as exc_info:
		assert_trace_has_service_name("trace", None)
	assert exc_info.value.rule_name == "trace_requires_service_name"


def test_assert_no_pii_in_logs_non_log_skips():
	assert_no_pii_in_logs("metric", contains_pii=True, pii_redacted=False)


def test_assert_no_pii_in_logs_redacted_ok():
	assert_no_pii_in_logs("log", contains_pii=True, pii_redacted=True)


def test_assert_no_pii_in_logs_pii_not_redacted_raises():
	with pytest.raises(RuleViolation) as exc_info:
		assert_no_pii_in_logs("log", contains_pii=True, pii_redacted=False)
	assert exc_info.value.rule_name == "pii_logs_blocked"


def test_assert_cardinality_within_limit_ok():
	assert_cardinality_within_limit(5000)


def test_assert_cardinality_within_limit_exception_recorded_ok():
	assert_cardinality_within_limit(50000, exception_recorded=True)


def test_assert_cardinality_within_limit_high_raises():
	with pytest.raises(RuleViolation) as exc_info:
		assert_cardinality_within_limit(15000)
	assert exc_info.value.rule_name == "high_cardinality_metric_requires_review"


# ─── Alert rules ──────────────────────────────────────────────────────────────

def test_assert_critical_alert_has_route_ok():
	assert_critical_alert_has_route("critical", "pagerduty:team")


def test_assert_critical_alert_has_route_medium_no_route_ok():
	assert_critical_alert_has_route("medium", None)


def test_assert_critical_alert_has_route_critical_no_route_raises():
	with pytest.raises(RuleViolation) as exc_info:
		assert_critical_alert_has_route("critical", None)
	assert exc_info.value.rule_name == "critical_alert_requires_route"


def test_assert_critical_alert_has_owner_ok():
	assert_critical_alert_has_owner("critical", "ops-team")


def test_assert_critical_alert_has_owner_raises():
	with pytest.raises(RuleViolation) as exc_info:
		assert_critical_alert_has_owner("critical", None)
	assert exc_info.value.rule_name == "critical_alert_requires_owner"


def test_assert_alert_rule_has_metric_ok():
	assert_alert_rule_has_metric("cpu_usage_percent")


def test_assert_alert_rule_has_metric_raises_on_empty():
	with pytest.raises(RuleViolation) as exc_info:
		assert_alert_rule_has_metric("")
	assert exc_info.value.rule_name == "alert_rule_requires_metric"


def test_assert_threshold_operator_valid_all_valid():
	for op in ("gt", "lt", "gte", "lte", "eq", "ne"):
		assert_threshold_operator_valid(op)


def test_assert_threshold_operator_valid_raises_on_invalid():
	with pytest.raises(RuleViolation) as exc_info:
		assert_threshold_operator_valid("greater_than")
	assert exc_info.value.rule_name == "invalid_threshold_operator"


# ─── Incident rules ───────────────────────────────────────────────────────────

def test_assert_critical_incident_has_owner_ok():
	assert_critical_incident_has_owner("critical", "sre-lead")


def test_assert_critical_incident_has_owner_raises():
	with pytest.raises(RuleViolation) as exc_info:
		assert_critical_incident_has_owner("critical", None)
	assert exc_info.value.rule_name == "critical_incident_requires_owner"


def test_assert_incident_not_closed_open_ok():
	assert_incident_not_closed("open", "inc-1")


def test_assert_incident_not_closed_resolved_raises():
	with pytest.raises(RuleViolation) as exc_info:
		assert_incident_not_closed("resolved", "inc-1")
	assert exc_info.value.rule_name == "incident_already_closed"


# ─── SLO rules ────────────────────────────────────────────────────────────────

def test_assert_slo_has_alert_route_ok():
	assert_slo_has_alert_route("pagerduty:orders")


def test_assert_slo_has_alert_route_raises():
	with pytest.raises(RuleViolation) as exc_info:
		assert_slo_has_alert_route(None)
	assert exc_info.value.rule_name == "slo_requires_alert_route"


def test_assert_slo_objective_valid_ok():
	assert_slo_objective_valid(99.9)
	assert_slo_objective_valid(100.0)
	assert_slo_objective_valid(0.1)


def test_assert_slo_objective_valid_zero_raises():
	with pytest.raises(RuleViolation) as exc_info:
		assert_slo_objective_valid(0.0)
	assert exc_info.value.rule_name == "slo_objective_out_of_range"


def test_assert_slo_objective_valid_over_100_raises():
	with pytest.raises(RuleViolation):
		assert_slo_objective_valid(100.1)


def test_assert_slo_window_positive_ok():
	assert_slo_window_positive(30)


def test_assert_slo_window_positive_zero_raises():
	with pytest.raises(RuleViolation) as exc_info:
		assert_slo_window_positive(0)
	assert exc_info.value.rule_name == "slo_window_too_short"


# ─── Remediation rules ────────────────────────────────────────────────────────

def test_assert_production_remediation_has_runbook_ok():
	assert_production_remediation_has_runbook("production", True)


def test_assert_production_remediation_has_runbook_staging_no_runbook_ok():
	assert_production_remediation_has_runbook("staging", False)


def test_assert_production_remediation_has_runbook_raises():
	with pytest.raises(RuleViolation) as exc_info:
		assert_production_remediation_has_runbook("production", False)
	assert exc_info.value.rule_name == "production_remediation_requires_runbook"


def test_assert_independent_reviewer_ok():
	assert_independent_reviewer("sre-lead", "dev-team")


def test_assert_independent_reviewer_same_raises():
	with pytest.raises(RuleViolation) as exc_info:
		assert_independent_reviewer("alice", "alice")
	assert exc_info.value.rule_name == "remediation_review_requires_independent_reviewer"


def test_assert_review_notes_present_ok():
	assert_review_notes_present("Reviewed and approved per runbook.")


def test_assert_review_notes_present_raises_on_empty():
	with pytest.raises(RuleViolation) as exc_info:
		assert_review_notes_present("")
	assert exc_info.value.rule_name == "review_notes_required"


def test_assert_review_notes_present_raises_on_none():
	with pytest.raises(RuleViolation):
		assert_review_notes_present(None)


# ─── Health check rules ───────────────────────────────────────────────────────

def test_assert_health_check_interval_sane_ok():
	assert_health_check_interval_sane(30)


def test_assert_health_check_interval_sane_too_short_raises():
	with pytest.raises(RuleViolation) as exc_info:
		assert_health_check_interval_sane(4)
	assert exc_info.value.rule_name == "health_check_interval_too_short"


def test_assert_health_check_timeout_lt_interval_ok():
	assert_health_check_timeout_lt_interval(5, 30)


def test_assert_health_check_timeout_lt_interval_raises():
	with pytest.raises(RuleViolation) as exc_info:
		assert_health_check_timeout_lt_interval(30, 30)
	assert exc_info.value.rule_name == "health_check_timeout_exceeds_interval"


# ─── Monitoring agent rules ───────────────────────────────────────────────────

def test_supported_runtimes_constant():
	assert "codex" in SUPPORTED_RUNTIMES
	assert "claude_code" in SUPPORTED_RUNTIMES


def test_supported_roles_constant():
	assert "slo_reviewer" in SUPPORTED_ROLES
	assert "alert_reviewer" in SUPPORTED_ROLES


def test_privileged_roles_subset_of_supported():
	assert PRIVILEGED_ROLES.issubset(SUPPORTED_ROLES)


def test_assert_agent_runtime_supported_ok():
	for rt in SUPPORTED_RUNTIMES:
		assert_agent_runtime_supported(rt)


def test_assert_agent_runtime_supported_raises():
	with pytest.raises(RuleViolation) as exc_info:
		assert_agent_runtime_supported("unknown_rt")
	assert exc_info.value.rule_name == "monitoring_agent_runtime_supported"


def test_assert_agent_role_supported_ok():
	for role in SUPPORTED_ROLES:
		assert_agent_role_supported(role)


def test_assert_agent_role_supported_raises():
	with pytest.raises(RuleViolation) as exc_info:
		assert_agent_role_supported("hacker")
	assert exc_info.value.rule_name == "monitoring_agent_role_supported"


def test_assert_agent_contribution_disclosed_ok():
	assert_agent_contribution_disclosed(True)


def test_assert_agent_contribution_disclosed_raises():
	with pytest.raises(RuleViolation) as exc_info:
		assert_agent_contribution_disclosed(False)
	assert exc_info.value.rule_name == "monitoring_agent_requires_contribution_disclosure"


def test_assert_privileged_agent_has_human_approval_non_privileged_no_approval_ok():
	assert_privileged_agent_has_human_approval("metric_quality_reviewer", False)


def test_assert_privileged_agent_has_human_approval_privileged_with_approval_ok():
	for role in PRIVILEGED_ROLES:
		assert_privileged_agent_has_human_approval(role, True)


def test_assert_privileged_agent_has_human_approval_privileged_without_approval_raises():
	with pytest.raises(RuleViolation) as exc_info:
		assert_privileged_agent_has_human_approval("slo_reviewer", False)
	assert exc_info.value.rule_name == "monitoring_agent_privileged_role_requires_human_approval"


# ─── Streaming ────────────────────────────────────────────────────────────────

def test_assert_bytewax_stream_ok():
	assert_bytewax_stream("bytewax")


def test_assert_bytewax_stream_raises():
	with pytest.raises(RuleViolation) as exc_info:
		assert_bytewax_stream("kafka")
	assert exc_info.value.rule_name == "bytewax_monitoring_stream_required"


# ─── Retention ────────────────────────────────────────────────────────────────

def test_retention_limits_constant():
	assert RETENTION_LIMITS_DAYS["metrics"] == 90
	assert RETENTION_LIMITS_DAYS["logs"] == 30
	assert RETENTION_LIMITS_DAYS["traces"] == 14


def test_assert_retention_within_limit_ok():
	assert_retention_within_limit("metrics", 30)


def test_assert_retention_within_limit_over_limit_raises():
	with pytest.raises(RuleViolation) as exc_info:
		assert_retention_within_limit("logs", 60)
	assert exc_info.value.rule_name == "retention_above_limit_requires_review"


def test_assert_retention_within_limit_exception_recorded_ok():
	assert_retention_within_limit("metrics", 365, exception_recorded=True)


# ─── Anomaly detection ────────────────────────────────────────────────────────

def test_assert_anomaly_sensitivity_valid_ok():
	assert_anomaly_sensitivity_valid(0.0)
	assert_anomaly_sensitivity_valid(0.5)
	assert_anomaly_sensitivity_valid(1.0)


def test_assert_anomaly_sensitivity_valid_raises():
	with pytest.raises(RuleViolation) as exc_info:
		assert_anomaly_sensitivity_valid(1.1)
	assert exc_info.value.rule_name == "invalid_anomaly_sensitivity"


def test_assert_baseline_sufficient_ok():
	assert_baseline_sufficient(10)
	assert_baseline_sufficient(100)


def test_assert_baseline_sufficient_raises():
	with pytest.raises(RuleViolation) as exc_info:
		assert_baseline_sufficient(5)
	assert exc_info.value.rule_name == "insufficient_baseline_samples"


# ─── Query time range ─────────────────────────────────────────────────────────

def test_assert_query_time_range_valid_ok():
	start = datetime.utcnow() - timedelta(hours=1)
	end = datetime.utcnow()
	assert_query_time_range_valid(start, end)


def test_assert_query_time_range_inverted_raises():
	now = datetime.utcnow()
	with pytest.raises(RuleViolation) as exc_info:
		assert_query_time_range_valid(now, now - timedelta(seconds=1))
	assert exc_info.value.rule_name == "invalid_query_time_range"


def test_assert_query_time_range_too_wide_raises():
	start = datetime.utcnow() - timedelta(days=31)
	end = datetime.utcnow()
	with pytest.raises(RuleViolation) as exc_info:
		assert_query_time_range_valid(start, end)
	assert exc_info.value.rule_name == "query_time_range_too_wide"
