"""Regression coverage for the LOGT executable capability contract."""

from capabilities.common.logt import register_capability
from capabilities.common.logt.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-logs", {"tracing": {"span_retention_days": 7}})

	assert contract["capability"] == "logt"
	assert contract["configuration"]["tenant_id"] == "tenant-logs"
	assert contract["configuration"]["tracing"]["span_retention_days"] == 7
	assert contract["configuration_schema"]["required"] == ["tenant_id", "ingestion", "tracing", "privacy", "governance", "ui", "theme"]
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "logs", "traces", "spans", "pipelines", "retention", "analytics", "settings"}
	assert contract["theme"]["name"] == "logt_observability_console"


def test_rule_engine_enforces_logging_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "create_pipeline", "pipeline_owner_assigned": False, "sensitive_log_content": True, "redaction_applied": False, "query_window_hours": 240, "query_review_recorded": False})
	trace_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "ingest_trace", "trace_context_present": False})
	export_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "export_logs", "approval_recorded": False})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "pipeline_requires_owner", "sensitive_log_requires_redaction", "large_query_requires_review"}
	assert trace_result["matched_rules"] == ["trace_context_required"]
	assert export_result["matched_rules"] == ["log_export_requires_approval"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "logt"
	assert "moni" in registration["dependencies"]
	assert registration["ui_components"]["traces"] == "/logt/traces"
	assert "logt:query" in registration["permissions"]
