"""Regression coverage for the MONI executable capability contract."""

from capabilities.common.moni import register_capability
from capabilities.common.moni.capability_contract import (
	evaluate_capability_rules,
	get_capability_contract
)


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract(
		"tenant-signals",
		{"retention": {"metrics_days": 180}}
	)

	assert contract["capability"] == "moni"
	assert contract["configuration"]["tenant_id"] == "tenant-signals"
	assert contract["configuration"]["retention"]["metrics_days"] == 180
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"collection",
		"alerts",
		"analytics",
		"retention",
		"remediation",
		"security",
		"ui",
		"theme"
	]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"metrics",
		"alerts",
		"traces",
		"analytics",
		"rules",
		"remediation",
		"settings"
	}
	assert contract["ui"]["api_prefix"] == "/moni/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "alert_correlation_stack" in contract["theme"]["components"]


def test_rule_engine_enforces_observability_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "ingest_metric",
		"source_present": False,
		"alert_severity": "critical",
		"notification_route_configured": False,
		"log_contains_pii": True,
		"pii_redacted": False,
		"metric_cardinality": 20000,
		"cardinality_exception_recorded": False,
		"environment": "production",
		"remediation_requested": True,
		"runbook_approved": False
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"tenant_context_required",
		"metric_ingestion_requires_source",
		"critical_alert_requires_route",
		"pii_logs_blocked",
		"high_cardinality_metric_requires_review",
		"production_remediation_requires_runbook"
	}


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "moni_signal_console"
	assert registration["ui_components"]["alerts"] == "/moni/alerts"
	assert "auth" in registration["dependencies"]
