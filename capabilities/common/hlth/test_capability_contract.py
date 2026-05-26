"""Regression coverage for the HLTH executable capability contract."""

from capabilities.common.hlth import register_capability
from capabilities.common.hlth.capability_contract import (
	evaluate_capability_rules,
	get_capability_contract
)


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract(
		"tenant-health",
		{"prediction": {"prediction_window_hours": 48}}
	)

	assert contract["capability"] == "hlth"
	assert contract["configuration"]["tenant_id"] == "tenant-health"
	assert contract["configuration"]["prediction"]["prediction_window_hours"] == 48
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"assessment",
		"baselines",
		"alerts",
		"prediction",
		"remediation",
		"incidents",
		"ui",
		"theme"
	]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"components",
		"alerts",
		"incidents",
		"predictions",
		"remediation",
		"reports",
		"settings"
	}
	assert contract["ui"]["api_prefix"] == "/hlth/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "health_score_card" in contract["theme"]["components"]


def test_rule_engine_enforces_health_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "track_component_health",
		"component_id_present": False,
		"health_score": 25,
		"alert_created": False,
		"remediation_requested": True,
		"runbook_attached": False,
		"baseline_age_days": 45,
		"baseline_review_recorded": False,
		"deployment_requested": True,
		"unresolved_critical_incidents": 1
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"tenant_context_required",
		"component_health_requires_component_id",
		"critical_health_score_creates_alert",
		"remediation_requires_runbook",
		"stale_baseline_requires_review",
		"unresolved_critical_incident_blocks_deploy"
	}


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "hlth_health_console"
	assert registration["ui_components"]["predictions"] == "/hlth/predictions"
	assert "moni" in registration["dependencies"]
