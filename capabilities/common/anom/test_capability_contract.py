"""Regression coverage for the ANOM executable capability contract."""

from capabilities.common.anom import register_capability
from capabilities.common.anom.capability_contract import (
	evaluate_capability_rules,
	get_capability_contract
)


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-signals", {"detection": {"default_sensitivity": "high"}})

	assert contract["capability"] == "anom"
	assert contract["configuration"]["tenant_id"] == "tenant-signals"
	assert contract["configuration"]["detection"]["default_sensitivity"] == "high"
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"detection",
		"baselines",
		"investigation",
		"governance",
		"ui",
		"theme"
	]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "signals", "baselines", "investigations", "rules", "feedback", "settings"}
	assert contract["ui"]["api_prefix"] == "/anom/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "signal_card" in contract["theme"]["components"]


def test_rule_engine_enforces_anomaly_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "detect",
		"monitoring_source_present": False,
		"history_points": 20,
		"severity": "critical",
		"owner_assigned": False,
		"approval_recorded": False,
		"false_positive_rate": 0.4,
		"tuning_review_recorded": False
	})
	baseline_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "create_baseline",
		"history_points": 20
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"tenant_context_required",
		"detection_requires_monitoring_source",
		"critical_anomaly_requires_owner",
		"high_false_positive_rate_requires_tuning"
	}
	assert baseline_result["decision"] == "deny"
	assert baseline_result["matched_rules"] == ["baseline_requires_history"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "anom"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "anom_signal_console"
	assert registration["ui_components"]["investigations"] == "/anom/investigations"
	assert "pred" in registration["dependencies"]
	assert "anom:investigate" in registration["permissions"]
