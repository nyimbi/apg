"""Regression coverage for the CONN executable capability contract."""

from .. import register_capability
from ..capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-a")

	assert contract["capability"] == "conn"
	assert contract["configuration"]["tenant_id"] == "tenant-a"
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"singer",
		"security",
		"ai",
		"ui",
		"theme"
	]
	assert len(contract["rule_engine"]["rules"]) >= 3
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"designer",
		"marketplace",
		"lineage",
		"data_quality",
		"rules",
		"settings"
	}
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "connection_node" in contract["theme"]["components"]


def test_rule_engine_denies_unsafe_activation():
	result = evaluate_capability_rules({
		"requested_status": "active",
		"last_test_passed": False,
		"contains_credentials": True,
		"credentials_encrypted": False,
		"batch_size": 20000,
		"monitoring_enabled": False
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"require_connection_test_before_activation",
		"encrypt_credentials",
		"large_batch_requires_monitoring"
	}


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "conn_enterprise"
	assert registration["ui_components"]["rules"] == "/conn/rules"
