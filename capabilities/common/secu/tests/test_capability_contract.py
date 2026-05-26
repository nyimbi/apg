"""Regression coverage for the SECU executable capability contract."""

from .. import get_capability_info, register_capability
from ..capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-a", {"risk": {"critical_threshold": 95}})

	assert contract["capability"] == "secu"
	assert contract["configuration"]["tenant_id"] == "tenant-a"
	assert contract["configuration"]["risk"]["critical_threshold"] == 95
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"zero_trust",
		"risk",
		"threat_detection",
		"compliance",
		"ui",
		"theme"
	]
	assert len(contract["rule_engine"]["rules"]) >= 5
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"risk",
		"threats",
		"policies",
		"compliance",
		"rules",
		"settings"
	}
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "risk_score_meter" in contract["theme"]["components"]


def test_rule_engine_denies_high_risk_context():
	result = evaluate_capability_rules({
		"is_known_malicious": True,
		"device_trust": "compromised",
		"risk_score": 92,
		"challenge_completed": False,
		"compliance_violation": True,
		"audit_evidence_attached": False
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"known_malicious_network_denied",
		"compromised_device_quarantined",
		"critical_risk_denied",
		"high_risk_requires_challenge",
		"compliance_violation_alert"
	}


def test_capability_info_and_registration_include_manifest_and_theme():
	info = get_capability_info()
	registration = register_capability()

	assert info["metadata"]["capability_name"] == "secu"
	assert info["configuration"]["tenant_id"] == "default"
	assert info["ui_manifest"]["requires_theme"] is True
	assert info["theme"]["name"] == "secu_zero_trust"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_components"]["policies"] == "/secu/policies"
