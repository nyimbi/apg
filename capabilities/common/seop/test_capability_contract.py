"""Regression coverage for the SEOP executable capability contract."""

from capabilities.common.seop import register_capability
from capabilities.common.seop.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-seop", {"detection": {"confidence_threshold": 0.8}})

	assert contract["capability"] == "seop"
	assert contract["configuration"]["tenant_id"] == "tenant-seop"
	assert contract["configuration"]["detection"]["confidence_threshold"] == 0.8
	assert contract["configuration_schema"]["required"] == ["tenant_id", "detection", "incidents", "response", "governance", "ui", "theme"]
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "detections", "incidents", "triage", "playbooks", "responses", "posture", "settings"}
	assert contract["theme"]["name"] == "seop_security_ops"


def test_rule_engine_enforces_seop_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "open_incident", "incident_owner_assigned": False, "incident_severity": "critical", "escalation_recorded": False, "anomaly_confidence": 0.95, "triage_review_recorded": False})
	detection_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "create_detection", "alert_source_present": False})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "incident_requires_owner", "critical_incident_requires_escalation", "high_confidence_anomaly_requires_review"}
	assert detection_result["matched_rules"] == ["detection_requires_alert_source"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "seop"
	assert "anom" in registration["dependencies"]
	assert registration["ui_components"]["incidents"] == "/seop/incidents"
	assert "seop:respond" in registration["permissions"]
