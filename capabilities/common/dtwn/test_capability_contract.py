"""Regression coverage for the DTWN executable capability contract."""

from capabilities.common.dtwn import register_capability
from capabilities.common.dtwn.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-dtwn", {"simulation": {"prediction_confidence_threshold": 0.9}})

	assert contract["capability"] == "dtwn"
	assert contract["configuration"]["tenant_id"] == "tenant-dtwn"
	assert contract["configuration"]["simulation"]["prediction_confidence_threshold"] == 0.9
	assert contract["configuration_schema"]["required"] == ["tenant_id", "twins", "telemetry", "simulation", "governance", "ui", "theme"]
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "twins", "models", "telemetry", "simulations", "predictions", "topology", "settings"}
	assert contract["theme"]["name"] == "dtwn_digital_twin_ops"


def test_rule_engine_enforces_dtwn_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "create_twin", "twin_owner_assigned": False, "telemetry_source_authenticated": False, "prediction_risk_score": 0.95, "prediction_review_recorded": False})
	simulation_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "run_simulation", "model_present": False, "telemetry_source_authenticated": True})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "twin_requires_owner", "telemetry_requires_authenticated_source", "high_risk_prediction_requires_review"}
	assert simulation_result["matched_rules"] == ["simulation_requires_model"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "dtwn"
	assert "iotd" in registration["dependencies"]
	assert registration["ui_components"]["topology"] == "/dtwn/topology"
	assert "dtwn:simulate" in registration["permissions"]
