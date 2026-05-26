"""Regression coverage for the ESGC executable capability contract."""

from capabilities.common.esgc import register_capability
from capabilities.common.esgc.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-esgc", {"reporting": {"target_tracking_enabled": False}})

	assert contract["capability"] == "esgc"
	assert contract["configuration"]["tenant_id"] == "tenant-esgc"
	assert contract["configuration"]["reporting"]["target_tracking_enabled"] is False
	assert contract["configuration_schema"]["required"] == ["tenant_id", "emissions", "data_sources", "reporting", "governance", "ui", "theme"]
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "emissions", "factors", "data_sources", "reports", "targets", "audit", "settings"}
	assert contract["theme"]["name"] == "esgc_sustainability_ops"


def test_rule_engine_enforces_esgc_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "create_inventory", "organization_owner_assigned": False, "factor_source_approved": False, "geospatial_boundary_present": False, "emission_anomaly_detected": True, "anomaly_review_recorded": False})
	report_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "publish_report", "approval_recorded": False, "factor_source_approved": True, "geospatial_boundary_present": True})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "inventory_requires_owner", "factor_requires_approved_source", "emission_requires_boundary", "emission_anomaly_requires_review"}
	assert report_result["matched_rules"] == ["report_requires_approval"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "esgc"
	assert "comp" in registration["dependencies"]
	assert registration["ui_components"]["reports"] == "/esgc/reports"
	assert "esgc:report" in registration["permissions"]
