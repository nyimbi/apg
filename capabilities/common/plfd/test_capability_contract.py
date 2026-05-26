"""Regression coverage for the PLFD executable capability contract."""

from capabilities.common.plfd import register_capability
from capabilities.common.plfd.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-plfd", {"operations": {"change_window_required": False}})

	assert contract["capability"] == "plfd"
	assert contract["configuration"]["tenant_id"] == "tenant-plfd"
	assert contract["configuration"]["operations"]["change_window_required"] is False
	assert contract["configuration_schema"]["required"] == ["tenant_id", "foundation", "baselines", "operations", "governance", "ui", "theme"]
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "services", "dependencies", "baselines", "readiness", "changes", "governance", "settings"}
	assert contract["theme"]["name"] == "plfd_platform_foundation"


def test_rule_engine_enforces_plfd_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "register_foundation_service", "service_owner_assigned": False, "configuration_baseline_present": False, "affected_capability_count": 12, "broad_review_recorded": False})
	change_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "approve_platform_change", "dependencies_healthy": False, "approval_recorded": False, "configuration_baseline_present": True})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "foundation_service_requires_owner", "configuration_baseline_required", "broad_platform_change_requires_review"}
	assert change_result["matched_rules"] == ["dependency_health_required", "platform_change_requires_approval"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "plfd"
	assert "mten" in registration["dependencies"]
	assert registration["ui_components"]["baselines"] == "/plfd/baselines"
	assert "plfd:manage_baselines" in registration["permissions"]
