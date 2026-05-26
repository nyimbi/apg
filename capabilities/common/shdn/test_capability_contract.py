"""Regression coverage for the SHDN executable capability contract."""

from capabilities.common.shdn import register_capability
from capabilities.common.shdn.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-shdn", {"services": {"drain_timeout_seconds": 120}})

	assert contract["capability"] == "shdn"
	assert contract["configuration"]["tenant_id"] == "tenant-shdn"
	assert contract["configuration"]["services"]["drain_timeout_seconds"] == 120
	assert contract["configuration_schema"]["required"] == ["tenant_id", "services", "lifecycle", "recovery", "governance", "ui", "theme"]
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "services", "plans", "executions", "approvals", "recovery", "audit", "settings"}
	assert contract["theme"]["name"] == "shdn_lifecycle_control"


def test_rule_engine_enforces_shdn_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "register_service", "service_owner_assigned": False, "production_service": True, "approval_recorded": False, "force_shutdown": True, "force_review_recorded": False})
	shutdown_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "execute_shutdown", "health_gate_passed": False, "backup_snapshot_present": False})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "service_requires_owner", "production_shutdown_requires_approval", "force_shutdown_requires_review"}
	assert shutdown_result["matched_rules"] == ["shutdown_requires_health_gate", "shutdown_requires_backup_snapshot"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "shdn"
	assert "bkup" in registration["dependencies"]
	assert registration["ui_components"]["executions"] == "/shdn/executions"
	assert "shdn:execute" in registration["permissions"]
