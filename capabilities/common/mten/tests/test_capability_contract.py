"""Regression coverage for the MTEN executable capability contract."""

from .. import register_capability
from ..capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract(
		"tenant-a",
		{"resources": {"quota_alert_threshold_percent": 90}}
	)

	assert contract["capability"] == "mten"
	assert contract["configuration"]["tenant_id"] == "tenant-a"
	assert contract["configuration"]["resources"]["quota_alert_threshold_percent"] == 90
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"provisioning",
		"isolation",
		"resources",
		"orchestration",
		"analytics",
		"ui",
		"theme"
	]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"tenants",
		"provisioning",
		"templates",
		"analytics",
		"optimization",
		"settings"
	}
	assert contract["ui"]["api_prefix"] == "/mten/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "12px"
	assert "tenant_health_card" in contract["theme"]["components"]


def test_rule_engine_enforces_multi_tenant_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"cross_tenant_operation": True,
		"tenant_membership_confirmed": False,
		"tenant_status": "suspended",
		"requested_operation_is_mutation": True,
		"custom_domain_requested": True,
		"dns_validated": False,
		"projected_compute_units": 1400,
		"capacity_approval_recorded": False,
		"requested_operation": "live_migration",
		"runbook_attached": False
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"tenant_context_required",
		"cross_tenant_access_requires_membership",
		"suspended_tenants_block_mutations",
		"custom_domain_requires_dns_validation",
		"capacity_overcommit_requires_review",
		"live_migration_requires_runbook"
	}


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "mten_control_fabric"
	assert registration["ui_components"]["optimization"] == "/mten/optimization"
	assert "auth_rbac" in registration["dependencies"]
