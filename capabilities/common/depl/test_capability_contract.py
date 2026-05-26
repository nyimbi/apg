"""Regression coverage for the DEPL executable capability contract."""

from capabilities.common.depl import register_capability
from capabilities.common.depl.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-deploy", {"rollouts": {"max_canary_percent": 10}})

	assert contract["capability"] == "depl"
	assert contract["configuration"]["tenant_id"] == "tenant-deploy"
	assert contract["configuration"]["rollouts"]["max_canary_percent"] == 10
	assert contract["configuration_schema"]["required"] == ["tenant_id", "releases", "rollouts", "evidence", "governance", "ui", "theme"]
	assert contract["theme"]["name"] == "depl_release_ops"


def test_rule_engine_enforces_deployment_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "create_release", "release_owner_assigned": False, "target_environment": "production", "approval_recorded": False, "canary_percent": 50, "canary_review_recorded": False})
	deploy_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "deploy", "health_gate_passed": False, "rollback_plan_attached": False})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "release_requires_owner", "production_requires_approval", "large_canary_requires_review"}
	assert set(deploy_result["matched_rules"]) == {"deployment_requires_health_gate", "rollback_requires_plan"}


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "depl"
	assert "logt" in registration["dependencies"]
	assert registration["ui_components"]["rollback"] == "/depl/rollback"
	assert "depl:deploy" in registration["permissions"]
