"""Regression coverage for the ENVM executable capability contract."""

from capabilities.common.envm import register_capability
from capabilities.common.envm.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-env", {"drift": {"drift_threshold_percent": 10}})

	assert contract["capability"] == "envm"
	assert contract["configuration"]["tenant_id"] == "tenant-env"
	assert contract["configuration"]["drift"]["drift_threshold_percent"] == 10
	assert contract["configuration_schema"]["required"] == ["tenant_id", "environments", "promotion", "drift", "governance", "ui", "theme"]
	assert contract["theme"]["name"] == "envm_environment_ops"


def test_rule_engine_enforces_environment_guardrails():
	result = evaluate_capability_rules({"tenant_context_present": False, "operation": "create_environment", "environment_owner_assigned": False, "environment": "production", "approval_recorded": False, "secret_scope_present": True, "secret_policy_attached": False, "drift_percent": 15, "drift_review_recorded": False})
	promote_result = evaluate_capability_rules({"tenant_context_present": True, "operation": "promote", "promotion_path_attached": False})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {"tenant_context_required", "environment_requires_owner", "production_change_requires_approval", "secret_scope_requires_policy", "high_drift_requires_review"}
	assert promote_result["matched_rules"] == ["promotion_requires_path"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "envm"
	assert "depl" in registration["dependencies"]
	assert registration["ui_components"]["promotion"] == "/envm/promotion"
	assert "envm:promote" in registration["permissions"]
