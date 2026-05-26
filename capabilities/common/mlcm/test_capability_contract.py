"""Regression coverage for the MLCM executable capability contract."""

from capabilities.common.mlcm import register_capability
from capabilities.common.mlcm.capability_contract import (
	evaluate_capability_rules,
	get_capability_contract
)


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-models", {"evaluation": {"minimum_eval_score": 0.9}})

	assert contract["capability"] == "mlcm"
	assert contract["configuration"]["tenant_id"] == "tenant-models"
	assert contract["configuration"]["evaluation"]["minimum_eval_score"] == 0.9
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"registry",
		"promotion",
		"evaluation",
		"governance",
		"ui",
		"theme"
	]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"registry",
		"versions",
		"evaluation",
		"deployments",
		"drift",
		"governance",
		"settings"
	}
	assert contract["ui"]["api_prefix"] == "/mlcm/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "promotion_gate_panel" in contract["theme"]["components"]


def test_rule_engine_enforces_model_lifecycle_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "deploy_model",
		"owner_assigned": False,
		"target_stage": "production",
		"approval_recorded": False,
		"model_card_present": False,
		"eval_score": 0.3,
		"promotion_requested": True,
		"drift_detected": True,
		"drift_review_recorded": False
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"tenant_context_required",
		"production_promotion_requires_approval",
		"deployment_requires_model_card",
		"low_eval_score_blocks_promotion",
		"drifted_model_requires_review"
	}


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "mlcm"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "mlcm_model_ops_console"
	assert registration["ui_components"]["deployments"] == "/mlcm/deployments"
	assert "aicr" in registration["dependencies"]
	assert "mlcm:deploy" in registration["permissions"]
