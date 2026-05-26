"""Regression coverage for the AICR executable capability contract."""

from capabilities.common.aicr import register_capability
from capabilities.common.aicr.capability_contract import (
	evaluate_capability_rules,
	get_capability_contract
)


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-ai", {"inference": {"max_concurrent_requests": 32}})

	assert contract["capability"] == "aicr"
	assert contract["configuration"]["tenant_id"] == "tenant-ai"
	assert contract["configuration"]["inference"]["max_concurrent_requests"] == 32
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"services",
		"inference",
		"orchestration",
		"governance",
		"ui",
		"theme"
	]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"services",
		"inference",
		"models",
		"workflows",
		"governance",
		"metrics",
		"settings"
	}
	assert contract["ui"]["api_prefix"] == "/aicr/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "workflow_graph" in contract["theme"]["components"]


def test_rule_engine_enforces_ai_core_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "register_service",
		"owner_assigned": False,
		"workflow_risk": "high",
		"approval_recorded": False,
		"context_tokens": 256000,
		"review_recorded": False
	})
	inference_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "run_inference",
		"model_policy_attached": False
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"tenant_context_required",
		"service_registration_requires_owner",
		"high_risk_workflow_requires_approval",
		"large_context_requires_review"
	}
	assert inference_result["decision"] == "deny"
	assert inference_result["matched_rules"] == ["inference_requires_model_policy"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "aicr"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "aicr_ai_control_console"
	assert registration["ui_components"]["services"] == "/aicr/services"
	assert "auth" in registration["dependencies"]
	assert "aicr:run_inference" in registration["permissions"]
