"""Regression coverage for the IMEX executable capability contract."""

from capabilities.common.imex import imex_capability, register_capability
from capabilities.common.imex.capability_contract import (
	evaluate_capability_rules,
	get_capability_contract
)


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-transfer", {"jobs": {"max_concurrent_jobs": 5}})

	assert contract["capability"] == "imex"
	assert contract["configuration"]["tenant_id"] == "tenant-transfer"
	assert contract["configuration"]["jobs"]["max_concurrent_jobs"] == 5
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"jobs",
		"formats",
		"validation",
		"security",
		"orchestration",
		"ui",
		"theme"
	]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"jobs",
		"designer",
		"mappings",
		"monitor",
		"validation",
		"workflows",
		"settings"
	}
	assert contract["ui"]["api_prefix"] == "/imex/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "schema_mapping_canvas" in contract["theme"]["components"]


def test_rule_engine_enforces_transfer_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "execute_job",
		"owner_assigned": False,
		"environment": "production",
		"approval_recorded": False,
		"data_classification": "sensitive",
		"export_encrypted": False,
		"preview_validated": False,
		"quality_score": 55.0,
		"quality_review_recorded": False
	})
	export_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "export",
		"data_classification": "sensitive",
		"export_encrypted": False
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"tenant_context_required",
		"job_execution_requires_owner",
		"production_transfer_requires_approval",
		"execution_requires_preview_validation",
		"low_quality_transfer_requires_review"
	}
	assert export_result["decision"] == "deny"
	assert export_result["matched_rules"] == ["sensitive_export_requires_encryption"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "imex_transfer_console"
	assert registration["ui_components"]["mappings"] == "/imex/mappings"
	assert "etlp" in registration["dependencies"]
	assert callable(imex_capability.health_check)
