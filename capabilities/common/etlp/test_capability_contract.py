"""Regression coverage for the ETLP executable capability contract."""

from capabilities.common.etlp import register_capability
from capabilities.common.etlp.capability_contract import (
	evaluate_capability_rules,
	get_capability_contract
)


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-pipeline", {"pipelines": {"max_concurrent_executions": 4}})

	assert contract["capability"] == "etlp"
	assert contract["configuration"]["tenant_id"] == "tenant-pipeline"
	assert contract["configuration"]["pipelines"]["max_concurrent_executions"] == 4
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"pipelines",
		"processing",
		"quality",
		"governance",
		"optimization",
		"ui",
		"theme"
	]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"pipelines",
		"designer",
		"field_mapper",
		"executions",
		"quality",
		"datasources",
		"settings"
	}
	assert contract["ui"]["api_prefix"] == "/etlp/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "field_mapping_canvas" in contract["theme"]["components"]


def test_rule_engine_enforces_pipeline_guardrails():
	execution_result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "execute_pipeline",
		"owner_assigned": False,
		"environment": "production",
		"approval_recorded": False,
		"transformation_present": True,
		"lineage_emitted": False,
		"estimated_cost": 1500.0,
		"cost_review_recorded": False
	})
	publish_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "publish_output",
		"quality_gate_passed": False
	})

	assert execution_result["decision"] == "deny"
	assert set(execution_result["matched_rules"]) == {
		"tenant_context_required",
		"pipeline_execution_requires_owner",
		"production_execution_requires_approval",
		"lineage_required_for_transformations",
		"high_cost_execution_requires_review"
	}
	assert publish_result["decision"] == "deny"
	assert publish_result["matched_rules"] == ["publish_requires_quality_gate"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "etlp_pipeline_console"
	assert registration["ui_components"]["field_mapper"] == "/etlp/field-mapper"
	assert "metadata" in registration["dependencies"]
