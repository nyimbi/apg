"""Regression coverage for the MDM executable capability contract."""

from capabilities.common.mdm import register_capability
from capabilities.common.mdm.capability_contract import (
	evaluate_capability_rules,
	get_capability_contract
)


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract(
		"tenant-master",
		{"quality": {"minimum_quality_score": 88.0}}
	)

	assert contract["capability"] == "mdm"
	assert contract["configuration"]["tenant_id"] == "tenant-master"
	assert contract["configuration"]["quality"]["minimum_quality_score"] == 88.0
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"entities",
		"quality",
		"matching",
		"governance",
		"integration",
		"ui",
		"theme"
	]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"entities",
		"golden_records",
		"quality",
		"duplicates",
		"stewardship",
		"analytics",
		"settings"
	}
	assert contract["ui"]["api_prefix"] == "/mdm/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "golden_record_card" in contract["theme"]["components"]


def test_rule_engine_enforces_master_data_guardrails():
	publish_result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "publish_entity",
		"data_owner_assigned": False,
		"quality_score": 40.0,
		"duplicate_confidence": 82.0,
		"steward_review_recorded": False,
		"entity_classification": "restricted",
		"audit_evidence_present": False
	})

	merge_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "merge_golden_record",
		"survivorship_policy_present": False
	})

	assert publish_result["decision"] == "deny"
	assert set(publish_result["matched_rules"]) == {
		"tenant_context_required",
		"entity_publish_requires_data_owner",
		"low_quality_blocks_publish",
		"duplicate_candidates_require_review",
		"restricted_entity_requires_audit_trail"
	}
	assert merge_result["decision"] == "deny"
	assert merge_result["matched_rules"] == ["golden_record_merge_requires_survivorship"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "mdm_golden_record_console"
	assert registration["ui_components"]["duplicates"] == "/mdm/duplicates"
	assert "mten" in registration["dependencies"]
