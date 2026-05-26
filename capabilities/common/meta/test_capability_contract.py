"""Regression coverage for the META executable capability contract."""

from capabilities.common.meta import register_capability
from capabilities.common.meta.capability_contract import (
	evaluate_capability_rules,
	get_capability_contract
)


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract(
		"tenant-catalog",
		{"quality": {"minimum_certification_score": 92.0}}
	)

	assert contract["capability"] == "meta"
	assert contract["configuration"]["tenant_id"] == "tenant-catalog"
	assert contract["configuration"]["quality"]["minimum_certification_score"] == 92.0
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"catalog",
		"discovery",
		"classification",
		"lineage",
		"quality",
		"governance",
		"ui",
		"theme"
	]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"catalog",
		"discovery",
		"lineage",
		"classification",
		"quality",
		"search",
		"settings"
	}
	assert contract["ui"]["api_prefix"] == "/meta/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "lineage_graph_viewer" in contract["theme"]["components"]


def test_rule_engine_enforces_metadata_governance_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "publish_asset",
		"asset_owner_assigned": False,
		"asset_sensitivity": "restricted",
		"classification_complete": False,
		"certification_requested": True,
		"lineage_available": False,
		"classification_confidence": 0.5,
		"steward_review_recorded": False,
		"asset_age_days": 120,
		"freshness_review_recorded": False
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"tenant_context_required",
		"published_asset_requires_owner",
		"restricted_asset_requires_classification",
		"certified_asset_requires_lineage",
		"low_classification_confidence_requires_review",
		"stale_asset_requires_review"
	}


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "meta_catalog_console"
	assert registration["ui_components"]["lineage"] == "/meta/lineage"
	assert "mdm" in registration["dependencies"]
