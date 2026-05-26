"""Regression coverage for the KNGR executable capability contract."""

from capabilities.common.kngr import register_capability
from capabilities.common.kngr.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-knowledge", {"reasoning": {"max_reasoning_depth": 3}})

	assert contract["capability"] == "kngr"
	assert contract["configuration"]["tenant_id"] == "tenant-knowledge"
	assert contract["configuration"]["reasoning"]["max_reasoning_depth"] == 3
	assert contract["configuration_schema"]["required"] == ["tenant_id", "knowledge", "reasoning", "governance", "ui", "theme"]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "entities", "curation", "reasoning", "context", "settings"}
	assert contract["ui"]["api_prefix"] == "/kngr/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "semantic_graph" in contract["theme"]["components"]


def test_rule_engine_enforces_knowledge_graph_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "reason",
		"source_evidence_present": False,
		"confidence_score": 0.2,
		"evidence_links_present": False,
		"reasoning_depth": 9,
		"review_recorded": False,
		"curation_recorded": False
	})
	publish_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "publish_graph",
		"curation_recorded": False
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"tenant_context_required",
		"reasoning_requires_evidence",
		"deep_reasoning_requires_review"
	}
	assert publish_result["decision"] == "deny"
	assert publish_result["matched_rules"] == ["uncurated_public_graph_blocked"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "kngr"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "kngr_semantic_console"
	assert registration["ui_components"]["reasoning"] == "/kngr/reasoning"
	assert "grph" in registration["dependencies"]
	assert "kngr:reason" in registration["permissions"]
