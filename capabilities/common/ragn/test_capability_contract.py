"""Regression coverage for the RAGN executable capability contract."""

from capabilities.common.ragn import register_capability
from capabilities.common.ragn.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-rag", {"retrieval": {"minimum_context_confidence": 0.8}})

	assert contract["capability"] == "ragn"
	assert contract["configuration"]["tenant_id"] == "tenant-rag"
	assert contract["configuration"]["retrieval"]["minimum_context_confidence"] == 0.8
	assert contract["configuration_schema"]["required"] == ["tenant_id", "knowledge_bases", "retrieval", "generation", "governance", "ui", "theme"]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "studio", "knowledge_bases", "documents", "conversations", "curation", "settings"}
	assert contract["ui"]["api_prefix"] == "/ragn/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "answer_panel" in contract["theme"]["components"]


def test_rule_engine_enforces_rag_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "generate_answer",
		"owner_assigned": False,
		"source_classification": "restricted",
		"access_filter_applied": False,
		"citations_attached": False,
		"context_confidence": 0.2,
		"review_recorded": False,
		"model_location": "external",
		"model_policy_attached": False
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"tenant_context_required",
		"restricted_sources_require_filter",
		"generation_requires_citations",
		"low_context_confidence_requires_review",
		"external_model_requires_policy"
	}


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "ragn"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "ragn_answer_studio"
	assert registration["ui_components"]["studio"] == "/ragn/studio"
	assert "srch" in registration["dependencies"]
	assert "ragn:query" in registration["permissions"]
