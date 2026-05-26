"""Regression coverage for the SRCH executable capability contract."""

from capabilities.common.srch import register_capability
from capabilities.common.srch.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-search", {"query": {"max_result_window": 250}})

	assert contract["capability"] == "srch"
	assert contract["configuration"]["tenant_id"] == "tenant-search"
	assert contract["configuration"]["query"]["max_result_window"] == 250
	assert contract["configuration_schema"]["required"] == ["tenant_id", "indexing", "query", "governance", "ui", "theme"]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "search", "indices", "documents", "analytics", "governance", "settings"}
	assert contract["ui"]["api_prefix"] == "/srch/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "result_card" in contract["theme"]["components"]


def test_rule_engine_enforces_search_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "bulk_index",
		"owner_assigned": False,
		"content_classification": "restricted",
		"rbac_filter_applied": False,
		"query_type": "semantic",
		"embedding_index_ready": False,
		"result_window": 5000,
		"review_recorded": False,
		"source_lineage_present": False
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"tenant_context_required",
		"restricted_query_requires_rbac_filter",
		"semantic_query_requires_embeddings",
		"large_result_window_requires_review",
		"bulk_index_requires_lineage"
	}


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "srch"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "srch_discovery_console"
	assert registration["ui_components"]["indices"] == "/srch/indices"
	assert "nlpc" in registration["dependencies"]
	assert "srch:query" in registration["permissions"]
