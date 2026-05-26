"""Regression coverage for the GRPH executable capability contract."""

from capabilities.common.grph import register_capability
from capabilities.common.grph.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-graph", {"graph": {"max_traversal_depth": 4}})

	assert contract["capability"] == "grph"
	assert contract["configuration"]["tenant_id"] == "tenant-graph"
	assert contract["configuration"]["graph"]["max_traversal_depth"] == 4
	assert contract["configuration_schema"]["required"] == ["tenant_id", "graph", "storage", "governance", "ui", "theme"]
	assert len(contract["rule_engine"]["rules"]) >= 6
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "explorer", "schema", "lineage", "quality", "settings"}
	assert contract["ui"]["api_prefix"] == "/grph/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "graph_canvas" in contract["theme"]["components"]


def test_rule_engine_enforces_graph_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "write_edge",
		"owner_assigned": False,
		"edge_type_present": False,
		"relationship_classification": "restricted",
		"review_recorded": False,
		"traversal_depth": 12,
		"graph_type": "lineage",
		"source_asset_present": False
	})
	node_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "write_node",
		"owner_assigned": False
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) == {
		"tenant_context_required",
		"edge_write_requires_type",
		"restricted_relationship_requires_review",
		"deep_traversal_requires_review",
		"lineage_graph_requires_source_asset"
	}
	assert node_result["decision"] == "deny"
	assert node_result["matched_rules"] == ["node_write_requires_owner"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "grph"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "grph_relationship_console"
	assert registration["ui_components"]["schema"] == "/grph/schema"
	assert "mdm" in registration["dependencies"]
	assert "grph:query" in registration["permissions"]
