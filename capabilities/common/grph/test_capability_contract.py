"""Regression coverage for the GRPH executable capability contract."""

import pytest

from capabilities.common.grph import register_capability
from capabilities.common.grph.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.grph.service import GrphService
from capabilities.common.grph.views import (
	dashboard_model,
	graph_explorer_model,
	lineage_viewer_model,
	quality_console_model,
	schema_manager_model,
)


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


def test_graph_lifecycle_is_executable():
	service = GrphService()

	schema = service.create_schema(
		schema_id="schema-lineage",
		tenant_id="tenant-graph",
		name="Data lineage graph",
		graph_kind="lineage",
		node_types={"Dataset": ["name", "system"]},
		edge_types={"DERIVES_FROM": {"classification": "restricted"}},
		source_asset_id="asset-lakehouse",
	)
	source = service.create_node(
		node_id="dataset-bronze",
		tenant_id="tenant-graph",
		schema_id=schema["id"],
		node_type="Dataset",
		owner_id="data-owner",
		labels=["bronze"],
		properties={"name": "raw orders"},
		source_asset_id="asset-lakehouse",
	)
	target = service.create_node(
		node_id="dataset-silver",
		tenant_id="tenant-graph",
		schema_id=schema["id"],
		node_type="Dataset",
		owner_id="data-owner",
		labels=["silver"],
		properties={"name": "clean orders"},
		source_asset_id="asset-lakehouse",
	)
	edge = service.create_edge(
		edge_id="edge-transform",
		tenant_id="tenant-graph",
		schema_id=schema["id"],
		from_node_id=source["id"],
		to_node_id=target["id"],
		edge_type="DERIVES_FROM",
		owner_id="data-owner",
		classification="restricted",
		review_recorded=True,
	)
	traversal = service.lineage_path(
		traversal_id="lineage-orders",
		tenant_id="tenant-graph",
		source_asset_id="asset-lakehouse",
		start_node_id=source["id"],
		max_depth=2,
	)
	quality = service.quality_report("quality-lineage", "tenant-graph", schema["id"])

	assert edge["classification"] == "restricted"
	assert traversal["node_ids"] == ["dataset-bronze", "dataset-silver"]
	assert traversal["edge_ids"] == ["edge-transform"]
	assert quality["status"] == "healthy"
	assert service.dashboard_summary("tenant-graph") == {
		"tenant_id": "tenant-graph",
		"schema_count": 1,
		"node_count": 2,
		"edge_count": 1,
		"restricted_edge_count": 1,
		"traversal_count": 1,
		"quality_report_count": 1,
	}
	assert dashboard_model(service, "tenant-graph")["summary"]["node_count"] == 2
	assert graph_explorer_model(service, "tenant-graph")["edges"][0]["id"] == "edge-transform"
	assert schema_manager_model(service, "tenant-graph")["node_type_count"] == 1
	assert lineage_viewer_model(service, "tenant-graph")["lineage_schemas"][0]["id"] == "schema-lineage"
	assert quality_console_model(service, "tenant-graph")["reports"][0]["id"] == "quality-lineage"


def test_graph_service_enforces_policy_guardrails():
	service = GrphService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.create_schema(
			schema_id="schema-no-tenant",
			tenant_id="",
			name="No tenant graph",
		)

	with pytest.raises(PermissionError, match="source_asset_required"):
		service.create_schema(
			schema_id="schema-lineage-no-source",
			tenant_id="tenant-graph",
			name="Lineage without source",
			graph_kind="lineage",
		)

	service.create_schema(
		schema_id="schema-main",
		tenant_id="tenant-graph",
		name="Main graph",
		node_types={"Entity": []},
		edge_types={"LINKS_TO": {}},
	)

	with pytest.raises(PermissionError, match="node_owner_required"):
		service.create_node(
			node_id="node-no-owner",
			tenant_id="tenant-graph",
			schema_id="schema-main",
			node_type="Entity",
			owner_id="",
		)

	with pytest.raises(ValueError, match="node_type_not_in_schema"):
		service.create_node(
			node_id="node-bad-type",
			tenant_id="tenant-graph",
			schema_id="schema-main",
			node_type="Unknown",
			owner_id="owner",
		)

	service.create_node("node-a", "tenant-graph", "schema-main", "Entity", "owner")
	service.create_node("node-b", "tenant-graph", "schema-main", "Entity", "owner")

	with pytest.raises(PermissionError, match="edge_type_required"):
		service.create_edge(
			edge_id="edge-no-type",
			tenant_id="tenant-graph",
			schema_id="schema-main",
			from_node_id="node-a",
			to_node_id="node-b",
			edge_type="",
			owner_id="owner",
		)

	with pytest.raises(PermissionError, match="restricted_relationship_review_required"):
		service.create_edge(
			edge_id="edge-restricted",
			tenant_id="tenant-graph",
			schema_id="schema-main",
			from_node_id="node-a",
			to_node_id="node-b",
			edge_type="LINKS_TO",
			owner_id="owner",
			classification="restricted",
			review_recorded=False,
		)

	with pytest.raises(PermissionError, match="deep_traversal_review_required"):
		service.traverse(
			traversal_id="deep",
			tenant_id="tenant-graph",
			start_node_id="node-a",
			max_depth=12,
			review_recorded=False,
		)

	with pytest.raises(PermissionError, match="schema_missing"):
		service.create_node(
			node_id="node-cross",
			tenant_id="tenant-other",
			schema_id="schema-main",
			node_type="Entity",
			owner_id="owner",
		)
