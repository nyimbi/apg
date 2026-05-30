"""Regression coverage for the GRPH executable capability contract."""

from __future__ import annotations

import pytest

from capabilities.common.grph import api, register_capability, views
from capabilities.common.grph.capability_contract import evaluate_capability_rules, get_capability_contract
from capabilities.common.grph.service import GrphService


def test_contract_exposes_full_lifecycle_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-graph", {"traversal": {"max_depth": 4}})

	assert contract["capability"] == "grph"
	assert contract["configuration"]["tenant_id"] == "tenant-graph"
	assert contract["configuration"]["traversal"]["max_depth"] == 4
	assert contract["configuration_schema"]["required"] == [
		"tenant_id",
		"schemas",
		"nodes",
		"edges",
		"traversal",
		"lineage",
		"quality",
		"security",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	]
	assert len(contract["rule_engine"]["rules"]) >= 30
	assert {route["name"] for route in contract["ui"]["routes"]} >= {
		"dashboard",
		"explorer",
		"schemas",
		"nodes",
		"edges",
		"traversal",
		"lineage",
		"impact",
		"quality",
		"governance",
		"audit",
		"settings",
	}
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["configuration"]["adapters"]["generated_app_runtime"] == "service.GrphService"
	assert next(route for route in contract["ui"]["routes"] if route["name"] == "audit")["permission"] == "grph:audit"
	assert contract["ui"]["api_prefix"] == "/grph/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert {"graph_canvas", "node_panel", "edge_panel", "traversal_panel", "impact_map", "audit_timeline"} <= set(contract["theme"]["components"])


def test_rule_engine_enforces_graph_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "write_edge",
		"schema_present": False,
		"edge_id_present": False,
		"source_node_present": False,
		"target_node_present": False,
		"edge_type_present": False,
		"owner_assigned": False,
		"classification_present": False,
		"classification_known": False,
		"edge_type_allowed": False,
		"cross_tenant_edge": True,
		"relationship_classification": "restricted",
		"self_edge": True,
		"review_recorded": False,
	})
	traversal_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "traverse",
		"start_node_present": False,
		"traversal_depth": 12,
		"restricted_relationships_in_scope": True,
		"rbac_filter_applied": False,
		"review_recorded": False,
	})
	state_change_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"state_change_requested": True,
		"audit_event_recorded": False,
	})

	assert result["decision"] == "deny"
	assert {
		"tenant_context_required",
		"edge_requires_schema",
		"edge_requires_id",
		"edge_requires_source",
		"edge_requires_target",
		"edge_write_requires_type",
		"edge_requires_owner",
		"edge_requires_classification",
		"edge_classification_requires_review",
		"edge_type_requires_schema_membership",
		"cross_tenant_edge_denied",
		"restricted_relationship_requires_review",
		"self_edge_requires_review",
	} <= set(result["matched_rules"])
	assert traversal_result["decision"] == "deny"
	assert set(traversal_result["matched_rules"]) == {
		"traversal_requires_start_node",
		"deep_traversal_requires_review",
		"restricted_traversal_requires_rbac_filter",
	}
	assert state_change_result["matched_rules"] == ["graph_state_change_requires_audit"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "grph"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["adapters"]["event_stream"] == "bytewax"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "grph_relationship_console"
	assert registration["ui_components"]["schemas"] == "/grph/schemas"
	assert registration["ui_components"]["audit"] == "/grph/audit"
	assert "mdm" in registration["dependencies"]
	assert "impact_analysis" in registration["capabilities"]
	assert "graph_audit" in registration["capabilities"]
	assert "grph:audit" in registration["permissions"]


def test_graph_lifecycle_and_view_models_execute():
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
		labels=["data-bronze"],
		properties={"name": "raw orders"},
		source_asset_id="asset-lakehouse",
	)
	target = service.create_node(
		node_id="dataset-silver",
		tenant_id="tenant-graph",
		schema_id=schema["id"],
		node_type="Dataset",
		owner_id="data-owner",
		labels=["data-silver"],
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
		rbac_filter_applied=True,
	)
	impact = service.impact_analysis("impact-orders", "tenant-graph", source["id"], max_depth=2, rbac_filter_applied=True)
	quality = service.quality_report("quality-lineage", "tenant-graph", schema["id"])

	assert edge["classification"] == "restricted"
	assert traversal["node_ids"] == ["dataset-bronze", "dataset-silver"]
	assert traversal["edge_ids"] == ["edge-transform"]
	assert impact["edge_ids"] == ["edge-transform"]
	assert quality["status"] == "healthy"
	assert service.dashboard_summary("tenant-graph")["audit_event_count"] >= 6
	assert views.dashboard_model(service, "tenant-graph")["summary"]["node_count"] == 2
	assert views.graph_explorer_model(service, "tenant-graph")["edges"][0]["id"] == "edge-transform"
	assert views.schema_manager_model(service, "tenant-graph")["node_type_count"] == 1
	assert views.node_manager_model(service, "tenant-graph")["owner_required"] is True
	assert views.edge_manager_model(service, "tenant-graph")["classifications"][-1] == "restricted"
	assert views.traversal_workbench_model(service, "tenant-graph")["max_depth"] == 8
	assert views.lineage_viewer_model(service, "tenant-graph")["lineage_schemas"][0]["id"] == "schema-lineage"
	assert views.impact_analysis_model(service, "tenant-graph")["traversals"]
	assert views.quality_console_model(service, "tenant-graph")["reports"][0]["id"] == "quality-lineage"
	assert views.governance_model(service, "tenant-graph")["restricted_edges"][0]["id"] == "edge-transform"
	assert views.audit_timeline_model(service, "tenant-graph")["audit_events"]
	assert views.settings_model("tenant-graph")["theme"]["name"] == "grph_relationship_console"


def test_graph_service_enforces_policy_guardrails():
	service = GrphService()

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.create_schema(
			schema_id="schema-no-tenant",
			tenant_id="",
			name="No tenant graph",
			node_types={"Entity": []},
			edge_types={"LINKS_TO": {}},
		)

	with pytest.raises(PermissionError, match="source_asset_required"):
		service.create_schema(
			schema_id="schema-lineage-no-source",
			tenant_id="tenant-graph",
			name="Lineage without source",
			graph_kind="lineage",
			node_types={"Entity": []},
			edge_types={"LINKS_TO": {}},
		)

	with pytest.raises(PermissionError, match="schema_node_types_required"):
		service.create_schema("schema-empty", "tenant-graph", "Empty graph", node_types={}, edge_types={"LINKS_TO": {}})

	schema = service.create_schema(
		schema_id="schema-main",
		tenant_id="tenant-graph",
		name="Main graph",
		node_types={"Entity": []},
		edge_types={"LINKS_TO": {}},
	)

	with pytest.raises(PermissionError, match="node_owner_required"):
		service.create_node("node-no-owner", "tenant-graph", schema["id"], "Entity", "")

	with pytest.raises(PermissionError, match="node_type_not_in_schema"):
		service.create_node("node-bad-type", "tenant-graph", schema["id"], "Unknown", "owner")

	with pytest.raises(PermissionError, match="node_label_review_required"):
		service.create_node("node-bad-label", "tenant-graph", schema["id"], "Entity", "owner", labels=["unreviewed"])

	service.create_node("node-a", "tenant-graph", schema["id"], "Entity", "owner", labels=["entity-a"])
	service.create_node("node-b", "tenant-graph", schema["id"], "Entity", "owner", labels=["entity-b"])

	with pytest.raises(PermissionError, match="edge_type_required"):
		service.create_edge("edge-no-type", "tenant-graph", schema["id"], "node-a", "node-b", "", "owner")

	with pytest.raises(PermissionError, match="restricted_relationship_review_required"):
		service.create_edge(
			edge_id="edge-restricted",
			tenant_id="tenant-graph",
			schema_id=schema["id"],
			from_node_id="node-a",
			to_node_id="node-b",
			edge_type="LINKS_TO",
			owner_id="owner",
			classification="restricted",
			review_recorded=False,
		)

	with pytest.raises(PermissionError, match="deep_traversal_review_required"):
		service.traverse("deep", "tenant-graph", "node-a", max_depth=12, review_recorded=False)

	with pytest.raises(PermissionError, match="start_node_required"):
		service.traverse("missing-start", "tenant-graph", "missing", max_depth=1)


def test_review_evidence_unlocks_review_required_paths():
	service = GrphService()

	reviewed_schema = service.create_schema(
		schema_id="schema-custom",
		tenant_id="tenant-graph",
		name="Reviewed custom graph",
		graph_kind="semantic-network",
		node_types={"Entity": []},
		edge_types={"LINKS_TO": {}},
		review_recorded=True,
	)
	assert reviewed_schema["graph_kind"] == "property"

	node = service.create_node(
		"node-reviewed-label",
		"tenant-graph",
		reviewed_schema["id"],
		"Entity",
		"owner",
		labels=["custom-label"],
		review_recorded=True,
	)
	assert node["labels"] == ["custom-label"]

	service.create_edge("edge-internal", "tenant-graph", reviewed_schema["id"], node["id"], node["id"], "LINKS_TO", "owner", review_recorded=True)
	with pytest.raises(PermissionError, match="edge_classification_review_required"):
		service.create_edge("edge-unknown-class", "tenant-graph", reviewed_schema["id"], node["id"], node["id"], "LINKS_TO", "owner", classification="regulated", review_recorded=False)
	reviewed_edge = service.create_edge("edge-reviewed-class", "tenant-graph", reviewed_schema["id"], node["id"], node["id"], "LINKS_TO", "owner", classification="regulated", review_recorded=True)
	reviewed_traversal = service.traverse("deep-reviewed", "tenant-graph", node["id"], max_depth=12, review_recorded=True)
	assert reviewed_edge["classification"] == "restricted"
	assert reviewed_traversal["max_depth"] == 12

	quality_result = service.evaluate({
		"tenant_context_present": True,
		"operation": "quality_report",
		"quality_issue_count": 51,
		"review_recorded": True,
	})
	assert quality_result["decision"] == "allow"


def test_api_helpers_expose_graph_surfaces():
	local_service = GrphService()
	api.SERVICE = local_service

	schema = api.create_schema({
		"id": "schema-api",
		"tenant_id": "tenant-api",
		"name": "API graph",
		"node_types": {"Entity": []},
		"edge_types": {"LINKS_TO": {}},
	})
	node_a = api.create_node({
		"id": "node-api-a",
		"tenant_id": "tenant-api",
		"schema_id": schema["id"],
		"node_type": "Entity",
		"owner_id": "owner",
		"labels": ["entity-a"],
	})
	node_b = api.create_node({
		"id": "node-api-b",
		"tenant_id": "tenant-api",
		"schema_id": schema["id"],
		"node_type": "Entity",
		"owner_id": "owner",
		"labels": ["entity-b"],
	})
	api.create_edge({
		"id": "edge-api",
		"tenant_id": "tenant-api",
		"schema_id": schema["id"],
		"from_node_id": node_a["id"],
		"to_node_id": node_b["id"],
		"edge_type": "LINKS_TO",
		"owner_id": "owner",
	})
	traversal = api.traverse({"id": "trav-api", "tenant_id": "tenant-api", "start_node_id": node_a["id"]})
	quality = api.quality_report({"id": "quality-api", "tenant_id": "tenant-api", "schema_id": schema["id"]})
	status = api.capability_status("tenant-api")
	graph_data = api.list_graph_data("tenant-api")

	assert traversal["node_ids"] == ["node-api-a", "node-api-b"]
	assert quality["status"] == "healthy"
	assert status["summary"]["edge_count"] == 1
	assert graph_data["summary"]["audit_event_count"] >= 5
