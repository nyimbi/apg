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
		"agents",
		"streaming",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	]
	assert len(contract["rule_engine"]["rules"]) >= 43
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
		"agents",
		"lifecycle",
		"audit",
		"settings",
	}
	assert contract["provides"] == ["graph_data_management", "relationship_intelligence", "graph_agent_composition"]
	assert contract["requires"] == ["mdm", "meta", "etlp", "srch", "aicr", "conf"]
	assert contract["agents"]["first_class"] is True
	assert contract["agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert "lineage_reviewer" in contract["agents"]["privileged_roles"]
	assert contract["streaming"]["required_processor"] == "bytewax"
	assert "graph_agent_batch" in contract["streaming"]["required_operations"]
	assert contract["configuration"]["adapters"]["event_stream"] == "bytewax"
	assert contract["configuration"]["adapters"]["generated_app_runtime"] == "service.GrphService"
	assert next(route for route in contract["ui"]["routes"] if route["name"] == "audit")["permission"] == "grph:audit"
	assert contract["ui"]["api_prefix"] == "/grph/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert {"graph_canvas", "node_panel", "edge_panel", "traversal_panel", "impact_map", "graph_agent_roster", "bytewax_lifecycle_panel", "audit_timeline"} <= set(contract["theme"]["components"])


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
	agent_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "register_graph_agent",
		"agent_runtime_supported": False,
		"agent_role_supported": False,
		"scope_present": False,
		"owner_present": False,
		"purpose_present": False,
		"contribution_disclosed": False,
		"privileged_role": True,
		"human_approval_required": False,
	})
	lifecycle_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "validate_grph_lifecycle_batch",
		"event_stream": "legacy_queue",
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
	assert agent_result["decision"] == "deny"
	assert {
		"graph_agent_runtime_supported",
		"graph_agent_role_supported",
		"graph_agent_requires_scope",
		"graph_agent_requires_owner",
		"graph_agent_requires_purpose",
		"graph_agent_requires_contribution_disclosure",
		"graph_agent_privileged_role_requires_human_approval",
	} <= set(agent_result["matched_rules"])
	assert lifecycle_result["decision"] == "deny"
	assert lifecycle_result["matched_rules"] == ["bytewax_grph_stream_required"]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "grph"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["adapters"]["event_stream"] == "bytewax"
	assert registration["agents"]["first_class"] is True
	assert registration["streaming"]["required_processor"] == "bytewax"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "grph_relationship_console"
	assert registration["ui_components"]["schemas"] == "/grph/schemas"
	assert registration["ui_components"]["agents"] == "/grph/agents"
	assert registration["ui_components"]["lifecycle"] == "/grph/lifecycle"
	assert registration["ui_components"]["audit"] == "/grph/audit"
	assert "mdm" in registration["dependencies"]
	assert "aicr" in registration["dependencies"]
	assert "impact_analysis" in registration["capabilities"]
	assert "graph_agent_composition" in registration["capabilities"]
	assert "graph_lifecycle_batches" in registration["capabilities"]
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
	agent = service.register_graph_agent(
		agent_id="graph-agent-1",
		tenant_id="tenant-graph",
		name="Graph Steward",
		runtime="codex",
		role="graph_steward",
		scope="schema node edge lineage quality review",
		owner="graph-platform",
		purpose="govern lineage and relationship quality",
	)
	batch = service.validate_grph_lifecycle_batch(
		tenant_id="tenant-graph",
		event_stream="bytewax",
		mutation_count=3,
		operation="graph_agent_batch",
		batch_id="grph-batch-001",
	)

	assert edge["classification"] == "restricted"
	assert traversal["node_ids"] == ["dataset-bronze", "dataset-silver"]
	assert traversal["edge_ids"] == ["edge-transform"]
	assert impact["edge_ids"] == ["edge-transform"]
	assert quality["status"] == "healthy"
	assert agent["runtime"] == "codex"
	assert agent["status"] == "active"
	assert batch["required_processor"] == "bytewax"
	assert service.dashboard_summary("tenant-graph")["graph_agent_count"] == 1
	assert service.dashboard_summary("tenant-graph")["lifecycle_batch_count"] == 1
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
	assert views.governance_model(service, "tenant-graph")["agents"]["first_class"] is True
	assert views.graph_agent_roster_model(service, "tenant-graph")["agents"][0]["id"] == "graph-agent-1"
	assert views.lifecycle_batch_model(service, "tenant-graph")["batches"][0]["id"] == "grph-batch-001"
	assert views.audit_timeline_model(service, "tenant-graph")["audit_events"]
	assert views.settings_model("tenant-graph")["theme"]["name"] == "grph_relationship_console"
	assert views.settings_model("tenant-graph")["streaming"]["required_processor"] == "bytewax"


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


def test_graph_agent_and_lifecycle_guardrails_execute():
	service = GrphService()
	tenant_id = "tenant-agents"

	with pytest.raises(PermissionError, match="unsupported_graph_agent_runtime"):
		service.register_graph_agent(
			"agent-bad-runtime",
			tenant_id,
			"Bad Runtime",
			"unknown",
			"graph_steward",
			"schema review",
			"owner",
			"purpose",
		)

	with pytest.raises(PermissionError, match="graph_agent_scope_required"):
		service.register_graph_agent("agent-no-scope", tenant_id, "No Scope", "codex", "graph_steward", "", "owner", "purpose")

	with pytest.raises(PermissionError, match="graph_agent_contribution_disclosure_required"):
		service.register_graph_agent(
			"agent-no-disclosure",
			tenant_id,
			"No Disclosure",
			"codex",
			"graph_steward",
			"schema review",
			"owner",
			"purpose",
			contribution_disclosed=False,
		)

	agent = service.register_graph_agent(
		"agent-privileged",
		tenant_id,
		"Privileged Agent",
		"claude-code",
		"lineage reviewer",
		"lineage and impact policy",
		"owner",
		"review privileged graph decisions",
		human_approval_required=False,
	)
	assert agent["runtime"] == "claude_code"
	assert agent["role"] == "lineage_reviewer"
	assert agent["status"] == "pending_review"
	assert service.dashboard_summary(tenant_id)["pending_agent_review_count"] == 1

	with pytest.raises(ValueError, match="grph_lifecycle_batch_empty"):
		service.validate_grph_lifecycle_batch(tenant_id, "bytewax", 0)

	with pytest.raises(ValueError, match="unsupported_grph_lifecycle_operation"):
		service.validate_grph_lifecycle_batch(tenant_id, "bytewax", 1, "unknown_batch")

	with pytest.raises(PermissionError, match="bytewax_lifecycle_stream_required"):
		service.validate_grph_lifecycle_batch(tenant_id, "legacy_queue", 1)

	assert service.dashboard_summary(tenant_id)["denied_lifecycle_batch_count"] == 1


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
	agent = api.register_graph_agent({
		"id": "api-agent",
		"tenant_id": "tenant-api",
		"name": "API Graph Agent",
		"runtime": "opencode",
		"role": "quality_reviewer",
		"scope": "quality and edge review",
		"owner": "owner",
		"purpose": "review graph quality",
	})
	batch = api.validate_grph_lifecycle_batch({
		"id": "api-batch",
		"tenant_id": "tenant-api",
		"event_stream": "bytewax",
		"mutation_count": 2,
		"operation": "quality_batch",
	})
	status = api.capability_status("tenant-api")
	graph_data = api.list_graph_data("tenant-api")

	assert traversal["node_ids"] == ["node-api-a", "node-api-b"]
	assert quality["status"] == "healthy"
	assert agent["id"] == "api-agent"
	assert batch["status"] == "accepted"
	assert api.list_graph_agents("tenant-api")[0]["id"] == "api-agent"
	assert api.list_lifecycle_batches("tenant-api")[0]["id"] == "api-batch"
	assert status["summary"]["edge_count"] == 1
	assert status["agent_count"] == 1
	assert graph_data["summary"]["audit_event_count"] >= 5
	assert graph_data["graph_agents"][0]["runtime"] == "opencode"
