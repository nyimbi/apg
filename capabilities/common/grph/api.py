"""API helpers for the Graph Data Management capability."""

from __future__ import annotations

from typing import Any

from .service import GrphService


SERVICE = GrphService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"record_count": len(SERVICE.list_records(tenant_id)),
		"agent_count": len(SERVICE.list_graph_agents(tenant_id)),
		"lifecycle_batch_count": len(SERVICE.list_lifecycle_batches(tenant_id)),
		"summary": SERVICE.dashboard_summary(tenant_id),
	}


def create_schema(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_schema(
		schema_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		graph_kind=str(payload.get("graph_kind") or "property"),
		node_types=dict(payload.get("node_types") or {}),
		edge_types=dict(payload.get("edge_types") or {}),
		source_asset_id=payload.get("source_asset_id"),
		review_recorded=bool(payload.get("review_recorded", False)),
	)


def create_node(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_node(
		node_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		schema_id=str(payload["schema_id"]),
		node_type=str(payload["node_type"]),
		owner_id=str(payload["owner_id"]),
		labels=list(payload.get("labels") or []),
		properties=dict(payload.get("properties") or {}),
		source_asset_id=payload.get("source_asset_id"),
		review_recorded=bool(payload.get("review_recorded", False)),
	)


def create_edge(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_edge(
		edge_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		schema_id=str(payload["schema_id"]),
		from_node_id=str(payload["from_node_id"]),
		to_node_id=str(payload["to_node_id"]),
		edge_type=str(payload["edge_type"]),
		owner_id=str(payload["owner_id"]),
		classification=str(payload.get("classification") or "internal"),
		properties=dict(payload.get("properties") or {}),
		review_recorded=bool(payload.get("review_recorded", False)),
	)


def traverse(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.traverse(
		traversal_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		start_node_id=str(payload["start_node_id"]),
		max_depth=int(payload.get("max_depth", 1)),
		review_recorded=bool(payload.get("review_recorded", False)),
		rbac_filter_applied=bool(payload.get("rbac_filter_applied", True)),
	)


def lineage_path(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.lineage_path(
		traversal_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		source_asset_id=str(payload.get("source_asset_id") or ""),
		start_node_id=str(payload["start_node_id"]),
		max_depth=int(payload.get("max_depth", 2)),
		review_recorded=bool(payload.get("review_recorded", False)),
		rbac_filter_applied=bool(payload.get("rbac_filter_applied", True)),
	)


def impact_analysis(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.impact_analysis(
		traversal_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		start_node_id=str(payload["start_node_id"]),
		max_depth=int(payload.get("max_depth", 3)),
		review_recorded=bool(payload.get("review_recorded", False)),
		rbac_filter_applied=bool(payload.get("rbac_filter_applied", True)),
	)


def neighborhood(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.neighborhood(
		traversal_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		start_node_id=str(payload["start_node_id"]),
		review_recorded=bool(payload.get("review_recorded", False)),
		rbac_filter_applied=bool(payload.get("rbac_filter_applied", True)),
	)


def quality_report(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.quality_report(
		report_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		schema_id=str(payload["schema_id"]),
		review_recorded=bool(payload.get("review_recorded", False)),
	)


def retire_schema(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.retire_schema(
		tenant_id=str(payload.get("tenant_id") or "default"),
		schema_id=str(payload["schema_id"]),
		review_recorded=bool(payload.get("review_recorded", False)),
	)


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_record(
		record_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		metadata=dict(payload.get("metadata") or {}),
		status=str(payload.get("status") or "active"),
	)


def register_graph_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_graph_agent(
		agent_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		runtime=str(payload["runtime"]),
		role=str(payload["role"]),
		scope=str(payload["scope"]),
		owner=str(payload["owner"]),
		purpose=str(payload["purpose"]),
		contribution_disclosed=bool(payload.get("contribution_disclosed", True)),
		human_approval_required=bool(payload.get("human_approval_required", False)),
	)


def validate_grph_lifecycle_batch(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.validate_grph_lifecycle_batch(
		tenant_id=str(payload.get("tenant_id") or "default"),
		event_stream=str(payload.get("event_stream") or "bytewax"),
		mutation_count=int(payload.get("mutation_count", 1)),
		operation=str(payload.get("operation") or "graph_agent_batch"),
		batch_id=payload.get("id") or payload.get("batch_id"),
	)


def list_records(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_records(tenant_id)


def list_graph_agents(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_graph_agents(tenant_id)


def list_lifecycle_batches(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_lifecycle_batches(tenant_id)


def dashboard_summary(tenant_id: str | None = None) -> dict[str, Any]:
	return SERVICE.dashboard_summary(tenant_id)


def list_graph_data(tenant_id: str = "default") -> dict[str, Any]:
	return {
		"schemas": SERVICE.list_schemas(tenant_id),
		"nodes": SERVICE.list_nodes(tenant_id),
		"edges": SERVICE.list_edges(tenant_id),
		"traversals": SERVICE.list_traversals(tenant_id),
		"quality_reports": SERVICE.list_quality_reports(tenant_id),
		"graph_agents": SERVICE.list_graph_agents(tenant_id),
		"lifecycle_batches": SERVICE.list_lifecycle_batches(tenant_id),
		"audit_events": SERVICE.list_audit_events(tenant_id),
		"summary": SERVICE.dashboard_summary(tenant_id),
	}
