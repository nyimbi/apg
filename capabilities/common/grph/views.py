"""UI metadata and view-model helpers for graph data management."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import GrphService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
	service: GrphService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or GrphService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"routes": capability_routes(tenant_id),
		"summary": service.dashboard_summary(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
	}


def graph_explorer_model(service: GrphService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or GrphService()
	return {
		"title": "Graph Explorer",
		"schemas": service.list_schemas(tenant_id),
		"nodes": service.list_nodes(tenant_id),
		"edges": service.list_edges(tenant_id),
		"traversals": service.list_traversals(tenant_id),
	}


def schema_manager_model(service: GrphService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or GrphService()
	return {
		"title": "Graph Schema Manager",
		"schemas": service.list_schemas(tenant_id),
		"node_type_count": sum(len(schema["node_types"]) for schema in service.list_schemas(tenant_id)),
		"edge_type_count": sum(len(schema["edge_types"]) for schema in service.list_schemas(tenant_id)),
	}


def lineage_viewer_model(service: GrphService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or GrphService()
	return {
		"title": "Lineage Graph Viewer",
		"lineage_schemas": [schema for schema in service.list_schemas(tenant_id) if schema["graph_kind"] == "lineage"],
		"traversals": service.list_traversals(tenant_id),
	}


def quality_console_model(service: GrphService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or GrphService()
	return {
		"title": "Graph Quality Console",
		"reports": service.list_quality_reports(tenant_id),
		"summary": service.dashboard_summary(tenant_id),
	}
