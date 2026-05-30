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
		"route": "/grph/explorer",
		"tenant_id": tenant_id,
		"schemas": service.list_schemas(tenant_id),
		"nodes": service.list_nodes(tenant_id),
		"edges": service.list_edges(tenant_id),
		"traversals": service.list_traversals(tenant_id),
	}


def schema_manager_model(service: GrphService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or GrphService()
	return {
		"route": "/grph/schemas",
		"tenant_id": tenant_id,
		"schemas": service.list_schemas(tenant_id),
		"graph_kinds": service.describe(tenant_id)["configuration"]["schemas"]["allowed_graph_kinds"],
		"node_type_count": sum(len(schema["node_types"]) for schema in service.list_schemas(tenant_id)),
		"edge_type_count": sum(len(schema["edge_types"]) for schema in service.list_schemas(tenant_id)),
	}


def node_manager_model(service: GrphService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or GrphService()
	return {
		"route": "/grph/nodes",
		"tenant_id": tenant_id,
		"nodes": service.list_nodes(tenant_id),
		"schemas": service.list_schemas(tenant_id),
		"owner_required": service.describe(tenant_id)["configuration"]["nodes"]["owner_required"],
	}


def edge_manager_model(service: GrphService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or GrphService()
	return {
		"route": "/grph/edges",
		"tenant_id": tenant_id,
		"edges": service.list_edges(tenant_id),
		"classifications": service.describe(tenant_id)["configuration"]["edges"]["allowed_classifications"],
	}


def traversal_workbench_model(service: GrphService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or GrphService()
	contract = service.describe(tenant_id)
	return {
		"route": "/grph/traversal",
		"tenant_id": tenant_id,
		"traversals": service.list_traversals(tenant_id),
		"query_types": contract["configuration"]["traversal"]["allowed_query_types"],
		"max_depth": contract["configuration"]["traversal"]["max_depth"],
	}


def lineage_viewer_model(service: GrphService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or GrphService()
	return {
		"route": "/grph/lineage",
		"tenant_id": tenant_id,
		"lineage_schemas": [schema for schema in service.list_schemas(tenant_id) if schema["graph_kind"] == "lineage"],
		"traversals": service.list_traversals(tenant_id),
		"source_asset_required": service.describe(tenant_id)["configuration"]["lineage"]["source_asset_required"],
	}


def impact_analysis_model(service: GrphService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or GrphService()
	return {
		"route": "/grph/impact",
		"tenant_id": tenant_id,
		"nodes": service.list_nodes(tenant_id),
		"edges": service.list_edges(tenant_id),
		"traversals": service.list_traversals(tenant_id),
	}


def quality_console_model(service: GrphService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or GrphService()
	return {
		"route": "/grph/quality",
		"tenant_id": tenant_id,
		"reports": service.list_quality_reports(tenant_id),
		"summary": service.dashboard_summary(tenant_id),
		"thresholds": service.describe(tenant_id)["configuration"]["quality"],
	}


def governance_model(service: GrphService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or GrphService()
	return {
		"route": "/grph/governance",
		"tenant_id": tenant_id,
		"restricted_edges": [edge for edge in service.list_edges(tenant_id) if edge["classification"] == "restricted"],
		"quality_reports": service.list_quality_reports(tenant_id),
		"audit_events": service.list_audit_events(tenant_id),
	}


def audit_timeline_model(service: GrphService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or GrphService()
	return {
		"route": "/grph/audit",
		"tenant_id": tenant_id,
		"audit_events": service.list_audit_events(tenant_id),
	}


def settings_model(tenant_id: str = "default") -> dict[str, object]:
	contract = get_capability_contract(tenant_id)
	return {
		"route": "/grph/settings",
		"tenant_id": tenant_id,
		"configuration": contract["configuration"],
		"theme": contract["theme"],
	}
