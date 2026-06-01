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
	schemas = service.list_schemas(tenant_id)
	nodes = service.list_nodes(tenant_id)
	edges = service.list_edges(tenant_id)
	traversals = service.list_traversals(tenant_id)
	quality_reports = service.list_quality_reports(tenant_id)
	graph_agents = service.list_graph_agents(tenant_id)
	lifecycle_batches = service.list_lifecycle_batches(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"routes": capability_routes(tenant_id),
		"summary": service.dashboard_summary(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"agents": contract["agents"],
		"streaming": contract["streaming"],
		"graph_agents": graph_agents,
		"lifecycle_batches": lifecycle_batches,
		"pending_reviews": {
			"schemas": _pending_review(schemas),
			"nodes": _pending_review(nodes),
			"edges": _pending_review(edges),
			"traversals": _pending_review(traversals),
			"quality_reports": _pending_review(quality_reports),
			"agents": _pending_review(graph_agents),
		},
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
	schemas = service.list_schemas(tenant_id)
	return {
		"route": "/grph/schemas",
		"tenant_id": tenant_id,
		"schemas": schemas,
		"pending_review": _pending_review(schemas),
		"graph_kinds": service.describe(tenant_id)["configuration"]["schemas"]["allowed_graph_kinds"],
		"node_type_count": sum(len(schema["node_types"]) for schema in schemas),
		"edge_type_count": sum(len(schema["edge_types"]) for schema in schemas),
	}


def node_manager_model(service: GrphService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or GrphService()
	nodes = service.list_nodes(tenant_id)
	return {
		"route": "/grph/nodes",
		"tenant_id": tenant_id,
		"nodes": nodes,
		"pending_review": _pending_review(nodes),
		"schemas": service.list_schemas(tenant_id),
		"owner_required": service.describe(tenant_id)["configuration"]["nodes"]["owner_required"],
	}


def edge_manager_model(service: GrphService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or GrphService()
	edges = service.list_edges(tenant_id)
	return {
		"route": "/grph/edges",
		"tenant_id": tenant_id,
		"edges": edges,
		"pending_review": _pending_review(edges),
		"classifications": service.describe(tenant_id)["configuration"]["edges"]["allowed_classifications"],
	}


def traversal_workbench_model(service: GrphService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or GrphService()
	contract = service.describe(tenant_id)
	traversals = service.list_traversals(tenant_id)
	return {
		"route": "/grph/traversal",
		"tenant_id": tenant_id,
		"traversals": traversals,
		"pending_review": _pending_review(traversals),
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
	reports = service.list_quality_reports(tenant_id)
	return {
		"route": "/grph/quality",
		"tenant_id": tenant_id,
		"reports": reports,
		"pending_review": _pending_review(reports),
		"summary": service.dashboard_summary(tenant_id),
		"thresholds": service.describe(tenant_id)["configuration"]["quality"],
	}


def governance_model(service: GrphService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or GrphService()
	contract = service.describe(tenant_id)
	schemas = service.list_schemas(tenant_id)
	nodes = service.list_nodes(tenant_id)
	edges = service.list_edges(tenant_id)
	traversals = service.list_traversals(tenant_id)
	quality_reports = service.list_quality_reports(tenant_id)
	graph_agents = service.list_graph_agents(tenant_id)
	return {
		"route": "/grph/governance",
		"tenant_id": tenant_id,
		"restricted_edges": [edge for edge in edges if edge["classification"] == "restricted"],
		"quality_reports": quality_reports,
		"pending_reviews": {
			"schemas": _pending_review(schemas),
			"nodes": _pending_review(nodes),
			"edges": _pending_review(edges),
			"traversals": _pending_review(traversals),
			"quality_reports": _pending_review(quality_reports),
			"agents": _pending_review(graph_agents),
		},
		"agents": contract["agents"],
		"streaming": contract["streaming"],
		"graph_agents": graph_agents,
		"lifecycle_batches": service.list_lifecycle_batches(tenant_id),
		"audit_events": service.list_audit_events(tenant_id),
	}


def graph_agent_roster_model(service: GrphService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or GrphService()
	contract = service.describe(tenant_id)
	agents = service.list_graph_agents(tenant_id)
	return {
		"route": "/grph/agents",
		"tenant_id": tenant_id,
		"agents": agents,
		"pending_review": [item for item in agents if item["status"] == "pending_review"],
		"supported_runtimes": contract["agents"]["supported_runtimes"],
		"supported_roles": contract["agents"]["supported_roles"],
		"privileged_roles": contract["agents"]["privileged_roles"],
	}


def lifecycle_batch_model(service: GrphService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or GrphService()
	contract = service.describe(tenant_id)
	batches = service.list_lifecycle_batches(tenant_id)
	return {
		"route": "/grph/lifecycle",
		"tenant_id": tenant_id,
		"batches": batches,
		"denied": [item for item in batches if item["status"] == "denied"],
		"required_processor": contract["streaming"]["required_processor"],
		"required_operations": contract["streaming"]["required_operations"],
		"topics": contract["streaming"]["topics"],
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
		"agents": contract["agents"],
		"streaming": contract["streaming"],
		"theme": contract["theme"],
	}


def _pending_review(records: list[dict[str, object]]) -> list[dict[str, object]]:
	return [item for item in records if item.get("status") == "pending_review"]
