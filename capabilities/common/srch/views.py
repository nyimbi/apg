"""UI metadata helpers for the Search Engine capability."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import SrchService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
	service: SrchService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or SrchService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"routes": capability_routes(tenant_id),
		"summary": service.dashboard_summary(tenant_id),
		"search_agents": service.list_search_agents(tenant_id),
		"lifecycle_batches": service.list_lifecycle_batches(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
	}


def search_console_model(
	service: SrchService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or SrchService()
	return {
		"route": "/srch/search",
		"tenant_id": tenant_id,
		"indices": service.list_indices(tenant_id),
		"query_types": ["keyword", "semantic", "hybrid"],
		"rbac_filter_required": True,
	}


def index_manager_model(
	service: SrchService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or SrchService()
	return {
		"route": "/srch/indices",
		"tenant_id": tenant_id,
		"indices": service.list_indices(tenant_id),
		"classifications": ["public", "internal", "confidential", "restricted"],
		"states": ["creating", "ready", "embedding_pending", "embedding_ready", "degraded", "retired"],
	}


def document_indexer_model(
	service: SrchService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or SrchService()
	return {
		"route": "/srch/documents",
		"tenant_id": tenant_id,
		"documents": service.list_documents(tenant_id),
		"lineage_required": True,
	}


def bulk_index_model(
	service: SrchService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or SrchService()
	contract = service.describe(tenant_id)
	return {
		"route": "/srch/bulk",
		"tenant_id": tenant_id,
		"indices": service.list_indices(tenant_id),
		"event_stream": contract["configuration"]["adapters"]["event_stream"],
		"streaming": contract["streaming"],
		"lifecycle_batches": service.list_lifecycle_batches(tenant_id),
		"max_documents_per_batch": contract["configuration"]["indexing"]["max_documents_per_batch"],
		"lineage_required": contract["configuration"]["indexing"]["bulk_lineage_required"],
	}


def facet_explorer_model(
	service: SrchService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or SrchService()
	return {
		"route": "/srch/facets",
		"tenant_id": tenant_id,
		"facets": service.facets(tenant_id),
		"allowed_facet_keys": service.describe(tenant_id)["configuration"]["facets"]["allowed_facet_keys"],
	}


def analytics_model(
	service: SrchService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or SrchService()
	return {
		"route": "/srch/analytics",
		"tenant_id": tenant_id,
		"queries": service.list_queries(tenant_id),
		"summary": service.dashboard_summary(tenant_id),
		"facets": service.facets(tenant_id),
	}


def ranking_model(
	service: SrchService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or SrchService()
	contract = service.describe(tenant_id)
	return {
		"route": "/srch/ranking",
		"tenant_id": tenant_id,
		"ranking": contract["configuration"]["ranking"],
		"queries": service.list_queries(tenant_id),
	}


def access_review_model(
	service: SrchService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or SrchService()
	return {
		"route": "/srch/access",
		"tenant_id": tenant_id,
		"restricted_indices": [
			index
			for index in service.list_indices(tenant_id)
			if index["classification"] == "restricted"
		],
		"denied_queries": [
			query
			for query in service.list_queries(tenant_id)
			if query["status"] == "denied"
		],
	}


def governance_model(
	service: SrchService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or SrchService()
	contract = service.describe(tenant_id)
	return {
		"route": "/srch/governance",
		"tenant_id": tenant_id,
		"audit_events": service.list_audit_events(tenant_id),
		"agents": contract["agents"],
		"streaming": contract["streaming"],
		"search_agents": service.list_search_agents(tenant_id),
		"lifecycle_batches": service.list_lifecycle_batches(tenant_id),
		"restricted_indices": [
			index
			for index in service.list_indices(tenant_id)
			if index["classification"] == "restricted"
		],
		"review_required_queries": [
			query
			for query in service.list_queries(tenant_id)
			if query["status"] == "review_required"
		],
	}


def search_agent_roster_model(
	service: SrchService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or SrchService()
	contract = service.describe(tenant_id)
	agents = service.list_search_agents(tenant_id)
	return {
		"route": "/srch/agents",
		"tenant_id": tenant_id,
		"agents": agents,
		"pending_review": [item for item in agents if item["status"] == "pending_review"],
		"supported_runtimes": contract["agents"]["supported_runtimes"],
		"supported_roles": contract["agents"]["supported_roles"],
		"privileged_roles": contract["agents"]["privileged_roles"],
	}


def lifecycle_batch_model(
	service: SrchService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or SrchService()
	contract = service.describe(tenant_id)
	batches = service.list_lifecycle_batches(tenant_id)
	return {
		"route": "/srch/lifecycle",
		"tenant_id": tenant_id,
		"batches": batches,
		"denied": [item for item in batches if item["status"] == "denied"],
		"required_processor": contract["streaming"]["required_processor"],
		"required_operations": contract["streaming"]["required_operations"],
		"topics": contract["streaming"]["topics"],
	}


def audit_timeline_model(
	service: SrchService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or SrchService()
	return {
		"route": "/srch/audit",
		"tenant_id": tenant_id,
		"audit_events": service.list_audit_events(tenant_id),
	}


def settings_model(tenant_id: str = "default") -> dict[str, object]:
	contract = get_capability_contract(tenant_id)
	return {
		"route": "/srch/settings",
		"tenant_id": tenant_id,
		"configuration": contract["configuration"],
		"agents": contract["agents"],
		"streaming": contract["streaming"],
		"theme": contract["theme"],
	}
