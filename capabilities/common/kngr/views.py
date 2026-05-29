"""UI metadata helpers for APG Knowledge Graph."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import KngrService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
	service: KngrService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or KngrService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"routes": capability_routes(tenant_id),
		"summary": service.dashboard_summary(tenant_id),
		"sources": service.list_sources(tenant_id),
		"entities": service.list_entities(tenant_id),
		"relationships": service.list_relationships(tenant_id),
		"enrichments": service.list_enrichments(tenant_id),
		"reasoning_paths": service.list_reasoning_paths(tenant_id),
		"curations": service.list_curations(tenant_id),
		"publications": service.list_publications(tenant_id),
		"audit_events": service.list_audit_events(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
	}


def entity_browser_model(service: KngrService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"tenant_id": tenant_id,
		"sources": service.list_sources(tenant_id),
		"entities": service.list_entities(tenant_id),
		"relationships": service.list_relationships(tenant_id),
	}


def curation_queue_model(service: KngrService, tenant_id: str = "default") -> dict[str, object]:
	entities = service.list_entities(tenant_id)
	return {
		"tenant_id": tenant_id,
		"curations": service.list_curations(tenant_id),
		"pending_entities": [
			entity for entity in entities
			if entity["curation_status"] != "curated"
		],
	}


def reasoning_paths_model(service: KngrService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"tenant_id": tenant_id,
		"reasoning_paths": service.list_reasoning_paths(tenant_id),
		"enrichments": service.list_enrichments(tenant_id),
	}


def context_explorer_model(service: KngrService, tenant_id: str, entity_id: str) -> dict[str, object]:
	return {
		"tenant_id": tenant_id,
		"neighborhood": service.context_neighborhood(tenant_id, entity_id),
	}


def governance_model(service: KngrService, tenant_id: str = "default") -> dict[str, object]:
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"rules": contract["rule_engine"]["rules"],
		"audit_events": service.list_audit_events(tenant_id),
		"publications": service.list_publications(tenant_id),
	}
