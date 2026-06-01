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
	sources = service.list_sources(tenant_id)
	entities = service.list_entities(tenant_id)
	relationships = service.list_relationships(tenant_id)
	enrichments = service.list_enrichments(tenant_id)
	reasoning_paths = service.list_reasoning_paths(tenant_id)
	knowledge_agents = service.list_knowledge_agents(tenant_id)
	lifecycle_batches = service.list_lifecycle_batches(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"routes": capability_routes(tenant_id),
		"summary": service.dashboard_summary(tenant_id),
		"sources": sources,
		"entities": entities,
		"relationships": relationships,
		"enrichments": enrichments,
		"reasoning_paths": reasoning_paths,
		"curations": service.list_curations(tenant_id),
		"publications": service.list_publications(tenant_id),
		"knowledge_agents": knowledge_agents,
		"lifecycle_batches": lifecycle_batches,
		"audit_events": service.list_audit_events(tenant_id),
		"pending_reviews": {
			"sources": _pending_review(sources),
			"entities": _pending_review(entities),
			"relationships": _pending_review(relationships),
			"enrichments": _pending_review(enrichments),
			"reasoning_paths": _pending_review(reasoning_paths),
			"agents": _pending_review(knowledge_agents),
		},
		"rules": contract["rule_engine"]["rules"],
		"agents": contract["agents"],
		"streaming": contract["streaming"],
		"theme": contract["theme"],
	}


def entity_browser_model(service: KngrService, tenant_id: str = "default") -> dict[str, object]:
	entities = service.list_entities(tenant_id)
	return {
		"tenant_id": tenant_id,
		"sources": service.list_sources(tenant_id),
		"entities": entities,
		"pending_review": _pending_review(entities),
		"relationships": service.list_relationships(tenant_id),
	}


def source_manager_model(service: KngrService, tenant_id: str = "default") -> dict[str, object]:
	sources = service.list_sources(tenant_id)
	return {
		"tenant_id": tenant_id,
		"sources": sources,
		"pending_review": _pending_review(sources),
		"audit_events": [
			event for event in service.list_audit_events(tenant_id)
			if event["event_type"] == "source_registered"
		],
	}


def relationship_browser_model(service: KngrService, tenant_id: str = "default") -> dict[str, object]:
	relationships = service.list_relationships(tenant_id)
	return {
		"tenant_id": tenant_id,
		"entities": service.list_entities(tenant_id),
		"relationships": relationships,
		"pending_review": _pending_review(relationships),
		"sources": service.list_sources(tenant_id),
	}


def enrichment_console_model(service: KngrService, tenant_id: str = "default") -> dict[str, object]:
	enrichments = service.list_enrichments(tenant_id)
	return {
		"tenant_id": tenant_id,
		"entities": service.list_entities(tenant_id),
		"enrichments": enrichments,
		"pending_review": _pending_review(enrichments),
		"review_required": [
			enrichment for enrichment in enrichments
			if enrichment["status"] in {"accepted_with_review", "pending_review"}
		],
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
	reasoning_paths = service.list_reasoning_paths(tenant_id)
	return {
		"tenant_id": tenant_id,
		"reasoning_paths": reasoning_paths,
		"pending_review": _pending_review(reasoning_paths),
		"enrichments": service.list_enrichments(tenant_id),
	}


def context_explorer_model(service: KngrService, tenant_id: str, entity_id: str) -> dict[str, object]:
	return {
		"tenant_id": tenant_id,
		"neighborhood": service.context_neighborhood(tenant_id, entity_id),
	}


def governance_model(service: KngrService, tenant_id: str = "default") -> dict[str, object]:
	contract = service.describe(tenant_id)
	sources = service.list_sources(tenant_id)
	entities = service.list_entities(tenant_id)
	relationships = service.list_relationships(tenant_id)
	enrichments = service.list_enrichments(tenant_id)
	reasoning_paths = service.list_reasoning_paths(tenant_id)
	knowledge_agents = service.list_knowledge_agents(tenant_id)
	return {
		"tenant_id": tenant_id,
		"rules": contract["rule_engine"]["rules"],
		"agents": contract["agents"],
		"streaming": contract["streaming"],
		"knowledge_agents": knowledge_agents,
		"lifecycle_batches": service.list_lifecycle_batches(tenant_id),
		"audit_events": service.list_audit_events(tenant_id),
		"publications": service.list_publications(tenant_id),
		"pending_reviews": {
			"sources": _pending_review(sources),
			"entities": _pending_review(entities),
			"relationships": _pending_review(relationships),
			"enrichments": _pending_review(enrichments),
			"reasoning_paths": _pending_review(reasoning_paths),
			"agents": _pending_review(knowledge_agents),
		},
	}


def knowledge_agent_roster_model(service: KngrService, tenant_id: str = "default") -> dict[str, object]:
	contract = service.describe(tenant_id)
	agents = service.list_knowledge_agents(tenant_id)
	return {
		"tenant_id": tenant_id,
		"agents": agents,
		"pending_review": [item for item in agents if item["status"] == "pending_review"],
		"supported_runtimes": contract["agents"]["supported_runtimes"],
		"supported_roles": contract["agents"]["supported_roles"],
		"privileged_roles": contract["agents"]["privileged_roles"],
	}


def lifecycle_batch_model(service: KngrService, tenant_id: str = "default") -> dict[str, object]:
	contract = service.describe(tenant_id)
	batches = service.list_lifecycle_batches(tenant_id)
	return {
		"tenant_id": tenant_id,
		"batches": batches,
		"denied": [item for item in batches if item["status"] == "denied"],
		"required_processor": contract["streaming"]["required_processor"],
		"required_operations": contract["streaming"]["required_operations"],
		"topics": contract["streaming"]["topics"],
	}


def publication_model(service: KngrService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"tenant_id": tenant_id,
		"curated_entities": [
			entity for entity in service.list_entities(tenant_id)
			if entity["curation_status"] == "curated"
		],
		"relationships": service.list_relationships(tenant_id),
		"publications": service.list_publications(tenant_id),
	}


def audit_timeline_model(service: KngrService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"tenant_id": tenant_id,
		"audit_events": service.list_audit_events(tenant_id),
	}


def settings_model(service: KngrService, tenant_id: str = "default") -> dict[str, object]:
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"agents": contract["agents"],
		"streaming": contract["streaming"],
		"theme": contract["theme"],
		"adapters": contract["configuration"]["adapters"],
	}


def _pending_review(records: list[dict[str, object]]) -> list[dict[str, object]]:
	return [item for item in records if item.get("status") == "pending_review"]
