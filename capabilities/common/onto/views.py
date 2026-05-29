"""UI metadata helpers for the Ontology Management capability."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import OntoService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
	service: OntoService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or OntoService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"summary": service.dashboard_summary(tenant_id),
		"routes": capability_routes(tenant_id),
		"ontologies": service.list_ontologies(tenant_id),
		"terms": service.list_terms(tenant_id),
		"mappings": service.list_mappings(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
	}


def ontology_registry_model(
	service: OntoService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or OntoService()
	return {
		"tenant_id": tenant_id,
		"ontologies": service.list_ontologies(tenant_id),
		"publications": service.list_publications(tenant_id),
		"route": "/onto/ontologies",
	}


def term_editor_model(
	service: OntoService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or OntoService()
	return {
		"tenant_id": tenant_id,
		"terms": service.list_terms(tenant_id),
		"reviews": service.list_reviews(tenant_id),
		"route": "/onto/terms",
	}


def mapping_workbench_model(
	service: OntoService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or OntoService()
	return {
		"tenant_id": tenant_id,
		"confidence_threshold": service.confidence_threshold,
		"mappings": service.list_mappings(tenant_id),
		"reviews": service.list_reviews(tenant_id),
		"route": "/onto/mappings",
	}


def taxonomy_model(
	service: OntoService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or OntoService()
	return {
		"tenant_id": tenant_id,
		"terms": service.list_terms(tenant_id),
		"edges": service.list_taxonomy_edges(tenant_id),
		"route": "/onto/terms",
	}


def publication_queue_model(
	service: OntoService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or OntoService()
	return {
		"tenant_id": tenant_id,
		"ontologies": service.list_ontologies(tenant_id),
		"publications": service.list_publications(tenant_id),
		"route": "/onto/publication",
	}


def governance_model(
	service: OntoService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or OntoService()
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"rules": contract["rule_engine"]["rules"],
		"reviews": service.list_reviews(tenant_id),
		"audit_events": service.list_audit_events(tenant_id),
		"route": "/onto/governance",
	}
