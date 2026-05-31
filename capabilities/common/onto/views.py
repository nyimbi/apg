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
		"namespaces": service.list_namespaces(tenant_id),
		"terms": service.list_terms(tenant_id),
		"taxonomy_edges": service.list_taxonomy_edges(tenant_id),
		"mappings": service.list_mappings(tenant_id),
		"validation_reports": service.list_validation_reports(tenant_id),
		"publications": service.list_publications(tenant_id),
		"exports": service.list_exports(tenant_id),
		"ontology_agents": service.list_ontology_agents(tenant_id),
		"lifecycle_batches": service.list_lifecycle_batches(tenant_id),
		"audit_events": service.list_audit_events(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"agents": contract["agents"],
		"streaming": contract["streaming"],
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


def namespace_model(
	service: OntoService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or OntoService()
	return {
		"tenant_id": tenant_id,
		"ontologies": service.list_ontologies(tenant_id),
		"namespaces": service.list_namespaces(tenant_id),
		"route": "/onto/namespaces",
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


def validation_model(
	service: OntoService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or OntoService()
	return {
		"tenant_id": tenant_id,
		"ontologies": service.list_ontologies(tenant_id),
		"validation_reports": service.list_validation_reports(tenant_id),
		"route": "/onto/validation",
	}


def exchange_model(
	service: OntoService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or OntoService()
	return {
		"tenant_id": tenant_id,
		"ontologies": service.list_ontologies(tenant_id),
		"exports": service.list_exports(tenant_id),
		"route": "/onto/exports",
	}


def ontology_agent_roster_model(
	service: OntoService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or OntoService()
	contract = service.describe(tenant_id)
	agents = service.list_ontology_agents(tenant_id)
	return {
		"tenant_id": tenant_id,
		"agents": agents,
		"pending_review": [item for item in agents if item["status"] == "pending_review"],
		"supported_runtimes": contract["agents"]["supported_runtimes"],
		"supported_roles": contract["agents"]["supported_roles"],
		"privileged_roles": contract["agents"]["privileged_roles"],
		"route": "/onto/agents",
	}


def lifecycle_batch_model(
	service: OntoService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or OntoService()
	contract = service.describe(tenant_id)
	batches = service.list_lifecycle_batches(tenant_id)
	return {
		"tenant_id": tenant_id,
		"batches": batches,
		"denied": [item for item in batches if item["status"] == "denied"],
		"required_processor": contract["streaming"]["required_processor"],
		"required_operations": contract["streaming"]["required_operations"],
		"topics": contract["streaming"]["topics"],
		"route": "/onto/lifecycle",
	}


def audit_timeline_model(
	service: OntoService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or OntoService()
	return {
		"tenant_id": tenant_id,
		"audit_events": service.list_audit_events(tenant_id),
		"route": "/onto/audit",
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
		"agents": contract["agents"],
		"streaming": contract["streaming"],
		"reviews": service.list_reviews(tenant_id),
		"ontology_agents": service.list_ontology_agents(tenant_id),
		"lifecycle_batches": service.list_lifecycle_batches(tenant_id),
		"audit_events": service.list_audit_events(tenant_id),
		"configuration": contract["configuration"],
		"route": "/onto/governance",
	}


def settings_model(
	service: OntoService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or OntoService()
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"agents": contract["agents"],
		"streaming": contract["streaming"],
		"theme": contract["theme"],
		"adapters": contract["configuration"]["adapters"],
		"route": "/onto/settings",
	}
