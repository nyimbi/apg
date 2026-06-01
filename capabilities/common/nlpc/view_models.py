"""Generated-app view models for APG NLP Core."""

from __future__ import annotations

from .capability_contract import SUPPORTED_LANGUAGES, get_capability_contract
from .nlpc_runtime import AFRICAN_LANGUAGE_CODES, NlpcService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	"""Return route metadata for generated NLPC applications."""
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(service: NlpcService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or NlpcService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"summary": service.dashboard_summary(tenant_id),
		"routes": capability_routes(tenant_id),
		"recent_documents": service.list_documents(tenant_id)[-10:],
		"recent_runs": service.list_processing_runs(tenant_id)[-10:],
		"nlp_agents": service.list_nlp_agents(tenant_id),
		"lifecycle_batches": service.list_lifecycle_batches(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
	}


def processing_console_model(service: NlpcService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or NlpcService()
	return {
		"tenant_id": tenant_id,
		"route": "/nlpc/process",
		"documents": service.list_documents(tenant_id),
		"processing_runs": service.list_processing_runs(tenant_id),
		"pending_review": [
			item for item in service.list_processing_runs(tenant_id)
			if item["status"] == "pending_review"
		],
		"enabled_tasks": service.describe(tenant_id)["configuration"]["tasks"]["enabled"],
		"required_policies": ["tenant", "language", "pii_redaction", "generation_safety", "model_policy"],
	}


def document_workbench_model(service: NlpcService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or NlpcService()
	return {
		"tenant_id": tenant_id,
		"route": "/nlpc/documents",
		"documents": service.list_documents(tenant_id),
		"supported_languages": list(SUPPORTED_LANGUAGES),
	}


def pipeline_designer_model(service: NlpcService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or NlpcService()
	return {
		"tenant_id": tenant_id,
		"route": "/nlpc/pipelines",
		"pipelines": service.list_pipelines(tenant_id),
		"models": service.list_models(tenant_id),
		"enabled_tasks": service.describe(tenant_id)["configuration"]["tasks"]["enabled"],
	}


def batch_queue_model(service: NlpcService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or NlpcService()
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"route": "/nlpc/batches",
		"engine": contract["configuration"]["adapters"]["event_stream"],
		"lifecycle_stream": contract["streaming"]["lifecycle_stream"],
		"async_threshold_documents": contract["configuration"]["processing"]["async_threshold_documents"],
		"pending_documents": [item for item in service.list_documents(tenant_id) if item["status"] == "ingested"],
		"lifecycle_batches": service.list_lifecycle_batches(tenant_id),
	}


def annotation_workbench_model(service: NlpcService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or NlpcService()
	return {
		"tenant_id": tenant_id,
		"route": "/nlpc/annotations",
		"projects": service.list_annotation_projects(tenant_id),
		"annotations": service.list_annotations(tenant_id),
		"documents": service.list_documents(tenant_id),
	}


def review_console_model(service: NlpcService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or NlpcService()
	return {
		"tenant_id": tenant_id,
		"route": "/nlpc/review",
		"low_confidence_runs": [
			item for item in service.list_processing_runs(tenant_id)
			if item["confidence_score"] < service.describe(tenant_id)["configuration"]["tasks"]["minimum_confidence_score"]
		],
		"pending_processing_reviews": [
			item for item in service.list_processing_runs(tenant_id)
			if item["status"] == "pending_review"
		],
		"low_consensus_annotations": [
			item for item in service.list_annotations(tenant_id)
			if item["consensus_score"] < service.describe(tenant_id)["configuration"]["annotation"]["minimum_consensus_score"]
		],
	}


def model_registry_model(service: NlpcService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or NlpcService()
	return {
		"tenant_id": tenant_id,
		"route": "/nlpc/models",
		"models": service.list_models(tenant_id),
		"mlcm_adapter": service.describe(tenant_id)["configuration"]["adapters"]["model_lifecycle"],
	}


def language_coverage_model(tenant_id: str = "default") -> dict[str, object]:
	return {
		"tenant_id": tenant_id,
		"route": "/nlpc/languages",
		"supported_languages": list(SUPPORTED_LANGUAGES),
		"supported_language_count": len(SUPPORTED_LANGUAGES),
		"african_languages": sorted(AFRICAN_LANGUAGE_CODES),
		"african_language_count": len(AFRICAN_LANGUAGE_CODES),
	}


def lexicon_manager_model(service: NlpcService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or NlpcService()
	return {
		"tenant_id": tenant_id,
		"route": "/nlpc/lexicons",
		"lexicons": service.list_lexicons(tenant_id),
		"supported_languages": list(SUPPORTED_LANGUAGES),
	}


def semantic_search_model(service: NlpcService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or NlpcService()
	return {
		"tenant_id": tenant_id,
		"route": "/nlpc/search",
		"search_index_adapter": service.describe(tenant_id)["configuration"]["adapters"]["search_index"],
		"search_runs": [
			item for item in service.list_processing_runs(tenant_id)
			if "semantic_search" in item["tasks"]
		],
	}


def governance_model(service: NlpcService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or NlpcService()
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"route": "/nlpc/governance",
		"configuration": contract["configuration"],
		"agents": contract["agents"],
		"streaming": contract["streaming"],
		"nlp_agents": service.list_nlp_agents(tenant_id),
		"lifecycle_batches": service.list_lifecycle_batches(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"audit_events": service.list_audit_events(tenant_id),
	}


def audit_timeline_model(service: NlpcService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or NlpcService()
	return {
		"tenant_id": tenant_id,
		"route": "/nlpc/audit",
		"audit_events": service.list_audit_events(tenant_id),
	}


def nlp_agent_roster_model(service: NlpcService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or NlpcService()
	contract = service.describe(tenant_id)
	agents = service.list_nlp_agents(tenant_id)
	return {
		"tenant_id": tenant_id,
		"route": "/nlpc/agents",
		"agents": agents,
		"pending_review": [item for item in agents if item["status"] == "pending_review"],
		"supported_runtimes": contract["agents"]["supported_runtimes"],
		"supported_roles": contract["agents"]["supported_roles"],
		"privileged_roles": contract["agents"]["privileged_roles"],
	}


def lifecycle_batch_model(service: NlpcService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or NlpcService()
	contract = service.describe(tenant_id)
	batches = service.list_lifecycle_batches(tenant_id)
	return {
		"tenant_id": tenant_id,
		"route": "/nlpc/lifecycle",
		"batches": batches,
		"denied": [item for item in batches if item["status"] == "denied"],
		"required_processor": contract["streaming"]["required_processor"],
		"required_operations": contract["streaming"]["required_operations"],
		"topics": contract["streaming"]["topics"],
	}
