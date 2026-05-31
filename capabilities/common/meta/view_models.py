"""Generated-application view models for the META capability."""

from __future__ import annotations

from typing import Any

from .capability_contract import get_capability_contract
from .service import MetaService


def capability_routes(tenant_id: str = "default") -> list[dict[str, Any]]:
	"""Return route metadata for generated APG shells."""
	return get_capability_contract(tenant_id)["ui"]["routes"]


def dashboard_model(service: MetaService, tenant_id: str = "default") -> dict[str, Any]:
	"""Return metadata dashboard state."""
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Metadata Management",
		"tenant_id": tenant_id,
		"summary": service.dashboard_summary(tenant_id),
		"theme": contract["theme"],
		"primary_actions": [
			{"id": "register_asset", "label": "Register asset", "permission": "meta:view_assets"},
			{"id": "schedule_discovery", "label": "Schedule discovery", "permission": "meta:run_discovery"},
			{"id": "review_classification", "label": "Review classification", "permission": "meta:classify"},
			{"id": "request_certification", "label": "Request certification", "permission": "meta:certify"},
			{"id": "register_agent", "label": "Register agent", "permission": "meta:admin"},
		],
	}


def asset_catalog_model(service: MetaService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"rows": service.list_records(tenant_id, "assets"),
		"columns": ["asset_id", "asset_type", "name", "source_system", "owner", "steward", "quality_score", "status"],
	}


def discovery_console_model(service: MetaService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"rows": service.list_records(tenant_id, "discovery_jobs"),
		"columns": ["created_at", "connector_type", "source_system", "schedule", "status", "matched_rules"],
	}


def classification_review_model(service: MetaService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"rows": service.list_records(tenant_id, "classifications"),
		"columns": ["created_at", "asset_id", "label", "confidence", "status", "steward", "review_notes"],
		"review_actions": ["accept", "correct", "defer"],
	}


def lineage_model(service: MetaService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"nodes": service.list_records(tenant_id, "assets"),
		"edges": service.list_records(tenant_id, "lineage"),
		"columns": ["source_asset_id", "target_asset_id", "lineage_type", "depth", "status"],
	}


def quality_model(service: MetaService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"rows": service.list_records(tenant_id, "quality_assessments"),
		"columns": ["created_at", "asset_id", "score", "status", "assessor"],
		"dimensions": get_capability_contract(tenant_id)["configuration"]["quality"]["dimensions"],
	}


def certification_model(service: MetaService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"rows": service.list_records(tenant_id, "certifications"),
		"columns": ["created_at", "asset_id", "requester", "decision", "status", "matched_rules"],
	}


def glossary_model(service: MetaService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"rows": service.list_records(tenant_id, "glossary_terms"),
		"columns": ["term", "definition", "owner", "linked_asset_ids", "status"],
	}


def impact_model(service: MetaService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"assets": service.list_records(tenant_id, "assets"),
		"lineage": service.list_records(tenant_id, "lineage"),
		"retirement_candidates": [
			row for row in service.list_records(tenant_id, "assets")
			if row["status"] in {"draft", "published", "certified"}
		],
	}


def search_model(service: MetaService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"assets": service.list_records(tenant_id, "assets"),
		"searchable_fields": ["name", "asset_type", "business_key", "source_system", "tags", "metadata"],
	}


def audit_timeline_model(service: MetaService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"events": service.list_records(tenant_id, "audit_events"),
		"columns": ["created_at", "event_type", "subject", "actor", "decision", "matched_rules"],
	}


def adapter_health_model(tenant_id: str = "default") -> dict[str, Any]:
	adapters = get_capability_contract(tenant_id)["configuration"]["adapters"]
	return {
		"tenant_id": tenant_id,
		"production_runtime": adapters["production_runtime"],
		"generated_app_runtime": adapters["generated_app_runtime"],
		"discovery_engine": adapters["discovery_engine"],
		"classification_engine": adapters["classification_engine"],
		"lineage_engine": adapters["lineage_engine"],
		"search_engine": adapters["search_engine"],
		"event_stream": adapters["event_stream"],
	}


def catalog_agent_roster_model(service: MetaService, tenant_id: str = "default") -> dict[str, Any]:
	"""Return first-class META catalog-agent roster state."""
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"rows": service.list_records(tenant_id, "catalog_agents"),
		"supported_runtimes": contract["agents"]["supported_runtimes"],
		"supported_roles": contract["agents"]["supported_roles"],
		"privileged_roles": contract["agents"]["privileged_roles"],
		"guardrails": contract["agents"]["guardrails"],
		"columns": ["name", "runtime", "role", "owner", "purpose", "status", "human_approval_required"],
	}


def lifecycle_batch_model(service: MetaService, tenant_id: str = "default") -> dict[str, Any]:
	"""Return Bytewax metadata lifecycle-batch monitor state."""
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"streaming": contract["streaming"],
		"rows": service.list_records(tenant_id, "lifecycle_batches"),
		"columns": ["event_stream", "mutation_count", "accepted", "decision", "required_processor", "status"],
	}


def settings_model(tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"agents": contract["agents"],
		"streaming": contract["streaming"],
		"theme": contract["theme"],
		"routes": contract["ui"]["routes"],
	}
