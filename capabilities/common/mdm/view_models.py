"""Generated-application view models for the MDM capability."""

from __future__ import annotations

from typing import Any

from .capability_contract import get_capability_contract
from .service import MdmService


def capability_routes(tenant_id: str = "default") -> list[dict[str, Any]]:
	"""Return route metadata for generated APG shells."""
	return get_capability_contract(tenant_id)["ui"]["routes"]


def dashboard_model(service: MdmService, tenant_id: str = "default") -> dict[str, Any]:
	"""Return MDM dashboard state."""
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Master Data Management",
		"tenant_id": tenant_id,
		"summary": service.dashboard_summary(tenant_id),
		"theme": contract["theme"],
		"primary_actions": [
			{"id": "register_entity", "label": "Register entity", "permission": "mdm:manage_entities"},
			{"id": "assess_quality", "label": "Assess quality", "permission": "mdm:view_quality"},
			{"id": "review_duplicate", "label": "Review duplicate", "permission": "mdm:review_duplicates"},
			{"id": "publish_entity", "label": "Publish entity", "permission": "mdm:publish"},
		],
	}


def entity_workbench_model(service: MdmService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"rows": service.list_records(tenant_id, "entities"),
		"columns": ["entity_id", "entity_type", "name", "business_key", "data_owner", "quality_score", "status"],
	}


def quality_console_model(service: MdmService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"rows": service.list_records(tenant_id, "quality_assessments"),
		"columns": ["created_at", "entity_id", "overall_score", "status", "assessor", "matched_rules"],
		"dimensions": get_capability_contract(tenant_id)["configuration"]["quality"]["dimensions"],
	}


def duplicate_review_model(service: MdmService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"rows": service.list_records(tenant_id, "duplicate_candidates"),
		"columns": ["created_at", "entity_id", "candidate_entity_id", "confidence", "status", "steward", "review_decision"],
		"review_actions": ["merge", "keep_separate", "defer"],
	}


def stewardship_model(service: MdmService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"duplicate_reviews": [
			row for row in service.list_records(tenant_id, "duplicate_candidates")
			if row["status"] in {"review_required", "review_denied"}
		],
		"merge_reviews": [
			row for row in service.list_records(tenant_id, "merge_requests")
			if row["status"] == "pending_review"
		],
	}


def golden_record_model(service: MdmService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"rows": service.list_records(tenant_id, "golden_records"),
		"columns": ["golden_record_id", "entity_type", "survivorship_policy", "source_entity_ids", "status"],
		"merge_requests": service.list_records(tenant_id, "merge_requests"),
	}


def lineage_model(service: MdmService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"entities": service.list_records(tenant_id, "entities"),
		"quality_assessments": service.list_records(tenant_id, "quality_assessments"),
		"duplicate_candidates": service.list_records(tenant_id, "duplicate_candidates"),
		"golden_records": service.list_records(tenant_id, "golden_records"),
		"publish_records": service.list_records(tenant_id, "publish_records"),
	}


def cross_reference_model(service: MdmService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"rows": service.list_records(tenant_id, "cross_references"),
		"columns": ["source_system", "source_identifier", "entity_id", "status", "evidence_reference"],
	}


def publish_model(service: MdmService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"rows": service.list_records(tenant_id, "publish_records"),
		"columns": ["created_at", "entity_id", "channel", "decision", "status", "quality_score", "matched_rules"],
	}


def analytics_model(service: MdmService, tenant_id: str = "default") -> dict[str, Any]:
	summary = service.dashboard_summary(tenant_id)
	return {
		"tenant_id": tenant_id,
		"summary": summary,
		"quality_dimensions": get_capability_contract(tenant_id)["configuration"]["quality"]["dimensions"],
		"sections": ["entities", "quality_assessments", "duplicate_candidates", "golden_records", "publish_records"],
	}


def audit_timeline_model(service: MdmService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"events": service.list_records(tenant_id, "audit_events"),
		"columns": ["created_at", "event_type", "subject", "actor", "decision", "matched_rules"],
	}


def adapter_health_model(tenant_id: str = "default") -> dict[str, Any]:
	adapters = get_capability_contract(tenant_id)["configuration"]["adapters"]
	return {
		"tenant_id": tenant_id,
		"database_runtime": adapters["database_runtime"],
		"generated_app_runtime": adapters["generated_app_runtime"],
		"event_stream": adapters["event_stream"],
		"quality_engine": adapters["quality_engine"],
		"matching_engine": adapters["matching_engine"],
		"lineage_adapter": adapters["lineage_adapter"],
	}


def settings_model(tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"theme": contract["theme"],
		"routes": contract["ui"]["routes"],
	}
