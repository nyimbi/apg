"""Generated-application view models for the ETLP capability."""

from __future__ import annotations

from typing import Any

from .capability_contract import get_capability_contract
from .service import ETLPLifecycleService


def capability_routes(tenant_id: str = "default") -> list[dict[str, Any]]:
	"""Return route metadata for generated APG shells."""
	return get_capability_contract(tenant_id)["ui"]["routes"]


def dashboard_model(service: ETLPLifecycleService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"title": "ETL/ELT Processing",
		"tenant_id": tenant_id,
		"summary": service.dashboard_summary(tenant_id),
		"theme": contract["theme"],
		"primary_actions": [
			{"id": "register_pipeline", "label": "Register pipeline", "permission": "etlp:pipeline:write"},
			{"id": "register_datasource", "label": "Register datasource", "permission": "etlp:datasource:write"},
			{"id": "execute_pipeline", "label": "Execute pipeline", "permission": "etlp:pipeline:execute"},
			{"id": "publish_output", "label": "Review publish", "permission": "etlp:publish:review"},
		],
	}


def pipeline_workbench_model(service: ETLPLifecycleService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"rows": service.list_records(tenant_id, "pipelines"),
		"columns": ["pipeline_id", "name", "mode", "owner", "status", "decision", "matched_rules"],
	}


def datasource_manager_model(service: ETLPLifecycleService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"rows": service.list_records(tenant_id, "datasources"),
		"columns": ["datasource_id", "name", "datasource_type", "owner", "approved", "status", "matched_rules"],
	}


def field_mapper_model(service: ETLPLifecycleService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"rows": service.list_records(tenant_id, "mappings"),
		"columns": ["mapping_id", "pipeline_id", "source_datasource_id", "target_datasource_id", "schema_validated", "lineage_emitted", "status"],
	}


def execution_monitor_model(service: ETLPLifecycleService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"rows": service.list_records(tenant_id, "executions"),
		"columns": ["created_at", "execution_id", "pipeline_id", "environment", "status", "estimated_cost", "matched_rules"],
	}


def quality_console_model(service: ETLPLifecycleService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"rows": service.list_records(tenant_id, "quality_results"),
		"columns": ["created_at", "execution_id", "pipeline_id", "score", "gate_passed", "assessor"],
		"dimensions": get_capability_contract(tenant_id)["configuration"]["quality"]["required_dimensions"],
	}


def schedule_console_model(service: ETLPLifecycleService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"rows": service.list_records(tenant_id, "schedules"),
		"columns": ["created_at", "schedule_id", "pipeline_id", "environment", "schedule", "status", "matched_rules"],
	}


def publish_review_model(service: ETLPLifecycleService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"rows": service.list_records(tenant_id, "publish_reviews"),
		"columns": ["created_at", "execution_id", "pipeline_id", "requester", "quality_score", "status", "matched_rules"],
	}


def replay_console_model(service: ETLPLifecycleService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"rows": service.list_records(tenant_id, "replay_requests"),
		"columns": ["created_at", "execution_id", "replay_type", "reason", "window_hours", "status", "matched_rules"],
	}


def lineage_model(service: ETLPLifecycleService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"pipelines": service.list_records(tenant_id, "pipelines"),
		"mappings": service.list_records(tenant_id, "mappings"),
		"lineage_edges": [
			row for row in service.list_records(tenant_id, "mappings")
			if row["lineage_emitted"]
		],
	}


def audit_timeline_model(service: ETLPLifecycleService, tenant_id: str = "default") -> dict[str, Any]:
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
		"execution_engine": adapters["execution_engine"],
		"connector_registry": adapters["connector_registry"],
		"event_stream": adapters["event_stream"],
		"metadata_catalog": adapters["metadata_catalog"],
		"quality_engine": adapters["quality_engine"],
		"lineage_emitter": adapters["lineage_emitter"],
		"secret_store": adapters["secret_store"],
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
