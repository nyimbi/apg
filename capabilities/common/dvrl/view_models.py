"""Generated-application view models for the DVRL capability."""

from __future__ import annotations

from typing import Any

from .capability_contract import get_capability_contract
from .service import DVRLLifecycleService


def capability_routes(tenant_id: str = "default") -> list[dict[str, Any]]:
	"""Return route metadata for generated APG shells."""
	return get_capability_contract(tenant_id)["ui"]["routes"]


def dashboard_model(service: DVRLLifecycleService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Data Virtualization",
		"tenant_id": tenant_id,
		"summary": service.dashboard_summary(tenant_id),
		"pending_reviews": service.list_pending_reviews(tenant_id),
		"review_evidence": contract["review_evidence"],
		"theme": contract["theme"],
		"primary_actions": [
			{"id": "register_source", "label": "Register source", "permission": "dvrl:manage_sources"},
			{"id": "refresh_schema", "label": "Refresh schema", "permission": "dvrl:manage_sources"},
			{"id": "publish_virtual_table", "label": "Publish virtual table", "permission": "dvrl:manage_sources"},
			{"id": "execute_query", "label": "Plan query", "permission": "dvrl:query"},
			{"id": "register_agent", "label": "Register agent", "permission": "dvrl:admin"},
		],
	}


def source_manager_model(service: DVRLLifecycleService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"rows": service.list_records(tenant_id, "sources"),
		"columns": ["source_id", "name", "source_type", "owner", "approved", "status", "decision", "matched_rules"],
	}


def schema_browser_model(service: DVRLLifecycleService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"rows": service.list_records(tenant_id, "schemas"),
		"columns": ["schema_id", "source_id", "name", "schema_age_days", "status", "matched_rules"],
	}


def virtual_table_catalog_model(service: DVRLLifecycleService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"rows": service.list_records(tenant_id, "virtual_tables"),
		"columns": ["table_id", "source_id", "name", "owner", "classification", "status", "matched_rules"],
	}


def query_workbench_model(service: DVRLLifecycleService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"rows": service.list_records(tenant_id, "queries"),
		"columns": ["query_id", "actor", "data_classification", "estimated_query_cost", "requested_rows", "status", "matched_rules"],
		"defaults": get_capability_contract(tenant_id)["configuration"]["queries"],
	}


def federation_map_model(service: DVRLLifecycleService, tenant_id: str = "default") -> dict[str, Any]:
	sources = service.list_records(tenant_id, "sources")
	tables = service.list_records(tenant_id, "virtual_tables")
	return {
		"tenant_id": tenant_id,
		"sources": sources,
		"virtual_tables": tables,
		"edges": [
			{"from": table["source_id"], "to": table["table_id"], "kind": "publishes"}
			for table in tables
		],
	}


def cache_console_model(service: DVRLLifecycleService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"rows": service.list_records(tenant_id, "caches"),
		"columns": ["cache_id", "query_id", "ttl_seconds", "status", "matched_rules"],
		"limits": get_capability_contract(tenant_id)["configuration"]["cache"],
	}


def policies_model(service: DVRLLifecycleService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"rows": service.list_records(tenant_id, "policies"),
		"columns": ["policy_id", "name", "actor", "status", "decision", "matched_rules"],
		"rules": get_capability_contract(tenant_id)["rule_engine"]["rules"],
	}


def metrics_model(service: DVRLLifecycleService, tenant_id: str = "default") -> dict[str, Any]:
	summary = service.dashboard_summary(tenant_id)
	return {
		"tenant_id": tenant_id,
		"summary": summary,
		"metrics": [
			{"name": "sources", "value": summary["source_count"]},
			{"name": "active_sources", "value": summary["active_source_count"]},
			{"name": "queries", "value": summary["query_count"]},
			{"name": "virtualization_agents", "value": summary["virtualization_agent_count"]},
			{"name": "lifecycle_batches", "value": summary["lifecycle_batch_count"]},
			{"name": "reviews", "value": summary["review_count"]},
			{"name": "audit_events", "value": summary["audit_event_count"]},
		],
	}


def adapter_health_model(tenant_id: str = "default") -> dict[str, Any]:
	adapters = get_capability_contract(tenant_id)["configuration"]["adapters"]
	return {
		"tenant_id": tenant_id,
		"production_runtime": adapters["production_runtime"],
		"generated_app_runtime": adapters["generated_app_runtime"],
		"connector_registry": adapters["connector_registry"],
		"query_planner": adapters["query_planner"],
		"execution_engine": adapters["execution_engine"],
		"metadata_catalog": adapters["metadata_catalog"],
		"cache_store": adapters["cache_store"],
		"credential_vault": adapters["credential_vault"],
		"audit_sink": adapters["audit_sink"],
		"event_stream": adapters["event_stream"],
	}


def virtualization_agent_roster_model(service: DVRLLifecycleService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"rows": service.list_records(tenant_id, "virtualization_agents"),
		"columns": [
			"agent_id",
			"name",
			"runtime",
			"role",
			"scope",
			"owner",
			"purpose",
			"human_approval_required",
			"status",
		],
		"supported_runtimes": contract["agents"]["supported_runtimes"],
		"supported_roles": contract["agents"]["supported_roles"],
		"privileged_roles": contract["agents"]["privileged_roles"],
		"pending_reviews": [
			review
			for review in service.list_pending_reviews(tenant_id)
			if "agent_id" in review
		],
	}


def lifecycle_batch_model(service: DVRLLifecycleService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"rows": service.list_records(tenant_id, "lifecycle_batches"),
		"columns": ["batch_id", "event_stream", "mutation_count", "accepted", "status", "matched_rules"],
		"streaming": contract["streaming"],
	}


def audit_timeline_model(service: DVRLLifecycleService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"events": service.list_records(tenant_id, "audit_events"),
		"columns": ["created_at", "event_type", "subject", "actor", "decision", "matched_rules"],
	}


def settings_model(tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"agents": contract["agents"],
		"streaming": contract["streaming"],
		"review_evidence": contract["review_evidence"],
		"theme": contract["theme"],
		"routes": contract["ui"]["routes"],
	}
