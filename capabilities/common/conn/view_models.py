"""Generated-app UI view models for the CONN capability."""

from __future__ import annotations

from typing import Any

from .capability_contract import get_capability_contract
from .conn_runtime import ConnService


def dashboard_model(service: ConnService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Connection Management",
		"tenant_id": tenant_id,
		"summary": service.dashboard_summary(tenant_id),
		"connector_agents": service.list_connector_agents(tenant_id),
		"lifecycle_batches": service.list_lifecycle_batches(tenant_id),
		"pending_reviews": service.list_pending_reviews(tenant_id),
		"review_evidence": contract["review_evidence"],
		"routes": contract["ui"]["routes"],
		"theme": contract["theme"],
	}


def connector_catalog_model(service: ConnService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"connectors": service.list_connectors(tenant_id),
		"config": get_capability_contract(tenant_id)["configuration"]["connectors"],
	}


def connection_workbench_model(service: ConnService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"connections": service.list_connections(tenant_id),
		"actions": ["register_connection", "record_connection_test", "activate_connection", "transfer_owner", "retire_connection"],
		"security": get_capability_contract(tenant_id)["configuration"]["security"],
	}


def flow_designer_model(service: ConnService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"flows": service.list_flows(tenant_id),
		"connections": service.list_connections(tenant_id),
		"defaults": get_capability_contract(tenant_id)["configuration"]["flows"],
	}


def sync_monitor_model(service: ConnService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"runs": service.list_sync_runs(tenant_id),
		"schedules": service.list_schedules(tenant_id),
		"sync": get_capability_contract(tenant_id)["configuration"]["sync"],
	}


def quality_console_model(service: ConnService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"quality": get_capability_contract(tenant_id)["configuration"]["quality"],
		"runs": service.list_sync_runs(tenant_id),
	}


def lineage_model(service: ConnService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"flows": service.list_flows(tenant_id),
		"lineage_required": get_capability_contract(tenant_id)["configuration"]["governance"]["lineage_capture_required"],
	}


def marketplace_model(service: ConnService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"connectors": service.list_connectors(tenant_id),
		"reviews": [review for review in service.list_reviews(tenant_id) if review["review_type"] == "marketplace"],
	}


def security_console_model(service: ConnService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"connections": service.list_connections(tenant_id),
		"security": get_capability_contract(tenant_id)["configuration"]["security"],
	}


def audit_timeline_model(service: ConnService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"events": service.list_audit_events(tenant_id),
		"observability": get_capability_contract(tenant_id)["configuration"]["observability"],
	}


def connector_agent_roster_model(service: ConnService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"agents": service.list_connector_agents(tenant_id),
		"pending_reviews": [
			record
			for record in service.list_pending_reviews(tenant_id)
			if record.get("role") in contract["agents"]["supported_roles"]
		],
		"agent_contract": contract["agents"],
		"supported_runtimes": contract["agents"]["supported_runtimes"],
		"supported_roles": contract["agents"]["supported_roles"],
		"privileged_roles": contract["agents"]["privileged_roles"],
		"required_fields": ["name", "runtime", "role", "scope", "owner", "purpose", "contribution_disclosed"],
	}


def lifecycle_batch_model(service: ConnService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"batches": service.list_lifecycle_batches(tenant_id),
		"streaming": contract["streaming"],
		"required_processor": contract["streaming"]["required_processor"],
		"summary": service.dashboard_summary(tenant_id),
	}


def settings_model(service: ConnService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"configuration": contract["configuration"],
		"schema": contract["configuration_schema"],
		"adapters": contract["configuration"]["adapters"],
		"agents": contract["agents"],
		"streaming": contract["streaming"],
		"review_evidence": contract["review_evidence"],
		"theme": contract["theme"],
	}
