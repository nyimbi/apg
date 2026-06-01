"""Generated-application view models for the MONI capability."""

from __future__ import annotations

from typing import Any

from .capability_contract import get_capability_contract
from .service import MoniService


def capability_routes(tenant_id: str = "default") -> list[dict[str, Any]]:
	"""Return route metadata for generated APG shells."""
	return get_capability_contract(tenant_id)["ui"]["routes"]


def dashboard_model(service: MoniService, tenant_id: str = "default") -> dict[str, Any]:
	"""Return MONI dashboard state."""
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Monitoring and Observability",
		"tenant_id": tenant_id,
		"summary": service.dashboard_summary(tenant_id),
		"pending_reviews": service.list_pending_reviews(tenant_id),
		"review_evidence": contract["review_evidence"],
		"theme": contract["theme"],
		"primary_actions": [
			{"id": "register_source", "label": "Register source", "permission": "moni:manage_sources"},
			{"id": "create_slo", "label": "Create SLO", "permission": "moni:manage_slos"},
			{"id": "create_alert", "label": "Create alert", "permission": "moni:manage_alerts"},
			{"id": "review_remediation", "label": "Review remediation", "permission": "moni:remediate"},
			{"id": "register_agent", "label": "Register agent", "permission": "moni:admin"},
		],
	}


def source_inventory_model(service: MoniService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"rows": service.list_records(tenant_id, "sources"),
		"columns": ["source_id", "service_name", "environment", "owner", "status", "notification_route"],
	}


def signal_explorer_model(service: MoniService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"rows": service.list_records(tenant_id, "signals"),
		"filters": ["source_id", "signal_type", "severity", "status"],
		"columns": ["created_at", "signal_type", "source_id", "name", "status", "decision", "matched_rules"],
	}


def slo_model(service: MoniService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"rows": service.list_records(tenant_id, "slos"),
		"columns": ["service_name", "objective", "threshold", "window_minutes", "owner", "status"],
	}


def alert_center_model(service: MoniService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"rows": service.list_records(tenant_id, "alerts"),
		"columns": ["created_at", "severity", "title", "source_id", "status", "owner", "incident_id"],
		"actions": ["acknowledge", "resolve", "open_incident"],
	}


def incident_model(service: MoniService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"rows": service.list_records(tenant_id, "incidents"),
		"columns": ["created_at", "severity", "title", "owner", "status", "alert_ids"],
	}


def remediation_model(service: MoniService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"rows": service.list_records(tenant_id, "remediation_requests"),
		"columns": ["created_at", "incident_id", "requester", "environment", "runbook_id", "status", "reviewer"],
		"review_actions": ["approved", "rejected"],
	}


def analytics_model(service: MoniService, tenant_id: str = "default") -> dict[str, Any]:
	summary = service.dashboard_summary(tenant_id)
	return {
		"tenant_id": tenant_id,
		"summary": summary,
		"panels": [
			{"id": "signal_volume", "value": summary["signal_count"]},
			{"id": "open_alerts", "value": summary["open_alert_count"]},
			{"id": "open_incidents", "value": summary["open_incident_count"]},
			{"id": "pending_remediation", "value": summary["pending_remediation_count"]},
			{"id": "monitoring_agents", "value": summary["monitoring_agent_count"]},
			{"id": "lifecycle_batches", "value": summary["lifecycle_batch_count"]},
		],
	}


def adapter_health_model(tenant_id: str = "default") -> dict[str, Any]:
	adapters = get_capability_contract(tenant_id)["configuration"]["adapters"]
	return {
		"tenant_id": tenant_id,
		"supported_collectors": adapters["supported_collectors"],
		"metrics_store": adapters["metrics_store"],
		"log_store": adapters["log_store"],
		"trace_store": adapters["trace_store"],
		"notification_adapter_required_for_critical": adapters["notification_adapter_required_for_critical"],
	}


def monitoring_agent_roster_model(service: MoniService, tenant_id: str = "default") -> dict[str, Any]:
	"""Return first-class monitoring-agent roster state."""
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"rows": service.list_records(tenant_id, "monitoring_agents"),
		"pending_reviews": [
			agent
			for agent in service.list_records(tenant_id, "monitoring_agents")
			if agent.get("status") == "pending_review"
		],
		"supported_runtimes": contract["agents"]["supported_runtimes"],
		"supported_roles": contract["agents"]["supported_roles"],
		"privileged_roles": contract["agents"]["privileged_roles"],
		"guardrails": contract["agents"]["guardrails"],
		"columns": ["name", "runtime", "role", "owner", "purpose", "status", "human_approval_required"],
	}


def lifecycle_batch_model(service: MoniService, tenant_id: str = "default") -> dict[str, Any]:
	"""Return Bytewax lifecycle-batch monitor state."""
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"streaming": contract["streaming"],
		"rows": service.list_records(tenant_id, "lifecycle_batches"),
		"columns": ["event_stream", "mutation_count", "accepted", "decision", "required_processor", "status"],
	}


def audit_timeline_model(service: MoniService, tenant_id: str = "default") -> dict[str, Any]:
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
