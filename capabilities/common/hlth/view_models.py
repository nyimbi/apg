"""Generated-application view models for the HLTH capability."""

from __future__ import annotations

from typing import Any

from .capability_contract import get_capability_contract
from .service import HlthService


def capability_routes(tenant_id: str = "default") -> list[dict[str, Any]]:
	"""Return route metadata for generated APG shells."""
	return get_capability_contract(tenant_id)["ui"]["routes"]


def dashboard_model(service: HlthService, tenant_id: str = "default") -> dict[str, Any]:
	"""Return HLTH dashboard state."""
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Health Checks and Diagnostics",
		"tenant_id": tenant_id,
		"summary": service.dashboard_summary(tenant_id),
		"theme": contract["theme"],
		"primary_actions": [
			{"id": "register_component", "label": "Register component", "permission": "health.manage"},
			{"id": "record_check", "label": "Record check", "permission": "health.manage"},
			{"id": "review_remediation", "label": "Review remediation", "permission": "health.remediate"},
			{"id": "evaluate_gate", "label": "Evaluate gate", "permission": "health.deployments.review"},
			{"id": "register_agent", "label": "Register agent", "permission": "health.admin"},
		],
	}


def component_inventory_model(service: HlthService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"rows": service.list_records(tenant_id, "components"),
		"columns": ["component_id", "name", "component_type", "environment", "owner", "criticality", "status"],
	}


def check_timeline_model(service: HlthService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"rows": service.list_records(tenant_id, "checks"),
		"filters": ["component_id", "dimension", "severity", "status"],
		"columns": ["created_at", "component_id", "dimension", "score", "status", "decision", "matched_rules"],
	}


def baseline_model(service: HlthService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"rows": service.list_records(tenant_id, "baselines"),
		"columns": ["component_id", "dimension", "expected_score", "sample_count", "reviewed", "status"],
	}


def prediction_model(service: HlthService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"rows": service.list_records(tenant_id, "predictions"),
		"columns": ["component_id", "predicted_score", "confidence", "risk", "status", "matched_rules"],
	}


def alert_center_model(service: HlthService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"rows": service.list_records(tenant_id, "alerts"),
		"columns": ["created_at", "severity", "title", "component_id", "status", "owner", "incident_id"],
		"actions": ["acknowledge", "resolve", "open_incident"],
	}


def incident_model(service: HlthService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"rows": service.list_records(tenant_id, "incidents"),
		"columns": ["created_at", "severity", "title", "owner", "status", "component_ids", "alert_ids"],
	}


def remediation_model(service: HlthService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"rows": service.list_records(tenant_id, "remediation_requests"),
		"columns": ["created_at", "incident_id", "requester", "environment", "runbook_id", "status", "reviewer"],
		"review_actions": ["approved", "rejected"],
	}


def deployment_gate_model(service: HlthService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"rows": service.list_records(tenant_id, "deployment_gates"),
		"columns": ["created_at", "deployment_id", "decision", "status", "unresolved_critical_incidents", "waiver_recorded"],
	}


def report_model(service: HlthService, tenant_id: str = "default") -> dict[str, Any]:
	summary = service.dashboard_summary(tenant_id)
	return {
		"tenant_id": tenant_id,
		"summary": summary,
		"sections": ["components", "checks", "incidents", "remediation", "deployment_gates", "health_agents", "lifecycle_batches"],
	}


def adapter_health_model(tenant_id: str = "default") -> dict[str, Any]:
	adapters = get_capability_contract(tenant_id)["configuration"]["adapters"]
	return {
		"tenant_id": tenant_id,
		"supported_probe_sources": adapters["supported_probe_sources"],
		"notification_adapter_required_for_critical": adapters["notification_adapter_required_for_critical"],
		"remediation_executor": adapters["remediation_executor"],
		"deployment_gate_adapter": adapters["deployment_gate_adapter"],
		"prediction_engine": adapters["prediction_engine"],
	}


def health_agent_roster_model(service: HlthService, tenant_id: str = "default") -> dict[str, Any]:
	"""Return first-class health-agent roster state."""
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"rows": service.list_records(tenant_id, "health_agents"),
		"supported_runtimes": contract["agents"]["supported_runtimes"],
		"supported_roles": contract["agents"]["supported_roles"],
		"privileged_roles": contract["agents"]["privileged_roles"],
		"guardrails": contract["agents"]["guardrails"],
		"columns": ["name", "runtime", "role", "owner", "purpose", "status", "human_approval_required"],
	}


def lifecycle_batch_model(service: HlthService, tenant_id: str = "default") -> dict[str, Any]:
	"""Return Bytewax lifecycle-batch monitor state."""
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"streaming": contract["streaming"],
		"rows": service.list_records(tenant_id, "lifecycle_batches"),
		"columns": ["event_stream", "mutation_count", "accepted", "decision", "required_processor", "status"],
	}


def audit_timeline_model(service: HlthService, tenant_id: str = "default") -> dict[str, Any]:
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
		"theme": contract["theme"],
		"routes": contract["ui"]["routes"],
	}
