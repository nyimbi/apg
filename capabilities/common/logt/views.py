"""UI metadata helpers for APG Logging and Tracing."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import LogtService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
	service: LogtService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or LogtService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"routes": capability_routes(tenant_id),
		"summary": service.dashboard_summary(tenant_id),
		"pipelines": service.list_pipelines(tenant_id),
		"logs": service.list_logs(tenant_id),
		"traces": service.list_traces(tenant_id),
		"spans": service.list_spans(tenant_id),
		"queries": service.list_queries(tenant_id),
		"exports": service.list_exports(tenant_id),
		"retention_policies": service.list_retention_policies(tenant_id),
		"audit_events": service.list_audit_events(tenant_id),
		"service_map": service.service_map(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
		"streaming": contract["streaming"],
	}


def log_search_model(service: LogtService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"tenant_id": tenant_id,
		"logs": service.list_logs(tenant_id),
		"queries": service.list_queries(tenant_id),
	}


def trace_explorer_model(service: LogtService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"tenant_id": tenant_id,
		"traces": service.list_traces(tenant_id),
		"spans": service.list_spans(tenant_id),
		"service_map": service.service_map(tenant_id),
	}


def pipeline_manager_model(service: LogtService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"tenant_id": tenant_id,
		"pipelines": service.list_pipelines(tenant_id),
		"retention_policies": service.list_retention_policies(tenant_id),
	}


def retention_center_model(service: LogtService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"tenant_id": tenant_id,
		"retention_policies": service.list_retention_policies(tenant_id),
		"exports": service.list_exports(tenant_id),
	}


def analytics_model(service: LogtService, tenant_id: str = "default") -> dict[str, object]:
	summary = service.dashboard_summary(tenant_id)
	return {
		"tenant_id": tenant_id,
		"summary": summary,
		"service_map": service.service_map(tenant_id),
		"slow_spans": [
			span for span in service.list_spans(tenant_id)
			if span["status"] == "slow"
		],
		"error_logs": [
			log for log in service.list_logs(tenant_id)
			if log["severity"] in {"error", "critical"}
		],
	}


def logt_agent_model(service: LogtService, tenant_id: str = "default") -> dict[str, object]:
	contract = service.describe(tenant_id)
	return {
		"route": "/logt/agents",
		"logt_agents": service.list_logt_agents(tenant_id),
		"supported_runtimes": contract["configuration"]["logt_agents"]["supported_runtimes"],
		"allowed_roles": contract["configuration"]["logt_agents"]["allowed_roles"],
		"permissions": ["logt:view", "logt:admin"],
	}


def audit_trail_model(service: LogtService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"route": "/logt/audit",
		"audit_events": service.list_audit_events(tenant_id),
		"permissions": ["logt:admin"],
	}


def diagnostic_policy_model(service: LogtService, tenant_id: str = "default") -> dict[str, object]:
	contract = service.describe(tenant_id)
	return {
		"route": "/logt/settings",
		"rules": contract["rule_engine"]["rules"],
		"retention_policies": service.list_retention_policies(tenant_id),
		"streaming": contract["streaming"],
		"configuration": contract["configuration"],
	}
