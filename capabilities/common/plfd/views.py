"""UI metadata and view-model helpers for Platform Foundation."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import PlfdService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
	service: PlfdService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or PlfdService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"summary": service.dashboard_summary(tenant_id),
		"routes": capability_routes(tenant_id),
		"services": service.list_services(tenant_id),
		"readiness": service.list_readiness_assessments(tenant_id),
		"changes": service.list_changes(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
	}


def services_model(service: PlfdService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"route": "/plfd/services",
		"services": service.list_services(tenant_id),
		"dependencies": service.list_dependencies(tenant_id),
	}


def dependency_map_model(service: PlfdService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"route": "/plfd/dependencies",
		"nodes": service.list_services(tenant_id),
		"edges": service.list_dependencies(tenant_id),
	}


def baseline_manager_model(service: PlfdService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"route": "/plfd/baselines",
		"baselines": service.list_baselines(tenant_id),
	}


def readiness_gate_model(service: PlfdService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"route": "/plfd/readiness",
		"assessments": service.list_readiness_assessments(tenant_id),
		"summary": service.dashboard_summary(tenant_id),
	}


def change_queue_model(service: PlfdService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"route": "/plfd/changes",
		"changes": service.list_changes(tenant_id),
	}


def governance_model(service: PlfdService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"route": "/plfd/governance",
		"audit_events": service.list_audit_events(tenant_id),
		"rules": service.describe(tenant_id)["rule_engine"]["rules"],
	}
