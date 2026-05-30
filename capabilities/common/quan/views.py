"""UI metadata helpers for the Quantum Computing capability."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import QuanService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
	service: QuanService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or QuanService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"routes": capability_routes(tenant_id),
		"summary": service.dashboard_summary(tenant_id),
		"backends": service.list_backends(tenant_id),
		"circuits": service.list_circuits(tenant_id),
		"jobs": service.list_jobs(tenant_id),
		"results": service.list_results(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
		"streaming": contract["streaming"],
	}


def backend_registry_model(
	service: QuanService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or QuanService()
	return {
		"tenant_id": tenant_id,
		"route": _route("backends", tenant_id),
		"backends": service.list_backends(tenant_id),
		"quota_policies": service.list_quota_policies(tenant_id),
		"theme": service.describe(tenant_id)["theme"]["components"]["backend_card"],
	}


def circuit_library_model(
	service: QuanService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or QuanService()
	return {
		"tenant_id": tenant_id,
		"route": _route("circuits", tenant_id),
		"circuits": service.list_circuits(tenant_id),
		"theme": service.describe(tenant_id)["theme"]["components"]["circuit_library"],
	}


def job_queue_model(
	service: QuanService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or QuanService()
	return {
		"tenant_id": tenant_id,
		"route": _route("jobs", tenant_id),
		"jobs": service.list_jobs(tenant_id),
		"summary": service.dashboard_summary(tenant_id),
		"theme": service.describe(tenant_id)["theme"]["components"]["job_queue"],
	}


def experiment_workbench_model(
	service: QuanService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or QuanService()
	return {
		"tenant_id": tenant_id,
		"route": _route("experiments", tenant_id),
		"experiments": service.list_experiments(tenant_id),
		"circuits": service.list_circuits(tenant_id),
		"jobs": service.list_jobs(tenant_id),
	}


def result_viewer_model(
	service: QuanService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or QuanService()
	return {
		"tenant_id": tenant_id,
		"route": _route("results", tenant_id),
		"results": service.list_results(tenant_id),
		"theme": service.describe(tenant_id)["theme"]["components"]["result_viewer"],
	}


def governance_model(
	service: QuanService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or QuanService()
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"route": _route("governance", tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"audit_events": service.list_audit_events(tenant_id),
		"streaming": contract["streaming"],
		"permissions": sorted({route["permission"] for route in contract["ui"]["routes"]}),
	}


def quan_agent_model(
	service: QuanService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or QuanService()
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"route": _route("agents", tenant_id),
		"quan_agents": service.list_quan_agents(tenant_id),
		"supported_runtimes": contract["configuration"]["quan_agents"]["supported_runtimes"],
		"allowed_roles": contract["configuration"]["quan_agents"]["allowed_roles"],
		"permissions": ["quan:view", "quan:admin"],
	}


def audit_trail_model(
	service: QuanService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or QuanService()
	return {
		"tenant_id": tenant_id,
		"route": _route("audit", tenant_id),
		"audit_events": service.list_audit_events(tenant_id),
		"permissions": ["quan:admin"],
	}


def quantum_policy_model(
	service: QuanService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or QuanService()
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"route": _route("governance", tenant_id),
		"configuration": contract["configuration"],
		"rules": contract["rule_engine"]["rules"],
		"streaming": contract["streaming"],
		"quota_policies": service.list_quota_policies(tenant_id),
	}


def _route(name: str, tenant_id: str) -> dict[str, str]:
	for route in capability_routes(tenant_id):
		if route["name"] == name:
			return route
	raise KeyError(f"quan_route_not_found:{name}")
