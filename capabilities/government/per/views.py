"""View models for generated Permits Management screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import PermitsManagementService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import PermitsManagementService  # type: ignore


def dashboard_model(service: PermitsManagementService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": "Permits Management", "tenant_id": tenant_id, "summary": service.dashboard_summary(tenant_id), "theme": contract["theme"], "routes": contract["ui"]["routes"]}


def permits_console_model(service: PermitsManagementService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"applications": _tenant_items(service.applications, tenant_id),
		"permits": _tenant_items(service.permits, tenant_id),
		"conditions": _tenant_items(service.conditions, tenant_id),
		"inspections": _tenant_items(service.inspections, tenant_id),
		"compliance": _tenant_items(service.compliance, tenant_id),
		"enforcement": _tenant_items(service.enforcement, tenant_id),
	}


def agent_workbench_model(service: PermitsManagementService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"],
		"supported_roles": contract["configuration"]["agents"]["supported_roles"],
		"agents": [item.to_dict() for item in service.agents.values() if item.tenant_id == tenant_id],
	}


def _tenant_items(items: dict[Any, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [item.to_dict() for item in sorted(items.values(), key=lambda v: v.id) if item.tenant_id == tenant_id]
