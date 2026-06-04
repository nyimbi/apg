"""View models for generated Citizen Services Portal screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import CitizenServicesService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import CitizenServicesService  # type: ignore


def dashboard_model(service: CitizenServicesService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": "Citizen Services Portal", "tenant_id": tenant_id, "summary": service.dashboard_summary(tenant_id), "theme": contract["theme"], "routes": contract["ui"]["routes"]}


def portal_console_model(service: CitizenServicesService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"services": _tenant_items(service.services, tenant_id),
		"applications": _tenant_items(service.applications, tenant_id),
		"payments": _tenant_items(service.payments, tenant_id),
		"verifications": _tenant_items(service.verifications, tenant_id),
		"notifications": _tenant_items(service.notifications, tenant_id),
		"deliveries": _tenant_items(service.deliveries, tenant_id),
	}


def agent_workbench_model(service: CitizenServicesService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"],
		"supported_roles": contract["configuration"]["agents"]["supported_roles"],
		"agents": [item.to_dict() for item in service.agents.values() if item.tenant_id == tenant_id],
	}


def _tenant_items(items: dict[Any, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [item.to_dict() for item in sorted(items.values(), key=lambda v: v.id) if item.tenant_id == tenant_id]
