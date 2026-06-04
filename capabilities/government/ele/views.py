"""View models for generated Electoral & Civil Registration screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import ElectoralService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import ElectoralService  # type: ignore


def dashboard_model(service: ElectoralService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": "Electoral and Civil Registration", "tenant_id": tenant_id, "summary": service.dashboard_summary(tenant_id), "theme": contract["theme"], "routes": contract["ui"]["routes"]}


def electoral_console_model(service: ElectoralService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"registrations": _tenant_items(service.registrations, tenant_id),
		"polling_stations": _tenant_items(service.polling_stations, tenant_id),
		"elections": _tenant_items(service.elections, tenant_id),
		"results": _tenant_items(service.results, tenant_id),
		"civil_events": _tenant_items(service.civil_events, tenant_id),
	}


def agent_workbench_model(service: ElectoralService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"],
		"supported_roles": contract["configuration"]["agents"]["supported_roles"],
		"agents": [item.to_dict() for item in service.agents.values() if item.tenant_id == tenant_id],
	}


def _tenant_items(items: dict[Any, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [item.to_dict() for item in sorted(items.values(), key=lambda v: v.id) if item.tenant_id == tenant_id]
