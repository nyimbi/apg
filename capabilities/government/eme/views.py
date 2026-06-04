"""View models for generated Emergency Management screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import EmergencyManagementService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import EmergencyManagementService  # type: ignore


def dashboard_model(service: EmergencyManagementService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": "Emergency Management", "tenant_id": tenant_id, "summary": service.dashboard_summary(tenant_id), "theme": contract["theme"], "routes": contract["ui"]["routes"]}


def emergency_console_model(service: EmergencyManagementService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"incidents": _tenant_items(service.incidents, tenant_id),
		"resources": _tenant_items(service.resources, tenant_id),
		"agencies": _tenant_items(service.agencies, tenant_id),
		"eoc_records": _tenant_items(service.eoc_records, tenant_id),
		"situation_reports": _tenant_items(service.situation_reports, tenant_id),
		"after_action_reviews": _tenant_items(service.after_action_reviews, tenant_id),
	}


def agent_workbench_model(service: EmergencyManagementService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"],
		"supported_roles": contract["configuration"]["agents"]["supported_roles"],
		"agents": [item.to_dict() for item in service.agents.values() if item.tenant_id == tenant_id],
	}


def _tenant_items(items: dict[Any, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [item.to_dict() for item in sorted(items.values(), key=lambda v: v.id) if item.tenant_id == tenant_id]
