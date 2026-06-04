"""View models for Generation Management screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import GenerationManagementService
except ImportError:
	from capability_contract import get_capability_contract  # type: ignore
	from service import GenerationManagementService  # type: ignore


def dashboard_model(svc: GenerationManagementService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Generation Management",
		"tenant_id": tenant_id,
		"summary": svc.dashboard_summary(tenant_id),
		"theme": contract["theme"],
		"routes": contract["ui"]["routes"],
	}


def plant_list_model(svc: GenerationManagementService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"plants": svc.list_plants(tenant_id),
		"supported_plant_types": contract["configuration"]["plants"]["supported_plant_types"],
		"supported_statuses": contract["configuration"]["plants"]["supported_statuses"],
	}


def plant_detail_model(svc: GenerationManagementService, tenant_id: str, plant_id: str) -> dict[str, Any]:
	plant = svc.get_plant(tenant_id, plant_id)
	outages = [o for o in svc.list_outages(tenant_id) if o["plant_id"] == plant_id]
	schedules = [s for s in svc.list_dispatch_schedules(tenant_id) if s["plant_id"] == plant_id]
	kpis = svc.list_kpis(tenant_id, plant_id=plant_id)
	fuel_stocks = [f for f in svc.list_fuel_stocks(tenant_id) if f["plant_id"] == plant_id]
	return {
		"tenant_id": tenant_id,
		"plant": plant,
		"outages": outages,
		"schedules": schedules,
		"kpis": kpis,
		"fuel_stocks": fuel_stocks,
	}


def dispatch_console_model(svc: GenerationManagementService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"schedules": svc.list_dispatch_schedules(tenant_id),
		"plants": svc.list_plants(tenant_id),
		"supported_modes": contract["configuration"]["dispatch"]["supported_modes"],
		"supported_statuses": contract["configuration"]["dispatch"]["supported_schedule_statuses"],
	}


def outage_manager_model(svc: GenerationManagementService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"outages": svc.list_outages(tenant_id),
		"plants": svc.list_plants(tenant_id),
		"supported_outage_types": contract["configuration"]["outages"]["supported_outage_types"],
		"supported_statuses": contract["configuration"]["outages"]["supported_statuses"],
	}


def kpi_model(svc: GenerationManagementService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"kpis": svc.list_kpis(tenant_id),
		"plants": svc.list_plants(tenant_id),
		"supported_kpi_types": contract["configuration"]["kpis"]["supported_kpi_types"],
		"supported_periods": contract["configuration"]["kpis"]["supported_periods"],
	}


def capacity_planner_model(svc: GenerationManagementService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"plans": svc.list_capacity_plans(tenant_id),
		"plants": svc.list_plants(tenant_id),
	}


def fuel_management_model(svc: GenerationManagementService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"fuel_stocks": svc.list_fuel_stocks(tenant_id),
		"low_fuel_alerts": svc.get_low_fuel_alerts(tenant_id),
		"plants": svc.list_plants(tenant_id),
	}


def agent_workbench_model(svc: GenerationManagementService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"agents": _tenant_items(svc.agents, tenant_id),
		"supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"],
		"supported_roles": contract["configuration"]["agents"]["supported_roles"],
	}


def _tenant_items(store: dict[Any, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [v.to_dict() for k, v in store.items() if k[0] == tenant_id]
