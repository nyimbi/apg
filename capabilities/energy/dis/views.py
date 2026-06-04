"""View models for Distribution Network screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import DistributionNetworkService
except ImportError:
	from capability_contract import get_capability_contract  # type: ignore
	from service import DistributionNetworkService  # type: ignore


def dashboard_model(svc: DistributionNetworkService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Distribution Network",
		"tenant_id": tenant_id,
		"summary": svc.dashboard_summary(tenant_id),
		"theme": contract["theme"],
		"routes": contract["ui"]["routes"],
	}


def topology_model(svc: DistributionNetworkService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"feeders": svc.list_feeders(tenant_id),
		"elements": svc.list_elements(tenant_id),
		"supported_voltage_levels": contract["configuration"]["network"]["supported_voltage_levels"],
	}


def fault_management_model(svc: DistributionNetworkService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"faults": svc.list_faults(tenant_id),
		"active_faults": svc.list_faults(tenant_id, status="detected"),
		"supported_fault_types": contract["configuration"]["faults"]["supported_fault_types"],
		"supported_statuses": contract["configuration"]["faults"]["supported_statuses"],
	}


def fault_detail_model(svc: DistributionNetworkService, tenant_id: str, fault_id: str) -> dict[str, Any]:
	fault = svc.faults.get((tenant_id, fault_id))
	if not fault:
		return {"error": "fault_not_found", "fault_id": fault_id}
	related_switching = [
		o for o in svc.list_switching_orders(tenant_id)
		if o["element_id"] == fault.element_id
	]
	return {
		"tenant_id": tenant_id,
		"fault": fault.to_dict(),
		"related_switching_orders": related_switching,
	}


def switching_orders_model(svc: DistributionNetworkService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"switching_orders": svc.list_switching_orders(tenant_id),
		"supported_operations": contract["configuration"]["switching"]["supported_operations"],
	}


def outage_manager_model(svc: DistributionNetworkService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"outages": svc.list_outages(tenant_id),
		"active_outages": [o for o in svc.list_outages(tenant_id) if not o.get("restored_at")],
	}


def scada_console_model(svc: DistributionNetworkService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	readings = _tenant_items(svc.scada_readings, tenant_id)
	return {
		"tenant_id": tenant_id,
		"recent_readings": readings[-100:],  # last 100 readings
		"supported_protocols": contract["configuration"]["scada"]["supported_protocols"],
		"polling_interval_seconds": contract["configuration"]["scada"]["polling_interval_seconds"],
	}


def load_balancing_model(svc: DistributionNetworkService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"load_balance_actions": _tenant_items(svc.load_balance_actions, tenant_id),
		"feeders": svc.list_feeders(tenant_id),
		"supported_modes": contract["configuration"]["load_balancing"]["supported_modes"],
		"voltage_limits": contract["configuration"]["load_balancing"]["voltage_limits"],
	}


def reliability_kpi_model(svc: DistributionNetworkService, tenant_id: str = "default") -> dict[str, Any]:
	outages = svc.list_outages(tenant_id)
	total_saidi = sum(o.get("saidi_minutes", 0) for o in outages)
	total_customers_interrupted = sum(o.get("affected_customers", 0) for o in outages)
	return {
		"tenant_id": tenant_id,
		"total_saidi_minutes": total_saidi,
		"total_customers_interrupted": total_customers_interrupted,
		"outage_count": len(outages),
	}


def agent_workbench_model(svc: DistributionNetworkService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"agents": _tenant_items(svc.agents, tenant_id),
		"supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"],
		"supported_roles": contract["configuration"]["agents"]["supported_roles"],
	}


def _tenant_items(store: dict[Any, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [v.to_dict() for k, v in store.items() if k[0] == tenant_id]
