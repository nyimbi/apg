"""View models for Smart Metering & AMI screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import SmartMeteringService
except ImportError:
	from capability_contract import get_capability_contract  # type: ignore
	from service import SmartMeteringService  # type: ignore


def dashboard_model(svc: SmartMeteringService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Smart Metering & AMI",
		"tenant_id": tenant_id,
		"summary": svc.dashboard_summary(tenant_id),
		"theme": contract["theme"],
		"routes": contract["ui"]["routes"],
	}


def meter_registry_model(svc: SmartMeteringService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"meters": svc.list_meters(tenant_id),
		"supported_meter_types": contract["configuration"]["meters"]["supported_types"],
		"supported_statuses": contract["configuration"]["meters"]["supported_statuses"],
		"supported_comm_tech": contract["configuration"]["meters"]["supported_comm_tech"],
	}


def meter_detail_model(svc: SmartMeteringService, tenant_id: str, meter_id: str) -> dict[str, Any]:
	meter = svc.get_meter(tenant_id, meter_id)
	readings = svc.list_readings(tenant_id, meter_id=meter_id)
	tampers = svc.list_tamper_events(tenant_id)
	meter_tampers = [t for t in tampers if t["meter_id"] == meter_id]
	commands = svc.list_commands(tenant_id, meter_id=meter_id)
	return {
		"tenant_id": tenant_id,
		"meter": meter,
		"recent_readings": readings[-48:],  # last 48 intervals
		"tamper_events": meter_tampers,
		"recent_commands": commands[-10:],
	}


def interval_data_model(svc: SmartMeteringService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"readings": _tenant_items(svc.readings, tenant_id),
		"supported_reading_types": contract["configuration"]["readings"]["supported_reading_types"],
		"supported_intervals": contract["configuration"]["readings"]["supported_intervals"],
		"supported_quality_flags": contract["configuration"]["readings"]["supported_quality_flags"],
	}


def tamper_alert_model(svc: SmartMeteringService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"tamper_events": svc.list_tamper_events(tenant_id),
		"open_tampers": svc.list_tamper_events(tenant_id, status="open"),
		"supported_tamper_types": contract["configuration"]["tamper"]["supported_tamper_types"],
	}


def remote_command_model(svc: SmartMeteringService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"commands": svc.list_commands(tenant_id),
		"pending_commands": [c for c in svc.list_commands(tenant_id) if c["status"] == "pending"],
		"supported_command_types": contract["configuration"]["commands"]["supported_types"],
		"supported_statuses": contract["configuration"]["commands"]["supported_statuses"],
	}


def demand_response_model(svc: SmartMeteringService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"dr_events": svc.list_dr_events(tenant_id),
		"active_dr_events": [d for d in svc.list_dr_events(tenant_id) if d["status"] == "active"],
		"supported_event_types": contract["configuration"]["demand_response"]["supported_event_types"],
		"opt_out_allowed": contract["configuration"]["demand_response"]["opt_out_allowed"],
	}


def data_quality_model(svc: SmartMeteringService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"quality_flags": _tenant_items(svc.quality_flags, tenant_id),
		"supported_quality_flags": contract["configuration"]["readings"]["supported_quality_flags"],
	}


def head_end_model(svc: SmartMeteringService, tenant_id: str = "default") -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"head_end_statuses": _tenant_items(svc.head_end_statuses, tenant_id),
	}


def agent_workbench_model(svc: SmartMeteringService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"agents": _tenant_items(svc.agents, tenant_id),
		"supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"],
		"supported_roles": contract["configuration"]["agents"]["supported_roles"],
	}


def _tenant_items(store: dict[Any, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [v.to_dict() for k, v in store.items() if k[0] == tenant_id]
