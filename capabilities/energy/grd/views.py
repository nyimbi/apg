"""View models for Grid Operations screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import GridOperationsService
except ImportError:
	from capability_contract import get_capability_contract  # type: ignore
	from service import GridOperationsService  # type: ignore


def dashboard_model(svc: GridOperationsService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Grid Operations",
		"tenant_id": tenant_id,
		"summary": svc.dashboard_summary(tenant_id),
		"theme": contract["theme"],
		"routes": contract["ui"]["routes"],
	}


def state_estimation_model(svc: GridOperationsService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"se_runs": svc.list_se_runs(tenant_id),
		"latest_run": svc.get_latest_se_run(tenant_id),
		"supported_types": contract["configuration"]["state_estimation"]["supported_types"],
		"run_interval_seconds": contract["configuration"]["state_estimation"]["run_interval_seconds"],
	}


def contingency_model(svc: GridOperationsService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"all_cases": svc.list_contingency_cases(tenant_id),
		"violation_cases": svc.list_contingency_cases(tenant_id, has_violations=True),
		"supported_types": contract["configuration"]["contingency"]["supported_types"],
		"supported_statuses": contract["configuration"]["contingency"]["supported_statuses"],
	}


def contingency_detail_model(svc: GridOperationsService, tenant_id: str, case_id: str) -> dict[str, Any]:
	case = svc.contingency_cases.get((tenant_id, case_id))
	if not case:
		return {"error": "contingency_case_not_found", "case_id": case_id}
	return {
		"tenant_id": tenant_id,
		"case": case.to_dict(),
		"remedial_action_count": len(case.remedial_actions),
		"violation_count": len(case.violations),
	}


def voltage_control_model(svc: GridOperationsService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"actions": svc.list_voltage_control_actions(tenant_id),
		"supported_methods": contract["configuration"]["voltage_control"]["supported_methods"],
		"target_pu": contract["configuration"]["voltage_control"]["target_pu"],
		"tolerance_pu": contract["configuration"]["voltage_control"]["tolerance_pu"],
	}


def frequency_control_model(svc: GridOperationsService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"actions": svc.list_frequency_control_actions(tenant_id),
		"supported_methods": contract["configuration"]["frequency_control"]["supported_methods"],
		"nominal_hz": contract["configuration"]["frequency_control"]["nominal_hz"],
		"deadband_hz": contract["configuration"]["frequency_control"]["deadband_hz"],
		"ufls_threshold_hz": contract["configuration"]["frequency_control"]["ufls_threshold_hz"],
	}


def market_settlement_model(svc: GridOperationsService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	settlements = svc.list_settlements(tenant_id)
	preliminary = [s for s in settlements if s["status"] == "preliminary"]
	final = [s for s in settlements if s["status"] == "final"]
	return {
		"tenant_id": tenant_id,
		"settlements": settlements,
		"preliminary_count": len(preliminary),
		"final_count": len(final),
		"total_settlement_amount": sum(s["settlement_amount"] for s in settlements),
		"supported_products": contract["configuration"]["market"]["supported_products"],
		"supported_statuses": contract["configuration"]["market"]["supported_settlement_statuses"],
	}


def settlement_detail_model(svc: GridOperationsService, tenant_id: str, interval_id: str) -> dict[str, Any]:
	interval = svc.settlement_intervals.get((tenant_id, interval_id))
	if not interval:
		return {"error": "settlement_interval_not_found", "interval_id": interval_id}
	return {"tenant_id": tenant_id, "interval": interval.to_dict()}


def alarm_console_model(svc: GridOperationsService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	all_alarms = svc.list_alarms(tenant_id)
	active_alarms = svc.list_alarms(tenant_id, active_only=True)
	critical = [a for a in active_alarms if a["severity"] in ("critical", "emergency")]
	return {
		"tenant_id": tenant_id,
		"all_alarms": all_alarms,
		"active_alarms": active_alarms,
		"critical_alarms": critical,
		"supported_severities": contract["configuration"]["alarms"]["supported_severities"],
		"supported_categories": contract["configuration"]["alarms"]["supported_categories"],
	}


def ems_console_model(svc: GridOperationsService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"executions": svc.list_ems_executions(tenant_id),
		"supported_functions": contract["configuration"]["ems"]["supported_functions"],
		"real_time_enabled": contract["configuration"]["ems"]["real_time_enabled"],
		"study_mode_enabled": contract["configuration"]["ems"]["study_mode_enabled"],
	}


def agent_workbench_model(svc: GridOperationsService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"agents": _tenant_items(svc.agents, tenant_id),
		"supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"],
		"supported_roles": contract["configuration"]["agents"]["supported_roles"],
	}


def _tenant_items(store: dict[Any, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [v.to_dict() for k, v in store.items() if k[0] == tenant_id]
