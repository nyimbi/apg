"""Process-local API helpers for APG Resource Management (res)."""

from __future__ import annotations

try:
	from .service import ResourceManagementService
except ImportError:  # pragma: no cover
	import sys as _sys, pathlib as _pl
	_here = str(_pl.Path(__file__).parent)
	if _here not in _sys.path:
		_sys.path.insert(0, _here)
	from service import ResourceManagementService  # type: ignore

_SERVICE = ResourceManagementService()


def service() -> ResourceManagementService:
	return _SERVICE


def create_resource(payload: dict):
	return _SERVICE.create_resource(
		payload["resource_id"], payload.get("tenant_id", "default"),
		payload["name"], payload["resource_type"],
		payload.get("status", "available"), payload.get("department", ""),
		payload["owner_id"], float(payload["cost_rate"]),
		payload.get("cost_rate_type", "standard_cost"),
		payload["evidence_reference"], payload.get("policy_attached", True),
	)


def add_skill(payload: dict):
	return _SERVICE.add_skill(
		payload["skill_id"], payload.get("tenant_id", "default"),
		payload["resource_id"], payload["skill_name"],
		payload["proficiency_level"], float(payload.get("years_experience", 0.0)),
		payload["evidence_reference"],
	)


def match_skills(payload: dict):
	return _SERVICE.match_skills(
		payload.get("tenant_id", "default"),
		payload["required_skills"],
		payload.get("algorithm", "exact_skill_match"),
	)


def create_allocation(payload: dict):
	return _SERVICE.create_allocation(
		payload["alloc_id"], payload.get("tenant_id", "default"),
		payload["resource_id"], payload["project_id"],
		payload.get("task_id", ""), payload.get("status", "confirmed"),
		payload.get("start_date", ""), payload.get("end_date", ""),
		float(payload.get("allocation_pct", 100.0)),
		payload.get("manager_approval_reference", ""),
		payload.get("over_allocated", False),
	)


def create_capacity_plan(payload: dict):
	return _SERVICE.create_capacity_plan(
		payload["plan_id"], payload.get("tenant_id", "default"),
		payload["plan_type"], payload["name"],
		payload.get("horizon", "medium_term_90d"),
		payload.get("demand_data", "{}"), payload.get("supply_data", "{}"),
		payload.get("gap_analysis", "{}"), payload.get("created_by", "system"),
	)


def take_utilisation_snapshot(payload: dict):
	return _SERVICE.take_utilisation_snapshot(
		payload["snapshot_id"], payload.get("tenant_id", "default"),
		payload["resource_id"], payload.get("snapshot_period", ""),
		float(payload["allocated_hours"]), float(payload["available_hours"]),
	)


def forecast_demand(payload: dict):
	return _SERVICE.forecast_demand(
		payload["forecast_id"], payload.get("tenant_id", "default"),
		payload["horizon"], payload.get("resource_type", "human"),
		payload.get("skill_filter", ""), float(payload["forecast_demand_fte"]),
		float(payload["current_supply_fte"]), payload.get("generated_by", "system"),
	)


def record_leave(payload: dict):
	return _SERVICE.record_leave(
		payload["leave_id"], payload.get("tenant_id", "default"),
		payload["resource_id"], payload["leave_type"],
		payload["start_date"], payload["end_date"],
		payload["approval_reference"],
	)


def set_cost_rate(payload: dict):
	return _SERVICE.set_cost_rate(
		payload["rate_id"], payload.get("tenant_id", "default"),
		payload["resource_id"], payload["rate_type"],
		float(payload["rate_amount"]), payload.get("currency", "USD"),
		payload["effective_date"], payload["finance_approval_reference"],
	)


def register_agent(payload: dict):
	return _SERVICE.register_agent(
		payload["agent_id"], payload.get("tenant_id", "default"),
		payload["name"], payload["runtime"], payload["role"],
		payload.get("scope", "resource management operations"),
	)


def validate_agent_action(payload: dict):
	return _SERVICE.validate_agent_action(
		payload.get("tenant_id", "default"),
		payload.get("privileged_scope", False),
		payload.get("human_approval_recorded", False),
	)


def validate_batch(payload: dict):
	return _SERVICE.validate_batch(
		payload.get("tenant_id", "default"),
		payload["item_count"],
		payload.get("event_stream", "bytewax"),
	)


def dashboard(payload: dict):
	return _SERVICE.dashboard_summary(payload.get("tenant_id", "default"))
