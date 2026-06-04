"""REST API for APG Generation Management."""

from __future__ import annotations

import json
from typing import Any

try:
	from .service import GenerationManagementService
	from .views import (
		agent_workbench_model, capacity_planner_model, dashboard_model,
		dispatch_console_model, fuel_management_model, kpi_model,
		outage_manager_model, plant_detail_model, plant_list_model,
	)
except ImportError:
	from service import GenerationManagementService  # type: ignore
	from views import (  # type: ignore
		agent_workbench_model, capacity_planner_model, dashboard_model,
		dispatch_console_model, fuel_management_model, kpi_model,
		outage_manager_model, plant_detail_model, plant_list_model,
	)

_SERVICE = GenerationManagementService()


def _ok(data: Any) -> dict[str, Any]:
	return {"status": "ok", "data": data}


def _err(reason: str, code: int = 400) -> dict[str, Any]:
	return {"status": "error", "code": code, "reason": reason}


def _tenant(payload: dict[str, Any]) -> str:
	return payload.get("tenant_id", "default")


# ── contract ──────────────────────────────────────────────────────────────────

def get_contract(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-gen/api/v1/contract — Return capability contract."""
	return _ok(_SERVICE.describe(_tenant(payload)))


def evaluate_rules(payload: dict[str, Any]) -> dict[str, Any]:
	"""POST /energy-gen/api/v1/evaluate — Evaluate rules against context."""
	return _ok(_SERVICE.evaluate(payload.get("context", {})))


# ── dashboard ─────────────────────────────────────────────────────────────────

def get_dashboard(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-gen/api/v1/dashboard — Return dashboard summary."""
	return _ok(dashboard_model(_SERVICE, _tenant(payload)))


# ── plants ────────────────────────────────────────────────────────────────────

def list_plants(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-gen/api/v1/plants — List all plants."""
	return _ok(plant_list_model(_SERVICE, _tenant(payload)))


def create_plant(payload: dict[str, Any]) -> dict[str, Any]:
	"""POST /energy-gen/api/v1/plants — Register a new plant."""
	try:
		result = _SERVICE.register_plant(
			plant_id=payload["plant_id"],
			tenant_id=_tenant(payload),
			name=payload["name"],
			plant_type=payload["plant_type"],
			fuel_type=payload["fuel_type"],
			capacity_mw=float(payload["capacity_mw"]),
			owner_id=payload["owner_id"],
			commissioning_date=payload["commissioning_date"],
			location_reference=payload.get("location_reference", ""),
			policy_attached=payload.get("policy_attached", True),
		)
		return _ok(result)
	except (KeyError, TypeError) as exc:
		return _err(f"missing_field: {exc}")
	except ValueError as exc:
		return _err(str(exc))


def get_plant(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-gen/api/v1/plants/<id> — Get plant detail."""
	try:
		plant_id = payload["plant_id"]
		return _ok(plant_detail_model(_SERVICE, _tenant(payload), plant_id))
	except KeyError as exc:
		return _err(f"not_found: {exc}", 404)


def update_plant_status(payload: dict[str, Any]) -> dict[str, Any]:
	"""PUT /energy-gen/api/v1/plants/<id>/status — Update plant status."""
	try:
		result = _SERVICE.update_plant_status(
			plant_id=payload["plant_id"],
			tenant_id=_tenant(payload),
			new_status=payload["status"],
		)
		return _ok(result)
	except KeyError as exc:
		return _err(f"not_found: {exc}", 404)
	except ValueError as exc:
		return _err(str(exc))


def decommission_plant(payload: dict[str, Any]) -> dict[str, Any]:
	"""DELETE /energy-gen/api/v1/plants/<id> — Decommission a plant."""
	try:
		result = _SERVICE.decommission_plant(
			plant_id=payload["plant_id"],
			tenant_id=_tenant(payload),
			approved_by=payload.get("approved_by", ""),
		)
		return _ok(result)
	except (KeyError, ValueError) as exc:
		return _err(str(exc))


# ── dispatch schedules ────────────────────────────────────────────────────────

def list_schedules(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-gen/api/v1/schedules — List dispatch schedules."""
	return _ok(dispatch_console_model(_SERVICE, _tenant(payload)))


def create_schedule(payload: dict[str, Any]) -> dict[str, Any]:
	"""POST /energy-gen/api/v1/schedules — Create a dispatch schedule."""
	try:
		result = _SERVICE.create_dispatch_schedule(
			schedule_id=payload["schedule_id"],
			tenant_id=_tenant(payload),
			plant_id=payload["plant_id"],
			dispatch_mode=payload["dispatch_mode"],
			scheduled_mw=float(payload["scheduled_mw"]),
			start_time=payload["start_time"],
			end_time=payload["end_time"],
			policy_attached=payload.get("policy_attached", True),
		)
		return _ok(result)
	except (KeyError, TypeError) as exc:
		return _err(f"missing_field: {exc}")
	except ValueError as exc:
		return _err(str(exc))


def approve_schedule(payload: dict[str, Any]) -> dict[str, Any]:
	"""PUT /energy-gen/api/v1/schedules/<id>/approve — Approve a schedule."""
	try:
		result = _SERVICE.approve_dispatch_schedule(
			schedule_id=payload["schedule_id"],
			tenant_id=_tenant(payload),
			approved_by=payload.get("approved_by", ""),
		)
		return _ok(result)
	except (KeyError, ValueError) as exc:
		return _err(str(exc))


# ── outages ───────────────────────────────────────────────────────────────────

def list_outages(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-gen/api/v1/outages — List plant outages."""
	return _ok(outage_manager_model(_SERVICE, _tenant(payload)))


def create_outage(payload: dict[str, Any]) -> dict[str, Any]:
	"""POST /energy-gen/api/v1/outages — Schedule an outage."""
	try:
		result = _SERVICE.schedule_outage(
			outage_id=payload["outage_id"],
			tenant_id=_tenant(payload),
			plant_id=payload["plant_id"],
			outage_type=payload["outage_type"],
			planned_start=payload["planned_start"],
			planned_end=payload["planned_end"],
			reason=payload.get("reason", ""),
			evidence_reference=payload.get("evidence_reference", ""),
			policy_attached=payload.get("policy_attached", True),
		)
		return _ok(result)
	except (KeyError, TypeError) as exc:
		return _err(f"missing_field: {exc}")
	except ValueError as exc:
		return _err(str(exc))


def approve_outage(payload: dict[str, Any]) -> dict[str, Any]:
	"""PUT /energy-gen/api/v1/outages/<id>/approve — Approve an outage."""
	try:
		result = _SERVICE.approve_outage(
			outage_id=payload["outage_id"],
			tenant_id=_tenant(payload),
			approved_by=payload.get("approved_by", ""),
		)
		return _ok(result)
	except (KeyError, ValueError) as exc:
		return _err(str(exc))


def start_outage(payload: dict[str, Any]) -> dict[str, Any]:
	"""PUT /energy-gen/api/v1/outages/<id>/start — Start an outage."""
	try:
		return _ok(_SERVICE.start_outage(payload["outage_id"], _tenant(payload)))
	except KeyError as exc:
		return _err(f"not_found: {exc}", 404)


def complete_outage(payload: dict[str, Any]) -> dict[str, Any]:
	"""PUT /energy-gen/api/v1/outages/<id>/complete — Complete an outage."""
	try:
		return _ok(_SERVICE.complete_outage(payload["outage_id"], _tenant(payload)))
	except KeyError as exc:
		return _err(f"not_found: {exc}", 404)


# ── KPIs ──────────────────────────────────────────────────────────────────────

def list_kpis(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-gen/api/v1/kpis — List generation KPIs."""
	return _ok(kpi_model(_SERVICE, _tenant(payload)))


def calculate_kpi(payload: dict[str, Any]) -> dict[str, Any]:
	"""POST /energy-gen/api/v1/kpis — Record a KPI calculation."""
	try:
		result = _SERVICE.calculate_kpi(
			kpi_id=payload["kpi_id"],
			tenant_id=_tenant(payload),
			plant_id=payload["plant_id"],
			kpi_type=payload["kpi_type"],
			period=payload["period"],
			period_start=payload["period_start"],
			period_end=payload["period_end"],
			value=float(payload["value"]),
			unit=payload["unit"],
		)
		return _ok(result)
	except (KeyError, TypeError) as exc:
		return _err(f"missing_field: {exc}")
	except ValueError as exc:
		return _err(str(exc))


# ── capacity plans ────────────────────────────────────────────────────────────

def list_capacity_plans(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-gen/api/v1/capacity — List capacity plans."""
	return _ok(capacity_planner_model(_SERVICE, _tenant(payload)))


def create_capacity_plan(payload: dict[str, Any]) -> dict[str, Any]:
	"""POST /energy-gen/api/v1/capacity — Create a capacity plan."""
	try:
		result = _SERVICE.create_capacity_plan(
			plan_id=payload["plan_id"],
			tenant_id=_tenant(payload),
			plan_name=payload["plan_name"],
			horizon_years=int(payload["horizon_years"]),
			base_year=int(payload["base_year"]),
			total_existing_mw=float(payload["total_existing_mw"]),
			total_planned_mw=float(payload["total_planned_mw"]),
			peak_demand_mw=float(payload["peak_demand_mw"]),
			reserve_margin_pct=float(payload["reserve_margin_pct"]),
			created_by=payload["created_by"],
		)
		return _ok(result)
	except (KeyError, TypeError) as exc:
		return _err(f"missing_field: {exc}")
	except ValueError as exc:
		return _err(str(exc))


# ── fuel stocks ───────────────────────────────────────────────────────────────

def list_fuel_stocks(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-gen/api/v1/fuel — List fuel stocks."""
	return _ok(fuel_management_model(_SERVICE, _tenant(payload)))


def update_fuel_stock(payload: dict[str, Any]) -> dict[str, Any]:
	"""POST /energy-gen/api/v1/fuel — Update fuel stock."""
	try:
		result = _SERVICE.update_fuel_stock(
			stock_id=payload["stock_id"],
			tenant_id=_tenant(payload),
			plant_id=payload["plant_id"],
			fuel_type=payload["fuel_type"],
			quantity=float(payload["quantity"]),
			unit=payload["unit"],
			days_of_supply=float(payload["days_of_supply"]),
			supplier_reference=payload.get("supplier_reference", ""),
		)
		return _ok(result)
	except (KeyError, TypeError) as exc:
		return _err(f"missing_field: {exc}")
	except ValueError as exc:
		return _err(str(exc))


# ── agents ────────────────────────────────────────────────────────────────────

def list_agents(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-gen/api/v1/agents — List agents."""
	return _ok(agent_workbench_model(_SERVICE, _tenant(payload)))


def register_agent(payload: dict[str, Any]) -> dict[str, Any]:
	"""POST /energy-gen/api/v1/agents — Register an agent."""
	try:
		result = _SERVICE.register_agent(
			agent_id=payload["agent_id"],
			tenant_id=_tenant(payload),
			name=payload["name"],
			runtime=payload["runtime"],
			role=payload["role"],
			scope=payload.get("scope", "generation management operations"),
		)
		return _ok(result)
	except (KeyError, ValueError) as exc:
		return _err(str(exc))


def service() -> GenerationManagementService:
	"""Return the process-local service singleton."""
	return _SERVICE
