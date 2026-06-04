"""REST API for APG Grid Operations."""

from __future__ import annotations

from typing import Any

try:
	from .service import GridOperationsService
	from .views import (
		agent_workbench_model, alarm_console_model, contingency_detail_model,
		contingency_model, dashboard_model, ems_console_model,
		frequency_control_model, market_settlement_model, settlement_detail_model,
		state_estimation_model, voltage_control_model,
	)
except ImportError:
	from service import GridOperationsService  # type: ignore
	from views import (  # type: ignore
		agent_workbench_model, alarm_console_model, contingency_detail_model,
		contingency_model, dashboard_model, ems_console_model,
		frequency_control_model, market_settlement_model, settlement_detail_model,
		state_estimation_model, voltage_control_model,
	)

_SERVICE = GridOperationsService()


def _ok(data: Any) -> dict[str, Any]:
	return {"status": "ok", "data": data}


def _err(reason: str, code: int = 400) -> dict[str, Any]:
	return {"status": "error", "code": code, "reason": reason}


def _tenant(payload: dict[str, Any]) -> str:
	return payload.get("tenant_id", "default")


def get_contract(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-grd/api/v1/contract"""
	return _ok(_SERVICE.describe(_tenant(payload)))


def get_dashboard(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-grd/api/v1/dashboard"""
	return _ok(dashboard_model(_SERVICE, _tenant(payload)))


# ── state estimation ──────────────────────────────────────────────────────────

def list_se_runs(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-grd/api/v1/state-estimation"""
	return _ok(state_estimation_model(_SERVICE, _tenant(payload)))


def run_state_estimation(payload: dict[str, Any]) -> dict[str, Any]:
	"""POST /energy-grd/api/v1/state-estimation"""
	try:
		result = _SERVICE.run_state_estimation(
			run_id=payload["run_id"],
			tenant_id=_tenant(payload),
			estimator_type=payload["estimator_type"],
			grid_area=payload.get("grid_area", "transmission"),
			network_model_ref=payload["network_model_ref"],
			measurement_snapshot_ref=payload["measurement_snapshot_ref"],
			iterations=int(payload.get("iterations", 0)),
			converged=bool(payload.get("converged", False)),
			residual=float(payload.get("residual", 0.0)),
			voltage_violations=int(payload.get("voltage_violations", 0)),
		)
		return _ok(result)
	except (KeyError, TypeError) as exc:
		return _err(f"missing_field: {exc}")
	except ValueError as exc:
		return _err(str(exc))


# ── contingency ───────────────────────────────────────────────────────────────

def list_contingencies(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-grd/api/v1/contingency"""
	return _ok(contingency_model(_SERVICE, _tenant(payload)))


def run_contingency(payload: dict[str, Any]) -> dict[str, Any]:
	"""POST /energy-grd/api/v1/contingency"""
	try:
		result = _SERVICE.run_contingency(
			case_id=payload["case_id"],
			tenant_id=_tenant(payload),
			contingency_type=payload["contingency_type"],
			contingency_name=payload["contingency_name"],
			base_case_ref=payload["base_case_ref"],
			base_case_converged=bool(payload.get("base_case_converged", True)),
			violations=payload.get("violations", []),
			max_overload_pct=float(payload.get("max_overload_pct", 0)),
			min_voltage_pu=float(payload.get("min_voltage_pu", 1.0)),
			max_voltage_pu=float(payload.get("max_voltage_pu", 1.0)),
			remedial_actions=payload.get("remedial_actions"),
		)
		return _ok(result)
	except (KeyError, TypeError) as exc:
		return _err(f"missing_field: {exc}")
	except ValueError as exc:
		return _err(str(exc))


def get_contingency(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-grd/api/v1/contingency/<id>"""
	return _ok(contingency_detail_model(_SERVICE, _tenant(payload), payload["case_id"]))


# ── voltage control ───────────────────────────────────────────────────────────

def list_voltage_actions(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-grd/api/v1/voltage-control"""
	return _ok(voltage_control_model(_SERVICE, _tenant(payload)))


def apply_voltage_control(payload: dict[str, Any]) -> dict[str, Any]:
	"""POST /energy-grd/api/v1/voltage-control"""
	try:
		result = _SERVICE.apply_voltage_control(
			action_id=payload["action_id"],
			tenant_id=_tenant(payload),
			control_method=payload["control_method"],
			element_id=payload["element_id"],
			target_voltage_pu=float(payload["target_voltage_pu"]),
			achieved_voltage_pu=float(payload.get("achieved_voltage_pu", 0)),
			approved_by=payload["approved_by"],
		)
		return _ok(result)
	except (KeyError, TypeError) as exc:
		return _err(f"missing_field: {exc}")
	except ValueError as exc:
		return _err(str(exc))


# ── frequency control ─────────────────────────────────────────────────────────

def list_frequency_actions(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-grd/api/v1/frequency-control"""
	return _ok(frequency_control_model(_SERVICE, _tenant(payload)))


def apply_frequency_control(payload: dict[str, Any]) -> dict[str, Any]:
	"""POST /energy-grd/api/v1/frequency-control"""
	try:
		result = _SERVICE.apply_frequency_control(
			action_id=payload["action_id"],
			tenant_id=_tenant(payload),
			control_method=payload["control_method"],
			trigger_frequency_hz=float(payload["trigger_frequency_hz"]),
			response_mw=float(payload["response_mw"]),
			response_mvar=float(payload.get("response_mvar", 0)),
		)
		return _ok(result)
	except (KeyError, TypeError) as exc:
		return _err(f"missing_field: {exc}")
	except ValueError as exc:
		return _err(str(exc))


def configure_ufls(payload: dict[str, Any]) -> dict[str, Any]:
	"""PUT /energy-grd/api/v1/frequency-control/ufls"""
	try:
		return _ok(_SERVICE.configure_ufls(
			_tenant(payload),
			threshold_hz=float(payload["threshold_hz"]),
		))
	except (KeyError, ValueError) as exc:
		return _err(str(exc))


# ── market settlement ─────────────────────────────────────────────────────────

def list_settlements(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-grd/api/v1/market-settlement"""
	return _ok(market_settlement_model(_SERVICE, _tenant(payload)))


def settle_interval(payload: dict[str, Any]) -> dict[str, Any]:
	"""POST /energy-grd/api/v1/market-settlement"""
	try:
		result = _SERVICE.settle_market_interval(
			interval_id=payload["interval_id"],
			tenant_id=_tenant(payload),
			market_product=payload["market_product"],
			interval_start=payload["interval_start"],
			interval_end=payload["interval_end"],
			metered_mwh=float(payload["metered_mwh"]),
			scheduled_mwh=float(payload["scheduled_mwh"]),
			price_per_mwh=float(payload["price_per_mwh"]),
			currency=payload.get("currency", "KES"),
			participant_id=payload.get("participant_id", ""),
			bid_offer_ref=payload.get("bid_offer_ref", ""),
			policy_attached=payload.get("policy_attached", True),
		)
		return _ok(result)
	except (KeyError, TypeError) as exc:
		return _err(f"missing_field: {exc}")
	except ValueError as exc:
		return _err(str(exc))


def get_settlement(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-grd/api/v1/market-settlement/<id>"""
	return _ok(settlement_detail_model(_SERVICE, _tenant(payload), payload["interval_id"]))


def finalize_settlement(payload: dict[str, Any]) -> dict[str, Any]:
	"""PUT /energy-grd/api/v1/market-settlement/<id>/finalize"""
	try:
		return _ok(_SERVICE.finalize_settlement(payload["interval_id"], _tenant(payload)))
	except (KeyError, ValueError) as exc:
		return _err(str(exc))


# ── alarms ────────────────────────────────────────────────────────────────────

def list_alarms(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-grd/api/v1/alarms"""
	return _ok(alarm_console_model(_SERVICE, _tenant(payload)))


def raise_alarm(payload: dict[str, Any]) -> dict[str, Any]:
	"""POST /energy-grd/api/v1/alarms"""
	try:
		result = _SERVICE.raise_alarm(
			alarm_id=payload["alarm_id"],
			tenant_id=_tenant(payload),
			alarm_category=payload["alarm_category"],
			severity=payload["severity"],
			element_id=payload["element_id"],
			description=payload.get("description", ""),
			policy_attached=payload.get("policy_attached", True),
		)
		return _ok(result)
	except (KeyError, TypeError) as exc:
		return _err(f"missing_field: {exc}")
	except ValueError as exc:
		return _err(str(exc))


def acknowledge_alarm(payload: dict[str, Any]) -> dict[str, Any]:
	"""PUT /energy-grd/api/v1/alarms/<id>/acknowledge"""
	try:
		return _ok(_SERVICE.acknowledge_alarm(
			payload["alarm_id"], _tenant(payload),
			acknowledged_by=payload.get("acknowledged_by", ""),
		))
	except (KeyError, ValueError) as exc:
		return _err(str(exc))


def clear_alarm(payload: dict[str, Any]) -> dict[str, Any]:
	"""PUT /energy-grd/api/v1/alarms/<id>/clear"""
	try:
		return _ok(_SERVICE.clear_alarm(payload["alarm_id"], _tenant(payload)))
	except (KeyError, ValueError) as exc:
		return _err(str(exc))


# ── EMS ───────────────────────────────────────────────────────────────────────

def list_ems(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-grd/api/v1/ems"""
	return _ok(ems_console_model(_SERVICE, _tenant(payload)))


def execute_ems(payload: dict[str, Any]) -> dict[str, Any]:
	"""POST /energy-grd/api/v1/ems"""
	try:
		result = _SERVICE.execute_ems_function(
			exec_id=payload["exec_id"],
			tenant_id=_tenant(payload),
			ems_function=payload["ems_function"],
			mode=payload.get("mode", "real_time"),
			triggered_by=payload.get("triggered_by", "operator"),
			result_summary=payload.get("result_summary", {}),
		)
		return _ok(result)
	except (KeyError, TypeError) as exc:
		return _err(f"missing_field: {exc}")
	except ValueError as exc:
		return _err(str(exc))


# ── agents ────────────────────────────────────────────────────────────────────

def list_agents(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-grd/api/v1/agents"""
	return _ok(agent_workbench_model(_SERVICE, _tenant(payload)))


def register_agent(payload: dict[str, Any]) -> dict[str, Any]:
	"""POST /energy-grd/api/v1/agents"""
	try:
		result = _SERVICE.register_agent(
			agent_id=payload["agent_id"],
			tenant_id=_tenant(payload),
			name=payload["name"],
			runtime=payload["runtime"],
			role=payload["role"],
			scope=payload.get("scope", "grid operations"),
		)
		return _ok(result)
	except (KeyError, ValueError) as exc:
		return _err(str(exc))


def service() -> GridOperationsService:
	return _SERVICE
