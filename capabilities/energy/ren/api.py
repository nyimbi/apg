"""REST API for APG Renewable Energy."""

from __future__ import annotations

from typing import Any

try:
	from .service import RenewableEnergyService
	from .views import (
		agent_workbench_model, asset_detail_model, asset_registry_model,
		carbon_credit_model, curtailment_tracker_model, dashboard_model,
		fit_manager_model, forecasting_model, performance_model, rec_manager_model,
	)
except ImportError:
	from service import RenewableEnergyService  # type: ignore
	from views import (  # type: ignore
		agent_workbench_model, asset_detail_model, asset_registry_model,
		carbon_credit_model, curtailment_tracker_model, dashboard_model,
		fit_manager_model, forecasting_model, performance_model, rec_manager_model,
	)

_SERVICE = RenewableEnergyService()


def _ok(data: Any) -> dict[str, Any]:
	return {"status": "ok", "data": data}


def _err(reason: str, code: int = 400) -> dict[str, Any]:
	return {"status": "error", "code": code, "reason": reason}


def _tenant(payload: dict[str, Any]) -> str:
	return payload.get("tenant_id", "default")


def get_contract(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-ren/api/v1/contract"""
	return _ok(_SERVICE.describe(_tenant(payload)))


def get_dashboard(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-ren/api/v1/dashboard"""
	return _ok(dashboard_model(_SERVICE, _tenant(payload)))


# ── assets ────────────────────────────────────────────────────────────────────

def list_assets(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-ren/api/v1/assets"""
	return _ok(asset_registry_model(_SERVICE, _tenant(payload)))


def create_asset(payload: dict[str, Any]) -> dict[str, Any]:
	"""POST /energy-ren/api/v1/assets"""
	try:
		result = _SERVICE.register_asset(
			asset_id=payload["asset_id"],
			tenant_id=_tenant(payload),
			name=payload["name"],
			renewable_type=payload["renewable_type"],
			capacity_mw=float(payload["capacity_mw"]),
			owner_id=payload["owner_id"],
			commissioning_date=payload["commissioning_date"],
			location_reference=payload.get("location_reference", ""),
			grid_connection_point=payload.get("grid_connection_point", ""),
			policy_attached=payload.get("policy_attached", True),
		)
		return _ok(result)
	except (KeyError, TypeError) as exc:
		return _err(f"missing_field: {exc}")
	except ValueError as exc:
		return _err(str(exc))


def get_asset(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-ren/api/v1/assets/<id>"""
	try:
		return _ok(asset_detail_model(_SERVICE, _tenant(payload), payload["asset_id"]))
	except KeyError as exc:
		return _err(f"not_found: {exc}", 404)


def update_asset_status(payload: dict[str, Any]) -> dict[str, Any]:
	"""PUT /energy-ren/api/v1/assets/<id>/status"""
	try:
		return _ok(_SERVICE.update_asset_status(
			asset_id=payload["asset_id"],
			tenant_id=_tenant(payload),
			new_status=payload["status"],
		))
	except (KeyError, ValueError) as exc:
		return _err(str(exc))


# ── curtailment ───────────────────────────────────────────────────────────────

def list_curtailments(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-ren/api/v1/curtailment"""
	return _ok(curtailment_tracker_model(_SERVICE, _tenant(payload)))


def record_curtailment(payload: dict[str, Any]) -> dict[str, Any]:
	"""POST /energy-ren/api/v1/curtailment"""
	try:
		result = _SERVICE.record_curtailment(
			curtailment_id=payload["curtailment_id"],
			tenant_id=_tenant(payload),
			asset_id=payload["asset_id"],
			reason=payload["reason"],
			curtailed_mwh=float(payload["curtailed_mwh"]),
			start_time=payload["start_time"],
			end_time=payload["end_time"],
			revenue_loss=float(payload.get("revenue_loss", 0)),
			currency=payload.get("currency", "KES"),
			operator_reference=payload.get("operator_reference", ""),
			policy_attached=payload.get("policy_attached", True),
		)
		return _ok(result)
	except (KeyError, TypeError) as exc:
		return _err(f"missing_field: {exc}")
	except ValueError as exc:
		return _err(str(exc))


def approve_curtailment(payload: dict[str, Any]) -> dict[str, Any]:
	"""PUT /energy-ren/api/v1/curtailment/<id>/approve"""
	try:
		return _ok(_SERVICE.approve_curtailment(
			payload["curtailment_id"], _tenant(payload),
			approved_by=payload.get("approved_by", ""),
		))
	except (KeyError, ValueError) as exc:
		return _err(str(exc))


# ── RECs ──────────────────────────────────────────────────────────────────────

def list_recs(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-ren/api/v1/recs"""
	return _ok(rec_manager_model(_SERVICE, _tenant(payload)))


def issue_rec(payload: dict[str, Any]) -> dict[str, Any]:
	"""POST /energy-ren/api/v1/recs"""
	try:
		result = _SERVICE.issue_rec(
			rec_id=payload["rec_id"],
			tenant_id=_tenant(payload),
			asset_id=payload["asset_id"],
			rec_type=payload["rec_type"],
			quantity_mwh=float(payload["quantity_mwh"]),
			vintage_year=int(payload["vintage_year"]),
			registry=payload["registry"],
			serial_number=payload.get("serial_number", ""),
			expires_at=payload.get("expires_at", ""),
			policy_attached=payload.get("policy_attached", True),
		)
		return _ok(result)
	except (KeyError, TypeError) as exc:
		return _err(f"missing_field: {exc}")
	except ValueError as exc:
		return _err(str(exc))


def transfer_rec(payload: dict[str, Any]) -> dict[str, Any]:
	"""PUT /energy-ren/api/v1/recs/<id>/transfer"""
	try:
		return _ok(_SERVICE.transfer_rec(
			payload["rec_id"], _tenant(payload),
			transferred_to=payload["transferred_to"],
		))
	except (KeyError, ValueError) as exc:
		return _err(str(exc))


def retire_rec(payload: dict[str, Any]) -> dict[str, Any]:
	"""PUT /energy-ren/api/v1/recs/<id>/retire"""
	try:
		return _ok(_SERVICE.retire_rec(payload["rec_id"], _tenant(payload)))
	except (KeyError, ValueError) as exc:
		return _err(str(exc))


# ── carbon credits ────────────────────────────────────────────────────────────

def list_carbon_credits(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-ren/api/v1/carbon-credits"""
	return _ok(carbon_credit_model(_SERVICE, _tenant(payload)))


def issue_carbon_credit(payload: dict[str, Any]) -> dict[str, Any]:
	"""POST /energy-ren/api/v1/carbon-credits"""
	try:
		result = _SERVICE.issue_carbon_credit(
			credit_id=payload["credit_id"],
			tenant_id=_tenant(payload),
			asset_id=payload["asset_id"],
			credit_type=payload["credit_type"],
			quantity_tco2e=float(payload["quantity_tco2e"]),
			vintage_year=int(payload["vintage_year"]),
			standard=payload["standard"],
			verification_reference=payload["verification_reference"],
			serial_number=payload.get("serial_number", ""),
			project_id=payload.get("project_id", ""),
			policy_attached=payload.get("policy_attached", True),
		)
		return _ok(result)
	except (KeyError, TypeError) as exc:
		return _err(f"missing_field: {exc}")
	except ValueError as exc:
		return _err(str(exc))


def retire_carbon_credit(payload: dict[str, Any]) -> dict[str, Any]:
	"""PUT /energy-ren/api/v1/carbon-credits/<id>/retire"""
	try:
		return _ok(_SERVICE.retire_carbon_credit(payload["credit_id"], _tenant(payload)))
	except (KeyError, ValueError) as exc:
		return _err(str(exc))


# ── FITs ──────────────────────────────────────────────────────────────────────

def list_fits(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-ren/api/v1/feed-in-tariffs"""
	return _ok(fit_manager_model(_SERVICE, _tenant(payload)))


def create_fit(payload: dict[str, Any]) -> dict[str, Any]:
	"""POST /energy-ren/api/v1/feed-in-tariffs"""
	try:
		result = _SERVICE.create_fit(
			fit_id=payload["fit_id"],
			tenant_id=_tenant(payload),
			asset_id=payload["asset_id"],
			fit_type=payload["fit_type"],
			rate_per_kwh=float(payload["rate_per_kwh"]),
			currency=payload.get("currency", "KES"),
			effective_date=payload["effective_date"],
			approved_by=payload["approved_by"],
			end_date=payload.get("end_date", ""),
			policy_attached=payload.get("policy_attached", True),
		)
		return _ok(result)
	except (KeyError, TypeError) as exc:
		return _err(f"missing_field: {exc}")
	except ValueError as exc:
		return _err(str(exc))


# ── forecasting ───────────────────────────────────────────────────────────────

def list_forecasts(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-ren/api/v1/forecasting"""
	return _ok(forecasting_model(_SERVICE, _tenant(payload)))


def publish_forecast(payload: dict[str, Any]) -> dict[str, Any]:
	"""POST /energy-ren/api/v1/forecasting"""
	try:
		result = _SERVICE.publish_forecast(
			forecast_id=payload["forecast_id"],
			tenant_id=_tenant(payload),
			asset_id=payload["asset_id"],
			forecast_type=payload["forecast_type"],
			horizon=payload["horizon"],
			forecast_start=payload["forecast_start"],
			forecast_end=payload["forecast_end"],
			values=payload.get("values", []),
			model_version=payload.get("model_version", "1.0"),
			rmse=float(payload.get("rmse", 0)),
			mae=float(payload.get("mae", 0)),
		)
		return _ok(result)
	except (KeyError, TypeError) as exc:
		return _err(f"missing_field: {exc}")
	except ValueError as exc:
		return _err(str(exc))


# ── performance ───────────────────────────────────────────────────────────────

def list_performance(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-ren/api/v1/performance"""
	return _ok(performance_model(_SERVICE, _tenant(payload)))


def record_metric(payload: dict[str, Any]) -> dict[str, Any]:
	"""POST /energy-ren/api/v1/performance"""
	try:
		result = _SERVICE.record_performance_metric(
			metric_id=payload["metric_id"],
			tenant_id=_tenant(payload),
			asset_id=payload["asset_id"],
			metric_type=payload["metric_type"],
			period_start=payload["period_start"],
			period_end=payload["period_end"],
			value=float(payload["value"]),
			unit=payload["unit"],
			benchmark_value=float(payload.get("benchmark_value", 0)),
		)
		return _ok(result)
	except (KeyError, TypeError) as exc:
		return _err(f"missing_field: {exc}")
	except ValueError as exc:
		return _err(str(exc))


# ── agents ────────────────────────────────────────────────────────────────────

def list_agents(payload: dict[str, Any]) -> dict[str, Any]:
	"""GET /energy-ren/api/v1/agents"""
	return _ok(agent_workbench_model(_SERVICE, _tenant(payload)))


def register_agent(payload: dict[str, Any]) -> dict[str, Any]:
	"""POST /energy-ren/api/v1/agents"""
	try:
		result = _SERVICE.register_agent(
			agent_id=payload["agent_id"],
			tenant_id=_tenant(payload),
			name=payload["name"],
			runtime=payload["runtime"],
			role=payload["role"],
			scope=payload.get("scope", "renewable energy operations"),
		)
		return _ok(result)
	except (KeyError, ValueError) as exc:
		return _err(str(exc))


def service() -> RenewableEnergyService:
	return _SERVICE
