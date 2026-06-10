"""
APG Budgeting & Forecasting — Flask REST API

Complete REST API with filtering, pagination, all lifecycle transitions,
and domain-specific reports.  Uses plain Flask Blueprint (no FAB dependency).

URL prefix: /api/v1/bfc

© 2025 Datacraft. Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from decimal import Decimal
from functools import wraps
from typing import Any

from flask import Blueprint, Response, jsonify, request

from .models import (
	BFBudgetApprovalCreate,
	BFBudgetCreate,
	BFBudgetLineCreate,
	BFBudgetStatus,
	BFBudgetTemplateCreate,
	BFBudgetType,
	BFBudgetUpdate,
	BFDistributionMethod,
	BFDriverAssumptionCreate,
	BFDriverType,
	BFDriverBasedAssumption,
	BFForecastCreate,
	BFForecastLineCreate,
	BFForecastStatus,
	BFForecastType,
	BFLineType,
	BFScenarioCreate,
	BFScenarioType,
)
from .service import BFCService
from .domain.rules import RuleViolation
from .context import get_current_user_id, get_tenant_id_from_request

_log = logging.getLogger(__name__)

bfc_bp = Blueprint("bfc", __name__, url_prefix="/api/v1/bfc")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _svc() -> BFCService:
	"""Resolve tenant + actor from request headers / JWT claims."""
	tenant_id = request.headers.get("X-Tenant-Id", "default")
	actor_id = request.headers.get("X-Actor-Id", "anonymous")
	return BFCService(tenant_id=tenant_id, actor_id=actor_id)


def _ok(data: Any, status: int = 200) -> Response:
	return jsonify({"ok": True, "data": _serialise(data)}), status


def _err(message: str, status: int = 400, code: str | None = None) -> Response:
	body: dict[str, Any] = {"ok": False, "error": message}
	if code:
		body["code"] = code
	return jsonify(body), status


def _serialise(obj: Any) -> Any:
	"""Recursively make Pydantic models / Decimals / dates JSON-safe."""
	if hasattr(obj, "model_dump"):
		return _serialise(obj.model_dump())
	if isinstance(obj, list):
		return [_serialise(i) for i in obj]
	if isinstance(obj, dict):
		return {k: _serialise(v) for k, v in obj.items()}
	if isinstance(obj, Decimal):
		return str(obj)
	if isinstance(obj, (date, datetime)):
		return obj.isoformat()
	return obj


def _parse_date(s: str | None) -> date | None:
	if not s:
		return None
	return date.fromisoformat(s)


def _run(coro: Any) -> Any:
	"""Run an async coroutine from a sync Flask handler."""
	import asyncio
	try:
		loop = asyncio.get_event_loop()
		if loop.is_running():
			import concurrent.futures
			with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
				future = pool.submit(asyncio.run, coro)
				return future.result()
		return asyncio.run(coro)
	except RuntimeError:
		return asyncio.run(coro)


def get_tenant_id(payload: dict | None = None) -> str:
	"""Resolve tenant id from request context."""
	return get_tenant_id_from_request(payload)


try:
	from flask_jwt_extended import get_jwt_identity as _get_jwt_identity  # optional dep
except ImportError:
	def _get_jwt_identity():  # type: ignore[misc]
		return None


def get_user_id(payload: dict | None = None) -> str:
	"""Resolve user id: JWT identity > request JSON body > request context > env fallback."""
	try:
		identity = get_jwt_identity()
		if identity:
			return str(identity)
	except Exception as _exc:
		_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
	# Check request JSON body when no explicit payload provided
	if payload is None:
		try:
			json_body = request.get_json(force=True, silent=True) or {}
			uid = json_body.get("user_id")
			if uid:
				return str(uid)
		except Exception as _exc:
			_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
	return get_current_user_id(payload)


def handle_api_error(e: Exception) -> tuple:
	"""Convert known exceptions to JSON error responses."""
	if isinstance(e, RuleViolation):
		return _err(str(e), 422, e.rule_name)
	if isinstance(e, KeyError):
		return _err(str(e), 404, "not_found")
	if isinstance(e, ValueError):
		return _err(str(e), 400, "validation_error")
	if isinstance(e, PermissionError):
		return _err(str(e), 403, "forbidden")
	_log.exception("Unhandled BFC API error")
	return _err("Internal server error", 500, "internal_error")


def handle_errors(fn):
	@wraps(fn)
	def wrapper(*args, **kwargs):
		try:
			return fn(*args, **kwargs)
		except RuleViolation as e:
			return _err(str(e), 422, e.rule_name)
		except KeyError as e:
			return _err(str(e), 404, "not_found")
		except ValueError as e:
			return _err(str(e), 400, "validation_error")
		except PermissionError as e:
			return _err(str(e), 403, "forbidden")
		except Exception as e:
			_log.exception("Unhandled BFC API error")
			return _err("Internal server error", 500, "internal_error")
	return wrapper


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@bfc_bp.get("/health")
def health():
	return _ok({"capability": "bfc_budgeting_forecasting", "status": "healthy"})


# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------

@bfc_bp.get("/budgets")
@handle_errors
def list_budgets():
	"""GET /budgets?status=draft&fiscal_year=2026&budget_type=annual&offset=0&limit=50"""
	svc = _svc()
	status_str = request.args.get("status")
	fy = request.args.get("fiscal_year")
	bt = request.args.get("budget_type")
	offset = int(request.args.get("offset", 0))
	limit = min(int(request.args.get("limit", 50)), 200)

	items = _run(svc.list_budgets(
		status=BFBudgetStatus(status_str) if status_str else None,
		fiscal_year=int(fy) if fy else None,
		budget_type=BFBudgetType(bt) if bt else None,
		offset=offset,
		limit=limit,
	))
	return _ok({"items": items, "count": len(items), "offset": offset, "limit": limit})


@bfc_bp.post("/budgets")
@handle_errors
def create_budget():
	"""POST /budgets — create a new budget cycle."""
	data = request.get_json(force=True) or {}
	payload = BFBudgetCreate(**data)
	budget = _run(_svc().create_budget_cycle(payload))
	return _ok(budget, 201)


@bfc_bp.get("/budgets/<budget_id>")
@handle_errors
def get_budget(budget_id: str):
	budget = _run(_svc().get_budget(budget_id))
	return _ok(budget)


@bfc_bp.put("/budgets/<budget_id>")
@handle_errors
def update_budget(budget_id: str):
	data = request.get_json(force=True) or {}
	payload = BFBudgetUpdate(**data)
	budget = _run(_svc().update_budget(budget_id, payload))
	return _ok(budget)


@bfc_bp.delete("/budgets/<budget_id>")
@handle_errors
def cancel_budget(budget_id: str):
	data = request.get_json(force=True) or {}
	reason = data.get("reason", "Cancelled via API")
	budget = _run(_svc().cancel_budget(budget_id, reason))
	return _ok(budget)


@bfc_bp.post("/budgets/<budget_id>/submit")
@handle_errors
def submit_budget(budget_id: str):
	budget = _run(_svc().submit_budget(budget_id))
	return _ok(budget)


@bfc_bp.post("/budgets/<budget_id>/approve")
@handle_errors
def approve_budget(budget_id: str):
	data = request.get_json(force=True) or {}
	approval_id = data["approval_id"]
	comments = data.get("comments")
	budget = _run(_svc().approve_budget(budget_id, approval_id, comments))
	return _ok(budget)


@bfc_bp.post("/budgets/<budget_id>/reject")
@handle_errors
def reject_budget(budget_id: str):
	data = request.get_json(force=True) or {}
	approval_id = data["approval_id"]
	reason = data.get("reason", "Rejected")
	budget = _run(_svc().reject_budget(budget_id, approval_id, reason))
	return _ok(budget)


@bfc_bp.post("/budgets/<budget_id>/lock")
@handle_errors
def lock_budget(budget_id: str):
	budget = _run(_svc().lock_budget(budget_id))
	return _ok(budget)


@bfc_bp.post("/budgets/<budget_id>/close")
@handle_errors
def close_budget(budget_id: str):
	budget = _run(_svc().close_budget(budget_id))
	return _ok(budget)


@bfc_bp.post("/budgets/<budget_id>/distribute")
@handle_errors
def distribute_budget(budget_id: str):
	"""POST /budgets/<id>/distribute  body: {method, department_weights?, seasonal_weights?, line_justifications?}"""
	data = request.get_json(force=True) or {}
	method = BFDistributionMethod(data.get("method", "equal"))
	result = _run(_svc().distribute_budget(
		budget_id,
		method=method,
		department_weights=data.get("department_weights"),
		seasonal_weights=data.get("seasonal_weights"),
		line_justifications=data.get("line_justifications"),
	))
	return _ok(result)


@bfc_bp.post("/budgets/<budget_id>/what-if")
@handle_errors
def what_if_simulation(budget_id: str):
	data = request.get_json(force=True) or {}
	adjustments = data.get("adjustments", {})
	result = _run(_svc().what_if_simulation(budget_id, adjustments))
	return _ok(result)


@bfc_bp.post("/budgets/<budget_id>/consolidate")
@handle_errors
def consolidate_budget(budget_id: str):
	"""Consolidate this budget with additional budget_ids."""
	data = request.get_json(force=True) or {}
	budget_ids = [budget_id] + data.get("additional_budget_ids", [])
	currency = data.get("currency_code", "USD")
	result = _run(_svc().budget_consolidation(budget_ids, currency))
	return _ok(result)


# ---------------------------------------------------------------------------
# Budget lines
# ---------------------------------------------------------------------------

@bfc_bp.get("/budgets/<budget_id>/lines")
@handle_errors
def list_budget_lines(budget_id: str):
	lines = _run(_svc().get_budget_lines(budget_id))
	return _ok({"items": lines, "count": len(lines)})


@bfc_bp.post("/budgets/<budget_id>/lines")
@handle_errors
def add_budget_line(budget_id: str):
	data = request.get_json(force=True) or {}
	data["budget_id"] = budget_id
	payload = BFBudgetLineCreate(**data)
	line = _run(_svc().add_budget_line(payload))
	return _ok(line, 201)


@bfc_bp.delete("/budgets/<budget_id>/lines/<line_id>")
@handle_errors
def delete_budget_line(budget_id: str, line_id: str):
	_run(_svc().delete_budget_line(line_id))
	return _ok({"deleted": line_id})


# ---------------------------------------------------------------------------
# Approvals
# ---------------------------------------------------------------------------

@bfc_bp.get("/approvals")
@handle_errors
def list_approvals():
	budget_id = request.args.get("budget_id")
	approvals = _run(_svc().get_pending_approvals(budget_id))
	return _ok({"items": approvals, "count": len(approvals)})


@bfc_bp.post("/approvals")
@handle_errors
def create_approval():
	data = request.get_json(force=True) or {}
	payload = BFBudgetApprovalCreate(**data)
	approval = _run(_svc().create_approval(payload))
	return _ok(approval, 201)


@bfc_bp.post("/approvals/<approval_id>/delegate")
@handle_errors
def delegate_approval(approval_id: str):
	data = request.get_json(force=True) or {}
	delegate_to = data["delegate_to"]
	approval = _run(_svc().delegate_approval(approval_id, delegate_to))
	return _ok(approval)


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

@bfc_bp.get("/templates")
@handle_errors
def list_templates():
	templates = _run(_svc().list_templates())
	return _ok({"items": templates, "count": len(templates)})


@bfc_bp.post("/templates")
@handle_errors
def create_template():
	data = request.get_json(force=True) or {}
	payload = BFBudgetTemplateCreate(**data)
	template = _run(_svc().create_template(payload))
	return _ok(template, 201)


@bfc_bp.post("/templates/<template_id>/instantiate")
@handle_errors
def instantiate_template(template_id: str):
	data = request.get_json(force=True) or {}
	fiscal_year = int(data["fiscal_year"])
	period_start = date.fromisoformat(data["period_start"])
	period_end = date.fromisoformat(data["period_end"])
	budget = _run(_svc().instantiate_template(template_id, fiscal_year, period_start, period_end))
	return _ok(budget, 201)


# ---------------------------------------------------------------------------
# Forecasts
# ---------------------------------------------------------------------------

@bfc_bp.get("/forecasts")
@handle_errors
def list_forecasts():
	status_str = request.args.get("status")
	ft_str = request.args.get("forecast_type")
	limit = min(int(request.args.get("limit", 50)), 200)
	items = _run(_svc().list_forecasts(
		status=BFForecastStatus(status_str) if status_str else None,
		forecast_type=BFForecastType(ft_str) if ft_str else None,
		limit=limit,
	))
	return _ok({"items": items, "count": len(items)})


@bfc_bp.post("/forecasts")
@handle_errors
def create_forecast():
	data = request.get_json(force=True) or {}
	payload = BFForecastCreate(**data)
	forecast = _run(_svc().create_forecast(payload))
	return _ok(forecast, 201)


@bfc_bp.post("/forecasts/<forecast_id>/lines")
@handle_errors
def add_forecast_line(forecast_id: str):
	data = request.get_json(force=True) or {}
	data["forecast_id"] = forecast_id
	payload = BFForecastLineCreate(**data)
	line = _run(_svc().add_forecast_line(payload))
	return _ok(line, 201)


@bfc_bp.post("/forecasts/<forecast_id>/rolling")
@handle_errors
def rolling_forecast(forecast_id: str):
	data = request.get_json(force=True) or {}
	periods = int(data.get("periods", 3))
	alpha = float(data.get("alpha", 0.3))
	result = _run(_svc().rolling_forecast(forecast_id, periods, alpha))
	return _ok(result)


@bfc_bp.post("/forecasts/<forecast_id>/reforecast")
@handle_errors
def reforecast(forecast_id: str):
	data = request.get_json(force=True) or {}
	period = data["period"]
	actuals = [float(v) for v in data["actuals"]]
	forecast = _run(_svc().reforecast(forecast_id, period, actuals))
	return _ok(forecast)


@bfc_bp.post("/forecasts/<forecast_id>/driver-based")
@handle_errors
def driver_based_forecast_endpoint(forecast_id: str):
	data = request.get_json(force=True) or {}
	driver_changes = {k: float(v) for k, v in data.get("driver_changes", {}).items()}
	lines = _run(_svc().driver_based_forecast(forecast_id, driver_changes))
	return _ok({"items": lines, "count": len(lines)})


@bfc_bp.post("/forecasts/<forecast_id>/ai-model")
@handle_errors
def ai_forecast_model(forecast_id: str):
	data = request.get_json(force=True) or {}
	result = _run(_svc().ai_forecast_model(forecast_id, data.get("model_params")))
	return _ok(result)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

@bfc_bp.get("/scenarios")
@handle_errors
def list_scenarios():
	active_only = request.args.get("active_only", "false").lower() == "true"
	items = _run(_svc().list_scenarios(active_only))
	return _ok({"items": items, "count": len(items)})


@bfc_bp.post("/scenarios")
@handle_errors
def create_scenario():
	data = request.get_json(force=True) or {}
	payload = BFScenarioCreate(**data)
	scenario = _run(_svc().create_scenario(payload))
	return _ok(scenario, 201)


@bfc_bp.post("/scenarios/analyze")
@handle_errors
def scenario_analysis():
	"""POST /scenarios/analyze  body: {budget_id, scenario_ids: [...]}"""
	data = request.get_json(force=True) or {}
	budget_id = data["budget_id"]
	scenario_ids = data["scenario_ids"]
	result = _run(_svc().scenario_analysis(budget_id, scenario_ids))
	return _ok(result)


# ---------------------------------------------------------------------------
# Driver assumptions
# ---------------------------------------------------------------------------

@bfc_bp.get("/drivers")
@handle_errors
def list_drivers():
	dt_str = request.args.get("driver_type")
	items = _run(_svc().list_driver_assumptions(
		driver_type=BFDriverType(dt_str) if dt_str else None
	))
	return _ok({"items": items, "count": len(items)})


@bfc_bp.post("/drivers")
@handle_errors
def create_driver():
	data = request.get_json(force=True) or {}
	payload = BFDriverAssumptionCreate(**data)
	driver = _run(_svc().create_driver_assumption(payload))
	return _ok(driver, 201)


@bfc_bp.post("/drivers/<driver_id>/sensitivity")
@handle_errors
def sensitivity_analysis(driver_id: str):
	data = request.get_json(force=True) or {}
	steps = data.get("steps")
	result = _run(_svc().sensitivity_analysis(driver_id, steps))
	return _ok(result)


# ---------------------------------------------------------------------------
# Variance reports
# ---------------------------------------------------------------------------

@bfc_bp.get("/variance-reports")
@handle_errors
def list_variance_reports():
	budget_id = request.args.get("budget_id")
	reports = _run(_svc().list_variance_reports(budget_id))
	return _ok({"items": reports, "count": len(reports)})


@bfc_bp.post("/variance-reports")
@handle_errors
def create_variance_report():
	"""POST /variance-reports  body: {budget_id, period_start, period_end, actuals_by_account}"""
	data = request.get_json(force=True) or {}
	budget_id = data["budget_id"]
	period_start = date.fromisoformat(data["period_start"])
	period_end = date.fromisoformat(data["period_end"])
	actuals = {k: Decimal(str(v)) for k, v in data.get("actuals_by_account", {}).items()}
	report = _run(_svc().variance_analysis(budget_id, period_start, period_end, actuals))
	return _ok(report, 201)


# ---------------------------------------------------------------------------
# Consolidation
# ---------------------------------------------------------------------------

@bfc_bp.post("/consolidation")
@handle_errors
def consolidate():
	"""POST /consolidation  body: {budget_ids: [...], currency_code: 'USD'}"""
	data = request.get_json(force=True) or {}
	budget_ids = data["budget_ids"]
	currency = data.get("currency_code", "USD")
	result = _run(_svc().budget_consolidation(budget_ids, currency))
	return _ok(result)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

@bfc_bp.get("/reports/dashboard")
@handle_errors
def report_dashboard():
	kpis = _run(_svc().dashboard_kpis())
	return _ok(kpis)


@bfc_bp.get("/reports/summary")
@handle_errors
def report_summary():
	period_start_str = request.args.get("period_start", date.today().replace(month=1, day=1).isoformat())
	period_end_str = request.args.get("period_end", date.today().isoformat())
	summary = _run(_svc().budget_summary(
		date.fromisoformat(period_start_str),
		date.fromisoformat(period_end_str),
	))
	return _ok(summary)


@bfc_bp.get("/reports/audit")
@handle_errors
def report_audit():
	entity_id = request.args.get("entity_id")
	limit = int(request.args.get("limit", 100))
	events = _run(_svc().audit_trail(entity_id))
	return _ok({"items": events[:limit], "count": len(events)})
