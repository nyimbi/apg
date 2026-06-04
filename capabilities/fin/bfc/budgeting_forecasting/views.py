"""
APG Budgeting & Forecasting — Flask Blueprint UI Views

Server-side view models and route handlers for the BFC UI.
All views return JSON-compatible dicts for API-driven frontends
or can be adapted to Jinja2 templates.

URL prefix: /bfc/ui

© 2025 Datacraft. Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from flask import Blueprint, Response, jsonify, request

from .service import BFCService
from .models import (
	BFBudgetStatus,
	BFBudgetType,
	BFForecastStatus,
	BFForecastType,
	BFDriverType,
	BFScenarioType,
)
from .context import get_current_user_id, get_tenant_id_from_request
from .domain.rules import RuleViolation

_log = logging.getLogger(__name__)

bfc_ui_bp = Blueprint("bfc_ui", __name__, url_prefix="/bfc/ui")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _svc() -> BFCService:
	tenant_id = get_tenant_id_from_request()
	actor_id = get_current_user_id()
	return BFCService(tenant_id=tenant_id, actor_id=actor_id)


def _run(coro: Any) -> Any:
	"""Run async coroutine from sync Flask handler."""
	try:
		loop = asyncio.get_event_loop()
		if loop.is_running():
			import concurrent.futures
			with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
				return pool.submit(asyncio.run, coro).result()
		return asyncio.run(coro)
	except RuntimeError:
		return asyncio.run(coro)


def _serialise(obj: Any) -> Any:
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


def _ok(data: Any, status: int = 200) -> Response:
	return jsonify({"ok": True, "data": _serialise(data)}), status


def _err(msg: str, status: int = 400, code: str | None = None) -> Response:
	body: dict[str, Any] = {"ok": False, "error": msg}
	if code:
		body["code"] = code
	return jsonify(body), status


def _handle(fn):
	"""Decorator: convert exceptions to JSON error responses."""
	from functools import wraps
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
		except Exception:
			_log.exception("BFC UI view error")
			return _err("Internal server error", 500, "internal_error")
	return wrapper


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@bfc_ui_bp.get("/dashboard")
@_handle
def dashboard():
	"""
	GET /bfc/ui/dashboard

	Returns KPIs for the tenant dashboard:
	  - budget counts by status
	  - total budget/actual amounts
	  - overall variance %
	  - pending approvals
	  - material variances count
	  - forecast and scenario counts
	"""
	kpis = _run(_svc().dashboard_kpis())
	return _ok({
		"screen": "dashboard",
		"title": "Budgeting & Forecasting",
		"kpis": kpis,
		"sections": [
			{"id": "budgets",   "label": "Budgets",           "href": "/bfc/ui/budgets"},
			{"id": "forecasts", "label": "Forecasts",         "href": "/bfc/ui/forecasts"},
			{"id": "scenarios", "label": "Scenarios",         "href": "/bfc/ui/scenarios"},
			{"id": "variances", "label": "Variance Reports",  "href": "/bfc/ui/variance-reports"},
			{"id": "approvals", "label": "Pending Approvals", "href": "/bfc/ui/approvals"},
			{"id": "drivers",   "label": "Driver Assumptions","href": "/bfc/ui/drivers"},
		],
	})


# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------

@bfc_ui_bp.get("/budgets")
@_handle
def list_budgets():
	"""
	GET /bfc/ui/budgets?status=&fiscal_year=&budget_type=&offset=0&limit=50

	Returns list view model for the budget list screen.
	Columns: id, name, fiscal_year, budget_type, status, total_revenue,
	         total_expense, net_amount, owner_id, period_start, period_end.
	"""
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
	return _ok({
		"screen": "budgets",
		"title": "Budgets",
		"items": items,
		"count": len(items),
		"offset": offset,
		"limit": limit,
		"columns": [
			"id", "name", "fiscal_year", "budget_type", "status",
			"total_revenue", "total_expense", "net_amount",
			"owner_id", "period_start", "period_end",
		],
		"actions": [
			{"id": "create",   "label": "New Budget",     "method": "POST", "href": "/api/v1/bfc/budgets"},
			{"id": "submit",   "label": "Submit",         "method": "POST", "href": "/api/v1/bfc/budgets/{id}/submit"},
			{"id": "approve",  "label": "Approve",        "method": "POST", "href": "/api/v1/bfc/budgets/{id}/approve"},
			{"id": "lock",     "label": "Lock",           "method": "POST", "href": "/api/v1/bfc/budgets/{id}/lock"},
			{"id": "close",    "label": "Close",          "method": "POST", "href": "/api/v1/bfc/budgets/{id}/close"},
			{"id": "distribute","label": "Distribute",    "method": "POST", "href": "/api/v1/bfc/budgets/{id}/distribute"},
			{"id": "what_if",  "label": "What-If",        "method": "POST", "href": "/api/v1/bfc/budgets/{id}/what-if"},
			{"id": "cancel",   "label": "Cancel",         "method": "DELETE","href": "/api/v1/bfc/budgets/{id}"},
		],
		"filter_options": {
			"statuses": [s.value for s in BFBudgetStatus],
			"budget_types": [t.value for t in BFBudgetType],
		},
	})


@bfc_ui_bp.get("/budgets/<budget_id>")
@_handle
def detail_budget(budget_id: str):
	"""
	GET /bfc/ui/budgets/<id>

	Returns detail view model for a single budget including its lines,
	pending approvals, and variance summary.
	"""
	svc = _svc()
	budget = _run(svc.get_budget(budget_id))
	lines = _run(svc.get_budget_lines(budget_id))
	approvals = _run(svc.get_pending_approvals(budget_id))
	variance_reports = _run(svc.list_variance_reports(budget_id))

	return _ok({
		"screen": "budget_detail",
		"title": f"Budget: {budget.name}",
		"budget": budget,
		"lines": lines,
		"line_count": len(lines),
		"pending_approvals": approvals,
		"variance_reports": variance_reports,
		"actions": [
			{"id": "add_line",  "label": "Add Line",  "method": "POST"},
			{"id": "submit",    "label": "Submit",    "method": "POST"},
			{"id": "distribute","label": "Distribute","method": "POST"},
			{"id": "what_if",   "label": "What-If",   "method": "POST"},
			{"id": "consolidate","label": "Consolidate","method": "POST"},
		],
	})


@bfc_ui_bp.get("/budgets/create")
@_handle
def create_budget_form():
	"""GET /bfc/ui/budgets/create — returns form schema for budget creation."""
	return _ok({
		"screen": "create_budget",
		"title": "New Budget",
		"fields": [
			{"name": "name",         "type": "text",   "required": True,  "label": "Budget Name"},
			{"name": "description",  "type": "textarea","required": False, "label": "Description"},
			{"name": "budget_type",  "type": "select", "required": True,  "label": "Type",
			 "options": [t.value for t in BFBudgetType]},
			{"name": "fiscal_year",  "type": "number", "required": True,  "label": "Fiscal Year"},
			{"name": "period_start", "type": "date",   "required": True,  "label": "Period Start"},
			{"name": "period_end",   "type": "date",   "required": True,  "label": "Period End"},
			{"name": "currency_code","type": "text",   "required": False, "label": "Currency",
			 "default": "USD"},
			{"name": "owner_id",     "type": "text",   "required": True,  "label": "Owner ID"},
			{"name": "department_id","type": "text",   "required": False, "label": "Department ID"},
			{"name": "cost_center_id","type": "text",  "required": False, "label": "Cost Center ID"},
			{"name": "notes",        "type": "textarea","required": False, "label": "Notes"},
			{"name": "tags",         "type": "tags",   "required": False, "label": "Tags"},
		],
		"submit": {"method": "POST", "href": "/api/v1/bfc/budgets"},
	})


@bfc_ui_bp.get("/budgets/<budget_id>/edit")
@_handle
def edit_budget_form(budget_id: str):
	"""GET /bfc/ui/budgets/<id>/edit — returns current values pre-filled for edit form."""
	budget = _run(_svc().get_budget(budget_id))
	return _ok({
		"screen": "edit_budget",
		"title": f"Edit Budget: {budget.name}",
		"budget": budget,
		"editable_fields": ["name", "description", "notes", "tags", "metadata"],
		"submit": {"method": "PUT", "href": f"/api/v1/bfc/budgets/{budget_id}"},
	})


# ---------------------------------------------------------------------------
# Forecasts
# ---------------------------------------------------------------------------

@bfc_ui_bp.get("/forecasts")
@_handle
def list_forecasts():
	"""
	GET /bfc/ui/forecasts?status=&forecast_type=&limit=50

	Returns list view model for forecasts.
	"""
	svc = _svc()
	status_str = request.args.get("status")
	ft_str = request.args.get("forecast_type")
	limit = min(int(request.args.get("limit", 50)), 200)

	items = _run(svc.list_forecasts(
		status=BFForecastStatus(status_str) if status_str else None,
		forecast_type=BFForecastType(ft_str) if ft_str else None,
		limit=limit,
	))
	return _ok({
		"screen": "forecasts",
		"title": "Forecasts",
		"items": items,
		"count": len(items),
		"columns": [
			"id", "name", "forecast_type", "status",
			"period_start", "period_end", "total_forecasted",
			"mape", "rmse", "generated_at",
		],
		"actions": [
			{"id": "create",       "label": "New Forecast",     "method": "POST", "href": "/api/v1/bfc/forecasts"},
			{"id": "rolling",      "label": "Rolling Forecast",  "method": "POST", "href": "/api/v1/bfc/forecasts/{id}/rolling"},
			{"id": "reforecast",   "label": "Reforecast",        "method": "POST", "href": "/api/v1/bfc/forecasts/{id}/reforecast"},
			{"id": "driver_based", "label": "Driver-Based",      "method": "POST", "href": "/api/v1/bfc/forecasts/{id}/driver-based"},
			{"id": "ai_model",     "label": "AI Forecast Model", "method": "POST", "href": "/api/v1/bfc/forecasts/{id}/ai-model"},
		],
		"filter_options": {
			"statuses": [s.value for s in BFForecastStatus],
			"forecast_types": [t.value for t in BFForecastType],
		},
	})


@bfc_ui_bp.get("/forecasts/create")
@_handle
def create_forecast_form():
	"""GET /bfc/ui/forecasts/create — form schema for forecast creation."""
	return _ok({
		"screen": "create_forecast",
		"title": "New Forecast",
		"fields": [
			{"name": "name",              "type": "text",   "required": True,  "label": "Forecast Name"},
			{"name": "forecast_type",     "type": "select", "required": True,  "label": "Type",
			 "options": [t.value for t in BFForecastType]},
			{"name": "period_start",      "type": "date",   "required": True,  "label": "Period Start"},
			{"name": "period_end",        "type": "date",   "required": True,  "label": "Period End"},
			{"name": "currency_code",     "type": "text",   "required": False, "label": "Currency", "default": "USD"},
			{"name": "forecast_model_id", "type": "text",   "required": False, "label": "Model ID"},
			{"name": "budget_id",         "type": "text",   "required": False, "label": "Linked Budget ID"},
		],
		"submit": {"method": "POST", "href": "/api/v1/bfc/forecasts"},
	})


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

@bfc_ui_bp.get("/scenarios")
@_handle
def list_scenarios():
	"""GET /bfc/ui/scenarios — scenario list view."""
	active_only = request.args.get("active_only", "false").lower() == "true"
	items = _run(_svc().list_scenarios(active_only=active_only))
	return _ok({
		"screen": "scenarios",
		"title": "Scenarios",
		"items": items,
		"count": len(items),
		"columns": [
			"id", "name", "scenario_type", "probability",
			"is_active", "net_impact", "net_impact_pct", "ran_at",
		],
		"actions": [
			{"id": "create",  "label": "New Scenario", "method": "POST", "href": "/api/v1/bfc/scenarios"},
			{"id": "analyze", "label": "Run Analysis", "method": "POST", "href": "/api/v1/bfc/scenarios/analyze"},
		],
		"filter_options": {
			"scenario_types": [t.value for t in BFScenarioType],
		},
	})


@bfc_ui_bp.get("/scenarios/create")
@_handle
def create_scenario_form():
	"""GET /bfc/ui/scenarios/create — form schema."""
	return _ok({
		"screen": "create_scenario",
		"title": "New Scenario",
		"fields": [
			{"name": "name",            "type": "text",   "required": True,  "label": "Scenario Name"},
			{"name": "description",     "type": "textarea","required": False, "label": "Description"},
			{"name": "scenario_type",   "type": "select", "required": True,  "label": "Type",
			 "options": [t.value for t in BFScenarioType]},
			{"name": "probability",     "type": "number", "required": False, "label": "Probability (0–1)", "default": 0.5},
			{"name": "base_budget_id",  "type": "text",   "required": False, "label": "Base Budget ID"},
			{"name": "base_forecast_id","type": "text",   "required": False, "label": "Base Forecast ID"},
			{"name": "assumptions",     "type": "json",   "required": False, "label": "Assumptions"},
			{"name": "adjustments",     "type": "json_array","required": False,"label": "Adjustments"},
		],
		"submit": {"method": "POST", "href": "/api/v1/bfc/scenarios"},
	})


# ---------------------------------------------------------------------------
# Variance Reports
# ---------------------------------------------------------------------------

@bfc_ui_bp.get("/variance-reports")
@_handle
def list_variance_reports():
	"""GET /bfc/ui/variance-reports?budget_id= — variance report list."""
	budget_id = request.args.get("budget_id")
	reports = _run(_svc().list_variance_reports(budget_id))
	return _ok({
		"screen": "variance_reports",
		"title": "Variance Reports",
		"items": reports,
		"count": len(reports),
		"columns": [
			"id", "budget_id", "report_period_start", "report_period_end",
			"total_budget", "total_actual", "total_variance",
			"variance_pct", "variance_type", "significance",
		],
		"actions": [
			{"id": "create", "label": "Generate Report", "method": "POST", "href": "/api/v1/bfc/variance-reports"},
		],
	})


# ---------------------------------------------------------------------------
# Approvals
# ---------------------------------------------------------------------------

@bfc_ui_bp.get("/approvals")
@_handle
def list_approvals():
	"""GET /bfc/ui/approvals?budget_id= — pending approvals queue."""
	budget_id = request.args.get("budget_id")
	approvals = _run(_svc().get_pending_approvals(budget_id))
	return _ok({
		"screen": "approvals",
		"title": "Pending Approvals",
		"items": approvals,
		"count": len(approvals),
		"columns": [
			"id", "budget_id", "approver_id", "approver_name",
			"approver_role", "sequence", "required_by", "conditions",
		],
		"actions": [
			{"id": "approve",  "label": "Approve",  "method": "POST", "href": "/api/v1/bfc/budgets/{budget_id}/approve"},
			{"id": "reject",   "label": "Reject",   "method": "POST", "href": "/api/v1/bfc/budgets/{budget_id}/reject"},
			{"id": "delegate", "label": "Delegate", "method": "POST", "href": "/api/v1/bfc/approvals/{id}/delegate"},
		],
	})


# ---------------------------------------------------------------------------
# Driver Assumptions
# ---------------------------------------------------------------------------

@bfc_ui_bp.get("/drivers")
@_handle
def list_drivers():
	"""GET /bfc/ui/drivers?driver_type= — driver assumptions list."""
	dt_str = request.args.get("driver_type")
	items = _run(_svc().list_driver_assumptions(
		driver_type=BFDriverType(dt_str) if dt_str else None,
	))
	return _ok({
		"screen": "driver_assumptions",
		"title": "Driver Assumptions",
		"items": items,
		"count": len(items),
		"columns": [
			"id", "name", "driver_type", "value", "unit",
			"period_start", "period_end", "growth_rate", "confidence",
		],
		"actions": [
			{"id": "create",      "label": "New Driver",  "method": "POST", "href": "/api/v1/bfc/drivers"},
			{"id": "sensitivity", "label": "Sensitivity", "method": "POST", "href": "/api/v1/bfc/drivers/{id}/sensitivity"},
		],
		"filter_options": {
			"driver_types": [t.value for t in BFDriverType],
		},
	})


@bfc_ui_bp.get("/drivers/create")
@_handle
def create_driver_form():
	"""GET /bfc/ui/drivers/create — form schema for driver assumption creation."""
	return _ok({
		"screen": "create_driver",
		"title": "New Driver Assumption",
		"fields": [
			{"name": "name",         "type": "text",   "required": True,  "label": "Driver Name"},
			{"name": "driver_type",  "type": "select", "required": True,  "label": "Type",
			 "options": [t.value for t in BFDriverType]},
			{"name": "value",        "type": "number", "required": True,  "label": "Base Value"},
			{"name": "unit",         "type": "text",   "required": False, "label": "Unit"},
			{"name": "period_start", "type": "date",   "required": True,  "label": "Period Start"},
			{"name": "period_end",   "type": "date",   "required": True,  "label": "Period End"},
			{"name": "growth_rate",  "type": "number", "required": False, "label": "Annual Growth Rate"},
			{"name": "confidence",   "type": "number", "required": False, "label": "Confidence %", "default": 80.0},
			{"name": "source",       "type": "text",   "required": False, "label": "Source"},
			{"name": "linked_accounts","type": "tags", "required": False, "label": "Linked Accounts"},
		],
		"submit": {"method": "POST", "href": "/api/v1/bfc/drivers"},
	})


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

@bfc_ui_bp.get("/templates")
@_handle
def list_templates():
	"""GET /bfc/ui/templates — budget template list."""
	templates = _run(_svc().list_templates())
	return _ok({
		"screen": "templates",
		"title": "Budget Templates",
		"items": templates,
		"count": len(templates),
		"columns": ["id", "name", "budget_type", "is_active", "industry", "usage_count"],
		"actions": [
			{"id": "create",      "label": "New Template", "method": "POST", "href": "/api/v1/bfc/templates"},
			{"id": "instantiate", "label": "Use Template", "method": "POST", "href": "/api/v1/bfc/templates/{id}/instantiate"},
		],
	})


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

@bfc_ui_bp.get("/reports/summary")
@_handle
def report_summary():
	"""GET /bfc/ui/reports/summary?period_start=&period_end= — budget summary report."""
	ps = request.args.get("period_start", date.today().replace(month=1, day=1).isoformat())
	pe = request.args.get("period_end", date.today().isoformat())
	summary = _run(_svc().budget_summary(date.fromisoformat(ps), date.fromisoformat(pe)))
	return _ok({
		"screen": "report_summary",
		"title": "Budget Summary Report",
		"summary": summary,
		"period_start": ps,
		"period_end": pe,
	})


@bfc_ui_bp.get("/reports/audit")
@_handle
def report_audit():
	"""GET /bfc/ui/reports/audit?entity_id=&limit=100 — audit trail view."""
	entity_id = request.args.get("entity_id")
	limit = int(request.args.get("limit", 100))
	events = _run(_svc().audit_trail(entity_id))
	return _ok({
		"screen": "audit_trail",
		"title": "Audit Trail",
		"events": events[:limit],
		"count": len(events),
		"columns": ["occurred_at", "event", "actor_id", "entity_id", "payload"],
	})
