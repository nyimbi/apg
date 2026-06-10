"""Flask Blueprint REST API for Supply Planning (scm_spl)."""
from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, jsonify, request

from .service import SupplyPlanningService

_log = logging.getLogger(__name__)

bp = Blueprint("scm_spl", __name__, url_prefix="/api/scm/spl")
_svc = SupplyPlanningService()


def _run(coro: Any) -> Any:
	import asyncio
	try:
		loop = asyncio.get_event_loop()
	except RuntimeError:
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
	return loop.run_until_complete(coro)


@bp.get("/health")
def health():
	return jsonify(_run(_svc.health_check()))


@bp.get("/describe")
def describe():
	return jsonify(_run(_svc.describe()))


# ── Demand forecasts ──────────────────────────────────────────────────────────

@bp.get("/demand-forecasts")
def list_forecasts():
	tenant = request.args.get("tenant_id", "default")
	sku = request.args.get("sku")
	period = request.args.get("period")
	return jsonify(_run(_svc.list_demand_forecasts(sku=sku, period=period, tenant_id=tenant)))


@bp.get("/demand-forecasts/<forecast_id>")
def get_forecast(forecast_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.get_demand_forecast(forecast_id, tenant_id=tenant)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/demand-forecasts")
def create_forecast():
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.create_demand_forecast(tenant_id=tenant, **data))), 201
	except (ValueError, KeyError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.delete("/demand-forecasts/<forecast_id>")
def delete_forecast(forecast_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.delete_demand_forecast(forecast_id, tenant_id=tenant)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


# ── MRP runs ──────────────────────────────────────────────────────────────────

@bp.get("/mrp-runs")
def list_mrp_runs():
	tenant = request.args.get("tenant_id", "default")
	return jsonify(_run(_svc.list_mrp_runs(tenant_id=tenant)))


@bp.get("/mrp-runs/<run_id>")
def get_mrp_run(run_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.get_mrp_run(run_id, tenant_id=tenant)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/mrp-runs")
def run_mrp():
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.run_mrp(tenant_id=tenant, **data))), 201
	except (ValueError, KeyError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── Safety stock ──────────────────────────────────────────────────────────────

@bp.get("/safety-stocks")
def list_safety_stocks():
	tenant = request.args.get("tenant_id", "default")
	sku = request.args.get("sku")
	return jsonify(_run(_svc.list_safety_stocks(sku=sku, tenant_id=tenant)))


@bp.post("/safety-stocks")
def calculate_safety_stock():
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.calculate_safety_stock(tenant_id=tenant, **data))), 201
	except (ValueError, KeyError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── Replenishment rules ───────────────────────────────────────────────────────

@bp.get("/replenishment-rules")
def list_rules():
	tenant = request.args.get("tenant_id", "default")
	sku = request.args.get("sku")
	return jsonify(_run(_svc.list_replenishment_rules(sku=sku, tenant_id=tenant)))


@bp.get("/replenishment-rules/<rule_id>")
def get_rule(rule_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.get_replenishment_rule(rule_id, tenant_id=tenant)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/replenishment-rules")
def create_rule():
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.create_replenishment_rule(tenant_id=tenant, **data))), 201
	except (ValueError, KeyError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/replenishment-rules/<rule_id>")
def update_rule(rule_id: str):
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.update_replenishment_rule(rule_id, data, tenant_id=tenant)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.delete("/replenishment-rules/<rule_id>")
def delete_rule(rule_id: str):
	tenant = request.args.get("tenant_id", "default")
	try:
		return jsonify(_run(_svc.delete_replenishment_rule(rule_id, tenant_id=tenant)))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/replenishment-rules/evaluate")
def evaluate_triggers():
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	current_stocks = data.get("current_stocks", {})
	return jsonify(_run(_svc.evaluate_replenishment_triggers(current_stocks, tenant_id=tenant)))


# ── Capacity plans ────────────────────────────────────────────────────────────

@bp.get("/capacity-plans")
def list_capacity_plans():
	tenant = request.args.get("tenant_id", "default")
	period = request.args.get("period")
	return jsonify(_run(_svc.list_capacity_plans(period=period, tenant_id=tenant)))


@bp.post("/capacity-plans")
def create_capacity_plan():
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.create_capacity_plan(tenant_id=tenant, **data))), 201
	except (ValueError, KeyError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── Supply/demand balances ────────────────────────────────────────────────────

@bp.get("/supply-demand-balances")
def list_balances():
	tenant = request.args.get("tenant_id", "default")
	sku = request.args.get("sku")
	period = request.args.get("period")
	return jsonify(_run(_svc.list_supply_demand_balances(sku=sku, period=period, tenant_id=tenant)))


@bp.post("/supply-demand-balances")
def create_balance():
	data = request.get_json(force=True) or {}
	tenant = data.pop("tenant_id", "default")
	try:
		return jsonify(_run(_svc.create_supply_demand_balance(tenant_id=tenant, **data))), 201
	except (ValueError, KeyError) as exc:
		return jsonify({"error": str(exc)}), 400


# ── Analytics ─────────────────────────────────────────────────────────────────

@bp.get("/analytics/dashboard")
def planning_dashboard():
	tenant = request.args.get("tenant_id", "default")
	return jsonify(_run(_svc.planning_dashboard(tenant_id=tenant)))


@bp.get("/audit-events")
def audit_events():
	tenant = request.args.get("tenant_id", "default")
	return jsonify(_run(_svc.get_audit_events(tenant_id=tenant)))
