"""Flask Blueprint — REST API for Revenue Management & Rates."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from flask import Blueprint, jsonify, request

from .service import RMSService

_log = logging.getLogger(__name__)

rms_bp = Blueprint("hos_rms", __name__, url_prefix="/api/hospitality/rms")
_svc = RMSService()


def _run(coro: Any) -> Any:
	try:
		loop = asyncio.get_event_loop()
	except RuntimeError:
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
	return loop.run_until_complete(coro)


def _tenant() -> str:
	return request.headers.get("X-Tenant-ID", "default")


@rms_bp.get("/health")
def health():
	return jsonify(_run(_svc.health_check()))


@rms_bp.get("/rate-plans")
def list_rate_plans():
	room_type = request.args.get("room_type")
	active_only = request.args.get("active_only", "false").lower() == "true"
	return jsonify(_run(_svc.list_rate_plans(_tenant(), room_type=room_type, active_only=active_only)))


@rms_bp.get("/rate-plans/<plan_id>")
def get_rate_plan(plan_id: str):
	try:
		return jsonify(_run(_svc.get_rate_plan(plan_id, _tenant())))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@rms_bp.post("/rate-plans")
def create_rate_plan():
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.create_rate_plan(
			code=data["code"],
			name=data["name"],
			base_rate=float(data["base_rate"]),
			room_type=data["room_type"],
			min_stay=data.get("min_stay", 1),
			meal_plan=data.get("meal_plan", "room_only"),
			cancellation_policy=data.get("cancellation_policy", "flexible"),
			is_public=data.get("is_public", True),
			description=data.get("description"),
			tenant_id=_tenant(),
		))), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@rms_bp.put("/rate-plans/<plan_id>")
def update_rate_plan(plan_id: str):
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.update_rate_plan(plan_id, data, _tenant())))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@rms_bp.delete("/rate-plans/<plan_id>")
def delete_rate_plan(plan_id: str):
	try:
		return jsonify(_run(_svc.delete_rate_plan(plan_id, _tenant())))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@rms_bp.get("/rate-plans/<plan_id>/effective-rate")
def effective_rate(plan_id: str):
	date = request.args.get("date", "")
	try:
		return jsonify(_run(_svc.get_effective_rate(plan_id, date, _tenant())))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@rms_bp.get("/forecasts")
def list_forecasts():
	room_type = request.args.get("room_type")
	date_from = request.args.get("date_from")
	return jsonify(_run(_svc.list_demand_forecasts(_tenant(), room_type=room_type, date_from=date_from)))


@rms_bp.post("/forecasts")
def create_forecast():
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.create_demand_forecast(
			forecast_date=data["forecast_date"],
			room_type=data["room_type"],
			predicted_demand=float(data["predicted_demand"]),
			confidence=float(data.get("confidence", 0.8)),
			events=data.get("events", []),
			notes=data.get("notes"),
			tenant_id=_tenant(),
		))), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@rms_bp.get("/competitor-rates")
def list_competitor_rates():
	date = request.args.get("date")
	room_type = request.args.get("room_type")
	return jsonify(_run(_svc.list_competitor_rates(_tenant(), date=date, room_type=room_type)))


@rms_bp.post("/competitor-rates")
def create_competitor_rate():
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.create_competitor_rate(
			competitor_name=data["competitor_name"],
			room_type=data["room_type"],
			rate=float(data["rate"]),
			date=data["date"],
			source=data.get("source", "manual"),
			channel=data.get("channel"),
			tenant_id=_tenant(),
		))), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@rms_bp.get("/parity-alerts")
def list_parity_alerts():
	severity = request.args.get("severity")
	status = request.args.get("status")
	return jsonify(_run(_svc.list_parity_alerts(_tenant(), severity=severity, status=status)))


@rms_bp.post("/parity-alerts/<alert_id>/resolve")
def resolve_parity_alert(alert_id: str):
	data = request.get_json(force=True) or {}
	try:
		return jsonify(_run(_svc.resolve_parity_alert(alert_id, data.get("resolution", "acknowledged"), _tenant())))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@rms_bp.post("/yield-optimisation")
def run_yield():
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.run_yield_optimisation(
			date_from=data["date_from"],
			date_to=data["date_to"],
			room_type=data["room_type"],
			current_occupancy=float(data["current_occupancy"]),
			target_occupancy=float(data.get("target_occupancy", 0.85)),
			tenant_id=_tenant(),
		))), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@rms_bp.get("/yield-reports")
def list_yield_reports():
	return jsonify(_run(_svc.list_yield_reports(_tenant())))


@rms_bp.post("/seasonal-rules")
def create_seasonal_rule():
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.create_seasonal_rule(
			name=data["name"],
			date_from=data["date_from"],
			date_to=data["date_to"],
			multiplier=float(data["multiplier"]),
			room_type=data.get("room_type"),
			tenant_id=_tenant(),
		))), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@rms_bp.get("/seasonal-rules")
def list_seasonal_rules():
	return jsonify(_run(_svc.list_seasonal_rules(_tenant())))


@rms_bp.post("/revenue-targets")
def set_revenue_target():
	data = request.get_json(force=True)
	try:
		return jsonify(_run(_svc.set_revenue_target(
			period=data["period"],
			room_type=data["room_type"],
			target_revpar=float(data["target_revpar"]),
			target_adr=float(data["target_adr"]),
			target_occupancy=float(data["target_occupancy"]),
			tenant_id=_tenant(),
		))), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@rms_bp.get("/parity-report")
def parity_report():
	date_from = request.args.get("date_from", "")
	date_to = request.args.get("date_to", "9999-12-31")
	room_type = request.args.get("room_type")
	return jsonify(_run(_svc.rate_parity_report(date_from, date_to, room_type, _tenant())))


@rms_bp.get("/dashboard")
def dashboard():
	return jsonify(_run(_svc.dashboard_summary(_tenant())))


@rms_bp.get("/audit-events")
def audit_events():
	return jsonify(_run(_svc.get_audit_events(_tenant())))
