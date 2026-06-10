"""Weather & Climate Analytics Flask Blueprint — agr_wth."""
from __future__ import annotations
import logging
from flask import Blueprint, jsonify, request
from .service import WeatherClimateService

_log = logging.getLogger(__name__)
bp = Blueprint("agr_wth", __name__, url_prefix="/api/agriculture/wth")
_svc: dict[str, WeatherClimateService] = {}


def _get_svc(t: str = "default") -> WeatherClimateService:
	if t not in _svc:
		_svc[t] = WeatherClimateService(tenant_id=t)
	return _svc[t]


def _t() -> str:
	return request.headers.get("X-Tenant-ID", "default")


@bp.get("/health")
async def health():
	return jsonify(await _get_svc(_t()).health_check()), 200


@bp.get("/forecasts")
async def list_forecasts():
	svc = _get_svc(_t())
	items = await svc.list_forecasts(region=request.args.get("region"), source=request.args.get("source"))
	return jsonify({"items": items, "count": len(items)}), 200


@bp.get("/forecasts/latest")
async def latest_forecast():
	svc = _get_svc(_t())
	region = request.args.get("region", "")
	result = await svc.get_latest_forecast(region)
	if not result:
		return jsonify({"error": "no_forecast_found"}), 404
	return jsonify(result), 200


@bp.post("/forecasts")
async def create_forecast():
	try:
		return jsonify(await _get_svc(_t()).create_forecast(request.get_json(force=True) or {})), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.delete("/forecasts/<forecast_id>")
async def delete_forecast(forecast_id: str):
	try:
		return jsonify(await _get_svc(_t()).delete_forecast(forecast_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/thresholds")
async def list_thresholds():
	svc = _get_svc(_t())
	active_str = request.args.get("active")
	active = None if active_str is None else active_str.lower() == "true"
	items = await svc.list_thresholds(region=request.args.get("region"), active=active)
	return jsonify({"items": items, "count": len(items)}), 200


@bp.post("/thresholds")
async def create_threshold():
	try:
		return jsonify(await _get_svc(_t()).create_threshold(request.get_json(force=True) or {})), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.put("/thresholds/<threshold_id>")
async def update_threshold(threshold_id: str):
	try:
		return jsonify(await _get_svc(_t()).update_threshold(threshold_id, request.get_json(force=True) or {})), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.delete("/thresholds/<threshold_id>")
async def delete_threshold(threshold_id: str):
	try:
		return jsonify(await _get_svc(_t()).delete_threshold(threshold_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/alerts")
async def list_alerts():
	svc = _get_svc(_t())
	ack_str = request.args.get("acknowledged")
	ack = None if ack_str is None else ack_str.lower() == "true"
	items = await svc.list_alerts(region=request.args.get("region"), acknowledged=ack, severity=request.args.get("severity"))
	return jsonify({"items": items, "count": len(items)}), 200


@bp.post("/alerts/<alert_id>/acknowledge")
async def acknowledge_alert(alert_id: str):
	try:
		return jsonify(await _get_svc(_t()).acknowledge_alert(alert_id)), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/history")
async def list_history():
	svc = _get_svc(_t())
	year_str = request.args.get("year")
	month_str = request.args.get("month")
	items = await svc.list_historical_patterns(
		region=request.args.get("region"),
		year=int(year_str) if year_str else None,
		month=int(month_str) if month_str else None,
	)
	return jsonify({"items": items, "count": len(items)}), 200


@bp.post("/history")
async def create_history():
	try:
		return jsonify(await _get_svc(_t()).create_historical_pattern(request.get_json(force=True) or {})), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/history/normals")
async def compute_normals():
	svc = _get_svc(_t())
	result = await svc.compute_monthly_normals(
		region=request.args.get("region", ""),
		month=int(request.args.get("month", 1)),
	)
	return jsonify(result), 200


@bp.post("/risk-assessments")
async def assess_risk():
	try:
		body = request.get_json(force=True) or {}
		result = await _get_svc(_t()).assess_climate_risk(
			region=body["region"], crop_type=body["crop_type"], season=body["season"]
		)
		return jsonify(result), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/risk-assessments")
async def list_risk():
	svc = _get_svc(_t())
	items = await svc.list_risk_assessments(
		region=request.args.get("region"), crop_type=request.args.get("crop_type")
	)
	return jsonify({"items": items, "count": len(items)}), 200


@bp.get("/audit")
async def get_audit():
	events = await _get_svc(_t()).get_audit_events(int(request.args.get("limit", 100)))
	return jsonify({"events": events, "count": len(events)}), 200
