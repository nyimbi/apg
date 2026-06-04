"""REST API Blueprint for APG Store Intelligence."""

from __future__ import annotations

from typing import Any

from flask import Blueprint, g, jsonify, request

from .service import SinService
from .capability_contract import get_capability_contract, evaluate_capability_rules

api = Blueprint("retail_sin_api", __name__, url_prefix="/retail-sin/api/v1")
_svc = SinService()


def _tenant_id() -> str:
	return getattr(g, "tenant_id", request.headers.get("X-Tenant-ID", "default"))


def _run(coro: Any) -> Any:
	import asyncio
	loop = asyncio.new_event_loop()
	try:
		return asyncio.run(coro)
	finally:
		loop.close()


def _err(msg: str, code: int = 400) -> Any:
	return jsonify({"error": msg, "status": code}), code


@api.get("/contract")
def contract() -> Any:
	"""Return capability contract. GET /retail-sin/api/v1/contract"""
	return jsonify(get_capability_contract(_tenant_id()))


@api.post("/rules/evaluate")
def evaluate_rules() -> Any:
	"""Evaluate rules. POST /retail-sin/api/v1/rules/evaluate"""
	return jsonify(evaluate_capability_rules(request.get_json(force=True) or {}))


# Stores
@api.get("/stores")
def list_stores() -> Any:
	"""List stores. GET /retail-sin/api/v1/stores?format=<f>"""
	recs = _run(_svc.list_stores(_tenant_id(), request.args.get("format")))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@api.post("/stores")
def create_store() -> Any:
	"""Create store. POST /retail-sin/api/v1/stores"""
	from .models import SinStoreCreate
	body = request.get_json(force=True) or {}
	body["tenant_id"] = _tenant_id()
	try:
		return jsonify(_run(_svc.create_store(SinStoreCreate(**body))).model_dump()), 201
	except Exception as exc:
		return _err(str(exc))


@api.get("/stores/<store_id>")
def get_store(store_id: str) -> Any:
	"""Get store performance summary. GET /retail-sin/api/v1/stores/<store_id>"""
	summary = _run(_svc.store_performance_summary(_tenant_id(), store_id))
	return jsonify(summary) if summary else _err("not_found", 404)


# Zones
@api.get("/zones")
def list_zones() -> Any:
	"""List zones. GET /retail-sin/api/v1/zones?store_id=<id>"""
	store_id = request.args.get("store_id", "")
	recs = _run(_svc.list_zones(_tenant_id(), store_id))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@api.post("/zones")
def create_zone() -> Any:
	"""Create zone. POST /retail-sin/api/v1/zones"""
	from .models import SinZoneCreate
	body = request.get_json(force=True) or {}
	body["tenant_id"] = _tenant_id()
	try:
		return jsonify(_run(_svc.create_zone(SinZoneCreate(**body))).model_dump()), 201
	except Exception as exc:
		return _err(str(exc))


# Sensors
@api.get("/sensors")
def list_sensors() -> Any:
	"""List sensors. GET /retail-sin/api/v1/sensors?store_id=<id>&zone_id=<id>"""
	recs = _run(_svc.list_sensors(_tenant_id(), request.args.get("store_id"), request.args.get("zone_id")))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@api.post("/sensors")
def register_sensor() -> Any:
	"""Register sensor. POST /retail-sin/api/v1/sensors"""
	from .models import SinSensorCreate
	body = request.get_json(force=True) or {}
	body["tenant_id"] = _tenant_id()
	try:
		return jsonify(_run(_svc.register_sensor(SinSensorCreate(**body))).model_dump()), 201
	except Exception as exc:
		return _err(str(exc))


@api.post("/sensors/<sensor_id>/heartbeat")
def sensor_heartbeat(sensor_id: str) -> Any:
	"""Sensor heartbeat. POST /retail-sin/api/v1/sensors/<sensor_id>/heartbeat"""
	rec = _run(_svc.sensor_heartbeat(_tenant_id(), sensor_id))
	return jsonify(rec.model_dump()) if rec else _err("not_found", 404)


# Traffic
@api.get("/traffic")
def list_traffic() -> Any:
	"""List traffic counts. GET /retail-sin/api/v1/traffic?store_id=<id>&zone_id=<id>"""
	recs = _run(_svc.list_traffic_counts(_tenant_id(), request.args.get("store_id",""), request.args.get("zone_id")))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@api.post("/traffic")
def record_traffic() -> Any:
	"""Record traffic count. POST /retail-sin/api/v1/traffic"""
	from .models import SinTrafficCountCreate
	body = request.get_json(force=True) or {}
	body["tenant_id"] = _tenant_id()
	try:
		return jsonify(_run(_svc.record_traffic_count(SinTrafficCountCreate(**body))).model_dump()), 201
	except Exception as exc:
		return _err(str(exc))


@api.get("/traffic/summary")
def traffic_summary() -> Any:
	"""Traffic summary. GET /retail-sin/api/v1/traffic/summary?store_id=<id>"""
	from datetime import datetime
	store_id = request.args.get("store_id", "")
	result = _run(_svc.get_traffic_summary(_tenant_id(), store_id, datetime.utcnow(), datetime.utcnow()))
	return jsonify(result)


# Planogram
@api.get("/planogram")
def list_planogram() -> Any:
	"""List planogram audits. GET /retail-sin/api/v1/planogram?store_id=<id>"""
	recs = _run(_svc.list_planogram_audits(_tenant_id(), request.args.get("store_id",""), request.args.get("zone_id")))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@api.post("/planogram")
def record_planogram() -> Any:
	"""Record planogram audit. POST /retail-sin/api/v1/planogram"""
	from .models import SinPlanogramAuditCreate
	body = request.get_json(force=True) or {}
	body["tenant_id"] = _tenant_id()
	try:
		return jsonify(_run(_svc.record_planogram_audit(SinPlanogramAuditCreate(**body))).model_dump()), 201
	except Exception as exc:
		return _err(str(exc))


@api.get("/planogram/compliance/<store_id>")
def store_compliance(store_id: str) -> Any:
	"""Get store compliance rate. GET /retail-sin/api/v1/planogram/compliance/<store_id>"""
	rate = _run(_svc.get_store_compliance_rate(_tenant_id(), store_id))
	return jsonify({"store_id": store_id, "compliance_rate_pct": rate})


# Shelf alerts
@api.get("/shelf-alerts")
def list_shelf_alerts() -> Any:
	"""List shelf alerts. GET /retail-sin/api/v1/shelf-alerts?store_id=<id>&status=<s>"""
	recs = _run(_svc.list_shelf_alerts(_tenant_id(), request.args.get("store_id",""), request.args.get("status")))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@api.post("/shelf-alerts")
def raise_shelf_alert() -> Any:
	"""Raise shelf alert. POST /retail-sin/api/v1/shelf-alerts"""
	from .models import SinShelfAlertCreate
	body = request.get_json(force=True) or {}
	body["tenant_id"] = _tenant_id()
	try:
		return jsonify(_run(_svc.raise_shelf_alert(SinShelfAlertCreate(**body))).model_dump()), 201
	except Exception as exc:
		return _err(str(exc))


@api.put("/shelf-alerts/<alert_id>/resolve")
def resolve_alert(alert_id: str) -> Any:
	"""Resolve shelf alert. PUT /retail-sin/api/v1/shelf-alerts/<alert_id>/resolve"""
	body = request.get_json(force=True) or {}
	rec = _run(_svc.resolve_shelf_alert(_tenant_id(), alert_id, body.get("notes",""), body.get("by","system")))
	return jsonify(rec.model_dump()) if rec else _err("not_found", 404)


@api.post("/shelf-alerts/<alert_id>/replenish")
def trigger_replenishment(alert_id: str) -> Any:
	"""Trigger replenishment. POST /retail-sin/api/v1/shelf-alerts/<alert_id>/replenish"""
	rec = _run(_svc.trigger_replenishment(_tenant_id(), alert_id))
	return jsonify(rec.model_dump()) if rec else _err("not_found", 404)


# Conversion
@api.get("/conversion")
def conversion_funnel() -> Any:
	"""Conversion funnel. GET /retail-sin/api/v1/conversion?store_id=<id>"""
	result = _run(_svc.get_conversion_funnel(_tenant_id(), request.args.get("store_id","")))
	return jsonify(result)


@api.post("/conversion")
def record_conversion() -> Any:
	"""Record conversion event. POST /retail-sin/api/v1/conversion"""
	from .models import SinConversionEventCreate
	body = request.get_json(force=True) or {}
	body["tenant_id"] = _tenant_id()
	try:
		return jsonify(_run(_svc.record_conversion_event(SinConversionEventCreate(**body))).model_dump()), 201
	except Exception as exc:
		return _err(str(exc))


# KPIs
@api.get("/kpis")
def list_kpis() -> Any:
	"""List KPI snapshots. GET /retail-sin/api/v1/kpis?store_id=<id>&category=<c>"""
	recs = _run(_svc.list_kpi_snapshots(_tenant_id(), request.args.get("store_id",""), request.args.get("category")))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@api.post("/kpis")
def record_kpi() -> Any:
	"""Record KPI snapshot. POST /retail-sin/api/v1/kpis"""
	from .models import SinKpiSnapshotCreate
	body = request.get_json(force=True) or {}
	body["tenant_id"] = _tenant_id()
	try:
		return jsonify(_run(_svc.record_kpi_snapshot(SinKpiSnapshotCreate(**body))).model_dump()), 201
	except Exception as exc:
		return _err(str(exc))


# Heatmaps
@api.get("/heatmaps")
def list_heatmaps() -> Any:
	"""List heatmaps. GET /retail-sin/api/v1/heatmaps?store_id=<id>"""
	recs = _run(_svc.list_heatmaps(_tenant_id(), request.args.get("store_id","")))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@api.post("/heatmaps")
def create_heatmap() -> Any:
	"""Create heatmap. POST /retail-sin/api/v1/heatmaps"""
	from .models import SinHeatmapCreate
	body = request.get_json(force=True) or {}
	body["tenant_id"] = _tenant_id()
	try:
		return jsonify(_run(_svc.create_heatmap(SinHeatmapCreate(**body))).model_dump()), 201
	except Exception as exc:
		return _err(str(exc))
