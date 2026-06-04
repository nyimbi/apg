"""Flask Blueprint views for APG Store Intelligence."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

from flask import Blueprint, g, jsonify, request

from .service import SinService

bp = Blueprint("retail_sin_views", __name__, url_prefix="/retail-sin")
_svc = SinService()


def _tenant_id() -> str:
	return getattr(g, "tenant_id", request.headers.get("X-Tenant-ID", "default"))


def has_access(permission: str) -> Callable:
	def decorator(fn: Callable) -> Callable:
		@wraps(fn)
		def wrapper(*args: Any, **kwargs: Any) -> Any:
			perms: set[str] = getattr(g, "permissions", set())
			if permission not in perms and "superadmin" not in perms:
				return jsonify({"error": "forbidden", "required_permission": permission}), 403
			return fn(*args, **kwargs)
		return wrapper
	return decorator


def _run(coro: Any) -> Any:
	import asyncio
	loop = asyncio.new_event_loop()
	try:
		return loop.run_until_complete(coro)
	finally:
		loop.close()


@bp.get("/dashboard")
@has_access("retail_sin:view")
def dashboard() -> Any:
	tid = _tenant_id()
	stores = _run(_svc.list_stores(tid))
	return jsonify({"tenant_id": tid, "store_count": len(stores)})


@bp.get("/stores")
@has_access("retail_sin:view")
def list_stores() -> Any:
	tid = _tenant_id()
	store_format = request.args.get("format")
	recs = _run(_svc.list_stores(tid, store_format))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@bp.post("/stores")
@has_access("retail_sin:admin")
def create_store() -> Any:
	from .models import SinStoreCreate
	tid = _tenant_id()
	body = request.get_json(force=True) or {}
	body["tenant_id"] = tid
	try:
		rec = _run(_svc.create_store(SinStoreCreate(**body)))
		return jsonify(rec.model_dump()), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/stores/<store_id>")
@has_access("retail_sin:view")
def store_detail(store_id: str) -> Any:
	tid = _tenant_id()
	summary = _run(_svc.store_performance_summary(tid, store_id))
	if not summary:
		return jsonify({"error": "not_found"}), 404
	return jsonify(summary)


@bp.get("/traffic")
@has_access("retail_sin:view")
def traffic() -> Any:
	tid = _tenant_id()
	store_id = request.args.get("store_id", "")
	zone_id = request.args.get("zone_id")
	recs = _run(_svc.list_traffic_counts(tid, store_id, zone_id))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@bp.post("/traffic")
@has_access("retail_sin:write")
def record_traffic() -> Any:
	from .models import SinTrafficCountCreate
	tid = _tenant_id()
	body = request.get_json(force=True) or {}
	body["tenant_id"] = tid
	try:
		rec = _run(_svc.record_traffic_count(SinTrafficCountCreate(**body)))
		return jsonify(rec.model_dump()), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/heatmaps")
@has_access("retail_sin:view")
def list_heatmaps() -> Any:
	tid = _tenant_id()
	store_id = request.args.get("store_id", "")
	recs = _run(_svc.list_heatmaps(tid, store_id))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@bp.post("/heatmaps")
@has_access("retail_sin:write")
def create_heatmap() -> Any:
	from .models import SinHeatmapCreate
	tid = _tenant_id()
	body = request.get_json(force=True) or {}
	body["tenant_id"] = tid
	try:
		rec = _run(_svc.create_heatmap(SinHeatmapCreate(**body)))
		return jsonify(rec.model_dump()), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/planogram")
@has_access("retail_sin:view")
def list_planogram_audits() -> Any:
	tid = _tenant_id()
	store_id = request.args.get("store_id", "")
	zone_id = request.args.get("zone_id")
	recs = _run(_svc.list_planogram_audits(tid, store_id, zone_id))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@bp.post("/planogram")
@has_access("retail_sin:write")
def record_planogram_audit() -> Any:
	from .models import SinPlanogramAuditCreate
	tid = _tenant_id()
	body = request.get_json(force=True) or {}
	body["tenant_id"] = tid
	try:
		rec = _run(_svc.record_planogram_audit(SinPlanogramAuditCreate(**body)))
		return jsonify(rec.model_dump()), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/shelf-alerts")
@has_access("retail_sin:view")
def list_shelf_alerts() -> Any:
	tid = _tenant_id()
	store_id = request.args.get("store_id", "")
	status = request.args.get("status")
	recs = _run(_svc.list_shelf_alerts(tid, store_id, status))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@bp.post("/shelf-alerts")
@has_access("retail_sin:write")
def raise_shelf_alert() -> Any:
	from .models import SinShelfAlertCreate
	tid = _tenant_id()
	body = request.get_json(force=True) or {}
	body["tenant_id"] = tid
	try:
		rec = _run(_svc.raise_shelf_alert(SinShelfAlertCreate(**body)))
		return jsonify(rec.model_dump()), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/shelf-alerts/<alert_id>/resolve")
@has_access("retail_sin:write")
def resolve_shelf_alert(alert_id: str) -> Any:
	tid = _tenant_id()
	body = request.get_json(force=True) or {}
	rec = _run(_svc.resolve_shelf_alert(tid, alert_id, body.get("notes", ""), body.get("by", "system")))
	if rec is None:
		return jsonify({"error": "not_found"}), 404
	return jsonify(rec.model_dump())


@bp.get("/conversion")
@has_access("retail_sin:view")
def conversion_funnel() -> Any:
	tid = _tenant_id()
	store_id = request.args.get("store_id", "")
	result = _run(_svc.get_conversion_funnel(tid, store_id))
	return jsonify(result)


@bp.get("/kpis")
@has_access("retail_sin:view")
def list_kpis() -> Any:
	tid = _tenant_id()
	store_id = request.args.get("store_id", "")
	category = request.args.get("category")
	recs = _run(_svc.list_kpi_snapshots(tid, store_id, category))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@bp.post("/kpis")
@has_access("retail_sin:write")
def record_kpi() -> Any:
	from .models import SinKpiSnapshotCreate
	tid = _tenant_id()
	body = request.get_json(force=True) or {}
	body["tenant_id"] = tid
	try:
		rec = _run(_svc.record_kpi_snapshot(SinKpiSnapshotCreate(**body)))
		return jsonify(rec.model_dump()), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/sensors")
@has_access("retail_sin:admin")
def list_sensors() -> Any:
	tid = _tenant_id()
	store_id = request.args.get("store_id")
	zone_id = request.args.get("zone_id")
	recs = _run(_svc.list_sensors(tid, store_id, zone_id))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@bp.post("/sensors")
@has_access("retail_sin:admin")
def register_sensor() -> Any:
	from .models import SinSensorCreate
	tid = _tenant_id()
	body = request.get_json(force=True) or {}
	body["tenant_id"] = tid
	try:
		rec = _run(_svc.register_sensor(SinSensorCreate(**body)))
		return jsonify(rec.model_dump()), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/sensors/<sensor_id>/heartbeat")
@has_access("retail_sin:write")
def sensor_heartbeat(sensor_id: str) -> Any:
	tid = _tenant_id()
	rec = _run(_svc.sensor_heartbeat(tid, sensor_id))
	if rec is None:
		return jsonify({"error": "not_found"}), 404
	return jsonify(rec.model_dump())
