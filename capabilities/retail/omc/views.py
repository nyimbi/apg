"""Flask Blueprint views for APG Omnichannel Commerce."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

from flask import Blueprint, g, jsonify, request

from .service import OmcService

bp = Blueprint("retail_omc_views", __name__, url_prefix="/retail-omc")
_svc = OmcService()


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
@has_access("retail_omc:view")
def dashboard() -> Any:
	tid = _tenant_id()
	channels = _run(_svc.list_channels(tid))
	orders = _run(_svc.list_orders(tid))
	return jsonify({
		"tenant_id": tid,
		"channel_count": len(channels),
		"order_count": len(orders),
		"open_orders": sum(1 for o in orders if o.status not in ("delivered", "collected", "cancelled", "refunded")),
	})


@bp.get("/channels")
@has_access("retail_omc:view")
def list_channels() -> Any:
	tid = _tenant_id()
	recs = _run(_svc.list_channels(tid))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@bp.post("/channels")
@has_access("retail_omc:admin")
def create_channel() -> Any:
	from .models import OmcChannelCreate
	tid = _tenant_id()
	body = request.get_json(force=True) or {}
	body["tenant_id"] = tid
	try:
		rec = _run(_svc.create_channel(OmcChannelCreate(**body)))
		return jsonify(rec.model_dump()), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/inventory")
@has_access("retail_omc:view")
def inventory() -> Any:
	tid = _tenant_id()
	sku = request.args.get("sku", "")
	location_id = request.args.get("location_id")
	if not sku:
		return jsonify({"error": "sku parameter required"}), 400
	recs = _run(_svc.get_inventory(tid, sku, location_id))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@bp.post("/inventory")
@has_access("retail_omc:write")
def upsert_inventory() -> Any:
	from .models import OmcInventoryRecord
	tid = _tenant_id()
	body = request.get_json(force=True) or {}
	body["tenant_id"] = tid
	try:
		rec = _run(_svc.upsert_inventory(OmcInventoryRecord(**body)))
		return jsonify(rec.model_dump()), 200
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/orders")
@has_access("retail_omc:view")
def list_orders() -> Any:
	tid = _tenant_id()
	channel_id = request.args.get("channel_id")
	status = request.args.get("status")
	recs = _run(_svc.list_orders(tid, channel_id, status))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@bp.post("/orders")
@has_access("retail_omc:write")
def create_order() -> Any:
	from .models import OmcOrderCreate
	tid = _tenant_id()
	body = request.get_json(force=True) or {}
	body["tenant_id"] = tid
	try:
		rec = _run(_svc.create_order(OmcOrderCreate(**body)))
		return jsonify(rec.model_dump()), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/orders/<order_id>")
@has_access("retail_omc:view")
def order_detail(order_id: str) -> Any:
	tid = _tenant_id()
	rec = _run(_svc.get_order(tid, order_id))
	if rec is None:
		return jsonify({"error": "not_found"}), 404
	return jsonify(rec.model_dump())


@bp.put("/orders/<order_id>")
@has_access("retail_omc:write")
def update_order(order_id: str) -> Any:
	from .models import OmcOrderUpdate
	tid = _tenant_id()
	body = request.get_json(force=True) or {}
	try:
		rec = _run(_svc.update_order(tid, order_id, OmcOrderUpdate(**body)))
		if rec is None:
			return jsonify({"error": "not_found"}), 404
		return jsonify(rec.model_dump())
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/orders/<order_id>/cancel")
@has_access("retail_omc:write")
def cancel_order(order_id: str) -> Any:
	tid = _tenant_id()
	body = request.get_json(force=True) or {}
	rec = _run(_svc.cancel_order(tid, order_id, body.get("reason", ""), body.get("by", "system")))
	if rec is None:
		return jsonify({"error": "not_found"}), 404
	return jsonify(rec.model_dump())


@bp.post("/orders/<order_id>/ship")
@has_access("retail_omc:write")
def ship_order(order_id: str) -> Any:
	tid = _tenant_id()
	body = request.get_json(force=True) or {}
	rec = _run(_svc.mark_order_shipped(tid, order_id, body.get("tracking", ""), body.get("by", "system")))
	if rec is None:
		return jsonify({"error": "not_found"}), 404
	return jsonify(rec.model_dump())


@bp.post("/orders/<order_id>/collect")
@has_access("retail_omc:write")
def collect_order(order_id: str) -> Any:
	tid = _tenant_id()
	body = request.get_json(force=True) or {}
	rec = _run(_svc.mark_order_collected(tid, order_id, body.get("by", "system")))
	if rec is None:
		return jsonify({"error": "not_found"}), 404
	return jsonify(rec.model_dump())


@bp.get("/returns")
@has_access("retail_omc:view")
def list_returns() -> Any:
	tid = _tenant_id()
	order_id = request.args.get("order_id")
	recs = _run(_svc.list_returns(tid, order_id))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@bp.post("/returns")
@has_access("retail_omc:write")
def initiate_return() -> Any:
	from .models import OmcReturnCreate
	tid = _tenant_id()
	body = request.get_json(force=True) or {}
	body["tenant_id"] = tid
	try:
		rec = _run(_svc.initiate_return(OmcReturnCreate(**body)))
		return jsonify(rec.model_dump()), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/journey/<session_id>")
@has_access("retail_omc:view")
def journey(session_id: str) -> Any:
	tid = _tenant_id()
	recs = _run(_svc.get_session_journey(tid, session_id))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@bp.post("/journey")
@has_access("retail_omc:write")
def record_journey() -> Any:
	from .models import OmcJourneyEventCreate
	tid = _tenant_id()
	body = request.get_json(force=True) or {}
	body["tenant_id"] = tid
	try:
		rec = _run(_svc.record_journey_event(OmcJourneyEventCreate(**body)))
		return jsonify(rec.model_dump()), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/pricing")
@has_access("retail_omc:admin")
def list_pricing_rules() -> Any:
	tid = _tenant_id()
	channel_id = request.args.get("channel_id")
	recs = _run(_svc.list_pricing_rules(tid, channel_id))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@bp.post("/pricing")
@has_access("retail_omc:admin")
def create_pricing_rule() -> Any:
	from .models import OmcPricingRuleCreate
	tid = _tenant_id()
	body = request.get_json(force=True) or {}
	body["tenant_id"] = tid
	try:
		rec = _run(_svc.create_pricing_rule(OmcPricingRuleCreate(**body)))
		return jsonify(rec.model_dump()), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400
