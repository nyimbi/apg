"""REST API Blueprint for APG Omnichannel Commerce."""

from __future__ import annotations

from typing import Any

from flask import Blueprint, g, jsonify, request

from .service import OmcService
from .capability_contract import get_capability_contract, evaluate_capability_rules

api = Blueprint("retail_omc_api", __name__, url_prefix="/retail-omc/api/v1")
_svc = OmcService()


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
	"""Return the capability contract. GET /retail-omc/api/v1/contract"""
	return jsonify(get_capability_contract(_tenant_id()))


@api.post("/rules/evaluate")
def evaluate_rules() -> Any:
	"""Evaluate rules against context. POST /retail-omc/api/v1/rules/evaluate"""
	return jsonify(evaluate_capability_rules(request.get_json(force=True) or {}))


# Channels
@api.get("/channels")
def list_channels() -> Any:
	"""List channels. GET /retail-omc/api/v1/channels"""
	recs = _run(_svc.list_channels(_tenant_id()))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@api.post("/channels")
def create_channel() -> Any:
	"""Create channel. POST /retail-omc/api/v1/channels"""
	from .models import OmcChannelCreate
	body = request.get_json(force=True) or {}
	body["tenant_id"] = _tenant_id()
	try:
		return jsonify(_run(_svc.create_channel(OmcChannelCreate(**body))).model_dump()), 201
	except Exception as exc:
		return _err(str(exc))


@api.get("/channels/<channel_id>")
def get_channel(channel_id: str) -> Any:
	"""Get channel. GET /retail-omc/api/v1/channels/<channel_id>"""
	rec = _run(_svc.get_channel(_tenant_id(), channel_id))
	return jsonify(rec.model_dump()) if rec else _err("not_found", 404)


# Catalogue
@api.get("/catalogue")
def list_catalogue() -> Any:
	"""List catalogue items. GET /retail-omc/api/v1/catalogue"""
	recs = _run(_svc.list_catalogue_items(_tenant_id()))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@api.post("/catalogue")
def create_catalogue_item() -> Any:
	"""Create catalogue item. POST /retail-omc/api/v1/catalogue"""
	from .models import OmcCatalogueItemCreate
	body = request.get_json(force=True) or {}
	body["tenant_id"] = _tenant_id()
	try:
		return jsonify(_run(_svc.create_catalogue_item(OmcCatalogueItemCreate(**body))).model_dump()), 201
	except Exception as exc:
		return _err(str(exc))


@api.get("/catalogue/<item_id>")
def get_catalogue_item(item_id: str) -> Any:
	"""Get catalogue item. GET /retail-omc/api/v1/catalogue/<item_id>"""
	rec = _run(_svc.get_catalogue_item(_tenant_id(), item_id))
	return jsonify(rec.model_dump()) if rec else _err("not_found", 404)


@api.put("/catalogue/<item_id>/price")
def set_channel_price(item_id: str) -> Any:
	"""Set channel-specific price. PUT /retail-omc/api/v1/catalogue/<item_id>/price"""
	body = request.get_json(force=True) or {}
	try:
		rec = _run(_svc.set_channel_price(_tenant_id(), item_id, body["channel_id"], float(body["price"])))
		return jsonify(rec.model_dump()) if rec else _err("not_found", 404)
	except Exception as exc:
		return _err(str(exc))


# Inventory
@api.get("/inventory")
def get_inventory() -> Any:
	"""Get inventory. GET /retail-omc/api/v1/inventory?sku=<sku>&location_id=<id>"""
	sku = request.args.get("sku", "")
	if not sku:
		return _err("sku parameter required")
	recs = _run(_svc.get_inventory(_tenant_id(), sku, request.args.get("location_id")))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@api.post("/inventory")
def upsert_inventory() -> Any:
	"""Upsert inventory. POST /retail-omc/api/v1/inventory"""
	from .models import OmcInventoryRecord
	body = request.get_json(force=True) or {}
	body["tenant_id"] = _tenant_id()
	try:
		return jsonify(_run(_svc.upsert_inventory(OmcInventoryRecord(**body))).model_dump())
	except Exception as exc:
		return _err(str(exc))


@api.post("/inventory/reserve")
def reserve_inventory() -> Any:
	"""Reserve inventory. POST /retail-omc/api/v1/inventory/reserve"""
	body = request.get_json(force=True) or {}
	tid = _tenant_id()
	result = _run(_svc.reserve_inventory(tid, body.get("sku",""), body.get("location_id",""), body.get("channel_id",""), int(body.get("qty", 0))))
	return jsonify({"reserved": result})


# Orders
@api.get("/orders")
def list_orders() -> Any:
	"""List orders. GET /retail-omc/api/v1/orders?channel_id=<id>&status=<s>"""
	recs = _run(_svc.list_orders(_tenant_id(), request.args.get("channel_id"), request.args.get("status")))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@api.post("/orders")
def create_order() -> Any:
	"""Create order. POST /retail-omc/api/v1/orders"""
	from .models import OmcOrderCreate
	body = request.get_json(force=True) or {}
	body["tenant_id"] = _tenant_id()
	try:
		return jsonify(_run(_svc.create_order(OmcOrderCreate(**body))).model_dump()), 201
	except Exception as exc:
		return _err(str(exc))


@api.get("/orders/<order_id>")
def get_order(order_id: str) -> Any:
	"""Get order. GET /retail-omc/api/v1/orders/<order_id>"""
	rec = _run(_svc.get_order(_tenant_id(), order_id))
	return jsonify(rec.model_dump()) if rec else _err("not_found", 404)


@api.put("/orders/<order_id>")
def update_order(order_id: str) -> Any:
	"""Update order. PUT /retail-omc/api/v1/orders/<order_id>"""
	from .models import OmcOrderUpdate
	body = request.get_json(force=True) or {}
	try:
		rec = _run(_svc.update_order(_tenant_id(), order_id, OmcOrderUpdate(**body)))
		return jsonify(rec.model_dump()) if rec else _err("not_found", 404)
	except Exception as exc:
		return _err(str(exc))


@api.delete("/orders/<order_id>")
def cancel_order(order_id: str) -> Any:
	"""Cancel order. DELETE /retail-omc/api/v1/orders/<order_id>"""
	body = request.get_json(force=True) or {}
	rec = _run(_svc.cancel_order(_tenant_id(), order_id, body.get("reason",""), body.get("by","system")))
	return jsonify({"status": "cancelled"}) if rec else _err("not_found", 404)


# Returns
@api.get("/returns")
def list_returns() -> Any:
	"""List returns. GET /retail-omc/api/v1/returns?order_id=<id>"""
	recs = _run(_svc.list_returns(_tenant_id(), request.args.get("order_id")))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@api.post("/returns")
def initiate_return() -> Any:
	"""Initiate return. POST /retail-omc/api/v1/returns"""
	from .models import OmcReturnCreate
	body = request.get_json(force=True) or {}
	body["tenant_id"] = _tenant_id()
	try:
		return jsonify(_run(_svc.initiate_return(OmcReturnCreate(**body))).model_dump()), 201
	except Exception as exc:
		return _err(str(exc))


@api.get("/returns/<return_id>")
def get_return(return_id: str) -> Any:
	"""Get return. GET /retail-omc/api/v1/returns/<return_id>"""
	rec = _run(_svc.get_return(_tenant_id(), return_id))
	return jsonify(rec.model_dump()) if rec else _err("not_found", 404)


@api.put("/returns/<return_id>/approve")
def approve_return(return_id: str) -> Any:
	"""Approve return. PUT /retail-omc/api/v1/returns/<return_id>/approve"""
	body = request.get_json(force=True) or {}
	try:
		rec = _run(_svc.approve_return(_tenant_id(), return_id, float(body.get("refund_amount", 0)), body.get("by","system")))
		return jsonify(rec.model_dump()) if rec else _err("not_found", 404)
	except Exception as exc:
		return _err(str(exc))


# Pricing
@api.get("/pricing")
def list_pricing() -> Any:
	"""List pricing rules. GET /retail-omc/api/v1/pricing"""
	recs = _run(_svc.list_pricing_rules(_tenant_id(), request.args.get("channel_id")))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@api.post("/pricing")
def create_pricing() -> Any:
	"""Create pricing rule. POST /retail-omc/api/v1/pricing"""
	from .models import OmcPricingRuleCreate
	body = request.get_json(force=True) or {}
	body["tenant_id"] = _tenant_id()
	try:
		return jsonify(_run(_svc.create_pricing_rule(OmcPricingRuleCreate(**body))).model_dump()), 201
	except Exception as exc:
		return _err(str(exc))


@api.post("/pricing/apply")
def apply_pricing() -> Any:
	"""Apply pricing rules to a SKU. POST /retail-omc/api/v1/pricing/apply"""
	body = request.get_json(force=True) or {}
	price = _run(_svc.apply_pricing_rules(_tenant_id(), body.get("sku",""), float(body.get("base_price", 0)), body.get("channel_id","")))
	return jsonify({"applied_price": price})
