"""REST API Blueprint for APG Promotions Management."""

from __future__ import annotations

from typing import Any

from flask import Blueprint, g, jsonify, request

from .service import PrmService
from .capability_contract import get_capability_contract, evaluate_capability_rules

api = Blueprint("retail_prm_api", __name__, url_prefix="/retail-prm/api/v1")
_svc = PrmService()


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
	"""Return capability contract. GET /retail-prm/api/v1/contract"""
	return jsonify(get_capability_contract(_tenant_id()))


@api.post("/rules/evaluate")
def evaluate_rules() -> Any:
	"""Evaluate rules. POST /retail-prm/api/v1/rules/evaluate"""
	return jsonify(evaluate_capability_rules(request.get_json(force=True) or {}))


# Promotions
@api.get("/promotions")
def list_promotions() -> Any:
	"""List promotions. GET /retail-prm/api/v1/promotions?status=<s>"""
	recs = _run(_svc.list_promotions(_tenant_id(), request.args.get("status")))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@api.post("/promotions")
def create_promotion() -> Any:
	"""Create promotion. POST /retail-prm/api/v1/promotions"""
	from .models import PrmPromotionCreate
	body = request.get_json(force=True) or {}
	body["tenant_id"] = _tenant_id()
	try:
		return jsonify(_run(_svc.create_promotion(PrmPromotionCreate(**body))).model_dump()), 201
	except Exception as exc:
		return _err(str(exc))


@api.get("/promotions/<promotion_id>")
def get_promotion(promotion_id: str) -> Any:
	"""Get promotion summary. GET /retail-prm/api/v1/promotions/<promotion_id>"""
	summary = _run(_svc.promotion_summary(_tenant_id(), promotion_id))
	return jsonify(summary) if summary else _err("not_found", 404)


@api.put("/promotions/<promotion_id>")
def update_promotion(promotion_id: str) -> Any:
	"""Update promotion. PUT /retail-prm/api/v1/promotions/<promotion_id>"""
	from .models import PrmPromotionUpdate
	body = request.get_json(force=True) or {}
	try:
		rec = _run(_svc.update_promotion(_tenant_id(), promotion_id, PrmPromotionUpdate(**body)))
		return jsonify(rec.model_dump()) if rec else _err("not_found", 404)
	except Exception as exc:
		return _err(str(exc))


@api.post("/promotions/<promotion_id>/submit")
def submit_promotion(promotion_id: str) -> Any:
	"""Submit for approval. POST /retail-prm/api/v1/promotions/<promotion_id>/submit"""
	body = request.get_json(force=True) or {}
	try:
		rec = _run(_svc.submit_for_approval(_tenant_id(), promotion_id, body.get("by","system")))
		return jsonify(rec.model_dump()) if rec else _err("not_found", 404)
	except AssertionError as exc:
		return _err(str(exc), 422)


@api.post("/promotions/<promotion_id>/approve")
def approve_promotion(promotion_id: str) -> Any:
	"""Approve promotion. POST /retail-prm/api/v1/promotions/<promotion_id>/approve"""
	body = request.get_json(force=True) or {}
	try:
		rec = _run(_svc.approve_promotion(_tenant_id(), promotion_id, body.get("by","system")))
		return jsonify(rec.model_dump()) if rec else _err("not_found", 404)
	except AssertionError as exc:
		return _err(str(exc), 422)


@api.post("/promotions/<promotion_id>/activate")
def activate_promotion(promotion_id: str) -> Any:
	"""Activate promotion. POST /retail-prm/api/v1/promotions/<promotion_id>/activate"""
	try:
		rec = _run(_svc.activate_promotion(_tenant_id(), promotion_id))
		return jsonify(rec.model_dump()) if rec else _err("not_found", 404)
	except AssertionError as exc:
		return _err(str(exc), 422)


@api.post("/promotions/<promotion_id>/pause")
def pause_promotion(promotion_id: str) -> Any:
	"""Pause promotion. POST /retail-prm/api/v1/promotions/<promotion_id>/pause"""
	rec = _run(_svc.pause_promotion(_tenant_id(), promotion_id))
	return jsonify(rec.model_dump()) if rec else _err("not_found", 404)


@api.delete("/promotions/<promotion_id>")
def reject_promotion(promotion_id: str) -> Any:
	"""Reject promotion. DELETE /retail-prm/api/v1/promotions/<promotion_id>"""
	body = request.get_json(force=True) or {}
	try:
		rec = _run(_svc.reject_promotion(_tenant_id(), promotion_id, body.get("reason",""), body.get("by","system")))
		return jsonify({"status": "rejected"}) if rec else _err("not_found", 404)
	except Exception as exc:
		return _err(str(exc))


@api.post("/promotions/<promotion_id>/apply")
def apply_promotion(promotion_id: str) -> Any:
	"""Apply promotion to basket. POST /retail-prm/api/v1/promotions/<promotion_id>/apply"""
	body = request.get_json(force=True) or {}
	result = _run(_svc.apply_promotion(_tenant_id(), promotion_id, float(body.get("basket_value",0)), int(body.get("item_count",0))))
	return jsonify(result)


# Coupons
@api.get("/coupons")
def list_coupons() -> Any:
	"""List coupons. GET /retail-prm/api/v1/coupons?promotion_id=<id>"""
	recs = _run(_svc.list_coupons(_tenant_id(), request.args.get("promotion_id")))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@api.post("/coupons")
def create_coupon() -> Any:
	"""Create coupon. POST /retail-prm/api/v1/coupons"""
	from .models import PrmCouponCreate
	body = request.get_json(force=True) or {}
	body["tenant_id"] = _tenant_id()
	try:
		return jsonify(_run(_svc.create_coupon(PrmCouponCreate(**body))).model_dump()), 201
	except Exception as exc:
		return _err(str(exc))


@api.post("/coupons/redeem")
def redeem_coupon() -> Any:
	"""Redeem coupon. POST /retail-prm/api/v1/coupons/redeem"""
	from .models import PrmCouponRedemptionCreate
	body = request.get_json(force=True) or {}
	body["tenant_id"] = _tenant_id()
	try:
		return jsonify(_run(_svc.redeem_coupon(PrmCouponRedemptionCreate(**body))).model_dump()), 201
	except Exception as exc:
		return _err(str(exc))


# Pricing rules
@api.get("/pricing")
def list_pricing() -> Any:
	"""List pricing rules. GET /retail-prm/api/v1/pricing"""
	recs = _run(_svc.list_pricing_rules(_tenant_id()))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@api.post("/pricing")
def create_pricing() -> Any:
	"""Create pricing rule. POST /retail-prm/api/v1/pricing"""
	from .models import PrmPricingRuleCreate
	body = request.get_json(force=True) or {}
	body["tenant_id"] = _tenant_id()
	try:
		return jsonify(_run(_svc.create_pricing_rule(PrmPricingRuleCreate(**body))).model_dump()), 201
	except Exception as exc:
		return _err(str(exc))


# Markdown
@api.get("/markdown")
def list_markdowns() -> Any:
	"""List markdowns. GET /retail-prm/api/v1/markdown"""
	recs = _run(_svc.list_markdowns(_tenant_id(), request.args.get("type")))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@api.post("/markdown")
def create_markdown() -> Any:
	"""Create markdown. POST /retail-prm/api/v1/markdown"""
	from .models import PrmMarkdownCreate
	body = request.get_json(force=True) or {}
	body["tenant_id"] = _tenant_id()
	try:
		return jsonify(_run(_svc.create_markdown(PrmMarkdownCreate(**body))).model_dump()), 201
	except Exception as exc:
		return _err(str(exc))


@api.put("/markdown/<markdown_id>/approve")
def approve_markdown(markdown_id: str) -> Any:
	"""Approve markdown. PUT /retail-prm/api/v1/markdown/<markdown_id>/approve"""
	body = request.get_json(force=True) or {}
	rec = _run(_svc.approve_markdown(_tenant_id(), markdown_id, body.get("by","system")))
	return jsonify(rec.model_dump()) if rec else _err("not_found", 404)


# Effectiveness
@api.get("/effectiveness/<promotion_id>")
def get_effectiveness(promotion_id: str) -> Any:
	"""Get effectiveness history. GET /retail-prm/api/v1/effectiveness/<promotion_id>"""
	recs = _run(_svc.get_effectiveness(_tenant_id(), promotion_id))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@api.post("/effectiveness")
def record_effectiveness() -> Any:
	"""Record effectiveness. POST /retail-prm/api/v1/effectiveness"""
	from .models import PrmEffectivenessRecord
	body = request.get_json(force=True) or {}
	body["tenant_id"] = _tenant_id()
	try:
		return jsonify(_run(_svc.record_effectiveness(PrmEffectivenessRecord(**body))).model_dump()), 201
	except Exception as exc:
		return _err(str(exc))
