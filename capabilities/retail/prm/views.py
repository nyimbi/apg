"""Flask Blueprint views for APG Promotions Management."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

from flask import Blueprint, g, jsonify, request

from .service import PrmService

bp = Blueprint("retail_prm_views", __name__, url_prefix="/retail-prm")
_svc = PrmService()


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
@has_access("retail_prm:view")
def dashboard() -> Any:
	tid = _tenant_id()
	active = _run(_svc.list_promotions(tid, "active"))
	pending = _run(_svc.list_promotions(tid, "pending_review"))
	return jsonify({
		"tenant_id": tid,
		"active_promotions": len(active),
		"pending_approval": len(pending),
	})


@bp.get("/promotions")
@has_access("retail_prm:view")
def list_promotions() -> Any:
	tid = _tenant_id()
	status = request.args.get("status")
	recs = _run(_svc.list_promotions(tid, status))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@bp.post("/promotions")
@has_access("retail_prm:write")
def create_promotion() -> Any:
	from .models import PrmPromotionCreate
	tid = _tenant_id()
	body = request.get_json(force=True) or {}
	body["tenant_id"] = tid
	try:
		rec = _run(_svc.create_promotion(PrmPromotionCreate(**body)))
		return jsonify(rec.model_dump()), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/promotions/<promotion_id>")
@has_access("retail_prm:view")
def promotion_detail(promotion_id: str) -> Any:
	tid = _tenant_id()
	summary = _run(_svc.promotion_summary(tid, promotion_id))
	if not summary:
		return jsonify({"error": "not_found"}), 404
	return jsonify(summary)


@bp.put("/promotions/<promotion_id>")
@has_access("retail_prm:write")
def update_promotion(promotion_id: str) -> Any:
	from .models import PrmPromotionUpdate
	tid = _tenant_id()
	body = request.get_json(force=True) or {}
	try:
		rec = _run(_svc.update_promotion(tid, promotion_id, PrmPromotionUpdate(**body)))
		if rec is None:
			return jsonify({"error": "not_found"}), 404
		return jsonify(rec.model_dump())
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/promotions/<promotion_id>/submit")
@has_access("retail_prm:write")
def submit_promotion(promotion_id: str) -> Any:
	tid = _tenant_id()
	body = request.get_json(force=True) or {}
	try:
		rec = _run(_svc.submit_for_approval(tid, promotion_id, body.get("by", "system")))
		if rec is None:
			return jsonify({"error": "not_found"}), 404
		return jsonify(rec.model_dump())
	except AssertionError as exc:
		return jsonify({"error": str(exc)}), 422


@bp.post("/promotions/<promotion_id>/approve")
@has_access("retail_prm:approve")
def approve_promotion(promotion_id: str) -> Any:
	tid = _tenant_id()
	body = request.get_json(force=True) or {}
	try:
		rec = _run(_svc.approve_promotion(tid, promotion_id, body.get("by", "system")))
		if rec is None:
			return jsonify({"error": "not_found"}), 404
		return jsonify(rec.model_dump())
	except AssertionError as exc:
		return jsonify({"error": str(exc)}), 422


@bp.post("/promotions/<promotion_id>/activate")
@has_access("retail_prm:approve")
def activate_promotion(promotion_id: str) -> Any:
	tid = _tenant_id()
	try:
		rec = _run(_svc.activate_promotion(tid, promotion_id))
		if rec is None:
			return jsonify({"error": "not_found"}), 404
		return jsonify(rec.model_dump())
	except AssertionError as exc:
		return jsonify({"error": str(exc)}), 422


@bp.post("/promotions/<promotion_id>/pause")
@has_access("retail_prm:write")
def pause_promotion(promotion_id: str) -> Any:
	tid = _tenant_id()
	rec = _run(_svc.pause_promotion(tid, promotion_id))
	if rec is None:
		return jsonify({"error": "not_found"}), 404
	return jsonify(rec.model_dump())


@bp.post("/promotions/<promotion_id>/apply")
@has_access("retail_prm:write")
def apply_promotion(promotion_id: str) -> Any:
	tid = _tenant_id()
	body = request.get_json(force=True) or {}
	result = _run(_svc.apply_promotion(tid, promotion_id, float(body.get("basket_value", 0)), int(body.get("item_count", 0))))
	return jsonify(result)


@bp.get("/coupons")
@has_access("retail_prm:view")
def list_coupons() -> Any:
	tid = _tenant_id()
	promotion_id = request.args.get("promotion_id")
	recs = _run(_svc.list_coupons(tid, promotion_id))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@bp.post("/coupons")
@has_access("retail_prm:write")
def create_coupon() -> Any:
	from .models import PrmCouponCreate
	tid = _tenant_id()
	body = request.get_json(force=True) or {}
	body["tenant_id"] = tid
	try:
		rec = _run(_svc.create_coupon(PrmCouponCreate(**body)))
		return jsonify(rec.model_dump()), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/coupons/redeem")
@has_access("retail_prm:write")
def redeem_coupon() -> Any:
	from .models import PrmCouponRedemptionCreate
	tid = _tenant_id()
	body = request.get_json(force=True) or {}
	body["tenant_id"] = tid
	try:
		rec = _run(_svc.redeem_coupon(PrmCouponRedemptionCreate(**body)))
		return jsonify(rec.model_dump()), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/markdown")
@has_access("retail_prm:view")
def list_markdowns() -> Any:
	tid = _tenant_id()
	markdown_type = request.args.get("type")
	recs = _run(_svc.list_markdowns(tid, markdown_type))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@bp.post("/markdown")
@has_access("retail_prm:write")
def create_markdown() -> Any:
	from .models import PrmMarkdownCreate
	tid = _tenant_id()
	body = request.get_json(force=True) or {}
	body["tenant_id"] = tid
	try:
		rec = _run(_svc.create_markdown(PrmMarkdownCreate(**body)))
		return jsonify(rec.model_dump()), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/effectiveness/<promotion_id>")
@has_access("retail_prm:view")
def effectiveness(promotion_id: str) -> Any:
	tid = _tenant_id()
	recs = _run(_svc.get_effectiveness(tid, promotion_id))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@bp.get("/pricing")
@has_access("retail_prm:admin")
def list_pricing_rules() -> Any:
	tid = _tenant_id()
	recs = _run(_svc.list_pricing_rules(tid))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@bp.post("/pricing")
@has_access("retail_prm:admin")
def create_pricing_rule() -> Any:
	from .models import PrmPricingRuleCreate
	tid = _tenant_id()
	body = request.get_json(force=True) or {}
	body["tenant_id"] = tid
	try:
		rec = _run(_svc.create_pricing_rule(PrmPricingRuleCreate(**body)))
		return jsonify(rec.model_dump()), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400
