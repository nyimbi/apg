"""Flask Blueprint views for APG Loyalty & Rewards."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

from flask import Blueprint, g, jsonify, request

from .service import LoyService

bp = Blueprint("retail_loy_views", __name__, url_prefix="/retail-loy")
_svc = LoyService()


# ------------------------------------------------------------------
# Auth/tenant helpers
# ------------------------------------------------------------------

def _tenant_id() -> str:
	return getattr(g, "tenant_id", request.headers.get("X-Tenant-ID", "default"))


def has_access(permission: str) -> Callable:
	"""Decorator that enforces permission check via g.permissions set."""
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
	"""Run an async coroutine from a sync Flask view."""
	import asyncio
	loop = asyncio.new_event_loop()
	try:
		return loop.run_until_complete(coro)
	finally:
		loop.close()


# ------------------------------------------------------------------
# Dashboard
# ------------------------------------------------------------------

@bp.get("/dashboard")
@has_access("retail_loy:view")
def dashboard() -> Any:
	"""Loyalty programme dashboard."""
	tid = _tenant_id()
	programmes = _run(_svc.list_programmes(tid))
	members = _run(_svc.list_members(tid))
	return jsonify({
		"tenant_id": tid,
		"programme_count": len(programmes),
		"member_count": len(members),
		"programmes": [p.model_dump() for p in programmes],
	})


# ------------------------------------------------------------------
# Programme views
# ------------------------------------------------------------------

@bp.get("/programmes")
@has_access("retail_loy:view")
def list_programmes() -> Any:
	tid = _tenant_id()
	recs = _run(_svc.list_programmes(tid))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@bp.post("/programmes")
@has_access("retail_loy:write")
def create_programme() -> Any:
	from .models import LoyProgrammeCreate
	tid = _tenant_id()
	body = request.get_json(force=True) or {}
	body["tenant_id"] = tid
	try:
		rec = _run(_svc.create_programme(LoyProgrammeCreate(**body)))
		return jsonify(rec.model_dump()), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


# ------------------------------------------------------------------
# Member views
# ------------------------------------------------------------------

@bp.get("/members")
@has_access("retail_loy:view")
def list_members() -> Any:
	tid = _tenant_id()
	programme_id = request.args.get("programme_id")
	recs = _run(_svc.list_members(tid, programme_id))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@bp.post("/members/enrol")
@has_access("retail_loy:write")
def enrol_member() -> Any:
	from .models import LoyMemberCreate
	tid = _tenant_id()
	body = request.get_json(force=True) or {}
	body["tenant_id"] = tid
	try:
		rec = _run(_svc.enrol_member(LoyMemberCreate(**body)))
		return jsonify(rec.model_dump()), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.get("/members/<member_id>")
@has_access("retail_loy:view")
def member_detail(member_id: str) -> Any:
	tid = _tenant_id()
	summary = _run(_svc.get_member_summary(tid, member_id))
	if not summary:
		return jsonify({"error": "not_found"}), 404
	return jsonify(summary)


@bp.put("/members/<member_id>")
@has_access("retail_loy:write")
def update_member(member_id: str) -> Any:
	from .models import LoyMemberUpdate
	tid = _tenant_id()
	body = request.get_json(force=True) or {}
	try:
		rec = _run(_svc.update_member(tid, member_id, LoyMemberUpdate(**body)))
		if rec is None:
			return jsonify({"error": "not_found"}), 404
		return jsonify(rec.model_dump())
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/members/<member_id>/freeze")
@has_access("retail_loy:write")
def freeze_member(member_id: str) -> Any:
	tid = _tenant_id()
	body = request.get_json(force=True) or {}
	rec = _run(_svc.freeze_member(tid, member_id, body.get("reason", ""), body.get("by", "system")))
	if rec is None:
		return jsonify({"error": "not_found"}), 404
	return jsonify(rec.model_dump())


@bp.post("/members/<member_id>/reactivate")
@has_access("retail_loy:write")
def reactivate_member(member_id: str) -> Any:
	tid = _tenant_id()
	body = request.get_json(force=True) or {}
	rec = _run(_svc.reactivate_member(tid, member_id, body.get("by", "system")))
	if rec is None:
		return jsonify({"error": "not_found"}), 404
	return jsonify(rec.model_dump())


# ------------------------------------------------------------------
# Transactions
# ------------------------------------------------------------------

@bp.get("/members/<member_id>/transactions")
@has_access("retail_loy:view")
def member_transactions(member_id: str) -> Any:
	tid = _tenant_id()
	limit = int(request.args.get("limit", 50))
	recs = _run(_svc.get_transaction_history(tid, member_id, limit))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@bp.post("/earn")
@has_access("retail_loy:write")
def earn_points() -> Any:
	from .models import LoyTransactionCreate
	tid = _tenant_id()
	body = request.get_json(force=True) or {}
	body.setdefault("tenant_id", tid)
	body.setdefault("transaction_type", "earn")
	try:
		rec = _run(_svc.earn_points(LoyTransactionCreate(**body)))
		return jsonify(rec.model_dump()), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/redeem")
@has_access("retail_loy:write")
def redeem_points() -> Any:
	from .models import LoyTransactionCreate
	tid = _tenant_id()
	body = request.get_json(force=True) or {}
	body.setdefault("tenant_id", tid)
	body.setdefault("transaction_type", "redeem")
	try:
		rec = _run(_svc.redeem_points(LoyTransactionCreate(**body)))
		return jsonify(rec.model_dump()), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


# ------------------------------------------------------------------
# Tiers
# ------------------------------------------------------------------

@bp.get("/tiers")
@has_access("retail_loy:view")
def list_tiers() -> Any:
	tid = _tenant_id()
	programme_id = request.args.get("programme_id", "")
	recs = _run(_svc.list_tiers(tid, programme_id))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@bp.post("/tiers")
@has_access("retail_loy:admin")
def create_tier() -> Any:
	from .models import LoyTierCreate
	tid = _tenant_id()
	body = request.get_json(force=True) or {}
	body["tenant_id"] = tid
	try:
		rec = _run(_svc.create_tier(LoyTierCreate(**body)))
		return jsonify(rec.model_dump()), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


# ------------------------------------------------------------------
# Campaigns
# ------------------------------------------------------------------

@bp.get("/campaigns")
@has_access("retail_loy:view")
def list_campaigns() -> Any:
	tid = _tenant_id()
	programme_id = request.args.get("programme_id")
	recs = _run(_svc.list_campaigns(tid, programme_id))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@bp.post("/campaigns")
@has_access("retail_loy:write")
def create_campaign() -> Any:
	from .models import LoyCampaignCreate
	tid = _tenant_id()
	body = request.get_json(force=True) or {}
	body["tenant_id"] = tid
	try:
		rec = _run(_svc.create_campaign(LoyCampaignCreate(**body)))
		return jsonify(rec.model_dump()), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/campaigns/<campaign_id>/approve")
@has_access("retail_loy:admin")
def approve_campaign(campaign_id: str) -> Any:
	tid = _tenant_id()
	body = request.get_json(force=True) or {}
	rec = _run(_svc.approve_campaign(tid, campaign_id, body.get("by", "system")))
	if rec is None:
		return jsonify({"error": "not_found"}), 404
	return jsonify(rec.model_dump())


@bp.post("/campaigns/<campaign_id>/activate")
@has_access("retail_loy:admin")
def activate_campaign(campaign_id: str) -> Any:
	tid = _tenant_id()
	try:
		rec = _run(_svc.activate_campaign(tid, campaign_id))
		if rec is None:
			return jsonify({"error": "not_found"}), 404
		return jsonify(rec.model_dump())
	except AssertionError as exc:
		return jsonify({"error": str(exc)}), 422


# ------------------------------------------------------------------
# Rewards
# ------------------------------------------------------------------

@bp.get("/rewards")
@has_access("retail_loy:view")
def list_rewards() -> Any:
	tid = _tenant_id()
	programme_id = request.args.get("programme_id", "")
	recs = _run(_svc.list_rewards(tid, programme_id))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@bp.post("/rewards")
@has_access("retail_loy:write")
def create_reward() -> Any:
	from .models import LoyRewardCreate
	tid = _tenant_id()
	body = request.get_json(force=True) or {}
	body["tenant_id"] = tid
	try:
		rec = _run(_svc.create_reward(LoyRewardCreate(**body)))
		return jsonify(rec.model_dump()), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400


# ------------------------------------------------------------------
# CLV
# ------------------------------------------------------------------

@bp.get("/clv/<member_id>")
@has_access("retail_loy:view")
def clv_segment(member_id: str) -> Any:
	tid = _tenant_id()
	rec = _run(_svc.get_clv_segment(tid, member_id))
	if rec is None:
		return jsonify({"error": "not_found"}), 404
	return jsonify(rec.model_dump())


@bp.post("/clv")
@has_access("retail_loy:write")
def record_clv() -> Any:
	from .models import LoyClvSegmentRecord
	tid = _tenant_id()
	body = request.get_json(force=True) or {}
	body["tenant_id"] = tid
	try:
		rec = _run(_svc.record_clv_segment(LoyClvSegmentRecord(**body)))
		return jsonify(rec.model_dump()), 201
	except Exception as exc:
		return jsonify({"error": str(exc)}), 400
