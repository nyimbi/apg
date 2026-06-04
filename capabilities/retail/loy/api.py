"""REST API Blueprint for APG Loyalty & Rewards."""

from __future__ import annotations

from typing import Any

from flask import Blueprint, g, jsonify, request

from .service import LoyService
from .capability_contract import get_capability_contract, evaluate_capability_rules

api = Blueprint("retail_loy_api", __name__, url_prefix="/retail-loy/api/v1")
_svc = LoyService()


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


# ------------------------------------------------------------------
# Contract
# ------------------------------------------------------------------

@api.get("/contract")
def contract() -> Any:
	"""Return the capability contract for this tenant.

	GET /retail-loy/api/v1/contract
	"""
	return jsonify(get_capability_contract(_tenant_id()))


@api.post("/rules/evaluate")
def evaluate_rules() -> Any:
	"""Evaluate capability rules against a context payload.

	POST /retail-loy/api/v1/rules/evaluate
	Body: {context object}
	"""
	ctx = request.get_json(force=True) or {}
	return jsonify(evaluate_capability_rules(ctx))


# ------------------------------------------------------------------
# Programmes
# ------------------------------------------------------------------

@api.get("/programmes")
def list_programmes() -> Any:
	"""List loyalty programmes.

	GET /retail-loy/api/v1/programmes
	"""
	recs = _run(_svc.list_programmes(_tenant_id()))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@api.post("/programmes")
def create_programme() -> Any:
	"""Create a loyalty programme.

	POST /retail-loy/api/v1/programmes
	Body: LoyProgrammeCreate
	"""
	from .models import LoyProgrammeCreate
	body = request.get_json(force=True) or {}
	body["tenant_id"] = _tenant_id()
	try:
		rec = _run(_svc.create_programme(LoyProgrammeCreate(**body)))
		return jsonify(rec.model_dump()), 201
	except Exception as exc:
		return _err(str(exc))


@api.get("/programmes/<programme_id>")
def get_programme(programme_id: str) -> Any:
	"""Get a programme by ID.

	GET /retail-loy/api/v1/programmes/<programme_id>
	"""
	rec = _run(_svc.get_programme(_tenant_id(), programme_id))
	if rec is None:
		return _err("not_found", 404)
	return jsonify(rec.model_dump())


# ------------------------------------------------------------------
# Members
# ------------------------------------------------------------------

@api.get("/members")
def list_members() -> Any:
	"""List members, optionally filtered by programme_id.

	GET /retail-loy/api/v1/members?programme_id=<id>
	"""
	recs = _run(_svc.list_members(_tenant_id(), request.args.get("programme_id")))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@api.post("/members")
def enrol_member() -> Any:
	"""Enrol a new member.

	POST /retail-loy/api/v1/members
	Body: LoyMemberCreate
	"""
	from .models import LoyMemberCreate
	body = request.get_json(force=True) or {}
	body["tenant_id"] = _tenant_id()
	try:
		rec = _run(_svc.enrol_member(LoyMemberCreate(**body)))
		return jsonify(rec.model_dump()), 201
	except Exception as exc:
		return _err(str(exc))


@api.get("/members/<member_id>")
def get_member(member_id: str) -> Any:
	"""Get member summary including balance and CLV.

	GET /retail-loy/api/v1/members/<member_id>
	"""
	summary = _run(_svc.get_member_summary(_tenant_id(), member_id))
	if not summary:
		return _err("not_found", 404)
	return jsonify(summary)


@api.put("/members/<member_id>")
def update_member(member_id: str) -> Any:
	"""Update member profile.

	PUT /retail-loy/api/v1/members/<member_id>
	Body: LoyMemberUpdate
	"""
	from .models import LoyMemberUpdate
	body = request.get_json(force=True) or {}
	try:
		rec = _run(_svc.update_member(_tenant_id(), member_id, LoyMemberUpdate(**body)))
		if rec is None:
			return _err("not_found", 404)
		return jsonify(rec.model_dump())
	except Exception as exc:
		return _err(str(exc))


@api.delete("/members/<member_id>")
def deactivate_member(member_id: str) -> Any:
	"""Deactivate (soft-delete) a member.

	DELETE /retail-loy/api/v1/members/<member_id>
	"""
	from .models import LoyMemberUpdate
	body = request.get_json(force=True) or {}
	rec = _run(_svc.update_member(_tenant_id(), member_id, LoyMemberUpdate(status="inactive", updated_by=body.get("by", "system"))))
	if rec is None:
		return _err("not_found", 404)
	return jsonify({"status": "deactivated", "member_id": member_id})


# ------------------------------------------------------------------
# Transactions
# ------------------------------------------------------------------

@api.get("/members/<member_id>/transactions")
def member_transactions(member_id: str) -> Any:
	"""Get transaction ledger for a member.

	GET /retail-loy/api/v1/members/<member_id>/transactions?limit=50
	"""
	limit = int(request.args.get("limit", 50))
	recs = _run(_svc.get_transaction_history(_tenant_id(), member_id, limit))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@api.post("/transactions/earn")
def earn_points() -> Any:
	"""Post a points earn transaction.

	POST /retail-loy/api/v1/transactions/earn
	Body: LoyTransactionCreate (transaction_type=earn)
	"""
	from .models import LoyTransactionCreate
	body = request.get_json(force=True) or {}
	body.setdefault("tenant_id", _tenant_id())
	body["transaction_type"] = "earn"
	try:
		rec = _run(_svc.earn_points(LoyTransactionCreate(**body)))
		return jsonify(rec.model_dump()), 201
	except Exception as exc:
		return _err(str(exc))


@api.post("/transactions/redeem")
def redeem_points() -> Any:
	"""Post a points redemption transaction.

	POST /retail-loy/api/v1/transactions/redeem
	Body: LoyTransactionCreate (transaction_type=redeem, points negative)
	"""
	from .models import LoyTransactionCreate
	body = request.get_json(force=True) or {}
	body.setdefault("tenant_id", _tenant_id())
	body["transaction_type"] = "redeem"
	try:
		rec = _run(_svc.redeem_points(LoyTransactionCreate(**body)))
		return jsonify(rec.model_dump()), 201
	except Exception as exc:
		return _err(str(exc))


@api.post("/transactions/adjust")
def adjust_points() -> Any:
	"""Post an administrative points adjustment.

	POST /retail-loy/api/v1/transactions/adjust
	Body: LoyTransactionCreate (transaction_type=adjust)
	"""
	from .models import LoyTransactionCreate
	body = request.get_json(force=True) or {}
	body.setdefault("tenant_id", _tenant_id())
	body["transaction_type"] = "adjust"
	try:
		rec = _run(_svc.adjust_points(LoyTransactionCreate(**body)))
		return jsonify(rec.model_dump()), 201
	except Exception as exc:
		return _err(str(exc))


# ------------------------------------------------------------------
# Tiers
# ------------------------------------------------------------------

@api.get("/tiers")
def list_tiers() -> Any:
	"""List programme tiers.

	GET /retail-loy/api/v1/tiers?programme_id=<id>
	"""
	programme_id = request.args.get("programme_id", "")
	recs = _run(_svc.list_tiers(_tenant_id(), programme_id))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@api.post("/tiers")
def create_tier() -> Any:
	"""Create a programme tier.

	POST /retail-loy/api/v1/tiers
	Body: LoyTierCreate
	"""
	from .models import LoyTierCreate
	body = request.get_json(force=True) or {}
	body["tenant_id"] = _tenant_id()
	try:
		rec = _run(_svc.create_tier(LoyTierCreate(**body)))
		return jsonify(rec.model_dump()), 201
	except Exception as exc:
		return _err(str(exc))


# ------------------------------------------------------------------
# Campaigns
# ------------------------------------------------------------------

@api.get("/campaigns")
def list_campaigns() -> Any:
	"""List campaigns.

	GET /retail-loy/api/v1/campaigns?programme_id=<id>
	"""
	recs = _run(_svc.list_campaigns(_tenant_id(), request.args.get("programme_id")))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@api.post("/campaigns")
def create_campaign() -> Any:
	"""Create a campaign.

	POST /retail-loy/api/v1/campaigns
	Body: LoyCampaignCreate
	"""
	from .models import LoyCampaignCreate
	body = request.get_json(force=True) or {}
	body["tenant_id"] = _tenant_id()
	try:
		rec = _run(_svc.create_campaign(LoyCampaignCreate(**body)))
		return jsonify(rec.model_dump()), 201
	except Exception as exc:
		return _err(str(exc))


@api.post("/campaigns/<campaign_id>/approve")
def approve_campaign(campaign_id: str) -> Any:
	"""Approve a campaign for activation.

	POST /retail-loy/api/v1/campaigns/<campaign_id>/approve
	"""
	body = request.get_json(force=True) or {}
	rec = _run(_svc.approve_campaign(_tenant_id(), campaign_id, body.get("by", "system")))
	if rec is None:
		return _err("not_found", 404)
	return jsonify(rec.model_dump())


@api.post("/campaigns/<campaign_id>/activate")
def activate_campaign(campaign_id: str) -> Any:
	"""Activate an approved campaign.

	POST /retail-loy/api/v1/campaigns/<campaign_id>/activate
	"""
	try:
		rec = _run(_svc.activate_campaign(_tenant_id(), campaign_id))
		if rec is None:
			return _err("not_found", 404)
		return jsonify(rec.model_dump())
	except AssertionError as exc:
		return _err(str(exc), 422)


# ------------------------------------------------------------------
# Partners
# ------------------------------------------------------------------

@api.get("/partners")
def list_partners() -> Any:
	"""List coalition partners.

	GET /retail-loy/api/v1/partners?programme_id=<id>
	"""
	programme_id = request.args.get("programme_id", "")
	recs = _run(_svc.list_partners(_tenant_id(), programme_id))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@api.post("/partners")
def register_partner() -> Any:
	"""Register a coalition partner.

	POST /retail-loy/api/v1/partners
	Body: LoyPartnerCreate
	"""
	from .models import LoyPartnerCreate
	body = request.get_json(force=True) or {}
	body["tenant_id"] = _tenant_id()
	try:
		rec = _run(_svc.register_partner(LoyPartnerCreate(**body)))
		return jsonify(rec.model_dump()), 201
	except Exception as exc:
		return _err(str(exc))


# ------------------------------------------------------------------
# Rewards
# ------------------------------------------------------------------

@api.get("/rewards")
def list_rewards() -> Any:
	"""List available rewards.

	GET /retail-loy/api/v1/rewards?programme_id=<id>
	"""
	programme_id = request.args.get("programme_id", "")
	recs = _run(_svc.list_rewards(_tenant_id(), programme_id))
	return jsonify({"items": [r.model_dump() for r in recs], "count": len(recs)})


@api.post("/rewards")
def create_reward() -> Any:
	"""Add a reward to the catalogue.

	POST /retail-loy/api/v1/rewards
	Body: LoyRewardCreate
	"""
	from .models import LoyRewardCreate
	body = request.get_json(force=True) or {}
	body["tenant_id"] = _tenant_id()
	try:
		rec = _run(_svc.create_reward(LoyRewardCreate(**body)))
		return jsonify(rec.model_dump()), 201
	except Exception as exc:
		return _err(str(exc))


# ------------------------------------------------------------------
# CLV
# ------------------------------------------------------------------

@api.get("/clv/<member_id>")
def get_clv(member_id: str) -> Any:
	"""Get CLV segment for a member.

	GET /retail-loy/api/v1/clv/<member_id>
	"""
	rec = _run(_svc.get_clv_segment(_tenant_id(), member_id))
	if rec is None:
		return _err("not_found", 404)
	return jsonify(rec.model_dump())


@api.post("/clv")
def record_clv() -> Any:
	"""Record a CLV segment calculation.

	POST /retail-loy/api/v1/clv
	Body: LoyClvSegmentRecord
	"""
	from .models import LoyClvSegmentRecord
	body = request.get_json(force=True) or {}
	body["tenant_id"] = _tenant_id()
	try:
		rec = _run(_svc.record_clv_segment(LoyClvSegmentRecord(**body)))
		return jsonify(rec.model_dump()), 201
	except Exception as exc:
		return _err(str(exc))


# ------------------------------------------------------------------
# Expiry
# ------------------------------------------------------------------

@api.post("/expiry/run")
def run_expiry() -> Any:
	"""Run points expiry for a programme (dry_run optional).

	POST /retail-loy/api/v1/expiry/run
	Body: {programme_id, dry_run}
	"""
	body = request.get_json(force=True) or {}
	programme_id = body.get("programme_id", "")
	dry_run = bool(body.get("dry_run", True))
	result = _run(_svc.expire_points(_tenant_id(), programme_id, dry_run))
	return jsonify(result)
