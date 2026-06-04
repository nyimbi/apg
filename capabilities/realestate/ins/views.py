"""Flask Blueprint views for Property Insurance (ins)."""

from __future__ import annotations

import asyncio
from functools import wraps
from decimal import Decimal
from typing import Any, Callable

from flask import Blueprint, request, jsonify

from .service import InsService
from .models import (
	InsurerCreate,
	PolicyCreate, PolicyUpdate,
	InsuredAssetCreate,
	ClaimCreate,
	EndorsementCreate,
	PremiumAllocationCreate,
	CoverageGapCreate,
)

bp = Blueprint("ins_views", __name__, url_prefix="/realestate/ins")
_svc = InsService()


def _run(coro):
	return asyncio.get_event_loop().run_until_complete(coro)


def has_access(permission: str):
	def decorator(fn: Callable) -> Callable:
		@wraps(fn)
		def wrapper(*args, **kwargs):
			return fn(*args, **kwargs)
		return wrapper
	return decorator


def _tenant() -> str:
	return request.headers.get("X-Tenant-ID", "default")


def _ok(data: Any, status: int = 200):
	return jsonify({"status": "ok", "data": data}), status


def _err(msg: str, status: int = 400):
	return jsonify({"status": "error", "message": msg}), status


# ── Dashboard ─────────────────────────────────────────────────────────────────

@bp.get("/dashboard")
@has_access("realestate_ins:view")
def dashboard():
	return _ok(_run(_svc.get_insurance_summary(_tenant())))


# ── Insurers ──────────────────────────────────────────────────────────────────

@bp.get("/insurers")
@has_access("realestate_ins:insurers")
def list_insurers():
	return _ok([i.model_dump() for i in _run(_svc.list_insurers(_tenant(), request.args.get("grade")))])


@bp.post("/insurers")
@has_access("realestate_ins:insurers")
def create_insurer():
	try:
		return _ok(_run(_svc.register_insurer(InsurerCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.get("/insurers/<insurer_id>")
@has_access("realestate_ins:insurers")
def get_insurer(insurer_id: str):
	r = _run(_svc.get_insurer(insurer_id, _tenant()))
	return _ok(r.model_dump()) if r else _err("not found", 404)


# ── Policies ──────────────────────────────────────────────────────────────────

@bp.get("/policies")
@has_access("realestate_ins:policies")
def list_policies():
	return _ok([p.model_dump() for p in _run(_svc.list_policies(_tenant(), request.args.get("property_id"), request.args.get("status")))])


@bp.post("/policies")
@has_access("realestate_ins:policies")
def create_policy():
	try:
		return _ok(_run(_svc.create_policy(PolicyCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.get("/policies/<policy_id>")
@has_access("realestate_ins:policies")
def get_policy(policy_id: str):
	r = _run(_svc.get_policy(policy_id, _tenant()))
	return _ok(r.model_dump()) if r else _err("not found", 404)


@bp.put("/policies/<policy_id>")
@has_access("realestate_ins:policies")
def update_policy(policy_id: str):
	try:
		r = _run(_svc.update_policy(policy_id, _tenant(), PolicyUpdate(**request.json)))
		return _ok(r.model_dump()) if r else _err("not found", 404)
	except Exception as e:
		return _err(str(e))


@bp.post("/policies/<policy_id>/bind")
@has_access("realestate_ins:policies")
def bind_policy(policy_id: str):
	try:
		r = _run(_svc.bind_policy(policy_id, _tenant()))
		return _ok(r.model_dump()) if r else _err("not found", 404)
	except Exception as e:
		return _err(str(e))


@bp.get("/renewals")
@has_access("realestate_ins:renewals")
def renewal_pipeline():
	days = int(request.args.get("days", 90))
	return _ok(_run(_svc.get_renewal_pipeline(_tenant(), days)))


# ── Asset Schedule ────────────────────────────────────────────────────────────

@bp.get("/assets")
@has_access("realestate_ins:assets")
def list_assets():
	policy_id = request.args.get("policy_id", "")
	if not policy_id:
		return _err("policy_id required")
	return _ok([a.model_dump() for a in _run(_svc.list_policy_assets(_tenant(), policy_id))])


@bp.post("/assets")
@has_access("realestate_ins:assets")
def add_asset():
	try:
		return _ok(_run(_svc.add_asset_to_schedule(InsuredAssetCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.delete("/assets/<asset_id>")
@has_access("realestate_ins:assets")
def remove_asset(asset_id: str):
	removed = _run(_svc.remove_asset_from_schedule(asset_id, _tenant()))
	return _ok({"removed": removed})


# ── Claims ────────────────────────────────────────────────────────────────────

@bp.get("/claims")
@has_access("realestate_ins:claims")
def list_claims():
	return _ok([c.model_dump() for c in _run(_svc.list_claims(_tenant(), request.args.get("policy_id"), request.args.get("status")))])


@bp.post("/claims")
@has_access("realestate_ins:claims")
def lodge_claim():
	try:
		return _ok(_run(_svc.lodge_claim(ClaimCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.get("/claims/<claim_id>")
@has_access("realestate_ins:claims")
def get_claim(claim_id: str):
	r = _run(_svc.get_claim(claim_id, _tenant()))
	return _ok(r.model_dump()) if r else _err("not found", 404)


@bp.post("/claims/<claim_id>/approve")
@has_access("realestate_ins:claims")
def approve_claim(claim_id: str):
	try:
		data = request.json
		r = _run(_svc.approve_claim(claim_id, _tenant(), Decimal(str(data["approved_value"])), data.get("senior_approved", False)))
		return _ok(r.model_dump()) if r else _err("not found", 404)
	except Exception as e:
		return _err(str(e))


@bp.post("/claims/<claim_id>/settle")
@has_access("realestate_ins:claims")
def settle_claim(claim_id: str):
	try:
		r = _run(_svc.settle_claim(claim_id, _tenant(), Decimal(str(request.json["settlement_amount"]))))
		return _ok(r.model_dump()) if r else _err("not found", 404)
	except Exception as e:
		return _err(str(e))


# ── Endorsements ──────────────────────────────────────────────────────────────

@bp.get("/endorsements")
@has_access("realestate_ins:endorsements")
def list_endorsements():
	return _ok([e.model_dump() for e in _run(_svc.list_endorsements(_tenant(), request.args.get("policy_id")))])


@bp.post("/endorsements")
@has_access("realestate_ins:endorsements")
def issue_endorsement():
	try:
		return _ok(_run(_svc.issue_endorsement(EndorsementCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


# ── Premium Allocation ────────────────────────────────────────────────────────

@bp.post("/premiums")
@has_access("realestate_ins:premiums")
def allocate_premium():
	try:
		return _ok(_run(_svc.allocate_premium(PremiumAllocationCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


# ── Coverage Gaps ─────────────────────────────────────────────────────────────

@bp.get("/gaps")
@has_access("realestate_ins:gaps")
def list_gaps():
	return _ok([g.model_dump() for g in _run(_svc.list_coverage_gaps(_tenant(), request.args.get("property_id")))])


@bp.post("/gaps/detect/<property_id>")
@has_access("realestate_ins:gaps")
def detect_gaps(property_id: str):
	return _ok([g.model_dump() for g in _run(_svc.detect_coverage_gaps(_tenant(), property_id))])
