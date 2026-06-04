"""Flask Blueprint views for Property Valuation (val)."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from functools import wraps
from typing import Any, Callable

from flask import Blueprint, request, jsonify

from .service import ValService
from .models import (
	ValuerCreate,
	ComparableCreate,
	ValuationCreate, ValuationUpdate,
	DcfModelCreate,
	ValuationRollEntryCreate,
	MassAppraisalRunCreate,
	ValuationChallengeCreate,
)

bp = Blueprint("val_views", __name__, url_prefix="/realestate/val")
_svc = ValService()


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
@has_access("realestate_val:view")
def dashboard():
	return _ok(_run(_svc.get_valuation_summary(_tenant())))


# ── Valuers ───────────────────────────────────────────────────────────────────

@bp.get("/valuers")
@has_access("realestate_val:valuers")
def list_valuers():
	independent = request.args.get("independent", "false").lower() == "true"
	return _ok([v.model_dump() for v in _run(_svc.list_valuers(_tenant(), request.args.get("grade"), independent))])


@bp.post("/valuers")
@has_access("realestate_val:valuers")
def create_valuer():
	try:
		return _ok(_run(_svc.register_valuer(ValuerCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.get("/valuers/<valuer_id>")
@has_access("realestate_val:valuers")
def get_valuer(valuer_id: str):
	r = _run(_svc.get_valuer(valuer_id, _tenant()))
	return _ok(r.model_dump()) if r else _err("not found", 404)


# ── Comparables ───────────────────────────────────────────────────────────────

@bp.get("/comparables")
@has_access("realestate_val:comparables")
def list_comparables():
	verified = request.args.get("verified_only", "false").lower() == "true"
	return _ok([c.model_dump() for c in _run(_svc.list_comparables(_tenant(), request.args.get("comparable_type"), verified))])


@bp.post("/comparables")
@has_access("realestate_val:comparables")
def add_comparable():
	try:
		return _ok(_run(_svc.add_comparable(ComparableCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.post("/comparables/<comparable_id>/verify")
@has_access("realestate_val:comparables")
def verify_comparable(comparable_id: str):
	verified_by = request.json.get("verified_by", "unknown") if request.json else "unknown"
	r = _run(_svc.verify_comparable(comparable_id, _tenant(), verified_by))
	return _ok(r.model_dump()) if r else _err("not found", 404)


# ── Valuations ────────────────────────────────────────────────────────────────

@bp.get("/valuations")
@has_access("realestate_val:valuations")
def list_valuations():
	return _ok([v.model_dump() for v in _run(_svc.list_valuations(_tenant(), request.args.get("property_id"), request.args.get("status")))])


@bp.post("/valuations")
@has_access("realestate_val:valuations")
def instruct_valuation():
	try:
		return _ok(_run(_svc.instruct_valuation(ValuationCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.get("/valuations/<valuation_id>")
@has_access("realestate_val:valuations")
def get_valuation(valuation_id: str):
	r = _run(_svc.get_valuation(valuation_id, _tenant()))
	return _ok(r.model_dump()) if r else _err("not found", 404)


@bp.put("/valuations/<valuation_id>")
@has_access("realestate_val:valuations")
def update_valuation(valuation_id: str):
	try:
		r = _run(_svc.update_valuation(valuation_id, _tenant(), ValuationUpdate(**request.json)))
		return _ok(r.model_dump()) if r else _err("not found", 404)
	except Exception as e:
		return _err(str(e))


@bp.post("/valuations/<valuation_id>/sign-off")
@has_access("realestate_val:valuations")
def sign_off_valuation(valuation_id: str):
	try:
		data = request.json
		r = _run(_svc.sign_off_valuation(valuation_id, _tenant(), data["signed_off_by"], data["valuer_grade"]))
		return _ok(r.model_dump()) if r else _err("not found", 404)
	except Exception as e:
		return _err(str(e))


@bp.post("/valuations/<valuation_id>/publish")
@has_access("realestate_val:valuations")
def publish_valuation(valuation_id: str):
	try:
		r = _run(_svc.publish_valuation(valuation_id, _tenant()))
		return _ok(r.model_dump()) if r else _err("not found", 404)
	except Exception as e:
		return _err(str(e))


# ── DCF ───────────────────────────────────────────────────────────────────────

@bp.post("/dcf")
@has_access("realestate_val:dcf")
def run_dcf():
	try:
		return _ok(_run(_svc.run_dcf_model(DcfModelCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.get("/dcf/<model_id>")
@has_access("realestate_val:dcf")
def get_dcf(model_id: str):
	r = _run(_svc.get_dcf_model(model_id, _tenant()))
	return _ok(r.model_dump()) if r else _err("not found", 404)


# ── Mass Appraisal ────────────────────────────────────────────────────────────

@bp.post("/mass-appraisal")
@has_access("realestate_val:mass_appraisal")
def run_mass_appraisal():
	try:
		return _ok(_run(_svc.run_mass_appraisal(MassAppraisalRunCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.get("/mass-appraisal/<run_id>")
@has_access("realestate_val:mass_appraisal")
def get_mass_appraisal(run_id: str):
	r = _run(_svc.get_mass_appraisal_run(run_id, _tenant()))
	return _ok(r.model_dump()) if r else _err("not found", 404)


# ── Valuation Roll ────────────────────────────────────────────────────────────

@bp.get("/roll")
@has_access("realestate_val:roll")
def valuation_roll():
	return _ok([e.model_dump() for e in _run(_svc.get_valuation_roll(_tenant(), request.args.get("property_id")))])


@bp.post("/roll")
@has_access("realestate_val:roll")
def add_to_roll():
	try:
		return _ok(_run(_svc.add_to_valuation_roll(ValuationRollEntryCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


# ── Yield Analysis ────────────────────────────────────────────────────────────

@bp.get("/yields/<property_id>")
@has_access("realestate_val:yields")
def calculate_yield(property_id: str):
	try:
		annual_rent = Decimal(request.args.get("annual_rent", "0"))
		purchase_price = Decimal(request.args.get("purchase_price", "1"))
		yield_type = request.args.get("yield_type", "net_initial_yield")
		return _ok(_run(_svc.calculate_yield(_tenant(), property_id, annual_rent, purchase_price, yield_type)))
	except Exception as e:
		return _err(str(e))


# ── Challenges ────────────────────────────────────────────────────────────────

@bp.get("/challenges")
@has_access("realestate_val:challenges")
def list_challenges():
	return _ok([c.model_dump() for c in _run(_svc.list_challenges(_tenant(), request.args.get("valuation_id")))])


@bp.post("/challenges")
@has_access("realestate_val:challenges")
def raise_challenge():
	try:
		return _ok(_run(_svc.raise_challenge(ValuationChallengeCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.post("/challenges/<challenge_id>/resolve")
@has_access("realestate_val:challenges")
def resolve_challenge(challenge_id: str):
	try:
		data = request.json
		r = _run(_svc.resolve_challenge(challenge_id, _tenant(), data.get("upheld", False), data.get("resolution_notes", ""), data.get("reviewed_by", "unknown")))
		return _ok(r.model_dump()) if r else _err("not found", 404)
	except Exception as e:
		return _err(str(e))
