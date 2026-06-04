"""Flask Blueprint views for Lease Management (lea)."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from functools import wraps
from typing import Any, Callable

from flask import Blueprint, request, jsonify

from .service import LeaService
from .models import (
	LeaseCreate, LeaseUpdate,
	LeaseAbstractionCreate,
	RentEscalationCreate,
	LeaseOptionCreate,
	RentReviewCreate,
	Ifrs16ScheduleCreate,
	LeaseAssignmentCreate,
	Ifrs16Category,
)

bp = Blueprint("lea_views", __name__, url_prefix="/realestate/lea")
_svc = LeaService()


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
@has_access("realestate_lea:view")
def dashboard():
	expiry = _run(_svc.get_expiry_pipeline(_tenant()))
	options = _run(_svc.get_expiring_options(_tenant()))
	return _ok({"expiry_pipeline": expiry, "expiring_options": [o.model_dump() for o in options]})


# ── Leases ────────────────────────────────────────────────────────────────────

@bp.get("/leases")
@has_access("realestate_lea:leases")
def list_leases():
	return _ok([l.model_dump() for l in _run(_svc.list_leases(_tenant(), request.args.get("property_id"), request.args.get("status")))])


@bp.post("/leases")
@has_access("realestate_lea:leases")
def create_lease():
	try:
		return _ok(_run(_svc.create_lease(LeaseCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.get("/leases/<lease_id>")
@has_access("realestate_lea:leases")
def get_lease(lease_id: str):
	r = _run(_svc.get_lease(lease_id, _tenant()))
	return _ok(r.model_dump()) if r else _err("not found", 404)


@bp.put("/leases/<lease_id>")
@has_access("realestate_lea:leases")
def update_lease(lease_id: str):
	try:
		r = _run(_svc.update_lease(lease_id, _tenant(), LeaseUpdate(**request.json)))
		return _ok(r.model_dump()) if r else _err("not found", 404)
	except Exception as e:
		return _err(str(e))


@bp.post("/leases/<lease_id>/activate")
@has_access("realestate_lea:leases")
def activate_lease(lease_id: str):
	try:
		r = _run(_svc.activate_lease(lease_id, _tenant()))
		return _ok(r.model_dump()) if r else _err("not found", 404)
	except Exception as e:
		return _err(str(e))


@bp.post("/leases/<lease_id>/surrender")
@has_access("realestate_lea:leases")
def surrender_lease(lease_id: str):
	try:
		surrendered_by = request.json.get("surrendered_by", "unknown")
		r = _run(_svc.surrender_lease(lease_id, _tenant(), surrendered_by))
		return _ok(r.model_dump()) if r else _err("not found", 404)
	except Exception as e:
		return _err(str(e))


# ── Abstraction ───────────────────────────────────────────────────────────────

@bp.post("/abstraction")
@has_access("realestate_lea:abstraction")
def create_abstraction():
	try:
		return _ok(_run(_svc.create_abstraction(LeaseAbstractionCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.post("/abstraction/<abstraction_id>/verify")
@has_access("realestate_lea:abstraction")
def verify_abstraction(abstraction_id: str):
	verified_by = request.json.get("verified_by", "unknown")
	r = _run(_svc.verify_abstraction(abstraction_id, _tenant(), verified_by))
	return _ok(r.model_dump()) if r else _err("not found", 404)


# ── Escalations ───────────────────────────────────────────────────────────────

@bp.get("/escalations")
@has_access("realestate_lea:escalations")
def list_escalations():
	return _ok([e.model_dump() for e in _run(_svc.list_escalations(_tenant(), request.args.get("lease_id")))])


@bp.post("/escalations")
@has_access("realestate_lea:escalations")
def create_escalation():
	try:
		return _ok(_run(_svc.create_escalation(RentEscalationCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.post("/escalations/<escalation_id>/apply")
@has_access("realestate_lea:escalations")
def apply_escalation(escalation_id: str):
	try:
		applied_by = request.json.get("applied_by", "unknown")
		r = _run(_svc.apply_escalation(escalation_id, _tenant(), applied_by))
		return _ok(r.model_dump()) if r else _err("not found or already applied", 404)
	except Exception as e:
		return _err(str(e))


# ── Options ───────────────────────────────────────────────────────────────────

@bp.post("/options")
@has_access("realestate_lea:options")
def create_option():
	try:
		return _ok(_run(_svc.create_option(LeaseOptionCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.post("/options/<option_id>/exercise")
@has_access("realestate_lea:options")
def exercise_option(option_id: str):
	try:
		notice_served = request.json.get("notice_served", False)
		r = _run(_svc.exercise_option(option_id, _tenant(), notice_served))
		return _ok(r.model_dump()) if r else _err("not found", 404)
	except Exception as e:
		return _err(str(e))


@bp.get("/options/expiring")
@has_access("realestate_lea:options")
def expiring_options():
	days = int(request.args.get("days", 180))
	return _ok([o.model_dump() for o in _run(_svc.get_expiring_options(_tenant(), days))])


# ── Rent Reviews ──────────────────────────────────────────────────────────────

@bp.post("/rent-reviews")
@has_access("realestate_lea:rent_reviews")
def commence_rent_review():
	try:
		return _ok(_run(_svc.commence_rent_review(RentReviewCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.post("/rent-reviews/<review_id>/agree")
@has_access("realestate_lea:rent_reviews")
def agree_rent_review(review_id: str):
	try:
		data = request.json
		r = _run(_svc.agree_rent_review(review_id, _tenant(), Decimal(str(data["agreed_rent"])), data.get("backdating_authorised_by")))
		return _ok(r.model_dump()) if r else _err("not found", 404)
	except Exception as e:
		return _err(str(e))


# ── IFRS 16 ───────────────────────────────────────────────────────────────────

@bp.post("/ifrs16")
@has_access("realestate_lea:ifrs16")
def generate_ifrs16():
	try:
		return _ok(_run(_svc.generate_ifrs16_schedule(Ifrs16ScheduleCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.post("/ifrs16/<schedule_id>/reclassify")
@has_access("realestate_lea:ifrs16")
def reclassify_ifrs16(schedule_id: str):
	try:
		data = request.json
		r = _run(_svc.reclassify_ifrs16(schedule_id, _tenant(), Ifrs16Category(data["new_category"]), data["auditor_approved_by"]))
		return _ok(r.model_dump()) if r else _err("not found", 404)
	except Exception as e:
		return _err(str(e))


# ── Assignments ───────────────────────────────────────────────────────────────

@bp.post("/assignments")
@has_access("realestate_lea:assignments")
def create_assignment():
	try:
		return _ok(_run(_svc.create_assignment(LeaseAssignmentCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.post("/assignments/<assignment_id>/complete")
@has_access("realestate_lea:assignments")
def complete_assignment(assignment_id: str):
	r = _run(_svc.complete_assignment(assignment_id, _tenant()))
	return _ok(r.model_dump()) if r else _err("not found", 404)


# ── Expiry Pipeline ───────────────────────────────────────────────────────────

@bp.get("/expiry")
@has_access("realestate_lea:view")
def expiry_pipeline():
	months = int(request.args.get("months", 12))
	return _ok(_run(_svc.get_expiry_pipeline(_tenant(), months)))
