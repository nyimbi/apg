"""Flask Blueprint views for Rental Operations (ren)."""

from __future__ import annotations

import asyncio
from functools import wraps
from typing import Any, Callable

from flask import Blueprint, request, jsonify

from .service import RenService
from .models import (
	TenancyCreate, TenancyUpdate,
	RentPaymentCreate,
	ArrearsRecordCreate,
	DepositCreate, DepositDeductionCreate,
	NoticeCreate,
	TenancyRenewalCreate,
	ReferencingCreate,
)

bp = Blueprint("ren_views", __name__, url_prefix="/realestate/ren")
_svc = RenService()


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
@has_access("realestate_ren:view")
def dashboard():
	rent_roll = _run(_svc.generate_rent_roll(_tenant()))
	arrears = _run(_svc.get_arrears_report(_tenant()))
	renewal_pipeline = _run(_svc.get_renewal_pipeline(_tenant()))
	return _ok({"rent_roll_count": len(rent_roll), "active_arrears": len(arrears), "renewals_due_90_days": len(renewal_pipeline)})


# ── Tenancies ─────────────────────────────────────────────────────────────────

@bp.get("/tenancies")
@has_access("realestate_ren:tenancies")
def list_tenancies():
	return _ok([t.model_dump() for t in _run(_svc.list_tenancies(_tenant(), request.args.get("unit_id"), request.args.get("status")))])


@bp.post("/tenancies")
@has_access("realestate_ren:tenancies")
def create_tenancy():
	try:
		return _ok(_run(_svc.create_tenancy(TenancyCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.get("/tenancies/<tenancy_id>")
@has_access("realestate_ren:tenancies")
def get_tenancy(tenancy_id: str):
	r = _run(_svc.get_tenancy(tenancy_id, _tenant()))
	return _ok(r.model_dump()) if r else _err("not found", 404)


@bp.put("/tenancies/<tenancy_id>")
@has_access("realestate_ren:tenancies")
def update_tenancy(tenancy_id: str):
	try:
		r = _run(_svc.update_tenancy(tenancy_id, _tenant(), TenancyUpdate(**request.json)))
		return _ok(r.model_dump()) if r else _err("not found", 404)
	except Exception as e:
		return _err(str(e))


@bp.post("/tenancies/<tenancy_id>/activate")
@has_access("realestate_ren:tenancies")
def activate_tenancy(tenancy_id: str):
	try:
		r = _run(_svc.activate_tenancy(tenancy_id, _tenant()))
		return _ok(r.model_dump()) if r else _err("not found", 404)
	except Exception as e:
		return _err(str(e))


# ── Rent Collection ───────────────────────────────────────────────────────────

@bp.get("/rent-collection")
@has_access("realestate_ren:rent_collection")
def list_payments():
	return _ok([p.model_dump() for p in _run(_svc.list_payments(_tenant(), request.args.get("tenancy_id"), request.args.get("period")))])


@bp.post("/rent-collection")
@has_access("realestate_ren:rent_collection")
def record_payment():
	try:
		return _ok(_run(_svc.record_rent_payment(RentPaymentCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


# ── Arrears ───────────────────────────────────────────────────────────────────

@bp.get("/arrears")
@has_access("realestate_ren:arrears")
def list_arrears():
	return _ok([a.model_dump() for a in _run(_svc.get_arrears_report(_tenant()))])


@bp.post("/arrears/<arrears_id>/legal")
@has_access("realestate_ren:legal")
def escalate_to_legal(arrears_id: str):
	try:
		r = _run(_svc.escalate_arrears_to_legal(arrears_id, _tenant()))
		return _ok(r.model_dump()) if r else _err("not found", 404)
	except Exception as e:
		return _err(str(e))


# ── Deposits ──────────────────────────────────────────────────────────────────

@bp.post("/deposits")
@has_access("realestate_ren:deposits")
def register_deposit():
	try:
		return _ok(_run(_svc.register_deposit(DepositCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.get("/deposits/<deposit_id>")
@has_access("realestate_ren:deposits")
def get_deposit(deposit_id: str):
	r = _run(_svc.get_deposit(deposit_id, _tenant()))
	return _ok(r.model_dump()) if r else _err("not found", 404)


@bp.post("/deposits/<deposit_id>/deduct")
@has_access("realestate_ren:deposits")
def deduct_deposit(deposit_id: str):
	try:
		payload = DepositDeductionCreate(**request.json, tenant_id=_tenant(), deposit_id=deposit_id)
		return _ok(_run(_svc.deduct_from_deposit(payload)).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.post("/deposits/<deposit_id>/release")
@has_access("realestate_ren:deposits")
def release_deposit(deposit_id: str):
	released_by = request.json.get("released_by", "unknown")
	r = _run(_svc.release_deposit(deposit_id, _tenant(), released_by))
	return _ok(r.model_dump()) if r else _err("not found", 404)


# ── Notices ───────────────────────────────────────────────────────────────────

@bp.get("/notices")
@has_access("realestate_ren:notices")
def list_notices():
	return _ok([n.model_dump() for n in _run(_svc.list_notices(_tenant(), request.args.get("tenancy_id")))])


@bp.post("/notices")
@has_access("realestate_ren:notices")
def serve_notice():
	try:
		return _ok(_run(_svc.serve_notice(NoticeCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


# ── Renewals ──────────────────────────────────────────────────────────────────

@bp.post("/renewals")
@has_access("realestate_ren:renewals")
def initiate_renewal():
	try:
		return _ok(_run(_svc.initiate_renewal(TenancyRenewalCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.post("/renewals/<renewal_id>/accept")
@has_access("realestate_ren:renewals")
def accept_renewal(renewal_id: str):
	r = _run(_svc.accept_renewal(renewal_id, _tenant()))
	return _ok(r.model_dump()) if r else _err("not found", 404)


@bp.get("/renewals/pipeline")
@has_access("realestate_ren:renewals")
def renewal_pipeline():
	months = int(request.args.get("months", 3))
	return _ok(_run(_svc.get_renewal_pipeline(_tenant(), months)))


# ── Referencing ───────────────────────────────────────────────────────────────

@bp.post("/referencing")
@has_access("realestate_ren:referencing")
def run_referencing():
	try:
		return _ok(_run(_svc.run_referencing(ReferencingCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.post("/referencing/<ref_id>/complete")
@has_access("realestate_ren:referencing")
def complete_referencing(ref_id: str):
	try:
		data = request.json
		r = _run(_svc.complete_referencing(ref_id, _tenant(), data.get("passed", False), data.get("results", {})))
		return _ok(r.model_dump()) if r else _err("not found", 404)
	except Exception as e:
		return _err(str(e))


# ── Rent Roll ─────────────────────────────────────────────────────────────────

@bp.get("/rent-roll")
@has_access("realestate_ren:rent_roll")
def rent_roll():
	return _ok(_run(_svc.generate_rent_roll(_tenant(), request.args.get("property_id"))))
