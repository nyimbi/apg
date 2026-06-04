"""Flask Blueprint views for Property Contracts (con)."""

from __future__ import annotations

import asyncio
from functools import wraps
from typing import Any, Callable

from flask import Blueprint, request, jsonify

from .service import ConService
from .models import (
	ContractCreate, ContractUpdate,
	ContractorCreate, ContractorUpdate,
	MilestoneCreate,
	VariationOrderCreate,
	DisputeCreate,
	RetentionCreate,
	ClauseCreate,
	ContractorGrade,
)

bp = Blueprint("con_views", __name__, url_prefix="/realestate/con")
_svc = ConService()


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
@has_access("realestate_con:view")
def dashboard():
	return _ok(_run(_svc.get_contract_summary(_tenant())))


# ── Contracts ─────────────────────────────────────────────────────────────────

@bp.get("/contracts")
@has_access("realestate_con:contracts")
def list_contracts():
	return _ok([c.model_dump() for c in _run(_svc.list_contracts(_tenant(), request.args.get("contract_type"), request.args.get("status")))])


@bp.post("/contracts")
@has_access("realestate_con:contracts")
def create_contract():
	try:
		return _ok(_run(_svc.create_contract(ContractCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.get("/contracts/<contract_id>")
@has_access("realestate_con:contracts")
def get_contract(contract_id: str):
	r = _run(_svc.get_contract(contract_id, _tenant()))
	return _ok(r.model_dump()) if r else _err("not found", 404)


@bp.put("/contracts/<contract_id>")
@has_access("realestate_con:contracts")
def update_contract(contract_id: str):
	try:
		r = _run(_svc.update_contract(contract_id, _tenant(), ContractUpdate(**request.json)))
		return _ok(r.model_dump()) if r else _err("not found", 404)
	except Exception as e:
		return _err(str(e))


@bp.post("/contracts/<contract_id>/execute")
@has_access("realestate_con:contracts")
def execute_contract(contract_id: str):
	try:
		r = _run(_svc.execute_contract(contract_id, _tenant()))
		return _ok(r.model_dump()) if r else _err("not found", 404)
	except Exception as e:
		return _err(str(e))


@bp.post("/contracts/<contract_id>/terminate")
@has_access("realestate_con:contracts")
def terminate_contract(contract_id: str):
	try:
		data = request.json
		r = _run(_svc.terminate_contract(contract_id, _tenant(), data["reason"], data.get("notice_period_satisfied", False)))
		return _ok(r.model_dump()) if r else _err("not found", 404)
	except Exception as e:
		return _err(str(e))


@bp.post("/contracts/<contract_id>/sign/<party_id>")
@has_access("realestate_con:contracts")
def sign_contract(contract_id: str, party_id: str):
	try:
		sig_ref = request.json.get("signature_ref", "")
		r = _run(_svc.sign_contract_party(contract_id, _tenant(), party_id, sig_ref))
		return _ok(r.model_dump()) if r else _err("not found", 404)
	except Exception as e:
		return _err(str(e))


@bp.get("/expiry")
@has_access("realestate_con:view")
def expiry_pipeline():
	days = int(request.args.get("days", 90))
	return _ok(_run(_svc.get_expiry_pipeline(_tenant(), days)))


# ── Contractors ───────────────────────────────────────────────────────────────

@bp.get("/contractors")
@has_access("realestate_con:contractors")
def list_contractors():
	return _ok([c.model_dump() for c in _run(_svc.list_contractors(_tenant(), request.args.get("grade")))])


@bp.post("/contractors")
@has_access("realestate_con:contractors")
def create_contractor():
	try:
		return _ok(_run(_svc.register_contractor(ContractorCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.get("/contractors/<contractor_id>")
@has_access("realestate_con:contractors")
def get_contractor(contractor_id: str):
	r = _run(_svc.get_contractor(contractor_id, _tenant()))
	return _ok(r.model_dump()) if r else _err("not found", 404)


@bp.post("/contractors/<contractor_id>/grade")
@has_access("realestate_con:contractors")
def grade_contractor(contractor_id: str):
	try:
		data = request.json
		r = _run(_svc.grade_contractor(contractor_id, _tenant(), ContractorGrade(data["grade"]), data.get("graded_by", "unknown")))
		return _ok(r.model_dump()) if r else _err("not found", 404)
	except Exception as e:
		return _err(str(e))


# ── Milestones ────────────────────────────────────────────────────────────────

@bp.get("/milestones")
@has_access("realestate_con:milestones")
def list_milestones():
	return _ok([m.model_dump() for m in _run(_svc.list_milestones(_tenant(), request.args.get("contract_id")))])


@bp.post("/milestones")
@has_access("realestate_con:milestones")
def create_milestone():
	try:
		return _ok(_run(_svc.create_milestone(MilestoneCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.post("/milestones/<milestone_id>/complete")
@has_access("realestate_con:milestones")
def complete_milestone(milestone_id: str):
	evidence_ids = request.json.get("evidence_ids", []) if request.json else []
	r = _run(_svc.complete_milestone(milestone_id, _tenant(), evidence_ids))
	return _ok(r.model_dump()) if r else _err("not found", 404)


# ── Variations ────────────────────────────────────────────────────────────────

@bp.get("/variations")
@has_access("realestate_con:variations")
def list_variations():
	return _ok([v.model_dump() for v in _run(_svc.list_variations(_tenant(), request.args.get("contract_id")))])


@bp.post("/variations")
@has_access("realestate_con:variations")
def raise_variation():
	try:
		return _ok(_run(_svc.raise_variation(VariationOrderCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.post("/variations/<vo_id>/approve")
@has_access("realestate_con:variations")
def approve_variation(vo_id: str):
	try:
		data = request.json
		r = _run(_svc.approve_variation(vo_id, _tenant(), data.get("approved_by", "unknown"), data.get("board_approval", False)))
		return _ok(r.model_dump()) if r else _err("not found", 404)
	except Exception as e:
		return _err(str(e))


# ── Disputes ──────────────────────────────────────────────────────────────────

@bp.get("/disputes")
@has_access("realestate_con:disputes")
def list_disputes():
	return _ok([d.model_dump() for d in _run(_svc.list_disputes(_tenant(), request.args.get("contract_id")))])


@bp.post("/disputes")
@has_access("realestate_con:disputes")
def raise_dispute():
	try:
		return _ok(_run(_svc.raise_dispute(DisputeCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.post("/disputes/<dispute_id>/resolve")
@has_access("realestate_con:disputes")
def resolve_dispute(dispute_id: str):
	summary = request.json.get("resolution_summary", "") if request.json else ""
	r = _run(_svc.resolve_dispute(dispute_id, _tenant(), summary))
	return _ok(r.model_dump()) if r else _err("not found", 404)


# ── Retention ─────────────────────────────────────────────────────────────────

@bp.post("/retention")
@has_access("realestate_con:retention")
def create_retention():
	try:
		return _ok(_run(_svc.create_retention(RetentionCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.post("/retention/<retention_id>/release")
@has_access("realestate_con:retention")
def release_retention(retention_id: str):
	try:
		data = request.json
		r = _run(_svc.release_retention(retention_id, _tenant(), data.get("approved_by", "unknown"), data.get("defect_liability_cleared", False)))
		return _ok(r.model_dump()) if r else _err("not found", 404)
	except Exception as e:
		return _err(str(e))


# ── Clause Library ────────────────────────────────────────────────────────────

@bp.get("/clauses")
@has_access("realestate_con:clauses")
def list_clauses():
	return _ok([c.model_dump() for c in _run(_svc.search_clauses(_tenant(), request.args.get("clause_type"), request.args.get("q")))])


@bp.post("/clauses")
@has_access("realestate_con:clauses")
def create_clause():
	try:
		return _ok(_run(_svc.create_clause(ClauseCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))
