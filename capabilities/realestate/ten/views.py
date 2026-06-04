"""Flask Blueprint views for Tenant Management (ten)."""

from __future__ import annotations

import asyncio
from functools import wraps
from typing import Any, Callable

from flask import Blueprint, request, jsonify

from .service import TenService
from .models import (
	TenantEntityCreate, TenantEntityUpdate,
	OnboardingStepRecord, OnboardingStep,
	ServiceRequestCreate, ServiceRequestUpdate,
	CommunicationCreate,
	SatisfactionSurveyCreate,
	TenantScoreCreate,
	TenantEscalationCreate,
	CreditGrade,
)

bp = Blueprint("ten_views", __name__, url_prefix="/realestate/ten")
_svc = TenService()


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
@has_access("realestate_ten:view")
def dashboard():
	return _ok(_run(_svc.get_tenant_summary(_tenant())))


# ── Tenants ───────────────────────────────────────────────────────────────────

@bp.get("/tenants")
@has_access("realestate_ten:tenants")
def list_tenants():
	return _ok([t.model_dump() for t in _run(_svc.list_tenants(_tenant(), request.args.get("status"), request.args.get("tenant_type")))])


@bp.post("/tenants")
@has_access("realestate_ten:tenants")
def create_tenant():
	try:
		return _ok(_run(_svc.register_tenant(TenantEntityCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.get("/tenants/<tenant_entity_id>")
@has_access("realestate_ten:tenants")
def get_tenant(tenant_entity_id: str):
	r = _run(_svc.get_tenant(tenant_entity_id, _tenant()))
	return _ok(r.model_dump()) if r else _err("not found", 404)


@bp.put("/tenants/<tenant_entity_id>")
@has_access("realestate_ten:tenants")
def update_tenant(tenant_entity_id: str):
	try:
		r = _run(_svc.update_tenant(tenant_entity_id, _tenant(), TenantEntityUpdate(**request.json)))
		return _ok(r.model_dump()) if r else _err("not found", 404)
	except Exception as e:
		return _err(str(e))


@bp.post("/tenants/<tenant_entity_id>/activate")
@has_access("realestate_ten:tenants")
def activate_tenant(tenant_entity_id: str):
	try:
		r = _run(_svc.activate_tenant(tenant_entity_id, _tenant()))
		return _ok(r.model_dump()) if r else _err("not found", 404)
	except Exception as e:
		return _err(str(e))


@bp.post("/tenants/<tenant_entity_id>/blacklist")
@has_access("realestate_ten:tenants")
def blacklist_tenant(tenant_entity_id: str):
	reason = request.json.get("reason", "") if request.json else ""
	r = _run(_svc.blacklist_tenant(tenant_entity_id, _tenant(), reason))
	return _ok(r.model_dump()) if r else _err("not found", 404)


@bp.post("/tenants/<tenant_entity_id>/grade")
@has_access("realestate_ten:scoring")
def assign_credit_grade(tenant_entity_id: str):
	try:
		grade = CreditGrade(request.json["grade"])
		r = _run(_svc.assign_credit_grade(tenant_entity_id, _tenant(), grade))
		return _ok(r.model_dump()) if r else _err("not found", 404)
	except Exception as e:
		return _err(str(e))


# ── Onboarding ────────────────────────────────────────────────────────────────

@bp.get("/onboarding/<tenant_entity_id>")
@has_access("realestate_ten:onboarding")
def onboarding_progress(tenant_entity_id: str):
	return _ok(_run(_svc.get_onboarding_progress(tenant_entity_id, _tenant())))


@bp.post("/onboarding")
@has_access("realestate_ten:onboarding")
def complete_onboarding_step():
	try:
		return _ok(_run(_svc.complete_onboarding_step(OnboardingStepRecord(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


# ── Service Requests ──────────────────────────────────────────────────────────

@bp.get("/service-requests")
@has_access("realestate_ten:service_requests")
def list_service_requests():
	return _ok([r.model_dump() for r in _run(_svc.list_service_requests(_tenant(), request.args.get("tenant_entity_id"), request.args.get("status")))])


@bp.post("/service-requests")
@has_access("realestate_ten:service_requests")
def raise_service_request():
	try:
		return _ok(_run(_svc.raise_service_request(ServiceRequestCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.get("/service-requests/<request_id>")
@has_access("realestate_ten:service_requests")
def get_service_request(request_id: str):
	r = _run(_svc.get_service_request(request_id, _tenant()))
	return _ok(r.model_dump()) if r else _err("not found", 404)


@bp.put("/service-requests/<request_id>")
@has_access("realestate_ten:service_requests")
def update_service_request(request_id: str):
	try:
		r = _run(_svc.update_service_request(request_id, _tenant(), ServiceRequestUpdate(**request.json)))
		return _ok(r.model_dump()) if r else _err("not found", 404)
	except Exception as e:
		return _err(str(e))


@bp.post("/service-requests/<request_id>/resolve")
@has_access("realestate_ten:service_requests")
def resolve_service_request(request_id: str):
	try:
		data = request.json
		r = _run(_svc.resolve_service_request(request_id, _tenant(), data.get("resolution_notes", ""), data.get("satisfaction_rating")))
		return _ok(r.model_dump()) if r else _err("not found", 404)
	except Exception as e:
		return _err(str(e))


# ── Communications ────────────────────────────────────────────────────────────

@bp.get("/communications")
@has_access("realestate_ten:communications")
def list_communications():
	return _ok([c.model_dump() for c in _run(_svc.list_communications(_tenant(), request.args.get("tenant_entity_id"), request.args.get("channel")))])


@bp.post("/communications")
@has_access("realestate_ten:communications")
def send_communication():
	try:
		return _ok(_run(_svc.send_communication(CommunicationCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


# ── Satisfaction ──────────────────────────────────────────────────────────────

@bp.get("/satisfaction")
@has_access("realestate_ten:satisfaction")
def list_satisfaction():
	return _ok([s.model_dump() for s in _run(_svc.list_satisfaction_surveys(_tenant(), request.args.get("tenant_entity_id")))])


@bp.post("/satisfaction")
@has_access("realestate_ten:satisfaction")
def record_satisfaction():
	try:
		return _ok(_run(_svc.record_satisfaction_survey(SatisfactionSurveyCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.get("/satisfaction/<tenant_entity_id>/trend")
@has_access("realestate_ten:satisfaction")
def satisfaction_trend(tenant_entity_id: str):
	return _ok(_run(_svc.get_satisfaction_trend(_tenant(), tenant_entity_id)))


# ── Scoring ───────────────────────────────────────────────────────────────────

@bp.post("/scoring")
@has_access("realestate_ten:scoring")
def calculate_score():
	try:
		return _ok(_run(_svc.calculate_tenant_score(TenantScoreCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


# ── Escalations ───────────────────────────────────────────────────────────────

@bp.get("/escalations")
@has_access("realestate_ten:escalations")
def list_escalations():
	return _ok([e.model_dump() for e in _run(_svc.list_escalations(_tenant(), request.args.get("tenant_entity_id")))])


@bp.post("/escalations")
@has_access("realestate_ten:escalations")
def raise_escalation():
	try:
		return _ok(_run(_svc.raise_escalation(TenantEscalationCreate(**request.json, tenant_id=_tenant()))).model_dump(), 201)
	except Exception as e:
		return _err(str(e))


@bp.post("/escalations/<escalation_id>/resolve")
@has_access("realestate_ten:escalations")
def resolve_escalation(escalation_id: str):
	notes = request.json.get("resolution_notes", "") if request.json else ""
	r = _run(_svc.resolve_escalation(escalation_id, _tenant(), notes))
	return _ok(r.model_dump()) if r else _err("not found", 404)


# ── Retention ─────────────────────────────────────────────────────────────────

@bp.get("/retention/at-risk")
@has_access("realestate_ten:retention")
def retention_at_risk():
	return _ok([t.model_dump() for t in _run(_svc.get_retention_at_risk(_tenant()))])
