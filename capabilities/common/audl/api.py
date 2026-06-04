"""
APG Audit Logging — REST API (Flask Blueprint).

URL prefix: /api/audl/v1

All endpoints are synchronous wrappers around async service methods,
using asyncio.run() per request.  In production, mount this blueprint
inside a Quart or ASGI-wrapped Flask app for native async support.

© 2025 Datacraft  www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from flask import Blueprint, current_app, g, jsonify, request
from pydantic import ValidationError

from .models import (
	AuditEventCreate,
	AuditQueryCreate,
	AuditTrailCreate,
	AuditTrailUpdate,
	ComplianceFramework,
	ComplianceReportCreate,
	DataSubjectRequestCreate,
	DataSubjectRequestUpdate,
	EvidencePackageCreate,
	RetentionPolicyCreate,
	RetentionPolicyUpdate,
	TamperDetectionCreate,
	uuid7str,
)
from .service import AuditLoggingService

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Blueprint
# ---------------------------------------------------------------------------

audl_bp = Blueprint("audl", __name__, url_prefix="/api/audl/v1")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _svc() -> AuditLoggingService:
	"""
	Resolve the AuditLoggingService for the current request.

	Reads ``tenant_id`` and ``actor_id`` from Flask ``g`` (set by your auth
	middleware).  Falls back to header values for development convenience.
	"""
	tenant_id = getattr(g, "tenant_id", None) or request.headers.get("X-Tenant-Id", "default")
	actor_id  = getattr(g, "actor_id",  None) or request.headers.get("X-Actor-Id",  "anonymous")
	db        = getattr(g, "db_session", None)   # None → service uses in-memory store
	return AuditLoggingService(db_session=db, tenant_id=tenant_id, actor_id=actor_id)


def _run(coro):
	"""Execute an async coroutine from a sync Flask view."""
	try:
		loop = asyncio.get_event_loop()
		if loop.is_running():
			# Inside an async server (Quart / ASGI) — schedule and wait
			import concurrent.futures
			with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
				fut = pool.submit(asyncio.run, coro)
				return fut.result()
		return asyncio.run(coro)
	except RuntimeError:
		return asyncio.run(coro)


def _ok(data: Any, status: int = 200):
	if hasattr(data, "model_dump"):
		return jsonify(data.model_dump(mode="json")), status
	if isinstance(data, list) and data and hasattr(data[0], "model_dump"):
		return jsonify([d.model_dump(mode="json") for d in data]), status
	return jsonify(data), status


def _err(msg: str, status: int = 400):
	return jsonify({"error": msg}), status


def _parse(model_cls):
	"""Parse JSON body into a Pydantic model; abort 400 on failure."""
	body = request.get_json(silent=True) or {}
	# Inject tenant / actor from context if not in body
	tenant_id = getattr(g, "tenant_id", None) or request.headers.get("X-Tenant-Id", "default")
	body.setdefault("tenant_id", tenant_id)
	try:
		return model_cls(**body)
	except ValidationError as exc:
		raise _ValidationError(exc) from exc


class _ValidationError(Exception):
	def __init__(self, exc: ValidationError):
		self.exc = exc


@audl_bp.errorhandler(_ValidationError)
def _handle_validation(exc):
	return _err(str(exc.exc), 422)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@audl_bp.get("/health")
def health():
	"""Service liveness check."""
	return _ok({"status": "ok", "capability": "audl", "ts": datetime.now(timezone.utc).isoformat()})


# ---------------------------------------------------------------------------
# AuditEvent  — POST /events, GET /events, GET /events/<id>
# ---------------------------------------------------------------------------

@audl_bp.post("/events")
def log_event():
	"""
	Log a single audit event.

	Body: AuditEventCreate JSON.
	Returns the created AuditEventResponse.
	"""
	try:
		payload = _parse(AuditEventCreate)
	except _ValidationError as exc:
		return _err(str(exc.exc), 422)

	svc = _svc()
	result = _run(svc.log_event(
		who     = payload.actor_id or svc.actor_id,
		what    = payload.action,
		on_what = payload.resource_id or "",
		how     = payload.event_type,
		where   = payload.ip_address,
		when    = None,
		result  = payload.success,
		payload = payload,
	))
	return _ok(result, 201)


@audl_bp.post("/events/batch")
def log_event_batch():
	"""
	Write a batch of audit events atomically (max 10 000 per call).

	Body: { "events": [ AuditEventCreate, ... ] }
	"""
	body = request.get_json(silent=True) or {}
	raw_events = body.get("events", [])
	if not raw_events:
		return _err("events array must be non-empty", 400)
	tenant_id = getattr(g, "tenant_id", None) or request.headers.get("X-Tenant-Id", "default")
	try:
		events = [AuditEventCreate(**{**e, "tenant_id": tenant_id}) for e in raw_events]
	except ValidationError as exc:
		return _err(str(exc), 422)

	svc = _svc()
	results = _run(svc.immutable_log_write(events))
	return _ok([r.model_dump(mode="json") for r in results], 201)


@audl_bp.get("/events/<event_id>")
def get_event(event_id: str):
	"""Retrieve a single audit event by ID."""
	svc = _svc()
	ev  = svc._events.get(event_id)
	if ev is None or ev.tenant_id != svc.tenant_id:
		return _err("event not found", 404)
	return _ok(ev)


# ---------------------------------------------------------------------------
# AuditQuery / search  — POST /search
# ---------------------------------------------------------------------------

@audl_bp.post("/search")
def search_events():
	"""
	Structured / NLP audit log search.

	Body: AuditQueryCreate JSON.
	Returns AuditSearchResult.
	"""
	try:
		q = _parse(AuditQueryCreate)
	except _ValidationError as exc:
		return _err(str(exc.exc), 422)

	svc    = _svc()
	result = _run(svc.audit_trail_search(q))
	return _ok(result)


@audl_bp.get("/queries")
def list_queries():
	"""List saved audit queries for this tenant."""
	svc = _svc()
	return _ok(_run(svc.list_queries()))


@audl_bp.get("/queries/<query_id>")
def get_query(query_id: str):
	"""Retrieve a saved query by ID."""
	svc = _svc()
	try:
		return _ok(_run(svc.get_query(query_id)))
	except KeyError:
		return _err("query not found", 404)


# ---------------------------------------------------------------------------
# AuditTrail  — CRUD
# ---------------------------------------------------------------------------

@audl_bp.post("/trails")
def create_trail():
	try:
		req = _parse(AuditTrailCreate)
	except _ValidationError as exc:
		return _err(str(exc.exc), 422)
	svc = _svc()
	return _ok(_run(svc.create_trail(req)), 201)


@audl_bp.get("/trails")
def list_trails():
	svc = _svc()
	active_only = request.args.get("active_only", "true").lower() != "false"
	return _ok(_run(svc.list_trails(active_only=active_only)))


@audl_bp.get("/trails/<trail_id>")
def get_trail(trail_id: str):
	svc = _svc()
	try:
		return _ok(_run(svc.get_trail(trail_id)))
	except KeyError:
		return _err("trail not found", 404)


@audl_bp.put("/trails/<trail_id>")
def update_trail(trail_id: str):
	body = request.get_json(silent=True) or {}
	try:
		upd = AuditTrailUpdate(**body)
	except ValidationError as exc:
		return _err(str(exc), 422)
	svc = _svc()
	try:
		return _ok(_run(svc.update_trail(trail_id, upd)))
	except KeyError:
		return _err("trail not found", 404)


@audl_bp.delete("/trails/<trail_id>")
def delete_trail(trail_id: str):
	svc = _svc()
	try:
		_run(svc.delete_trail(trail_id))
		return _ok({"deleted": True})
	except KeyError:
		return _err("trail not found", 404)


# ---------------------------------------------------------------------------
# ComplianceReport
# ---------------------------------------------------------------------------

@audl_bp.post("/compliance/reports")
def create_compliance_report():
	"""
	Generate a compliance report for a framework and time window.

	Body: ComplianceReportCreate JSON.
	"""
	try:
		req = _parse(ComplianceReportCreate)
	except _ValidationError as exc:
		return _err(str(exc.exc), 422)
	svc = _svc()
	return _ok(_run(svc.compliance_report(req)), 202)


@audl_bp.get("/compliance/reports")
def list_compliance_reports():
	svc = _svc()
	reports = [r for r in svc._reports.values() if r.tenant_id == svc.tenant_id]
	return _ok([r.model_dump(mode="json") for r in reports])


@audl_bp.get("/compliance/reports/<report_id>")
def get_compliance_report(report_id: str):
	svc = _svc()
	rec = svc._reports.get(report_id)
	if rec is None or rec.tenant_id != svc.tenant_id:
		return _err("report not found", 404)
	return _ok(rec)


# ---------------------------------------------------------------------------
# RetentionPolicy
# ---------------------------------------------------------------------------

@audl_bp.post("/retention-policies")
def create_retention_policy():
	try:
		req = _parse(RetentionPolicyCreate)
	except _ValidationError as exc:
		return _err(str(exc.exc), 422)
	svc = _svc()
	return _ok(_run(svc.create_retention_policy(req)), 201)


@audl_bp.get("/retention-policies")
def list_retention_policies():
	svc = _svc()
	return _ok(_run(svc.list_retention_policies()))


@audl_bp.put("/retention-policies/<policy_id>")
def update_retention_policy(policy_id: str):
	body = request.get_json(silent=True) or {}
	try:
		upd = RetentionPolicyUpdate(**body)
	except ValidationError as exc:
		return _err(str(exc), 422)
	svc = _svc()
	try:
		return _ok(_run(svc.update_retention_policy(policy_id, upd)))
	except KeyError:
		return _err("policy not found", 404)


@audl_bp.delete("/retention-policies/<policy_id>")
def delete_retention_policy(policy_id: str):
	svc = _svc()
	try:
		_run(svc.delete_retention_policy(policy_id))
		return _ok({"deleted": True})
	except KeyError:
		return _err("policy not found", 404)


@audl_bp.post("/retention-policies/enforce")
def enforce_retention():
	"""Run retention enforcement for the current tenant."""
	svc = _svc()
	return _ok(_run(svc.retention_enforcement()))


# ---------------------------------------------------------------------------
# DataSubjectRequest (GDPR)
# ---------------------------------------------------------------------------

@audl_bp.post("/dsr")
def create_dsr():
	"""
	Submit a data subject request (access, erasure, portability, etc.).

	Body: DataSubjectRequestCreate JSON.
	"""
	try:
		req = _parse(DataSubjectRequestCreate)
	except _ValidationError as exc:
		return _err(str(exc.exc), 422)
	is_admin = request.headers.get("X-Admin", "false").lower() == "true"
	svc      = _svc()
	return _ok(_run(svc.create_dsr(req, is_admin=is_admin)), 201)


@audl_bp.get("/dsr")
def list_dsrs():
	svc = _svc()
	return _ok(_run(svc.list_dsrs()))


@audl_bp.get("/dsr/<dsr_id>")
def get_dsr(dsr_id: str):
	svc = _svc()
	rec = svc._dsrs.get(dsr_id)
	if rec is None or rec.tenant_id != svc.tenant_id:
		return _err("DSR not found", 404)
	return _ok(rec)


@audl_bp.put("/dsr/<dsr_id>")
def update_dsr(dsr_id: str):
	body = request.get_json(silent=True) or {}
	try:
		upd = DataSubjectRequestUpdate(**body)
	except ValidationError as exc:
		return _err(str(exc), 422)
	svc = _svc()
	try:
		return _ok(_run(svc.update_dsr(dsr_id, upd)))
	except KeyError:
		return _err("DSR not found", 404)


@audl_bp.get("/dsr/erasure-impact/<subject_id>")
def erasure_impact(subject_id: str):
	"""Assess audit-log impact of a GDPR Art. 17 erasure request without modifying data."""
	svc = _svc()
	return _ok(_run(svc.right_to_erasure_audit_impact(subject_id)))


# ---------------------------------------------------------------------------
# EvidencePackage
# ---------------------------------------------------------------------------

@audl_bp.post("/evidence-packages")
def create_evidence_package():
	"""
	Assemble and seal a tamper-evident evidence package.

	Body: EvidencePackageCreate JSON.
	"""
	try:
		req = _parse(EvidencePackageCreate)
	except _ValidationError as exc:
		return _err(str(exc.exc), 422)
	svc = _svc()
	return _ok(_run(svc.evidence_package_export(req)), 201)


@audl_bp.get("/evidence-packages")
def list_evidence_packages():
	svc = _svc()
	return _ok(_run(svc.list_evidence_packages()))


@audl_bp.get("/evidence-packages/<pkg_id>")
def get_evidence_package(pkg_id: str):
	svc = _svc()
	try:
		return _ok(_run(svc.get_evidence_package(pkg_id)))
	except KeyError:
		return _err("evidence package not found", 404)


# ---------------------------------------------------------------------------
# TamperDetection
# ---------------------------------------------------------------------------

@audl_bp.post("/tamper-detection")
def run_tamper_detection():
	"""
	Run a tamper-detection scan over all stored events.

	Body: TamperDetectionCreate JSON.
	"""
	try:
		req = _parse(TamperDetectionCreate)
	except _ValidationError as exc:
		return _err(str(exc.exc), 422)
	svc = _svc()
	return _ok(_run(svc.tamper_detection(req)))


@audl_bp.get("/tamper-detection")
def list_tamper_scans():
	svc = _svc()
	scans = [s for s in svc._tampers.values() if s.tenant_id == svc.tenant_id]
	return _ok([s.model_dump(mode="json") for s in scans])


# ---------------------------------------------------------------------------
# Risk summary
# ---------------------------------------------------------------------------

@audl_bp.get("/reports/risk-summary")
def risk_summary():
	"""
	Return risk and compliance aggregates for a time window.

	Query params: period_start (ISO), period_end (ISO)
	"""
	svc   = _svc()
	start = request.args.get("period_start")
	end   = request.args.get("period_end")
	now   = datetime.now(timezone.utc)
	from datetime import timedelta
	ps = datetime.fromisoformat(start) if start else (now - timedelta(days=30))
	pe = datetime.fromisoformat(end)   if end   else now
	return _ok(_run(svc.risk_summary(ps, pe)))


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@audl_bp.errorhandler(404)
def not_found(e):
	return _err("not found", 404)


@audl_bp.errorhandler(405)
def method_not_allowed(e):
	return _err("method not allowed", 405)


@audl_bp.errorhandler(500)
def server_error(e):
	log.exception("audl unhandled error")
	return _err("internal server error", 500)


__all__ = ["audl_bp"]
