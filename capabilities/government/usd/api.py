"""Flask Blueprint REST API for USSD Government Services (gov_usd)."""
from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, jsonify, request

try:
	from .service import USSDGovService
	from .models import (
		USSDSessionCreate, USSDSessionUpdate,
		PermitEnquiryCreate,
		TaxBalanceEnquiryCreate,
		IDVerificationCreate,
		CertificateRequestCreate, CertificateRequestUpdate,
		USSDMenuCreate,
	)
except ImportError:
	from service import USSDGovService  # type: ignore
	from models import (  # type: ignore
		USSDSessionCreate, USSDSessionUpdate,
		PermitEnquiryCreate,
		TaxBalanceEnquiryCreate,
		IDVerificationCreate,
		CertificateRequestCreate, CertificateRequestUpdate,
		USSDMenuCreate,
	)

_log = logging.getLogger(__name__)

bp = Blueprint("gov_usd", __name__, url_prefix="/api/government/usd")

_svc = USSDGovService()


def _json_response(data: Any, status: int = 200):
	return jsonify(data), status


def _error(message: str, status: int = 400):
	return jsonify({"error": message}), status


# ── Health ─────────────────────────────────────────────────────────────────────

@bp.get("/health")
async def health():
	"""Service health check."""
	try:
		result = await _svc.health_check()
		return _json_response(result)
	except Exception as exc:
		_log.error("health_check failed: %s", exc)
		return _error(str(exc), 500)


# ── Sessions ───────────────────────────────────────────────────────────────────

@bp.get("/sessions")
async def list_sessions():
	"""List USSD sessions."""
	try:
		tenant_id = request.args.get("tenant_id", "default")
		msisdn = request.args.get("msisdn")
		status = request.args.get("status")
		result = await _svc.list_sessions(tenant_id=tenant_id, msisdn=msisdn, status=status)
		return _json_response({"items": result, "total": len(result)})
	except Exception as exc:
		_log.error("list_sessions failed: %s", exc)
		return _error(str(exc), 500)


@bp.get("/sessions/<session_id>")
async def get_session(session_id: str):
	"""Get a USSD session."""
	try:
		tenant_id = request.args.get("tenant_id", "default")
		result = await _svc.get_session(session_id, tenant_id=tenant_id)
		return _json_response(result)
	except KeyError:
		return _error("session not found", 404)
	except Exception as exc:
		_log.error("get_session failed: %s", exc)
		return _error(str(exc), 500)


@bp.post("/sessions")
async def create_session():
	"""Create a USSD session."""
	try:
		body = USSDSessionCreate(**request.get_json(force=True))
		result = await _svc.create_session(
			msisdn=body.msisdn,
			service_code=body.service_code,
			tenant_id=body.tenant_id,
			session_data=body.session_data,
		)
		return _json_response(result, 201)
	except (ValueError, TypeError) as exc:
		return _error(str(exc), 422)
	except Exception as exc:
		_log.error("create_session failed: %s", exc)
		return _error(str(exc), 500)


@bp.put("/sessions/<session_id>")
async def update_session(session_id: str):
	"""Advance a USSD session."""
	try:
		body = USSDSessionUpdate(**request.get_json(force=True))
		tenant_id = request.args.get("tenant_id", "default")
		result = await _svc.update_session(
			session_id=session_id,
			input_text=body.input_text,
			tenant_id=tenant_id,
			menu_level=body.menu_level or None,
			session_data=body.session_data,
		)
		return _json_response(result)
	except KeyError:
		return _error("session not found", 404)
	except Exception as exc:
		_log.error("update_session failed: %s", exc)
		return _error(str(exc), 500)


@bp.delete("/sessions/<session_id>")
async def delete_session(session_id: str):
	"""Delete a USSD session."""
	try:
		tenant_id = request.args.get("tenant_id", "default")
		result = await _svc.delete_session(session_id, tenant_id=tenant_id)
		return _json_response(result)
	except KeyError:
		return _error("session not found", 404)
	except Exception as exc:
		_log.error("delete_session failed: %s", exc)
		return _error(str(exc), 500)


# ── Permit enquiries ───────────────────────────────────────────────────────────

@bp.get("/permits/enquiries")
async def list_permit_enquiries():
	"""List permit enquiries."""
	try:
		tenant_id = request.args.get("tenant_id", "default")
		msisdn = request.args.get("msisdn")
		result = await _svc.list_permit_enquiries(tenant_id=tenant_id, msisdn=msisdn)
		return _json_response({"items": result, "total": len(result)})
	except Exception as exc:
		_log.error("list_permit_enquiries failed: %s", exc)
		return _error(str(exc), 500)


@bp.post("/permits/enquiries")
async def enquire_permit():
	"""Query permit status."""
	try:
		body = PermitEnquiryCreate(**request.get_json(force=True))
		result = await _svc.enquire_permit_status(
			msisdn=body.msisdn,
			permit_number=body.permit_number,
			permit_type=body.permit_type,
			tenant_id=body.tenant_id,
		)
		return _json_response(result, 201)
	except ValueError as exc:
		return _error(str(exc), 422)
	except Exception as exc:
		_log.error("enquire_permit failed: %s", exc)
		return _error(str(exc), 500)


# ── Tax enquiries ──────────────────────────────────────────────────────────────

@bp.get("/tax/enquiries")
async def list_tax_enquiries():
	"""List tax balance enquiries."""
	try:
		tenant_id = request.args.get("tenant_id", "default")
		result = await _svc.list_tax_enquiries(tenant_id=tenant_id)
		return _json_response({"items": result, "total": len(result)})
	except Exception as exc:
		_log.error("list_tax_enquiries failed: %s", exc)
		return _error(str(exc), 500)


@bp.post("/tax/enquiries")
async def enquire_tax():
	"""Query tax balance."""
	try:
		body = TaxBalanceEnquiryCreate(**request.get_json(force=True))
		result = await _svc.enquire_tax_balance(
			msisdn=body.msisdn,
			tax_pin=body.tax_pin,
			tax_type=body.tax_type,
			tenant_id=body.tenant_id,
		)
		return _json_response(result, 201)
	except ValueError as exc:
		return _error(str(exc), 422)
	except Exception as exc:
		_log.error("enquire_tax failed: %s", exc)
		return _error(str(exc), 500)


# ── ID verification ────────────────────────────────────────────────────────────

@bp.get("/id-verifications")
async def list_id_verifications():
	"""List ID verifications."""
	try:
		tenant_id = request.args.get("tenant_id", "default")
		result = await _svc.list_id_verifications(tenant_id=tenant_id)
		return _json_response({"items": result, "total": len(result)})
	except Exception as exc:
		_log.error("list_id_verifications failed: %s", exc)
		return _error(str(exc), 500)


@bp.post("/id-verifications")
async def verify_id():
	"""Verify a national ID."""
	try:
		body = IDVerificationCreate(**request.get_json(force=True))
		result = await _svc.verify_id(
			msisdn=body.msisdn,
			id_number=body.id_number,
			id_type=body.id_type,
			full_name=body.full_name,
			tenant_id=body.tenant_id,
		)
		return _json_response(result, 201)
	except ValueError as exc:
		return _error(str(exc), 422)
	except Exception as exc:
		_log.error("verify_id failed: %s", exc)
		return _error(str(exc), 500)


# ── Certificate requests ───────────────────────────────────────────────────────

@bp.get("/certificates")
async def list_certificate_requests():
	"""List certificate requests."""
	try:
		tenant_id = request.args.get("tenant_id", "default")
		cert_type = request.args.get("certificate_type")
		status = request.args.get("status")
		result = await _svc.list_certificate_requests(tenant_id=tenant_id, certificate_type=cert_type, status=status)
		return _json_response({"items": result, "total": len(result)})
	except Exception as exc:
		_log.error("list_certificate_requests failed: %s", exc)
		return _error(str(exc), 500)


@bp.get("/certificates/<request_id>")
async def get_certificate_request(request_id: str):
	"""Get a certificate request."""
	try:
		tenant_id = request.args.get("tenant_id", "default")
		result = await _svc.get_certificate_request(request_id, tenant_id=tenant_id)
		return _json_response(result)
	except KeyError:
		return _error("certificate request not found", 404)
	except Exception as exc:
		_log.error("get_certificate_request failed: %s", exc)
		return _error(str(exc), 500)


@bp.post("/certificates")
async def create_certificate_request():
	"""Submit a certificate request."""
	try:
		body = CertificateRequestCreate(**request.get_json(force=True))
		result = await _svc.request_certificate(
			msisdn=body.msisdn,
			certificate_type=body.certificate_type,
			applicant_id=body.applicant_id,
			applicant_name=body.applicant_name,
			tenant_id=body.tenant_id,
			reference_number=body.reference_number,
			metadata=body.metadata,
		)
		return _json_response(result, 201)
	except ValueError as exc:
		return _error(str(exc), 422)
	except Exception as exc:
		_log.error("create_certificate_request failed: %s", exc)
		return _error(str(exc), 500)


@bp.put("/certificates/<request_id>")
async def update_certificate_request(request_id: str):
	"""Update a certificate request."""
	try:
		body = CertificateRequestUpdate(**request.get_json(force=True))
		tenant_id = request.args.get("tenant_id", "default")
		result = await _svc.update_certificate_request(
			request_id=request_id,
			tenant_id=tenant_id,
			status=body.status,
			certificate_number=body.certificate_number,
			issued_by=body.issued_by,
			notes=body.notes,
		)
		return _json_response(result)
	except KeyError:
		return _error("certificate request not found", 404)
	except Exception as exc:
		_log.error("update_certificate_request failed: %s", exc)
		return _error(str(exc), 500)


@bp.delete("/certificates/<request_id>")
async def delete_certificate_request(request_id: str):
	"""Delete a certificate request."""
	try:
		tenant_id = request.args.get("tenant_id", "default")
		result = await _svc.delete_certificate_request(request_id, tenant_id=tenant_id)
		return _json_response(result)
	except KeyError:
		return _error("certificate request not found", 404)
	except Exception as exc:
		_log.error("delete_certificate_request failed: %s", exc)
		return _error(str(exc), 500)


# ── Dashboard ──────────────────────────────────────────────────────────────────

@bp.get("/dashboard")
async def dashboard():
	"""USSD service dashboard metrics."""
	try:
		tenant_id = request.args.get("tenant_id", "default")
		result = await _svc.dashboard_summary(tenant_id=tenant_id)
		return _json_response(result)
	except Exception as exc:
		_log.error("dashboard failed: %s", exc)
		return _error(str(exc), 500)


@bp.get("/audit-events")
async def audit_events():
	"""List audit events."""
	try:
		tenant_id = request.args.get("tenant_id", "default")
		result = await _svc.get_audit_events(tenant_id=tenant_id)
		return _json_response({"items": result, "total": len(result)})
	except Exception as exc:
		_log.error("audit_events failed: %s", exc)
		return _error(str(exc), 500)
