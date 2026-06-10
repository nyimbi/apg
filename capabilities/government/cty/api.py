"""Flask Blueprint REST API for County / Devolved Services (gov_cty)."""
from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, jsonify, request

try:
	from .service import CountyServicesService
	from .models import (
		RevenueCollectionCreate,
		CountyPermitCreate, CountyPermitUpdate,
		SocialWelfareApplicationCreate, SocialWelfareApplicationUpdate,
		HealthFacilityCreate,
		PatientRegistrationCreate,
		PublicWorksTicketCreate, PublicWorksTicketUpdate,
	)
except ImportError:
	from service import CountyServicesService  # type: ignore
	from models import (  # type: ignore
		RevenueCollectionCreate,
		CountyPermitCreate, CountyPermitUpdate,
		SocialWelfareApplicationCreate, SocialWelfareApplicationUpdate,
		HealthFacilityCreate,
		PatientRegistrationCreate,
		PublicWorksTicketCreate, PublicWorksTicketUpdate,
	)

_log = logging.getLogger(__name__)

bp = Blueprint("gov_cty", __name__, url_prefix="/api/government/cty")

_svc = CountyServicesService()


def _json_response(data: Any, status: int = 200):
	return jsonify(data), status


def _error(message: str, status: int = 400):
	return jsonify({"error": message}), status


# ── Health ─────────────────────────────────────────────────────────────────────

@bp.get("/health")
async def health():
	try:
		return _json_response(await _svc.health_check())
	except Exception as exc:
		_log.error("health_check failed: %s", exc)
		return _error(str(exc), 500)


# ── Revenue collection ─────────────────────────────────────────────────────────

@bp.get("/revenues")
async def list_revenues():
	try:
		tenant_id = request.args.get("tenant_id", "default")
		revenue_type = request.args.get("revenue_type")
		status = request.args.get("status")
		period = request.args.get("period")
		result = await _svc.list_revenues(tenant_id=tenant_id, revenue_type=revenue_type, status=status, period=period)
		return _json_response({"items": result, "total": len(result)})
	except Exception as exc:
		_log.error("list_revenues failed: %s", exc)
		return _error(str(exc), 500)


@bp.post("/revenues")
async def collect_revenue():
	try:
		body = RevenueCollectionCreate(**request.get_json(force=True))
		result = await _svc.collect_revenue(
			payer_id=body.payer_id,
			payer_name=body.payer_name,
			revenue_type=body.revenue_type,
			amount_kes=body.amount_kes,
			period=body.period,
			tenant_id=body.tenant_id,
			payment_method=body.payment_method,
			receipt_number=body.receipt_number,
			metadata=body.metadata,
		)
		return _json_response(result, 201)
	except (ValueError, PermissionError) as exc:
		return _error(str(exc), 422)
	except Exception as exc:
		_log.error("collect_revenue failed: %s", exc)
		return _error(str(exc), 500)


@bp.get("/revenues/<revenue_id>")
async def get_revenue(revenue_id: str):
	try:
		tenant_id = request.args.get("tenant_id", "default")
		return _json_response(await _svc.get_revenue(revenue_id, tenant_id=tenant_id))
	except KeyError:
		return _error("revenue record not found", 404)
	except Exception as exc:
		_log.error("get_revenue failed: %s", exc)
		return _error(str(exc), 500)


@bp.post("/revenues/<revenue_id>/confirm")
async def confirm_revenue(revenue_id: str):
	try:
		tenant_id = request.args.get("tenant_id", "default")
		return _json_response(await _svc.confirm_revenue(revenue_id, tenant_id=tenant_id))
	except (KeyError, PermissionError) as exc:
		return _error(str(exc), 422)
	except Exception as exc:
		_log.error("confirm_revenue failed: %s", exc)
		return _error(str(exc), 500)


@bp.get("/revenues/summary")
async def revenue_summary():
	try:
		tenant_id = request.args.get("tenant_id", "default")
		period = request.args.get("period")
		return _json_response(await _svc.revenue_summary(tenant_id=tenant_id, period=period))
	except Exception as exc:
		_log.error("revenue_summary failed: %s", exc)
		return _error(str(exc), 500)


# ── Permits ────────────────────────────────────────────────────────────────────

@bp.get("/permits")
async def list_permits():
	try:
		tenant_id = request.args.get("tenant_id", "default")
		permit_type = request.args.get("permit_type")
		status = request.args.get("status")
		sub_county = request.args.get("sub_county")
		result = await _svc.list_permits(tenant_id=tenant_id, permit_type=permit_type, status=status, sub_county=sub_county)
		return _json_response({"items": result, "total": len(result)})
	except Exception as exc:
		_log.error("list_permits failed: %s", exc)
		return _error(str(exc), 500)


@bp.get("/permits/<permit_id>")
async def get_permit(permit_id: str):
	try:
		tenant_id = request.args.get("tenant_id", "default")
		return _json_response(await _svc.get_permit(permit_id, tenant_id=tenant_id))
	except KeyError:
		return _error("permit not found", 404)
	except Exception as exc:
		_log.error("get_permit failed: %s", exc)
		return _error(str(exc), 500)


@bp.post("/permits")
async def apply_permit():
	try:
		body = CountyPermitCreate(**request.get_json(force=True))
		result = await _svc.apply_permit(
			applicant_id=body.applicant_id,
			applicant_name=body.applicant_name,
			business_name=body.business_name,
			permit_type=body.permit_type,
			location=body.location,
			sub_county=body.sub_county,
			fee_paid_kes=body.fee_paid_kes,
			tenant_id=body.tenant_id,
			supporting_documents=body.supporting_documents,
			metadata=body.metadata,
		)
		return _json_response(result, 201)
	except (ValueError, PermissionError) as exc:
		return _error(str(exc), 422)
	except Exception as exc:
		_log.error("apply_permit failed: %s", exc)
		return _error(str(exc), 500)


@bp.put("/permits/<permit_id>")
async def update_permit(permit_id: str):
	try:
		body = CountyPermitUpdate(**request.get_json(force=True))
		tenant_id = request.args.get("tenant_id", "default")
		result = await _svc.update_permit(
			permit_id=permit_id,
			tenant_id=tenant_id,
			status=body.status,
			issued_by=body.issued_by,
		)
		return _json_response(result)
	except KeyError:
		return _error("permit not found", 404)
	except Exception as exc:
		_log.error("update_permit failed: %s", exc)
		return _error(str(exc), 500)


@bp.delete("/permits/<permit_id>")
async def delete_permit(permit_id: str):
	try:
		tenant_id = request.args.get("tenant_id", "default")
		return _json_response(await _svc.delete_permit(permit_id, tenant_id=tenant_id))
	except (KeyError, PermissionError) as exc:
		return _error(str(exc), 422)
	except Exception as exc:
		_log.error("delete_permit failed: %s", exc)
		return _error(str(exc), 500)


# ── Social welfare ─────────────────────────────────────────────────────────────

@bp.get("/welfare")
async def list_welfare():
	try:
		tenant_id = request.args.get("tenant_id", "default")
		programme_type = request.args.get("programme_type")
		status = request.args.get("status")
		sub_county = request.args.get("sub_county")
		result = await _svc.list_welfare_applications(tenant_id=tenant_id, programme_type=programme_type, status=status, sub_county=sub_county)
		return _json_response({"items": result, "total": len(result)})
	except Exception as exc:
		_log.error("list_welfare failed: %s", exc)
		return _error(str(exc), 500)


@bp.get("/welfare/<application_id>")
async def get_welfare(application_id: str):
	try:
		tenant_id = request.args.get("tenant_id", "default")
		return _json_response(await _svc.get_welfare_application(application_id, tenant_id=tenant_id))
	except KeyError:
		return _error("welfare application not found", 404)
	except Exception as exc:
		_log.error("get_welfare failed: %s", exc)
		return _error(str(exc), 500)


@bp.post("/welfare")
async def apply_welfare():
	try:
		body = SocialWelfareApplicationCreate(**request.get_json(force=True))
		result = await _svc.apply_welfare(
			applicant_id=body.applicant_id,
			applicant_name=body.applicant_name,
			id_number=body.id_number,
			programme_type=body.programme_type,
			sub_county=body.sub_county,
			ward=body.ward,
			household_size=body.household_size,
			tenant_id=body.tenant_id,
			monthly_income_kes=body.monthly_income_kes,
			needs_assessment=body.needs_assessment,
			metadata=body.metadata,
		)
		return _json_response(result, 201)
	except (ValueError, PermissionError) as exc:
		return _error(str(exc), 422)
	except Exception as exc:
		_log.error("apply_welfare failed: %s", exc)
		return _error(str(exc), 500)


@bp.put("/welfare/<application_id>")
async def update_welfare(application_id: str):
	try:
		body = SocialWelfareApplicationUpdate(**request.get_json(force=True))
		tenant_id = request.args.get("tenant_id", "default")
		result = await _svc.update_welfare_application(
			application_id=application_id,
			tenant_id=tenant_id,
			status=body.status,
			case_worker_id=body.case_worker_id,
			notes=body.notes,
		)
		return _json_response(result)
	except KeyError:
		return _error("welfare application not found", 404)
	except Exception as exc:
		_log.error("update_welfare failed: %s", exc)
		return _error(str(exc), 500)


# ── Health facilities ──────────────────────────────────────────────────────────

@bp.get("/health-facilities")
async def list_health_facilities():
	try:
		tenant_id = request.args.get("tenant_id", "default")
		facility_type = request.args.get("facility_type")
		sub_county = request.args.get("sub_county")
		result = await _svc.list_health_facilities(tenant_id=tenant_id, facility_type=facility_type, sub_county=sub_county)
		return _json_response({"items": result, "total": len(result)})
	except Exception as exc:
		_log.error("list_health_facilities failed: %s", exc)
		return _error(str(exc), 500)


@bp.post("/health-facilities")
async def register_health_facility():
	try:
		body = HealthFacilityCreate(**request.get_json(force=True))
		result = await _svc.register_health_facility(
			facility_code=body.facility_code,
			facility_name=body.facility_name,
			facility_type=body.facility_type,
			sub_county=body.sub_county,
			ward=body.ward,
			tenant_id=body.tenant_id,
			beds=body.beds,
			services=body.services,
		)
		return _json_response(result, 201)
	except (ValueError, PermissionError) as exc:
		return _error(str(exc), 422)
	except Exception as exc:
		_log.error("register_health_facility failed: %s", exc)
		return _error(str(exc), 500)


@bp.post("/patients")
async def register_patient():
	try:
		body = PatientRegistrationCreate(**request.get_json(force=True))
		result = await _svc.register_patient(
			facility_id=body.facility_id,
			patient_name=body.patient_name,
			id_number=body.id_number,
			date_of_birth=body.date_of_birth,
			gender=body.gender,
			sub_county=body.sub_county,
			tenant_id=body.tenant_id,
			phone=body.phone,
		)
		return _json_response(result, 201)
	except (KeyError, ValueError) as exc:
		return _error(str(exc), 422)
	except Exception as exc:
		_log.error("register_patient failed: %s", exc)
		return _error(str(exc), 500)


# ── Public works tickets ───────────────────────────────────────────────────────

@bp.get("/tickets")
async def list_tickets():
	try:
		tenant_id = request.args.get("tenant_id", "default")
		ticket_type = request.args.get("ticket_type")
		status = request.args.get("status")
		priority = request.args.get("priority")
		sub_county = request.args.get("sub_county")
		result = await _svc.list_tickets(tenant_id=tenant_id, ticket_type=ticket_type, status=status, priority=priority, sub_county=sub_county)
		return _json_response({"items": result, "total": len(result)})
	except Exception as exc:
		_log.error("list_tickets failed: %s", exc)
		return _error(str(exc), 500)


@bp.get("/tickets/<ticket_id>")
async def get_ticket(ticket_id: str):
	try:
		tenant_id = request.args.get("tenant_id", "default")
		return _json_response(await _svc.get_ticket(ticket_id, tenant_id=tenant_id))
	except KeyError:
		return _error("ticket not found", 404)
	except Exception as exc:
		_log.error("get_ticket failed: %s", exc)
		return _error(str(exc), 500)


@bp.post("/tickets")
async def create_ticket():
	try:
		body = PublicWorksTicketCreate(**request.get_json(force=True))
		result = await _svc.create_ticket(
			reporter_id=body.reporter_id,
			reporter_name=body.reporter_name,
			ticket_type=body.ticket_type,
			description=body.description,
			location=body.location,
			sub_county=body.sub_county,
			ward=body.ward,
			tenant_id=body.tenant_id,
			priority=body.priority,
			reporter_phone=body.reporter_phone,
			metadata=body.metadata,
		)
		return _json_response(result, 201)
	except (ValueError, PermissionError) as exc:
		return _error(str(exc), 422)
	except Exception as exc:
		_log.error("create_ticket failed: %s", exc)
		return _error(str(exc), 500)


@bp.put("/tickets/<ticket_id>")
async def update_ticket(ticket_id: str):
	try:
		body = PublicWorksTicketUpdate(**request.get_json(force=True))
		tenant_id = request.args.get("tenant_id", "default")
		result = await _svc.update_ticket(
			ticket_id=ticket_id,
			tenant_id=tenant_id,
			status=body.status,
			assigned_to=body.assigned_to,
			resolution_notes=body.resolution_notes,
			estimated_completion=body.estimated_completion,
		)
		return _json_response(result)
	except KeyError:
		return _error("ticket not found", 404)
	except Exception as exc:
		_log.error("update_ticket failed: %s", exc)
		return _error(str(exc), 500)


@bp.delete("/tickets/<ticket_id>")
async def delete_ticket(ticket_id: str):
	try:
		tenant_id = request.args.get("tenant_id", "default")
		return _json_response(await _svc.delete_ticket(ticket_id, tenant_id=tenant_id))
	except KeyError:
		return _error("ticket not found", 404)
	except Exception as exc:
		_log.error("delete_ticket failed: %s", exc)
		return _error(str(exc), 500)


# ── Dashboard & audit ──────────────────────────────────────────────────────────

@bp.get("/dashboard")
async def dashboard():
	try:
		tenant_id = request.args.get("tenant_id", "default")
		return _json_response(await _svc.dashboard_summary(tenant_id=tenant_id))
	except Exception as exc:
		_log.error("dashboard failed: %s", exc)
		return _error(str(exc), 500)


@bp.get("/audit-events")
async def audit_events():
	try:
		tenant_id = request.args.get("tenant_id", "default")
		result = await _svc.get_audit_events(tenant_id=tenant_id)
		return _json_response({"items": result, "total": len(result)})
	except Exception as exc:
		_log.error("audit_events failed: %s", exc)
		return _error(str(exc), 500)
