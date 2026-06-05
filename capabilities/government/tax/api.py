"""Flask Blueprint REST API for APG Tax Administration.

All endpoints are async-compatible via Flask's async support (Werkzeug 3+).
Auth: expects X-Tenant-ID header; X-Actor-ID optional (defaults to "system").
All responses: {"data": ..., "meta": {...}} or {"error": ..., "code": ...}.
"""
from __future__ import annotations

import asyncio
from datetime import date
from decimal import Decimal
from functools import wraps
from typing import Any

from flask import Blueprint, jsonify, request

try:
	from .service import TaxAdministrationService
	from .models import (
		TaxpayerCreate, TaxpayerUpdate, TaxReturnCreate, TaxReturnUpdate,
		TaxAssessmentCreate, TaxAssessmentUpdate, TaxPaymentCreate,
		TaxDebtCreate, TaxAuditCreate, TaxAuditUpdate,
		AuditFindingCreate, ObjectionCreate, ObjectionUpdate,
		AppealCreate, TaxRefundCreate, PenaltyCreate, InterestCreate,
		TaxClearanceCertificateCreate,
		TaxType, ReturnType, AuditType, TaxpayerType,
	)
except ImportError:
	from service import TaxAdministrationService  # type: ignore
	from models import (  # type: ignore
		TaxpayerCreate, TaxpayerUpdate, TaxReturnCreate, TaxReturnUpdate,
		TaxAssessmentCreate, TaxAssessmentUpdate, TaxPaymentCreate,
		TaxDebtCreate, TaxAuditCreate, TaxAuditUpdate,
		AuditFindingCreate, ObjectionCreate, ObjectionUpdate,
		AppealCreate, TaxRefundCreate, PenaltyCreate, InterestCreate,
		TaxClearanceCertificateCreate,
		TaxType, ReturnType, AuditType, TaxpayerType,
	)

# ---------------------------------------------------------------------------
# Blueprint registration
# ---------------------------------------------------------------------------

tax_bp = Blueprint("tax", __name__, url_prefix="/api/v1/tax")

# Module-level service instance (swap for DI/per-request in production)
_svc = TaxAdministrationService()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tenant() -> str:
	return request.headers.get("X-Tenant-ID", "default")


def _actor() -> str:
	return request.headers.get("X-Actor-ID", "system")


def _body() -> dict[str, Any]:
	data = request.get_json(silent=True) or {}
	return data


def _ok(data: Any, status: int = 200, meta: dict[str, Any] | None = None) -> Any:
	payload: dict[str, Any] = {"data": data}
	if meta:
		payload["meta"] = meta
	return jsonify(payload), status


def _err(message: str, code: int = 400) -> Any:
	return jsonify({"error": message, "code": code}), code


def _run(coro):
    """Run a coroutine from Flask sync context. Python 3.12+ compatible."""
    import asyncio
    try:
        asyncio.get_running_loop()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result(timeout=30)
    except RuntimeError:
        return asyncio.run(coro)
def handle_errors(fn):
	"""Decorator: map common exceptions to HTTP responses."""
	@wraps(fn)
	def _wrapper(*args, **kwargs):
		try:
			return fn(*args, **kwargs)
		except AssertionError as exc:
			return _err(str(exc), 422)
		except PermissionError as exc:
			return _err(str(exc), 403)
		except KeyError as exc:
			return _err(f"missing field: {exc}", 400)
		except ValueError as exc:
			return _err(str(exc), 400)
		except Exception as exc:
			return _err(f"internal error: {exc}", 500)
	_wrapper.__name__ = fn.__name__
	return _wrapper


# ---------------------------------------------------------------------------
# Taxpayer endpoints
# ---------------------------------------------------------------------------

@tax_bp.get("/taxpayers")
@handle_errors
def list_taxpayers():
	"""List taxpayers for tenant with optional search."""
	tenant = _tenant()
	q = request.args.get("q")
	search_type = request.args.get("search_type", "name")
	limit = int(request.args.get("limit", 50))
	offset = int(request.args.get("offset", 0))

	if q:
		results = _svc.taxpayer_search(q, search_type, tenant_id=tenant)
	else:
		results = [r.model_dump(mode="json") for r in _svc._taxpayers.tenant_values(tenant) if not r.is_deleted]

	total = len(results)
	page = results[offset: offset + limit]
	return _ok(page, meta={"total": total, "limit": limit, "offset": offset})


@tax_bp.post("/taxpayers")
@handle_errors
def create_taxpayer():
	"""Register a new taxpayer."""
	body = _body()
	tenant = _tenant()
	actor = _actor()

	result = _svc.register_taxpayer(
		taxpayer_id=body.get("taxpayer_id", ""),
		tenant_id=tenant,
		tax_type=body.get("tax_type", "income_tax"),
		tax_pin=body.get("tax_pin", ""),
		id_number=body.get("national_id", body.get("id_number", "")),
		legal_name=body["taxpayer_name"],
		entity_type=body.get("taxpayer_type", "individual"),
		trade_name=body.get("trade_name"),
		email=body.get("email"),
		phone=body.get("phone"),
		address=body.get("physical_address", ""),
		tax_types=body.get("tax_types", []),
		evidence_reference=body.get("evidence_reference", "api_registration"),
		created_by=actor,
	)
	return _ok(result, 201)


@tax_bp.get("/taxpayers/<tin>")
@handle_errors
def get_taxpayer(tin: str):
	"""Get taxpayer by PIN."""
	tenant = _tenant()
	rec = _svc._find_taxpayer_by_pin(tin, tenant)
	if rec is None:
		return _err(f"taxpayer not found: {tin}", 404)
	return _ok(rec.model_dump(mode="json"))


@tax_bp.put("/taxpayers/<tin>")
@handle_errors
def update_taxpayer(tin: str):
	"""Update taxpayer fields."""
	body = _body()
	tenant = _tenant()
	result = _svc.update_taxpayer(tin, tenant_id=tenant, **body)
	return _ok(result)


@tax_bp.delete("/taxpayers/<tin>")
@handle_errors
def deregister_taxpayer(tin: str):
	"""Soft-deregister a taxpayer."""
	body = _body()
	tenant = _tenant()
	result = _svc.deregister_taxpayer(
		tin,
		reason=body.get("reason", "voluntary_deregistration"),
		deregistration_date=body.get("deregistration_date", date.today().isoformat()),
		tenant_id=tenant,
	)
	return _ok(result)


@tax_bp.get("/taxpayers/<tin>/verify")
@handle_errors
def verify_tin(tin: str):
	"""Verify TIN format and registry existence."""
	country = request.args.get("country", "KE")
	return _ok(_svc.verify_tin(tin, country))


@tax_bp.get("/taxpayers/<tin>/compliance-risk")
@handle_errors
def taxpayer_compliance_risk(tin: str):
	"""Get compliance risk profile for a taxpayer."""
	tenant = _tenant()
	tp = _svc._find_taxpayer_by_pin(tin, tenant)
	if tp is None:
		return _err(f"taxpayer not found: {tin}", 404)

	returns = [r for r in _svc._returns.tenant_values(tenant) if r.tax_pin.upper() == tin.upper()]
	payments = [p for p in _svc._payments.tenant_values(tenant) if p.taxpayer_id == tp.id]
	audits = [a for a in _svc._audits.tenant_values(tenant) if a.taxpayer_id == tp.id]

	try:
		from .domain.calculations import calculate_compliance_risk_score
	except ImportError:
		from domain.calculations import calculate_compliance_risk_score  # type: ignore

	score, category = calculate_compliance_risk_score(
		years_registered=max(1, (date.today() - tp.created_at.date()).days // 365),
		returns_filed=len(returns),
		returns_due=max(len(returns), 1),
		payments_on_time=len(payments),
		payments_due=max(len(payments), 1),
		open_audits=sum(1 for a in audits if a.status.value in ("planned", "in_progress")),
		prior_fraud_flags=0,
		days_avg_late_filing=0.0,
		debt_to_turnover_ratio=0.0,
	)
	return _ok({
		"taxpayer_id": tp.id,
		"tax_pin": tin,
		"risk_score": str(score),
		"risk_category": category,
	})


# ---------------------------------------------------------------------------
# Tax Returns
# ---------------------------------------------------------------------------

@tax_bp.get("/returns")
@handle_errors
def list_returns():
	"""List returns for tenant, filterable by tin/status/period."""
	tenant = _tenant()
	tin = request.args.get("tin")
	status = request.args.get("status")
	limit = int(request.args.get("limit", 50))
	offset = int(request.args.get("offset", 0))

	results = [r for r in _svc._returns.tenant_values(tenant) if not r.is_deleted]
	if tin:
		results = [r for r in results if r.tax_pin.upper() == tin.upper()]
	if status:
		results = [r for r in results if r.status.value == status]

	total = len(results)
	page = [r.model_dump(mode="json") for r in results[offset: offset + limit]]
	return _ok(page, meta={"total": total, "limit": limit, "offset": offset})


@tax_bp.post("/returns")
@handle_errors
def file_return():
	"""File a new tax return."""
	body = _body()
	tenant = _tenant()
	actor = _actor()

	result = _svc.submit_return(
		tin=body["tax_pin"],
		tax_type=body.get("tax_type", "income_tax"),
		period=body["period"],
		return_data=body,
		attachments=body.get("attachments"),
		tenant_id=tenant,
		created_by=actor,
	)
	return _ok(result, 201)


@tax_bp.post("/returns/nil")
@handle_errors
def file_nil_return():
	"""File a nil return."""
	body = _body()
	tenant = _tenant()
	actor = _actor()
	result = _svc.file_nil_return(
		tin=body["tax_pin"],
		tax_type=body["tax_type"],
		period=body["period"],
		tenant_id=tenant,
		created_by=actor,
	)
	return _ok(result, 201)


@tax_bp.get("/returns/<return_id>")
@handle_errors
def get_return(return_id: str):
	tenant = _tenant()
	rec = _svc._returns.get_item(tenant, return_id)
	if rec is None:
		return _err(f"return not found: {return_id}", 404)
	return _ok(rec.model_dump(mode="json"))


@tax_bp.put("/returns/<return_id>")
@handle_errors
def update_return(return_id: str):
	body = _body()
	tenant = _tenant()
	rec = _svc._returns.get_item(tenant, return_id)
	if rec is None:
		return _err(f"return not found: {return_id}", 404)
	result = _svc.amend_return(
		return_id=return_id,
		amendment_reason=body.get("amendment_reason", "correction"),
		amended_data=body,
		tenant_id=tenant,
		created_by=_actor(),
	)
	return _ok(result)


@tax_bp.post("/returns/<return_id>/validate")
@handle_errors
def validate_return(return_id: str):
	tenant = _tenant()
	return _ok(_svc.validate_return(return_id, tenant_id=tenant))


@tax_bp.get("/returns/status")
@handle_errors
def return_filing_status():
	"""Check whether a return was filed for a given TIN/tax_type/period."""
	tenant = _tenant()
	tin = request.args.get("tin", "")
	tax_type = request.args.get("tax_type", "income_tax")
	period = request.args.get("period", "")
	return _ok(_svc.return_filing_status(tin, tax_type, period, tenant_id=tenant))


# ---------------------------------------------------------------------------
# Assessments
# ---------------------------------------------------------------------------

@tax_bp.get("/assessments")
@handle_errors
def list_assessments():
	tenant = _tenant()
	status = request.args.get("status")
	tin = request.args.get("tin")
	limit = int(request.args.get("limit", 50))
	offset = int(request.args.get("offset", 0))

	results = [a for a in _svc._assessments.tenant_values(tenant) if not a.is_deleted]
	if status:
		results = [a for a in results if a.status.value == status]

	total = len(results)
	page = [a.model_dump(mode="json") for a in results[offset: offset + limit]]
	return _ok(page, meta={"total": total, "limit": limit, "offset": offset})


@tax_bp.post("/assessments")
@handle_errors
def create_assessment():
	body = _body()
	tenant = _tenant()
	actor = _actor()
	result = _svc.issue_assessment(
		tin=body["tax_pin"],
		tax_type=body.get("tax_type", "income_tax"),
		period=body["period"],
		assessed_amount=float(body["assessed_amount"]),
		reason=body.get("reason", ""),
		assessment_type=body.get("assessment_type", "best_judgement"),
		tenant_id=tenant,
		assessor_id=body.get("assessor_id", actor),
		created_by=actor,
	)
	return _ok(result, 201)


@tax_bp.get("/assessments/<assessment_id>")
@handle_errors
def get_assessment(assessment_id: str):
	tenant = _tenant()
	rec = _svc._assessments.get_item(tenant, assessment_id)
	if rec is None:
		return _err(f"assessment not found: {assessment_id}", 404)
	return _ok(rec.model_dump(mode="json"))


@tax_bp.put("/assessments/<assessment_id>")
@handle_errors
def update_assessment(assessment_id: str):
	body = _body()
	tenant = _tenant()
	rec = _svc._assessments.get_item(tenant, assessment_id)
	if rec is None:
		return _err(f"assessment not found: {assessment_id}", 404)
	data = rec.model_dump()
	for k, v in body.items():
		if k in data:
			data[k] = v
	from datetime import datetime as _dt
	data["updated_at"] = _dt.utcnow()
	try:
		from .models import TaxAssessmentResponse
	except ImportError:
		from models import TaxAssessmentResponse  # type: ignore
	updated = TaxAssessmentResponse(**data)
	_svc._assessments.put(tenant, assessment_id, updated)
	return _ok(updated.model_dump(mode="json"))


@tax_bp.post("/assessments/<assessment_id>/penalty-interest")
@handle_errors
def calc_penalty_interest(assessment_id: str):
	body = _body()
	tenant = _tenant()
	payment_date = body.get("payment_date", date.today().isoformat())
	return _ok(_svc.calculate_penalty_and_interest(assessment_id, payment_date, tenant_id=tenant))


# ---------------------------------------------------------------------------
# Payments
# ---------------------------------------------------------------------------

@tax_bp.get("/payments")
@handle_errors
def list_payments():
	tenant = _tenant()
	tin = request.args.get("tin")
	limit = int(request.args.get("limit", 50))
	offset = int(request.args.get("offset", 0))
	results = [p for p in _svc._payments.tenant_values(tenant) if not p.is_deleted]
	if tin:
		tp = _svc._find_taxpayer_by_pin(tin, tenant)
		if tp:
			results = [p for p in results if p.taxpayer_id == tp.id]
	total = len(results)
	page = [p.model_dump(mode="json") for p in results[offset: offset + limit]]
	return _ok(page, meta={"total": total, "limit": limit, "offset": offset})


@tax_bp.post("/payments")
@handle_errors
def create_payment():
	body = _body()
	tenant = _tenant()
	actor = _actor()
	result = _svc.process_tax_payment(
		tin=body["tax_pin"],
		tax_type=body.get("tax_type", "income_tax"),
		period=body.get("period", ""),
		amount=float(body["amount"]),
		payment_method=body.get("payment_method", "bank_transfer"),
		reference=body["payment_reference"],
		tenant_id=tenant,
		assessment_id=body.get("assessment_id"),
		created_by=actor,
	)
	return _ok(result, 201)


@tax_bp.get("/payments/<payment_id>")
@handle_errors
def get_payment(payment_id: str):
	tenant = _tenant()
	rec = _svc._payments.get_item(tenant, payment_id)
	if rec is None:
		return _err(f"payment not found: {payment_id}", 404)
	return _ok(rec.model_dump(mode="json"))


@tax_bp.post("/payments/<payment_id>/allocate")
@handle_errors
def allocate_payment(payment_id: str):
	tenant = _tenant()
	return _ok(_svc.allocate_payment_to_assessments(payment_id, tenant_id=tenant))


# ---------------------------------------------------------------------------
# Debts
# ---------------------------------------------------------------------------

@tax_bp.get("/debts")
@handle_errors
def list_debts():
	tenant = _tenant()
	tin = request.args.get("tin")
	status = request.args.get("status")
	limit = int(request.args.get("limit", 50))
	offset = int(request.args.get("offset", 0))
	results = [d for d in _svc._debts.tenant_values(tenant) if not d.is_deleted]
	if tin:
		tp = _svc._find_taxpayer_by_pin(tin, tenant)
		if tp:
			results = [d for d in results if d.taxpayer_id == tp.id]
	if status:
		results = [d for d in results if d.status.value == status]
	total = len(results)
	page = [d.model_dump(mode="json") for d in results[offset: offset + limit]]
	return _ok(page, meta={"total": total, "limit": limit, "offset": offset})


@tax_bp.get("/debts/<debt_id>")
@handle_errors
def get_debt(debt_id: str):
	tenant = _tenant()
	rec = _svc._debts.get_item(tenant, debt_id)
	if rec is None:
		return _err(f"debt not found: {debt_id}", 404)
	return _ok(rec.model_dump(mode="json"))


@tax_bp.post("/debts/demand-notice")
@handle_errors
def issue_demand_notice():
	body = _body()
	tenant = _tenant()
	actor = _actor()
	result = _svc.issue_demand_notice(
		tin=body["tax_pin"],
		outstanding_amount=float(body["outstanding_amount"]),
		deadline=body.get("deadline", (date.today().replace(day=28)).isoformat()),
		tenant_id=tenant,
		issued_by=actor,
	)
	return _ok(result, 201)


@tax_bp.post("/debts/collection-action")
@handle_errors
def debt_collection_action():
	body = _body()
	tenant = _tenant()
	actor = _actor()
	result = _svc.debt_collection_action(
		tin=body["tax_pin"],
		action_type=body["action_type"],
		officer_id=body.get("officer_id", actor),
		tenant_id=tenant,
		notes=body.get("notes"),
	)
	return _ok(result, 201)


# ---------------------------------------------------------------------------
# Audits
# ---------------------------------------------------------------------------

@tax_bp.get("/audits")
@handle_errors
def list_audits():
	tenant = _tenant()
	status = request.args.get("status")
	tin = request.args.get("tin")
	limit = int(request.args.get("limit", 50))
	offset = int(request.args.get("offset", 0))
	results = [a for a in _svc._audits.tenant_values(tenant) if not a.is_deleted]
	if tin:
		results = [a for a in results if a.tax_pin.upper() == tin.upper()]
	if status:
		results = [a for a in results if a.status.value == status]
	total = len(results)
	page = [a.model_dump(mode="json") for a in results[offset: offset + limit]]
	return _ok(page, meta={"total": total, "limit": limit, "offset": offset})


@tax_bp.post("/audits")
@handle_errors
def create_audit():
	body = _body()
	tenant = _tenant()
	actor = _actor()
	result = _svc.open_audit_case(
		tin=body["tax_pin"],
		audit_type=body.get("audit_type", "desk_audit"),
		audit_period=body["period"],
		assigned_officer=body.get("auditor_id", actor),
		tenant_id=tenant,
		scope_description=body.get("scope_description"),
		risk_score=body.get("risk_score"),
		created_by=actor,
	)
	return _ok(result, 201)


@tax_bp.get("/audits/<audit_id>")
@handle_errors
def get_audit(audit_id: str):
	tenant = _tenant()
	rec = _svc._audits.get_item(tenant, audit_id)
	if rec is None:
		return _err(f"audit not found: {audit_id}", 404)
	return _ok(rec.model_dump(mode="json"))


@tax_bp.put("/audits/<audit_id>")
@handle_errors
def update_audit(audit_id: str):
	body = _body()
	tenant = _tenant()
	rec = _svc._audits.get_item(tenant, audit_id)
	if rec is None:
		return _err(f"audit not found: {audit_id}", 404)
	data = rec.model_dump()
	for k, v in body.items():
		if k in data:
			data[k] = v
	from datetime import datetime as _dt
	data["updated_at"] = _dt.utcnow()
	try:
		from .models import TaxAuditResponse
	except ImportError:
		from models import TaxAuditResponse  # type: ignore
	updated = TaxAuditResponse(**data)
	_svc._audits.put(tenant, audit_id, updated)
	return _ok(updated.model_dump(mode="json"))


@tax_bp.post("/audits/<audit_id>/findings")
@handle_errors
def record_findings(audit_id: str):
	body = _body()
	tenant = _tenant()
	actor = _actor()
	findings = body.get("findings", [body])
	result = _svc.conduct_audit(audit_id, findings, tenant_id=tenant, created_by=actor)
	return _ok(result)


@tax_bp.post("/audits/<audit_id>/close")
@handle_errors
def close_audit(audit_id: str):
	body = _body()
	tenant = _tenant()
	result = _svc.close_audit_case(
		audit_id,
		outcome=body.get("outcome", "completed"),
		final_tax_due=float(body.get("final_tax_due", 0)),
		penalties=float(body.get("penalties", 0)),
		tenant_id=tenant,
	)
	return _ok(result)


# ---------------------------------------------------------------------------
# Objections
# ---------------------------------------------------------------------------

@tax_bp.get("/objections")
@handle_errors
def list_objections():
	tenant = _tenant()
	status = request.args.get("status")
	limit = int(request.args.get("limit", 50))
	offset = int(request.args.get("offset", 0))
	results = [o for o in _svc._objections.tenant_values(tenant) if not o.is_deleted]
	if status:
		results = [o for o in results if o.status.value == status]
	total = len(results)
	page = [o.model_dump(mode="json") for o in results[offset: offset + limit]]
	return _ok(page, meta={"total": total, "limit": limit, "offset": offset})


@tax_bp.post("/objections")
@handle_errors
def create_objection():
	body = _body()
	tenant = _tenant()
	actor = _actor()
	result = _svc.raise_objection(
		assessment_id=body["assessment_id"],
		grounds=body["grounds"],
		amount_disputed=float(body["amount_disputed"]),
		objection_date=body.get("filed_date"),
		tenant_id=tenant,
		tax_pin=body.get("tax_pin", ""),
		created_by=actor,
	)
	return _ok(result, 201)


@tax_bp.get("/objections/<objection_id>")
@handle_errors
def get_objection(objection_id: str):
	tenant = _tenant()
	rec = _svc._objections.get_item(tenant, objection_id)
	if rec is None:
		return _err(f"objection not found: {objection_id}", 404)
	return _ok(rec.model_dump(mode="json"))


@tax_bp.post("/objections/<objection_id>/determine")
@handle_errors
def determine_objection(objection_id: str):
	body = _body()
	tenant = _tenant()
	actor = _actor()
	result = _svc.process_objection(
		objection_id=objection_id,
		decision=body["decision"],
		revised_amount=float(body.get("revised_amount", 0)),
		officer_id=body.get("officer_id", actor),
		tenant_id=tenant,
	)
	return _ok(result)


# ---------------------------------------------------------------------------
# Appeals
# ---------------------------------------------------------------------------

@tax_bp.get("/appeals")
@handle_errors
def list_appeals():
	tenant = _tenant()
	limit = int(request.args.get("limit", 50))
	offset = int(request.args.get("offset", 0))
	results = [a for a in _svc._appeals.tenant_values(tenant) if not a.is_deleted]
	total = len(results)
	page = [a.model_dump(mode="json") for a in results[offset: offset + limit]]
	return _ok(page, meta={"total": total, "limit": limit, "offset": offset})


@tax_bp.post("/appeals")
@handle_errors
def create_appeal():
	body = _body()
	tenant = _tenant()
	actor = _actor()
	result = _svc.file_appeal(
		objection_id=body["objection_id"],
		appeal_grounds=body["grounds"],
		tenant_id=tenant,
		created_by=actor,
	)
	return _ok(result, 201)


@tax_bp.get("/appeals/<appeal_id>")
@handle_errors
def get_appeal(appeal_id: str):
	tenant = _tenant()
	rec = _svc._appeals.get_item(tenant, appeal_id)
	if rec is None:
		return _err(f"appeal not found: {appeal_id}", 404)
	return _ok(rec.model_dump(mode="json"))


# ---------------------------------------------------------------------------
# Refunds
# ---------------------------------------------------------------------------

@tax_bp.get("/refunds")
@handle_errors
def list_refunds():
	tenant = _tenant()
	status = request.args.get("status")
	limit = int(request.args.get("limit", 50))
	offset = int(request.args.get("offset", 0))
	results = [r for r in _svc._refunds.tenant_values(tenant) if not r.is_deleted]
	if status:
		results = [r for r in results if r.status.value == status]
	total = len(results)
	page = [r.model_dump(mode="json") for r in results[offset: offset + limit]]
	return _ok(page, meta={"total": total, "limit": limit, "offset": offset})


@tax_bp.post("/refunds")
@handle_errors
def create_refund():
	body = _body()
	tenant = _tenant()
	actor = _actor()
	result = _svc.refund_application(
		tin=body["tax_pin"],
		tax_type=body.get("tax_type", "vat"),
		period=body["period"],
		refund_amount=float(body["claimed_amount"]),
		reason=body.get("refund_type", "overpayment"),
		tenant_id=tenant,
		bank_account_number=body.get("bank_account_number"),
		bank_name=body.get("bank_name"),
		supporting_documents=body.get("supporting_documents"),
		created_by=actor,
	)
	return _ok(result, 201)


@tax_bp.get("/refunds/<refund_id>")
@handle_errors
def get_refund(refund_id: str):
	tenant = _tenant()
	rec = _svc._refunds.get_item(tenant, refund_id)
	if rec is None:
		return _err(f"refund not found: {refund_id}", 404)
	return _ok(rec.model_dump(mode="json"))


@tax_bp.post("/refunds/<refund_id>/review")
@handle_errors
def review_refund(refund_id: str):
	body = _body()
	tenant = _tenant()
	actor = _actor()
	result = _svc.verify_refund(
		refund_id, officer_id=body.get("officer_id", actor),
		tenant_id=tenant, notes=body.get("notes"),
	)
	return _ok(result)


@tax_bp.post("/refunds/<refund_id>/approve")
@handle_errors
def approve_refund(refund_id: str):
	body = _body()
	tenant = _tenant()
	actor = _actor()
	result = _svc.approve_refund(
		refund_id,
		approved_by=body.get("approved_by", actor),
		payment_method=body.get("payment_method", "bank_transfer"),
		tenant_id=tenant,
		approved_amount=body.get("approved_amount"),
		notes=body.get("notes"),
	)
	return _ok(result)


# ---------------------------------------------------------------------------
# Tax Clearance Certificates
# ---------------------------------------------------------------------------

@tax_bp.get("/clearances")
@handle_errors
def list_clearances():
	tenant = _tenant()
	limit = int(request.args.get("limit", 50))
	offset = int(request.args.get("offset", 0))
	results = [c for c in _svc._clearances.tenant_values(tenant) if not c.is_deleted]
	total = len(results)
	page = [c.model_dump(mode="json") for c in results[offset: offset + limit]]
	return _ok(page, meta={"total": total, "limit": limit, "offset": offset})


@tax_bp.post("/clearances")
@handle_errors
def request_clearance():
	body = _body()
	tenant = _tenant()
	actor = _actor()
	result = _svc.issue_tax_clearance_certificate(
		tin=body["tax_pin"],
		validity_days=int(body.get("validity_days", 180)),
		tenant_id=tenant,
		purpose=body.get("purpose", "general"),
		created_by=actor,
	)
	return _ok(result, 201)


@tax_bp.get("/clearances/<cert_id>")
@handle_errors
def get_clearance(cert_id: str):
	tenant = _tenant()
	rec = _svc._clearances.get_item(tenant, cert_id)
	if rec is None:
		return _err(f"certificate not found: {cert_id}", 404)
	return _ok(rec.model_dump(mode="json"))


@tax_bp.get("/clearances/verify/<certificate_number>")
@handle_errors
def verify_clearance(certificate_number: str):
	"""Verify a TCC by certificate number."""
	tenant = _tenant()
	rec = next(
		(c for c in _svc._clearances.tenant_values(tenant)
		 if c.certificate_number == certificate_number),
		None,
	)
	if rec is None:
		return _ok({"valid": False, "certificate_number": certificate_number})
	from datetime import date as _date
	valid = rec.status.value == "issued" and (rec.expiry_date is None or rec.expiry_date >= _date.today())
	return _ok({"valid": valid, "certificate": rec.model_dump(mode="json")})


# ---------------------------------------------------------------------------
# Exchange of Information
# ---------------------------------------------------------------------------

@tax_bp.post("/eoi")
@handle_errors
def exchange_of_information():
	body = _body()
	tenant = _tenant()
	result = _svc.exchange_of_information(
		request_source=body["treaty_partner"],
		tin=body["tax_pin"],
		data_type=body["information_requested"],
		tenant_id=tenant,
		urgency=body.get("urgency", "routine"),
	)
	return _ok(result, 201)


@tax_bp.get("/eoi")
@handle_errors
def list_eoi():
	tenant = _tenant()
	results = [v for (tid, _), v in _svc._eoi_requests.items() if tid == tenant and hasattr(v, "model_dump")]
	return _ok([r.model_dump(mode="json") for r in results])


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

@tax_bp.get("/reports/dashboard")
@handle_errors
def dashboard():
	tenant = _tenant()
	return _ok(_svc.dashboard_summary(tenant))


@tax_bp.get("/reports/revenue")
@handle_errors
def revenue_report():
	tenant = _tenant()
	period = request.args.get("period", str(date.today().year))
	tax_type = request.args.get("tax_type")
	return _ok(_svc.revenue_collection_report(period, tax_type, tenant_id=tenant))


@tax_bp.get("/reports/compliance")
@handle_errors
def compliance_report():
	tenant = _tenant()
	period = request.args.get("period", str(date.today().year))
	sector = request.args.get("sector")
	return _ok(_svc.compliance_rate_report(period, sector, tenant_id=tenant))


@tax_bp.get("/reports/delinquency")
@handle_errors
def delinquency_report():
	tenant = _tenant()
	as_of = request.args.get("as_of", date.today().isoformat())
	return _ok(_svc.delinquency_report(as_of, tenant_id=tenant))


@tax_bp.get("/reports/audits")
@handle_errors
def audit_analytics():
	tenant = _tenant()
	period = request.args.get("period", str(date.today().year))
	return _ok(_svc.audit_case_analytics(period, tenant_id=tenant))


@tax_bp.get("/reports/refunds")
@handle_errors
def refund_analytics():
	tenant = _tenant()
	period = request.args.get("period", str(date.today().year))
	return _ok(_svc.refund_analytics(period, tenant_id=tenant))


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@tax_bp.get("/health")
def health():
	return _ok({"status": "ok", "capability": "government_tax", "version": "1.0.0"})
