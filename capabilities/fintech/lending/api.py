"""
Flask Blueprint REST API for APG Digital Lending.

Registers at /api/v1/lending — full CRUD + lifecycle operations for all entities.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from __future__ import annotations

import functools
import traceback
from datetime import date
from typing import Any

from flask import Blueprint, jsonify, request

try:
	from .service import LendingService
	from .models import (
		LoanApplicationCreate, LoanApplicationUpdate,
		LoanProductCreate, LoanProductUpdate,
		RepaymentTransactionCreate,
		CollateralItemCreate, GuarantorRecordCreate,
		RestructureCreate, WriteOffCreate,
		AmortisationScheduleRequest,
	)
	from .domain.rules import RuleViolation
except ImportError:  # pragma: no cover
	from service import LendingService  # type: ignore
	from models import (  # type: ignore
		LoanApplicationCreate, LoanApplicationUpdate,
		LoanProductCreate, LoanProductUpdate,
		RepaymentTransactionCreate,
		CollateralItemCreate, GuarantorRecordCreate,
		RestructureCreate, WriteOffCreate,
		AmortisationScheduleRequest,
	)
	from domain.rules import RuleViolation  # type: ignore


# ---------------------------------------------------------------------------
# Blueprint & shared service
# ---------------------------------------------------------------------------

lending_bp = Blueprint("lending", __name__, url_prefix="/api/v1/lending")

# Dependency-light singleton — swap for DI in production
_SERVICE = LendingService()


def _svc() -> LendingService:
	return _SERVICE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok(data: Any, status: int = 200):
	return jsonify({"status": "ok", "data": data}), status


def _err(msg: str, detail: str | None = None, status: int = 400):
	body: dict[str, Any] = {"status": "error", "error": msg}
	if detail:
		body["detail"] = detail
	return jsonify(body), status


def _paginate(items: list[Any]) -> dict[str, Any]:
	page = max(1, request.args.get("page", 1, type=int))
	page_size = min(200, max(1, request.args.get("page_size", 20, type=int)))
	total = len(items)
	start = (page - 1) * page_size
	sliced = items[start : start + page_size]
	import math
	return {
		"items": sliced,
		"total": total,
		"page": page,
		"page_size": page_size,
		"pages": max(1, math.ceil(total / page_size)),
	}


def _handle(fn):
	"""Decorator: catches common exceptions and serialises them."""
	@functools.wraps(fn)
	def wrapper(*args, **kwargs):
		try:
			return fn(*args, **kwargs)
		except RuleViolation as exc:
			return _err(exc.rule_name, exc.reason, 422)
		except KeyError as exc:
			return _err("not_found", str(exc), 404)
		except (ValueError, AssertionError) as exc:
			return _err("validation_error", str(exc), 400)
		except PermissionError as exc:
			return _err("permission_denied", str(exc), 403)
		except Exception as exc:  # noqa: BLE001
			return _err("internal_error", traceback.format_exc(), 500)
	return wrapper


def _tenant() -> str:
	return (
		request.headers.get("X-Tenant-Id")
		or request.args.get("tenant_id", "default")
	)


def _actor() -> str:
	return (
		request.headers.get("X-Actor-Id")
		or request.args.get("actor_id", "system")
	)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@lending_bp.get("/health")
@_handle
def health():
	"""Capability health check."""
	return _ok({"capability": "fintech_lending", "status": "healthy"})


# ---------------------------------------------------------------------------
# Loan Products
# ---------------------------------------------------------------------------

@lending_bp.get("/products")
@_handle
def list_products():
	"""List loan products. ?active_only=true (default)."""
	active_only = request.args.get("active_only", "true").lower() != "false"
	items = _svc().list_products(active_only=active_only)
	return _ok(_paginate(items))


@lending_bp.post("/products")
@_handle
def create_product():
	"""Create a new loan product."""
	payload = request.get_json(force=True) or {}
	p = LoanProductCreate(**payload)
	result = _svc().create_loan_product(
		product_code=p.code,
		name=p.name,
		product_type=p.product_type,
		rate_type="reducing_balance",
		min_amount=p.min_amount,
		max_amount=p.max_amount,
		min_tenor=p.min_tenor_months,
		max_tenor=p.max_tenor_months,
		fees=[],
		tenant_id=p.tenant_id,
		owner_id=p.created_by,
		currency=p.currency,
		repayment_frequency=p.repayment_frequency,
	)
	return _ok(result, 201)


@lending_bp.get("/products/<product_id>")
@_handle
def get_product(product_id: str):
	"""Get product by ID/code."""
	product = _svc()._require_product(product_id)
	return _ok(product.to_dict())


@lending_bp.put("/products/<product_id>")
@_handle
def update_product(product_id: str):
	"""Update product rates."""
	payload = request.get_json(force=True) or {}
	p = LoanProductUpdate(**payload)
	updated: dict[str, float] = {}
	if p.base_annual_rate is not None:
		updated["annual_rate"] = p.base_annual_rate
	result = _svc().update_product_rates(
		product_code=product_id,
		new_rates=updated,
		effective_date=date.today().isoformat(),
	)
	return _ok(result)


@lending_bp.get("/products/<product_id>/performance")
@_handle
def product_performance(product_id: str):
	"""Product performance report."""
	period = request.args.get("period", date.today().strftime("%Y-%m"))
	return _ok(_svc().product_performance_report(product_id, period))


# ---------------------------------------------------------------------------
# Loan Applications
# ---------------------------------------------------------------------------

@lending_bp.get("/applications")
@_handle
def list_applications():
	"""List applications with optional filters."""
	filters: dict[str, Any] = {}
	for key in ("status", "borrower_id", "product_id", "purpose"):
		val = request.args.get(key)
		if val:
			filters[key] = val
	items = _svc().list_applications(filters=filters, tenant_id=_tenant())
	return _ok(_paginate(items))


@lending_bp.post("/applications")
@_handle
def submit_application():
	"""Submit a new loan application."""
	payload = request.get_json(force=True) or {}
	p = LoanApplicationCreate(**payload)
	from uuid6 import uuid7
	app_id = str(uuid7())
	result = _svc().submit_application(
		application_id=app_id,
		tenant_id=p.tenant_id,
		borrower_id=p.borrower_id,
		product_id=p.product_id,
		requested_amount=p.requested_amount,
		purpose=p.purpose,
		affordability_reference=p.bank_statement_ref,
		bank_statement_reference=p.bank_statement_ref,
		aml_reference=p.aml_ref,
		fraud_reference=p.fraud_ref,
		behavior_evidence_reference="",
		human_review="",
	)
	return _ok(result, 201)


@lending_bp.get("/applications/<application_id>")
@_handle
def get_application(application_id: str):
	"""Get application detail with linked underwriting/documents."""
	return _ok(_svc().retrieve_application(application_id))


@lending_bp.put("/applications/<application_id>")
@_handle
def update_application(application_id: str):
	"""Partial application update (status, notes, amount)."""
	payload = request.get_json(force=True) or {}
	LoanApplicationUpdate(**payload)  # validate
	app = _svc()._require_application(application_id)
	if "notes" in payload:
		app.notes = payload["notes"]  # type: ignore[attr-defined]
	return _ok(app.to_dict())


@lending_bp.delete("/applications/<application_id>")
@_handle
def withdraw_application(application_id: str):
	"""Withdraw an application (soft delete / status change)."""
	payload = request.get_json(force=True) or {}
	reason = payload.get("reason", "borrower_requested")
	return _ok(_svc().withdraw_application(application_id, reason))


@lending_bp.post("/applications/<application_id>/assign-underwriter")
@_handle
def assign_underwriter(application_id: str):
	"""Assign underwriter to application."""
	payload = request.get_json(force=True) or {}
	underwriter_id = payload.get("underwriter_id", _actor())
	return _ok(_svc().assign_underwriter(application_id, underwriter_id))


@lending_bp.post("/applications/<application_id>/request-documents")
@_handle
def request_documents(application_id: str):
	"""Request additional documents for an application."""
	payload = request.get_json(force=True) or {}
	docs = payload.get("documents", [])
	return _ok(_svc().request_documents(application_id, docs))


@lending_bp.post("/applications/<application_id>/site-visit")
@_handle
def record_site_visit(application_id: str):
	"""Record a site inspection visit."""
	payload = request.get_json(force=True) or {}
	return _ok(_svc().record_site_visit(
		application_id,
		visit_notes=payload.get("notes", ""),
		inspector_id=payload.get("inspector_id", _actor()),
		visit_date=payload.get("visit_date", date.today().isoformat()),
	))


@lending_bp.post("/applications/<application_id>/underwrite")
@_handle
def underwrite(application_id: str):
	"""Record underwriting decision."""
	payload = request.get_json(force=True) or {}
	return _ok(_svc().underwriting_decision(
		application_id=application_id,
		decision=payload.get("decision", "refer"),
		conditions=payload.get("conditions", []),
		underwriter_id=payload.get("underwriter_id", _actor()),
	))


@lending_bp.post("/applications/<application_id>/generate-offers")
@_handle
def generate_offers(application_id: str):
	"""Generate tiered loan offers for an approved application."""
	return _ok(_svc().generate_loan_offers(application_id))


@lending_bp.get("/applications/analytics")
@_handle
def application_analytics():
	"""Aggregate analytics for a given period."""
	period = request.args.get("period", date.today().strftime("%Y-%m"))
	return _ok(_svc().application_analytics(period))


# ---------------------------------------------------------------------------
# Credit Assessment
# ---------------------------------------------------------------------------

@lending_bp.post("/credit/score")
@_handle
def calculate_credit_score():
	"""Calculate composite credit score for a customer."""
	payload = request.get_json(force=True) or {}
	customer_id = payload.get("customer_id", "")
	assert customer_id, "customer_id required"
	return _ok(_svc().credit_score_calculate(customer_id))


@lending_bp.post("/credit/bureau-check")
@_handle
def bureau_check():
	"""Query credit bureau for a customer."""
	payload = request.get_json(force=True) or {}
	return _ok(_svc().credit_bureau_check(
		customer_id=payload.get("customer_id", ""),
		id_number=payload.get("id_number", ""),
		country=payload.get("country", "KE"),
	))


@lending_bp.post("/credit/income-verify")
@_handle
def income_verify():
	"""Verify income for a customer."""
	payload = request.get_json(force=True) or {}
	return _ok(_svc().income_verification(
		customer_id=payload.get("customer_id", ""),
		income_source=payload.get("income_source", "employed"),
		stated_amount=payload.get("stated_amount", 0),
		docs=payload.get("docs", []),
	))


@lending_bp.post("/credit/dsr")
@_handle
def debt_service_ratio():
	"""Calculate debt service ratio for a prospective loan."""
	payload = request.get_json(force=True) or {}
	return _ok(_svc().debt_service_ratio(
		customer_id=payload.get("customer_id", ""),
		new_loan_amount=payload.get("amount", 0),
		new_loan_rate=payload.get("annual_rate", 0.18),
		tenor_months=payload.get("tenor_months", 12),
	))


@lending_bp.post("/credit/eligibility")
@_handle
def loan_eligibility():
	"""Compute loan eligibility for a customer + product."""
	payload = request.get_json(force=True) or {}
	return _ok(_svc().calculate_loan_eligibility(
		customer_id=payload.get("customer_id", ""),
		product_code=payload.get("product_code", ""),
	))


# ---------------------------------------------------------------------------
# Amortisation (standalone calculator)
# ---------------------------------------------------------------------------

@lending_bp.post("/amortisation")
@_handle
def amortisation():
	"""Compute amortisation schedule (no loan required)."""
	payload = request.get_json(force=True) or {}
	req = AmortisationScheduleRequest(**payload)
	try:
		from .domain.calculations import build_amortisation_schedule
	except ImportError:
		from domain.calculations import build_amortisation_schedule  # type: ignore
	sched = build_amortisation_schedule(
		principal=req.principal,
		annual_rate=req.annual_rate,
		tenor_months=req.tenor_months,
		start_date=req.start_date,
		schedule_type=req.schedule_type,
	)
	sched["currency"] = req.currency
	return _ok(sched)


# ---------------------------------------------------------------------------
# Loans
# ---------------------------------------------------------------------------

@lending_bp.get("/loans")
@_handle
def list_loans():
	"""List loans. ?status=active&borrower_id=xxx"""
	status = request.args.get("status")
	borrower_id = request.args.get("borrower_id")
	items = _svc().list_loans(status=status, borrower_id=borrower_id)
	return _ok(_paginate(items))


@lending_bp.post("/loans/disburse")
@_handle
def disburse_loan():
	"""Disburse a loan from an approved application."""
	payload = request.get_json(force=True) or {}
	from uuid6 import uuid7
	loan_id = payload.get("loan_id", str(uuid7()))
	return _ok(_svc().disburse_loan(
		loan_id=loan_id,
		application_id=payload.get("application_id", ""),
		bank_account=payload.get("bank_account", ""),
		disbursement_date=payload.get("disbursement_date", date.today().isoformat()),
	), 201)


@lending_bp.get("/loans/<loan_id>")
@_handle
def get_loan(loan_id: str):
	"""Get loan detail."""
	return _ok(_svc()._require_loan(loan_id).to_dict())


@lending_bp.get("/loans/<loan_id>/statement")
@_handle
def loan_statement(loan_id: str):
	"""Full loan statement: all transactions, fees, restructures, collateral."""
	return _ok(_svc().get_loan_statement(loan_id))


@lending_bp.get("/loans/<loan_id>/schedule")
@_handle
def loan_schedule(loan_id: str):
	"""Repayment schedule for a loan."""
	schedule_type = request.args.get("schedule_type", "reducing_balance")
	return _ok(_svc().generate_repayment_schedule(loan_id, schedule_type))


@lending_bp.get("/loans/<loan_id>/dpd")
@_handle
def loan_dpd(loan_id: str):
	"""Days Past Due per installment."""
	return _ok(_svc().calculate_dpd(loan_id))


@lending_bp.post("/loans/<loan_id>/repay")
@_handle
def process_repayment(loan_id: str):
	"""Apply a repayment to a loan."""
	payload = request.get_json(force=True) or {}
	p = RepaymentTransactionCreate(loan_id=loan_id, **{k: v for k, v in payload.items() if k != "loan_id"})
	return _ok(_svc().process_repayment(
		loan_id=loan_id,
		amount=p.amount,
		payment_date=p.payment_date.isoformat(),
		payment_method=p.payment_method,
		reference=p.reference,
	))


@lending_bp.get("/loans/<loan_id>/early-settlement")
@_handle
def early_settlement(loan_id: str):
	"""Calculate early settlement amount."""
	settlement_date = request.args.get("settlement_date", date.today().isoformat())
	return _ok(_svc().early_settlement(loan_id, settlement_date))


@lending_bp.post("/loans/<loan_id>/restructure")
@_handle
def restructure_loan(loan_id: str):
	"""Restructure a loan (extend tenor, reduce rate, capitalise arrears)."""
	payload = request.get_json(force=True) or {}
	p = RestructureCreate(loan_id=loan_id, **{k: v for k, v in payload.items() if k != "loan_id"})
	new_terms: dict[str, Any] = {}
	if p.new_annual_rate is not None:
		new_terms["annual_rate"] = p.new_annual_rate
	if p.new_tenor_months is not None:
		new_terms["tenor_months"] = p.new_tenor_months
	new_terms["capitalise_arrears"] = p.capitalise_arrears
	return _ok(_svc().restructure_loan(
		loan_id=loan_id,
		new_terms=new_terms,
		reason=p.reason,
		approved_by=p.approved_by,
	))


@lending_bp.post("/loans/<loan_id>/write-off")
@_handle
def write_off_loan(loan_id: str):
	"""Write off a non-performing loan."""
	payload = request.get_json(force=True) or {}
	p = WriteOffCreate(loan_id=loan_id, **{k: v for k, v in payload.items() if k != "loan_id"})
	return _ok(_svc().write_off_loan(
		loan_id=loan_id,
		reason=p.reason,
		write_off_date=p.write_off_date.isoformat(),
		approved_by=p.approved_by,
	))


@lending_bp.post("/loans/<loan_id>/close")
@_handle
def close_loan(loan_id: str):
	"""Close a loan."""
	payload = request.get_json(force=True) or {}
	return _ok(_svc().close_loan(loan_id, payload.get("reason", "settled")))


@lending_bp.post("/loans/<loan_id>/fee")
@_handle
def add_fee(loan_id: str):
	"""Charge a fee to a loan."""
	payload = request.get_json(force=True) or {}
	return _ok(_svc().add_loan_fee(
		loan_id=loan_id,
		fee_type=payload.get("fee_type", "other"),
		amount=payload.get("amount", 0),
		reason=payload.get("reason", ""),
	))


@lending_bp.post("/loans/<loan_id>/waive-fee")
@_handle
def waive_fee(loan_id: str):
	"""Waive a fee on a loan."""
	payload = request.get_json(force=True) or {}
	return _ok(_svc().waive_fee_or_penalty(
		loan_id=loan_id,
		fee_id=payload.get("fee_id", ""),
		waiver_reason=payload.get("reason", ""),
		approved_by=payload.get("approved_by", _actor()),
	))


# ---------------------------------------------------------------------------
# Collateral
# ---------------------------------------------------------------------------

@lending_bp.post("/loans/<loan_id>/collateral")
@_handle
def add_collateral(loan_id: str):
	"""Register collateral item against a loan."""
	payload = request.get_json(force=True) or {}
	p = CollateralItemCreate(loan_id=loan_id, **{k: v for k, v in payload.items() if k != "loan_id"})
	from uuid6 import uuid7
	coll_id = str(uuid7())
	from service import _Collateral  # type: ignore
	coll = _Collateral(
		collateral_id=coll_id,
		loan_id=loan_id,
		collateral_type=str(p.collateral_type),
		description=p.description,
		market_value=p.market_value,
		currency=p.currency,
	)
	_svc().collateral[coll_id] = coll
	loan = _svc()._require_loan(loan_id)
	loan.collateral_ids.append(coll_id)
	return _ok(coll.to_dict(), 201)


@lending_bp.post("/loans/<loan_id>/collateral/<collateral_id>/release")
@_handle
def release_collateral(loan_id: str, collateral_id: str):
	"""Release a collateral item."""
	payload = request.get_json(force=True) or {}
	return _ok(_svc().record_collateral_release(
		loan_id=loan_id,
		collateral_id=collateral_id,
		reason=payload.get("reason", "loan_settled"),
		released_by=payload.get("released_by", _actor()),
	))


@lending_bp.post("/collateral/assess")
@_handle
def assess_collateral():
	"""Assess collateral items (no loan required)."""
	payload = request.get_json(force=True) or {}
	items = payload.get("items", [])
	return _ok(_svc().assess_collateral(items))


# ---------------------------------------------------------------------------
# Collections & Delinquency
# ---------------------------------------------------------------------------

@lending_bp.get("/loans/<loan_id>/demand-notice")
@_handle
def issue_demand_notice(loan_id: str):
	"""Issue a demand notice at a given level (1–4)."""
	level = request.args.get("level", 1, type=int)
	return _ok(_svc().generate_demand_notice(loan_id, level))


@lending_bp.post("/loans/<loan_id>/assign-collector")
@_handle
def assign_collector(loan_id: str):
	"""Assign a collections officer to a delinquent loan."""
	payload = request.get_json(force=True) or {}
	return _ok(_svc().assign_to_collector(
		loan_id=loan_id,
		collector_id=payload.get("collector_id", _actor()),
	))


@lending_bp.post("/loans/<loan_id>/collection-activity")
@_handle
def collection_activity(loan_id: str):
	"""Record a collections activity."""
	payload = request.get_json(force=True) or {}
	return _ok(_svc().record_collection_activity(
		loan_id=loan_id,
		activity_type=payload.get("activity_type", "call"),
		outcome=payload.get("outcome", "contacted"),
		notes=payload.get("notes", ""),
		next_action=payload.get("next_action", ""),
	))


@lending_bp.post("/loans/<loan_id>/legal-action")
@_handle
def legal_action(loan_id: str):
	"""Record a legal action against a borrower."""
	payload = request.get_json(force=True) or {}
	return _ok(_svc().legal_action(
		loan_id=loan_id,
		action_type=payload.get("action_type", "file_suit"),
		lawyer_id=payload.get("lawyer_id", ""),
		court_date=payload.get("court_date"),
	))


# ---------------------------------------------------------------------------
# Portfolio & Reports
# ---------------------------------------------------------------------------

@lending_bp.get("/reports/portfolio")
@_handle
def portfolio_report():
	"""Portfolio summary: book, PAR, NPL, yield."""
	as_of = request.args.get("as_of_date")
	return _ok(_svc().portfolio_summary(as_of))


@lending_bp.get("/reports/delinquency")
@_handle
def delinquency_report():
	"""Delinquency bucket report with PAR ratios."""
	as_of = request.args.get("as_of_date")
	return _ok(_svc().delinquency_report(as_of))


@lending_bp.get("/reports/ifrs9")
@_handle
def ifrs9_report():
	"""IFRS 9 ECL provision calculation."""
	method = request.args.get("method", "ifrs9")
	return _ok(_svc().provision_calculation(method))


@lending_bp.get("/reports/vintage")
@_handle
def vintage_report():
	"""Vintage analysis by origination cohort."""
	cohort_months = request.args.get("cohort_months", 12, type=int)
	return _ok(_svc().vintage_analysis(cohort_months))


@lending_bp.get("/reports/concentration")
@_handle
def concentration_report():
	"""Concentration risk by sector, geography, ticket size."""
	return _ok(_svc().concentration_risk_report())


@lending_bp.post("/reports/stress-test")
@_handle
def stress_test():
	"""Run default rate sensitivity scenarios."""
	payload = request.get_json(force=True) or {}
	scenarios = payload.get("scenarios", [
		{"name": "mild",     "additional_default_rate": 0.05, "lgd": 0.40},
		{"name": "moderate", "additional_default_rate": 0.15, "lgd": 0.40},
		{"name": "severe",   "additional_default_rate": 0.30, "lgd": 0.45},
	])
	return _ok(_svc().stress_test(scenarios))


@lending_bp.get("/reports/collection-performance")
@_handle
def collection_performance():
	"""Collections performance report."""
	period = request.args.get("period", date.today().strftime("%Y-%m"))
	collector_id = request.args.get("collector_id")
	return _ok(_svc().collection_performance_report(period, collector_id))


@lending_bp.get("/dashboard")
@_handle
def dashboard():
	"""Aggregated dashboard KPIs."""
	tenant_id = _tenant()
	summary = _svc().dashboard_summary(tenant_id)
	portfolio = _svc().portfolio_summary()
	return _ok({**summary, "portfolio": portfolio})


# ---------------------------------------------------------------------------
# Borrower onboarding
# ---------------------------------------------------------------------------

@lending_bp.post("/borrowers")
@_handle
def onboard_borrower():
	"""Onboard a borrower (KYC link)."""
	payload = request.get_json(force=True) or {}
	from uuid6 import uuid7
	borrower_id = payload.get("borrower_id", str(uuid7()))
	return _ok(_svc().onboard_borrower(
		borrower_id=borrower_id,
		tenant_id=payload.get("tenant_id", _tenant()),
		customer_reference=payload.get("customer_reference", ""),
		kyc_profile_id=payload.get("kyc_profile_id", ""),
		country=payload.get("country", "KE"),
		income_evidence_id=payload.get("income_evidence_id", ""),
		consent_reference=payload.get("consent_reference", ""),
	), 201)


# ---------------------------------------------------------------------------
# App factory helper
# ---------------------------------------------------------------------------

def create_app() -> "Flask":  # noqa: F821
	"""Minimal Flask app for standalone deployment / testing."""
	from flask import Flask
	app = Flask(__name__)
	app.register_blueprint(lending_bp)
	return app
