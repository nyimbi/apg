"""Complete REST API Blueprint for Lease Management (realestate_lea).

Base path: /realestate/lea

Endpoints
---------
GET    /dashboard                         → KPI dashboard + expiry pipeline
GET    /leases                            → list with filter/sort/page
POST   /leases                            → create lease (IFRS16 optional)
GET    /leases/<id>                       → full detail
PUT    /leases/<id>                       → partial update
DELETE /leases/<id>                       → soft delete
POST   /leases/<id>/activate              → activate
POST   /leases/<id>/execute               → execute / sign
POST   /leases/<id>/surrender             → surrender
POST   /leases/<id>/terminate             → terminate
POST   /leases/<id>/renew                 → renew (creates successor lease)
POST   /leases/<id>/classify-ifrs16       → classify as finance/operating
POST   /leases/<id>/calculate-rou         → calculate ROU asset
POST   /leases/<id>/calculate-liability   → calculate lease liability
POST   /leases/<id>/amortise              → amortise ROU asset for a period
POST   /leases/<id>/interest-expense      → interest expense for a period
POST   /leases/<id>/process-payment       → process a lease payment
POST   /leases/<id>/remeasure             → modification remeasurement
POST   /leases/<id>/journal-entries       → IFRS16 journal entries for period

GET    /abstraction                       → list abstractions
POST   /abstraction                       → create abstraction record
POST   /abstraction/<id>/verify           → verify abstraction

GET    /escalations                       → list (?lease_id)
POST   /escalations                       → create escalation clause
POST   /escalations/<id>/apply            → apply escalation

GET    /options                           → list options (?lease_id)
POST   /options                           → create option
GET    /options/expiring                  → options expiring soon (?days=180)
POST   /options/<id>/exercise             → exercise option
POST   /options/<id>/assess               → assess renewal/termination certainty

GET    /modifications                     → list modifications (?lease_id)
POST   /modifications                     → create modification
POST   /modifications/<id>/approve        → approve
POST   /modifications/<id>/apply          → apply modification

GET    /rent-reviews                      → list (?lease_id)
POST   /rent-reviews                      → commence rent review
POST   /rent-reviews/<id>/agree           → agree outcome

GET    /subleases                         → list subleases (?head_lease_id)
POST   /subleases                         → create sublease
PUT    /subleases/<id>                    → update sublease

GET    /expiry                            → expiry pipeline (?days=180)
POST   /expiry/<lease_id>/flag            → flag for expiry action

POST   /ifrs16                            → generate IFRS16 schedule
POST   /ifrs16/<id>/reclassify            → reclassify category

GET    /assignments                       → list assignments (?lease_id)
POST   /assignments                       → create assignment
POST   /assignments/<id>/complete         → complete assignment

GET    /reports/portfolio-analytics       → portfolio KPIs
GET    /reports/ifrs16-disclosure         → IFRS16 disclosure notes (?fiscal_year)
GET    /reports/maturity-profile          → rent roll maturity (?years=5)
GET    /reports/walt                      → weighted average lease term
GET    /reports/arrears/<lease_id>        → rent arrears analysis
GET    /reports/cost-analysis             → occupancy cost analysis (?cost_type)
GET    /reports/expiry-pipeline           → full pipeline with 180-day look-ahead
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any

from flask import Blueprint, request, jsonify, Response

from .service import LeaseManagementService, present_str
from .models import (
	LeaseCreate, LeaseUpdate,
	LeaseAbstractionCreate,
	RentEscalationCreate,
	LeaseOptionCreate, LeaseOptionUpdate,
	RentReviewCreate,
	Ifrs16ScheduleCreate,
	LeaseAssignmentCreate,
	LeaseModificationCreate, LeaseModificationUpdate,
	SubleaseCreate, SubleaseUpdate,
	LeaseExpiryCreate,
	Ifrs16Category,
	LeaseModificationRequest,
)
from .domain.rules import RuleViolation

bp = Blueprint("lea_api", __name__, url_prefix="/realestate/lea")

# Module-level service — in production, replace with app.extensions or DI
_svc = LeaseManagementService()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro) -> Any:
	"""Run a coroutine in the current event loop or a new one."""
	try:
		loop = asyncio.get_event_loop()
		if loop.is_closed():
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
		return loop.run_until_complete(coro)
	except RuntimeError:
		return asyncio.run(coro)


def _tenant() -> str:
	"""Extract tenant from request header."""
	return request.headers.get("X-Tenant-ID", "default")


def _actor() -> str:
	"""Extract actor/user from request header."""
	return request.headers.get("X-Actor-ID", request.headers.get("X-User-ID", "system"))


def _ok(data: Any, status: int = 200) -> tuple[Response, int]:
	return jsonify({"status": "ok", "data": data}), status


def _err(msg: str, status: int = 400) -> tuple[Response, int]:
	return jsonify({"status": "error", "message": msg}), status


def _body() -> dict[str, Any]:
	"""Parse JSON body, return empty dict on failure."""
	try:
		return request.get_json(force=True) or {}
	except Exception:
		return {}


def _paginate(items: list, page: int, per_page: int) -> dict[str, Any]:
	total = len(items)
	start = (page - 1) * per_page
	end = start + per_page
	return {
		"items": items[start:end],
		"total": total,
		"page": page,
		"per_page": per_page,
		"pages": max(1, (total + per_page - 1) // per_page),
	}


def _handle(fn):
	"""Decorator: catch ValueError, RuleViolation, pydantic ValidationError and return 400/422."""
	from functools import wraps
	@wraps(fn)
	def wrapper(*args, **kwargs):
		try:
			return fn(*args, **kwargs)
		except RuleViolation as e:
			return _err(f"{e.rule_name}: {e.reason}", 422)
		except ValueError as e:
			return _err(str(e), 400)
		except Exception as e:
			return _err(f"internal error: {e}", 500)
	return wrapper


# ===========================================================================
# DASHBOARD
# ===========================================================================

@bp.get("/dashboard")
def dashboard():
	"""KPI dashboard: expiry pipeline + expiring options + portfolio snapshot."""
	tenant = _tenant()
	pipeline = _run(_svc.get_expiry_pipeline(tenant, months_ahead=6))
	options = _run(_svc.get_expiring_options(tenant, days_ahead=180))
	summary = _run(_svc.lease_portfolio_summary({"tenant_id": tenant}))
	return _ok({
		"expiry_pipeline": pipeline,
		"expiring_options": [o.model_dump() for o in options],
		"portfolio_summary": summary,
	})


# ===========================================================================
# LEASES — CRUD + lifecycle
# ===========================================================================

@bp.get("/leases")
def list_leases():
	"""List leases with optional filters, sorting and pagination."""
	tenant = _tenant()
	page = int(request.args.get("page", 1))
	per_page = min(int(request.args.get("per_page", 50)), 200)
	leases = _run(_svc.list_leases(
		tenant_id=tenant,
		property_id=request.args.get("property_id"),
		status=request.args.get("status"),
	))
	items = [l.model_dump() for l in leases]
	# Client-side sort
	sort_by = request.args.get("sort_by", "created_at")
	sort_dir = request.args.get("sort_dir", "desc")
	items.sort(key=lambda x: str(x.get(sort_by, "")), reverse=(sort_dir == "desc"))
	return _ok(_paginate(items, page, per_page))


@bp.post("/leases")
@_handle
def create_lease():
	"""Create a lease. Set ifrs16_applicable=true to auto-generate IFRS16 schedule."""
	data = _body()
	tenant = _tenant()
	actor = _actor()
	data["tenant_id"] = tenant
	data["created_by"] = data.get("created_by", actor)

	payload = LeaseCreate(**data)
	lease = _run(_svc.create_lease_v2(payload))
	return _ok(lease.model_dump() if hasattr(lease, "model_dump") else lease, 201)


@bp.get("/leases/<lease_id>")
def get_lease(lease_id: str):
	"""Get full lease detail."""
	r = _run(_svc.get_lease(lease_id, _tenant()))
	return _ok(r.model_dump()) if r else _err("lease not found", 404)


@bp.put("/leases/<lease_id>")
@_handle
def update_lease(lease_id: str):
	"""Partial update of mutable lease fields."""
	r = _run(_svc.update_lease(lease_id, _tenant(), LeaseUpdate(**_body())))
	return _ok(r.model_dump()) if r else _err("lease not found", 404)


@bp.delete("/leases/<lease_id>")
@_handle
def delete_lease(lease_id: str):
	"""Soft delete a lease."""
	r = _run(_svc.soft_delete_lease(lease_id, _tenant(), _actor()))
	return _ok({"deleted": True, "id": lease_id}) if r else _err("lease not found", 404)


@bp.post("/leases/<lease_id>/activate")
@_handle
def activate_lease(lease_id: str):
	r = _run(_svc.activate_lease(lease_id, _tenant()))
	return _ok(r.model_dump()) if r else _err("lease not found", 404)


@bp.post("/leases/<lease_id>/execute")
@_handle
def execute_lease(lease_id: str):
	data = _body()
	result = _run(_svc.execute_lease(
		lease_id=lease_id,
		executed_by=data.get("executed_by", _actor()),
		execution_date=data.get("execution_date"),
	))
	return _ok(result)


@bp.post("/leases/<lease_id>/surrender")
@_handle
def surrender_lease(lease_id: str):
	data = _body()
	result = _run(_svc.surrender_lease(
		lease_id=lease_id,
		surrender_date=data.get("surrender_date"),
		agreed_compensation=Decimal(str(data.get("agreed_compensation", "0"))),
	))
	return _ok(result)


@bp.post("/leases/<lease_id>/terminate")
@_handle
def terminate_lease(lease_id: str):
	data = _body()
	result = _run(_svc.terminate_lease(
		lease_id=lease_id,
		termination_type=data.get("termination_type", "expiry"),
		effective_date=data.get("effective_date"),
		notice_date=data.get("notice_date"),
	))
	return _ok(result)


@bp.post("/leases/<lease_id>/renew")
@_handle
def renew_lease(lease_id: str):
	data = _body()
	result = _run(_svc.renew_lease(
		lease_id=lease_id,
		new_terms=data.get("new_terms", {}),
		renewal_date=data.get("renewal_date"),
	))
	return _ok(result, 201)


# ---------------------------------------------------------------------------
# Lease — IFRS 16 actions
# ---------------------------------------------------------------------------

@bp.post("/leases/<lease_id>/classify-ifrs16")
@_handle
def classify_ifrs16(lease_id: str):
	"""Classify a lease as finance or operating under IFRS 16."""
	result = _run(_svc.classify_lease_ifrs16(lease_id))
	return _ok(result)


@bp.post("/leases/<lease_id>/calculate-rou")
@_handle
def calculate_rou(lease_id: str):
	"""Calculate ROU asset at commencement."""
	result = _run(_svc.calculate_rou_asset(lease_id))
	return _ok(result)


@bp.post("/leases/<lease_id>/calculate-liability")
@_handle
def calculate_liability(lease_id: str):
	data = _body()
	result = _run(_svc.calculate_lease_liability(
		lease_id,
		discount_rate=float(data["discount_rate"]) if "discount_rate" in data else None,
	))
	return _ok(result)


@bp.post("/leases/<lease_id>/amortise")
@_handle
def amortise_rou(lease_id: str):
	data = _body()
	period = data.get("period")
	if not period:
		return _err("period required (YYYY-MM)", 400)
	result = _run(_svc.amortise_rou_asset(lease_id, period))
	return _ok(result)


@bp.post("/leases/<lease_id>/interest-expense")
@_handle
def interest_expense(lease_id: str):
	data = _body()
	period = data.get("period")
	if not period:
		return _err("period required (YYYY-MM)", 400)
	result = _run(_svc.calculate_interest_expense(lease_id, period))
	return _ok(result)


@bp.post("/leases/<lease_id>/process-payment")
@_handle
def process_payment(lease_id: str):
	data = _body()
	result = _run(_svc.process_lease_payment(
		lease_id=lease_id,
		payment_amount=Decimal(str(data["payment_amount"])),
		payment_date=data.get("payment_date"),
	))
	return _ok(result)


@bp.post("/leases/<lease_id>/remeasure")
@_handle
def remeasure(lease_id: str):
	"""Trigger lease liability remeasurement after a modification event."""
	data = _body()
	result = _run(_svc.lease_modification_remeasurement(
		lease_id=lease_id,
		event_type=data.get("event_type", "revised_payment"),
		new_terms=data.get("new_terms", {}),
	))
	return _ok(result)


@bp.post("/leases/<lease_id>/journal-entries")
@_handle
def journal_entries(lease_id: str):
	data = _body()
	period = data.get("period")
	if not period:
		return _err("period required (YYYY-MM)", 400)
	result = _run(_svc.ifrs16_journal_entries(lease_id, period))
	return _ok(result)


@bp.post("/leases/<lease_id>/handle-modification")
@_handle
def handle_modification(lease_id: str):
	"""Full modification handler: create, approve, apply and remeasure in one call."""
	data = _body()
	req = LeaseModificationRequest(**data)
	result = _run(_svc.handle_lease_modification(lease_id, req))
	return _ok(result)


@bp.post("/leases/<lease_id>/assess-extension")
@_handle
def assess_extension(lease_id: str):
	"""Assess whether a lease extension option is reasonably certain."""
	data = _body()
	result = _run(_svc.assess_lease_extension_option(
		lease_id=lease_id,
		option_id=data.get("option_id"),
		assessment_data=data,
	))
	return _ok(result)


@bp.post("/leases/<lease_id>/cpi-remeasure")
@_handle
def cpi_remeasure(lease_id: str):
	"""Apply CPI-indexed remeasurement to a variable payment lease."""
	data = _body()
	result = _run(_svc.apply_cpi_remeasurement(
		lease_id=lease_id,
		current_cpi=Decimal(str(data["current_cpi"])),
		actor_id=_actor(),
	))
	return _ok(result)


# ===========================================================================
# ABSTRACTION
# ===========================================================================

@bp.get("/abstraction")
def list_abstractions():
	tenant = _tenant()
	results = _run(_svc.list_abstractions(tenant, request.args.get("lease_id")))
	return _ok([r.model_dump() for r in results])


@bp.post("/abstraction")
@_handle
def create_abstraction():
	data = _body()
	data["tenant_id"] = _tenant()
	r = _run(_svc.create_abstraction(LeaseAbstractionCreate(**data)))
	return _ok(r.model_dump(), 201)


@bp.post("/abstraction/<abstraction_id>/verify")
@_handle
def verify_abstraction(abstraction_id: str):
	data = _body()
	r = _run(_svc.verify_abstraction(
		abstraction_id, _tenant(), data.get("verified_by", _actor())
	))
	return _ok(r.model_dump()) if r else _err("abstraction not found", 404)


# ===========================================================================
# ESCALATIONS
# ===========================================================================

@bp.get("/escalations")
def list_escalations():
	return _ok([e.model_dump() for e in _run(
		_svc.list_escalations(_tenant(), request.args.get("lease_id"))
	)])


@bp.post("/escalations")
@_handle
def create_escalation():
	data = _body()
	data["tenant_id"] = _tenant()
	data["created_by"] = data.get("created_by", _actor())
	r = _run(_svc.create_escalation(RentEscalationCreate(**data)))
	return _ok(r.model_dump(), 201)


@bp.post("/escalations/<escalation_id>/apply")
@_handle
def apply_escalation(escalation_id: str):
	data = _body()
	r = _run(_svc.apply_escalation(
		escalation_id, _tenant(), data.get("applied_by", _actor())
	))
	return _ok(r.model_dump()) if r else _err("not found or already applied", 404)


# ===========================================================================
# OPTIONS
# ===========================================================================

@bp.get("/options")
def list_options():
	tenant = _tenant()
	lease_id = request.args.get("lease_id")
	results = _run(_svc.list_options(tenant, lease_id))
	return _ok([o.model_dump() for o in results])


@bp.post("/options")
@_handle
def create_option():
	data = _body()
	data["tenant_id"] = _tenant()
	data["created_by"] = data.get("created_by", _actor())
	r = _run(_svc.create_option(LeaseOptionCreate(**data)))
	return _ok(r.model_dump(), 201)


@bp.get("/options/expiring")
def expiring_options():
	days = int(request.args.get("days", 180))
	return _ok([o.model_dump() for o in _run(
		_svc.get_expiring_options(_tenant(), days)
	)])


@bp.post("/options/<option_id>/exercise")
@_handle
def exercise_option(option_id: str):
	data = _body()
	r = _run(_svc.exercise_option(
		option_id, _tenant(), data.get("notice_served", False)
	))
	return _ok(r.model_dump()) if r else _err("option not found", 404)


@bp.post("/options/<option_id>/assess")
@_handle
def assess_option(option_id: str):
	data = _body()
	result = _run(_svc.assess_option(option_id, _tenant(), data, _actor()))
	return _ok(result)


# ===========================================================================
# MODIFICATIONS
# ===========================================================================

@bp.get("/modifications")
def list_modifications():
	tenant = _tenant()
	lease_id = request.args.get("lease_id")
	results = _run(_svc.list_modifications(tenant, lease_id))
	return _ok([r.model_dump() for r in results])


@bp.post("/modifications")
@_handle
def create_modification():
	data = _body()
	data["tenant_id"] = _tenant()
	data["created_by"] = data.get("created_by", _actor())
	r = _run(_svc.create_modification(LeaseModificationCreate(**data)))
	return _ok(r.model_dump(), 201)


@bp.post("/modifications/<mod_id>/approve")
@_handle
def approve_modification(mod_id: str):
	data = _body()
	r = _run(_svc.approve_modification(
		mod_id, _tenant(), data.get("approved_by", _actor())
	))
	return _ok(r.model_dump()) if r else _err("modification not found", 404)


@bp.post("/modifications/<mod_id>/apply")
@_handle
def apply_modification(mod_id: str):
	r = _run(_svc.apply_modification(mod_id, _tenant(), _actor()))
	return _ok(r) if r else _err("modification not found or not approved", 404)


# ===========================================================================
# RENT REVIEWS
# ===========================================================================

@bp.get("/rent-reviews")
def list_rent_reviews():
	tenant = _tenant()
	lease_id = request.args.get("lease_id")
	results = _run(_svc.list_rent_reviews(tenant, lease_id))
	return _ok([r.model_dump() for r in results])


@bp.post("/rent-reviews")
@_handle
def commence_rent_review():
	data = _body()
	data["tenant_id"] = _tenant()
	data["created_by"] = data.get("created_by", _actor())
	r = _run(_svc.commence_rent_review(RentReviewCreate(**data)))
	return _ok(r.model_dump(), 201)


@bp.post("/rent-reviews/<review_id>/agree")
@_handle
def agree_rent_review(review_id: str):
	data = _body()
	r = _run(_svc.agree_rent_review(
		review_id, _tenant(),
		Decimal(str(data["agreed_rent"])),
		data.get("backdating_authorised_by"),
	))
	return _ok(r.model_dump()) if r else _err("rent review not found", 404)


@bp.get("/leases/<lease_id>/rent-review-schedule")
def rent_review_schedule(lease_id: str):
	result = _run(_svc.rent_review_schedule(lease_id))
	return _ok(result)


# ===========================================================================
# SUBLEASES
# ===========================================================================

@bp.get("/subleases")
def list_subleases():
	tenant = _tenant()
	head_lease_id = request.args.get("head_lease_id")
	results = _run(_svc.list_subleases(tenant, head_lease_id))
	return _ok([r.model_dump() for r in results])


@bp.post("/subleases")
@_handle
def create_sublease():
	data = _body()
	data["tenant_id"] = _tenant()
	data["created_by"] = data.get("created_by", _actor())
	r = _run(_svc.create_sublease_record(SubleaseCreate(**data)))
	return _ok(r.model_dump(), 201)


@bp.put("/subleases/<sublease_id>")
@_handle
def update_sublease(sublease_id: str):
	r = _run(_svc.update_sublease(sublease_id, _tenant(), SubleaseUpdate(**_body())))
	return _ok(r.model_dump()) if r else _err("sublease not found", 404)


# ===========================================================================
# EXPIRY PIPELINE
# ===========================================================================

@bp.get("/expiry")
def expiry_pipeline():
	days = int(request.args.get("days", 180))
	months = max(1, days // 30)
	result = _run(_svc.lease_expiry_pipeline(days_ahead=days))
	return _ok(result)


@bp.post("/expiry/<lease_id>/flag")
@_handle
def flag_expiry(lease_id: str):
	data = _body()
	data["tenant_id"] = _tenant()
	data["lease_id"] = lease_id
	data["created_by"] = data.get("created_by", _actor())
	r = _run(_svc.flag_lease_expiry(LeaseExpiryCreate(**data)))
	return _ok(r.model_dump(), 201)


# ===========================================================================
# IFRS 16 SCHEDULE
# ===========================================================================

@bp.post("/ifrs16")
@_handle
def generate_ifrs16():
	data = _body()
	data["tenant_id"] = _tenant()
	r = _run(_svc.generate_ifrs16_schedule(Ifrs16ScheduleCreate(**data)))
	return _ok(r.model_dump(), 201)


@bp.post("/ifrs16/<schedule_id>/reclassify")
@_handle
def reclassify_ifrs16(schedule_id: str):
	data = _body()
	r = _run(_svc.reclassify_ifrs16(
		schedule_id, _tenant(),
		Ifrs16Category(data["new_category"]),
		data["auditor_approved_by"],
	))
	return _ok(r.model_dump()) if r else _err("schedule not found", 404)


# ===========================================================================
# ASSIGNMENTS
# ===========================================================================

@bp.get("/assignments")
def list_assignments():
	tenant = _tenant()
	lease_id = request.args.get("lease_id")
	results = _run(_svc.list_assignments(tenant, lease_id))
	return _ok([r.model_dump() for r in results])


@bp.post("/assignments")
@_handle
def create_assignment():
	data = _body()
	data["tenant_id"] = _tenant()
	data["created_by"] = data.get("created_by", _actor())
	r = _run(_svc.create_assignment(LeaseAssignmentCreate(**data)))
	return _ok(r.model_dump(), 201)


@bp.post("/assignments/<assignment_id>/complete")
@_handle
def complete_assignment(assignment_id: str):
	r = _run(_svc.complete_assignment(assignment_id, _tenant()))
	return _ok(r.model_dump()) if r else _err("assignment not found", 404)


# ===========================================================================
# REPORTS
# ===========================================================================

@bp.get("/reports/portfolio-analytics")
def report_portfolio_analytics():
	"""Full portfolio analytics dashboard."""
	result = _run(_svc.portfolio_lease_analytics(_tenant()))
	return _ok(result)


@bp.get("/reports/ifrs16-disclosure")
def report_ifrs16_disclosure():
	"""IFRS 16.53–59 disclosure notes for a fiscal year."""
	fiscal_year = request.args.get("fiscal_year", str(__import__("datetime").date.today().year))
	result = _run(_svc.ifrs16_disclosure_notes(fiscal_year))
	return _ok(result)


@bp.get("/reports/maturity-profile")
def report_maturity_profile():
	"""Rent roll maturity by year."""
	years = int(request.args.get("years", 5))
	result = _run(_svc.lease_maturity_profile(years))
	return _ok(result)


@bp.get("/reports/walt")
def report_walt():
	"""Weighted Average Lease Term (years)."""
	tenant = _tenant()
	walt = _run(_svc.weighted_average_lease_term({"tenant_id": tenant}))
	return _ok({"walt_years": walt, "tenant_id": tenant})


@bp.get("/reports/arrears/<lease_id>")
def report_arrears(lease_id: str):
	"""Rent arrears aged analysis."""
	as_of = request.args.get("as_of_date", str(__import__("datetime").date.today()))
	result = _run(_svc.calculate_rent_arrears(lease_id, as_of))
	return _ok(result)


@bp.get("/reports/cost-analysis")
def report_cost_analysis():
	"""Occupancy cost analysis across portfolio."""
	cost_type = request.args.get("cost_type", "total_occupancy_cost")
	result = _run(_svc.lease_cost_analysis(cost_type))
	return _ok(result)


@bp.get("/reports/expiry-pipeline")
def report_expiry_pipeline():
	"""Full expiry pipeline with 180-day look-ahead."""
	days = int(request.args.get("days", 180))
	result = _run(_svc.lease_expiry_pipeline(days_ahead=days))
	return _ok(result)


@bp.get("/reports/service-charge/<property_id>")
def report_service_charge(property_id: str):
	"""Service charge reconciliation for a property."""
	period = request.args.get("period", str(__import__("datetime").date.today().year))
	result = _run(_svc.service_charge_reconciliation(property_id, period))
	return _ok(result)


# ---------------------------------------------------------------------------
# API Blueprint alias for compatibility
# ---------------------------------------------------------------------------
api_bp = bp

__all__ = ["bp", "api_bp"]
