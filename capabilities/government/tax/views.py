"""Flask Blueprint UI views for APG Tax Administration.

Jinja2-rendered HTML views backed by TaxAdministrationService.
All view functions return template context dicts (for testability);
actual rendering happens via render_template() in the route handlers.

Pydantic v2 view-layer models are defined here per APG conventions.
"""
from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Annotated, Any

from flask import Blueprint, render_template, request, redirect, url_for, flash
from pydantic import AfterValidator, BaseModel, ConfigDict, Field

try:
	from .service import TaxAdministrationService
	from .models import (
		TaxpayerStatus, ReturnStatus, AssessmentStatus, DebtStatus,
		AuditStatus, RefundStatus, ClearanceCertificateStatus, uuid7str,
	)
except ImportError:
	from service import TaxAdministrationService  # type: ignore
	from models import (  # type: ignore
		TaxpayerStatus, ReturnStatus, AssessmentStatus, DebtStatus,
		AuditStatus, RefundStatus, ClearanceCertificateStatus, uuid7str,
	)


# ---------------------------------------------------------------------------
# View-layer Pydantic v2 models
# ---------------------------------------------------------------------------

def _non_empty(v: str) -> str:
	assert v and v.strip(), "must be non-empty"
	return v.strip()


NonEmpty = Annotated[str, AfterValidator(_non_empty)]


class TaxpayerFormModel(BaseModel):
	"""Form validation for taxpayer registration."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	taxpayer_name: NonEmpty
	taxpayer_type: str = "individual"
	tax_pin: str = ""
	national_id: str | None = None
	business_registration_number: str | None = None
	email: str | None = None
	phone: str | None = None
	physical_address: str | None = None
	tax_types: list[str] = Field(default_factory=list)
	evidence_reference: NonEmpty


class ReturnFormModel(BaseModel):
	"""Form validation for return filing."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tax_pin: NonEmpty
	tax_type: str = "income_tax"
	period: NonEmpty
	gross_income: Decimal = Decimal("0")
	allowable_deductions: Decimal = Decimal("0")
	tax_liability: Decimal = Decimal("0")
	tax_credits: Decimal = Decimal("0")
	tax_paid: Decimal = Decimal("0")
	evidence_reference: NonEmpty


class PaymentFormModel(BaseModel):
	"""Form validation for payment recording."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tax_pin: NonEmpty
	amount: Decimal
	payment_method: str = "bank_transfer"
	payment_reference: NonEmpty
	tax_type: str = "income_tax"
	period: str = ""
	assessment_id: str | None = None


class ObjectionFormModel(BaseModel):
	"""Form validation for objection filing."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	assessment_id: NonEmpty
	grounds: NonEmpty
	amount_disputed: Decimal
	tax_pin: str = ""


class DashboardKPIView(BaseModel):
	"""View-layer dashboard KPI model."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	as_of: datetime = Field(default_factory=datetime.utcnow)
	registered_taxpayers: int = 0
	active_taxpayers: int = 0
	returns_filed_ytd: int = 0
	returns_overdue: int = 0
	assessments_pending: int = 0
	total_tax_assessed: Decimal = Decimal("0")
	total_tax_collected: Decimal = Decimal("0")
	total_outstanding_debt: Decimal = Decimal("0")
	open_objections: int = 0
	open_audits: int = 0
	pending_refunds: int = 0
	pending_clearance_certs: int = 0
	compliance_rate: Decimal = Decimal("0")
	collection_rate: Decimal = Decimal("0")
	collection_efficiency_pct: str = "0.00%"
	compliance_pct: str = "0.00%"


# ---------------------------------------------------------------------------
# Blueprint
# ---------------------------------------------------------------------------

views_bp = Blueprint(
	"tax_views",
	__name__,
	url_prefix="/tax",
	template_folder="templates",
	static_folder="static",
)

# Module-level service (same instance as API)
_svc = TaxAdministrationService()


def _tenant() -> str:
	return request.headers.get("X-Tenant-ID", request.args.get("tenant_id", "default"))


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@views_bp.get("/")
@views_bp.get("/dashboard")
def dashboard():
	"""Main tax administration dashboard with KPIs."""
	tenant = _tenant()
	kpi_raw = _svc.dashboard_summary(tenant)
	kpi = DashboardKPIView(
		tenant_id=tenant,
		**{k: v for k, v in kpi_raw.items() if k in DashboardKPIView.model_fields},
	)
	cr = float(kpi.collection_rate) * 100
	comp = float(kpi.compliance_rate) * 100
	kpi = kpi.model_copy(update={
		"collection_efficiency_pct": f"{cr:.1f}%",
		"compliance_pct": f"{comp:.1f}%",
	})

	recent_returns = sorted(
		_svc._returns.tenant_values(tenant),
		key=lambda r: r.created_at,
		reverse=True,
	)[:10]
	recent_payments = sorted(
		_svc._payments.tenant_values(tenant),
		key=lambda p: p.created_at,
		reverse=True,
	)[:10]

	ctx = {
		"title": "Tax Administration Dashboard",
		"kpi": kpi.model_dump(mode="json"),
		"recent_returns": [r.model_dump(mode="json") for r in recent_returns],
		"recent_payments": [p.model_dump(mode="json") for p in recent_payments],
		"tenant_id": tenant,
		"today": date.today().isoformat(),
	}
	try:
		return render_template("dashboards/tax_dashboard.html", **ctx)
	except Exception:
		return ctx  # return context dict in test/headless mode


# ---------------------------------------------------------------------------
# Taxpayers
# ---------------------------------------------------------------------------

@views_bp.get("/taxpayers")
def list_taxpayers():
	"""List all taxpayers with search and filter."""
	tenant = _tenant()
	q = request.args.get("q", "")
	status_filter = request.args.get("status", "")
	page = max(1, int(request.args.get("page", 1)))
	per_page = 25

	taxpayers = [t for t in _svc._taxpayers.tenant_values(tenant) if not t.is_deleted]
	if q:
		ql = q.lower()
		taxpayers = [
			t for t in taxpayers
			if ql in t.taxpayer_name.lower()
			or ql in t.tax_pin.lower()
			or ql in (t.email or "").lower()
		]
	if status_filter:
		taxpayers = [t for t in taxpayers if t.status.value == status_filter]

	total = len(taxpayers)
	start = (page - 1) * per_page
	paged = taxpayers[start: start + per_page]

	ctx = {
		"title": "Taxpayers",
		"taxpayers": [t.model_dump(mode="json") for t in paged],
		"total": total,
		"page": page,
		"per_page": per_page,
		"pages": max(1, (total + per_page - 1) // per_page),
		"q": q,
		"status_filter": status_filter,
		"statuses": [s.value for s in TaxpayerStatus],
		"tenant_id": tenant,
	}
	try:
		return render_template("dashboards/taxpayer_list.html", **ctx)
	except Exception:
		return ctx


@views_bp.get("/taxpayers/<tin>")
def taxpayer_detail(tin: str):
	"""Taxpayer detail: returns, assessments, debts, audits."""
	tenant = _tenant()
	tp = _svc._find_taxpayer_by_pin(tin, tenant)
	if tp is None:
		flash(f"Taxpayer {tin} not found", "error")
		return redirect(url_for("tax_views.list_taxpayers"))

	returns = [r for r in _svc._returns.tenant_values(tenant) if r.tax_pin.upper() == tin.upper()]
	assessments = [a for a in _svc._assessments.tenant_values(tenant) if a.taxpayer_id == tp.id]
	debts = [d for d in _svc._debts.tenant_values(tenant) if d.taxpayer_id == tp.id]
	audits = [a for a in _svc._audits.tenant_values(tenant) if a.taxpayer_id == tp.id]
	payments = [p for p in _svc._payments.tenant_values(tenant) if p.taxpayer_id == tp.id]

	total_debt = sum(d.balance for d in debts if d.status.value in ("outstanding", "partially_paid"))
	total_paid = sum(p.amount for p in payments)

	ctx = {
		"title": f"Taxpayer: {tp.taxpayer_name}",
		"taxpayer": tp.model_dump(mode="json"),
		"returns": [r.model_dump(mode="json") for r in sorted(returns, key=lambda r: r.created_at, reverse=True)],
		"assessments": [a.model_dump(mode="json") for a in assessments],
		"debts": [d.model_dump(mode="json") for d in debts],
		"audits": [a.model_dump(mode="json") for a in audits],
		"payments": [p.model_dump(mode="json") for p in payments],
		"total_debt": str(total_debt),
		"total_paid": str(total_paid),
		"tenant_id": tenant,
	}
	try:
		return render_template("dashboards/taxpayer_detail.html", **ctx)
	except Exception:
		return ctx


@views_bp.get("/taxpayers/register")
def register_taxpayer_form():
	ctx = {"title": "Register Taxpayer", "tenant_id": _tenant()}
	try:
		return render_template("forms/register_taxpayer.html", **ctx)
	except Exception:
		return ctx


@views_bp.post("/taxpayers/register")
def register_taxpayer_submit():
	tenant = _tenant()
	form = request.form
	try:
		result = _svc.register_taxpayer(
			taxpayer_id="",
			tenant_id=tenant,
			tax_type=form.get("tax_type", "income_tax"),
			tax_pin=form.get("tax_pin", ""),
			id_number=form.get("national_id", ""),
			legal_name=form["taxpayer_name"],
			entity_type=form.get("taxpayer_type", "individual"),
			email=form.get("email"),
			phone=form.get("phone"),
			address=form.get("physical_address", ""),
			evidence_reference=form.get("evidence_reference", "ui_registration"),
		)
		flash(f"Taxpayer registered: {result['tax_pin']}", "success")
		return redirect(url_for("tax_views.taxpayer_detail", tin=result["tax_pin"]))
	except (AssertionError, ValueError, PermissionError) as exc:
		flash(str(exc), "error")
		return redirect(url_for("tax_views.register_taxpayer_form"))


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------

@views_bp.get("/returns")
def list_returns():
	tenant = _tenant()
	status_filter = request.args.get("status", "")
	page = max(1, int(request.args.get("page", 1)))
	per_page = 25

	results = [r for r in _svc._returns.tenant_values(tenant) if not r.is_deleted]
	if status_filter:
		results = [r for r in results if r.status.value == status_filter]
	results.sort(key=lambda r: r.created_at, reverse=True)

	total = len(results)
	paged = results[(page - 1) * per_page: page * per_page]

	ctx = {
		"title": "Tax Returns",
		"returns": [r.model_dump(mode="json") for r in paged],
		"total": total,
		"page": page,
		"pages": max(1, (total + per_page - 1) // per_page),
		"status_filter": status_filter,
		"statuses": [s.value for s in ReturnStatus],
		"tenant_id": tenant,
	}
	try:
		return render_template("dashboards/return_list.html", **ctx)
	except Exception:
		return ctx


@views_bp.get("/returns/file")
def file_return_form():
	ctx = {"title": "File Tax Return", "tenant_id": _tenant()}
	try:
		return render_template("forms/file_return.html", **ctx)
	except Exception:
		return ctx


@views_bp.post("/returns/file")
def file_return_submit():
	tenant = _tenant()
	form = request.form
	try:
		result = _svc.submit_return(
			tin=form["tax_pin"],
			tax_type=form.get("tax_type", "income_tax"),
			period=form["period"],
			return_data={
				"gross_income": float(form.get("gross_income", 0)),
				"allowable_deductions": float(form.get("allowable_deductions", 0)),
				"tax_liability": float(form.get("tax_liability", 0)),
				"tax_credits": float(form.get("tax_credits", 0)),
				"tax_paid": float(form.get("tax_paid", 0)),
				"evidence_reference": form.get("evidence_reference", "ui_filed"),
			},
			tenant_id=tenant,
		)
		flash(f"Return filed: {result['id']}", "success")
		return redirect(url_for("tax_views.list_returns"))
	except (AssertionError, ValueError, PermissionError) as exc:
		flash(str(exc), "error")
		return redirect(url_for("tax_views.file_return_form"))


# ---------------------------------------------------------------------------
# Assessments
# ---------------------------------------------------------------------------

@views_bp.get("/assessments")
def list_assessments():
	tenant = _tenant()
	status_filter = request.args.get("status", "")
	page = max(1, int(request.args.get("page", 1)))
	per_page = 25

	results = [a for a in _svc._assessments.tenant_values(tenant) if not a.is_deleted]
	if status_filter:
		results = [a for a in results if a.status.value == status_filter]
	results.sort(key=lambda a: a.assessment_date, reverse=True)

	total = len(results)
	paged = results[(page - 1) * per_page: page * per_page]

	ctx = {
		"title": "Tax Assessments",
		"assessments": [a.model_dump(mode="json") for a in paged],
		"total": total,
		"page": page,
		"pages": max(1, (total + per_page - 1) // per_page),
		"status_filter": status_filter,
		"statuses": [s.value for s in AssessmentStatus],
		"tenant_id": tenant,
	}
	try:
		return render_template("dashboards/assessment_list.html", **ctx)
	except Exception:
		return ctx


# ---------------------------------------------------------------------------
# Debts
# ---------------------------------------------------------------------------

@views_bp.get("/debts")
def list_debts():
	tenant = _tenant()
	status_filter = request.args.get("status", "outstanding")
	page = max(1, int(request.args.get("page", 1)))
	per_page = 25

	results = [d for d in _svc._debts.tenant_values(tenant) if not d.is_deleted]
	if status_filter:
		results = [d for d in results if d.status.value == status_filter]
	results.sort(key=lambda d: d.due_date)

	total = len(results)
	paged = results[(page - 1) * per_page: page * per_page]

	# Aging summary
	today = date.today()
	aging: dict[str, Decimal] = {"0-30": Decimal("0"), "31-90": Decimal("0"), "91-180": Decimal("0"), "180+": Decimal("0")}
	for d in results:
		age = (today - d.due_date).days
		if age <= 30:
			aging["0-30"] += d.balance
		elif age <= 90:
			aging["31-90"] += d.balance
		elif age <= 180:
			aging["91-180"] += d.balance
		else:
			aging["180+"] += d.balance

	ctx = {
		"title": "Tax Debts",
		"debts": [d.model_dump(mode="json") for d in paged],
		"total": total,
		"page": page,
		"pages": max(1, (total + per_page - 1) // per_page),
		"status_filter": status_filter,
		"statuses": [s.value for s in DebtStatus],
		"aging": {k: str(v) for k, v in aging.items()},
		"total_outstanding": str(sum(aging.values())),
		"tenant_id": tenant,
	}
	try:
		return render_template("dashboards/debt_list.html", **ctx)
	except Exception:
		return ctx


# ---------------------------------------------------------------------------
# Audits
# ---------------------------------------------------------------------------

@views_bp.get("/audits")
def list_audits():
	tenant = _tenant()
	status_filter = request.args.get("status", "")
	page = max(1, int(request.args.get("page", 1)))
	per_page = 25

	results = [a for a in _svc._audits.tenant_values(tenant) if not a.is_deleted]
	if status_filter:
		results = [a for a in results if a.status.value == status_filter]
	results.sort(key=lambda a: a.created_at, reverse=True)

	total = len(results)
	paged = results[(page - 1) * per_page: page * per_page]

	ctx = {
		"title": "Audit Cases",
		"audits": [a.model_dump(mode="json") for a in paged],
		"total": total,
		"page": page,
		"pages": max(1, (total + per_page - 1) // per_page),
		"status_filter": status_filter,
		"statuses": [s.value for s in AuditStatus],
		"tenant_id": tenant,
	}
	try:
		return render_template("dashboards/audit_list.html", **ctx)
	except Exception:
		return ctx


@views_bp.get("/audits/<audit_id>")
def audit_detail(audit_id: str):
	tenant = _tenant()
	audit = _svc._audits.get_item(tenant, audit_id)
	if audit is None:
		flash(f"Audit {audit_id} not found", "error")
		return redirect(url_for("tax_views.list_audits"))

	findings = [f for f in _svc._findings.tenant_values(tenant) if f.audit_id == audit_id]
	ctx = {
		"title": f"Audit: {audit_id[:8]}",
		"audit": audit.model_dump(mode="json"),
		"findings": [f.model_dump(mode="json") for f in findings],
		"tenant_id": tenant,
	}
	try:
		return render_template("dashboards/audit_detail.html", **ctx)
	except Exception:
		return ctx


# ---------------------------------------------------------------------------
# Objections & Appeals
# ---------------------------------------------------------------------------

@views_bp.get("/objections")
def list_objections():
	tenant = _tenant()
	results = [o for o in _svc._objections.tenant_values(tenant) if not o.is_deleted]
	results.sort(key=lambda o: o.filed_date, reverse=True)
	ctx = {
		"title": "Objections",
		"objections": [o.model_dump(mode="json") for o in results],
		"total": len(results),
		"tenant_id": tenant,
	}
	try:
		return render_template("dashboards/objection_list.html", **ctx)
	except Exception:
		return ctx


@views_bp.get("/appeals")
def list_appeals():
	tenant = _tenant()
	results = [a for a in _svc._appeals.tenant_values(tenant) if not a.is_deleted]
	results.sort(key=lambda a: a.created_at, reverse=True)
	ctx = {
		"title": "Appeals",
		"appeals": [a.model_dump(mode="json") for a in results],
		"total": len(results),
		"tenant_id": tenant,
	}
	try:
		return render_template("dashboards/appeal_list.html", **ctx)
	except Exception:
		return ctx


# ---------------------------------------------------------------------------
# Refunds
# ---------------------------------------------------------------------------

@views_bp.get("/refunds")
def list_refunds():
	tenant = _tenant()
	status_filter = request.args.get("status", "")
	results = [r for r in _svc._refunds.tenant_values(tenant) if not r.is_deleted]
	if status_filter:
		results = [r for r in results if r.status.value == status_filter]
	results.sort(key=lambda r: r.created_at, reverse=True)
	ctx = {
		"title": "Refund Applications",
		"refunds": [r.model_dump(mode="json") for r in results],
		"total": len(results),
		"status_filter": status_filter,
		"statuses": [s.value for s in RefundStatus],
		"tenant_id": tenant,
	}
	try:
		return render_template("dashboards/refund_list.html", **ctx)
	except Exception:
		return ctx


# ---------------------------------------------------------------------------
# Clearance Certificates
# ---------------------------------------------------------------------------

@views_bp.get("/clearances")
def list_clearances():
	tenant = _tenant()
	results = [c for c in _svc._clearances.tenant_values(tenant) if not c.is_deleted]
	results.sort(key=lambda c: c.created_at, reverse=True)
	ctx = {
		"title": "Tax Clearance Certificates",
		"clearances": [c.model_dump(mode="json") for c in results],
		"total": len(results),
		"statuses": [s.value for s in ClearanceCertificateStatus],
		"tenant_id": tenant,
	}
	try:
		return render_template("dashboards/clearance_list.html", **ctx)
	except Exception:
		return ctx


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

@views_bp.get("/reports")
def reports_index():
	tenant = _tenant()
	ctx = {
		"title": "Tax Reports",
		"tenant_id": tenant,
		"current_year": date.today().year,
	}
	try:
		return render_template("dashboards/reports_index.html", **ctx)
	except Exception:
		return ctx


@views_bp.get("/reports/revenue")
def revenue_report():
	tenant = _tenant()
	period = request.args.get("period", str(date.today().year))
	data = _svc.revenue_collection_report(period, tenant_id=tenant)
	ctx = {"title": f"Revenue Report — {period}", "report": data, "period": period, "tenant_id": tenant}
	try:
		return render_template("dashboards/revenue_report.html", **ctx)
	except Exception:
		return ctx


@views_bp.get("/reports/compliance")
def compliance_report():
	tenant = _tenant()
	period = request.args.get("period", str(date.today().year))
	data = _svc.compliance_rate_report(period, tenant_id=tenant)
	ctx = {"title": f"Compliance Report — {period}", "report": data, "period": period, "tenant_id": tenant}
	try:
		return render_template("dashboards/compliance_report.html", **ctx)
	except Exception:
		return ctx


@views_bp.get("/reports/delinquency")
def delinquency_report():
	tenant = _tenant()
	as_of = request.args.get("as_of", date.today().isoformat())
	data = _svc.delinquency_report(as_of, tenant_id=tenant)
	ctx = {"title": "Debt Aging Report", "report": data, "as_of": as_of, "tenant_id": tenant}
	try:
		return render_template("dashboards/delinquency_report.html", **ctx)
	except Exception:
		return ctx
