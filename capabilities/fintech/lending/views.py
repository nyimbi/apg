"""
Flask Blueprint UI views for APG Digital Lending.

Provides Jinja2-rendered HTML views for all lending entities.
Registered at /lending/* for browser-based access.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from __future__ import annotations

from datetime import date
from typing import Any

from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify

try:
	from .service import LendingService
	from .capability_contract import get_capability_contract
	from .models import LoanProductCreate, LoanApplicationCreate
except ImportError:  # pragma: no cover
	from service import LendingService  # type: ignore
	from capability_contract import get_capability_contract  # type: ignore
	from models import LoanProductCreate, LoanApplicationCreate  # type: ignore


lending_ui_bp = Blueprint(
	"lending_ui",
	__name__,
	url_prefix="/lending",
	template_folder="templates",
	static_folder="static",
	static_url_path="/lending/static",
)

_SERVICE = LendingService()


def _svc() -> LendingService:
	return _SERVICE


def _tenant() -> str:
	return request.headers.get("X-Tenant-Id", "default")


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@lending_ui_bp.get("/")
@lending_ui_bp.get("/dashboard")
def dashboard():
	"""Main lending dashboard with KPI cards."""
	tenant_id = _tenant()
	summary = _svc().dashboard_summary(tenant_id)
	portfolio = _svc().portfolio_summary()
	delinquency = _svc().delinquency_report()
	ifrs9 = _svc().provision_calculation("ifrs9")

	kpis: list[dict[str, Any]] = [
		{"label": "Active Loans",       "value": len(_svc().list_loans(status="active")),  "icon": "fa-hand-holding-usd", "color": "primary"},
		{"label": "Total Book",          "value": f"{portfolio['total_book']:,.0f}",         "icon": "fa-coins",            "color": "success"},
		{"label": "PAR 30",             "value": f"{portfolio['par_30']:.1%}",              "icon": "fa-exclamation-circle","color": "warning"},
		{"label": "NPL Ratio",          "value": f"{portfolio['npl_ratio']:.1%}",           "icon": "fa-times-circle",     "color": "danger"},
		{"label": "Applications",       "value": summary["application_count"],              "icon": "fa-file-alt",         "color": "info"},
		{"label": "ECL Provision",      "value": f"{ifrs9['total_ecl']:,.0f}",              "icon": "fa-shield-alt",       "color": "secondary"},
	]

	return render_template(
		"dashboards/lending_dashboard.html",
		kpis=kpis,
		portfolio=portfolio,
		delinquency=delinquency,
		ifrs9=ifrs9,
		summary=summary,
		tenant_id=tenant_id,
		page_title="Lending Dashboard",
	)


# ---------------------------------------------------------------------------
# Loan Products
# ---------------------------------------------------------------------------

@lending_ui_bp.get("/products")
def list_products():
	"""Product catalogue list view."""
	active_only = request.args.get("active_only", "true").lower() != "false"
	products = _svc().list_products(active_only=active_only)
	return render_template(
		"lending/product_list.html",
		products=products,
		active_only=active_only,
		page_title="Loan Products",
	)


@lending_ui_bp.get("/products/new")
def create_product_form():
	"""New product form."""
	return render_template("forms/product_form.html", product=None, page_title="New Loan Product")


@lending_ui_bp.post("/products/new")
def create_product_submit():
	"""Handle new product form submission."""
	try:
		payload = {**request.form}
		for float_field in ("min_amount", "max_amount", "base_annual_rate",
		                    "processing_fee_pct", "insurance_fee_pct", "late_penalty_pct"):
			if float_field in payload:
				payload[float_field] = float(payload[float_field])
		for int_field in ("min_tenor_months", "max_tenor_months"):
			if int_field in payload:
				payload[int_field] = int(payload[int_field])
		p = LoanProductCreate(**payload)
		_svc().create_loan_product(
			product_code=p.code, name=p.name, product_type=p.product_type,
			rate_type="reducing_balance", min_amount=p.min_amount, max_amount=p.max_amount,
			min_tenor=p.min_tenor_months, max_tenor=p.max_tenor_months, fees=[],
			tenant_id=p.tenant_id, owner_id=p.created_by, currency=p.currency,
		)
		flash("Loan product created successfully.", "success")
		return redirect(url_for("lending_ui.list_products"))
	except Exception as exc:
		flash(f"Error: {exc}", "danger")
		return render_template("forms/product_form.html", product=request.form, page_title="New Loan Product")


@lending_ui_bp.get("/products/<product_id>")
def detail_product(product_id: str):
	"""Product detail with performance metrics."""
	product = _svc()._require_product(product_id)
	period = request.args.get("period", date.today().strftime("%Y-%m"))
	try:
		perf = _svc().product_performance_report(product_id, period)
	except Exception:
		perf = {}
	return render_template(
		"lending/product_detail.html",
		product=product.to_dict(),
		performance=perf,
		page_title=f"Product: {product.name}",
	)


# ---------------------------------------------------------------------------
# Loan Applications
# ---------------------------------------------------------------------------

@lending_ui_bp.get("/applications")
def list_applications():
	"""Application queue with status filters."""
	filters: dict[str, Any] = {}
	for key in ("status", "borrower_id", "product_id"):
		val = request.args.get(key)
		if val:
			filters[key] = val
	apps = _svc().list_applications(filters=filters, tenant_id=_tenant())
	status_counts: dict[str, int] = {}
	for app in _svc().list_applications(tenant_id=_tenant()):
		s = app.get("status", "unknown")
		status_counts[s] = status_counts.get(s, 0) + 1

	return render_template(
		"lending/application_list.html",
		applications=apps,
		status_counts=status_counts,
		filters=filters,
		page_title="Loan Applications",
	)


@lending_ui_bp.get("/applications/new")
def create_application_form():
	"""New application form."""
	products = _svc().list_products(active_only=True)
	return render_template(
		"forms/application_form.html",
		application=None,
		products=products,
		page_title="New Loan Application",
	)


@lending_ui_bp.post("/applications/new")
def create_application_submit():
	"""Handle new application form submission."""
	try:
		payload = {**request.form}
		payload["requested_amount"] = float(payload.get("requested_amount", 0))
		payload["requested_tenor_months"] = int(payload.get("requested_tenor_months", 12))
		payload["monthly_income"] = float(payload.get("monthly_income", 0))
		p = LoanApplicationCreate(**payload)
		from uuid6 import uuid7
		app_id = str(uuid7())
		_svc().submit_application(
			application_id=app_id,
			tenant_id=p.tenant_id, borrower_id=p.borrower_id,
			product_id=p.product_id, requested_amount=p.requested_amount,
			purpose=p.purpose,
			affordability_reference=p.bank_statement_ref,
			bank_statement_reference=p.bank_statement_ref,
			aml_reference=p.aml_ref, fraud_reference=p.fraud_ref,
			behavior_evidence_reference="", human_review="",
		)
		flash("Application submitted successfully.", "success")
		return redirect(url_for("lending_ui.list_applications"))
	except Exception as exc:
		flash(f"Error: {exc}", "danger")
		products = _svc().list_products(active_only=True)
		return render_template("forms/application_form.html", application=request.form, products=products, page_title="New Loan Application")


@lending_ui_bp.get("/applications/<application_id>")
def detail_application(application_id: str):
	"""Application detail with linked entities."""
	app_data = _svc().retrieve_application(application_id)
	offers = _svc().generate_loan_offers(application_id) if app_data.get("status") in ("approved", "conditionally_approved") else []
	return render_template(
		"lending/application_detail.html",
		application=app_data,
		offers=offers,
		page_title=f"Application {application_id[:8]}...",
	)


@lending_ui_bp.post("/applications/<application_id>/underwrite")
def underwrite_application(application_id: str):
	"""Process underwriting decision from form."""
	try:
		result = _svc().underwriting_decision(
			application_id=application_id,
			decision=request.form.get("decision", "refer"),
			conditions=request.form.getlist("conditions"),
			underwriter_id=request.form.get("underwriter_id", "system"),
		)
		flash(f"Decision recorded: {result['decision']}", "success")
	except Exception as exc:
		flash(f"Error: {exc}", "danger")
	return redirect(url_for("lending_ui.detail_application", application_id=application_id))


# ---------------------------------------------------------------------------
# Loans
# ---------------------------------------------------------------------------

@lending_ui_bp.get("/loans")
def list_loans():
	"""Active loan book list view."""
	status = request.args.get("status")
	borrower_id = request.args.get("borrower_id")
	loans = _svc().list_loans(status=status, borrower_id=borrower_id)
	return render_template(
		"lending/loan_list.html",
		loans=loans,
		status_filter=status,
		page_title="Loan Book",
	)


@lending_ui_bp.get("/loans/<loan_id>")
def detail_loan(loan_id: str):
	"""Loan detail with repayment schedule and transaction history."""
	loan = _svc()._require_loan(loan_id)
	statement = _svc().get_loan_statement(loan_id)
	schedule = _svc().generate_repayment_schedule(loan_id)
	dpd_data = _svc().calculate_dpd(loan_id)
	early_settlement = _svc().early_settlement(loan_id, date.today().isoformat()) if loan.status == "active" else None
	return render_template(
		"lending/loan_detail.html",
		loan=loan.to_dict(),
		statement=statement,
		schedule=schedule,
		dpd=dpd_data,
		early_settlement=early_settlement,
		page_title=f"Loan {loan_id[:8]}...",
	)


@lending_ui_bp.post("/loans/<loan_id>/repay")
def post_repayment(loan_id: str):
	"""Post a repayment from UI form."""
	try:
		result = _svc().process_repayment(
			loan_id=loan_id,
			amount=float(request.form.get("amount", 0)),
			payment_date=request.form.get("payment_date", date.today().isoformat()),
			payment_method=request.form.get("payment_method", "mobile_money"),
			reference=request.form.get("reference", ""),
		)
		flash(f"Repayment of {result['payment_amount']:,.2f} applied. Outstanding: {result['outstanding_principal']:,.2f}", "success")
	except Exception as exc:
		flash(f"Error: {exc}", "danger")
	return redirect(url_for("lending_ui.detail_loan", loan_id=loan_id))


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------

@lending_ui_bp.get("/collections")
def list_collections():
	"""Delinquent loans collection queue."""
	report = _svc().delinquency_report()
	# Build list of delinquent loans
	delinquent_loans = []
	for loan in _svc().loans.values():
		dpd_data = _svc().calculate_dpd(loan.loan_id)
		if dpd_data["max_dpd"] > 0:
			loan_dict = loan.to_dict()
			loan_dict["max_dpd"] = dpd_data["max_dpd"]
			loan_dict["bucket"] = dpd_data["delinquency_bucket"]
			delinquent_loans.append(loan_dict)
	delinquent_loans.sort(key=lambda x: x["max_dpd"], reverse=True)
	return render_template(
		"lending/collection_list.html",
		loans=delinquent_loans,
		report=report,
		page_title="Collections Queue",
	)


# ---------------------------------------------------------------------------
# Portfolio & Reports
# ---------------------------------------------------------------------------

@lending_ui_bp.get("/portfolio")
def portfolio_view():
	"""Portfolio analytics dashboard."""
	portfolio = _svc().portfolio_summary()
	vintage = _svc().vintage_analysis(12)
	concentration = _svc().concentration_risk_report()
	ifrs9 = _svc().provision_calculation("ifrs9")
	stress = _svc().stress_test([
		{"name": "mild",     "additional_default_rate": 0.05, "lgd": 0.40},
		{"name": "moderate", "additional_default_rate": 0.15, "lgd": 0.40},
		{"name": "severe",   "additional_default_rate": 0.30, "lgd": 0.45},
	])
	return render_template(
		"dashboards/portfolio_dashboard.html",
		portfolio=portfolio,
		vintage=vintage,
		concentration=concentration,
		ifrs9=ifrs9,
		stress=stress,
		page_title="Portfolio Analytics",
	)


@lending_ui_bp.get("/reports/amortisation")
def amortisation_calculator():
	"""Interactive amortisation schedule calculator."""
	result = None
	if request.args.get("principal"):
		try:
			from .domain.calculations import build_amortisation_schedule
			result = build_amortisation_schedule(
				principal=float(request.args["principal"]),
				annual_rate=float(request.args.get("annual_rate", 0.18)),
				tenor_months=int(request.args.get("tenor_months", 12)),
				start_date=date.today(),
				schedule_type=request.args.get("schedule_type", "reducing_balance"),
			)
		except Exception as exc:
			flash(f"Error: {exc}", "danger")
	return render_template(
		"lending/amortisation_calculator.html",
		result=result,
		args=request.args,
		page_title="Amortisation Calculator",
	)
