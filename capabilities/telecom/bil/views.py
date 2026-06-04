"""Flask Blueprint UI views for APG Telecom Billing.

Provides HTML endpoints for the billing management UI.
All views are tenant-scoped and return Jinja2-rendered templates
with rich context dicts for the dashboard, lists, and detail pages.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""
from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any

from flask import Blueprint, redirect, render_template, request, url_for

from .capability_contract import get_capability_contract
from .service import TelecomBillingService

bil_views = Blueprint(
	"telecom_bil_views",
	__name__,
	url_prefix="/telecom/bil",
	template_folder="templates",
	static_folder="static",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tenant() -> str:
	return request.args.get("tenant_id", "default").strip() or "default"


def _svc(tenant_id: str | None = None) -> TelecomBillingService:
	return TelecomBillingService(tenant_id=tenant_id or _tenant())


def _run(coro: Any) -> Any:
	try:
		loop = asyncio.get_event_loop()
		return loop.run_until_complete(coro)
	except RuntimeError:
		return asyncio.run(coro)


def _items(store: dict[Any, Any], tenant_id: str) -> list[dict[str, Any]]:
	result = []
	for k, v in store.items():
		if isinstance(k, tuple) and k[0] == tenant_id:
			result.append(v.to_dict() if hasattr(v, "to_dict") else v)
	return sorted(result, key=lambda x: x.get("id", ""))


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def dashboard_model(svc: TelecomBillingService, tenant_id: str = "default") -> dict[str, Any]:
	"""Build context dict for the billing dashboard."""
	contract = get_capability_contract(tenant_id)
	summary = svc.dashboard_summary()
	return {
		"title": "Telecom Billing Dashboard",
		"tenant_id": tenant_id,
		"summary": summary,
		"theme": contract.get("theme", {}),
		"routes": contract.get("ui", {}).get("routes", []),
		"kpis": [
			{"label": "Total CDRs", "value": summary.get("cdr_count", 0), "icon": "phone"},
			{"label": "Charges", "value": summary.get("charge_count", 0), "icon": "zap"},
			{"label": "Invoices", "value": summary.get("invoice_count", 0), "icon": "file-text"},
			{"label": "Payments", "value": summary.get("payment_count", 0), "icon": "credit-card"},
			{"label": "Open Disputes", "value": summary.get("open_disputes", 0), "icon": "alert-triangle"},
			{"label": "Suspended", "value": summary.get("suspended_accounts", 0), "icon": "pause-circle"},
		],
	}


@bil_views.get("/")
@bil_views.get("/dashboard")
def dashboard():
	tenant_id = _tenant()
	svc = _svc(tenant_id)
	ctx = dashboard_model(svc, tenant_id)
	try:
		return render_template("dashboards/billing_dashboard.html", **ctx)
	except Exception:
		return _render_json(ctx)


# ---------------------------------------------------------------------------
# CDR views
# ---------------------------------------------------------------------------

def list_cdr_model(svc: TelecomBillingService, tenant_id: str) -> dict[str, Any]:
	items = _items(svc.cdrs, tenant_id)
	status_filter = request.args.get("status")
	if status_filter:
		items = [i for i in items if i.get("mediation_status") == status_filter]
	return {
		"title": "Call Detail Records",
		"tenant_id": tenant_id,
		"items": items,
		"total": len(items),
		"status_filter": status_filter,
	}


@bil_views.get("/cdrs")
def list_cdr():
	tenant_id = _tenant()
	svc = _svc(tenant_id)
	ctx = list_cdr_model(svc, tenant_id)
	try:
		return render_template("billing/cdr_list.html", **ctx)
	except Exception:
		return _render_json(ctx)


@bil_views.get("/cdrs/<cdr_id>")
def detail_cdr(cdr_id: str):
	tenant_id = _tenant()
	svc = _svc(tenant_id)
	item = svc.cdrs.get((tenant_id, cdr_id))
	ctx = {
		"title": f"CDR {cdr_id}",
		"tenant_id": tenant_id,
		"cdr": item.to_dict() if item else None,
		"not_found": item is None,
	}
	try:
		return render_template("billing/cdr_detail.html", **ctx)
	except Exception:
		return _render_json(ctx)


# ---------------------------------------------------------------------------
# Invoice views
# ---------------------------------------------------------------------------

def list_invoice_model(svc: TelecomBillingService, tenant_id: str) -> dict[str, Any]:
	items = _items(svc.invoices, tenant_id)
	status_filter = request.args.get("status")
	if status_filter:
		items = [i for i in items if i.get("status") == status_filter]
	overdue = [i for i in items if i.get("status") == "overdue"]
	draft = [i for i in items if i.get("status") == "draft"]
	return {
		"title": "Invoices",
		"tenant_id": tenant_id,
		"items": items,
		"total": len(items),
		"overdue_count": len(overdue),
		"draft_count": len(draft),
		"status_filter": status_filter,
		"status_options": [
			"draft", "pending_approval", "approved", "sent", "paid",
			"partially_paid", "overdue", "disputed", "cancelled", "written_off",
		],
	}


@bil_views.get("/invoices")
def list_invoice():
	tenant_id = _tenant()
	svc = _svc(tenant_id)
	ctx = list_invoice_model(svc, tenant_id)
	try:
		return render_template("billing/invoice_list.html", **ctx)
	except Exception:
		return _render_json(ctx)


@bil_views.get("/invoices/<invoice_id>")
def detail_invoice(invoice_id: str):
	tenant_id = _tenant()
	svc = _svc(tenant_id)
	result = _run(svc.view_bill(invoice_id))
	adjustments = svc._adjustments.get(invoice_id, [])
	ctx = {
		"title": f"Invoice {invoice_id}",
		"tenant_id": tenant_id,
		"invoice": result,
		"adjustments": adjustments,
		"not_found": not result.get("found"),
	}
	try:
		return render_template("billing/invoice_detail.html", **ctx)
	except Exception:
		return _render_json(ctx)


@bil_views.get("/invoices/create")
def create_invoice():
	ctx = {
		"title": "Create Invoice",
		"tenant_id": _tenant(),
	}
	try:
		return render_template("forms/invoice_create.html", **ctx)
	except Exception:
		return _render_json(ctx)


@bil_views.get("/invoices/<invoice_id>/edit")
def edit_invoice(invoice_id: str):
	tenant_id = _tenant()
	svc = _svc(tenant_id)
	inv = svc.invoices.get((tenant_id, invoice_id))
	ctx = {
		"title": f"Edit Invoice {invoice_id}",
		"tenant_id": tenant_id,
		"invoice": inv.to_dict() if inv else None,
	}
	try:
		return render_template("forms/invoice_edit.html", **ctx)
	except Exception:
		return _render_json(ctx)


# ---------------------------------------------------------------------------
# Payment views
# ---------------------------------------------------------------------------

def list_payment_model(svc: TelecomBillingService, tenant_id: str) -> dict[str, Any]:
	payments = _items(svc.payments, tenant_id)
	discounts = _items(svc.discounts, tenant_id)
	total_received = sum(
		Decimal(str(p.get("amount", 0))) for p in payments
	)
	return {
		"title": "Payment Ledger",
		"tenant_id": tenant_id,
		"payments": payments,
		"discounts": discounts,
		"total_received": str(total_received),
		"payment_count": len(payments),
		"discount_count": len(discounts),
	}


@bil_views.get("/payments")
def list_payment():
	tenant_id = _tenant()
	svc = _svc(tenant_id)
	ctx = list_payment_model(svc, tenant_id)
	try:
		return render_template("billing/payment_list.html", **ctx)
	except Exception:
		return _render_json(ctx)


@bil_views.get("/payments/create")
def create_payment():
	ctx = {"title": "Record Payment", "tenant_id": _tenant()}
	try:
		return render_template("forms/payment_create.html", **ctx)
	except Exception:
		return _render_json(ctx)


# ---------------------------------------------------------------------------
# Dispute views
# ---------------------------------------------------------------------------

def list_dispute_model(svc: TelecomBillingService, tenant_id: str) -> dict[str, Any]:
	disputes = [
		d for d in svc._disputes.values()
		if d.get("tenant_id") == tenant_id
	]
	status_filter = request.args.get("status")
	if status_filter:
		disputes = [d for d in disputes if d.get("status") == status_filter]
	open_count = sum(1 for d in disputes if d.get("status") == "open")
	return {
		"title": "Billing Disputes",
		"tenant_id": tenant_id,
		"disputes": disputes,
		"total": len(disputes),
		"open_count": open_count,
		"status_filter": status_filter,
	}


@bil_views.get("/disputes")
def list_dispute():
	tenant_id = _tenant()
	svc = _svc(tenant_id)
	ctx = list_dispute_model(svc, tenant_id)
	try:
		return render_template("billing/dispute_list.html", **ctx)
	except Exception:
		return _render_json(ctx)


@bil_views.get("/disputes/<dispute_id>")
def detail_dispute(dispute_id: str):
	tenant_id = _tenant()
	svc = _svc(tenant_id)
	d = svc._disputes.get(dispute_id)
	ctx = {
		"title": f"Dispute {dispute_id}",
		"tenant_id": tenant_id,
		"dispute": d,
		"not_found": d is None or d.get("tenant_id") != tenant_id,
	}
	try:
		return render_template("billing/dispute_detail.html", **ctx)
	except Exception:
		return _render_json(ctx)


@bil_views.get("/disputes/create")
def create_dispute():
	ctx = {"title": "Raise Dispute", "tenant_id": _tenant()}
	try:
		return render_template("forms/dispute_create.html", **ctx)
	except Exception:
		return _render_json(ctx)


# ---------------------------------------------------------------------------
# Mediation console
# ---------------------------------------------------------------------------

def mediation_console_model(svc: TelecomBillingService, tenant_id: str) -> dict[str, Any]:
	return {
		"title": "Mediation Console",
		"tenant_id": tenant_id,
		"cdrs": _items(svc.cdrs, tenant_id),
		"charges": _items(svc.charges, tenant_id),
		"cdr_count": sum(1 for k in svc.cdrs if k[0] == tenant_id),
		"charge_count": sum(1 for k in svc.charges if k[0] == tenant_id),
	}


@bil_views.get("/mediation")
def mediation_console():
	tenant_id = _tenant()
	svc = _svc(tenant_id)
	ctx = mediation_console_model(svc, tenant_id)
	try:
		return render_template("billing/mediation_console.html", **ctx)
	except Exception:
		return _render_json(ctx)


# ---------------------------------------------------------------------------
# Convergent billing console
# ---------------------------------------------------------------------------

def convergent_console_model(svc: TelecomBillingService, tenant_id: str) -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Convergent Billing",
		"tenant_id": tenant_id,
		"convergent_accounts": _items(svc.convergent_accounts, tenant_id),
		"supported_modes": contract.get("configuration", {}).get("convergent", {}).get("supported_modes", []),
	}


@bil_views.get("/convergent")
def convergent_console():
	tenant_id = _tenant()
	svc = _svc(tenant_id)
	ctx = convergent_console_model(svc, tenant_id)
	try:
		return render_template("billing/convergent_console.html", **ctx)
	except Exception:
		return _render_json(ctx)


# ---------------------------------------------------------------------------
# Agent workbench
# ---------------------------------------------------------------------------

def agent_workbench_model(svc: TelecomBillingService, tenant_id: str) -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Billing Agent Workbench",
		"tenant_id": tenant_id,
		"agents": _items(svc.agents, tenant_id),
		"supported_runtimes": contract.get("configuration", {}).get("agents", {}).get("supported_runtimes", []),
		"supported_roles": contract.get("configuration", {}).get("agents", {}).get("supported_roles", []),
	}


@bil_views.get("/agents")
def agent_workbench():
	tenant_id = _tenant()
	svc = _svc(tenant_id)
	ctx = agent_workbench_model(svc, tenant_id)
	try:
		return render_template("billing/agent_workbench.html", **ctx)
	except Exception:
		return _render_json(ctx)


# ---------------------------------------------------------------------------
# Report views
# ---------------------------------------------------------------------------

@bil_views.get("/reports")
def reports_index():
	ctx = {
		"title": "Billing Reports",
		"tenant_id": _tenant(),
		"reports": [
			{"id": "revenue", "name": "Revenue Report", "description": "Total revenue by period and segment"},
			{"id": "arpu", "name": "ARPU Analysis", "description": "Average revenue per user"},
			{"id": "disputes", "name": "Dispute Analytics", "description": "Dispute volume, rates, outcomes"},
			{"id": "leakage", "name": "Revenue Leakage", "description": "Unrated CDRs and leakage estimation"},
			{"id": "churn", "name": "Churn Revenue Impact", "description": "Revenue impact from churned subscribers"},
			{"id": "interconnect", "name": "Interconnect Reconciliation", "description": "Carrier settlement reconciliation"},
		],
	}
	try:
		return render_template("billing/reports_index.html", **ctx)
	except Exception:
		return _render_json(ctx)


@bil_views.get("/reports/<report_type>")
def view_report(report_type: str):
	tenant_id = _tenant()
	svc = _svc(tenant_id)
	period = {
		"start": request.args.get("start", ""),
		"end": request.args.get("end", ""),
	}
	data: dict[str, Any] = {}

	if report_type == "revenue":
		data = _run(svc.revenue_report(period))
	elif report_type == "arpu":
		data = _run(svc.arpu_analysis(period))
	elif report_type == "disputes":
		data = _run(svc.dispute_analytics(period))
	elif report_type == "leakage":
		data = _run(svc.revenue_leakage_detection(period))
	elif report_type == "churn":
		data = _run(svc.churn_revenue_impact(period))
	elif report_type == "interconnect":
		carrier = request.args.get("carrier", "")
		data = _run(svc.interconnect_reconciliation(carrier, period))
	else:
		data = {"error": f"Unknown report type: {report_type}"}

	ctx = {
		"title": f"Report: {report_type.replace('_', ' ').title()}",
		"tenant_id": tenant_id,
		"report_type": report_type,
		"period": period,
		"data": data,
	}
	try:
		return render_template("billing/report_view.html", **ctx)
	except Exception:
		return _render_json(ctx)


# ---------------------------------------------------------------------------
# Fallback: render as JSON when templates are missing
# ---------------------------------------------------------------------------

def _render_json(ctx: dict[str, Any]):
	"""Return JSON when templates are not yet rendered."""
	from flask import jsonify
	# Strip non-serialisable objects
	safe = {k: v for k, v in ctx.items() if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
	return jsonify(safe)
