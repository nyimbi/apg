"""Flask Blueprint UI views for APG Laboratory Information System.

Provides server-rendered view model builders and a Flask Blueprint with
HTML-serving routes for all LIS screens.  Each view function builds a
structured context dict that templates consume via Jinja2.

Routes
------
GET /healthcare-lab/                       → redirect to dashboard
GET /healthcare-lab/dashboard              → LabDashboard
GET /healthcare-lab/orders                 → LabOrderQueue (filterable)
GET /healthcare-lab/orders/new             → LabOrderEntry form
GET /healthcare-lab/orders/<id>            → LabOrderDetail
GET /healthcare-lab/specimens              → LabSpecimenTracker
GET /healthcare-lab/specimens/<id>         → LabSpecimenDetail (with custody chain)
GET /healthcare-lab/results                → LabResultWorkbench
GET /healthcare-lab/results/entry          → LabResultEntry form
GET /healthcare-lab/results/<id>           → LabResultDetail
GET /healthcare-lab/critical-values        → LabCriticalValues (unacked highlighted)
GET /healthcare-lab/qc                     → LabQCConsole
GET /healthcare-lab/qc/<id>               → LabQCRunDetail
GET /healthcare-lab/instruments            → LabInstrumentPanel
GET /healthcare-lab/instruments/<id>       → LabInstrumentDetail
GET /healthcare-lab/referrals              → LabReferralList
GET /healthcare-lab/referrals/<id>         → LabReferralDetail
GET /healthcare-lab/reports                → LabReportSelector
GET /healthcare-lab/reports/tat            → TATReport
GET /healthcare-lab/reports/workload       → WorkloadReport
GET /healthcare-lab/reports/qc-summary     → QCSummaryReport
GET /healthcare-lab/reports/critical-values → CriticalValueReport
GET /healthcare-lab/settings               → LabSettings

© 2025 Datacraft — nyimbi@gmail.com
"""

from __future__ import annotations

import asyncio
from typing import Any

from flask import Blueprint, jsonify, redirect, render_template, request, url_for

from .capability_contract import get_capability_contract
from .service import LaboratoryInformationService

ui_bp = Blueprint(
	"healthcare_lab_ui",
	__name__,
	url_prefix="/healthcare-lab",
	template_folder="templates",
	static_folder="static",
)

_svc = LaboratoryInformationService()


# ── Internal helpers ──────────────────────────────────────────────────────────

def _run(coro: Any) -> Any:
	"""Run an async coroutine from a synchronous view function."""
	return asyncio.run(coro)


def _tenant() -> str:
	return request.headers.get("X-Tenant-ID", request.args.get("tenant_id", "default"))


def _base_ctx(tenant_id: str, title: str) -> dict[str, Any]:
	"""Build the shared context injected into every template."""
	contract = get_capability_contract(tenant_id)
	return {
		"title": title,
		"tenant_id": tenant_id,
		"theme": contract["theme"],
		"routes": contract["ui"]["routes"],
		"capability": contract["capability"],
		"version": contract["version"],
	}


def _render_json(ctx: dict[str, Any]) -> Any:
	"""Return JSON when Accept header prefers JSON, else render template."""
	if "application/json" in request.headers.get("Accept", ""):
		return jsonify(ctx)
	template = ctx.pop("_template", None)
	if template:
		return render_template(template, **ctx)
	# Fallback: return JSON if no template configured
	return jsonify(ctx)


# ── View model builders (pure functions, no I/O) ──────────────────────────────

def dashboard_view_model(tenant_id: str) -> dict[str, Any]:
	"""Build view model for the LIS dashboard with KPI tiles."""
	summary = _run(_svc.dashboard_summary(tenant_id))
	ctx = _base_ctx(tenant_id, "Laboratory Dashboard")
	ctx.update({
		"summary": summary,
		"orders": summary.get("orders", {}),
		"results": summary.get("results", {}),
		"critical_values": summary.get("critical_values", {}),
		"instruments": summary.get("instruments", {}),
		"external_referrals": summary.get("external_referrals", {}),
		"_template": "healthcare/lab/dashboard.html",
	})
	return ctx


def order_queue_view_model(
	tenant_id: str,
	status: str | None = None,
	priority: str | None = None,
	patient_id: str | None = None,
) -> dict[str, Any]:
	"""Build view model for the lab order queue with STAT highlighting."""
	orders = _run(_svc.list_orders(tenant_id, status=status, patient_id=patient_id, priority=priority))
	stat_orders = [o for o in orders if o.collection_priority == "stat"]
	pending = [o for o in orders if o.status == "pending"]
	ctx = _base_ctx(tenant_id, "Lab Order Queue")
	ctx.update({
		"orders": [o.model_dump(mode="json") for o in orders],
		"stat_count": len(stat_orders),
		"pending_count": len(pending),
		"total_count": len(orders),
		"filter": {"status": status, "priority": priority, "patient_id": patient_id},
		"_template": "healthcare/lab/orders/list.html",
	})
	return ctx


def order_detail_view_model(tenant_id: str, order_id: str) -> dict[str, Any]:
	"""Build view model for a single order detail page."""
	order = _run(_svc.get_order(tenant_id, order_id))
	specimens = _run(_svc.list_specimens(tenant_id, order_id=order_id)) if order else []
	results = _run(_svc.list_results(tenant_id, order_id=order_id)) if order else []
	ctx = _base_ctx(tenant_id, "Order Detail")
	ctx.update({
		"order": order.model_dump(mode="json") if order else None,
		"specimens": [s.model_dump(mode="json") for s in specimens],
		"results": [r.model_dump(mode="json") for r in results],
		"not_found": order is None,
		"_template": "healthcare/lab/orders/detail.html",
	})
	return ctx


def specimen_tracker_view_model(
	tenant_id: str,
	order_id: str | None = None,
	status: str | None = None,
) -> dict[str, Any]:
	"""Build view model for the specimen tracker screen."""
	specimens = _run(_svc.list_specimens(tenant_id, order_id=order_id, status=status))
	rejected = [s for s in specimens if s.status == "rejected"]
	in_transit = [s for s in specimens if s.status == "in_transit"]
	ctx = _base_ctx(tenant_id, "Specimen Tracker")
	ctx.update({
		"specimens": [s.model_dump(mode="json") for s in specimens],
		"rejected_count": len(rejected),
		"in_transit_count": len(in_transit),
		"total_count": len(specimens),
		"filter": {"order_id": order_id, "status": status},
		"_template": "healthcare/lab/specimens/list.html",
	})
	return ctx


def specimen_detail_view_model(tenant_id: str, specimen_id: str) -> dict[str, Any]:
	"""Build view model for specimen detail including full custody chain."""
	specimen = _run(_svc.get_specimen(tenant_id, specimen_id))
	custody = _run(_svc.get_custody_chain(tenant_id, specimen_id)) if specimen else []
	ctx = _base_ctx(tenant_id, "Specimen Detail")
	ctx.update({
		"specimen": specimen.model_dump(mode="json") if specimen else None,
		"custody_chain": custody,
		"not_found": specimen is None,
		"_template": "healthcare/lab/specimens/detail.html",
	})
	return ctx


def result_workbench_view_model(
	tenant_id: str,
	order_id: str | None = None,
	critical_only: bool = False,
) -> dict[str, Any]:
	"""Build view model for the result workbench with critical flag highlighting."""
	results = _run(_svc.list_results(tenant_id, order_id=order_id, critical_only=critical_only))
	critical_results = [r for r in results if r.critical_value]
	preliminary = [r for r in results if r.result_status == "preliminary"]
	ctx = _base_ctx(tenant_id, "Result Workbench")
	ctx.update({
		"results": [r.model_dump(mode="json") for r in results],
		"critical_count": len(critical_results),
		"preliminary_count": len(preliminary),
		"total_count": len(results),
		"filter": {"order_id": order_id, "critical_only": critical_only},
		"_template": "healthcare/lab/results/workbench.html",
	})
	return ctx


def result_detail_view_model(tenant_id: str, result_id: str) -> dict[str, Any]:
	"""Build view model for a single result detail page."""
	result = _run(_svc.get_result(tenant_id, result_id))
	ctx = _base_ctx(tenant_id, "Result Detail")
	ctx.update({
		"result": result.model_dump(mode="json") if result else None,
		"not_found": result is None,
		"_template": "healthcare/lab/results/detail.html",
	})
	return ctx


def critical_values_view_model(tenant_id: str) -> dict[str, Any]:
	"""Build view model for the critical values alert screen."""
	all_cv = _run(_svc.list_critical_values(tenant_id))
	unacked = [n for n in all_cv if n.acknowledged_by is None]
	ctx = _base_ctx(tenant_id, "Critical Values")
	ctx.update({
		"notifications": [n.model_dump(mode="json") for n in all_cv],
		"unacknowledged": [n.model_dump(mode="json") for n in unacked],
		"unacknowledged_count": len(unacked),
		"total_count": len(all_cv),
		"_template": "healthcare/lab/critical_values/list.html",
	})
	return ctx


def qc_console_view_model(
	tenant_id: str,
	instrument_id: str | None = None,
) -> dict[str, Any]:
	"""Build view model for the QC console with Westgard violation highlights."""
	qc_runs = _run(_svc.list_qc_runs(tenant_id, instrument_id=instrument_id))
	instruments = _run(_svc.list_instruments(tenant_id))
	failed = [q for q in qc_runs if q.status == "failed"]
	pending_review = [q for q in qc_runs if q.status == "pending_review"]
	qc_hold_instruments = [i for i in instruments if i.status == "qc_hold"]
	ctx = _base_ctx(tenant_id, "QC Console")
	ctx.update({
		"qc_runs": [q.model_dump(mode="json") for q in qc_runs],
		"instruments": [i.model_dump(mode="json") for i in instruments],
		"failed_count": len(failed),
		"pending_review_count": len(pending_review),
		"qc_hold_count": len(qc_hold_instruments),
		"filter": {"instrument_id": instrument_id},
		"_template": "healthcare/lab/qc/console.html",
	})
	return ctx


def qc_run_detail_view_model(tenant_id: str, qc_id: str) -> dict[str, Any]:
	"""Build view model for a single QC run detail page."""
	qc_run = _run(_svc.get_qc_run(tenant_id, qc_id))
	ctx = _base_ctx(tenant_id, "QC Run Detail")
	ctx.update({
		"qc_run": qc_run.model_dump(mode="json") if qc_run else None,
		"not_found": qc_run is None,
		"_template": "healthcare/lab/qc/detail.html",
	})
	return ctx


def instrument_panel_view_model(tenant_id: str) -> dict[str, Any]:
	"""Build view model for the instrument management panel."""
	instruments = _run(_svc.list_instruments(tenant_id))
	online = [i for i in instruments if i.status == "online"]
	on_hold = [i for i in instruments if i.status == "qc_hold"]
	offline = [i for i in instruments if i.status == "offline"]
	ctx = _base_ctx(tenant_id, "Instrument Panel")
	ctx.update({
		"instruments": [i.model_dump(mode="json") for i in instruments],
		"online_count": len(online),
		"qc_hold_count": len(on_hold),
		"offline_count": len(offline),
		"total_count": len(instruments),
		"_template": "healthcare/lab/instruments/panel.html",
	})
	return ctx


def instrument_detail_view_model(tenant_id: str, instrument_id: str) -> dict[str, Any]:
	"""Build view model for a single instrument detail page."""
	instrument = _run(_svc.get_instrument(tenant_id, instrument_id))
	qc_runs = _run(_svc.list_qc_runs(tenant_id, instrument_id=instrument_id)) if instrument else []
	ctx = _base_ctx(tenant_id, "Instrument Detail")
	ctx.update({
		"instrument": instrument.model_dump(mode="json") if instrument else None,
		"qc_runs": [q.model_dump(mode="json") for q in qc_runs],
		"not_found": instrument is None,
		"_template": "healthcare/lab/instruments/detail.html",
	})
	return ctx


def referral_list_view_model(
	tenant_id: str,
	status: str | None = None,
) -> dict[str, Any]:
	"""Build view model for the external referral list."""
	referrals = _run(_svc.list_referrals(tenant_id, status=status))
	ctx = _base_ctx(tenant_id, "External Referrals")
	ctx.update({
		"referrals": referrals,
		"total_count": len(referrals),
		"filter": {"status": status},
		"_template": "healthcare/lab/referrals/list.html",
	})
	return ctx


def referral_detail_view_model(tenant_id: str, referral_id: str) -> dict[str, Any]:
	"""Build view model for a single referral detail page."""
	referral = _run(_svc.get_referral(tenant_id, referral_id))
	ctx = _base_ctx(tenant_id, "Referral Detail")
	ctx.update({
		"referral": referral,
		"not_found": referral is None,
		"_template": "healthcare/lab/referrals/detail.html",
	})
	return ctx


def tat_report_view_model(tenant_id: str, period: str = "today") -> dict[str, Any]:
	"""Build view model for the TAT analysis report."""
	report = _run(_svc.tat_monitoring(tenant_id, period=period))
	ctx = _base_ctx(tenant_id, "TAT Analysis")
	ctx.update({
		"report": report,
		"period": period,
		"_template": "healthcare/lab/reports/tat.html",
	})
	return ctx


def workload_report_view_model(tenant_id: str, period: str = "today") -> dict[str, Any]:
	"""Build view model for the workload summary report."""
	report = _run(_svc.lab_workload_report(tenant_id, period=period))
	ctx = _base_ctx(tenant_id, "Workload Report")
	ctx.update({
		"report": report,
		"period": period,
		"_template": "healthcare/lab/reports/workload.html",
	})
	return ctx


def qc_summary_report_view_model(tenant_id: str) -> dict[str, Any]:
	"""Build view model for the QC summary report."""
	report = _run(_svc.generate_qc_summary(tenant_id))
	ctx = _base_ctx(tenant_id, "QC Summary Report")
	ctx.update({
		"report": report,
		"_template": "healthcare/lab/reports/qc_summary.html",
	})
	return ctx


def critical_value_report_view_model(
	tenant_id: str,
	date_from: str | None = None,
	date_to: str | None = None,
) -> dict[str, Any]:
	"""Build view model for the critical value compliance report."""
	report = _run(_svc.generate_critical_value_report(tenant_id, date_from=date_from, date_to=date_to))
	ctx = _base_ctx(tenant_id, "Critical Value Report")
	ctx.update({
		"report": report,
		"date_from": date_from,
		"date_to": date_to,
		"_template": "healthcare/lab/reports/critical_values.html",
	})
	return ctx


def rejection_rate_report_view_model(tenant_id: str) -> dict[str, Any]:
	"""Build view model for the specimen rejection rate report."""
	report = _run(_svc.generate_rejection_report(tenant_id))
	ctx = _base_ctx(tenant_id, "Rejection Rate Report")
	ctx.update({
		"report": report,
		"_template": "healthcare/lab/reports/rejection_rate.html",
	})
	return ctx


# ── Flask Blueprint routes ────────────────────────────────────────────────────

@ui_bp.get("/")
def index():
	"""Redirect root to dashboard."""
	return redirect(url_for("healthcare_lab_ui.dashboard"))


@ui_bp.get("/dashboard")
def dashboard():
	"""LIS dashboard with KPI tiles."""
	return _render_json(dashboard_view_model(_tenant()))


@ui_bp.get("/orders")
def list_orders():
	"""Lab order queue with optional filters."""
	return _render_json(order_queue_view_model(
		_tenant(),
		status=request.args.get("status"),
		priority=request.args.get("priority"),
		patient_id=request.args.get("patient_id"),
	))


@ui_bp.get("/orders/new")
def new_order():
	"""Lab order entry form."""
	ctx = _base_ctx(_tenant(), "New Lab Order")
	ctx["_template"] = "healthcare/lab/orders/new.html"
	return _render_json(ctx)


@ui_bp.get("/orders/<order_id>")
def order_detail(order_id: str):
	"""Lab order detail page."""
	return _render_json(order_detail_view_model(_tenant(), order_id))


@ui_bp.get("/specimens")
def list_specimens():
	"""Specimen tracker."""
	return _render_json(specimen_tracker_view_model(
		_tenant(),
		order_id=request.args.get("order_id"),
		status=request.args.get("status"),
	))


@ui_bp.get("/specimens/<specimen_id>")
def specimen_detail(specimen_id: str):
	"""Specimen detail with full custody chain."""
	return _render_json(specimen_detail_view_model(_tenant(), specimen_id))


@ui_bp.get("/results")
def list_results():
	"""Result workbench."""
	critical_only = request.args.get("critical_only", "false").lower() == "true"
	return _render_json(result_workbench_view_model(
		_tenant(),
		order_id=request.args.get("order_id"),
		critical_only=critical_only,
	))


@ui_bp.get("/results/entry")
def result_entry():
	"""Result entry form."""
	ctx = _base_ctx(_tenant(), "Enter Result")
	ctx["_template"] = "healthcare/lab/results/entry.html"
	return _render_json(ctx)


@ui_bp.get("/results/<result_id>")
def result_detail(result_id: str):
	"""Result detail page."""
	return _render_json(result_detail_view_model(_tenant(), result_id))


@ui_bp.get("/critical-values")
def list_critical_values():
	"""Critical values alert screen."""
	return _render_json(critical_values_view_model(_tenant()))


@ui_bp.get("/qc")
def qc_console():
	"""QC console."""
	return _render_json(qc_console_view_model(
		_tenant(),
		instrument_id=request.args.get("instrument_id"),
	))


@ui_bp.get("/qc/<qc_id>")
def qc_run_detail(qc_id: str):
	"""QC run detail page."""
	return _render_json(qc_run_detail_view_model(_tenant(), qc_id))


@ui_bp.get("/instruments")
def instrument_panel():
	"""Instrument management panel."""
	return _render_json(instrument_panel_view_model(_tenant()))


@ui_bp.get("/instruments/<instrument_id>")
def instrument_detail(instrument_id: str):
	"""Instrument detail page."""
	return _render_json(instrument_detail_view_model(_tenant(), instrument_id))


@ui_bp.get("/referrals")
def list_referrals():
	"""External referral list."""
	return _render_json(referral_list_view_model(
		_tenant(),
		status=request.args.get("status"),
	))


@ui_bp.get("/referrals/<referral_id>")
def referral_detail(referral_id: str):
	"""Referral detail page."""
	return _render_json(referral_detail_view_model(_tenant(), referral_id))


@ui_bp.get("/reports")
def report_selector():
	"""Report selection screen."""
	ctx = _base_ctx(_tenant(), "Lab Reports")
	ctx.update({
		"available_reports": [
			{"id": "tat", "name": "Turnaround Time Analysis", "url": url_for("healthcare_lab_ui.report_tat")},
			{"id": "workload", "name": "Workload Summary", "url": url_for("healthcare_lab_ui.report_workload")},
			{"id": "qc_summary", "name": "QC Summary", "url": url_for("healthcare_lab_ui.report_qc_summary")},
			{"id": "critical_values", "name": "Critical Value Compliance", "url": url_for("healthcare_lab_ui.report_critical_values")},
			{"id": "rejection_rate", "name": "Rejection Rate", "url": url_for("healthcare_lab_ui.report_rejection_rate")},
		],
		"_template": "healthcare/lab/reports/index.html",
	})
	return _render_json(ctx)


@ui_bp.get("/reports/tat")
def report_tat():
	"""TAT analysis report."""
	return _render_json(tat_report_view_model(
		_tenant(),
		period=request.args.get("period", "today"),
	))


@ui_bp.get("/reports/workload")
def report_workload():
	"""Workload summary report."""
	return _render_json(workload_report_view_model(
		_tenant(),
		period=request.args.get("period", "today"),
	))


@ui_bp.get("/reports/qc-summary")
def report_qc_summary():
	"""QC summary report."""
	return _render_json(qc_summary_report_view_model(_tenant()))


@ui_bp.get("/reports/critical-values")
def report_critical_values():
	"""Critical value compliance report."""
	return _render_json(critical_value_report_view_model(
		_tenant(),
		date_from=request.args.get("date_from"),
		date_to=request.args.get("date_to"),
	))


@ui_bp.get("/reports/rejection-rate")
def report_rejection_rate():
	"""Specimen rejection rate report."""
	return _render_json(rejection_rate_report_view_model(_tenant()))


@ui_bp.get("/settings")
def settings():
	"""Lab settings page."""
	contract = get_capability_contract(_tenant())
	ctx = _base_ctx(_tenant(), "Lab Settings")
	ctx.update({
		"contract": contract,
		"configuration": contract["configuration"],
		"_template": "healthcare/lab/settings.html",
	})
	return _render_json(ctx)
