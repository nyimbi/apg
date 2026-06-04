"""
APG Audit Logging — Flask Blueprint UI views.

Plain Flask blueprint (no flask_appbuilder).
URL prefix: /audl

Views
-----
  GET  /audl/                      dashboard (KPIs, recent events, compliance status)
  GET  /audl/events                event list with filters
  GET  /audl/events/<id>           event detail
  GET  /audl/trails                trail list
  GET  /audl/trails/<id>           trail detail
  GET  /audl/compliance            compliance report list
  GET  /audl/dsr                   data subject request list
  GET  /audl/dsr/<id>              DSR detail
  GET  /audl/evidence              evidence package list
  GET  /audl/tamper                tamper scan list
  GET  /audl/reports/risk          risk summary report

All views render via render_template() with Jinja2 templates from
templates/audit/.  JSON fallback is returned when the Accept header
prefers application/json (useful for headless clients).

© 2025 Datacraft  www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from flask import (
	Blueprint,
	Response,
	flash,
	g,
	jsonify,
	redirect,
	render_template,
	request,
	url_for,
)

from .models import (
	AuditQueryCreate,
	ComplianceFramework,
	ComplianceReportCreate,
	EvidencePackageCreate,
	TamperDetectionCreate,
	uuid7str,
)
from .service import AuditLoggingService

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Blueprint
# ---------------------------------------------------------------------------

audl_ui = Blueprint(
	"audl_ui",
	__name__,
	url_prefix="/audl",
	template_folder="templates",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _svc() -> AuditLoggingService:
	tenant_id = getattr(g, "tenant_id", None) or request.headers.get("X-Tenant-Id", "default")
	actor_id  = getattr(g, "actor_id",  None) or request.headers.get("X-Actor-Id",  "anonymous")
	db        = getattr(g, "db_session", None)
	return AuditLoggingService(db_session=db, tenant_id=tenant_id, actor_id=actor_id)


def _run(coro):
	try:
		loop = asyncio.get_event_loop()
		if loop.is_running():
			import concurrent.futures
			with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
				return pool.submit(asyncio.run, coro).result()
		return loop.run_until_complete(coro)
	except RuntimeError:
		return asyncio.run(coro)


def _wants_json() -> bool:
	best = request.accept_mimetypes.best_match(["application/json", "text/html"])
	return best == "application/json"


def _render_or_json(template: str, ctx: dict[str, Any], status: int = 200) -> Response:
	if _wants_json():
		# Sanitise for JSON: convert pydantic models
		safe: dict[str, Any] = {}
		for k, v in ctx.items():
			if hasattr(v, "model_dump"):
				safe[k] = v.model_dump(mode="json")
			elif isinstance(v, list) and v and hasattr(v[0], "model_dump"):
				safe[k] = [i.model_dump(mode="json") for i in v]
			else:
				safe[k] = v
		return jsonify(safe), status
	return render_template(template, **ctx), status


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@audl_ui.get("/")
@audl_ui.get("/dashboard")
def dashboard():
	"""
	Main audit dashboard.

	KPIs: total events (30d), high-risk events, compliance violations, active trails.
	"""
	svc  = _svc()
	now  = datetime.now(timezone.utc)
	ps   = now - timedelta(days=30)

	try:
		summary = _run(svc.risk_summary(ps, now))
		trails  = _run(svc.list_trails(active_only=True))
	except Exception as exc:
		log.exception("dashboard error")
		flash(f"Dashboard data unavailable: {exc}", "error")
		summary = None
		trails  = []

	recent_events = sorted(
		[ev for ev in svc._events.values() if ev.tenant_id == svc.tenant_id],
		key=lambda e: e.created_at,
		reverse=True,
	)[:20]

	ctx = {
		"summary":       summary,
		"trails":        trails,
		"recent_events": recent_events,
		"tenant_id":     svc.tenant_id,
		"now":           now,
	}
	return _render_or_json("audit/dashboard.html", ctx)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

@audl_ui.get("/events")
def list_events():
	"""
	Event list with optional filter params:
	  event_type, actor_id, resource_id, date_start, date_end,
	  risk_min, success, limit, offset
	"""
	svc = _svc()
	p   = request.args

	def _dt(key: str) -> datetime | None:
		v = p.get(key)
		return datetime.fromisoformat(v) if v else None

	from .models import AuditEventType, EventSource
	q = AuditQueryCreate(
		tenant_id    = svc.tenant_id,
		query_type   = "structured",
		event_types  = [AuditEventType(p["event_type"])] if p.get("event_type") else [],
		actor_ids    = [p["actor_id"]]    if p.get("actor_id")    else [],
		resource_ids = [p["resource_id"]] if p.get("resource_id") else [],
		date_start   = _dt("date_start"),
		date_end     = _dt("date_end"),
		risk_score_min = float(p["risk_min"]) if p.get("risk_min") else None,
		success      = None if not p.get("success") else p["success"].lower() == "true",
		limit        = int(p.get("limit",  100)),
		offset       = int(p.get("offset", 0)),
		requested_by = svc.actor_id,
	)
	result = _run(svc.audit_trail_search(q))
	ctx    = {"result": result, "query": q, "tenant_id": svc.tenant_id}
	return _render_or_json("audit/events.html", ctx)


@audl_ui.get("/events/<event_id>")
def event_detail(event_id: str):
	"""Full event detail view with integrity status."""
	svc = _svc()
	ev  = svc._events.get(event_id)
	if ev is None or ev.tenant_id != svc.tenant_id:
		flash("Event not found", "error")
		return redirect(url_for("audl_ui.list_events"))
	ctx = {
		"event":           ev,
		"integrity_ok":    ev.verify_integrity(),
		"tenant_id":       svc.tenant_id,
	}
	return _render_or_json("audit/event_detail.html", ctx)


# ---------------------------------------------------------------------------
# Trails
# ---------------------------------------------------------------------------

@audl_ui.get("/trails")
def list_trails():
	svc         = _svc()
	active_only = request.args.get("active_only", "true").lower() != "false"
	trails      = _run(svc.list_trails(active_only=active_only))
	ctx         = {"trails": trails, "tenant_id": svc.tenant_id}
	return _render_or_json("audit/trails.html", ctx)


@audl_ui.get("/trails/<trail_id>")
def trail_detail(trail_id: str):
	svc = _svc()
	try:
		trail = _run(svc.get_trail(trail_id))
	except KeyError:
		flash("Trail not found", "error")
		return redirect(url_for("audl_ui.list_trails"))
	# Events associated with trail (naive: all events for tenant, in production filter by trail_id FK)
	trail_events = sorted(
		[ev for ev in svc._events.values() if ev.tenant_id == svc.tenant_id],
		key=lambda e: e.created_at, reverse=True,
	)[:50]
	ctx = {"trail": trail, "events": trail_events, "tenant_id": svc.tenant_id}
	return _render_or_json("audit/trail_detail.html", ctx)


# ---------------------------------------------------------------------------
# Compliance reports
# ---------------------------------------------------------------------------

@audl_ui.get("/compliance")
def list_compliance_reports():
	svc     = _svc()
	reports = [r for r in svc._reports.values() if r.tenant_id == svc.tenant_id]
	reports.sort(key=lambda r: r.created_at, reverse=True)
	ctx = {"reports": reports, "frameworks": list(ComplianceFramework), "tenant_id": svc.tenant_id}
	return _render_or_json("audit/compliance.html", ctx)


@audl_ui.post("/compliance/generate")
def generate_compliance_report():
	"""
	Trigger report generation from a form POST.

	Form fields: framework, period_start, period_end, export_format
	"""
	svc     = _svc()
	form    = request.form
	now     = datetime.now(timezone.utc)

	try:
		req = ComplianceReportCreate(
			tenant_id    = svc.tenant_id,
			framework    = ComplianceFramework(form.get("framework", "GDPR")),
			period_start = datetime.fromisoformat(form.get("period_start",
								(now - timedelta(days=30)).isoformat())),
			period_end   = datetime.fromisoformat(form.get("period_end", now.isoformat())),
			requested_by = svc.actor_id,
			export_format= form.get("export_format", "json"),
		)
		report = _run(svc.compliance_report(req))
		flash(f"Report {report.id} generated ({report.violation_count} violations).", "success")
	except Exception as exc:
		log.exception("compliance report generation failed")
		flash(f"Report generation failed: {exc}", "error")

	return redirect(url_for("audl_ui.list_compliance_reports"))


# ---------------------------------------------------------------------------
# Data Subject Requests
# ---------------------------------------------------------------------------

@audl_ui.get("/dsr")
def list_dsrs():
	svc  = _svc()
	dsrs = _run(svc.list_dsrs())
	ctx  = {"dsrs": dsrs, "tenant_id": svc.tenant_id}
	return _render_or_json("audit/dsr_list.html", ctx)


@audl_ui.get("/dsr/<dsr_id>")
def dsr_detail(dsr_id: str):
	svc = _svc()
	rec = svc._dsrs.get(dsr_id)
	if rec is None or rec.tenant_id != svc.tenant_id:
		flash("DSR not found", "error")
		return redirect(url_for("audl_ui.list_dsrs"))
	ctx = {"dsr": rec, "tenant_id": svc.tenant_id}
	return _render_or_json("audit/dsr_detail.html", ctx)


# ---------------------------------------------------------------------------
# Evidence packages
# ---------------------------------------------------------------------------

@audl_ui.get("/evidence")
def list_evidence_packages():
	svc      = _svc()
	packages = _run(svc.list_evidence_packages())
	ctx      = {"packages": packages, "tenant_id": svc.tenant_id}
	return _render_or_json("audit/evidence.html", ctx)


@audl_ui.get("/evidence/<pkg_id>")
def evidence_detail(pkg_id: str):
	svc = _svc()
	try:
		pkg = _run(svc.get_evidence_package(pkg_id))
	except KeyError:
		flash("Evidence package not found", "error")
		return redirect(url_for("audl_ui.list_evidence_packages"))
	ctx = {"package": pkg, "tenant_id": svc.tenant_id}
	return _render_or_json("audit/evidence_detail.html", ctx)


# ---------------------------------------------------------------------------
# Tamper detection
# ---------------------------------------------------------------------------

@audl_ui.get("/tamper")
def list_tamper_scans():
	svc   = _svc()
	scans = [s for s in svc._tampers.values() if s.tenant_id == svc.tenant_id]
	scans.sort(key=lambda s: s.created_at, reverse=True)
	ctx = {"scans": scans, "tenant_id": svc.tenant_id}
	return _render_or_json("audit/tamper.html", ctx)


@audl_ui.post("/tamper/run")
def run_tamper_scan():
	"""Trigger an on-demand tamper detection scan."""
	svc = _svc()
	try:
		scan = _run(svc.tamper_detection(TamperDetectionCreate(
			tenant_id  = svc.tenant_id,
			scan_type  = "on-demand",
			scanned_by = svc.actor_id,
		)))
		flash(
			f"Scan {scan.id}: {scan.events_scanned} events checked, "
			f"{scan.events_suspect} suspect — status {scan.status}.",
			"success" if scan.events_suspect == 0 else "warning",
		)
	except Exception as exc:
		log.exception("tamper scan failed")
		flash(f"Scan failed: {exc}", "error")
	return redirect(url_for("audl_ui.list_tamper_scans"))


# ---------------------------------------------------------------------------
# Risk summary report
# ---------------------------------------------------------------------------

@audl_ui.get("/reports/risk")
def risk_report():
	"""Risk summary for the last 30 days (or custom window via query params)."""
	svc = _svc()
	now = datetime.now(timezone.utc)
	p   = request.args
	ps  = datetime.fromisoformat(p["period_start"]) if p.get("period_start") else now - timedelta(days=30)
	pe  = datetime.fromisoformat(p["period_end"])   if p.get("period_end")   else now
	summary = _run(svc.risk_summary(ps, pe))
	ctx     = {"summary": summary, "tenant_id": svc.tenant_id}
	return _render_or_json("audit/risk_report.html", ctx)


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@audl_ui.errorhandler(404)
def not_found(e):
	if _wants_json():
		return jsonify({"error": "not found"}), 404
	return render_template("audit/404.html"), 404


@audl_ui.errorhandler(500)
def server_error(e):
	log.exception("audl_ui unhandled error")
	if _wants_json():
		return jsonify({"error": "internal server error"}), 500
	return render_template("audit/500.html"), 500


__all__ = ["audl_ui"]
