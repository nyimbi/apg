"""APG Know Your Customer — Flask Blueprint UI views.

Mounts at ``/kyc``.  All views are tenant-scoped via X-Tenant-ID header
(falls back to query-param ``tenant_id``, then to ``"default"``).

Routes
------
GET  /kyc/                              dashboard
GET  /kyc/applications                  list with filtering
GET  /kyc/applications/new              new application form
POST /kyc/applications/new              submit new application
GET  /kyc/applications/<id>             full detail view
GET  /kyc/applications/<id>/edit        edit form
POST /kyc/applications/<id>/edit        submit edit
POST /kyc/applications/<id>/approve     approve action
POST /kyc/applications/<id>/reject      reject action
GET  /kyc/screening                     screening queue
GET  /kyc/reviews                       review queue
GET  /kyc/reports/expiry                expiry report
GET  /kyc/reports/risk                  risk distribution
GET  /kyc/reports/onboarding            onboarding analytics

Backward-compatible exports (used by existing tests and api.py callers):
    capability_routes(), dashboard_model(), profile_console_model(),
    rule_console_model(), KnowYourCustomerService (re-export).
"""

from __future__ import annotations

import asyncio
import datetime as _dt
from typing import Any

from flask import (
	Blueprint,
	flash,
	redirect,
	render_template_string,
	request,
	url_for,
)

try:
	from .service import KYCService, KnowYourCustomerService
	from .domain.rules import RuleViolation
	from .capability_contract import get_capability_contract
except ImportError:  # pragma: no cover
	from service import KYCService, KnowYourCustomerService  # type: ignore[no-redef]
	from domain.rules import RuleViolation  # type: ignore[no-redef]
	from capability_contract import get_capability_contract  # type: ignore[no-redef]


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatible view-model helpers (used by existing api.py / tests)
# ─────────────────────────────────────────────────────────────────────────────

def capability_routes(tenant_id: str = "default") -> list[dict[str, Any]]:
	"""Return the list of UI routes from the capability contract."""
	return get_capability_contract(tenant_id)["ui"]["routes"]


def dashboard_model(
	service: KnowYourCustomerService,
	tenant_id: str = "default",
) -> dict[str, Any]:
	"""Build the legacy dashboard view-model dict."""
	contract = get_capability_contract(tenant_id)
	return {
		"title": contract["display_name"],
		"tenant_id": tenant_id,
		"summary": service.dashboard_summary(tenant_id),
		"routes": contract["ui"]["routes"],
		"theme": contract["theme"],
		"streaming": contract["streaming"],
	}


def profile_console_model(
	service: KnowYourCustomerService,
	tenant_id: str = "default",
) -> dict[str, Any]:
	"""Build the legacy profile console view-model dict."""
	return {
		"tenant_id": tenant_id,
		"profiles": service.list_profiles(tenant_id),
		"actions": [
			"open_profile",
			"register_document",
			"record_screening",
			"score_risk",
			"record_decision",
			"register_kyc_agent",
		],
	}


def rule_console_model(tenant_id: str = "default") -> dict[str, Any]:
	"""Build the legacy rule console view-model dict."""
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"rules": contract["rule_engine"]["rules"],
		"configuration": contract["configuration"],
	}


# ─────────────────────────────────────────────────────────────────────────────
# Flask Blueprint
# ─────────────────────────────────────────────────────────────────────────────

kyc_ui_bp = Blueprint("kyc_ui", __name__, url_prefix="/kyc")


def _tenant() -> str:
	return (
		request.headers.get("X-Tenant-ID")
		or request.args.get("tenant_id")
		or "default"
	)


def _actor() -> str:
	return (
		request.headers.get("X-Actor-ID")
		or request.args.get("actor_id")
		or "ui_user"
	)


def _svc() -> KYCService:
	return KYCService(tenant_id=_tenant(), actor_id=_actor())


def _run(coro: Any) -> Any:
	"""Execute an async coroutine from sync Flask context."""
	loop = asyncio.new_event_loop()
	try:
		return loop.run_until_complete(coro)
	finally:
		loop.close()


# ─────────────────────────────────────────────────────────────────────────────
# Base template
# ─────────────────────────────────────────────────────────────────────────────

_BASE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ title }} — KYC | Datacraft</title>
  <link rel="stylesheet"
        href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
  <style>
    body { background: #f8f9fa; }
    .sidebar { min-height: 100vh; background: #1a1a2e; }
    .sidebar a { color: #adb5bd; text-decoration: none;
                 display: block; padding: .5rem 1rem; }
    .sidebar a:hover, .sidebar a.active
                { background: #16213e; color: #fff; }
    .badge-low          { background: #198754; }
    .badge-medium       { background: #ffc107; color: #000; }
    .badge-high         { background: #fd7e14; }
    .badge-very_high    { background: #dc3545; }
    .badge-unacceptable { background: #6f42c1; }
  </style>
</head>
<body>
<div class="container-fluid">
  <div class="row">
    <nav class="col-md-2 sidebar py-3">
      <div class="text-white fw-bold px-3 mb-3">KYC</div>
      <a href="{{ url_for('kyc_ui.dashboard') }}"
         {% if active == 'dashboard' %}class="active"{% endif %}>Dashboard</a>
      <a href="{{ url_for('kyc_ui.list_applications') }}"
         {% if active == 'applications' %}class="active"{% endif %}>Applications</a>
      <a href="{{ url_for('kyc_ui.screening_queue') }}"
         {% if active == 'screening' %}class="active"{% endif %}>Screening Queue</a>
      <a href="{{ url_for('kyc_ui.review_queue') }}"
         {% if active == 'reviews' %}class="active"{% endif %}>Reviews</a>
      <hr class="text-secondary">
      <a href="{{ url_for('kyc_ui.expiry_report') }}"
         {% if active == 'expiry' %}class="active"{% endif %}>Expiry Report</a>
      <a href="{{ url_for('kyc_ui.risk_report') }}"
         {% if active == 'risk' %}class="active"{% endif %}>Risk Report</a>
      <a href="{{ url_for('kyc_ui.onboarding_report') }}"
         {% if active == 'onboarding' %}class="active"{% endif %}>Onboarding</a>
    </nav>
    <main class="col-md-10 py-4 px-4">
      {% with messages = get_flashed_messages(with_categories=true) %}
        {% for cat, msg in messages %}
          <div class="alert alert-{{ 'danger' if cat == 'error' else cat }}
                      alert-dismissible fade show" role="alert">
            {{ msg }}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
          </div>
        {% endfor %}
      {% endwith %}
      {{ content }}
    </main>
  </div>
</div>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js">
</script>
</body>
</html>"""

_STATUS_BADGE_MAP: dict[str, str] = {
	"draft": "secondary",
	"in_progress": "info",
	"pending_review": "warning",
	"pending_edd": "danger",
	"approved": "success",
	"rejected": "danger",
	"expired": "dark",
	"suspended": "dark",
	"reactivation_required": "warning",
}


def _status_badge(status: str) -> str:
	cls = _STATUS_BADGE_MAP.get(status, "secondary")
	return f'<span class="badge bg-{cls}">{status.replace("_", " ").title()}</span>'


def _risk_badge(band: str) -> str:
	return (
		f'<span class="badge badge-{band}">'
		f'{band.replace("_", " ").title()}</span>'
	)


def _render(title: str, content: str, active: str = "") -> str:
	return render_template_string(_BASE, title=title, content=content, active=active)


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard
# ─────────────────────────────────────────────────────────────────────────────

@kyc_ui_bp.get("/")
def dashboard() -> str:
	"""KYC dashboard with KPI cards and quick-action buttons."""
	svc = _svc()
	apps = _run(svc.list_applications())
	total = len(apps)
	by_status: dict[str, int] = {}
	by_band: dict[str, int] = {}
	for a in apps:
		by_status[a.get("status", "unknown")] = (
			by_status.get(a.get("status", "unknown"), 0) + 1
		)
		by_band[a.get("risk_band", "unknown")] = (
			by_band.get(a.get("risk_band", "unknown"), 0) + 1
		)

	approved  = by_status.get("approved", 0)
	pending   = by_status.get("pending_review", 0)
	rejected  = by_status.get("rejected", 0)
	edd       = by_status.get("pending_edd", 0)
	high_risk = (
		by_band.get("high", 0)
		+ by_band.get("very_high", 0)
		+ by_band.get("unacceptable", 0)
	)

	expiry_res  = _run(svc.kyc_expiry_report(90))
	expiring_30 = len(expiry_res.get("expiring_within_30_days", []))

	new_url    = url_for("kyc_ui.create_application")
	all_url    = url_for("kyc_ui.list_applications")
	review_url = url_for("kyc_ui.review_queue")

	content = f"""
<h4 class="mb-4">KYC Dashboard
  <small class="text-muted fs-6">Tenant: {svc.tenant_id}</small>
</h4>
<div class="row g-3 mb-4">
  <div class="col-md-2"><div class="card text-bg-primary">
    <div class="card-body"><div class="fs-1 fw-bold">{total}</div><div>Total</div>
  </div></div></div>
  <div class="col-md-2"><div class="card text-bg-success">
    <div class="card-body"><div class="fs-1 fw-bold">{approved}</div><div>Approved</div>
  </div></div></div>
  <div class="col-md-2"><div class="card text-bg-warning">
    <div class="card-body"><div class="fs-1 fw-bold">{pending}</div><div>Pending Review</div>
  </div></div></div>
  <div class="col-md-2"><div class="card text-bg-danger">
    <div class="card-body"><div class="fs-1 fw-bold">{rejected}</div><div>Rejected</div>
  </div></div></div>
  <div class="col-md-2"><div class="card text-bg-dark">
    <div class="card-body"><div class="fs-1 fw-bold">{edd}</div><div>Pending EDD</div>
  </div></div></div>
  <div class="col-md-2"><div class="card text-bg-danger">
    <div class="card-body"><div class="fs-1 fw-bold">{high_risk}</div><div>High Risk</div>
  </div></div></div>
</div>
<div class="row g-3 mb-4">
  <div class="col-md-3"><div class="card border-warning">
    <div class="card-body">
      <h6>Expiring &le;30 days</h6>
      <div class="fs-3 fw-bold text-warning">{expiring_30}</div>
  </div></div></div>
</div>
<div class="mb-3">
  <a href="{new_url}" class="btn btn-primary me-2">+ New Application</a>
  <a href="{all_url}" class="btn btn-outline-secondary me-2">View All</a>
  <a href="{review_url}" class="btn btn-outline-warning">
    Review Queue ({pending})</a>
</div>
"""
	return _render("Dashboard", content, "dashboard")


# ─────────────────────────────────────────────────────────────────────────────
# Applications list
# ─────────────────────────────────────────────────────────────────────────────

@kyc_ui_bp.get("/applications")
def list_applications() -> str:
	"""Application list with status / risk-band filtering."""
	svc            = _svc()
	status_filter  = request.args.get("status", "")
	risk_filter    = request.args.get("risk_band", "")
	filters: dict[str, Any] = {}
	if status_filter:
		filters["status"] = status_filter
	if risk_filter:
		filters["risk_band"] = risk_filter

	apps = _run(svc.list_applications(filters or None))

	rows = ""
	for a in apps:
		aid = a["id"]
		detail_url = url_for("kyc_ui.detail_application", application_id=aid)
		rows += f"""<tr>
  <td><a href="{detail_url}">{aid[:12]}…</a></td>
  <td>{a.get('legal_name', '')}</td>
  <td>{a.get('customer_type', '')}</td>
  <td>{a.get('country_code', '')}</td>
  <td>{_status_badge(a.get('status', ''))}</td>
  <td>{_risk_badge(a.get('risk_band', 'low'))}</td>
  <td>{a.get('risk_score', 0)}</td>
  <td>{str(a.get('expiry_date', ''))[:10]}</td>
  <td><a href="{detail_url}" class="btn btn-sm btn-outline-primary">View</a></td>
</tr>"""

	status_opts = "".join(
		f'<option value="{s}"{"selected" if s == status_filter else ""}>{s}</option>'
		for s in [
			"draft", "in_progress", "pending_review", "pending_edd",
			"approved", "rejected", "expired",
		]
	)
	band_opts = "".join(
		f'<option value="{b}"{"selected" if b == risk_filter else ""}>{b}</option>'
		for b in ["low", "medium", "high", "very_high", "unacceptable"]
	)

	empty_row = (
		'<tr><td colspan="9" class="text-center text-muted">'
		'No applications found</td></tr>'
	)

	content = f"""
<div class="d-flex justify-content-between align-items-center mb-3">
  <h4>Applications ({len(apps)})</h4>
  <a href="{url_for('kyc_ui.create_application')}" class="btn btn-primary">+ New</a>
</div>
<form class="row g-2 mb-3">
  <div class="col-auto">
    <select name="status" class="form-select form-select-sm">
      <option value="">All Statuses</option>{status_opts}
    </select>
  </div>
  <div class="col-auto">
    <select name="risk_band" class="form-select form-select-sm">
      <option value="">All Risk Bands</option>{band_opts}
    </select>
  </div>
  <div class="col-auto">
    <button class="btn btn-sm btn-secondary">Filter</button>
  </div>
</form>
<div class="table-responsive">
  <table class="table table-hover table-sm">
    <thead><tr>
      <th>ID</th><th>Name</th><th>Type</th><th>Country</th>
      <th>Status</th><th>Risk Band</th><th>Score</th><th>Expiry</th><th></th>
    </tr></thead>
    <tbody>{rows or empty_row}</tbody>
  </table>
</div>
"""
	return _render("Applications", content, "applications")


# ─────────────────────────────────────────────────────────────────────────────
# Application detail
# ─────────────────────────────────────────────────────────────────────────────

@kyc_ui_bp.get("/applications/<application_id>")
def detail_application(application_id: str) -> str:
	"""Full application detail."""
	svc = _svc()
	try:
		app = _run(svc.get_application(application_id))
	except KeyError:
		flash("Application not found.", "error")
		return redirect(url_for("kyc_ui.list_applications"))

	edit_url    = url_for("kyc_ui.edit_application",    application_id=application_id)
	approve_url = url_for("kyc_ui.approve_application_ui", application_id=application_id)
	reject_url  = url_for("kyc_ui.reject_application_ui",  application_id=application_id)

	content = f"""
<div class="d-flex justify-content-between mb-3">
  <h4>{app.get('legal_name', '')}
    <small class="text-muted fs-6">{application_id}</small>
  </h4>
  <div>
    {_status_badge(app.get('status', ''))}
    &nbsp;
    {_risk_badge(app.get('risk_band', 'low'))}
  </div>
</div>
<div class="row">
  <div class="col-md-6">
    <div class="card mb-3">
      <div class="card-header">Customer</div>
      <div class="card-body">
        <table class="table table-sm mb-0">
          <tr><th>Customer ID</th><td>{app.get('customer_id', '')}</td></tr>
          <tr><th>Customer Type</th><td>{app.get('customer_type', '')}</td></tr>
          <tr><th>Country</th><td>{app.get('country_code', '')}</td></tr>
          <tr><th>KYC Tier</th><td>{app.get('kyc_tier', '')}</td></tr>
          <tr><th>Refugee</th>
              <td>{'Yes' if app.get('is_refugee') else 'No'}</td></tr>
          <tr><th>Informal Sector</th>
              <td>{'Yes' if app.get('is_informal_sector') else 'No'}</td></tr>
          <tr><th>Language</th>
              <td>{app.get('preferred_language', 'en')}</td></tr>
        </table>
      </div>
    </div>
  </div>
  <div class="col-md-6">
    <div class="card mb-3">
      <div class="card-header">Risk &amp; Dates</div>
      <div class="card-body">
        <table class="table table-sm mb-0">
          <tr><th>Risk Score</th>
              <td>{app.get('risk_score', 0)} / 100</td></tr>
          <tr><th>Risk Band</th>
              <td>{_risk_badge(app.get('risk_band', 'low'))}</td></tr>
          <tr><th>Expiry Date</th>
              <td>{str(app.get('expiry_date', ''))[:10] or '—'}</td></tr>
          <tr><th>Last Verified</th>
              <td>{str(app.get('last_verified_at', ''))[:10] or '—'}</td></tr>
          <tr><th>EDD Triggered</th>
              <td>{str(app.get('edd_triggered_at', ''))[:10] or '—'}</td></tr>
          <tr><th>Created</th>
              <td>{str(app.get('created_at', ''))[:10]}</td></tr>
        </table>
      </div>
    </div>
  </div>
</div>
<div class="mb-3">
  <a href="{edit_url}" class="btn btn-outline-primary me-2">Edit</a>
  <form method="post" action="{approve_url}" class="d-inline">
    <button class="btn btn-success me-2"
            onclick="return confirm('Approve this application?')">Approve</button>
  </form>
  <form method="post" action="{reject_url}" class="d-inline">
    <input type="hidden" name="reason" value="Rejected via UI">
    <button class="btn btn-danger"
            onclick="return confirm('Reject this application?')">Reject</button>
  </form>
</div>
<a href="{url_for('kyc_ui.list_applications')}" class="btn btn-link">
  &larr; Back to list</a>
"""
	return _render(f"Application — {app.get('legal_name', '')}", content, "applications")


# ─────────────────────────────────────────────────────────────────────────────
# Create application
# ─────────────────────────────────────────────────────────────────────────────

_CREATE_FORM = """
<h4 class="mb-4">New KYC Application</h4>
<form method="post" class="row g-3">
  <div class="col-md-6">
    <label class="form-label">Customer ID *</label>
    <input name="customer_id" class="form-control" required>
  </div>
  <div class="col-md-6">
    <label class="form-label">Legal Name *</label>
    <input name="legal_name" class="form-control" required>
  </div>
  <div class="col-md-4">
    <label class="form-label">Customer Type *</label>
    <select name="customer_type" class="form-select" required>
      <option value="individual">Individual</option>
      <option value="sole_proprietor">Sole Proprietor</option>
      <option value="business">Business</option>
      <option value="nonprofit">Nonprofit</option>
      <option value="government">Government</option>
      <option value="trust">Trust</option>
      <option value="partnership">Partnership</option>
    </select>
  </div>
  <div class="col-md-4">
    <label class="form-label">Jurisdiction (ISO-2) *</label>
    <input name="jurisdiction" class="form-control"
           maxlength="2" placeholder="KE" required>
  </div>
  <div class="col-md-4">
    <label class="form-label">KYC Tier</label>
    <select name="kyc_tier" class="form-select">
      <option value="standard">Standard</option>
      <option value="simplified">Simplified</option>
      <option value="enhanced">Enhanced</option>
    </select>
  </div>
  <div class="col-md-6">
    <label class="form-label">Consent Reference</label>
    <input name="consent_reference" class="form-control"
           placeholder="CONSENT-2026-…">
  </div>
  <div class="col-md-3">
    <div class="form-check mt-4">
      <input class="form-check-input" type="checkbox"
             name="is_refugee" id="refugee">
      <label class="form-check-label" for="refugee">Refugee</label>
    </div>
  </div>
  <div class="col-md-3">
    <div class="form-check mt-4">
      <input class="form-check-input" type="checkbox"
             name="is_informal_sector" id="informal">
      <label class="form-check-label" for="informal">Informal Sector</label>
    </div>
  </div>
  <div class="col-12">
    <button type="submit" class="btn btn-primary me-2">Create Application</button>
    <a href="{{ url_for('kyc_ui.list_applications') }}"
       class="btn btn-outline-secondary">Cancel</a>
  </div>
</form>
"""


@kyc_ui_bp.get("/applications/new")
def create_application() -> str:
	return _render("New Application", render_template_string(_CREATE_FORM), "applications")


@kyc_ui_bp.post("/applications/new")
def submit_create_application() -> Any:
	f = request.form
	try:
		result = _run(_svc().start_kyc_application(
			customer_id=f["customer_id"],
			customer_type=f["customer_type"],
			jurisdiction=f["jurisdiction"],
			legal_name=f.get("legal_name", ""),
			consent_reference=f.get("consent_reference", ""),
			kyc_tier=f.get("kyc_tier", "standard"),
			is_refugee="is_refugee" in f,
			is_informal_sector="is_informal_sector" in f,
		))
		flash(f"Application created: {result['id']}", "success")
		return redirect(url_for("kyc_ui.detail_application",
		                        application_id=result["id"]))
	except (RuleViolation, ValueError, KeyError) as exc:
		flash(str(exc), "error")
		return redirect(url_for("kyc_ui.create_application"))


# ─────────────────────────────────────────────────────────────────────────────
# Edit application
# ─────────────────────────────────────────────────────────────────────────────

@kyc_ui_bp.get("/applications/<application_id>/edit")
def edit_application(application_id: str) -> str:
	svc = _svc()
	try:
		app = _run(svc.get_application(application_id))
	except KeyError:
		flash("Application not found.", "error")
		return redirect(url_for("kyc_ui.list_applications"))

	tier_opts = "".join(
		f'<option value="{t}"{"selected" if t == app.get("kyc_tier") else ""}>{t}</option>'
		for t in ["simplified", "standard", "enhanced"]
	)
	status_opts = "".join(
		f'<option value="{s}"{"selected" if s == app.get("status") else ""}>{s}</option>'
		for s in [
			"draft", "in_progress", "pending_review", "pending_edd",
			"approved", "rejected", "expired", "suspended",
		]
	)

	content = f"""
<h4>Edit Application — {app.get('legal_name', '')}
  <small class="text-muted">{application_id[:12]}…</small>
</h4>
<form method="post" class="row g-3">
  <div class="col-md-6">
    <label class="form-label">Legal Name</label>
    <input name="legal_name" class="form-control"
           value="{app.get('legal_name', '')}">
  </div>
  <div class="col-md-4">
    <label class="form-label">KYC Tier</label>
    <select name="kyc_tier" class="form-select">{tier_opts}</select>
  </div>
  <div class="col-md-4">
    <label class="form-label">Status</label>
    <select name="status" class="form-select">{status_opts}</select>
  </div>
  <div class="col-12">
    <button type="submit" class="btn btn-primary me-2">Save</button>
    <a href="{url_for('kyc_ui.detail_application', application_id=application_id)}"
       class="btn btn-outline-secondary">Cancel</a>
  </div>
</form>
"""
	return _render("Edit Application", content, "applications")


@kyc_ui_bp.post("/applications/<application_id>/edit")
def submit_edit_application(application_id: str) -> Any:
	fields: dict[str, Any] = {k: v for k, v in request.form.items() if v}
	try:
		_run(_svc().update_application(application_id, **fields))
		flash("Application updated.", "success")
	except (RuleViolation, ValueError) as exc:
		flash(str(exc), "error")
	return redirect(url_for("kyc_ui.detail_application",
	                        application_id=application_id))


# ─────────────────────────────────────────────────────────────────────────────
# Approve / reject POST actions (called from detail page)
# ─────────────────────────────────────────────────────────────────────────────

@kyc_ui_bp.post("/applications/<application_id>/approve")
def approve_application_ui(application_id: str) -> Any:
	try:
		_run(_svc().approve_application(application_id, reviewer_id=_actor()))
		flash("Application approved.", "success")
	except (RuleViolation, ValueError, KeyError) as exc:
		flash(str(exc), "error")
	return redirect(url_for("kyc_ui.detail_application",
	                        application_id=application_id))


@kyc_ui_bp.post("/applications/<application_id>/reject")
def reject_application_ui(application_id: str) -> Any:
	reason = request.form.get("reason", "Rejected via UI")
	try:
		_run(_svc().reject_application(
			application_id, reason=reason, reviewer_id=_actor()
		))
		flash("Application rejected.", "warning")
	except (RuleViolation, ValueError, KeyError) as exc:
		flash(str(exc), "error")
	return redirect(url_for("kyc_ui.detail_application",
	                        application_id=application_id))


# ─────────────────────────────────────────────────────────────────────────────
# Screening queue
# ─────────────────────────────────────────────────────────────────────────────

@kyc_ui_bp.get("/screening")
def screening_queue() -> str:
	"""Applications pending PEP / sanctions screening."""
	svc  = _svc()
	apps = (
		_run(svc.list_applications({"status": "in_progress"}))
		+ _run(svc.list_applications({"status": "pending_review"}))
	)

	rows = ""
	for a in apps:
		aid = a["id"]
		detail_url = url_for("kyc_ui.detail_application", application_id=aid)
		rows += f"""<tr>
  <td><a href="{detail_url}">{aid[:12]}…</a></td>
  <td>{a.get('legal_name', '')}</td>
  <td>{a.get('country_code', '')}</td>
  <td>{_status_badge(a.get('status', ''))}</td>
  <td>{_risk_badge(a.get('risk_band', 'low'))}</td>
  <td>
    <a href="{detail_url}" class="btn btn-sm btn-outline-warning">Review</a>
  </td>
</tr>"""

	empty = (
		'<tr><td colspan="6" class="text-center text-muted">'
		'No applications pending screening</td></tr>'
	)
	content = f"""
<h4 class="mb-3">Screening Queue ({len(apps)})</h4>
<div class="table-responsive">
  <table class="table table-hover table-sm">
    <thead><tr>
      <th>ID</th><th>Name</th><th>Country</th>
      <th>Status</th><th>Risk</th><th></th>
    </tr></thead>
    <tbody>{rows or empty}</tbody>
  </table>
</div>
"""
	return _render("Screening Queue", content, "screening")


# ─────────────────────────────────────────────────────────────────────────────
# Review queue
# ─────────────────────────────────────────────────────────────────────────────

@kyc_ui_bp.get("/reviews")
def review_queue() -> str:
	"""Applications awaiting human review, including EDD queue."""
	svc      = _svc()
	apps     = _run(svc.list_applications({"status": "pending_review"}))
	edd_apps = _run(svc.list_applications({"status": "pending_edd"}))

	def _table(items: list[dict[str, Any]], label: str) -> str:
		rows = ""
		for a in items:
			aid        = a["id"]
			detail_url = url_for("kyc_ui.detail_application", application_id=aid)
			approve_url = url_for("kyc_ui.approve_application_ui",
			                      application_id=aid)
			rows += f"""<tr>
  <td><a href="{detail_url}">{aid[:12]}…</a></td>
  <td>{a.get('legal_name', '')}</td>
  <td>{a.get('customer_type', '')}</td>
  <td>{_risk_badge(a.get('risk_band', 'low'))}</td>
  <td>
    <form method="post" action="{approve_url}" class="d-inline">
      <button class="btn btn-sm btn-success me-1"
              onclick="return confirm('Approve?')">Approve</button>
    </form>
    <a href="{detail_url}" class="btn btn-sm btn-outline-primary">View</a>
  </td>
</tr>"""
		empty = (
			'<tr><td colspan="5" class="text-center text-muted">None</td></tr>'
		)
		return f"""
<h5 class="mt-4">{label} ({len(items)})</h5>
<div class="table-responsive">
  <table class="table table-hover table-sm">
    <thead><tr>
      <th>ID</th><th>Name</th><th>Type</th><th>Risk</th><th></th>
    </tr></thead>
    <tbody>{rows or empty}</tbody>
  </table>
</div>"""

	content = (
		"<h4 class='mb-3'>Review Queue</h4>"
		+ _table(apps, "Pending Standard Review")
		+ _table(edd_apps, "Pending Enhanced Due Diligence (EDD)")
	)
	return _render("Review Queue", content, "reviews")


# ─────────────────────────────────────────────────────────────────────────────
# Reports
# ─────────────────────────────────────────────────────────────────────────────

@kyc_ui_bp.get("/reports/expiry")
def expiry_report() -> str:
	"""KYC expiry pipeline report."""
	days       = int(request.args.get("days_ahead", 90))
	svc        = _svc()
	result     = _run(svc.kyc_expiry_report(days_ahead=days))

	def _id_links(ids: list[str]) -> str:
		if not ids:
			return "<p class='text-muted'>None.</p>"
		return "<ul class='list-unstyled mb-0'>" + "".join(
			f'<li><a href="{url_for("kyc_ui.detail_application", application_id=i)}">'
			f'{i[:16]}…</a></li>'
			for i in ids
		) + "</ul>"

	content = f"""
<h4 class="mb-3">KYC Expiry Report</h4>
<form class="row g-2 mb-3">
  <div class="col-auto">
    <label class="form-label">Days ahead</label>
    <input name="days_ahead" type="number" class="form-control"
           value="{days}" min="1" max="365">
  </div>
  <div class="col-auto align-self-end">
    <button class="btn btn-secondary">Refresh</button>
  </div>
</form>
<div class="row">
  <div class="col-md-4">
    <div class="card border-danger mb-3">
      <div class="card-header text-danger fw-bold">
        Already Expired ({len(result['already_expired'])})
      </div>
      <div class="card-body">
        {_id_links(result['already_expired'])}
      </div>
    </div>
  </div>
  <div class="col-md-4">
    <div class="card border-warning mb-3">
      <div class="card-header text-warning fw-bold">
        Expiring &le;30 days ({len(result['expiring_within_30_days'])})
      </div>
      <div class="card-body">
        {_id_links(result['expiring_within_30_days'])}
      </div>
    </div>
  </div>
  <div class="col-md-4">
    <div class="card border-info mb-3">
      <div class="card-header text-info fw-bold">
        Expiring &le;{days} days ({len(result['expiring_within_90_days'])})
      </div>
      <div class="card-body">
        {_id_links(result['expiring_within_90_days'])}
      </div>
    </div>
  </div>
</div>
<p class="text-muted small">Generated: {result['generated_at'][:19]}</p>
"""
	return _render("Expiry Report", content, "expiry")


@kyc_ui_bp.get("/reports/risk")
def risk_report() -> str:
	"""Risk band and country distribution."""
	svc        = _svc()
	apps       = _run(svc.list_applications())
	by_band: dict[str, int]    = {}
	by_country: dict[str, int] = {}
	for a in apps:
		b = a.get("risk_band", "unknown")
		c = a.get("country_code", "UNKNOWN")
		by_band[b]    = by_band.get(b, 0) + 1
		by_country[c] = by_country.get(c, 0) + 1

	band_rows = "".join(
		f'<tr><td>{_risk_badge(b)}</td><td>{cnt}</td></tr>'
		for b, cnt in sorted(by_band.items(), key=lambda x: -x[1])
	) or '<tr><td colspan="2" class="text-muted">No data</td></tr>'

	country_rows = "".join(
		f'<tr><td>{c}</td><td>{cnt}</td></tr>'
		for c, cnt in sorted(by_country.items(), key=lambda x: -x[1])[:20]
	) or '<tr><td colspan="2" class="text-muted">No data</td></tr>'

	content = f"""
<h4 class="mb-3">Risk Report</h4>
<div class="row">
  <div class="col-md-4">
    <h6>By Risk Band</h6>
    <table class="table table-sm">
      <thead><tr><th>Band</th><th>Count</th></tr></thead>
      <tbody>{band_rows}</tbody>
    </table>
  </div>
  <div class="col-md-4">
    <h6>By Country (top 20)</h6>
    <table class="table table-sm">
      <thead><tr><th>Country</th><th>Count</th></tr></thead>
      <tbody>{country_rows}</tbody>
    </table>
  </div>
</div>
"""
	return _render("Risk Report", content, "risk")


@kyc_ui_bp.get("/reports/onboarding")
def onboarding_report() -> str:
	"""Onboarding funnel analytics."""
	period = request.args.get("period", _dt.date.today().strftime("%Y-%m"))
	svc    = _svc()
	result = _run(svc.onboarding_analytics(period))

	total      = result.get("total_started", 0)
	completed  = result.get("completed", 0)
	abandoned  = result.get("abandoned", 0)
	comp_rate  = round(result.get("completion_rate", 0) * 100, 1)
	avg_mins   = round(result.get("avg_time_to_complete_seconds", 0) / 60, 1)

	drop_rows = "".join(
		f'<tr><td>{step}</td><td>{cnt}</td></tr>'
		for step, cnt in result.get("drop_off_by_step", {}).items()
	) or '<tr><td colspan="2" class="text-muted">No drop-off data</td></tr>'

	channel_rows = "".join(
		f'<tr><td>{ch}</td><td>{cnt}</td></tr>'
		for ch, cnt in result.get("by_channel", {}).items()
	) or '<tr><td colspan="2" class="text-muted">No data</td></tr>'

	content = f"""
<h4 class="mb-3">Onboarding Analytics</h4>
<form class="row g-2 mb-3">
  <div class="col-auto">
    <input name="period" class="form-control"
           value="{period}" placeholder="YYYY-MM">
  </div>
  <div class="col-auto">
    <button class="btn btn-secondary">Load</button>
  </div>
</form>
<div class="row g-3 mb-4">
  <div class="col-md-2"><div class="card text-bg-primary">
    <div class="card-body">
      <div class="fs-2 fw-bold">{total}</div><div>Started</div>
  </div></div></div>
  <div class="col-md-2"><div class="card text-bg-success">
    <div class="card-body">
      <div class="fs-2 fw-bold">{completed}</div><div>Completed</div>
  </div></div></div>
  <div class="col-md-2"><div class="card text-bg-warning">
    <div class="card-body">
      <div class="fs-2 fw-bold">{abandoned}</div><div>Abandoned</div>
  </div></div></div>
  <div class="col-md-2"><div class="card text-bg-info">
    <div class="card-body">
      <div class="fs-2 fw-bold">{comp_rate}%</div><div>Completion Rate</div>
  </div></div></div>
  <div class="col-md-2"><div class="card text-bg-secondary">
    <div class="card-body">
      <div class="fs-2 fw-bold">{avg_mins}m</div><div>Avg Time</div>
  </div></div></div>
</div>
<div class="row">
  <div class="col-md-4">
    <h6>Drop-off by Step</h6>
    <table class="table table-sm">
      <thead><tr><th>Step</th><th>Drop-offs</th></tr></thead>
      <tbody>{drop_rows}</tbody>
    </table>
  </div>
  <div class="col-md-4">
    <h6>By Channel</h6>
    <table class="table table-sm">
      <thead><tr><th>Channel</th><th>Count</th></tr></thead>
      <tbody>{channel_rows}</tbody>
    </table>
  </div>
</div>
"""
	return _render("Onboarding Analytics", content, "onboarding")
