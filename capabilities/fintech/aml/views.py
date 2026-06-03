"""Flask Blueprint UI views for APG Anti-Money Laundering.

Renders Jinja2 templates for the AML workbench: dashboard, alerts, cases,
SAR/CTR queue, watchlist, reports.
"""
from __future__ import annotations

import asyncio
from typing import Any

from flask import Blueprint, jsonify, render_template_string, request

try:
	from .service import AMLService
except ImportError:  # pragma: no cover
	from service import AMLService  # type: ignore

views_bp = Blueprint("aml_views", __name__, url_prefix="/aml")

_DASHBOARD_TMPL = """
<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>AML Control Centre — {{ tenant_id }}</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{font-family:system-ui,sans-serif;background:#f8fafc;color:#111827;margin:0}
.topbar{background:#27374D;color:#fff;padding:.75rem 1.5rem;display:flex;align-items:center;gap:1rem}
.topbar h1{margin:0;font-size:1.1rem;font-weight:600}
.badge{background:#0f766e;color:#fff;border-radius:9999px;padding:.15rem .6rem;font-size:.75rem}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:1rem;padding:1.5rem}
.card{background:#fff;border-radius:8px;padding:1.25rem;box-shadow:0 1px 3px rgba(0,0,0,.08)}
.card h2{margin:0 0 .25rem;font-size:.75rem;font-weight:500;color:#6b7280;text-transform:uppercase;letter-spacing:.05em}
.card .val{font-size:2rem;font-weight:700}
.red{color:#b91c1c}.amber{color:#b45309}.green{color:#166534}
.section{padding:0 1.5rem 1.5rem}
.section h3{font-size:1rem;font-weight:600;margin:0 0 .75rem;color:#374151}
table{width:100%;border-collapse:collapse;font-size:.875rem}
th{background:#f1f5f9;padding:.5rem .75rem;text-align:left;font-weight:500;color:#374151}
td{padding:.5rem .75rem;border-bottom:1px solid #f1f5f9}
.chip{display:inline-block;border-radius:9999px;padding:.1rem .5rem;font-size:.7rem;font-weight:600}
.chip.critical{background:#fef2f2;color:#b91c1c}
.chip.high{background:#fff7ed;color:#c2410c}
.chip.medium{background:#fefce8;color:#92400e}
.chip.low{background:#f0fdf4;color:#166534}
.chip.open,.chip.under_review,.chip.escalated{background:#eff6ff;color:#1d4ed8}
.chip.closed,.chip.false_positive{background:#f9fafb;color:#6b7280}
a{color:#1d4ed8;text-decoration:none}a:hover{text-decoration:underline}
</style>
</head><body>
<div class="topbar"><h1>AML Control Centre</h1><span class="badge">{{ tenant_id }}</span></div>
<div class="grid">
  <div class="card"><h2>Open Alerts</h2>
    <div class="val {% if summary.open_alert_count > 10 %}red{% elif summary.open_alert_count > 5 %}amber{% else %}green{% endif %}">{{ summary.open_alert_count }}</div></div>
  <div class="card"><h2>Critical Alerts</h2><div class="val red">{{ summary.critical_alert_count }}</div></div>
  <div class="card"><h2>Open Cases</h2>
    <div class="val {% if summary.open_case_count > 5 %}amber{% else %}green{% endif %}">{{ summary.open_case_count }}</div></div>
  <div class="card"><h2>Pending SARs</h2><div class="val amber">{{ summary.pending_sar_count }}</div></div>
  <div class="card"><h2>CTRs Filed</h2><div class="val">{{ summary.ctr_count }}</div></div>
  <div class="card"><h2>FP Rate</h2>
    <div class="val {% if summary.false_positive_rate > 0.3 %}red{% elif summary.false_positive_rate > 0.15 %}amber{% else %}green{% endif %}">
      {{ "%.0f"|format(summary.false_positive_rate * 100) }}%</div></div>
  <div class="card"><h2>Watchlist Hits</h2><div class="val">{{ summary.watchlist_match_count }}</div></div>
  <div class="card"><h2>Active Rules</h2><div class="val green">{{ summary.rule_count }}</div></div>
</div>
<div class="section">
  <h3>Recent Alerts</h3>
  <table>
    <thead><tr><th>ID</th><th>Type</th><th>Severity</th><th>Score</th><th>Subject</th><th>Status</th><th>Created</th></tr></thead>
    <tbody>{% for a in alerts %}
    <tr>
      <td><a href="/aml/alerts/{{ a.id }}">{{ a.id[:8] }}</a></td>
      <td>{{ a.alert_type }}</td>
      <td><span class="chip {{ a.severity }}">{{ a.severity }}</span></td>
      <td>{{ a.risk_score }}</td>
      <td>{{ a.subject_reference }}</td>
      <td><span class="chip {{ a.status }}">{{ a.status }}</span></td>
      <td>{{ a.created_at }}</td>
    </tr>{% else %}
    <tr><td colspan="7" style="color:#9ca3af;text-align:center;padding:2rem">No alerts</td></tr>
    {% endfor %}</tbody>
  </table>
</div>
<div class="section">
  <h3>Open Cases</h3>
  <table>
    <thead><tr><th>ID</th><th>Type</th><th>Investigator</th><th>Subject</th><th>Priority</th><th>Status</th></tr></thead>
    <tbody>{% for c in cases %}
    <tr>
      <td><a href="/aml/cases/{{ c.id }}">{{ c.id[:8] }}</a></td>
      <td>{{ c.case_type }}</td>
      <td>{{ c.investigator_id }}</td>
      <td>{{ c.subject_reference }}</td>
      <td>{{ c.priority }}</td>
      <td><span class="chip open">{{ c.status }}</span></td>
    </tr>{% else %}
    <tr><td colspan="6" style="color:#9ca3af;text-align:center;padding:2rem">No open cases</td></tr>
    {% endfor %}</tbody>
  </table>
</div>
</body></html>
"""


def _run(coro: Any) -> Any:
	try:
		loop = asyncio.get_event_loop()
		if loop.is_running():
			import concurrent.futures
			with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
				return pool.submit(asyncio.run, coro).result()
		return loop.run_until_complete(coro)
	except RuntimeError:
		return asyncio.run(coro)


def _svc() -> AMLService:
	tenant_id = request.headers.get("X-Tenant-ID", request.args.get("tenant_id", "default"))
	actor_id = request.headers.get("X-Actor-ID", "system")
	return AMLService(tenant_id=tenant_id, actor_id=actor_id)


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

@views_bp.get("/")
@views_bp.get("/dashboard")
def dashboard_view() -> Any:
	svc = _svc()
	summary = _run(svc.dashboard_summary())
	alerts = _run(svc.list_alerts(limit=10))
	cases = _run(svc.list_cases(status="open", limit=10))
	return render_template_string(
		_DASHBOARD_TMPL,
		tenant_id=svc.tenant_id,
		summary=summary,
		alerts=alerts,
		cases=cases,
	)


@views_bp.get("/alerts")
def list_alerts_view() -> Any:
	svc = _svc()
	alerts = _run(svc.list_alerts(
		status=request.args.get("status"),
		severity=request.args.get("severity"),
		limit=int(request.args.get("limit", 50)),
	))
	return jsonify({"items": [a.model_dump(mode="json") for a in alerts], "count": len(alerts)})


@views_bp.get("/alerts/<alert_id>")
def detail_alert(alert_id: str) -> Any:
	svc = _svc()
	try:
		alert = _run(svc.get_alert(alert_id))
		return jsonify(alert.model_dump(mode="json"))
	except AssertionError as exc:
		return jsonify({"error": str(exc)}), 404


@views_bp.get("/cases")
def list_cases_view() -> Any:
	svc = _svc()
	cases = _run(svc.list_cases(
		status=request.args.get("status"),
		limit=int(request.args.get("limit", 50)),
	))
	return jsonify({"items": [c.model_dump(mode="json") for c in cases], "count": len(cases)})


@views_bp.get("/cases/<case_id>")
def detail_case(case_id: str) -> Any:
	svc = _svc()
	try:
		case = _run(svc.get_case(case_id))
		notes = _run(svc.list_notes(case_id))
		data = case.model_dump(mode="json")
		data["notes"] = [n.model_dump(mode="json") for n in notes]
		return jsonify(data)
	except AssertionError as exc:
		return jsonify({"error": str(exc)}), 404


@views_bp.get("/sar")
def list_sars_view() -> Any:
	svc = _svc()
	sars = _run(svc.list_sars())
	return jsonify({"items": [s.model_dump(mode="json") for s in sars], "count": len(sars)})


@views_bp.get("/ctr")
def list_ctrs_view() -> Any:
	svc = _svc()
	ctrs = _run(svc.list_ctrs())
	return jsonify({"items": [c.model_dump(mode="json") for c in ctrs], "count": len(ctrs)})


@views_bp.get("/watchlist")
def list_watchlist_view() -> Any:
	svc = _svc()
	matches = _run(svc.list_watchlist_matches())
	return jsonify({"items": [m.model_dump(mode="json") for m in matches], "count": len(matches)})


@views_bp.get("/reports")
def reports_view() -> Any:
	from datetime import datetime
	svc = _svc()
	report = _run(svc.regulatory_reporting(
		jurisdiction=request.args.get("jurisdiction", "US"),
		period_start=datetime(datetime.utcnow().year, 1, 1),
		period_end=datetime.utcnow(),
	))
	return jsonify(report.model_dump(mode="json"))


# ---------------------------------------------------------------------------
# View-model helpers (backward-compat + used by external renderers)
# ---------------------------------------------------------------------------

def _call(service: Any, method: str, *args: Any, **kwargs: Any) -> Any:
	"""Call a service method that may be sync or async."""
	import inspect
	result = getattr(service, method)(*args, **kwargs)
	if inspect.isawaitable(result):
		return _run(result)
	return result


def dashboard_model(service: Any, tenant_id: str = "default") -> dict[str, Any]:
	"""Serialisable dashboard context dict."""
	try:
		from .capability_contract import get_capability_contract
	except ImportError:
		from capability_contract import get_capability_contract  # type: ignore
	contract = get_capability_contract(tenant_id)
	summary = _call(service, "dashboard_summary", tenant_id) if _is_legacy(service) else _call(service, "dashboard_summary")
	return {
		"title": contract["display_name"],
		"tenant_id": tenant_id,
		"summary": summary,
		"routes": contract["ui"]["routes"],
		"theme": contract["theme"],
		"streaming": contract["streaming"],
	}


def _is_legacy(service: Any) -> bool:
	"""True if service is the legacy sync AntiMoneyLaunderingService."""
	return type(service).__name__ == "AntiMoneyLaunderingService"


def alert_console_model(service: Any, tenant_id: str = "default") -> dict[str, Any]:
	if _is_legacy(service):
		alerts = service.list_alerts(tenant_id)
		cases = service.list_cases(tenant_id)
	else:
		alerts_obj = _call(service, "list_alerts")
		cases_obj = _call(service, "list_cases")
		alerts = [a.model_dump(mode="json") if hasattr(a, "model_dump") else a for a in alerts_obj]
		cases = [c.model_dump(mode="json") if hasattr(c, "model_dump") else c for c in cases_obj]
	def _to_dict(item: Any) -> Any:
		if hasattr(item, "model_dump"):
			return item.model_dump(mode="json")
		return item

	return {
		"tenant_id": tenant_id,
		"alerts": [_to_dict(a) for a in alerts],
		"cases": [_to_dict(c) for c in cases],
		"actions": [
			"monitor_transaction", "create_alert", "triage_alert",
			"open_case", "draft_sar", "file_ctr", "watchlist_screen",
		],
	}


def rule_console_model(tenant_id: str = "default") -> dict[str, Any]:
	try:
		from .capability_contract import get_capability_contract
	except ImportError:
		from capability_contract import get_capability_contract  # type: ignore
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"rules": contract["rule_engine"]["rules"],
		"configuration": contract["configuration"],
	}


def capability_routes(tenant_id: str = "default") -> list[dict[str, Any]]:
	try:
		from .capability_contract import get_capability_contract
	except ImportError:
		from capability_contract import get_capability_contract  # type: ignore
	return get_capability_contract(tenant_id)["ui"]["routes"]
