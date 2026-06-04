"""Flask Blueprint UI views for Lease Management (realestate_lea).

Renders HTML pages for the lease management console.
All data is fetched from LeaseManagementService.

Routes
------
GET  /realestate/lea/                         → redirect → dashboard
GET  /realestate/lea/dashboard                → dashboard with KPIs
GET  /realestate/lea/leases                   → lease registry (list)
GET  /realestate/lea/leases/<id>              → lease detail
GET  /realestate/lea/leases/new               → create lease form
GET  /realestate/lea/leases/<id>/edit         → edit lease form
GET  /realestate/lea/escalations              → escalation scheduler
GET  /realestate/lea/options                  → option tracker
GET  /realestate/lea/rent-reviews             → rent review workflow
GET  /realestate/lea/modifications            → modifications list
GET  /realestate/lea/subleases                → sublease management
GET  /realestate/lea/expiry                   → expiry pipeline
GET  /realestate/lea/ifrs16                   → IFRS 16 compliance console
GET  /realestate/lea/abstraction              → abstraction workbench
GET  /realestate/lea/assignments              → assignment console
GET  /realestate/lea/reports                  → report builder
GET  /realestate/lea/reports/ifrs16           → IFRS 16 disclosure report
GET  /realestate/lea/reports/portfolio        → portfolio analytics
GET  /realestate/lea/reports/maturity         → maturity profile
GET  /realestate/lea/reports/walt             → WALT report
GET  /realestate/lea/settings                 → settings
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any

from flask import (
	Blueprint, redirect, url_for, request,
	render_template_string, jsonify,
)

from .service import LeaseManagementService
from .models import (
	LeaseCreate, LeaseUpdate,
	LeaseAbstractionCreate,
	RentEscalationCreate,
	LeaseOptionCreate,
	RentReviewCreate,
	Ifrs16ScheduleCreate,
	LeaseAssignmentCreate,
	LeaseModificationCreate,
	SubleaseCreate,
	LeaseExpiryCreate,
	Ifrs16Category,
	LeaseModificationRequest,
)

bp = Blueprint(
	"lea_views", __name__,
	url_prefix="/realestate/lea",
	template_folder="templates",
	static_folder="static",
)

_svc = LeaseManagementService()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro) -> Any:
	try:
		loop = asyncio.get_event_loop()
		if loop.is_closed():
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
		return loop.run_until_complete(coro)
	except RuntimeError:
		return asyncio.run(coro)


def _tenant() -> str:
	return request.headers.get("X-Tenant-ID", request.cookies.get("tenant_id", "default"))


def _actor() -> str:
	return request.headers.get("X-Actor-ID", request.cookies.get("actor_id", "system"))


def _ok(data: Any, status: int = 200):
	return jsonify({"status": "ok", "data": data}), status


def _err(msg: str, status: int = 400):
	return jsonify({"status": "error", "message": msg}), status


# ---------------------------------------------------------------------------
# Minimal inline template base (production: use real Jinja2 templates)
# ---------------------------------------------------------------------------

_BASE_TMPL = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{ title }} — APG Lease Management</title>
<style>
  :root{--c-primary:#4338CA;--c-accent:#0D9488;--c-success:#166534;
    --c-warn:#A16207;--c-danger:#B91C1C;--canvas:#F5F3FF;--panel:#fff;
    --text:#1E1B4B;--text2:#4B5563;--radius:8px;}
  *{box-sizing:border-box;margin:0;padding:0;}
  body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
    background:var(--canvas);color:var(--text);min-height:100vh;}
  nav{background:var(--c-primary);color:#fff;padding:0 1.5rem;display:flex;
    align-items:center;gap:1.5rem;height:52px;position:sticky;top:0;z-index:100;}
  nav a{color:#ffffffcc;text-decoration:none;font-size:.875rem;padding:.25rem .5rem;
    border-radius:4px;}
  nav a:hover,nav a.active{color:#fff;background:#ffffff20;}
  nav .brand{font-weight:700;font-size:1rem;color:#fff;letter-spacing:-.3px;}
  main{padding:1.5rem 2rem;max-width:1400px;margin:0 auto;}
  h1{font-size:1.5rem;font-weight:700;margin-bottom:1rem;color:var(--c-primary);}
  h2{font-size:1.125rem;font-weight:600;margin:.75rem 0 .5rem;}
  .card{background:var(--panel);border-radius:var(--radius);padding:1.25rem;
    box-shadow:0 1px 3px #0000001a;margin-bottom:1rem;}
  .kpi-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:1rem;
    margin-bottom:1.5rem;}
  .kpi{background:var(--panel);border-radius:var(--radius);padding:1rem;
    box-shadow:0 1px 3px #0000001a;border-top:3px solid var(--c-primary);}
  .kpi .label{font-size:.75rem;color:var(--text2);text-transform:uppercase;
    letter-spacing:.05em;margin-bottom:.25rem;}
  .kpi .value{font-size:1.75rem;font-weight:700;color:var(--text);}
  .kpi .sub{font-size:.8rem;color:var(--text2);margin-top:.25rem;}
  table{width:100%;border-collapse:collapse;font-size:.875rem;}
  th{text-align:left;padding:.5rem .75rem;background:#f1f0ff;color:var(--text2);
    font-size:.75rem;text-transform:uppercase;letter-spacing:.05em;border-bottom:1px solid #e2e0f5;}
  td{padding:.6rem .75rem;border-bottom:1px solid #f0eff8;vertical-align:middle;}
  tr:hover td{background:#f8f7ff;}
  .badge{display:inline-flex;align-items:center;padding:.15rem .5rem;border-radius:999px;
    font-size:.7rem;font-weight:600;text-transform:uppercase;letter-spacing:.04em;}
  .badge.active{background:#dcfce7;color:var(--c-success);}
  .badge.expired,.badge.terminated,.badge.forfeited{background:#fee2e2;color:var(--c-danger);}
  .badge.draft,.badge.heads_of_terms,.badge.negotiating{background:#fef9c3;color:var(--c-warn);}
  .badge.critical{background:#fee2e2;color:var(--c-danger);}
  .badge.high{background:#ffedd5;color:#9a3412;}
  .badge.medium{background:#fef9c3;color:var(--c-warn);}
  .badge.low{background:#dcfce7;color:var(--c-success);}
  .btn{display:inline-flex;align-items:center;gap:.375rem;padding:.4rem .875rem;
    border-radius:6px;border:none;cursor:pointer;font-size:.875rem;font-weight:500;
    text-decoration:none;}
  .btn-primary{background:var(--c-primary);color:#fff;}
  .btn-secondary{background:#e0e7ff;color:var(--c-primary);}
  .btn-danger{background:#fee2e2;color:var(--c-danger);}
  .btn:hover{filter:brightness(.92);}
  .actions{display:flex;gap:.5rem;align-items:center;}
  .section-header{display:flex;justify-content:space-between;align-items:center;
    margin-bottom:.75rem;}
  .empty-state{text-align:center;padding:3rem;color:var(--text2);}
  .tag-finance{background:#dbeafe;color:#1e40af;}
  .tag-operating{background:#d1fae5;color:#065f46;}
  .progress-bar{background:#e0e7ff;border-radius:999px;height:8px;overflow:hidden;}
  .progress-bar .fill{height:100%;background:var(--c-primary);border-radius:999px;}
  form{display:grid;gap:1rem;}
  .form-row{display:grid;grid-template-columns:1fr 1fr;gap:1rem;}
  label{font-size:.8rem;color:var(--text2);font-weight:500;display:block;
    margin-bottom:.25rem;}
  input,select,textarea{width:100%;padding:.45rem .75rem;border:1px solid #d1d5db;
    border-radius:6px;font-size:.875rem;background:#fff;}
  input:focus,select:focus,textarea:focus{outline:none;border-color:var(--c-primary);
    box-shadow:0 0 0 2px #4338ca22;}
  .alert{padding:.75rem 1rem;border-radius:6px;font-size:.875rem;margin-bottom:1rem;}
  .alert-warn{background:#fef9c3;color:var(--c-warn);border:1px solid #fde68a;}
  .alert-error{background:#fee2e2;color:var(--c-danger);border:1px solid #fca5a5;}
  @media(max-width:768px){main{padding:1rem;}.kpi-grid{grid-template-columns:1fr 1fr;}
    .form-row{grid-template-columns:1fr;}}
</style>
</head>
<body>
<nav>
  <span class="brand">APG · Lease Management</span>
  <a href="/realestate/lea/dashboard">Dashboard</a>
  <a href="/realestate/lea/leases">Leases</a>
  <a href="/realestate/lea/expiry">Expiry Pipeline</a>
  <a href="/realestate/lea/escalations">Escalations</a>
  <a href="/realestate/lea/options">Options</a>
  <a href="/realestate/lea/rent-reviews">Rent Reviews</a>
  <a href="/realestate/lea/modifications">Modifications</a>
  <a href="/realestate/lea/subleases">Subleases</a>
  <a href="/realestate/lea/ifrs16">IFRS 16</a>
  <a href="/realestate/lea/abstraction">Abstraction</a>
  <a href="/realestate/lea/assignments">Assignments</a>
  <a href="/realestate/lea/reports">Reports</a>
  <a href="/realestate/lea/settings">Settings</a>
</nav>
<main>{{ content }}</main>
</body></html>"""


def _page(title: str, content: str) -> str:
	return render_template_string(_BASE_TMPL, title=title, content=content)


def _fmt_money(v, currency="KES") -> str:
	if v is None:
		return "—"
	try:
		return f"{currency} {float(v):,.2f}"
	except Exception:
		return str(v)


def _badge(status: str) -> str:
	return f'<span class="badge {status}">{status.replace("_"," ")}</span>'


def _urgency_badge(days: int) -> str:
	if days <= 30:
		label, cls = "Critical", "critical"
	elif days <= 90:
		label, cls = "High", "high"
	elif days <= 180:
		label, cls = "Medium", "medium"
	else:
		label, cls = "Low", "low"
	return f'<span class="badge {cls}">{label} ({days}d)</span>'


# ===========================================================================
# ROOT REDIRECT
# ===========================================================================

@bp.get("/")
def index():
	return redirect(url_for("lea_views.dashboard"))


# ===========================================================================
# DASHBOARD
# ===========================================================================

@bp.get("/dashboard")
def dashboard():
	tenant = _tenant()
	pipeline = _run(_svc.get_expiry_pipeline(tenant, months_ahead=6))
	options = _run(_svc.get_expiring_options(tenant, days_ahead=90))
	summary = _run(_svc.lease_portfolio_summary({"tenant_id": tenant}))

	total = summary.get("total_leases", 0)
	total_ar = summary.get("total_annual_rent", 0)
	total_rou = summary.get("total_rou_assets", 0)
	total_ll = summary.get("total_lease_liabilities", 0)
	exp_90 = summary.get("expiring_within_90_days", 0)

	kpis = f"""
<div class="kpi-grid">
  <div class="kpi"><div class="label">Total Leases</div>
    <div class="value">{total}</div></div>
  <div class="kpi" style="border-color:var(--c-accent)"><div class="label">Annual Rent Roll</div>
    <div class="value" style="font-size:1.25rem">{_fmt_money(total_ar)}</div></div>
  <div class="kpi" style="border-color:var(--c-warn)"><div class="label">Expiring ≤90 days</div>
    <div class="value" style="color:var(--c-warn)">{exp_90}</div></div>
  <div class="kpi"><div class="label">ROU Assets</div>
    <div class="value" style="font-size:1.2rem">{_fmt_money(total_rou)}</div></div>
  <div class="kpi"><div class="label">Lease Liabilities</div>
    <div class="value" style="font-size:1.2rem">{_fmt_money(total_ll)}</div></div>
</div>
"""
	# Expiry pipeline table
	if pipeline:
		rows = "".join(
			f"<tr><td><a href='/realestate/lea/leases/{r['lease_id']}'>{r['lease_id'][:8]}…</a></td>"
			f"<td>{r.get('property_id','—')}</td>"
			f"<td>{r.get('end_date','—')}</td>"
			f"<td>{_urgency_badge(r.get('days_remaining',0))}</td>"
			f"<td>{_fmt_money(r.get('current_rent'), r.get('currency','KES'))}/mo</td></tr>"
			for r in pipeline[:10]
		)
		pipe_html = f"""
<div class="card">
  <div class="section-header"><h2>Expiry Pipeline (next 6 months)</h2>
    <a href="/realestate/lea/expiry" class="btn btn-secondary">View all</a></div>
  <table><thead><tr><th>Lease ID</th><th>Property</th>
    <th>Expiry</th><th>Urgency</th><th>Rent/mo</th></tr></thead>
  <tbody>{rows}</tbody></table>
</div>"""
	else:
		pipe_html = '<div class="card"><div class="empty-state">No leases expiring in the next 6 months.</div></div>'

	# Expiring options
	opt_rows = ""
	for o in options:
		od = o.model_dump()
		opt_rows += (
			f"<tr><td>{od['option_type'].replace('_',' ')}</td>"
			f"<td><a href='/realestate/lea/leases/{od['lease_id']}'>{od['lease_id'][:8]}…</a></td>"
			f"<td>{od.get('exercise_to','—')}</td>"
			f"<td>{_badge(od.get('status','open'))}</td></tr>"
		)
	opts_html = f"""
<div class="card">
  <div class="section-header"><h2>Options Expiring Soon (90 days)</h2>
    <a href="/realestate/lea/options" class="btn btn-secondary">View all</a></div>
  {'<table><thead><tr><th>Type</th><th>Lease</th><th>Deadline</th><th>Status</th></tr></thead><tbody>' + opt_rows + '</tbody></table>' if opt_rows else '<div class="empty-state">No options expiring soon.</div>'}
</div>"""

	content = f"<h1>Lease Management Dashboard</h1>{kpis}{pipe_html}{opts_html}"
	return _page("Dashboard", content)


# ===========================================================================
# LEASES — list
# ===========================================================================

@bp.get("/leases")
def list_leases():
	tenant = _tenant()
	leases = _run(_svc.list_leases(
		tenant,
		property_id=request.args.get("property_id"),
		status=request.args.get("status"),
	))

	if leases:
		rows = ""
		for l in leases:
			d = l.model_dump()
			rows += (
				f"<tr>"
				f"<td><a href='/realestate/lea/leases/{d['id']}'>"
				f"{d.get('lease_ref', d['id'][:8]+'…')}</a></td>"
				f"<td>{d.get('lease_type','—').replace('_',' ')}</td>"
				f"<td>{d.get('property_id','—')}</td>"
				f"<td>{d.get('commencement_date','—')}</td>"
				f"<td>{d.get('expiry_date','—')}</td>"
				f"<td>{_fmt_money(d.get('current_rent', d.get('initial_rent',0)), d.get('currency','KES'))}</td>"
				f"<td>{_badge(d.get('status','draft'))}</td>"
				f"<td>{d.get('ifrs16_category','—') or '—'}</td>"
				f"<td class='actions'>"
				f"<a href='/realestate/lea/leases/{d['id']}' class='btn btn-secondary'>View</a>"
				f"<a href='/realestate/lea/leases/{d['id']}/edit' class='btn btn-secondary'>Edit</a>"
				f"</td></tr>"
			)
		table = f"""
<table>
  <thead><tr><th>Ref</th><th>Type</th><th>Property</th>
    <th>Start</th><th>Expiry</th><th>Rent/mo</th>
    <th>Status</th><th>IFRS16</th><th>Actions</th></tr></thead>
  <tbody>{rows}</tbody>
</table>"""
	else:
		table = '<div class="empty-state">No leases found. <a href="/realestate/lea/leases/new">Create one.</a></div>'

	# Filter bar
	status_opts = "".join(
		f'<option value="{s}" {"selected" if request.args.get("status")==s else ""}>{s}</option>'
		for s in ["", "draft", "active", "expired", "surrendered", "terminated"]
	)
	filters = f"""
<form method="get" style="display:flex;gap:.75rem;align-items:flex-end;margin-bottom:1rem;flex-wrap:wrap;">
  <div><label>Status</label><select name="status">{status_opts}</select></div>
  <div><label>Property ID</label><input name="property_id" value="{request.args.get('property_id','')}"></div>
  <button type="submit" class="btn btn-primary">Filter</button>
</form>"""

	content = f"""
<div class="section-header">
  <h1>Lease Registry</h1>
  <a href="/realestate/lea/leases/new" class="btn btn-primary">+ New Lease</a>
</div>
{filters}
<div class="card">{table}</div>"""
	return _page("Lease Registry", content)


# ===========================================================================
# LEASES — detail
# ===========================================================================

@bp.get("/leases/<lease_id>")
def detail_lease(lease_id: str):
	tenant = _tenant()
	lease = _run(_svc.get_lease(lease_id, tenant))
	if lease is None:
		return _page("Not Found", '<div class="alert alert-error">Lease not found.</div>'), 404

	d = lease.model_dump()
	currency = d.get("currency", "KES")

	# Fetch related data in parallel-ish
	options = _run(_svc.list_options(tenant, lease_id))
	escalations = _run(_svc.list_escalations(tenant, lease_id))
	modifications = _run(_svc.list_modifications(tenant, lease_id))

	def _row(label, value):
		return f"<tr><td style='color:var(--text2);width:200px'>{label}</td><td><strong>{value}</strong></td></tr>"

	ifrs16_section = ""
	if d.get("rou_asset") or d.get("lease_liability"):
		ifrs16_section = f"""
<div class="card">
  <h2>IFRS 16 / ASC 842</h2>
  <table>
    {_row("Category", _badge(d.get("ifrs16_category","—") or "—"))}
    {_row("ROU Asset", _fmt_money(d.get("rou_asset"), currency))}
    {_row("Lease Liability", _fmt_money(d.get("lease_liability"), currency))}
    {_row("IBR", f"{float(d.get('incremental_borrowing_rate') or 0)*100:.2f}% p.a." if d.get('incremental_borrowing_rate') else '—')}
  </table>
  <div class="actions" style="margin-top:.75rem">
    <button class="btn btn-secondary" onclick="postAction('/realestate/lea/leases/{lease_id}/classify-ifrs16')">Classify</button>
    <button class="btn btn-secondary" onclick="postAction('/realestate/lea/leases/{lease_id}/calculate-rou')">Calc ROU</button>
    <button class="btn btn-secondary" onclick="postAction('/realestate/lea/leases/{lease_id}/calculate-liability')">Calc Liability</button>
  </div>
</div>"""

	options_html = ""
	if options:
		opt_rows = "".join(
			f"<tr><td>{o.model_dump()['option_type'].replace('_',' ')}</td>"
			f"<td>{o.model_dump().get('exercise_from','—')} – {o.model_dump().get('exercise_to','—')}</td>"
			f"<td>{_badge(o.model_dump().get('status','open'))}</td>"
			f"<td>{'Yes' if o.model_dump().get('reasonably_certain') else 'No'}</td></tr>"
			for o in options
		)
		options_html = f"""
<div class="card"><h2>Options</h2>
<table><thead><tr><th>Type</th><th>Window</th><th>Status</th><th>Certain?</th></tr></thead>
<tbody>{opt_rows}</tbody></table></div>"""

	lifecycle_buttons = ""
	status = d.get("status", "")
	if status in ("draft", "heads_of_terms", "negotiating", "signed"):
		lifecycle_buttons += f'<button class="btn btn-primary" onclick="postAction(\'/realestate/lea/leases/{lease_id}/activate\')">Activate</button>'
	if status == "active":
		lifecycle_buttons += (
			f'<button class="btn btn-secondary" onclick="postAction(\'/realestate/lea/leases/{lease_id}/surrender\',{{surrender_date:today(),agreed_compensation:0}})">Surrender</button>'
			f'<button class="btn btn-secondary" onclick="postAction(\'/realestate/lea/leases/{lease_id}/terminate\',{{termination_type:\'expiry\',effective_date:today(),notice_date:today()}})">Terminate</button>'
		)

	content = f"""
<div class="section-header">
  <h1>Lease: {d.get("lease_ref", lease_id)}</h1>
  <div class="actions">
    <a href="/realestate/lea/leases/{lease_id}/edit" class="btn btn-secondary">Edit</a>
    {lifecycle_buttons}
  </div>
</div>

<div class="card">
  <h2>Lease Details</h2>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:0">
    <table>
      {_row("Status", _badge(status))}
      {_row("Type", d.get("lease_type","—").replace("_"," "))}
      {_row("Property", d.get("property_id","—"))}
      {_row("Unit", d.get("unit_id","—"))}
      {_row("Tenant Entity", d.get("tenant_entity_id","—"))}
    </table>
    <table>
      {_row("Commencement", str(d.get("commencement_date","—")))}
      {_row("Expiry", str(d.get("expiry_date","—")))}
      {_row("Initial Rent", _fmt_money(d.get("initial_rent",0), currency)+"/mo")}
      {_row("Current Rent", _fmt_money(d.get("current_rent",0), currency)+"/mo")}
      {_row("Security Deposit", _fmt_money(d.get("security_deposit",0), currency))}
    </table>
  </div>
</div>

{ifrs16_section}
{options_html}

<script>
function today(){{return new Date().toISOString().split('T')[0];}}
async function postAction(url, body={{}}){{
  const r=await fetch(url,{{method:'POST',headers:{{'Content-Type':'application/json','X-Tenant-ID':'{tenant}'}},body:JSON.stringify(body)}});
  const j=await r.json();
  alert(j.status==='ok'?'Done: '+JSON.stringify(j.data).slice(0,120):'Error: '+j.message);
  location.reload();
}}
</script>"""
	return _page(f"Lease {d.get('lease_ref', lease_id)}", content)


# ===========================================================================
# LEASES — create form
# ===========================================================================

@bp.get("/leases/new")
def create_lease_form():
	content = """
<h1>Create New Lease</h1>
<div class="card">
<form method="post" action="/realestate/lea/leases/new" id="create-form">
  <div class="form-row">
    <div><label>Lease Reference *</label><input name="lease_ref" required></div>
    <div><label>Lease Type *</label>
      <select name="lease_type">
        <option value="commercial">Commercial</option>
        <option value="retail">Retail</option>
        <option value="office">Office</option>
        <option value="industrial">Industrial</option>
        <option value="residential">Residential</option>
        <option value="ground_lease">Ground Lease</option>
      </select>
    </div>
  </div>
  <div class="form-row">
    <div><label>Property ID *</label><input name="property_id" required></div>
    <div><label>Unit ID *</label><input name="unit_id" required></div>
  </div>
  <div class="form-row">
    <div><label>Tenant Entity ID *</label><input name="tenant_entity_id" required></div>
    <div><label>Currency</label>
      <select name="currency">
        <option value="KES">KES</option><option value="USD">USD</option>
        <option value="EUR">EUR</option><option value="GBP">GBP</option>
      </select>
    </div>
  </div>
  <div class="form-row">
    <div><label>Commencement Date *</label><input type="date" name="commencement_date" required></div>
    <div><label>Expiry Date *</label><input type="date" name="expiry_date" required></div>
  </div>
  <div class="form-row">
    <div><label>Initial Rent (per period) *</label><input type="number" step="0.01" name="initial_rent" required></div>
    <div><label>Rent Frequency</label>
      <select name="rent_frequency">
        <option value="monthly">Monthly</option>
        <option value="quarterly">Quarterly</option>
        <option value="annual">Annual</option>
      </select>
    </div>
  </div>
  <div class="form-row">
    <div><label>Security Deposit</label><input type="number" step="0.01" name="security_deposit" value="0"></div>
    <div><label>Incremental Borrowing Rate (decimal, e.g. 0.085)</label>
      <input type="number" step="0.0001" name="incremental_borrowing_rate"></div>
  </div>
  <div class="form-row">
    <div><label>Initial Direct Costs</label><input type="number" step="0.01" name="initial_direct_costs" value="0"></div>
    <div><label>Lease Incentives Received</label><input type="number" step="0.01" name="lease_incentives" value="0"></div>
  </div>
  <div><label>Notes</label><textarea name="notes" rows="3"></textarea></div>
  <div style="display:flex;gap:.75rem;margin-top:.5rem">
    <button type="submit" class="btn btn-primary">Create Lease</button>
    <a href="/realestate/lea/leases" class="btn btn-secondary">Cancel</a>
  </div>
</form>
</div>
<script>
document.getElementById('create-form').addEventListener('submit',async e=>{
  e.preventDefault();
  const data=Object.fromEntries(new FormData(e.target));
  data.tenant_id=document.cookie.match(/tenant_id=([^;]+)/)?.[1]||'default';
  data.created_by='ui_user';
  const r=await fetch('/realestate/lea/leases',{method:'POST',
    headers:{'Content-Type':'application/json','X-Tenant-ID':data.tenant_id},
    body:JSON.stringify(data)});
  const j=await r.json();
  if(j.status==='ok'){window.location='/realestate/lea/leases/'+j.data.id;}
  else{alert('Error: '+j.message);}
});
</script>"""
	return _page("Create Lease", content)


@bp.post("/leases/new")
def create_lease_post():
	"""Handle form POST — delegate to API and redirect."""
	from flask import redirect
	data = {k: v for k, v in request.form.items() if v}
	data["tenant_id"] = _tenant()
	data["created_by"] = _actor()
	try:
		payload = LeaseCreate(**data)
		lease = _run(_svc.create_lease_v2(payload))
		lease_id = lease.model_dump()["id"] if hasattr(lease, "model_dump") else lease.get("id", "")
		return redirect(url_for("lea_views.detail_lease", lease_id=lease_id))
	except Exception as e:
		content = f'<div class="alert alert-error">{e}</div>'
		return _page("Error", content), 400


# ===========================================================================
# LEASES — edit form
# ===========================================================================

@bp.get("/leases/<lease_id>/edit")
def edit_lease(lease_id: str):
	lease = _run(_svc.get_lease(lease_id, _tenant()))
	if lease is None:
		return _page("Not Found", '<div class="alert alert-error">Lease not found.</div>'), 404
	d = lease.model_dump()

	def _field(label, name, type_="text", step=None, value=None):
		v = value if value is not None else (d.get(name) or "")
		step_attr = f' step="{step}"' if step else ""
		return f'<div><label>{label}</label><input type="{type_}" name="{name}" value="{v}"{step_attr}></div>'

	content = f"""
<h1>Edit Lease: {d.get("lease_ref", lease_id)}</h1>
<div class="card">
<form id="edit-form">
  <div class="form-row">
    {_field("Current Rent", "current_rent", "number", "0.01")}
    {_field("Expiry Date", "expiry_date", "date")}
  </div>
  <div class="form-row">
    <div><label>IFRS16 Category</label>
      <select name="ifrs16_category">
        <option value="">—</option>
        <option value="finance_lease" {"selected" if d.get("ifrs16_category")=="finance_lease" else ""}>Finance Lease</option>
        <option value="operating_lease" {"selected" if d.get("ifrs16_category")=="operating_lease" else ""}>Operating Lease</option>
        <option value="short_term_exemption" {"selected" if d.get("ifrs16_category")=="short_term_exemption" else ""}>Short-term Exemption</option>
        <option value="low_value_exemption" {"selected" if d.get("ifrs16_category")=="low_value_exemption" else ""}>Low-value Exemption</option>
      </select>
    </div>
    {_field("IBR (decimal)", "incremental_borrowing_rate", "number", "0.0001")}
  </div>
  <div><label>Notes</label><textarea name="notes" rows="3">{d.get("notes","")}</textarea></div>
  <div style="display:flex;gap:.75rem;margin-top:.5rem">
    <button type="submit" class="btn btn-primary">Save Changes</button>
    <a href="/realestate/lea/leases/{lease_id}" class="btn btn-secondary">Cancel</a>
  </div>
</form>
</div>
<script>
document.getElementById('edit-form').addEventListener('submit',async e=>{{
  e.preventDefault();
  const data=Object.fromEntries(new FormData(e.target));
  const r=await fetch('/realestate/lea/leases/{lease_id}',{{method:'PUT',
    headers:{{'Content-Type':'application/json','X-Tenant-ID':'{_tenant()}'}},
    body:JSON.stringify(data)}});
  const j=await r.json();
  if(j.status==='ok'){{window.location='/realestate/lea/leases/{lease_id}';}}
  else{{alert('Error: '+j.message);}}
}});
</script>"""
	return _page(f"Edit Lease {d.get('lease_ref',lease_id)}", content)


# ===========================================================================
# EXPIRY PIPELINE
# ===========================================================================

@bp.get("/expiry")
def expiry_pipeline():
	days = int(request.args.get("days", 180))
	pipeline = _run(_svc.lease_expiry_pipeline(days_ahead=days))

	if pipeline:
		rows = "".join(
			f"<tr><td><a href='/realestate/lea/leases/{r['lease_id']}'>{r['lease_id'][:8]}…</a></td>"
			f"<td>{r.get('property_id','—')}</td>"
			f"<td>{r.get('end_date','—')}</td>"
			f"<td>{_urgency_badge(r.get('days_remaining',0))}</td>"
			f"<td>{_fmt_money(r.get('current_rent'), r.get('currency','KES'))}</td>"
			f"<td>{'✓' if r.get('has_renewal_option') else '—'}</td>"
			f"<td>{'✓' if r.get('has_break_option') else '—'}</td></tr>"
			for r in pipeline
		)
		table = f"""
<table><thead><tr><th>Lease</th><th>Property</th><th>Expiry</th>
<th>Urgency</th><th>Rent/mo</th><th>Renewal?</th><th>Break?</th></tr></thead>
<tbody>{rows}</tbody></table>"""
	else:
		table = f'<div class="empty-state">No leases expiring within {days} days.</div>'

	# Filter
	days_filter = f"""
<form method="get" style="display:flex;gap:.75rem;margin-bottom:1rem">
  <div><label>Days ahead</label>
    <select name="days" onchange="this.form.submit()">
      {"".join(f'<option value="{d}" {"selected" if days==d else ""}>{d} days</option>' for d in [60,90,180,365])}
    </select>
  </div>
</form>"""

	content = f"<h1>Lease Expiry Pipeline</h1>{days_filter}<div class='card'>{table}</div>"
	return _page("Expiry Pipeline", content)


# ===========================================================================
# ESCALATIONS
# ===========================================================================

@bp.get("/escalations")
def list_escalations():
	tenant = _tenant()
	escalations = _run(_svc.list_escalations(tenant, request.args.get("lease_id")))
	if escalations:
		rows = "".join(
			f"<tr><td>{e.model_dump()['escalation_type'].replace('_',' ')}</td>"
			f"<td><a href='/realestate/lea/leases/{e.model_dump()['lease_id']}'>{e.model_dump()['lease_id'][:8]}…</a></td>"
			f"<td>{e.model_dump().get('effective_date','—')}</td>"
			f"<td>{_fmt_money(e.model_dump().get('old_rent',0))}</td>"
			f"<td>{_fmt_money(e.model_dump().get('computed_new_rent') or e.model_dump().get('new_rent',0))}</td>"
			f"<td>{'✓' if e.model_dump().get('applied') else '—'}</td></tr>"
			for e in escalations
		)
		table = f"""
<table><thead><tr><th>Type</th><th>Lease</th><th>Effective</th>
<th>Old Rent</th><th>New Rent</th><th>Applied</th></tr></thead>
<tbody>{rows}</tbody></table>"""
	else:
		table = '<div class="empty-state">No escalations found.</div>'

	content = f"""
<div class="section-header">
  <h1>Rent Escalation Scheduler</h1>
</div>
<div class="card">{table}</div>"""
	return _page("Escalations", content)


# ===========================================================================
# OPTIONS
# ===========================================================================

@bp.get("/options")
def list_options():
	tenant = _tenant()
	opts = _run(_svc.list_options(tenant, request.args.get("lease_id")))
	if opts:
		rows = "".join(
			f"<tr><td>{o.model_dump()['option_type'].replace('_',' ')}</td>"
			f"<td><a href='/realestate/lea/leases/{o.model_dump()['lease_id']}'>{o.model_dump()['lease_id'][:8]}…</a></td>"
			f"<td>{o.model_dump().get('exercise_from','—')} – {o.model_dump().get('exercise_to','—')}</td>"
			f"<td>{_badge(o.model_dump().get('status','open'))}</td>"
			f"<td>{'Yes' if o.model_dump().get('reasonably_certain') else 'No'}</td></tr>"
			for o in opts
		)
		table = f"""
<table><thead><tr><th>Type</th><th>Lease</th><th>Exercise Window</th>
<th>Status</th><th>Reasonably Certain?</th></tr></thead>
<tbody>{rows}</tbody></table>"""
	else:
		table = '<div class="empty-state">No options found.</div>'

	expiring = _run(_svc.get_expiring_options(tenant, 180))
	warning = ""
	if expiring:
		warning = f'<div class="alert alert-warn">{len(expiring)} option(s) expiring within 180 days — review required.</div>'

	content = f"""
<div class="section-header"><h1>Lease Option Tracker</h1></div>
{warning}
<div class="card">{table}</div>"""
	return _page("Options", content)


# ===========================================================================
# MODIFICATIONS
# ===========================================================================

@bp.get("/modifications")
def list_modifications():
	tenant = _tenant()
	mods = _run(_svc.list_modifications(tenant, request.args.get("lease_id")))
	if mods:
		def _mod_row(m):
			d = m.model_dump()
			lid = d.get('lease_id', '')
			applied = '✓' if d.get('applied') else '—'
			return (
				f"<tr><td>{d.get('trigger','—').replace('_',' ')}</td>"
				f"<td><a href='/realestate/lea/leases/{lid}'>{lid[:8]}…</a></td>"
				f"<td>{d.get('modification_date','—')}</td>"
				f"<td>{_badge(d.get('status','pending'))}</td>"
				f"<td>{d.get('reason','—')}</td>"
				f"<td>{applied}</td></tr>"
			)
		rows = "".join(_mod_row(m) for m in mods)
		table = f"""
<table><thead><tr><th>Trigger</th><th>Lease</th><th>Date</th>
<th>Status</th><th>Reason</th><th>Applied</th></tr></thead>
<tbody>{rows}</tbody></table>"""
	else:
		table = '<div class="empty-state">No modifications found.</div>'

	content = f"""
<div class="section-header"><h1>Lease Modifications</h1></div>
<div class="card">{table}</div>"""
	return _page("Modifications", content)


# ===========================================================================
# RENT REVIEWS
# ===========================================================================

@bp.get("/rent-reviews")
def list_rent_reviews():
	tenant = _tenant()
	reviews = _run(_svc.list_rent_reviews(tenant, request.args.get("lease_id")))
	if reviews:
		def _rr_row(r):
			d = r.model_dump()
			lid = d.get('lease_id', '')
			return (
				f"<tr><td>{d.get('review_type','—').replace('_',' ')}</td>"
				f"<td><a href='/realestate/lea/leases/{lid}'>{lid[:8]}…</a></td>"
				f"<td>{d.get('review_date','—')}</td>"
				f"<td>{_badge(d.get('status','pending'))}</td>"
				f"<td>{_fmt_money(d.get('proposed_rent'))}</td>"
				f"<td>{_fmt_money(d.get('agreed_rent'))}</td></tr>"
			)
		rows = "".join(_rr_row(r) for r in reviews)
		table = f"""
<table><thead><tr><th>Type</th><th>Lease</th><th>Review Date</th>
<th>Status</th><th>Proposed</th><th>Agreed</th></tr></thead>
<tbody>{rows}</tbody></table>"""
	else:
		table = '<div class="empty-state">No rent reviews found.</div>'

	content = f"""
<div class="section-header"><h1>Rent Review Workflow</h1></div>
<div class="card">{table}</div>"""
	return _page("Rent Reviews", content)


# ===========================================================================
# SUBLEASES
# ===========================================================================

@bp.get("/subleases")
def list_subleases():
	tenant = _tenant()
	subleases = _run(_svc.list_subleases(tenant, request.args.get("head_lease_id")))
	if subleases:
		def _sl_row(s):
			d = s.model_dump()
			hlid = d.get('head_lease_id', '')
			return (
				f"<tr><td><a href='/realestate/lea/leases/{hlid}'>{hlid[:8]}…</a></td>"
				f"<td>{d.get('sublessee_entity_id','—')}</td>"
				f"<td>{d.get('commencement_date','—')} – {d.get('end_date','—')}</td>"
				f"<td>{_fmt_money(d.get('payment_amount',0))}/mo</td>"
				f"<td>{d.get('sublease_classification','—')}</td>"
				f"<td>{_badge(d.get('status','active'))}</td></tr>"
			)
		rows = "".join(_sl_row(s) for s in subleases)
		table = f"""
<table><thead><tr><th>Head Lease</th><th>Sublessee</th><th>Term</th>
<th>Rent/mo</th><th>Classification</th><th>Status</th></tr></thead>
<tbody>{rows}</tbody></table>"""
	else:
		table = '<div class="empty-state">No subleases found.</div>'

	content = f"""
<div class="section-header"><h1>Sublease Management</h1></div>
<div class="card">{table}</div>"""
	return _page("Subleases", content)


# ===========================================================================
# IFRS 16 CONSOLE
# ===========================================================================

@bp.get("/ifrs16")
def ifrs16_console():
	import datetime
	fiscal_year = request.args.get("fiscal_year", str(datetime.date.today().year))
	try:
		disc = _run(_svc.ifrs16_disclosure_notes(fiscal_year))
	except Exception as e:
		disc = {}

	def _kpi(label, value, color="var(--c-primary)"):
		return f'<div class="kpi" style="border-color:{color}"><div class="label">{label}</div><div class="value" style="font-size:1.1rem">{value}</div></div>'

	kpis = f"""
<div class="kpi-grid">
  {_kpi("Total ROU Assets", _fmt_money(disc.get("rou_asset_carrying_amount",0)))}
  {_kpi("Total Lease Liabilities", _fmt_money(disc.get("total_lease_liability",0)), "var(--c-warn)")}
  {_kpi("Depreciation Charge", _fmt_money(disc.get("rou_asset_depreciation_charge",0)))}
  {_kpi("Interest Expense", _fmt_money(disc.get("interest_expense_on_lease_liabilities",0)), "var(--c-accent)")}
  {_kpi("Cash Outflow", _fmt_money(disc.get("total_cash_outflow_for_leases",0)))}
  {_kpi("Active Leases", str(disc.get("active_lease_count",0)))}
  {_kpi("Avg IBR", f"{float(disc.get('weighted_average_lessee_incremental_borrowing_rate',0))*100:.2f}%")}
  {_kpi("WALT", f"{disc.get('weighted_average_lease_term_years',0)} yrs")}
</div>"""

	mat = disc.get("maturity_analysis_undiscounted", {})
	mat_html = ""
	if mat:
		mat_html = f"""
<div class="card"><h2>Maturity Analysis (Undiscounted)</h2>
<table><thead><tr><th>Band</th><th>Total Payments</th></tr></thead><tbody>
  <tr><td>Within 1 year</td><td>{_fmt_money(mat.get("within_1_year",0))}</td></tr>
  <tr><td>1 – 5 years</td><td>{_fmt_money(mat.get("1_to_5_years",0))}</td></tr>
  <tr><td>Beyond 5 years</td><td>{_fmt_money(mat.get("over_5_years",0))}</td></tr>
</tbody></table></div>"""

	yr_opts = "".join(
		f'<option value="{y}" {"selected" if fiscal_year==str(y) else ""}>{y}</option>'
		for y in range(2020, 2030)
	)
	fy_form = f"""
<form method="get" style="display:flex;gap:.75rem;margin-bottom:1rem">
  <div><label>Fiscal Year</label>
    <select name="fiscal_year" onchange="this.form.submit()">{yr_opts}</select></div>
</form>"""

	content = f"<h1>IFRS 16 Compliance Console — FY {fiscal_year}</h1>{fy_form}{kpis}{mat_html}"
	return _page("IFRS 16", content)


# ===========================================================================
# ABSTRACTION
# ===========================================================================

@bp.get("/abstraction")
def abstraction_workbench():
	tenant = _tenant()
	abstractions = _run(_svc.list_abstractions(tenant, None))
	if abstractions:
		def _abs_row(a):
			d = a.model_dump()
			lid = d.get('lease_id', '')
			return (
				f"<tr><td><a href='/realestate/lea/leases/{lid}'>{lid[:8]}…</a></td>"
				f"<td>{d.get('source_document_id','—')}</td>"
				f"<td>{d.get('abstracted_by','—')}</td>"
				f"<td>{_badge(d.get('status','pending'))}</td>"
				f"<td>{d.get('verified_by','—')}</td></tr>"
			)
		rows = "".join(_abs_row(a) for a in abstractions)
		table = f"""
<table><thead><tr><th>Lease</th><th>Document</th><th>Abstracted By</th>
<th>Status</th><th>Verified By</th></tr></thead>
<tbody>{rows}</tbody></table>"""
	else:
		table = '<div class="empty-state">No abstractions found. Upload a lease document to begin.</div>'

	content = f"""
<div class="section-header"><h1>Lease Abstraction Workbench</h1></div>
<div class="card">{table}</div>"""
	return _page("Abstraction", content)


# ===========================================================================
# ASSIGNMENTS
# ===========================================================================

@bp.get("/assignments")
def list_assignments():
	tenant = _tenant()
	assignments = _run(_svc.list_assignments(tenant, request.args.get("lease_id")))
	if assignments:
		def _asgn_row(a):
			d = a.model_dump()
			lid = d.get('lease_id', '')
			return (
				f"<tr><td>{d.get('assignment_type','—').replace('_',' ')}</td>"
				f"<td><a href='/realestate/lea/leases/{lid}'>{lid[:8]}…</a></td>"
				f"<td>{d.get('assignee_id','—')}</td>"
				f"<td>{d.get('effective_date','—')}</td>"
				f"<td>{_badge(d.get('status','pending'))}</td></tr>"
			)
		rows = "".join(_asgn_row(a) for a in assignments)
		table = f"""
<table><thead><tr><th>Type</th><th>Lease</th><th>Assignee</th>
<th>Effective</th><th>Status</th></tr></thead>
<tbody>{rows}</tbody></table>"""
	else:
		table = '<div class="empty-state">No assignments found.</div>'

	content = f"""
<div class="section-header"><h1>Lease Assignment Console</h1></div>
<div class="card">{table}</div>"""
	return _page("Assignments", content)


# ===========================================================================
# REPORTS
# ===========================================================================

@bp.get("/reports")
def reports_index():
	content = """
<h1>Lease Report Builder</h1>
<div class="kpi-grid" style="margin-top:1rem">
  <a href="/realestate/lea/reports/portfolio" class="card" style="text-decoration:none;display:block">
    <div class="label">Portfolio Analytics</div>
    <div style="margin-top:.5rem;color:var(--c-primary);font-weight:600">→ View</div>
  </a>
  <a href="/realestate/lea/reports/ifrs16" class="card" style="text-decoration:none;display:block">
    <div class="label">IFRS 16 Disclosure Notes</div>
    <div style="margin-top:.5rem;color:var(--c-primary);font-weight:600">→ View</div>
  </a>
  <a href="/realestate/lea/reports/maturity" class="card" style="text-decoration:none;display:block">
    <div class="label">Maturity Profile</div>
    <div style="margin-top:.5rem;color:var(--c-primary);font-weight:600">→ View</div>
  </a>
  <a href="/realestate/lea/reports/walt" class="card" style="text-decoration:none;display:block">
    <div class="label">WALT Analysis</div>
    <div style="margin-top:.5rem;color:var(--c-primary);font-weight:600">→ View</div>
  </a>
</div>"""
	return _page("Reports", content)


@bp.get("/reports/portfolio")
def report_portfolio():
	tenant = _tenant()
	data = _run(_svc.portfolio_lease_analytics(tenant))
	content = f"<h1>Portfolio Analytics</h1><div class='card'><pre style='font-size:.75rem;overflow:auto'>{__import__('json').dumps(data, indent=2, default=str)}</pre></div>"
	return _page("Portfolio Analytics", content)


@bp.get("/reports/ifrs16")
def report_ifrs16():
	import datetime
	fy = request.args.get("fiscal_year", str(datetime.date.today().year))
	data = _run(_svc.ifrs16_disclosure_notes(fy))
	content = f"<h1>IFRS 16 Disclosure Notes — FY {fy}</h1><div class='card'><pre style='font-size:.75rem;overflow:auto'>{__import__('json').dumps(data, indent=2, default=str)}</pre></div>"
	return _page("IFRS 16 Disclosure", content)


@bp.get("/reports/maturity")
def report_maturity():
	years = int(request.args.get("years", 5))
	data = _run(_svc.lease_maturity_profile(years))
	content = f"<h1>Lease Maturity Profile ({years} years)</h1><div class='card'><pre style='font-size:.75rem;overflow:auto'>{__import__('json').dumps(data, indent=2, default=str)}</pre></div>"
	return _page("Maturity Profile", content)


@bp.get("/reports/walt")
def report_walt():
	tenant = _tenant()
	walt = _run(_svc.weighted_average_lease_term({"tenant_id": tenant}))
	content = f"""
<h1>Weighted Average Lease Term</h1>
<div class="card">
  <div class="kpi-grid">
    <div class="kpi"><div class="label">WALT</div>
      <div class="value">{walt} <span style="font-size:1rem;color:var(--text2)">years</span></div>
    </div>
  </div>
  <p style="color:var(--text2);font-size:.85rem;margin-top:.5rem">
    Weighted by annual rent. Only active leases with future expiry dates are included.
  </p>
</div>"""
	return _page("WALT", content)


# ===========================================================================
# SETTINGS
# ===========================================================================

@bp.get("/settings")
def settings():
	from .capability_contract import get_capability_contract
	contract = get_capability_contract(_tenant())
	content = f"""
<h1>Lease Management Settings</h1>
<div class="card">
  <h2>Capability Contract</h2>
  <pre style="font-size:.75rem;overflow:auto">{__import__('json').dumps(contract, indent=2, default=str)}</pre>
</div>"""
	return _page("Settings", content)
