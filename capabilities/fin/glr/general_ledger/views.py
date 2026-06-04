"""
General Ledger — Flask Blueprint UI view models.

Each function returns a pure dict (screen model) suitable for rendering with
any templating engine (Jinja2, JSON API, etc.).  No HTTP concerns here.

© 2025 Datacraft. All rights reserved.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any

try:
	from .service import GeneralLedgerService
except ImportError:
	from service import GeneralLedgerService  # type: ignore


# ---------------------------------------------------------------------------
# Navigation manifest
# ---------------------------------------------------------------------------

NAVIGATION: list[dict[str, str]] = [
	{"name": "Dashboard",          "route": "/glr/dashboard",             "icon": "layout-dashboard"},
	{"name": "Chart of Accounts",  "route": "/glr/accounts",              "icon": "book-open"},
	{"name": "Periods",            "route": "/glr/periods",               "icon": "calendar-days"},
	{"name": "Journal Entries",    "route": "/glr/journals",              "icon": "receipt-text"},
	{"name": "Postings",           "route": "/glr/postings",              "icon": "send"},
	{"name": "Reconciliations",    "route": "/glr/reconciliations",       "icon": "check-square"},
	{"name": "Budgets",            "route": "/glr/budgets",               "icon": "target"},
	{"name": "Trial Balance",      "route": "/glr/reports/trial-balance", "icon": "scale"},
	{"name": "Balance Sheet",      "route": "/glr/reports/balance-sheet", "icon": "landmark"},
	{"name": "Income Statement",   "route": "/glr/reports/income",        "icon": "trending-up"},
	{"name": "Cash Flow",          "route": "/glr/reports/cashflow",      "icon": "activity"},
	{"name": "Budget vs Actual",   "route": "/glr/reports/bva",           "icon": "bar-chart-2"},
	{"name": "Consolidation",      "route": "/glr/consolidation",         "icon": "git-merge"},
	{"name": "Year-End Close",     "route": "/glr/year-end",              "icon": "archive"},
	{"name": "Settings",           "route": "/glr/settings",             "icon": "settings"},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base(screen: str, tenant_id: str) -> dict[str, Any]:
	return {
		"screen": screen,
		"tenant_id": tenant_id,
		"navigation": NAVIGATION,
		"capability": "glr_general_ledger",
	}


def _run(coro) -> Any:
	loop = asyncio.new_event_loop()
	try:
		return loop.run_until_complete(coro)
	finally:
		loop.close()


def _d(v: Any) -> Decimal:
	try:
		return Decimal(str(v))
	except Exception:
		return Decimal("0")


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def dashboard_model(svc: GeneralLedgerService, tenant_id: str) -> dict[str, Any]:
	"""
	KPI dashboard view model.

	Returns summary counts, work queue, and recent activity.
	"""
	model = _base("dashboard", tenant_id)
	summary = svc.dashboard_summary(tenant_id)
	model["summary"] = summary

	unposted_journals = [
		j for j in svc.journal_entries.values()
		if j["tenant_id"] == tenant_id and j["status"] not in {"posted", "reversed", "cancelled"}
	]
	open_periods = [
		p for p in svc.periods.values()
		if p["tenant_id"] == tenant_id and p["status"] == "open"
	]
	pending_approvals = [
		wf for wf in svc.approval_workflows.values()
		if wf["tenant_id"] == tenant_id and wf["status"] == "pending"
	]
	open_reconciliations = [
		r for r in svc.reconciliations.values()
		if r["tenant_id"] == tenant_id and r.get("status") == "open"
	]

	model["work_queue"] = {
		"unposted_journals": len(unposted_journals),
		"open_periods": len(open_periods),
		"pending_approvals": len(pending_approvals),
		"open_reconciliations": len(open_reconciliations),
	}

	# Last 10 audit events
	model["recent_events"] = sorted(
		[e for e in svc._audit_events if e["tenant_id"] == tenant_id],
		key=lambda e: e.get("emitted_at", ""),
		reverse=True,
	)[:10]

	return model


# ---------------------------------------------------------------------------
# Chart of Accounts
# ---------------------------------------------------------------------------

def account_list_model(svc: GeneralLedgerService, tenant_id: str) -> dict[str, Any]:
	"""Pageable account list with type grouping."""
	model = _base("accounts", tenant_id)
	accounts = _run(svc.chart_of_accounts(tenant_id, include_inactive=False))
	model["records"] = accounts
	model["total"] = len(accounts)
	model["columns"] = [
		{"key": "code",          "label": "Code",         "sortable": True},
		{"key": "name",          "label": "Name",         "sortable": True},
		{"key": "account_type",  "label": "Type",         "sortable": True},
		{"key": "currency",      "label": "Currency",     "sortable": False},
		{"key": "allow_posting", "label": "Postable",     "sortable": False},
		{"key": "status",        "label": "Status",       "sortable": True},
	]
	model["filters"] = {
		"account_types": ["asset", "liability", "equity", "revenue", "expense", "contra"],
		"statuses": ["active", "inactive", "archived"],
	}
	return model


def account_detail_model(svc: GeneralLedgerService, tenant_id: str, account_id: str) -> dict[str, Any]:
	"""Single account detail with recent transaction summary."""
	model = _base("account_detail", tenant_id)
	acct = svc.accounts.get(account_id)
	if not acct or acct["tenant_id"] != tenant_id:
		model["error"] = "account_not_found"
		return model
	model["account"] = acct
	# Children
	model["children"] = [
		a for a in svc.accounts.values()
		if a["tenant_id"] == tenant_id and a.get("parent_account_id") == account_id
	]
	return model


def account_hierarchy_model(svc: GeneralLedgerService, tenant_id: str) -> dict[str, Any]:
	"""Nested tree for hierarchy visualisation."""
	model = _base("account_hierarchy", tenant_id)
	model["hierarchy"] = _run(svc.account_hierarchy(tenant_id))
	return model


def create_account_model(tenant_id: str) -> dict[str, Any]:
	"""Empty form model for new account creation."""
	model = _base("create_account", tenant_id)
	model["form"] = {
		"account_code": "",
		"account_name": "",
		"account_type": "asset",
		"currency": "USD",
		"allow_posting": True,
		"parent_account_code": None,
		"description": "",
		"tags": [],
	}
	model["account_types"] = ["asset", "liability", "equity", "revenue", "expense", "contra"]
	model["currencies"] = ["USD", "EUR", "GBP", "KES", "ZAR", "NGN", "GHS"]
	return model


def edit_account_model(svc: GeneralLedgerService, tenant_id: str, account_id: str) -> dict[str, Any]:
	"""Pre-populated form for editing an account."""
	model = _base("edit_account", tenant_id)
	acct = svc.accounts.get(account_id)
	if not acct or acct["tenant_id"] != tenant_id:
		model["error"] = "account_not_found"
		return model
	model["form"] = {
		"account_name": acct.get("name", ""),
		"allow_posting": acct.get("allow_posting", True),
		"tags": acct.get("tags", []),
		"description": acct.get("description", ""),
	}
	model["account"] = acct
	return model


# ---------------------------------------------------------------------------
# Periods
# ---------------------------------------------------------------------------

def period_list_model(svc: GeneralLedgerService, tenant_id: str) -> dict[str, Any]:
	"""Period list grouped by fiscal year."""
	model = _base("periods", tenant_id)
	periods = sorted(
		[p for p in svc.periods.values() if p["tenant_id"] == tenant_id],
		key=lambda p: (p.get("fiscal_year", 0), p.get("period_start", "")),
	)
	model["records"] = periods
	model["total"] = len(periods)
	model["actions"] = ["open", "close", "lock", "reopen"]
	model["status_classes"] = {
		"future": "text-gray-500",
		"open": "text-green-600",
		"soft_closed": "text-yellow-600",
		"closed": "text-orange-600",
		"locked": "text-red-700",
	}
	return model


def period_detail_model(svc: GeneralLedgerService, tenant_id: str, period_code: str) -> dict[str, Any]:
	"""Period detail with checklist."""
	model = _base("period_detail", tenant_id)
	period = svc._period_by_code(tenant_id, period_code)
	if not period:
		model["error"] = "period_not_found"
		return model
	model["period"] = period
	try:
		model["checklist"] = _run(svc.period_end_checklist(tenant_id, period_code))
	except Exception as exc:
		model["checklist_error"] = str(exc)
	return model


def create_period_model(tenant_id: str) -> dict[str, Any]:
	"""Empty form for period creation."""
	model = _base("create_period", tenant_id)
	model["form"] = {
		"period_code": "",
		"fiscal_year": 2026,
		"period_number": 1,
		"start_date": "",
		"end_date": "",
		"allows_adjustments": False,
	}
	return model


# ---------------------------------------------------------------------------
# Journal Entries
# ---------------------------------------------------------------------------

def journal_list_model(svc: GeneralLedgerService, tenant_id: str) -> dict[str, Any]:
	"""Journal list with status filter tabs."""
	model = _base("journals", tenant_id)
	all_journals = [j for j in svc.journal_entries.values() if j["tenant_id"] == tenant_id]
	all_journals.sort(key=lambda j: j.get("created_at", ""), reverse=True)
	model["records"] = all_journals
	model["total"] = len(all_journals)
	model["status_tabs"] = [
		{"status": "all",              "count": len(all_journals)},
		{"status": "draft",            "count": sum(1 for j in all_journals if j.get("status") == "draft")},
		{"status": "pending_approval", "count": sum(1 for j in all_journals if j.get("status") == "pending_approval")},
		{"status": "posted",           "count": sum(1 for j in all_journals if j.get("status") == "posted")},
		{"status": "reversed",         "count": sum(1 for j in all_journals if j.get("status") == "reversed")},
	]
	model["columns"] = [
		{"key": "journal_number", "label": "Number"},
		{"key": "journal_date",   "label": "Date"},
		{"key": "description",    "label": "Description"},
		{"key": "total_debits",   "label": "Debit"},
		{"key": "total_credits",  "label": "Credit"},
		{"key": "status",         "label": "Status"},
	]
	return model


def journal_detail_model(svc: GeneralLedgerService, tenant_id: str, journal_id: str) -> dict[str, Any]:
	"""Full journal detail with lines."""
	model = _base("journal_detail", tenant_id)
	journal = svc.journal_entries.get(journal_id)
	if not journal or journal["tenant_id"] != tenant_id:
		model["error"] = "journal_not_found"
		return model
	model["journal"] = journal
	model["posting"] = svc.postings.get(journal_id)
	model["available_actions"] = _journal_actions(journal)
	return model


def create_journal_model(svc: GeneralLedgerService, tenant_id: str) -> dict[str, Any]:
	"""Form model for new journal entry."""
	model = _base("create_journal", tenant_id)
	accounts = [
		{"id": a["id"], "code": a["code"], "name": a["name"], "type": a["account_type"]}
		for a in svc.accounts.values()
		if a["tenant_id"] == tenant_id and a.get("status") == "active" and a.get("allow_posting", True)
	]
	accounts.sort(key=lambda a: a["code"])
	open_periods = [
		{"id": p["id"], "code": p.get("period_code", p["id"])}
		for p in svc.periods.values()
		if p["tenant_id"] == tenant_id and p["status"] == "open"
	]
	model["accounts"] = accounts
	model["open_periods"] = open_periods
	model["journal_types"] = ["standard", "adjustment", "accrual", "reversal", "intercompany"]
	model["form"] = {
		"journal_date": "",
		"journal_type": "standard",
		"description": "",
		"reference": "",
		"lines": [
			{"account_id": "", "debit": "0.00", "credit": "0.00", "description": ""},
			{"account_id": "", "debit": "0.00", "credit": "0.00", "description": ""},
		],
	}
	return model


def edit_journal_model(svc: GeneralLedgerService, tenant_id: str, journal_id: str) -> dict[str, Any]:
	"""Pre-populated form — only for draft/balanced journals."""
	model = _base("edit_journal", tenant_id)
	journal = svc.journal_entries.get(journal_id)
	if not journal or journal["tenant_id"] != tenant_id:
		model["error"] = "journal_not_found"
		return model
	if journal.get("status") == "posted":
		model["error"] = "posted_journal_cannot_be_edited"
		return model
	model["journal"] = journal
	return model


def _journal_actions(journal: dict[str, Any]) -> list[str]:
	status = journal.get("status", "draft")
	actions: list[str] = []
	if status in {"draft", "balanced"}:
		actions += ["edit", "approve", "delete"]
	elif status == "pending_approval":
		actions += ["approve", "reject"]
	elif status == "approved":
		actions += ["post"]
	elif status == "posted":
		actions += ["reverse", "view-audit"]
	return actions


# ---------------------------------------------------------------------------
# Reconciliations
# ---------------------------------------------------------------------------

def reconciliation_list_model(svc: GeneralLedgerService, tenant_id: str) -> dict[str, Any]:
	model = _base("reconciliations", tenant_id)
	recs = [r for r in svc.reconciliations.values() if r.get("tenant_id") == tenant_id]
	model["records"] = recs
	model["total"] = len(recs)
	model["columns"] = [
		{"key": "account_code",          "label": "Account"},
		{"key": "period_code",           "label": "Period"},
		{"key": "gl_balance",            "label": "GL Balance"},
		{"key": "unreconciled_difference", "label": "Difference"},
		{"key": "status",                "label": "Status"},
	]
	return model


def reconciliation_detail_model(
	svc: GeneralLedgerService, tenant_id: str, reconciliation_id: str
) -> dict[str, Any]:
	model = _base("reconciliation_detail", tenant_id)
	rec = svc.reconciliations.get(reconciliation_id)
	if not rec or rec.get("tenant_id") != tenant_id:
		model["error"] = "reconciliation_not_found"
		return model
	model["reconciliation"] = rec
	model["available_actions"] = (
		["submit"] if rec.get("status") == "open"
		else ["approve", "reject"] if rec.get("status") == "submitted"
		else []
	)
	return model


# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------

def budget_list_model(svc: GeneralLedgerService, tenant_id: str) -> dict[str, Any]:
	model = _base("budgets", tenant_id)
	budgets = [b for b in svc.budgets.values() if b.get("tenant_id") == tenant_id]
	model["records"] = budgets
	model["total"] = len(budgets)
	model["columns"] = [
		{"key": "account_code",   "label": "Account"},
		{"key": "period_code",    "label": "Period"},
		{"key": "budget_type",    "label": "Type"},
		{"key": "budget_amount",  "label": "Amount"},
		{"key": "currency",       "label": "Currency"},
	]
	return model


def create_budget_model(svc: GeneralLedgerService, tenant_id: str) -> dict[str, Any]:
	model = _base("create_budget", tenant_id)
	accounts = [
		{"code": a["code"], "name": a["name"]}
		for a in svc.accounts.values()
		if a["tenant_id"] == tenant_id and a.get("status") == "active"
	]
	accounts.sort(key=lambda a: a["code"])
	model["accounts"] = accounts
	model["budget_types"] = ["original", "revised", "forecast"]
	return model


# ---------------------------------------------------------------------------
# Report views
# ---------------------------------------------------------------------------

def trial_balance_model(
	svc: GeneralLedgerService, tenant_id: str, period_code: str, include_zero: bool = False
) -> dict[str, Any]:
	model = _base("trial_balance", tenant_id)
	model["period_code"] = period_code
	try:
		report = _run(svc.trial_balance(tenant_id, period_code, include_zero_balances=include_zero))
		model["report"] = report
		model["balanced"] = report.get("balanced", False)
	except Exception as exc:
		model["error"] = str(exc)
	return model


def balance_sheet_model(
	svc: GeneralLedgerService,
	tenant_id: str,
	period_code: str,
	comparative_period: str | None = None,
) -> dict[str, Any]:
	model = _base("balance_sheet", tenant_id)
	model["period_code"] = period_code
	model["comparative_period"] = comparative_period
	try:
		report = _run(svc.balance_sheet(tenant_id, period_code, comparative_period=comparative_period))
		model["report"] = report
		model["balanced"] = report.get("balanced", False)
		model["kpis"] = {
			"total_assets": report.get("total_assets"),
			"total_liabilities_and_equity": report.get("total_liabilities_and_equity"),
		}
	except Exception as exc:
		model["error"] = str(exc)
	return model


def income_statement_model(
	svc: GeneralLedgerService,
	tenant_id: str,
	period_code: str,
	comparative_period: str | None = None,
	segment: str | None = None,
) -> dict[str, Any]:
	model = _base("income_statement", tenant_id)
	model["period_code"] = period_code
	model["comparative_period"] = comparative_period
	model["segment"] = segment
	try:
		report = _run(svc.income_statement(tenant_id, period_code,
		                                    comparative_period=comparative_period,
		                                    segment=segment))
		model["report"] = report
		model["kpis"] = {
			"revenue": report.get("revenue"),
			"gross_profit": report.get("gross_profit"),
			"ebit": report.get("ebit"),
			"pat": report.get("pat"),
		}
	except Exception as exc:
		model["error"] = str(exc)
	return model


def cash_flow_model(
	svc: GeneralLedgerService, tenant_id: str, period_code: str, method: str = "indirect"
) -> dict[str, Any]:
	model = _base("cash_flow", tenant_id)
	model["period_code"] = period_code
	model["method"] = method
	try:
		report = _run(svc.cash_flow_statement(tenant_id, period_code, method=method))
		model["report"] = report
		model["net_change_in_cash"] = report.get("net_change_in_cash")
	except Exception as exc:
		model["error"] = str(exc)
	return model


def budget_vs_actual_model(
	svc: GeneralLedgerService,
	tenant_id: str,
	period_code: str,
	budget_version: str = "approved",
) -> dict[str, Any]:
	model = _base("budget_vs_actual", tenant_id)
	model["period_code"] = period_code
	model["budget_version"] = budget_version
	try:
		report = _run(svc.budget_vs_actual(tenant_id, period_code, budget_version=budget_version))
		model["report"] = report
		model["adverse_count"] = sum(1 for r in report.get("rows", []) if r.get("indicator") == "A")
		model["favourable_count"] = sum(1 for r in report.get("rows", []) if r.get("indicator") == "F")
	except Exception as exc:
		model["error"] = str(exc)
	return model


def segment_report_model(
	svc: GeneralLedgerService,
	tenant_id: str,
	period_code: str,
	segment_dimension: str = "cost_center",
) -> dict[str, Any]:
	model = _base("segment_report", tenant_id)
	model["period_code"] = period_code
	model["segment_dimension"] = segment_dimension
	try:
		report = _run(svc.segment_report(tenant_id, period_code, segment_dimension=segment_dimension))
		model["report"] = report
	except Exception as exc:
		model["error"] = str(exc)
	return model


def management_pack_model(
	svc: GeneralLedgerService, tenant_id: str, period_code: str
) -> dict[str, Any]:
	model = _base("management_pack", tenant_id)
	model["period_code"] = period_code
	try:
		pack = _run(svc.management_accounts_pack(tenant_id, period_code))
		model["pack"] = pack
		model["ratios"] = pack.get("ratios", {})
		model["commentary"] = pack.get("commentary_template", "")
	except Exception as exc:
		model["error"] = str(exc)
	return model


# ---------------------------------------------------------------------------
# Year-end
# ---------------------------------------------------------------------------

def year_end_model(svc: GeneralLedgerService, tenant_id: str) -> dict[str, Any]:
	"""Year-end close landing page."""
	model = _base("year_end", tenant_id)
	closed_years = [
		fy for fy in svc.fiscal_years.values()
		if fy["tenant_id"] == tenant_id and fy.get("status") == "closed"
	]
	model["closed_years"] = closed_years
	equity_accounts = [
		{"code": a["code"], "name": a["name"]}
		for a in svc.accounts.values()
		if a["tenant_id"] == tenant_id
		and a.get("account_type") == "equity"
		and a.get("status") == "active"
	]
	model["equity_accounts"] = equity_accounts
	return model


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def settings_model(svc: GeneralLedgerService, tenant_id: str) -> dict[str, Any]:
	model = _base("settings", tenant_id)
	model["currency_rates_count"] = len([
		r for r in svc.currency_rates.values() if r.get("tenant_id") == tenant_id
	])
	model["recurring_templates_count"] = len([
		t for t in svc.recurring_templates.values() if t.get("tenant_id") == tenant_id
	])
	model["agents_count"] = len([
		a for a in svc.agents.values() if a.get("tenant_id") == tenant_id
	])
	return model


# ---------------------------------------------------------------------------
# Backward-compatibility aliases expected by package contract tests
# ---------------------------------------------------------------------------

def account_model(svc: GeneralLedgerService, tenant_id: str) -> dict[str, Any]:
	"""Alias for account_list_model — returns pageable account list."""
	return account_list_model(svc, tenant_id)


def agent_workbench_model(svc: GeneralLedgerService, tenant_id: str) -> dict[str, Any]:
	"""Agent workbench view: lists registered GLR agents and their recent actions."""
	model = _base("agent_workbench", tenant_id)
	agents = [a for a in svc.agents.values() if a.get("tenant_id") == tenant_id]
	model["records"] = agents
	model["total"] = len(agents)
	model["columns"] = [
		{"key": "name",    "label": "Agent Name"},
		{"key": "runtime", "label": "Runtime"},
		{"key": "role",    "label": "Role"},
		{"key": "scope",   "label": "Scope"},
		{"key": "status",  "label": "Status"},
	]
	model["recent_actions"] = [
		e for e in svc._audit_events
		if e.get("tenant_id") == tenant_id and "agent" in e.get("event_type", "")
	][-20:]
	return model
