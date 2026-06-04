"""
General Ledger — Flask Blueprint REST API.

url_prefix: /api/glr

All endpoints enforce tenant isolation via X-Tenant-ID header or query param.
Responses use {data, error, meta} envelope.

© 2025 Datacraft. All rights reserved.
"""

from __future__ import annotations

import asyncio
from functools import wraps
from typing import Any

from flask import Blueprint, jsonify, request

try:
	from .context import get_current_user_id, get_tenant_id_from_request
	from .service import GeneralLedgerService
	from .models import (
		GLAccountCreate, GLAccountUpdate,
		GLBudgetCreate, GLBudgetUpdate,
		GLCurrencyRateCreate,
		GLJournalEntryCreate, GLJournalEntryUpdate,
		GLPeriodCreate, GLPeriodUpdate,
		GLReconciliationCreate, GLReconciliationSubmit,
		GLReportRequest, GLBudgetVsActualRequest,
		GLYearEndRequest, GLPriorYearAdjRequest,
		GLConsolidationRequest, GLIntercompanyRequest,
	)
	from .domain.rules import RuleViolation
except ImportError:
	from context import get_current_user_id, get_tenant_id_from_request  # type: ignore
	from service import GeneralLedgerService  # type: ignore
	from models import (  # type: ignore
		GLAccountCreate, GLAccountUpdate,
		GLBudgetCreate, GLBudgetUpdate,
		GLCurrencyRateCreate,
		GLJournalEntryCreate, GLJournalEntryUpdate,
		GLPeriodCreate, GLPeriodUpdate,
		GLReconciliationCreate, GLReconciliationSubmit,
		GLReportRequest, GLBudgetVsActualRequest,
		GLYearEndRequest, GLPriorYearAdjRequest,
		GLConsolidationRequest, GLIntercompanyRequest,
	)
	from domain.rules import RuleViolation  # type: ignore


# ---------------------------------------------------------------------------
# Blueprint
# ---------------------------------------------------------------------------

bp = Blueprint("glr_general_ledger", __name__, url_prefix="/api/glr")

# Process-local service instance — replace with a factory/DI in production.
_svc = GeneralLedgerService()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok(data: Any, status: int = 200, meta: dict[str, Any] | None = None) -> tuple:
	body: dict[str, Any] = {"data": data, "error": None}
	if meta:
		body["meta"] = meta
	return jsonify(body), status


def _err(message: str, status: int = 400, code: str | None = None) -> tuple:
	return jsonify({"data": None, "error": {"message": message, "code": code or "bad_request"}}), status


def _tenant() -> str | None:
	return (
		request.headers.get("X-Tenant-ID")
		or request.args.get("tenant_id")
		or (request.json or {}).get("tenant_id")
	)


def _run(coro) -> Any:
	"""Run an async service method from a sync Flask route."""
	loop = asyncio.new_event_loop()
	try:
		return asyncio.run(coro)
	finally:
		loop.close()


def _parse_int(name: str, default: int) -> int:
	try:
		return int(request.args.get(name, default))
	except (TypeError, ValueError):
		return default


def require_tenant(f):
	"""Decorator: abort with 400 if no tenant header/param present."""
	@wraps(f)
	def wrapper(*args, **kwargs):
		if not get_tenant_id_from_request():
			return _err("X-Tenant-ID header or tenant_id param required", 400, "tenant_required")
		return f(*args, **kwargs)
	return wrapper


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

@bp.get("/health")
def health():
	return _ok({"status": "ok", "capability": "glr_general_ledger"})


@bp.get("/dashboard")
@require_tenant
def dashboard():
	tenant = get_tenant_id_from_request()
	user_id = get_current_user_id()
	return _ok(_svc.dashboard_summary(tenant))


# ---------------------------------------------------------------------------
# Chart of Accounts
# ---------------------------------------------------------------------------

@bp.get("/accounts")
@require_tenant
def list_accounts():
	tenant = get_tenant_id_from_request()
	user_id = get_current_user_id()
	include_inactive = request.args.get("include_inactive", "false").lower() == "true"
	page = _parse_int("page", 1)
	page_size = _parse_int("page_size", 50)
	try:
		accounts = _run(_svc.chart_of_accounts(tenant, include_inactive=include_inactive))
	except (ValueError, PermissionError) as exc:
		return _err(str(exc))
	start = (page - 1) * page_size
	return _ok(
		accounts[start: start + page_size],
		meta={"total": len(accounts), "page": page, "page_size": page_size},
	)


@bp.get("/accounts/hierarchy")
@require_tenant
def account_hierarchy():
	tenant = get_tenant_id_from_request()
	user_id = get_current_user_id()
	try:
		return _ok(_run(_svc.account_hierarchy(tenant)))
	except (ValueError, PermissionError) as exc:
		return _err(str(exc))


@bp.post("/accounts/create")
@require_tenant
def create_account():
	body = request.get_json(force=True) or {}
	body["tenant_id"] = get_tenant_id_from_request()
	try:
		req = GLAccountCreate(**body)
	except Exception as exc:
		return _err(str(exc), 422, "validation_error")
	try:
		result = _run(_svc.create_account_v2(
			tenant_id=req.tenant_id,
			account_code=req.account_code,
			account_name=req.account_name,
			account_type=req.account_type.value,
			parent_code=req.parent_account_code,
			currency=req.currency,
		))
		# Patch extra fields from request that service v2 doesn't consume directly
		result.update({
			"ifrs_mapping": req.ifrs_mapping,
			"gaap_mapping": req.gaap_mapping,
			"tax_code": req.tax_code,
			"cost_center_required": req.cost_center_required,
			"project_required": req.project_required,
			"is_reconciliation_account": req.is_reconciliation_account,
			"tags": req.tags,
			"description": req.description,
		})
		return _ok(result, 201)
	except (ValueError, PermissionError, RuleViolation) as exc:
		return _err(str(exc), 400)


@bp.get("/accounts/<account_id>")
@require_tenant
def get_account(account_id: str):
	tenant = get_tenant_id_from_request()
	user_id = get_current_user_id()
	acct = _svc.accounts.get(account_id)
	if not acct or acct.get("tenant_id") != tenant:
		return _err("account not found", 404, "not_found")
	return _ok(acct)


@bp.put("/accounts/<account_id>")
@require_tenant
def update_account(account_id: str):
	tenant = get_tenant_id_from_request()
	user_id = get_current_user_id()
	acct = _svc.accounts.get(account_id)
	if not acct or acct.get("tenant_id") != tenant:
		return _err("account not found", 404, "not_found")
	body = request.get_json(force=True) or {}
	try:
		upd = GLAccountUpdate(**body)
	except Exception as exc:
		return _err(str(exc), 422, "validation_error")
	if upd.account_name is not None:
		acct["name"] = upd.account_name
	if upd.allow_posting is not None:
		acct["allow_posting"] = upd.allow_posting
	if upd.tags is not None:
		acct["tags"] = upd.tags
	acct["updated_by"] = upd.updated_by
	return _ok(acct)


@bp.delete("/accounts/<account_id>")
@require_tenant
def delete_account(account_id: str):
	tenant = get_tenant_id_from_request()
	user_id = get_current_user_id()
	acct = _svc.accounts.get(account_id)
	if not acct or acct.get("tenant_id") != tenant:
		return _err("account not found", 404, "not_found")
	acct["status"] = "inactive"
	acct["is_deleted"] = True
	return _ok({"id": account_id, "status": "deleted"})


@bp.get("/accounts/<account_id>/analysis")
@require_tenant
def account_analysis(account_id: str):
	tenant = get_tenant_id_from_request()
	user_id = get_current_user_id()
	acct = _svc.accounts.get(account_id)
	if not acct or acct.get("tenant_id") != tenant:
		return _err("account not found", 404, "not_found")
	period_code = request.args.get("period_code", "")
	if not period_code:
		return _err("period_code param required", 400)
	try:
		result = _run(_svc.account_analysis(tenant, acct["code"], period_code))
		return _ok(result)
	except (ValueError, PermissionError) as exc:
		return _err(str(exc))


# ---------------------------------------------------------------------------
# Periods
# ---------------------------------------------------------------------------

@bp.get("/periods")
@require_tenant
def list_periods():
	tenant = get_tenant_id_from_request()
	user_id = get_current_user_id()
	fiscal_year = _parse_int("fiscal_year", 0)
	try:
		if fiscal_year:
			periods = _run(_svc.get_period_status(tenant, fiscal_year))
		else:
			periods = _svc.list_records("periods", tenant)
		return _ok(periods, meta={"total": len(periods)})
	except (ValueError, PermissionError) as exc:
		return _err(str(exc))


@bp.post("/periods/create")
@require_tenant
def create_period():
	body = request.get_json(force=True) or {}
	body["tenant_id"] = get_tenant_id_from_request()
	try:
		req = GLPeriodCreate(**body)
	except Exception as exc:
		return _err(str(exc), 422, "validation_error")
	result = _svc.open_period(
		period_id=f"period-{req.period_code}",
		tenant_id=req.tenant_id,
		name=req.period_code,
		fiscal_year=req.fiscal_year,
		period_start=req.start_date.isoformat(),
		period_end=req.end_date.isoformat(),
	)
	return _ok(result, 201)


@bp.get("/periods/<period_code>")
@require_tenant
def get_period(period_code: str):
	tenant = get_tenant_id_from_request()
	user_id = get_current_user_id()
	period = _svc._period_by_code(tenant, period_code)
	if not period:
		return _err("period not found", 404, "not_found")
	return _ok(period)


@bp.post("/periods/<period_code>/open")
@require_tenant
def open_period(period_code: str):
	tenant = get_tenant_id_from_request()
	user_id = get_current_user_id()
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.open_period_v2(tenant, period_code, body.get("opened_by", "system")))
		return _ok(result)
	except (ValueError, PermissionError, RuleViolation) as exc:
		return _err(str(exc), 400)


@bp.post("/periods/<period_code>/close")
@require_tenant
def close_period(period_code: str):
	tenant = get_tenant_id_from_request()
	user_id = get_current_user_id()
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.close_period(tenant, period_code, body.get("closed_by", "system")))
		return _ok(result)
	except (ValueError, PermissionError, RuleViolation) as exc:
		return _err(str(exc), 400)


@bp.post("/periods/<period_code>/lock")
@require_tenant
def lock_period(period_code: str):
	tenant = get_tenant_id_from_request()
	user_id = get_current_user_id()
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.lock_period(tenant, period_code, body.get("locked_by", "system")))
		return _ok(result)
	except (ValueError, PermissionError, RuleViolation) as exc:
		return _err(str(exc), 400)


@bp.post("/periods/<period_code>/reopen")
@require_tenant
def reopen_period(period_code: str):
	tenant = get_tenant_id_from_request()
	user_id = get_current_user_id()
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.reopen_period(
			tenant, period_code,
			body.get("reason", ""),
			body.get("authorised_by", ""),
		))
		return _ok(result)
	except (ValueError, PermissionError, RuleViolation) as exc:
		return _err(str(exc), 400)


@bp.get("/periods/<period_code>/checklist")
@require_tenant
def period_checklist(period_code: str):
	tenant = get_tenant_id_from_request()
	user_id = get_current_user_id()
	try:
		result = _run(_svc.period_end_checklist(tenant, period_code))
		return _ok(result)
	except (ValueError, PermissionError) as exc:
		return _err(str(exc))


# ---------------------------------------------------------------------------
# Journal Entries
# ---------------------------------------------------------------------------

@bp.get("/journals")
@require_tenant
def list_journals():
	tenant = get_tenant_id_from_request()
	user_id = get_current_user_id()
	page = _parse_int("page", 1)
	page_size = _parse_int("page_size", 50)
	status_filter = request.args.get("status")
	journals = [
		j for j in _svc.journal_entries.values()
		if j.get("tenant_id") == tenant
		and (not status_filter or j.get("status") == status_filter)
	]
	journals.sort(key=lambda j: j.get("created_at", ""), reverse=True)
	start = (page - 1) * page_size
	return _ok(
		journals[start: start + page_size],
		meta={"total": len(journals), "page": page, "page_size": page_size},
	)


@bp.post("/journals/create")
@require_tenant
def create_journal():
	body = request.get_json(force=True) or {}
	body["tenant_id"] = get_tenant_id_from_request()
	try:
		req = GLJournalEntryCreate(**body)
	except Exception as exc:
		return _err(str(exc), 422, "validation_error")
	lines = [
		{
			"account_id": ln.account_id,
			"debit": str(ln.debit),
			"credit": str(ln.credit),
			"description": ln.description,
			"cost_center": ln.cost_center,
			"project": ln.project,
			"segment": ln.segment,
		}
		for ln in req.lines
	]
	try:
		result = _run(_svc.post_journal_v2(
			tenant_id=req.tenant_id,
			journal_date=req.journal_date.isoformat(),
			journal_type=req.journal_type.value,
			lines=lines,
			description=req.description,
			reference=req.reference or "",
			posted_by=req.posted_by,
		))
		return _ok(result, 201)
	except (ValueError, PermissionError, RuleViolation) as exc:
		return _err(str(exc), 400)


@bp.get("/journals/<journal_id>")
@require_tenant
def get_journal(journal_id: str):
	tenant = get_tenant_id_from_request()
	user_id = get_current_user_id()
	journal = _svc.journal_entries.get(journal_id)
	if not journal or journal.get("tenant_id") != tenant:
		return _err("journal not found", 404, "not_found")
	return _ok(journal)


@bp.put("/journals/<journal_id>")
@require_tenant
def update_journal(journal_id: str):
	tenant = get_tenant_id_from_request()
	user_id = get_current_user_id()
	journal = _svc.journal_entries.get(journal_id)
	if not journal or journal.get("tenant_id") != tenant:
		return _err("journal not found", 404, "not_found")
	if journal.get("status") == "posted":
		return _err("posted journals cannot be edited; use reversal", 400, "journal_posted")
	body = request.get_json(force=True) or {}
	try:
		upd = GLJournalEntryUpdate(**body)
	except Exception as exc:
		return _err(str(exc), 422, "validation_error")
	if upd.description is not None:
		journal["description"] = upd.description
	if upd.reference is not None:
		journal["reference"] = upd.reference
	return _ok(journal)


@bp.delete("/journals/<journal_id>")
@require_tenant
def cancel_journal(journal_id: str):
	tenant = get_tenant_id_from_request()
	user_id = get_current_user_id()
	journal = _svc.journal_entries.get(journal_id)
	if not journal or journal.get("tenant_id") != tenant:
		return _err("journal not found", 404, "not_found")
	if journal.get("status") == "posted":
		return _err("use reversal to cancel posted journals", 400, "use_reversal")
	journal["status"] = "cancelled"
	return _ok({"id": journal_id, "status": "cancelled"})


@bp.post("/journals/<journal_id>/approve")
@require_tenant
def approve_journal(journal_id: str):
	tenant = get_tenant_id_from_request()
	user_id = get_current_user_id()
	body = request.get_json(force=True) or {}
	try:
		result = _svc.approve_journal(journal_id, tenant, body.get("approved_by", ""))
		return _ok(result)
	except (ValueError, PermissionError, KeyError, RuleViolation) as exc:
		return _err(str(exc), 400)


@bp.post("/journals/<journal_id>/post")
@require_tenant
def post_journal(journal_id: str):
	tenant = get_tenant_id_from_request()
	user_id = get_current_user_id()
	body = request.get_json(force=True) or {}
	posted_by = body.get("posted_by", "system")
	idempotency_key = body.get("idempotency_key") or f"post-{journal_id}"
	try:
		result = _svc.post_journal(journal_id, tenant, posted_by, idempotency_key)
		return _ok(result)
	except (ValueError, PermissionError, KeyError, RuleViolation) as exc:
		return _err(str(exc), 400)


@bp.post("/journals/<journal_id>/reverse")
@require_tenant
def reverse_journal(journal_id: str):
	tenant = get_tenant_id_from_request()
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.reverse_journal_v2(
			tenant_id=tenant,
			journal_id=journal_id,
			reversal_date=body.get("reversal_date", _svc._today()),
			reversal_description=body.get("description", f"Reversal of {journal_id}"),
			reversed_by=body.get("reversed_by", "system"),
		))
		return _ok(result)
	except (ValueError, PermissionError, RuleViolation) as exc:
		return _err(str(exc), 400)


@bp.post("/journals/<journal_id>/approval-workflow")
@require_tenant
def journal_approval_workflow(journal_id: str):
	tenant = get_tenant_id_from_request()
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.journal_approval_workflow(
			tenant_id=tenant,
			journal_id=journal_id,
			amount_threshold=str(body.get("amount_threshold", "0")),
			approver_id=body.get("approver_id", ""),
		))
		return _ok(result)
	except (ValueError, PermissionError, RuleViolation) as exc:
		return _err(str(exc), 400)


@bp.post("/journals/bulk-import")
@require_tenant
def bulk_import_journals():
	tenant = get_tenant_id_from_request()
	csv_data = request.get_data(as_text=True)
	if not csv_data:
		body = request.get_json(force=True) or {}
		csv_data = body.get("csv", "")
	if not csv_data:
		return _err("csv body required", 400)
	try:
		result = _run(_svc.bulk_journal_import(tenant, csv_data))
		return _ok(result)
	except (ValueError, PermissionError) as exc:
		return _err(str(exc))


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------

@bp.get("/reconciliations")
@require_tenant
def list_reconciliations():
	tenant = get_tenant_id_from_request()
	recs = [r for r in _svc.reconciliations.values() if r.get("tenant_id") == tenant]
	return _ok(recs, meta={"total": len(recs)})


@bp.post("/reconciliations/create")
@require_tenant
def create_reconciliation():
	body = request.get_json(force=True) or {}
	body["tenant_id"] = get_tenant_id_from_request()
	try:
		req = GLReconciliationCreate(**body)
	except Exception as exc:
		return _err(str(exc), 422, "validation_error")
	try:
		result = _run(_svc.account_reconciliation(req.tenant_id, req.account_code, req.period_code))
		return _ok(result, 201)
	except (ValueError, PermissionError, RuleViolation) as exc:
		return _err(str(exc), 400)


@bp.get("/reconciliations/<reconciliation_id>")
@require_tenant
def get_reconciliation(reconciliation_id: str):
	tenant = get_tenant_id_from_request()
	rec = _svc.reconciliations.get(reconciliation_id)
	if not rec or rec.get("tenant_id") != tenant:
		return _err("reconciliation not found", 404, "not_found")
	return _ok(rec)


@bp.post("/reconciliations/<reconciliation_id>/submit")
@require_tenant
def submit_reconciliation(reconciliation_id: str):
	tenant = get_tenant_id_from_request()
	body = request.get_json(force=True) or {}
	try:
		req = GLReconciliationSubmit(**body)
	except Exception as exc:
		return _err(str(exc), 422, "validation_error")
	try:
		result = _run(_svc.submit_reconciliation(
			tenant,
			reconciliation_id,
			req.reconciled_by,
			[item.model_dump() for item in req.reconciling_items],
		))
		return _ok(result)
	except (ValueError, PermissionError, RuleViolation) as exc:
		return _err(str(exc), 400)


@bp.post("/reconciliations/<reconciliation_id>/approve")
@require_tenant
def approve_reconciliation(reconciliation_id: str):
	tenant = get_tenant_id_from_request()
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.approve_reconciliation(
			tenant, reconciliation_id, body.get("approved_by", "system")
		))
		return _ok(result)
	except (ValueError, PermissionError, RuleViolation) as exc:
		return _err(str(exc), 400)


@bp.post("/reconciliations/bank")
@require_tenant
def bank_reconciliation():
	tenant = get_tenant_id_from_request()
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.bank_reconciliation(
			tenant,
			body.get("bank_account_code", ""),
			body.get("statement_id", ""),
		))
		return _ok(result)
	except (ValueError, PermissionError, RuleViolation) as exc:
		return _err(str(exc), 400)


@bp.post("/reconciliations/subledger")
@require_tenant
def subledger_reconciliation():
	tenant = get_tenant_id_from_request()
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.subledger_reconciliation(tenant, body.get("period_code", "")))
		return _ok(result)
	except (ValueError, PermissionError) as exc:
		return _err(str(exc))


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

@bp.get("/budgets")
@require_tenant
def list_budgets():
	tenant = get_tenant_id_from_request()
	budgets = [b for b in _svc.budgets.values() if b.get("tenant_id") == tenant]
	return _ok(budgets, meta={"total": len(budgets)})


@bp.post("/budgets/create")
@require_tenant
def create_budget():
	body = request.get_json(force=True) or {}
	body["tenant_id"] = get_tenant_id_from_request()
	try:
		req = GLBudgetCreate(**body)
	except Exception as exc:
		return _err(str(exc), 422, "validation_error")
	budget_id = f"budget-{req.budget_code}-{req.period_code}"
	record = {
		"id": budget_id,
		"type": "gl_budget",
		"tenant_id": req.tenant_id,
		"budget_code": req.budget_code,
		"fiscal_year": req.fiscal_year,
		"budget_type": req.budget_type.value,
		"account_code": req.account_code,
		"period_code": req.period_code,
		"budget_amount": str(req.amount),
		"currency": req.currency,
		"budget_version": "approved",
		"status": "active",
	}
	_svc.budgets[budget_id] = record
	return _ok(record, 201)


@bp.put("/budgets/<budget_id>")
@require_tenant
def update_budget(budget_id: str):
	tenant = get_tenant_id_from_request()
	budget = _svc.budgets.get(budget_id)
	if not budget or budget.get("tenant_id") != tenant:
		return _err("budget not found", 404, "not_found")
	body = request.get_json(force=True) or {}
	try:
		upd = GLBudgetUpdate(**body)
	except Exception as exc:
		return _err(str(exc), 422, "validation_error")
	if upd.amount is not None:
		budget["budget_amount"] = str(upd.amount)
	if upd.budget_type is not None:
		budget["budget_type"] = upd.budget_type.value
	return _ok(budget)


@bp.delete("/budgets/<budget_id>")
@require_tenant
def delete_budget(budget_id: str):
	tenant = get_tenant_id_from_request()
	budget = _svc.budgets.get(budget_id)
	if not budget or budget.get("tenant_id") != tenant:
		return _err("budget not found", 404, "not_found")
	budget["status"] = "deleted"
	return _ok({"id": budget_id, "status": "deleted"})


# ---------------------------------------------------------------------------
# Currency Rates
# ---------------------------------------------------------------------------

@bp.get("/currency-rates")
@require_tenant
def list_currency_rates():
	tenant = get_tenant_id_from_request()
	rates = [r for r in _svc.currency_rates.values() if r.get("tenant_id") == tenant]
	return _ok(rates, meta={"total": len(rates)})


@bp.post("/currency-rates/create")
@require_tenant
def create_currency_rate():
	body = request.get_json(force=True) or {}
	body["tenant_id"] = get_tenant_id_from_request()
	try:
		req = GLCurrencyRateCreate(**body)
	except Exception as exc:
		return _err(str(exc), 422, "validation_error")
	result = _svc.record_currency_rate(
		rate_id=f"rate-{req.from_currency}-{req.to_currency}-{req.effective_date}",
		tenant_id=req.tenant_id,
		from_currency=req.from_currency,
		to_currency=req.to_currency,
		exchange_rate=float(req.exchange_rate),
	)
	return _ok(result, 201)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

@bp.get("/reports/trial-balance")
@require_tenant
def report_trial_balance():
	tenant = get_tenant_id_from_request()
	period_code = request.args.get("period_code", "")
	if not period_code:
		return _err("period_code required", 400)
	include_zero = request.args.get("include_zero_balances", "false").lower() == "true"
	try:
		result = _run(_svc.trial_balance(tenant, period_code, include_zero_balances=include_zero))
		return _ok(result)
	except (ValueError, PermissionError) as exc:
		return _err(str(exc))


@bp.get("/reports/balance-sheet")
@require_tenant
def report_balance_sheet():
	tenant = get_tenant_id_from_request()
	period_code = request.args.get("period_code", "")
	if not period_code:
		return _err("period_code required", 400)
	comparative = request.args.get("comparative_period")
	try:
		result = _run(_svc.balance_sheet(tenant, period_code, comparative_period=comparative))
		return _ok(result)
	except (ValueError, PermissionError) as exc:
		return _err(str(exc))


@bp.get("/reports/income-statement")
@require_tenant
def report_income_statement():
	tenant = get_tenant_id_from_request()
	period_code = request.args.get("period_code", "")
	if not period_code:
		return _err("period_code required", 400)
	comparative = request.args.get("comparative_period")
	segment = request.args.get("segment")
	try:
		result = _run(_svc.income_statement(tenant, period_code, comparative_period=comparative, segment=segment))
		return _ok(result)
	except (ValueError, PermissionError) as exc:
		return _err(str(exc))


@bp.get("/reports/cash-flow")
@require_tenant
def report_cash_flow():
	tenant = get_tenant_id_from_request()
	period_code = request.args.get("period_code", "")
	if not period_code:
		return _err("period_code required", 400)
	method = request.args.get("method", "indirect")
	try:
		result = _run(_svc.cash_flow_statement(tenant, period_code, method=method))
		return _ok(result)
	except (ValueError, PermissionError) as exc:
		return _err(str(exc))


@bp.get("/reports/budget-vs-actual")
@require_tenant
def report_budget_vs_actual():
	tenant = get_tenant_id_from_request()
	period_code = request.args.get("period_code", "")
	if not period_code:
		return _err("period_code required", 400)
	budget_version = request.args.get("budget_version", "approved")
	try:
		result = _run(_svc.budget_vs_actual(tenant, period_code, budget_version=budget_version))
		return _ok(result)
	except (ValueError, PermissionError) as exc:
		return _err(str(exc))


@bp.get("/reports/segment")
@require_tenant
def report_segment():
	tenant = get_tenant_id_from_request()
	period_code = request.args.get("period_code", "")
	if not period_code:
		return _err("period_code required", 400)
	dimension = request.args.get("dimension", "cost_center")
	try:
		result = _run(_svc.segment_report(tenant, period_code, segment_dimension=dimension))
		return _ok(result)
	except (ValueError, PermissionError) as exc:
		return _err(str(exc))


@bp.get("/reports/management-pack")
@require_tenant
def report_management_pack():
	tenant = get_tenant_id_from_request()
	period_code = request.args.get("period_code", "")
	if not period_code:
		return _err("period_code required", 400)
	try:
		result = _run(_svc.management_accounts_pack(tenant, period_code))
		return _ok(result)
	except (ValueError, PermissionError) as exc:
		return _err(str(exc))


@bp.get("/reports/statement-of-equity")
@require_tenant
def report_statement_of_equity():
	tenant = get_tenant_id_from_request()
	fiscal_year = _parse_int("fiscal_year", 0)
	if not fiscal_year:
		return _err("fiscal_year required", 400)
	try:
		result = _run(_svc.statement_of_equity(tenant, fiscal_year))
		return _ok(result)
	except (ValueError, PermissionError) as exc:
		return _err(str(exc))


@bp.get("/reports/xbrl")
@require_tenant
def report_xbrl():
	tenant = get_tenant_id_from_request()
	period_code = request.args.get("period_code", "")
	if not period_code:
		return _err("period_code required", 400)
	framework = request.args.get("framework", "IFRS")
	try:
		result = _run(_svc.xbrl_tagging_extract(tenant, period_code, framework=framework))
		return _ok(result)
	except (ValueError, PermissionError) as exc:
		return _err(str(exc))


# ---------------------------------------------------------------------------
# Year-end operations
# ---------------------------------------------------------------------------

@bp.post("/year-end/close")
@require_tenant
def year_end_close():
	body = request.get_json(force=True) or {}
	body["tenant_id"] = get_tenant_id_from_request()
	try:
		req = GLYearEndRequest(**body)
	except Exception as exc:
		return _err(str(exc), 422, "validation_error")
	try:
		result = _run(_svc.year_end_closing(req.tenant_id, req.fiscal_year, req.retained_earnings_account))
		return _ok(result)
	except (ValueError, PermissionError, RuleViolation) as exc:
		return _err(str(exc), 400)


@bp.post("/year-end/opening-balances")
@require_tenant
def year_end_opening_balances():
	tenant = get_tenant_id_from_request()
	body = request.get_json(force=True) or {}
	new_year = body.get("new_fiscal_year")
	if not new_year:
		return _err("new_fiscal_year required", 400)
	try:
		result = _run(_svc.opening_balances_new_year(tenant, int(new_year)))
		return _ok(result)
	except (ValueError, PermissionError, RuleViolation) as exc:
		return _err(str(exc), 400)


@bp.post("/year-end/prior-year-adjustment")
@require_tenant
def prior_year_adjustment():
	body = request.get_json(force=True) or {}
	body["tenant_id"] = get_tenant_id_from_request()
	try:
		req = GLPriorYearAdjRequest(**body)
	except Exception as exc:
		return _err(str(exc), 422, "validation_error")
	try:
		result = _run(_svc.prior_year_adjustment(
			req.tenant_id, req.account_code, str(req.amount), req.adjustment_reason
		))
		return _ok(result)
	except (ValueError, PermissionError, RuleViolation) as exc:
		return _err(str(exc), 400)


# ---------------------------------------------------------------------------
# IFRS Consolidation
# ---------------------------------------------------------------------------

@bp.post("/consolidation")
@require_tenant
def consolidation():
	body = request.get_json(force=True) or {}
	body["tenant_id"] = get_tenant_id_from_request()
	try:
		req = GLConsolidationRequest(**body)
	except Exception as exc:
		return _err(str(exc), 422, "validation_error")
	try:
		result = _run(_svc.ifrs_consolidation(
			req.tenant_id,
			req.subsidiaries,
			req.group_adjustments,
			req.minority_interest,
		))
		return _ok(result)
	except (ValueError, PermissionError, RuleViolation) as exc:
		return _err(str(exc), 400)


# ---------------------------------------------------------------------------
# Intercompany
# ---------------------------------------------------------------------------

@bp.post("/intercompany")
@require_tenant
def intercompany():
	body = request.get_json(force=True) or {}
	body["tenant_id"] = get_tenant_id_from_request()
	try:
		req = GLIntercompanyRequest(**body)
	except Exception as exc:
		return _err(str(exc), 422, "validation_error")
	try:
		result = _run(_svc.intercompany_journal(
			req.tenant_id,
			req.counterpart_entity,
			str(req.amount),
			req.currency,
			req.account_mapping,
		))
		return _ok(result, 201)
	except (ValueError, PermissionError, RuleViolation) as exc:
		return _err(str(exc), 400)


@bp.post("/intercompany/reconcile")
@require_tenant
def intercompany_reconcile():
	tenant = get_tenant_id_from_request()
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.intercompany_reconciliation(
			tenant,
			body.get("counterpart_entity", ""),
			body.get("period_code", ""),
		))
		return _ok(result)
	except (ValueError, PermissionError) as exc:
		return _err(str(exc))


# ---------------------------------------------------------------------------
# Recurring journals
# ---------------------------------------------------------------------------

@bp.get("/recurring-templates")
@require_tenant
def list_recurring_templates():
	tenant = get_tenant_id_from_request()
	templates = [t for t in _svc.recurring_templates.values() if t.get("tenant_id") == tenant]
	return _ok(templates, meta={"total": len(templates)})


@bp.post("/recurring-templates/create")
@require_tenant
def create_recurring_template():
	tenant = get_tenant_id_from_request()
	body = request.get_json(force=True) or {}
	body["tenant_id"] = tenant
	tmpl_id = _svc._record_id("tmpl")
	body["id"] = tmpl_id
	body["type"] = "recurring_template"
	body["status"] = "active"
	_svc.recurring_templates[tmpl_id] = body
	return _ok(body, 201)


@bp.post("/recurring-templates/<template_id>/run")
@require_tenant
def run_recurring_template(template_id: str):
	tenant = get_tenant_id_from_request()
	body = request.get_json(force=True) or {}
	period = body.get("period", "")
	if not period:
		return _err("period required", 400)
	tmpl = _svc.recurring_templates.get(template_id)
	if not tmpl or tmpl.get("tenant_id") != tenant:
		return _err("template not found", 404, "not_found")
	try:
		result = _run(_svc.recurring_journal_run(tenant, template_id, period))
		return _ok(result)
	except (ValueError, PermissionError, RuleViolation) as exc:
		return _err(str(exc), 400)


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

@bp.get("/audit-events")
@require_tenant
def audit_events():
	tenant = get_tenant_id_from_request()
	events = _svc.audit_events(tenant)
	page = _parse_int("page", 1)
	page_size = _parse_int("page_size", 100)
	start = (page - 1) * page_size
	return _ok(
		events[start: start + page_size],
		meta={"total": len(events), "page": page, "page_size": page_size},
	)


# ---------------------------------------------------------------------------
# Module-level compatibility helpers used by package contract tests
# (these mirror the old functional api.py signatures)
# ---------------------------------------------------------------------------

def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	"""Return health/status dict for the capability."""
	return {"ok": True, "capability": "glr_general_ledger", "summary": _svc.dashboard_summary(tenant_id)}


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	"""Generic composition helper — creates a GL account and returns its record."""
	return _svc.create_account(
		payload.get("account_id", "account"),
		payload["tenant_id"],
		payload.get("code", "1000"),
		payload.get("name", "API Account"),
		payload.get("account_type", "asset"),
		payload.get("parent_account_id"),
		payload.get("allow_posting", True),
		payload.get("currency", "USD"),
	)


def list_records(collection: str, tenant_id: str = "default") -> list[dict[str, Any]]:
	"""Return all records in the given in-memory collection for a tenant."""
	return _svc.list_records(collection, tenant_id)
