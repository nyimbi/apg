"""Flask Blueprint REST API for SACCO General Ledger."""
from __future__ import annotations

import asyncio
import logging
from decimal import Decimal
from typing import Any

from flask import Blueprint, jsonify, request

from .service import SACCOGLService

_log = logging.getLogger(__name__)

bp = Blueprint("sacco_gl", __name__, url_prefix="/api/fintech/sacco/gl")
_svc = SACCOGLService()


def _run(coro: Any) -> Any:
	loop = asyncio.new_event_loop()
	try:
		return loop.run_until_complete(coro)
	finally:
		loop.close()


def _tenant() -> str:
	return request.headers.get("X-Tenant-ID", "default")


def _dec(val: Any, default: str = "0") -> Decimal:
	return Decimal(str(val)) if val is not None else Decimal(default)


# ── Health ─────────────────────────────────────────────────────────────────────

@bp.get("/health")
def health():
	return jsonify(_run(_svc.health_check())), 200


# ── Chart of Accounts ──────────────────────────────────────────────────────────

@bp.post("/coa/init")
def init_coa():
	"""POST /api/fintech/sacco/gl/coa/init — initialise standard SACCO COA."""
	result = _run(_svc.initialise_sacco_coa(tenant_id=_tenant()))
	return jsonify(result), 200


# ── Generic Transaction ────────────────────────────────────────────────────────

@bp.post("/transactions")
def post_transaction():
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.post_transaction(
			tenant_id=_tenant(),
			transaction_type=body["transaction_type"],
			entries=body["entries"],
			reference=body["reference"],
			value_date=body["value_date"],
			posted_by=body.get("posted_by", "api"),
			narration=body.get("narration", ""),
		))
		return jsonify(result), 201
	except KeyError as exc:
		return jsonify({"error": f"missing_field: {exc}"}), 400
	except ValueError as exc:
		return jsonify({"error": str(exc)}), 422


# ── Standard Transaction Shortcuts ────────────────────────────────────────────

@bp.post("/deposits")
def post_deposit():
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.post_member_deposit(
			tenant_id=_tenant(),
			member_id=body["member_id"],
			account_type=body.get("account_type", "FOSA"),
			amount=_dec(body["amount"]),
			channel=body.get("channel", "cash"),
			value_date=body.get("value_date"),
			posted_by=body.get("posted_by", "api"),
		))
		return jsonify(result), 201
	except (KeyError, TypeError) as exc:
		return jsonify({"error": str(exc)}), 400
	except (ValueError, AssertionError) as exc:
		return jsonify({"error": str(exc)}), 422


@bp.post("/loans/disbursements")
def post_disbursement():
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.post_loan_disbursement(
			tenant_id=_tenant(),
			loan_id=body["loan_id"],
			amount=_dec(body["amount"]),
			loan_type=body.get("loan_type", "BOSA"),
			disbursement_channel=body.get("disbursement_channel", "savings_account"),
			value_date=body.get("value_date"),
			posted_by=body.get("posted_by", "api"),
		))
		return jsonify(result), 201
	except (KeyError, TypeError) as exc:
		return jsonify({"error": str(exc)}), 400
	except (ValueError, AssertionError) as exc:
		return jsonify({"error": str(exc)}), 422


@bp.post("/loans/repayments")
def post_repayment():
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.post_loan_repayment(
			tenant_id=_tenant(),
			loan_id=body["loan_id"],
			principal=_dec(body["principal"]),
			interest=_dec(body.get("interest", "0")),
			penalty=_dec(body.get("penalty", "0")),
			payment_channel=body.get("payment_channel", "cash"),
			value_date=body.get("value_date"),
			posted_by=body.get("posted_by", "api"),
		))
		return jsonify(result), 201
	except (KeyError, TypeError) as exc:
		return jsonify({"error": str(exc)}), 400
	except (ValueError, AssertionError) as exc:
		return jsonify({"error": str(exc)}), 422


@bp.post("/interest")
def post_interest():
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.post_interest_earned(
			tenant_id=_tenant(),
			account_id=body["account_id"],
			amount=_dec(body["amount"]),
			period=body["period"],
			account_type=body.get("account_type", "BOSA"),
			value_date=body.get("value_date"),
			posted_by=body.get("posted_by", "api"),
		))
		return jsonify(result), 201
	except (KeyError, TypeError) as exc:
		return jsonify({"error": str(exc)}), 400
	except (ValueError, AssertionError) as exc:
		return jsonify({"error": str(exc)}), 422


@bp.post("/dividends")
def post_dividend():
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.post_dividend(
			tenant_id=_tenant(),
			member_id=body["member_id"],
			amount=_dec(body["amount"]),
			year=int(body["year"]),
			pay_to_deposits=body.get("pay_to_deposits", False),
			value_date=body.get("value_date"),
			posted_by=body.get("posted_by", "api"),
		))
		return jsonify(result), 201
	except (KeyError, TypeError) as exc:
		return jsonify({"error": str(exc)}), 400
	except (ValueError, AssertionError) as exc:
		return jsonify({"error": str(exc)}), 422


@bp.post("/shares")
def post_share_purchase():
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.post_share_purchase(
			tenant_id=_tenant(),
			member_id=body["member_id"],
			amount=_dec(body["amount"]),
			channel=body.get("channel", "cash"),
			value_date=body.get("value_date"),
			posted_by=body.get("posted_by", "api"),
		))
		return jsonify(result), 201
	except (KeyError, TypeError) as exc:
		return jsonify({"error": str(exc)}), 400
	except (ValueError, AssertionError) as exc:
		return jsonify({"error": str(exc)}), 422


@bp.post("/withdrawals")
def post_withdrawal():
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.post_withdrawal(
			tenant_id=_tenant(),
			member_id=body["member_id"],
			amount=_dec(body["amount"]),
			account_type=body.get("account_type", "FOSA"),
			channel=body.get("channel", "cash"),
			value_date=body.get("value_date"),
			posted_by=body.get("posted_by", "api"),
		))
		return jsonify(result), 201
	except (KeyError, TypeError) as exc:
		return jsonify({"error": str(exc)}), 400
	except (ValueError, AssertionError) as exc:
		return jsonify({"error": str(exc)}), 422


@bp.post("/provisions")
def post_provision():
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.post_provision(
			tenant_id=_tenant(),
			loan_id=body["loan_id"],
			provision_amount=_dec(body["provision_amount"]),
			value_date=body.get("value_date"),
			posted_by=body.get("posted_by", "api"),
		))
		return jsonify(result), 201
	except (KeyError, TypeError) as exc:
		return jsonify({"error": str(exc)}), 400
	except (ValueError, AssertionError) as exc:
		return jsonify({"error": str(exc)}), 422


@bp.post("/write-offs")
def post_write_off():
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.post_write_off(
			tenant_id=_tenant(),
			loan_id=body["loan_id"],
			amount=_dec(body["amount"]),
			loan_type=body.get("loan_type", "BOSA"),
			value_date=body.get("value_date"),
			posted_by=body.get("posted_by", "api"),
		))
		return jsonify(result), 201
	except (KeyError, TypeError) as exc:
		return jsonify({"error": str(exc)}), 400
	except (ValueError, AssertionError) as exc:
		return jsonify({"error": str(exc)}), 422


# ── Reporting ──────────────────────────────────────────────────────────────────

@bp.get("/accounts/<account_code>/balance")
def account_balance(account_code: str):
	as_of = request.args.get("as_of_date")
	try:
		bal = _run(_svc.get_account_balance(_tenant(), account_code, as_of))
		return jsonify({"account_code": account_code, "balance": str(bal)}), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/trial-balance")
def trial_balance():
	as_of = request.args.get("as_of_date")
	if not as_of:
		from datetime import date
		as_of = date.today().isoformat()
	rows = _run(_svc.get_trial_balance(_tenant(), as_of))
	# Convert Decimals to str for JSON
	for r in rows:
		for k in ("debit", "credit", "net"):
			r[k] = str(r[k])
	return jsonify({"as_of_date": as_of, "rows": rows, "count": len(rows)}), 200


@bp.get("/balance-sheet")
def balance_sheet():
	as_of = request.args.get("as_of_date")
	if not as_of:
		from datetime import date
		as_of = date.today().isoformat()
	bs = _run(_svc.get_balance_sheet(_tenant(), as_of))
	return jsonify({
		"as_of_date": bs.as_of_date,
		"assets": {k: str(v) for k, v in bs.assets.items()},
		"liabilities": {k: str(v) for k, v in bs.liabilities.items()},
		"equity": {k: str(v) for k, v in bs.equity.items()},
		"total_assets": str(bs.total_assets),
		"total_liabilities": str(bs.total_liabilities),
		"total_equity": str(bs.total_equity),
		"total_liabilities_equity": str(bs.total_liabilities_equity),
		"is_balanced": bs.is_balanced,
	}), 200


@bp.get("/income-statement")
def income_statement():
	from_date = request.args.get("from_date")
	to_date = request.args.get("to_date")
	if not from_date or not to_date:
		return jsonify({"error": "from_date and to_date required"}), 400
	stmt = _run(_svc.get_income_statement(_tenant(), from_date, to_date))
	return jsonify({
		"from_date": stmt.from_date,
		"to_date": stmt.to_date,
		"income": {k: str(v) for k, v in stmt.income.items()},
		"expenses": {k: str(v) for k, v in stmt.expenses.items()},
		"total_income": str(stmt.total_income),
		"total_expenses": str(stmt.total_expenses),
		"surplus_deficit": str(stmt.surplus_deficit),
	}), 200


@bp.get("/journal-entries")
def journal_entries():
	from_date = request.args.get("from_date")
	to_date = request.args.get("to_date")
	if not from_date or not to_date:
		return jsonify({"error": "from_date and to_date required"}), 400
	entries = _run(_svc.get_journal_entries(
		tenant_id=_tenant(),
		from_date=from_date,
		to_date=to_date,
		account_code=request.args.get("account_code"),
		transaction_type=request.args.get("transaction_type"),
		limit=int(request.args.get("limit", 50)),
	))
	return jsonify({"entries": entries, "count": len(entries)}), 200


@bp.get("/summary")
def gl_summary():
	period = request.args.get("period")
	if not period:
		from datetime import date
		period = date.today().strftime("%Y-%m")
	summary = _run(_svc.get_gl_summary(_tenant(), period))
	data = summary.model_dump()
	for k, v in data.items():
		if isinstance(v, Decimal):
			data[k] = str(v)
	return jsonify(data), 200


@bp.get("/validate")
def validate():
	as_of = request.args.get("as_of_date")
	result = _run(_svc.validate_double_entry(_tenant(), as_of))
	return jsonify(result), 200


# ── Period Management ──────────────────────────────────────────────────────────

@bp.post("/periods/open")
def open_period():
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.open_period(_tenant(), int(body["year"]), int(body["month"])))
		return jsonify(result), 200
	except (KeyError, TypeError) as exc:
		return jsonify({"error": str(exc)}), 400


@bp.post("/periods/close")
def close_period():
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.close_period(
			_tenant(), int(body["year"]), int(body["month"]), body.get("closed_by", "api")
		))
		return jsonify(result), 200
	except (KeyError, TypeError) as exc:
		return jsonify({"error": str(exc)}), 400
	except ValueError as exc:
		return jsonify({"error": str(exc)}), 422


@bp.get("/periods/<int:year>/<int:month>")
def period_status(year: int, month: int):
	result = _run(_svc.get_period_status(_tenant(), year, month))
	return jsonify(result), 200


# ── Reconciliation ─────────────────────────────────────────────────────────────

@bp.get("/reconciliation")
def reconcile():
	as_of = request.args.get("as_of_date")
	if not as_of:
		from datetime import date
		as_of = date.today().isoformat()
	result = _run(_svc.reconcile_subsidiary_ledgers(_tenant(), as_of))
	data = result.model_dump()
	for k in ("gl_total_deposits", "subsidiary_total_deposits", "gl_total_loans", "subsidiary_total_loans"):
		data[k] = str(data[k])
	return jsonify(data), 200
