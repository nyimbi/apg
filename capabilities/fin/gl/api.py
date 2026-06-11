"""General Ledger — REST API endpoints."""
from __future__ import annotations

import logging
from datetime import date
from typing import Any

from flask import Blueprint, jsonify, request

from .service import GLService

_log = logging.getLogger(__name__)

gl_api = Blueprint("gl_api", __name__, url_prefix="/api/fin/gl")


def _svc() -> GLService:
	return GLService(tenant_id=request.headers.get("X-Tenant-Id", "default"))


# ── Health ────────────────────────────────────────────────────────

@gl_api.get("/health")
async def health():
	return jsonify(await _svc().health_check())


# ── Chart of Accounts ─────────────────────────────────────────────

@gl_api.post("/coa/initialise")
async def initialise_coa():
	return jsonify(_svc().initialise_standard_coa()), 201


@gl_api.get("/accounts")
async def list_accounts():
	svc = _svc()
	account_type = request.args.get("account_type")
	search = request.args.get("search")
	active_only = request.args.get("active_only", "true").lower() == "true"
	accounts = await svc.list_accounts(account_type=account_type, active_only=active_only, search=search)
	return jsonify({"accounts": accounts, "total": len(accounts)})


@gl_api.post("/accounts")
async def create_account():
	body = request.get_json(force=True) or {}
	svc = _svc()
	try:
		account = await svc.create_account(
			code=body["code"],
			name=body["name"],
			account_type=body["account_type"],
			normal_balance=body["normal_balance"],
			parent_code=body.get("parent_code"),
			currency=body.get("currency", "KES"),
		)
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400
	return jsonify(account), 201


@gl_api.get("/accounts/<code>")
async def get_account(code: str):
	svc = _svc()
	try:
		return jsonify(await svc.get_account(code))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@gl_api.get("/accounts/<code>/balance")
async def get_balance(code: str):
	svc = _svc()
	as_of_date = request.args.get("as_of_date")
	try:
		return jsonify(await svc.get_account_balance(code, as_of_date))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@gl_api.get("/accounts/<code>/movements")
async def get_movements(code: str):
	svc = _svc()
	period_id = request.args.get("period_id", "")
	return jsonify(await svc.get_account_movements(code, period_id))


@gl_api.get("/accounts/hierarchy")
async def get_hierarchy():
	return jsonify(await _svc().get_account_hierarchy())


# ── Periods ───────────────────────────────────────────────────────

@gl_api.post("/periods")
async def open_period():
	body = request.get_json(force=True) or {}
	svc = _svc()
	try:
		period = await svc.open_period(
			period_id=body["period_id"],
			year=int(body["year"]),
			month=int(body["month"]),
		)
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400
	return jsonify(period), 201


@gl_api.put("/periods/<period_id>/close")
async def close_period(period_id: str):
	svc = _svc()
	body = request.get_json(force=True) or {}
	try:
		return jsonify(await svc.close_period(period_id, closed_by=body.get("closed_by", "api")))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@gl_api.get("/periods")
async def list_periods():
	return jsonify({"periods": await _svc().list_periods()})


# ── Journal Entries ───────────────────────────────────────────────

@gl_api.post("/journal-entries")
async def post_journal_entry():
	body = request.get_json(force=True) or {}
	svc = _svc()
	try:
		from .service import GLImbalanceError, PostingToClosedPeriodError, AccountNotFoundError
		je = await svc.post_journal_entry(
			entries=body["entries"],
			description=body["description"],
			reference=body["reference"],
			posting_date=body.get("posting_date", str(date.today())),
			period_id=body["period_id"],
			posted_by=request.headers.get("X-Actor-Id", "api"),
		)
	except GLImbalanceError as exc:
		return jsonify({"error": "Journal entry not balanced", "detail": str(exc)}), 422
	except PostingToClosedPeriodError as exc:
		return jsonify({"error": "Period is closed", "detail": str(exc)}), 422
	except AccountNotFoundError as exc:
		return jsonify({"error": str(exc)}), 404
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 400
	return jsonify(je), 201


@gl_api.get("/journal-entries")
async def list_journal_entries():
	svc = _svc()
	result = await svc.get_journal_entries(
		account_code=request.args.get("account_code"),
		from_date=request.args.get("from_date"),
		to_date=request.args.get("to_date"),
		reference=request.args.get("reference"),
		limit=int(request.args.get("limit", "50")),
		page=int(request.args.get("page", "1")),
	)
	return jsonify(result)


@gl_api.get("/journal-entries/<journal_id>")
async def get_journal_entry(journal_id: str):
	svc = _svc()
	try:
		return jsonify(await svc.get_journal_entry(journal_id))
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@gl_api.post("/journal-entries/<journal_id>/reverse")
async def reverse_journal_entry(journal_id: str):
	svc = _svc()
	body = request.get_json(force=True) or {}
	try:
		je = await svc.reverse_journal_entry(
			journal_id=journal_id,
			reason=body.get("reason", "Manual reversal"),
			reversed_by=request.headers.get("X-Actor-Id", "api"),
		)
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404
	return jsonify(je), 201


# ── Reports ───────────────────────────────────────────────────────

@gl_api.get("/reports/trial-balance")
async def trial_balance():
	svc = _svc()
	rows = await svc.get_trial_balance(
		as_of_date=request.args.get("as_of_date"),
		period_id=request.args.get("period_id"),
	)
	return jsonify({"trial_balance": rows})


@gl_api.get("/reports/profit-and-loss")
async def profit_and_loss():
	svc = _svc()
	from_date = request.args.get("from_date", f"{date.today().year}-01-01")
	to_date = request.args.get("to_date", str(date.today()))
	return jsonify(await svc.get_profit_and_loss(from_date, to_date))


@gl_api.get("/reports/balance-sheet")
async def balance_sheet():
	svc = _svc()
	return jsonify(await svc.get_balance_sheet(request.args.get("as_of_date")))


@gl_api.get("/reports/suspense")
async def suspense_check():
	return jsonify(await _svc().check_suspense_accounts())


@gl_api.get("/reports/validate")
async def validate_coa():
	return jsonify(await _svc().validate_coa_balance())
