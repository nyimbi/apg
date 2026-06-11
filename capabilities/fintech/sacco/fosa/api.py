"""Flask Blueprint REST API for SACCO FOSA (Front Office Service Activity)."""
from __future__ import annotations

import asyncio
import logging
from decimal import Decimal
from typing import Any

from flask import Blueprint, jsonify, request

from .service import FOSAService

_log = logging.getLogger(__name__)

bp = Blueprint("sacco_fosa", __name__, url_prefix="/api/fintech/sacco/fosa")
_svc = FOSAService()


def _run(coro: Any) -> Any:
	loop = asyncio.new_event_loop()
	try:
		return loop.run_until_complete(coro)
	finally:
		loop.close()


def _tenant() -> str:
	return request.headers.get("X-Tenant-ID", "default")


def _dec(val: Any) -> Decimal:
	return Decimal(str(val))


# ── Health ────────────────────────────────────────────────────────────────────

@bp.get("/health")
def health():
	return jsonify(_run(_svc.health_check())), 200


# ── Accounts ──────────────────────────────────────────────────────────────────

@bp.post("/accounts")
def open_account():
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.open_fosa_account(
			tenant_id=_tenant(),
			member_id=body["member_id"],
			account_type=body["account_type"],
			opening_balance=_dec(body.get("opening_balance", "0")),
			currency=body.get("currency", "KES"),
			account_name=body.get("account_name"),
			daily_withdrawal_limit=_dec(body.get("daily_withdrawal_limit", "100000")),
			daily_transfer_limit=_dec(body.get("daily_transfer_limit", "200000")),
		))
		return jsonify(result), 201
	except (KeyError, AssertionError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422
	except Exception as exc:
		_log.error("open_account error: %s", exc)
		return jsonify({"error": str(exc)}), 500


@bp.delete("/accounts/<account_id>")
def close_account(account_id: str):
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.close_fosa_account(
			tenant_id=_tenant(),
			account_id=account_id,
			reason=body.get("reason", "member_request"),
			closed_by=body.get("closed_by", "system"),
		))
		return jsonify(result), 200
	except (KeyError, ValueError, AssertionError) as exc:
		return jsonify({"error": str(exc)}), 422


@bp.get("/accounts/<account_id>/balance")
def get_balance(account_id: str):
	try:
		result = _run(_svc.get_account_balance(_tenant(), account_id))
		return jsonify(result), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/accounts/<account_id>/mini-statement")
def mini_statement(account_id: str):
	try:
		n = int(request.args.get("n", 10))
		result = _run(_svc.get_mini_statement(_tenant(), account_id, last_n=n))
		return jsonify({"items": result, "total": len(result)}), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/accounts/<account_id>/statement")
def full_statement(account_id: str):
	from_date = request.args.get("from_date", "2020-01-01")
	to_date = request.args.get("to_date", "2099-12-31")
	try:
		result = _run(_svc.get_full_statement(_tenant(), account_id, from_date, to_date))
		return jsonify(result), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.put("/accounts/<account_id>/limits")
def set_limits(account_id: str):
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.set_daily_limit(
			tenant_id=_tenant(),
			account_id=account_id,
			withdrawal_limit=_dec(body["withdrawal_limit"]),
			transfer_limit=_dec(body["transfer_limit"]),
		))
		return jsonify(result), 200
	except (KeyError, AssertionError) as exc:
		return jsonify({"error": str(exc)}), 422


# ── Deposits ──────────────────────────────────────────────────────────────────

@bp.post("/deposits")
def deposit():
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.deposit(
			tenant_id=_tenant(),
			account_id=body["account_id"],
			amount=_dec(body["amount"]),
			channel=body["channel"],
			reference=body["reference"],
			depositor_name=body.get("depositor_name"),
			narration=body.get("narration"),
			teller_id=body.get("teller_id"),
		))
		return jsonify(result), 201
	except (KeyError, AssertionError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422
	except Exception as exc:
		_log.error("deposit error: %s", exc)
		return jsonify({"error": str(exc)}), 500


# ── Withdrawals ───────────────────────────────────────────────────────────────

@bp.post("/withdrawals")
def withdraw():
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.withdraw(
			tenant_id=_tenant(),
			account_id=body["account_id"],
			amount=_dec(body["amount"]),
			channel=body["channel"],
			reference=body.get("reference"),
			authorized_by=body.get("authorized_by"),
			narration=body.get("narration"),
			teller_id=body.get("teller_id"),
		))
		return jsonify(result), 201
	except (KeyError, AssertionError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422
	except Exception as exc:
		_log.error("withdraw error: %s", exc)
		return jsonify({"error": str(exc)}), 500


# ── BOSA Transfers ────────────────────────────────────────────────────────────

@bp.post("/transfers/to-bosa")
def transfer_to_bosa():
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.transfer_to_bosa(
			tenant_id=_tenant(),
			account_id=body["account_id"],
			amount=_dec(body["amount"]),
			bosa_account_id=body["bosa_account_id"],
			reference=body["reference"],
		))
		return jsonify(result), 201
	except (KeyError, AssertionError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


@bp.post("/transfers/from-bosa")
def transfer_from_bosa():
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.transfer_from_bosa(
			tenant_id=_tenant(),
			account_id=body["account_id"],
			amount=_dec(body["amount"]),
			bosa_account_id=body["bosa_account_id"],
			reference=body["reference"],
			approved_by=body.get("approved_by"),
		))
		return jsonify(result), 201
	except (KeyError, AssertionError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


# ── M-PESA ────────────────────────────────────────────────────────────────────

@bp.post("/mpesa/cash-in")
def mpesa_cash_in():
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.mpesa_cash_in(
			tenant_id=_tenant(),
			account_id=body["account_id"],
			mpesa_reference=body["mpesa_reference"],
			amount=_dec(body["amount"]),
			phone_number=body["phone_number"],
		))
		return jsonify(result), 201
	except (KeyError, AssertionError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


@bp.post("/mpesa/cash-out")
def mpesa_cash_out():
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.mpesa_cash_out(
			tenant_id=_tenant(),
			account_id=body["account_id"],
			amount=_dec(body["amount"]),
			phone_number=body["phone_number"],
			mpesa_reference=body.get("mpesa_reference"),
		))
		return jsonify(result), 201
	except (KeyError, AssertionError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


# ── ATM Cards ─────────────────────────────────────────────────────────────────

@bp.post("/cards")
def issue_card():
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.issue_atm_card(
			tenant_id=_tenant(),
			member_id=body["member_id"],
			account_id=body["account_id"],
			card_type=body["card_type"],
			card_name=body.get("card_name"),
		))
		return jsonify(result), 201
	except (KeyError, AssertionError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


@bp.post("/cards/<card_id>/block")
def block_card(card_id: str):
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.block_atm_card(
			tenant_id=_tenant(),
			card_id=card_id,
			reason=body.get("reason", "member_request"),
		))
		return jsonify(result), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


@bp.post("/cards/<card_id>/unblock")
def unblock_card(card_id: str):
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.unblock_atm_card(
			tenant_id=_tenant(),
			card_id=card_id,
			authorized_by=body.get("authorized_by", "system"),
		))
		return jsonify(result), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


# ── Standing Orders ───────────────────────────────────────────────────────────

@bp.post("/standing-orders")
def create_standing_order():
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.create_standing_order(
			tenant_id=_tenant(),
			account_id=body["account_id"],
			beneficiary_account=body["beneficiary_account"],
			amount=_dec(body["amount"]),
			frequency=body["frequency"],
			start_date=body["start_date"],
			end_date=body.get("end_date"),
			beneficiary_name=body.get("beneficiary_name"),
			narration=body.get("narration"),
		))
		return jsonify(result), 201
	except (KeyError, AssertionError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


@bp.delete("/standing-orders/<so_id>")
def cancel_standing_order(so_id: str):
	try:
		result = _run(_svc.cancel_standing_order(_tenant(), so_id))
		return jsonify(result), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


@bp.get("/accounts/<account_id>/standing-orders")
def get_standing_orders(account_id: str):
	try:
		result = _run(_svc.get_standing_orders(_tenant(), account_id))
		return jsonify({"items": result, "total": len(result)}), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/standing-orders/process")
def process_standing_orders():
	body = request.get_json(force=True) or {}
	from datetime import date
	processing_date = body.get("processing_date", date.today().isoformat())
	result = _run(_svc.process_standing_orders(_tenant(), processing_date))
	return jsonify(result), 200


# ── Overdrafts ────────────────────────────────────────────────────────────────

@bp.post("/overdrafts/request")
def request_overdraft():
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.request_overdraft(
			tenant_id=_tenant(),
			account_id=body["account_id"],
			requested_amount=_dec(body["requested_amount"]),
			purpose=body["purpose"],
		))
		return jsonify(result), 201
	except (KeyError, AssertionError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


@bp.post("/overdrafts/approve")
def approve_overdraft():
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.approve_overdraft(
			tenant_id=_tenant(),
			account_id=body["account_id"],
			approved_amount=_dec(body["approved_amount"]),
			approved_by=body["approved_by"],
			expiry_date=body["expiry_date"],
		))
		return jsonify(result), 200
	except (KeyError, AssertionError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


# ── Portfolio & Reporting ─────────────────────────────────────────────────────

@bp.get("/portfolio")
def portfolio():
	return jsonify(_run(_svc.get_fosa_portfolio(_tenant()))), 200


@bp.get("/dormant-accounts")
def dormant_accounts():
	result = _run(_svc.get_dormant_fosa_accounts(_tenant()))
	return jsonify({"items": result, "total": len(result)}), 200


@bp.post("/accounts/<account_id>/reactivate")
def reactivate_account(account_id: str):
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.reactivate_fosa_account(
			tenant_id=_tenant(),
			account_id=account_id,
			reactivation_deposit=_dec(body.get("reactivation_deposit", "500")),
		))
		return jsonify(result), 200
	except (KeyError, AssertionError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


@bp.get("/teller/<teller_id>/summary")
def teller_summary(teller_id: str):
	from datetime import date as _date
	d = request.args.get("date", _date.today().isoformat())
	result = _run(_svc.get_teller_summary(_tenant(), teller_id, d))
	return jsonify(result), 200


@bp.get("/accounts/<account_id>/interest")
def interest_earned(account_id: str):
	period_id = request.args.get("period_id", "")
	try:
		amount = _run(_svc.get_interest_earned(_tenant(), account_id, period_id))
		return jsonify({"account_id": account_id, "period_id": period_id, "interest_earned": str(amount)}), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/audit")
def audit_events():
	result = _run(_svc.get_audit_events(_tenant()))
	return jsonify({"items": result, "total": len(result)}), 200
