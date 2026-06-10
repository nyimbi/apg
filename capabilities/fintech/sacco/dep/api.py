"""Flask Blueprint REST API for SACCO Deposits & Savings."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from flask import Blueprint, jsonify, request

from .service import SaccoDepositsService

_log = logging.getLogger(__name__)

bp = Blueprint("sacco_dep", __name__, url_prefix="/api/fintech/sacco/dep")
_svc = SaccoDepositsService()


def _run(coro: Any) -> Any:
	loop = asyncio.new_event_loop()
	try:
		return loop.run_until_complete(coro)
	finally:
		loop.close()


def _tenant() -> str:
	return request.headers.get("X-Tenant-ID", "default")


# ── Health ────────────────────────────────────────────────────────────────────

@bp.get("/health")
def health():
	return jsonify(_run(_svc.health_check())), 200


# ── Products ──────────────────────────────────────────────────────────────────

@bp.get("/products")
def list_products():
	result = _run(_svc.list_products(
		tenant_id=_tenant(),
		product_type=request.args.get("product_type"),
		active_only=request.args.get("active_only", "true").lower() == "true",
	))
	return jsonify({"items": result, "total": len(result)}), 200


@bp.get("/products/<product_id>")
def get_product(product_id: str):
	try:
		return jsonify(_run(_svc.get_product(product_id, tenant_id=_tenant()))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/products")
def create_product():
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.create_product(
			product_code=body["product_code"],
			product_name=body["product_name"],
			product_type=body["product_type"],
			interest_rate_pa=body["interest_rate_pa"],
			tenant_id=_tenant(),
			min_balance=body.get("min_balance", 0.0),
			min_opening_balance=body.get("min_opening_balance", 0.0),
			max_balance=body.get("max_balance"),
			lock_in_months=body.get("lock_in_months", 0),
			interest_posting_frequency=body.get("interest_posting_frequency", "monthly"),
			withdrawal_notice_days=body.get("withdrawal_notice_days", 0),
			allow_overdraft=body.get("allow_overdraft", False),
			tax_exempt=body.get("tax_exempt", False),
			description=body.get("description"),
		))
		return jsonify(result), 201
	except ValueError as exc:
		return jsonify({"error": str(exc)}), 422
	except Exception as exc:
		_log.error("create_product error: %s", exc)
		return jsonify({"error": str(exc)}), 500


@bp.put("/products/<product_id>")
def update_product(product_id: str):
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.update_product(product_id, tenant_id=_tenant(), **body))
		return jsonify(result), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


@bp.delete("/products/<product_id>")
def delete_product(product_id: str):
	try:
		return jsonify(_run(_svc.delete_product(product_id, tenant_id=_tenant()))), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


# ── Accounts ──────────────────────────────────────────────────────────────────

@bp.get("/accounts")
def list_accounts():
	result = _run(_svc.list_accounts(
		tenant_id=_tenant(),
		member_id=request.args.get("member_id"),
		product_id=request.args.get("product_id"),
		status=request.args.get("status"),
	))
	return jsonify({"items": result, "total": len(result)}), 200


@bp.get("/accounts/<account_id>")
def get_account(account_id: str):
	try:
		return jsonify(_run(_svc.get_account(account_id, tenant_id=_tenant()))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.post("/accounts")
def open_account():
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.open_account(
			member_id=body["member_id"],
			product_id=body["product_id"],
			tenant_id=_tenant(),
			opening_balance=body.get("opening_balance", 0.0),
			currency=body.get("currency", "KES"),
			account_name=body.get("account_name"),
			maturity_date=body.get("maturity_date"),
		))
		return jsonify(result), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422
	except Exception as exc:
		_log.error("open_account error: %s", exc)
		return jsonify({"error": str(exc)}), 500


@bp.put("/accounts/<account_id>")
def update_account(account_id: str):
	body = request.get_json(force=True) or {}
	try:
		return jsonify(_run(_svc.update_account(account_id, tenant_id=_tenant(), **body))), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


@bp.delete("/accounts/<account_id>")
def close_account(account_id: str):
	body = request.get_json(force=True) or {}
	try:
		return jsonify(_run(_svc.close_account(account_id, closed_by=body.get("closed_by", "system"), reason=body.get("reason", "member_request"), tenant_id=_tenant()))), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


@bp.get("/accounts/<account_id>/statement")
def account_statement(account_id: str):
	try:
		result = _run(_svc.get_account_statement(
			account_id,
			from_date=request.args.get("from_date"),
			to_date=request.args.get("to_date"),
			tenant_id=_tenant(),
		))
		return jsonify(result), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


@bp.get("/accounts/<account_id>/min-balance")
def check_min_balance(account_id: str):
	try:
		return jsonify(_run(_svc.check_minimum_balance(account_id, tenant_id=_tenant()))), 200
	except KeyError as exc:
		return jsonify({"error": str(exc)}), 404


# ── Transactions ──────────────────────────────────────────────────────────────

@bp.post("/deposits")
def deposit():
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.deposit(
			account_id=body["account_id"],
			amount=body["amount"],
			payment_reference=body["payment_reference"],
			recorded_by=body["recorded_by"],
			tenant_id=_tenant(),
			payment_method=body.get("payment_method", "cash"),
			narration=body.get("narration"),
		))
		return jsonify(result), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


@bp.post("/withdrawals")
def withdraw():
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.withdraw(
			account_id=body["account_id"],
			amount=body["amount"],
			approved_by=body["approved_by"],
			tenant_id=_tenant(),
			payment_method=body.get("payment_method", "cash"),
			narration=body.get("narration"),
			payment_reference=body.get("payment_reference"),
		))
		return jsonify(result), 201
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


@bp.get("/transactions")
def list_transactions():
	result = _run(_svc.list_transactions(
		tenant_id=_tenant(),
		account_id=request.args.get("account_id"),
		member_id=request.args.get("member_id"),
		txn_type=request.args.get("type"),
		from_date=request.args.get("from_date"),
		to_date=request.args.get("to_date"),
	))
	return jsonify({"items": result, "total": len(result)}), 200


# ── Interest ──────────────────────────────────────────────────────────────────

@bp.post("/interest/accrue")
def accrue_interest():
	body = request.get_json(force=True) or {}
	try:
		result = _run(_svc.accrue_interest(
			period_start=body["period_start"],
			period_end=body["period_end"],
			posting_date=body["posting_date"],
			run_by=body["run_by"],
			tenant_id=_tenant(),
			account_ids=body.get("accounts"),
		))
		return jsonify(result), 200
	except (KeyError, ValueError) as exc:
		return jsonify({"error": str(exc)}), 422


@bp.get("/interest/postings")
def list_interest_postings():
	result = _run(_svc.list_interest_postings(
		account_id=request.args.get("account_id"),
		tenant_id=_tenant(),
	))
	return jsonify({"items": result, "total": len(result)}), 200


# ── Summary & Audit ───────────────────────────────────────────────────────────

@bp.get("/summary")
def summary():
	return jsonify(_run(_svc.portfolio_summary(tenant_id=_tenant()))), 200


@bp.get("/audit")
def audit_events():
	result = _run(_svc.get_audit_events(tenant_id=_tenant()))
	return jsonify({"items": result, "total": len(result)}), 200
