"""
Bank Account Management — Flask Blueprint REST API.

url_prefix: /api/fin/acct

All endpoints enforce tenant isolation via X-Tenant-ID header.
Responses use {data, error, meta} envelope.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any

from flask import Blueprint, jsonify, request

try:
	from .service import BankAccountService
	from .views import (
		OpenAccountView, CloseAccountView, FreezeAccountView,
		UnfreezeAccountView, CreditView, DebitView, TransferView,
		LockFundsView, ReleaseLockView, SetOverdraftView,
		BulkCreditView, StatementView, AddSignatoryView, SweepView,
		LinkProductView,
	)
except ImportError:
	from service import BankAccountService  # type: ignore
	from views import (  # type: ignore
		OpenAccountView, CloseAccountView, FreezeAccountView,
		UnfreezeAccountView, CreditView, DebitView, TransferView,
		LockFundsView, ReleaseLockView, SetOverdraftView,
		BulkCreditView, StatementView, AddSignatoryView, SweepView,
		LinkProductView,
	)

bp = Blueprint("fin_acct", __name__, url_prefix="/api/fin/acct")
_svc = BankAccountService()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok(data: Any, status: int = 200, meta: dict[str, Any] | None = None):
	body: dict[str, Any] = {"data": data, "error": None}
	if meta:
		body["meta"] = meta
	return jsonify(body), status


def _err(message: str, status: int = 400, code: str | None = None):
	return jsonify({"data": None, "error": {"message": message, "code": code or "bad_request"}}), status


def _tenant() -> str | None:
	return (
		request.headers.get("X-Tenant-ID")
		or request.args.get("tenant_id")
		or (request.get_json(silent=True) or {}).get("tenant_id")
	)


def _run(coro) -> Any:
	return asyncio.run(coro)


def _json() -> dict[str, Any]:
	return request.get_json(force=True, silent=True) or {}


def _int(name: str, default: int) -> int:
	try:
		return int(request.args.get(name, default))
	except (TypeError, ValueError):
		return default


def _date(name: str):
	from datetime import date
	raw = request.args.get(name)
	if not raw:
		return None
	try:
		return date.fromisoformat(raw)
	except ValueError:
		return None


# ---------------------------------------------------------------------------
# Account lifecycle
# ---------------------------------------------------------------------------

@bp.post("/accounts")
def open_account():
	tid = _tenant()
	if not tid:
		return _err("X-Tenant-ID required", 401, "missing_tenant")
	try:
		body = OpenAccountView(**_json())
		acct = _run(_svc.open_account(
			tid, body.customer_id, body.product_code, body.currency,
			body.account_number, body.opening_deposit, body.metadata,
		))
		return _ok(acct.model_dump(), 201)
	except (ValueError, AssertionError) as e:
		return _err(str(e))
	except Exception as e:
		return _err(str(e), 500, "internal_error")


@bp.get("/accounts")
def list_accounts():
	tid = _tenant()
	if not tid:
		return _err("X-Tenant-ID required", 401)
	try:
		accounts = _run(_svc.list_accounts(
			tid,
			customer_id=request.args.get("customer_id"),
			status=request.args.get("status"),
			account_type=request.args.get("account_type"),
		))
		return _ok([a.model_dump() for a in accounts],
		           meta={"count": len(accounts)})
	except Exception as e:
		return _err(str(e), 500)


@bp.get("/accounts/<account_id>")
def get_account(account_id: str):
	tid = _tenant()
	if not tid:
		return _err("X-Tenant-ID required", 401)
	try:
		acct = _run(_svc.get_account(tid, account_id))
		return _ok(acct.model_dump())
	except KeyError as e:
		return _err(str(e), 404, "not_found")
	except Exception as e:
		return _err(str(e), 500)


@bp.get("/accounts/by-number/<account_number>")
def get_account_by_number(account_number: str):
	tid = _tenant()
	if not tid:
		return _err("X-Tenant-ID required", 401)
	try:
		acct = _run(_svc.get_account_by_number(tid, account_number))
		return _ok(acct.model_dump())
	except KeyError as e:
		return _err(str(e), 404, "not_found")
	except Exception as e:
		return _err(str(e), 500)


@bp.post("/accounts/<account_id>/close")
def close_account(account_id: str):
	tid = _tenant()
	if not tid:
		return _err("X-Tenant-ID required", 401)
	try:
		body = CloseAccountView(**_json())
		acct = _run(_svc.close_account(tid, account_id, body.reason, body.closed_by))
		return _ok(acct.model_dump())
	except ValueError as e:
		return _err(str(e))
	except KeyError as e:
		return _err(str(e), 404)
	except Exception as e:
		return _err(str(e), 500)


@bp.post("/accounts/<account_id>/freeze")
def freeze_account(account_id: str):
	tid = _tenant()
	if not tid:
		return _err("X-Tenant-ID required", 401)
	try:
		body = FreezeAccountView(**_json())
		acct = _run(_svc.freeze_account(tid, account_id, body.reason, body.frozen_by))
		return _ok(acct.model_dump())
	except (ValueError, KeyError) as e:
		return _err(str(e), 400 if isinstance(e, ValueError) else 404)
	except Exception as e:
		return _err(str(e), 500)


@bp.post("/accounts/<account_id>/unfreeze")
def unfreeze_account(account_id: str):
	tid = _tenant()
	if not tid:
		return _err("X-Tenant-ID required", 401)
	try:
		body = UnfreezeAccountView(**_json())
		acct = _run(_svc.unfreeze_account(tid, account_id, body.reason, body.unfrozen_by))
		return _ok(acct.model_dump())
	except (ValueError, KeyError) as e:
		return _err(str(e), 400 if isinstance(e, ValueError) else 404)
	except Exception as e:
		return _err(str(e), 500)


@bp.post("/accounts/<account_id>/dormant")
def mark_dormant(account_id: str):
	tid = _tenant()
	if not tid:
		return _err("X-Tenant-ID required", 401)
	try:
		acct = _run(_svc.mark_dormant(tid, account_id))
		return _ok(acct.model_dump())
	except (ValueError, KeyError) as e:
		return _err(str(e), 400 if isinstance(e, ValueError) else 404)
	except Exception as e:
		return _err(str(e), 500)


@bp.post("/accounts/<account_id>/reactivate")
def reactivate_dormant(account_id: str):
	tid = _tenant()
	if not tid:
		return _err("X-Tenant-ID required", 401)
	try:
		acct = _run(_svc.reactivate_dormant(tid, account_id))
		return _ok(acct.model_dump())
	except (ValueError, KeyError) as e:
		return _err(str(e), 400 if isinstance(e, ValueError) else 404)
	except Exception as e:
		return _err(str(e), 500)


# ---------------------------------------------------------------------------
# Balance
# ---------------------------------------------------------------------------

@bp.get("/accounts/<account_id>/balance")
def get_balance(account_id: str):
	tid = _tenant()
	if not tid:
		return _err("X-Tenant-ID required", 401)
	try:
		bal = _run(_svc.get_balance(tid, account_id))
		return _ok(bal.model_dump())
	except KeyError as e:
		return _err(str(e), 404)
	except Exception as e:
		return _err(str(e), 500)


@bp.get("/accounts/<account_id>/balance/check")
def check_sufficient_funds(account_id: str):
	tid = _tenant()
	if not tid:
		return _err("X-Tenant-ID required", 401)
	try:
		amount = Decimal(request.args.get("amount", "0"))
		ok = _run(_svc.check_sufficient_funds(tid, account_id, amount))
		return _ok({"sufficient": ok, "amount_requested": str(amount)})
	except (ValueError, KeyError) as e:
		return _err(str(e), 400 if isinstance(e, ValueError) else 404)
	except Exception as e:
		return _err(str(e), 500)


# ---------------------------------------------------------------------------
# Transactions
# ---------------------------------------------------------------------------

@bp.post("/accounts/<account_id>/credit")
def credit_account(account_id: str):
	tid = _tenant()
	if not tid:
		return _err("X-Tenant-ID required", 401)
	try:
		body = CreditView(**_json())
		txn = _run(_svc.credit_account(
			tid, account_id, body.amount, body.currency,
			body.reference, body.description, body.transaction_type,
		))
		return _ok(txn.model_dump(), 201)
	except ValueError as e:
		return _err(str(e))
	except KeyError as e:
		return _err(str(e), 404)
	except Exception as e:
		return _err(str(e), 500)


@bp.post("/accounts/<account_id>/debit")
def debit_account(account_id: str):
	tid = _tenant()
	if not tid:
		return _err("X-Tenant-ID required", 401)
	try:
		body = DebitView(**_json())
		txn = _run(_svc.debit_account(
			tid, account_id, body.amount, body.currency,
			body.reference, body.description, body.transaction_type,
		))
		return _ok(txn.model_dump(), 201)
	except ValueError as e:
		return _err(str(e))
	except KeyError as e:
		return _err(str(e), 404)
	except Exception as e:
		return _err(str(e), 500)


@bp.post("/accounts/<account_id>/transfer")
def transfer_internal(account_id: str):
	tid = _tenant()
	if not tid:
		return _err("X-Tenant-ID required", 401)
	try:
		body = TransferView(**_json())
		debit_txn, credit_txn = _run(_svc.transfer_internal(
			tid, account_id, body.to_account_id, body.amount,
			body.reference, body.description,
		))
		return _ok({"debit": debit_txn.model_dump(), "credit": credit_txn.model_dump()}, 201)
	except ValueError as e:
		return _err(str(e))
	except KeyError as e:
		return _err(str(e), 404)
	except Exception as e:
		return _err(str(e), 500)


@bp.get("/accounts/<account_id>/transactions")
def get_transactions(account_id: str):
	tid = _tenant()
	if not tid:
		return _err("X-Tenant-ID required", 401)
	try:
		txns = _run(_svc.get_transactions(
			tid, account_id,
			from_date=_date("from_date"),
			to_date=_date("to_date"),
			limit=_int("limit", 50),
			page=_int("page", 1),
		))
		return _ok([t.model_dump() for t in txns], meta={"count": len(txns)})
	except (ValueError, KeyError) as e:
		return _err(str(e), 400 if isinstance(e, ValueError) else 404)
	except Exception as e:
		return _err(str(e), 500)


@bp.get("/transactions/<transaction_id>")
def get_transaction(transaction_id: str):
	tid = _tenant()
	if not tid:
		return _err("X-Tenant-ID required", 401)
	try:
		txn = _run(_svc.get_transaction(tid, transaction_id))
		return _ok(txn.model_dump())
	except KeyError as e:
		return _err(str(e), 404)
	except Exception as e:
		return _err(str(e), 500)


@bp.post("/accounts/<account_id>/statement")
def generate_statement(account_id: str):
	tid = _tenant()
	if not tid:
		return _err("X-Tenant-ID required", 401)
	try:
		body = StatementView(**_json())
		stmt = _run(_svc.generate_statement(
			tid, account_id, body.from_date, body.to_date, body.format.value,
		))
		return _ok(stmt)
	except (ValueError, KeyError) as e:
		return _err(str(e), 400 if isinstance(e, ValueError) else 404)
	except Exception as e:
		return _err(str(e), 500)


# ---------------------------------------------------------------------------
# Fund locks
# ---------------------------------------------------------------------------

@bp.post("/accounts/<account_id>/locks")
def lock_funds(account_id: str):
	tid = _tenant()
	if not tid:
		return _err("X-Tenant-ID required", 401)
	try:
		body = LockFundsView(**_json())
		lock = _run(_svc.lock_funds(tid, account_id, body.amount, body.lock_reference, body.reason, body.expires_at))
		return _ok(lock.model_dump(), 201)
	except (ValueError, KeyError) as e:
		return _err(str(e), 400 if isinstance(e, ValueError) else 404)
	except Exception as e:
		return _err(str(e), 500)


@bp.post("/accounts/<account_id>/locks/release")
def release_lock(account_id: str):
	tid = _tenant()
	if not tid:
		return _err("X-Tenant-ID required", 401)
	try:
		body = ReleaseLockView(**_json())
		lock = _run(_svc.release_lock(tid, account_id, body.lock_reference))
		return _ok(lock.model_dump())
	except (ValueError, KeyError) as e:
		return _err(str(e), 400 if isinstance(e, ValueError) else 404)
	except Exception as e:
		return _err(str(e), 500)


# ---------------------------------------------------------------------------
# Overdraft
# ---------------------------------------------------------------------------

@bp.put("/accounts/<account_id>/overdraft")
def set_overdraft_limit(account_id: str):
	tid = _tenant()
	if not tid:
		return _err("X-Tenant-ID required", 401)
	try:
		from .views import SetOverdraftView
		body = SetOverdraftView(**_json())
		acct = _run(_svc.set_overdraft_limit(tid, account_id, body.limit, body.approved_by))
		return _ok(acct.model_dump())
	except (ValueError, KeyError) as e:
		return _err(str(e), 400 if isinstance(e, ValueError) else 404)
	except Exception as e:
		return _err(str(e), 500)


# ---------------------------------------------------------------------------
# Product
# ---------------------------------------------------------------------------

@bp.get("/accounts/<account_id>/product")
def get_account_product(account_id: str):
	tid = _tenant()
	if not tid:
		return _err("X-Tenant-ID required", 401)
	try:
		prod = _run(_svc.get_account_product(tid, account_id))
		return _ok(prod.model_dump())
	except KeyError as e:
		return _err(str(e), 404)
	except Exception as e:
		return _err(str(e), 500)


@bp.put("/accounts/<account_id>/product")
def link_product(account_id: str):
	tid = _tenant()
	if not tid:
		return _err("X-Tenant-ID required", 401)
	try:
		body = LinkProductView(**_json())
		acct = _run(_svc.link_product(tid, account_id, body.product_code))
		return _ok(acct.model_dump())
	except (ValueError, KeyError) as e:
		return _err(str(e), 400 if isinstance(e, ValueError) else 404)
	except Exception as e:
		return _err(str(e), 500)


# ---------------------------------------------------------------------------
# Dormancy
# ---------------------------------------------------------------------------

@bp.get("/dormancy-candidates")
def get_dormancy_candidates():
	tid = _tenant()
	if not tid:
		return _err("X-Tenant-ID required", 401)
	try:
		days = _int("days_inactive", 180)
		accounts = _run(_svc.get_dormancy_candidates(tid, days))
		return _ok([a.model_dump() for a in accounts], meta={"count": len(accounts)})
	except Exception as e:
		return _err(str(e), 500)


# ---------------------------------------------------------------------------
# Stats, bulk, sweep
# ---------------------------------------------------------------------------

@bp.get("/stats/<customer_id>")
def get_account_stats(customer_id: str):
	tid = _tenant()
	if not tid:
		return _err("X-Tenant-ID required", 401)
	try:
		stats = _run(_svc.get_account_stats(tid, customer_id))
		return _ok(stats.model_dump())
	except Exception as e:
		return _err(str(e), 500)


@bp.post("/bulk-credit")
def bulk_credit():
	tid = _tenant()
	if not tid:
		return _err("X-Tenant-ID required", 401)
	try:
		body = BulkCreditView(**_json())
		result = _run(_svc.bulk_credit(tid, [c.model_dump() for c in body.credits]))
		return _ok(result.model_dump())
	except ValueError as e:
		return _err(str(e))
	except Exception as e:
		return _err(str(e), 500)


@bp.post("/accounts/<account_id>/sweep")
def sweep_to_linked(account_id: str):
	tid = _tenant()
	if not tid:
		return _err("X-Tenant-ID required", 401)
	try:
		body = SweepView(**_json())
		txn = _run(_svc.sweep_to_linked(
			tid, account_id, body.linked_account_id,
			body.sweep_threshold, body.retain_amount,
		))
		return _ok(txn.model_dump() if txn else None,
		           meta={"swept": txn is not None})
	except (ValueError, KeyError) as e:
		return _err(str(e), 400 if isinstance(e, ValueError) else 404)
	except Exception as e:
		return _err(str(e), 500)


# ---------------------------------------------------------------------------
# Signatories
# ---------------------------------------------------------------------------

@bp.post("/accounts/<account_id>/signatories")
def add_joint_holder(account_id: str):
	tid = _tenant()
	if not tid:
		return _err("X-Tenant-ID required", 401)
	try:
		body = AddSignatoryView(**_json())
		sig = _run(_svc.add_joint_holder(tid, account_id, body.customer_id, body.signing_authority.value))
		return _ok(sig.model_dump(), 201)
	except (ValueError, KeyError) as e:
		return _err(str(e), 400 if isinstance(e, ValueError) else 404)
	except Exception as e:
		return _err(str(e), 500)


@bp.get("/accounts/<account_id>/signatories")
def get_account_signatories(account_id: str):
	tid = _tenant()
	if not tid:
		return _err("X-Tenant-ID required", 401)
	try:
		sigs = _run(_svc.get_account_signatories(tid, account_id))
		return _ok([s.model_dump() for s in sigs])
	except (ValueError, KeyError) as e:
		return _err(str(e), 400 if isinstance(e, ValueError) else 404)
	except Exception as e:
		return _err(str(e), 500)


# ---------------------------------------------------------------------------
# History & summary
# ---------------------------------------------------------------------------

@bp.get("/accounts/<account_id>/history")
def get_account_history(account_id: str):
	tid = _tenant()
	if not tid:
		return _err("X-Tenant-ID required", 401)
	try:
		history = _run(_svc.get_account_history(tid, account_id))
		return _ok([h.model_dump() for h in history])
	except KeyError as e:
		return _err(str(e), 404)
	except Exception as e:
		return _err(str(e), 500)


@bp.get("/accounts/<account_id>/summary/<period>")
def get_transaction_summary(account_id: str, period: str):
	tid = _tenant()
	if not tid:
		return _err("X-Tenant-ID required", 401)
	try:
		summary = _run(_svc.get_transaction_summary(tid, account_id, period))
		return _ok(summary.model_dump())
	except (ValueError, KeyError) as e:
		return _err(str(e), 400 if isinstance(e, ValueError) else 404)
	except Exception as e:
		return _err(str(e), 500)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@bp.get("/health")
def health_check():
	try:
		result = _run(_svc.health_check())
		return _ok(result)
	except Exception as e:
		return _err(str(e), 500)
