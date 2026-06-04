"""REST API Blueprint for APG Point of Sale.

All endpoints are prefixed /retail-pos/api/v1.
Authentication: X-Tenant-ID header (APG platform injects g.tenant_id in production).
All responses are JSON. Errors follow {"error": "...", "status": <code>} convention.
"""
from __future__ import annotations

import asyncio
from typing import Any

from flask import Blueprint, g, jsonify, request

from .service import PointOfSaleService
from .capability_contract import get_capability_contract, evaluate_capability_rules

blueprint = Blueprint("retail_pos_api", __name__, url_prefix="/retail-pos/api/v1")
# backward-compat name used by app.py
api = blueprint

_svc: PointOfSaleService | None = None


def _get_svc() -> PointOfSaleService:
	global _svc
	if _svc is None:
		_svc = PointOfSaleService()
	return _svc


def _tid() -> str:
	return getattr(g, "tenant_id", None) or request.headers.get("X-Tenant-ID", "default")


def _run(coro: Any) -> Any:
	return asyncio.run(coro)


def _ok(data: Any, code: int = 200) -> Any:
	return jsonify(data), code


def _err(msg: str, code: int = 400) -> Any:
	return jsonify({"error": msg, "status": code}), code


def _body() -> dict[str, Any]:
	return request.get_json(force=True, silent=True) or {}


# ---------------------------------------------------------------------------
# Capability contract
# ---------------------------------------------------------------------------

@blueprint.get("/contract")
def contract() -> Any:
	"""GET /retail-pos/api/v1/contract — Capability contract."""
	return _ok(get_capability_contract(_tid()))


@blueprint.post("/rules/evaluate")
def evaluate_rules() -> Any:
	"""POST /retail-pos/api/v1/rules/evaluate — Evaluate capability rules."""
	return _ok(evaluate_capability_rules(_body()))


# ===========================================================================
# TERMINALS
# ===========================================================================

@blueprint.get("/terminals")
def list_terminals() -> Any:
	"""GET /terminals?store_id=<id> — List terminals."""
	store_id = request.args.get("store_id")
	items = _run(_get_svc().list_terminals(_tid(), store_id))
	return _ok({"items": items, "count": len(items)})


@blueprint.post("/terminals")
def register_terminal() -> Any:
	"""POST /terminals — Register a new terminal."""
	from .models import PosTerminalCreate
	body = _body()
	body["tenant_id"] = _tid()
	try:
		rec = _run(_get_svc().register_terminal(PosTerminalCreate(**body)))
		return _ok(rec, 201)
	except Exception as exc:
		return _err(str(exc))


@blueprint.get("/terminals/<terminal_id>")
def get_terminal(terminal_id: str) -> Any:
	"""GET /terminals/<id> — Get terminal details."""
	rec = _run(_get_svc().get_terminal(_tid(), terminal_id))
	return _ok(rec) if rec else _err("not_found", 404)


@blueprint.put("/terminals/<terminal_id>")
def update_terminal(terminal_id: str) -> Any:
	"""PUT /terminals/<id> — Update terminal (status, floor_limit, etc.)."""
	from .models import PosTerminalUpdate
	body = _body()
	body.setdefault("updated_by", "api")
	try:
		svc = _get_svc()
		rec = svc._terminals.get(terminal_id)
		if not rec or rec["tenant_id"] != _tid():
			return _err("not_found", 404)
		allowed = {"status", "floor_limit", "hardware_model", "offline_capable", "tax_profile_id", "terminal_type"}
		for k, v in body.items():
			if k in allowed and v is not None:
				rec[k] = v
		svc._terminals[terminal_id] = rec
		return _ok(rec)
	except Exception as exc:
		return _err(str(exc))


@blueprint.delete("/terminals/<terminal_id>")
def delete_terminal(terminal_id: str) -> Any:
	"""DELETE /terminals/<id> — Soft-delete terminal."""
	svc = _get_svc()
	rec = svc._terminals.get(terminal_id)
	if not rec or rec["tenant_id"] != _tid():
		return _err("not_found", 404)
	rec["is_deleted"] = True
	svc._terminals[terminal_id] = rec
	return _ok({"id": terminal_id, "is_deleted": True})


@blueprint.post("/terminals/<terminal_id>/heartbeat")
def terminal_heartbeat(terminal_id: str) -> Any:
	"""POST /terminals/<id>/heartbeat — Mark terminal online."""
	rec = _run(_get_svc().heartbeat_terminal(_tid(), terminal_id))
	return _ok(rec) if rec else _err("not_found", 404)


@blueprint.post("/terminals/<terminal_id>/offline")
def terminal_offline(terminal_id: str) -> Any:
	"""POST /terminals/<id>/offline — Mark terminal offline."""
	rec = _run(_get_svc().mark_terminal_offline(_tid(), terminal_id))
	return _ok(rec) if rec else _err("not_found", 404)


# ===========================================================================
# SESSIONS
# ===========================================================================

@blueprint.get("/sessions")
def list_sessions() -> Any:
	"""GET /sessions?store_id=<id>&status=<s> — List sessions."""
	items = _run(_get_svc().list_sessions(
		_tid(),
		request.args.get("store_id"),
		request.args.get("status"),
	))
	return _ok({"items": items, "count": len(items)})


@blueprint.post("/sessions")
def open_session() -> Any:
	"""POST /sessions — Open a new cashier session."""
	body = _body()
	try:
		rec = _run(_get_svc().open_session(
			terminal_id=body["terminal_id"],
			cashier_id=body["cashier_id"],
			opening_float=float(body.get("opening_float", 0)),
			tenant_id=_tid(),
			store_id=body.get("store_id", "default"),
			supervisor_id=body.get("supervisor_id"),
			created_by=body.get("created_by", "api"),
		))
		return _ok(rec, 201)
	except Exception as exc:
		return _err(str(exc))


@blueprint.get("/sessions/<session_id>")
def get_session(session_id: str) -> Any:
	"""GET /sessions/<id> — Get session details."""
	rec = _run(_get_svc().get_session(session_id, tenant_id=_tid()))
	return _ok(rec) if rec else _err("not_found", 404)


@blueprint.get("/sessions/<session_id>/summary")
def session_summary(session_id: str) -> Any:
	"""GET /sessions/<id>/summary — Full session summary with totals."""
	summary = _run(_get_svc().session_summary(session_id, tenant_id=_tid()))
	return _ok(summary) if summary else _err("not_found", 404)


@blueprint.post("/sessions/<session_id>/close")
def close_session(session_id: str) -> Any:
	"""POST /sessions/<id>/close — Close session with cash count."""
	body = _body()
	try:
		rec = _run(_get_svc().close_session(
			session_id=session_id,
			closing_float=float(body.get("closing_cash", 0)),
			closing_notes=body.get("notes"),
			tenant_id=_tid(),
		))
		return _ok(rec)
	except Exception as exc:
		return _err(str(exc))


@blueprint.post("/sessions/<session_id>/suspend")
def suspend_session(session_id: str) -> Any:
	"""POST /sessions/<id>/suspend — Suspend session."""
	rec = _run(_get_svc().suspend_session(_tid(), session_id))
	return _ok(rec) if rec else _err("not_found", 404)


@blueprint.post("/sessions/<session_id>/resume")
def resume_session(session_id: str) -> Any:
	"""POST /sessions/<id>/resume — Resume suspended session."""
	rec = _run(_get_svc().resume_session(_tid(), session_id))
	return _ok(rec) if rec else _err("not_found", 404)


# ===========================================================================
# TRANSACTIONS
# ===========================================================================

@blueprint.get("/transactions")
def list_transactions() -> Any:
	"""GET /transactions?session_id=<id>&status=<s>&page=<n>&page_size=<n> — List transactions."""
	session_id = request.args.get("session_id")
	page = int(request.args.get("page", 1))
	page_size = int(request.args.get("page_size", 50))
	items = _run(_get_svc().list_transactions(_tid(), session_id))
	# Simple pagination
	total = len(items)
	start = (page - 1) * page_size
	paginated = items[start: start + page_size]
	return _ok({
		"items": paginated,
		"total": total,
		"page": page,
		"page_size": page_size,
		"pages": max(1, -(-total // page_size)),
	})


@blueprint.post("/transactions")
def begin_transaction() -> Any:
	"""POST /transactions — Begin a new transaction basket."""
	body = _body()
	try:
		rec = _run(_get_svc().begin_transaction(
			session_id=body["session_id"],
			customer_id=body.get("customer_id"),
			tenant_id=_tid(),
			cashier_id=body.get("cashier_id", "api"),
			created_by=body.get("created_by", "api"),
		))
		return _ok(rec, 201)
	except Exception as exc:
		return _err(str(exc))


@blueprint.get("/transactions/<txn_id>")
def get_transaction(txn_id: str) -> Any:
	"""GET /transactions/<id> — Get transaction details."""
	rec = _run(_get_svc().get_transaction(txn_id, tenant_id=_tid()))
	return _ok(rec) if rec else _err("not_found", 404)


@blueprint.put("/transactions/<txn_id>")
def update_transaction(txn_id: str) -> Any:
	"""PUT /transactions/<id> — Update transaction (notes, customer_id)."""
	body = _body()
	try:
		svc = _get_svc()
		obj = svc._store_transactions.get_item(_tid(), txn_id)
		if obj is None:
			return _err("not_found", 404)
		data = obj.model_dump()
		if "notes" in body:
			data["notes"] = body["notes"]
		if "customer_id" in body:
			data["customer_id"] = body["customer_id"]
		from .models import SaleTransactionResponse
		svc._store_transactions.put(_tid(), txn_id, SaleTransactionResponse(**data))
		return _ok(data)
	except Exception as exc:
		return _err(str(exc))


@blueprint.delete("/transactions/<txn_id>")
def void_transaction(txn_id: str) -> Any:
	"""DELETE /transactions/<id> — Void transaction (supervisor required)."""
	body = _body()
	try:
		rec = _run(_get_svc().void_transaction(
			transaction_id=txn_id,
			reason=body.get("reason", ""),
			supervisor_id=body.get("supervisor_id", ""),
			tenant_id=_tid(),
		))
		return _ok(rec)
	except Exception as exc:
		return _err(str(exc))


@blueprint.post("/transactions/<txn_id>/items")
def add_item(txn_id: str) -> Any:
	"""POST /transactions/<id>/items — Add item to transaction."""
	body = _body()
	try:
		rec = _run(_get_svc().add_item(
			transaction_id=txn_id,
			sku=body["sku"],
			quantity=float(body["quantity"]),
			price_override=body.get("price_override"),
			tenant_id=_tid(),
			description=body.get("description"),
			tax_rate=body.get("tax_rate"),
			created_by=body.get("created_by", "api"),
		))
		return _ok(rec, 201)
	except Exception as exc:
		return _err(str(exc))


@blueprint.delete("/transactions/<txn_id>/items/<sku>")
def remove_item(txn_id: str, sku: str) -> Any:
	"""DELETE /transactions/<id>/items/<sku> — Remove item line."""
	try:
		rec = _run(_get_svc().remove_item(
			transaction_id=txn_id,
			line_id=sku,
			tenant_id=_tid(),
		))
		return _ok(rec)
	except Exception as exc:
		return _err(str(exc))


@blueprint.post("/transactions/<txn_id>/discount")
def apply_discount(txn_id: str) -> Any:
	"""POST /transactions/<id>/discount — Apply discount."""
	body = _body()
	try:
		rec = _run(_get_svc().apply_discount(
			transaction_id=txn_id,
			discount_type=body["discount_type"],
			value=float(body.get("value", 0)),
			approved_by=body.get("approved_by"),
			tenant_id=_tid(),
			coupon_code=body.get("coupon_code"),
		))
		return _ok(rec)
	except Exception as exc:
		return _err(str(exc))


@blueprint.post("/transactions/<txn_id>/pay")
def split_payment(txn_id: str) -> Any:
	"""POST /transactions/<id>/pay — Record payment split(s)."""
	body = _body()
	try:
		rec = _run(_get_svc().split_payment(
			transaction_id=txn_id,
			payment_splits=body.get("payments", []),
			tenant_id=_tid(),
			created_by=body.get("created_by", "api"),
		))
		return _ok(rec)
	except Exception as exc:
		return _err(str(exc))


@blueprint.post("/transactions/<txn_id>/complete")
def complete_transaction(txn_id: str) -> Any:
	"""POST /transactions/<id>/complete — Complete transaction and issue receipt."""
	body = _body()
	try:
		rec = _run(_get_svc().complete_transaction(
			transaction_id=txn_id,
			payments=body.get("payments"),
			tenant_id=_tid(),
			created_by=body.get("created_by", "api"),
		))
		return _ok(rec)
	except Exception as exc:
		return _err(str(exc))


@blueprint.post("/transactions/<txn_id>/park")
def park_transaction(txn_id: str) -> Any:
	"""POST /transactions/<id>/park — Suspend transaction for later retrieval."""
	try:
		rec = _run(_get_svc().park_transaction(txn_id, tenant_id=_tid()))
		return _ok(rec)
	except Exception as exc:
		return _err(str(exc))


@blueprint.post("/transactions/<txn_id>/retrieve")
def retrieve_parked(txn_id: str) -> Any:
	"""POST /transactions/<id>/retrieve — Retrieve parked transaction."""
	try:
		rec = _run(_get_svc().retrieve_parked_transaction(txn_id, tenant_id=_tid()))
		return _ok(rec)
	except Exception as exc:
		return _err(str(exc))


@blueprint.post("/transactions/<txn_id>/void")
def void_transaction_post(txn_id: str) -> Any:
	"""POST /transactions/<id>/void — Void transaction (alternate endpoint)."""
	body = _body()
	try:
		rec = _run(_get_svc().void_transaction(
			transaction_id=txn_id,
			reason=body.get("reason", ""),
			supervisor_id=body.get("supervisor_id", ""),
			tenant_id=_tid(),
		))
		return _ok(rec)
	except Exception as exc:
		return _err(str(exc))


# ===========================================================================
# PROCESS SALE (convenience: all-in-one)
# ===========================================================================

@blueprint.post("/process-sale")
def process_sale() -> Any:
	"""POST /process-sale — Full sale in one call: items + payments → complete.

	Body: {session_id, cashier_id, items: [{sku, qty, price?}], payments: [{method, amount}], customer_id?}
	"""
	body = _body()
	try:
		svc = _get_svc()
		tid = _tid()
		# 1. Begin transaction
		txn = _run(svc.begin_transaction(
			session_id=body["session_id"],
			customer_id=body.get("customer_id"),
			tenant_id=tid,
			cashier_id=body.get("cashier_id", "api"),
			created_by=body.get("created_by", "api"),
		))
		txn_id = txn["id"]
		# 2. Add items
		for item in body.get("items", []):
			_run(svc.add_item(
				transaction_id=txn_id,
				sku=item["sku"],
				quantity=float(item.get("quantity", item.get("qty", 1))),
				price_override=item.get("price"),
				tenant_id=tid,
				description=item.get("description"),
				created_by=body.get("created_by", "api"),
			))
		# 3. Apply discount if provided
		if body.get("discount"):
			d = body["discount"]
			_run(svc.apply_discount(
				transaction_id=txn_id,
				discount_type=d["type"],
				value=float(d.get("value", 0)),
				approved_by=d.get("approved_by"),
				tenant_id=tid,
				coupon_code=d.get("coupon_code"),
			))
		# 4. Complete
		completed = _run(svc.complete_transaction(
			transaction_id=txn_id,
			payments=body.get("payments", []),
			tenant_id=tid,
			created_by=body.get("created_by", "api"),
		))
		return _ok(completed, 201)
	except Exception as exc:
		return _err(str(exc))


# ===========================================================================
# REFUNDS
# ===========================================================================

@blueprint.get("/refunds")
def list_refunds() -> Any:
	"""GET /refunds?session_id=<id> — List refunds."""
	svc = _get_svc()
	tid = _tid()
	session_id = request.args.get("session_id")
	items = [
		r.model_dump(mode="json")
		for r in svc._store_refunds.tenant_values(tid)
		if not session_id or r.session_id == session_id
	]
	return _ok({"items": items, "count": len(items)})


@blueprint.post("/refunds")
def process_refund() -> Any:
	"""POST /refunds — Process a refund against a completed transaction."""
	from .models import RefundCreate, SaleItemCreate, RefundReason
	body = _body()
	body["tenant_id"] = _tid()
	try:
		items_raw = body.pop("items", [])
		items = [SaleItemCreate(**i) for i in items_raw]
		reason_str = body.pop("reason", "other")
		reason = RefundReason(reason_str)
		rec = _run(_get_svc().process_refund(RefundCreate(
			**body,
			items=items,
			reason=reason,
		)))
		return _ok(rec, 201)
	except Exception as exc:
		return _err(str(exc))


@blueprint.get("/refunds/<refund_id>")
def get_refund(refund_id: str) -> Any:
	"""GET /refunds/<id> — Get refund details."""
	svc = _get_svc()
	obj = svc._store_refunds.get_item(_tid(), refund_id)
	return _ok(obj.model_dump(mode="json")) if obj else _err("not_found", 404)


# ===========================================================================
# PAYMENTS
# ===========================================================================

@blueprint.post("/transactions/<txn_id>/payments/cash")
def cash_payment(txn_id: str) -> Any:
	"""POST /transactions/<id>/payments/cash — Process cash tender."""
	body = _body()
	try:
		rec = _run(_get_svc().process_cash_payment(
			transaction_id=txn_id,
			amount_tendered=float(body["amount_tendered"]),
			tenant_id=_tid(),
			created_by=body.get("created_by", "api"),
		))
		return _ok(rec)
	except Exception as exc:
		return _err(str(exc))


@blueprint.post("/transactions/<txn_id>/payments/card")
def card_payment(txn_id: str) -> Any:
	"""POST /transactions/<id>/payments/card — Process card payment."""
	body = _body()
	try:
		rec = _run(_get_svc().process_card_payment(
			transaction_id=txn_id,
			amount=float(body["amount"]),
			card_type=body.get("card_type", "debit"),
			auth_code=body.get("auth_code", ""),
			tenant_id=_tid(),
			terminal_ref=body.get("terminal_ref"),
			created_by=body.get("created_by", "api"),
		))
		return _ok(rec)
	except Exception as exc:
		return _err(str(exc))


@blueprint.post("/transactions/<txn_id>/payments/mpesa")
def mpesa_payment(txn_id: str) -> Any:
	"""POST /transactions/<id>/payments/mpesa — Record M-Pesa confirmation."""
	body = _body()
	try:
		rec = _run(_get_svc().process_mpesa_payment(
			transaction_id=txn_id,
			phone=body.get("phone", ""),
			amount=float(body["amount"]),
			mpesa_ref=body.get("mpesa_ref", ""),
			tenant_id=_tid(),
			created_by=body.get("created_by", "api"),
		))
		return _ok(rec)
	except Exception as exc:
		return _err(str(exc))


# ===========================================================================
# DISCOUNTS
# ===========================================================================

@blueprint.get("/discounts")
def list_discounts() -> Any:
	"""GET /discounts?active=true — List discount catalogue."""
	svc = _get_svc()
	tid = _tid()
	active_only = request.args.get("active", "").lower() == "true"
	items = [
		d.model_dump(mode="json")
		for d in svc._store_discounts.tenant_values(tid)
		if not active_only or d.is_active
	]
	return _ok({"items": items, "count": len(items)})


@blueprint.post("/discounts")
def create_discount() -> Any:
	"""POST /discounts — Create a discount definition."""
	from .models import DiscountCreate
	body = _body()
	body["tenant_id"] = _tid()
	try:
		rec = _run(_get_svc().create_discount(DiscountCreate(**body)))
		return _ok(rec, 201)
	except Exception as exc:
		return _err(str(exc))


@blueprint.get("/discounts/<discount_id>")
def get_discount(discount_id: str) -> Any:
	"""GET /discounts/<id> — Get discount details."""
	obj = _get_svc()._store_discounts.get_item(_tid(), discount_id)
	return _ok(obj.model_dump(mode="json")) if obj else _err("not_found", 404)


@blueprint.put("/discounts/<discount_id>")
def update_discount(discount_id: str) -> Any:
	"""PUT /discounts/<id> — Update discount (activate/deactivate, extend validity)."""
	body = _body()
	svc = _get_svc()
	obj = svc._store_discounts.get_item(_tid(), discount_id)
	if obj is None:
		return _err("not_found", 404)
	data = obj.model_dump()
	for k in ("is_active", "valid_until", "max_uses", "value"):
		if k in body:
			data[k] = body[k]
	from .models import DiscountResponse
	svc._store_discounts.put(_tid(), discount_id, DiscountResponse(**data))
	return _ok(data)


@blueprint.delete("/discounts/<discount_id>")
def delete_discount(discount_id: str) -> Any:
	"""DELETE /discounts/<id> — Soft-delete discount."""
	svc = _get_svc()
	obj = svc._store_discounts.get_item(_tid(), discount_id)
	if obj is None:
		return _err("not_found", 404)
	data = obj.model_dump()
	data["is_deleted"] = True
	data["is_active"] = False
	from .models import DiscountResponse
	svc._store_discounts.put(_tid(), discount_id, DiscountResponse(**data))
	return _ok({"id": discount_id, "is_deleted": True})


# ===========================================================================
# CASH MANAGEMENT
# ===========================================================================

@blueprint.get("/cash")
def list_cash_events() -> Any:
	"""GET /cash?session_id=<id> — List cash events."""
	session_id = request.args.get("session_id", "")
	recs = _run(_get_svc().list_cash_events(_tid(), session_id))
	return _ok({"items": [r.model_dump() for r in recs], "count": len(recs)})


@blueprint.post("/cash")
def record_cash_event() -> Any:
	"""POST /cash — Record cash float event (safe drop, petty cash, etc.)."""
	from .models import CashFloatCreate
	body = _body()
	body["tenant_id"] = _tid()
	try:
		rec = _run(_get_svc().record_cash_float(CashFloatCreate(**body)))
		return _ok(rec, 201)
	except Exception as exc:
		return _err(str(exc))


@blueprint.post("/sessions/<session_id>/reconcile")
def cash_reconciliation(session_id: str) -> Any:
	"""POST /sessions/<id>/reconcile — Submit physical cash count."""
	body = _body()
	try:
		rec = _run(_get_svc().cash_count_reconciliation(
			session_id=session_id,
			counted_cash=float(body["counted_cash"]),
			denominations=body.get("denominations"),
			tenant_id=_tid(),
			counted_by=body.get("counted_by", "api"),
		))
		return _ok(rec)
	except Exception as exc:
		return _err(str(exc))


# ===========================================================================
# LOYALTY
# ===========================================================================

@blueprint.get("/loyalty/<customer_id>")
def loyalty_balance(customer_id: str) -> Any:
	"""GET /loyalty/<customer_id> — Get loyalty points balance."""
	balance = _get_svc()._loyalty.balance(_tid(), customer_id)
	return _ok({
		"customer_id": customer_id,
		"tenant_id": _tid(),
		"points_balance": balance,
		"redemption_value": round(balance * 0.01, 2),
	})


@blueprint.get("/loyalty/<customer_id>/history")
def loyalty_history(customer_id: str) -> Any:
	"""GET /loyalty/<customer_id>/history — Get loyalty transaction history."""
	history = _get_svc()._loyalty.customer_history(_tid(), customer_id)
	return _ok({"customer_id": customer_id, "history": history, "count": len(history)})


@blueprint.post("/loyalty/earn-redeem")
def loyalty_earn_redeem() -> Any:
	"""POST /loyalty/earn-redeem — Earn or redeem loyalty points."""
	body = _body()
	try:
		rec = _run(_get_svc().loyalty_points_earn_redeem(
			customer_id=body["customer_id"],
			transaction_id=body["transaction_id"],
			points_earned=int(body.get("points_earned", 0)),
			points_to_redeem=int(body.get("points_to_redeem", 0)),
			tenant_id=_tid(),
			created_by=body.get("created_by", "api"),
		))
		return _ok(rec)
	except Exception as exc:
		return _err(str(exc))


# ===========================================================================
# RECEIPTS
# ===========================================================================

@blueprint.get("/receipts")
def list_receipts() -> Any:
	"""GET /receipts?transaction_id=<id> — List receipts."""
	transaction_id = request.args.get("transaction_id", "")
	recs = _run(_get_svc().list_receipts(_tid(), transaction_id))
	return _ok({"items": [r.model_dump() for r in recs], "count": len(recs)})


@blueprint.post("/receipts")
def generate_receipt() -> Any:
	"""POST /receipts — Generate receipt for a completed transaction."""
	body = _body()
	try:
		rec = _run(_get_svc().receipt_generation(
			transaction_id=body["transaction_id"],
			fmt=body.get("format", "thermal"),
			tenant_id=_tid(),
			recipient_email=body.get("recipient_email"),
			recipient_mobile=body.get("recipient_mobile"),
			created_by=body.get("created_by", "api"),
		))
		return _ok(rec, 201)
	except Exception as exc:
		return _err(str(exc))


@blueprint.get("/receipts/<receipt_id>")
def get_receipt(receipt_id: str) -> Any:
	"""GET /receipts/<id> — Get receipt."""
	obj = _get_svc()._store_receipts_v2.get_item(_tid(), receipt_id)
	return _ok(obj.model_dump(mode="json")) if obj else _err("not_found", 404)


# ===========================================================================
# PRICE CHECK & INVENTORY
# ===========================================================================

@blueprint.get("/price-check")
def price_check() -> Any:
	"""GET /price-check?sku=<sku>&store_id=<id>&tier=<tier> — Check item price."""
	sku = request.args.get("sku", "")
	if not sku:
		return _err("sku is required")
	result = _run(_get_svc().price_check(
		sku=sku,
		customer_tier=request.args.get("tier"),
		tenant_id=_tid(),
		store_id=request.args.get("store_id"),
	))
	return _ok(result)


@blueprint.get("/stock-check")
def stock_check() -> Any:
	"""GET /stock-check?sku=<sku>&store_id=<id> — Check stock level."""
	sku = request.args.get("sku", "")
	store_id = request.args.get("store_id", "default")
	if not sku:
		return _err("sku is required")
	result = _run(_get_svc().stock_check(sku, store_id, tenant_id=_tid()))
	return _ok(result)


@blueprint.get("/inventory/movements")
def list_movements() -> Any:
	"""GET /inventory/movements?store_id=<id> — List inventory movements."""
	svc = _get_svc()
	tid = _tid()
	store_id = request.args.get("store_id")
	items = [
		m.model_dump(mode="json")
		for m in svc._store_movements.tenant_values(tid)
		if not store_id or m.store_id == store_id
	]
	return _ok({"items": items, "count": len(items)})


@blueprint.get("/inventory/low-stock")
def low_stock_alerts() -> Any:
	"""GET /inventory/low-stock?store_id=<id>&threshold_days=<n> — Low stock alerts."""
	store_id = request.args.get("store_id", "default")
	threshold = int(request.args.get("threshold_days", 7))
	alerts = _run(_get_svc().low_stock_alerts(store_id, threshold, tenant_id=_tid()))
	return _ok({"alerts": alerts, "count": len(alerts)})


# ===========================================================================
# SUPERVISOR OVERRIDES
# ===========================================================================

@blueprint.post("/overrides")
def supervisor_override() -> Any:
	"""POST /overrides — Create a supervisor override."""
	from .models import SupervisorOverrideCreate
	body = _body()
	body["tenant_id"] = _tid()
	try:
		rec = _run(_get_svc().supervisor_override(SupervisorOverrideCreate(**body)))
		return _ok(rec, 201)
	except Exception as exc:
		return _err(str(exc))


@blueprint.get("/overrides")
def list_overrides() -> Any:
	"""GET /overrides?session_id=<id> — List supervisor overrides."""
	svc = _get_svc()
	tid = _tid()
	session_id = request.args.get("session_id")
	items = [
		o.model_dump(mode="json")
		for o in svc._store_supervisor_overrides.tenant_values(tid)
		if not session_id or o.session_id == session_id
	]
	return _ok({"items": items, "count": len(items)})


# ===========================================================================
# PRICE OVERRIDES
# ===========================================================================

@blueprint.post("/price-overrides")
def create_price_override() -> Any:
	"""POST /price-overrides — Create a supervisor-authorised price override."""
	from .models import PriceOverrideCreate
	body = _body()
	body["tenant_id"] = _tid()
	try:
		from .models import PriceOverrideResponse
		rec = PriceOverrideResponse(**body)
		_get_svc()._store_price_overrides.put(_tid(), rec.id, rec)
		return _ok(rec.model_dump(mode="json"), 201)
	except Exception as exc:
		return _err(str(exc))


# ===========================================================================
# OFFLINE SYNC
# ===========================================================================

@blueprint.post("/offline/sync")
def offline_sync() -> Any:
	"""POST /offline/sync — Submit offline transaction batch for sync."""
	from .models import OfflineSyncBatch
	body = _body()
	body["tenant_id"] = _tid()
	try:
		result = _run(_get_svc().offline_mode_sync(
			OfflineSyncBatch(**body),
			tenant_id=_tid(),
		))
		return _ok(result)
	except Exception as exc:
		return _err(str(exc))


# ===========================================================================
# END OF DAY
# ===========================================================================

@blueprint.post("/eod")
def end_of_day() -> Any:
	"""POST /eod — Generate end-of-day report for a store."""
	body = _body()
	try:
		report = _run(_get_svc().end_of_day_closing(
			store_id=body["store_id"],
			business_date=body["business_date"],
			tenant_id=_tid(),
			generated_by=body.get("generated_by", "api"),
			created_by=body.get("created_by", "api"),
		))
		return _ok(report, 201)
	except Exception as exc:
		return _err(str(exc))


@blueprint.get("/eod")
def list_eod_reports() -> Any:
	"""GET /eod?store_id=<id> — List EOD reports."""
	svc = _get_svc()
	tid = _tid()
	store_id = request.args.get("store_id")
	items = [
		r.model_dump(mode="json")
		for r in svc._store_eod_reports.tenant_values(tid)
		if not store_id or r.store_id == store_id
	]
	return _ok({"items": items, "count": len(items)})


@blueprint.get("/eod/<report_id>")
def get_eod_report(report_id: str) -> Any:
	"""GET /eod/<id> — Get EOD report."""
	obj = _get_svc()._store_eod_reports.get_item(_tid(), report_id)
	return _ok(obj.model_dump(mode="json")) if obj else _err("not_found", 404)


@blueprint.post("/eod/<report_id>/approve")
def approve_eod_report(report_id: str) -> Any:
	"""POST /eod/<id>/approve — Approve EOD report."""
	body = _body()
	svc = _get_svc()
	obj = svc._store_eod_reports.get_item(_tid(), report_id)
	if obj is None:
		return _err("not_found", 404)
	data = obj.model_dump()
	data["status"] = "approved"
	data["approved_by"] = body.get("approved_by", "api")
	from .models import EndOfDayReportResponse
	import datetime
	data["approved_at"] = datetime.datetime.utcnow()
	svc._store_eod_reports.put(_tid(), report_id, EndOfDayReportResponse(**data))
	return _ok(data)


# ===========================================================================
# REPORTS
# ===========================================================================

@blueprint.get("/reports/sales-summary")
def sales_summary_report() -> Any:
	"""GET /reports/sales-summary?period=today|week|month|<YYYY-MM-DD> — Sales summary."""
	period = request.args.get("period", "today")
	store_id = request.args.get("store_id")
	try:
		result = _run(_get_svc().sales_summary_report(
			period=period,
			store_id=store_id,
			tenant_id=_tid(),
		))
		return _ok(result)
	except Exception as exc:
		return _err(str(exc))


@blueprint.get("/reports/till-variance")
def till_variance_report() -> Any:
	"""GET /reports/till-variance?period=today — Till variance report."""
	period = request.args.get("period", "today")
	store_id = request.args.get("store_id")
	try:
		result = _run(_get_svc().till_variance_report(
			period=period,
			store_id=store_id,
			tenant_id=_tid(),
		))
		return _ok(result)
	except Exception as exc:
		return _err(str(exc))


@blueprint.get("/reports/hourly")
def hourly_report() -> Any:
	"""GET /reports/hourly?store_id=<id>&date=<YYYY-MM-DD> — Hourly sales breakdown."""
	svc = _get_svc()
	tid = _tid()
	store_id = request.args.get("store_id")
	date_str = request.args.get("date")
	from .domain.calculations import hourly_sales_breakdown
	txns = [
		t.model_dump(mode="json")
		for t in svc._store_transactions.tenant_values(tid)
		if t.transaction_type.value == "sale"
		and t.status.value == "completed"
		and (not store_id or t.store_id == store_id)
	]
	return _ok({"hourly": hourly_sales_breakdown(txns), "count": len(txns)})


@blueprint.get("/reports/top-skus")
def top_skus_report() -> Any:
	"""GET /reports/top-skus?store_id=<id>&top_n=<n> — Top selling SKUs."""
	svc = _get_svc()
	tid = _tid()
	store_id = request.args.get("store_id")
	top_n = int(request.args.get("top_n", 10))
	from .domain.calculations import top_selling_skus
	items: list[dict] = []
	for t in svc._store_transactions.tenant_values(tid):
		if t.transaction_type.value == "sale" and t.status.value == "completed":
			if not store_id or t.store_id == store_id:
				for item in (t.items or []):
					items.append(item if isinstance(item, dict) else item.model_dump())
	return _ok({"top_skus": top_selling_skus(items, top_n)})


@blueprint.get("/reports/eod/<store_id>/<business_date>")
def eod_report_by_date(store_id: str, business_date: str) -> Any:
	"""GET /reports/eod/<store_id>/<YYYY-MM-DD> — Get or generate EOD for date."""
	svc = _get_svc()
	tid = _tid()
	existing = next(
		(r for r in svc._store_eod_reports.tenant_values(tid)
		 if r.store_id == store_id and r.business_date == business_date),
		None,
	)
	if existing:
		return _ok(existing.model_dump(mode="json"))
	return _err(f"no EOD report for {store_id} on {business_date}", 404)
