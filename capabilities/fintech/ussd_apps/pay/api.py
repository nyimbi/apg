"""Flask Blueprint — Payment USSD App REST API."""
from __future__ import annotations

import logging
from decimal import Decimal, InvalidOperation
from typing import Any

from flask import Blueprint, jsonify, request

from .models import (
	PayBillCreate,
	PayMerchantCreate,
	PayAirtimeCreate,
	PayUtilityCreate,
	PaySendMoneyCreate,
	PaySendMoneyConfirmation,
	PayBillerCreate,
	PayUssdSessionCreate,
)
from .service import PayUssdService

_log = logging.getLogger(__name__)

pay_bp = Blueprint("fintech_ussd_pay", __name__, url_prefix="/api/fintech/ussd/pay")

# Service singleton (replace with DI in production)
_svc = PayUssdService()


def _run(coro: Any) -> Any:
	"""Run async coroutine from sync Flask context."""
	import asyncio
	try:
		loop = asyncio.get_event_loop()
	except RuntimeError:
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
	return loop.run_until_complete(coro)


def _err(msg: str, code: int = 400) -> tuple[Any, int]:
	return jsonify({"error": msg}), code


# ── Health ────────────────────────────────────────────────────────────────────

@pay_bp.get("/health")
def health():
	result = _run(_svc.health_check())
	return jsonify(result), 200


@pay_bp.get("/describe")
def describe():
	result = _run(_svc.describe())
	return jsonify(result), 200


# ── Billers ───────────────────────────────────────────────────────────────────

@pay_bp.get("/billers")
def list_billers():
	tenant_id = request.args.get("tenant_id", "default")
	category = request.args.get("category")
	try:
		items = _run(_svc.list_billers(category=category, tenant_id=tenant_id))
		return jsonify({"items": items, "total": len(items)}), 200
	except Exception as exc:
		_log.error("list_billers error: %s", exc)
		return _err(str(exc))


@pay_bp.post("/billers")
def create_biller():
	body = request.get_json(silent=True) or {}
	try:
		payload = PayBillerCreate(**body)
	except Exception as exc:
		return _err(f"validation_error: {exc}")
	try:
		record = _run(_svc.create_biller(
			biller_code=payload.biller_code,
			biller_name=payload.biller_name,
			category=payload.category,
			paybill_number=payload.paybill_number,
			account_mask=payload.account_mask,
			min_amount=payload.min_amount,
			max_amount=payload.max_amount,
			tenant_id=payload.tenant_id,
		))
		return jsonify(record), 201
	except ValueError as exc:
		return _err(str(exc))
	except Exception as exc:
		_log.error("create_biller error: %s", exc)
		return _err(str(exc), 500)


@pay_bp.get("/billers/<biller_id>")
def get_biller(biller_id: str):
	tenant_id = request.args.get("tenant_id", "default")
	try:
		record = _run(_svc.get_biller(biller_id, tenant_id=tenant_id))
		return jsonify(record), 200
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("get_biller error: %s", exc)
		return _err(str(exc), 500)


@pay_bp.put("/billers/<biller_id>")
def update_biller(biller_id: str):
	body = request.get_json(silent=True) or {}
	tenant_id = body.pop("tenant_id", "default")
	try:
		record = _run(_svc.update_biller(biller_id, tenant_id=tenant_id, **body))
		return jsonify(record), 200
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("update_biller error: %s", exc)
		return _err(str(exc), 500)


@pay_bp.delete("/billers/<biller_id>")
def delete_biller(biller_id: str):
	tenant_id = request.args.get("tenant_id", "default")
	try:
		record = _run(_svc.delete_biller(biller_id, tenant_id=tenant_id))
		return jsonify(record), 200
	except KeyError as exc:
		return _err(str(exc), 404)
	except PermissionError as exc:
		return _err(str(exc), 403)
	except Exception as exc:
		_log.error("delete_biller error: %s", exc)
		return _err(str(exc), 500)


# ── Bill payments ─────────────────────────────────────────────────────────────

@pay_bp.get("/bills")
def list_bill_payments():
	tenant_id = request.args.get("tenant_id", "default")
	phone_number = request.args.get("phone_number")
	try:
		items = _run(_svc.list_bill_payments(phone_number=phone_number, tenant_id=tenant_id))
		return jsonify({"items": items, "total": len(items)}), 200
	except Exception as exc:
		_log.error("list_bill_payments error: %s", exc)
		return _err(str(exc))


@pay_bp.post("/bills")
def pay_bill():
	body = request.get_json(silent=True) or {}
	try:
		payload = PayBillCreate(**body)
	except Exception as exc:
		return _err(f"validation_error: {exc}")
	try:
		record = _run(_svc.pay_bill(
			phone_number=payload.phone_number,
			biller_code=payload.biller_code,
			account_reference=payload.account_reference,
			amount=payload.amount,
			pin=payload.pin,
			narration=payload.narration,
			tenant_id=payload.tenant_id,
		))
		return jsonify(record), 201
	except PermissionError as exc:
		return _err(str(exc), 403)
	except KeyError as exc:
		return _err(str(exc), 404)
	except ValueError as exc:
		return _err(str(exc))
	except Exception as exc:
		_log.error("pay_bill error: %s", exc)
		return _err(str(exc), 500)


@pay_bp.get("/bills/<payment_id>")
def get_bill_payment(payment_id: str):
	tenant_id = request.args.get("tenant_id", "default")
	try:
		record = _run(_svc.get_bill_payment(payment_id, tenant_id=tenant_id))
		return jsonify(record), 200
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("get_bill_payment error: %s", exc)
		return _err(str(exc), 500)


@pay_bp.post("/bills/<payment_id>/reverse")
def reverse_bill_payment(payment_id: str):
	body = request.get_json(silent=True) or {}
	tenant_id = body.get("tenant_id", "default")
	reason = body.get("reason", "")
	if not reason:
		return _err("reason_required")
	try:
		record = _run(_svc.reverse_bill_payment(payment_id, reason=reason, tenant_id=tenant_id))
		return jsonify(record), 200
	except PermissionError as exc:
		return _err(str(exc), 403)
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("reverse_bill_payment error: %s", exc)
		return _err(str(exc), 500)


# ── Merchant payments ─────────────────────────────────────────────────────────

@pay_bp.get("/merchants")
def list_merchant_payments():
	tenant_id = request.args.get("tenant_id", "default")
	phone_number = request.args.get("phone_number")
	try:
		items = _run(_svc.list_merchant_payments(phone_number=phone_number, tenant_id=tenant_id))
		return jsonify({"items": items, "total": len(items)}), 200
	except Exception as exc:
		_log.error("list_merchant_payments error: %s", exc)
		return _err(str(exc))


@pay_bp.post("/merchants")
def pay_merchant():
	body = request.get_json(silent=True) or {}
	try:
		payload = PayMerchantCreate(**body)
	except Exception as exc:
		return _err(f"validation_error: {exc}")
	try:
		record = _run(_svc.pay_merchant(
			phone_number=payload.phone_number,
			merchant_till=payload.merchant_till,
			amount=payload.amount,
			pin=payload.pin,
			narration=payload.narration,
			tenant_id=payload.tenant_id,
		))
		return jsonify(record), 201
	except PermissionError as exc:
		return _err(str(exc), 403)
	except ValueError as exc:
		return _err(str(exc))
	except Exception as exc:
		_log.error("pay_merchant error: %s", exc)
		return _err(str(exc), 500)


@pay_bp.get("/merchants/<payment_id>")
def get_merchant_payment(payment_id: str):
	tenant_id = request.args.get("tenant_id", "default")
	try:
		record = _run(_svc.get_merchant_payment(payment_id, tenant_id=tenant_id))
		return jsonify(record), 200
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("get_merchant_payment error: %s", exc)
		return _err(str(exc), 500)


# ── Airtime top-ups ───────────────────────────────────────────────────────────

@pay_bp.get("/airtime")
def list_airtime_topups():
	tenant_id = request.args.get("tenant_id", "default")
	phone_number = request.args.get("phone_number")
	try:
		items = _run(_svc.list_airtime_topups(phone_number=phone_number, tenant_id=tenant_id))
		return jsonify({"items": items, "total": len(items)}), 200
	except Exception as exc:
		_log.error("list_airtime_topups error: %s", exc)
		return _err(str(exc))


@pay_bp.post("/airtime")
def buy_airtime():
	body = request.get_json(silent=True) or {}
	try:
		payload = PayAirtimeCreate(**body)
	except Exception as exc:
		return _err(f"validation_error: {exc}")
	try:
		record = _run(_svc.buy_airtime(
			phone_number=payload.phone_number,
			recipient_phone=payload.recipient_phone,
			amount=payload.amount,
			pin=payload.pin,
			telco=payload.telco,
			tenant_id=payload.tenant_id,
		))
		return jsonify(record), 201
	except PermissionError as exc:
		return _err(str(exc), 403)
	except ValueError as exc:
		return _err(str(exc))
	except Exception as exc:
		_log.error("buy_airtime error: %s", exc)
		return _err(str(exc), 500)


@pay_bp.get("/airtime/<topup_id>")
def get_airtime_topup(topup_id: str):
	tenant_id = request.args.get("tenant_id", "default")
	try:
		record = _run(_svc.get_airtime_topup(topup_id, tenant_id=tenant_id))
		return jsonify(record), 200
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("get_airtime_topup error: %s", exc)
		return _err(str(exc), 500)


# ── Utility payments ──────────────────────────────────────────────────────────

@pay_bp.get("/utilities")
def list_utility_payments():
	tenant_id = request.args.get("tenant_id", "default")
	phone_number = request.args.get("phone_number")
	utility_code = request.args.get("utility_code")
	try:
		items = _run(_svc.list_utility_payments(phone_number=phone_number, utility_code=utility_code, tenant_id=tenant_id))
		return jsonify({"items": items, "total": len(items)}), 200
	except Exception as exc:
		_log.error("list_utility_payments error: %s", exc)
		return _err(str(exc))


@pay_bp.post("/utilities")
def pay_utility():
	body = request.get_json(silent=True) or {}
	try:
		payload = PayUtilityCreate(**body)
	except Exception as exc:
		return _err(f"validation_error: {exc}")
	try:
		record = _run(_svc.pay_utility(
			phone_number=payload.phone_number,
			utility_code=payload.utility_code,
			meter_number=payload.meter_number,
			amount=payload.amount,
			pin=payload.pin,
			tenant_id=payload.tenant_id,
		))
		return jsonify(record), 201
	except PermissionError as exc:
		return _err(str(exc), 403)
	except ValueError as exc:
		return _err(str(exc))
	except Exception as exc:
		_log.error("pay_utility error: %s", exc)
		return _err(str(exc), 500)


@pay_bp.get("/utilities/<payment_id>")
def get_utility_payment(payment_id: str):
	tenant_id = request.args.get("tenant_id", "default")
	try:
		record = _run(_svc.get_utility_payment(payment_id, tenant_id=tenant_id))
		return jsonify(record), 200
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("get_utility_payment error: %s", exc)
		return _err(str(exc), 500)


# ── Send money ────────────────────────────────────────────────────────────────

@pay_bp.get("/send-money")
def list_send_money():
	tenant_id = request.args.get("tenant_id", "default")
	phone_number = request.args.get("phone_number")
	status = request.args.get("status")
	try:
		items = _run(_svc.list_send_money_transactions(phone_number=phone_number, status=status, tenant_id=tenant_id))
		return jsonify({"items": items, "total": len(items)}), 200
	except Exception as exc:
		_log.error("list_send_money error: %s", exc)
		return _err(str(exc))


@pay_bp.post("/send-money")
def initiate_send_money():
	body = request.get_json(silent=True) or {}
	try:
		payload = PaySendMoneyCreate(**body)
	except Exception as exc:
		return _err(f"validation_error: {exc}")
	try:
		record = _run(_svc.initiate_send_money(
			from_phone=payload.from_phone,
			to_phone=payload.to_phone,
			amount=payload.amount,
			pin=payload.pin,
			narration=payload.narration,
			tenant_id=payload.tenant_id,
		))
		return jsonify(record), 201
	except PermissionError as exc:
		return _err(str(exc), 403)
	except ValueError as exc:
		return _err(str(exc))
	except Exception as exc:
		_log.error("initiate_send_money error: %s", exc)
		return _err(str(exc), 500)


@pay_bp.get("/send-money/<transaction_id>")
def get_send_money(transaction_id: str):
	tenant_id = request.args.get("tenant_id", "default")
	try:
		record = _run(_svc.get_send_money_transaction(transaction_id, tenant_id=tenant_id))
		return jsonify(record), 200
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("get_send_money error: %s", exc)
		return _err(str(exc), 500)


@pay_bp.post("/send-money/<transaction_id>/confirm")
def confirm_send_money(transaction_id: str):
	body = request.get_json(silent=True) or {}
	try:
		payload = PaySendMoneyConfirmation(**{**body, "transaction_id": transaction_id})
	except Exception as exc:
		return _err(f"validation_error: {exc}")
	try:
		record = _run(_svc.confirm_send_money(
			transaction_id=transaction_id,
			pin=payload.pin,
			tenant_id=payload.tenant_id,
		))
		return jsonify(record), 200
	except PermissionError as exc:
		return _err(str(exc), 403)
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("confirm_send_money error: %s", exc)
		return _err(str(exc), 500)


@pay_bp.post("/send-money/<transaction_id>/cancel")
def cancel_send_money(transaction_id: str):
	body = request.get_json(silent=True) or {}
	tenant_id = body.get("tenant_id", "default")
	reason = body.get("reason", "")
	if not reason:
		return _err("reason_required")
	try:
		record = _run(_svc.cancel_send_money(transaction_id, reason=reason, tenant_id=tenant_id))
		return jsonify(record), 200
	except PermissionError as exc:
		return _err(str(exc), 403)
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("cancel_send_money error: %s", exc)
		return _err(str(exc), 500)


# ── USSD session ──────────────────────────────────────────────────────────────

@pay_bp.post("/ussd")
def handle_ussd():
	body = request.get_json(silent=True) or {}
	if not body:
		body = {
			"session_id": request.form.get("sessionId", ""),
			"phone_number": request.form.get("phoneNumber", ""),
			"service_code": request.form.get("serviceCode", ""),
			"input_text": request.form.get("text", ""),
		}
	try:
		payload = PayUssdSessionCreate(**body)
	except Exception as exc:
		return _err(f"validation_error: {exc}")
	try:
		result = _run(_svc.handle_ussd_request(
			session_id=payload.session_id,
			phone_number=payload.phone_number,
			service_code=payload.service_code,
			input_text=payload.input_text,
			tenant_id=payload.tenant_id,
		))
		if request.content_type and "form" in request.content_type:
			return result["response_text"], 200, {"Content-Type": "text/plain"}
		return jsonify(result), 200
	except Exception as exc:
		_log.error("handle_ussd error: %s", exc)
		return _err(str(exc), 500)


# ── Analytics & Search ────────────────────────────────────────────────────────

@pay_bp.get("/history")
def get_payment_history():
	phone_number = request.args.get("phone_number", "")
	tenant_id = request.args.get("tenant_id", "default")
	if not phone_number:
		return _err("phone_number_required")
	try:
		result = _run(_svc.get_payment_history(phone_number=phone_number, tenant_id=tenant_id))
		return jsonify(result), 200
	except Exception as exc:
		_log.error("get_payment_history error: %s", exc)
		return _err(str(exc), 500)


@pay_bp.get("/statistics")
def get_statistics():
	tenant_id = request.args.get("tenant_id", "default")
	try:
		result = _run(_svc.get_tenant_statistics(tenant_id=tenant_id))
		return jsonify(result), 200
	except Exception as exc:
		_log.error("get_statistics error: %s", exc)
		return _err(str(exc), 500)


@pay_bp.get("/volume/daily")
def get_daily_volume():
	tenant_id = request.args.get("tenant_id", "default")
	date = request.args.get("date")
	try:
		result = _run(_svc.get_daily_volume(tenant_id=tenant_id, date=date))
		return jsonify(result), 200
	except Exception as exc:
		_log.error("get_daily_volume error: %s", exc)
		return _err(str(exc), 500)


@pay_bp.get("/search")
def search_payments():
	tenant_id = request.args.get("tenant_id", "default")
	phone_number = request.args.get("phone_number")
	payment_type = request.args.get("payment_type")
	date_from = request.args.get("date_from")
	date_to = request.args.get("date_to")
	try:
		result = _run(_svc.search_payments(
			phone_number=phone_number,
			payment_type=payment_type,
			date_from=date_from,
			date_to=date_to,
			tenant_id=tenant_id,
		))
		return jsonify(result), 200
	except Exception as exc:
		_log.error("search_payments error: %s", exc)
		return _err(str(exc), 500)


# ── Audit ─────────────────────────────────────────────────────────────────────

@pay_bp.get("/audit-events")
def get_audit_events():
	tenant_id = request.args.get("tenant_id", "default")
	event_type = request.args.get("event_type")
	try:
		events = _run(_svc.get_audit_events(tenant_id=tenant_id, event_type=event_type))
		return jsonify({"events": events, "total": len(events)}), 200
	except Exception as exc:
		_log.error("get_audit_events error: %s", exc)
		return _err(str(exc), 500)
