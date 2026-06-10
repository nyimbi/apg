"""Flask Blueprint — Mobile Banking USSD REST API."""
from __future__ import annotations

import logging
from decimal import Decimal, InvalidOperation
from typing import Any

from flask import Blueprint, jsonify, request

from .models import (
	MobAccountCreate,
	MobTransferCreate,
	MobStandingOrderCreate,
	MobPinChangeRequest,
	MobPinResetRequest,
	MobUssdSessionCreate,
)
from .service import MobUssdService

_log = logging.getLogger(__name__)

mob_bp = Blueprint("fintech_ussd_mob", __name__, url_prefix="/api/fintech/ussd/mob")

# Service singleton (replace with DI in production)
_svc = MobUssdService()


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

@mob_bp.get("/health")
def health():
	result = _run(_svc.health_check())
	return jsonify(result), 200


@mob_bp.get("/describe")
def describe():
	result = _run(_svc.describe())
	return jsonify(result), 200


# ── Accounts ──────────────────────────────────────────────────────────────────

@mob_bp.get("/accounts")
def list_accounts():
	tenant_id = request.args.get("tenant_id", "default")
	status = request.args.get("status")
	try:
		items = _run(_svc.list_accounts(tenant_id=tenant_id, status=status))
		return jsonify({"items": items, "total": len(items)}), 200
	except Exception as exc:
		_log.error("list_accounts error: %s", exc)
		return _err(str(exc))


@mob_bp.post("/accounts")
def create_account():
	body = request.get_json(silent=True) or {}
	try:
		payload = MobAccountCreate(**body)
	except Exception as exc:
		return _err(f"validation_error: {exc}")
	try:
		record = _run(_svc.create_account(
			phone_number=payload.phone_number,
			account_number=payload.account_number,
			account_type=payload.account_type,
			customer_name=payload.customer_name,
			national_id=payload.national_id,
			pin=payload.pin,
			currency=payload.currency,
			tenant_id=payload.tenant_id,
		))
		return jsonify(record), 201
	except ValueError as exc:
		return _err(str(exc))
	except Exception as exc:
		_log.error("create_account error: %s", exc)
		return _err(str(exc), 500)


@mob_bp.get("/accounts/<account_id>")
def get_account(account_id: str):
	tenant_id = request.args.get("tenant_id", "default")
	try:
		record = _run(_svc.get_account(account_id, tenant_id=tenant_id))
		return jsonify(record), 200
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("get_account error: %s", exc)
		return _err(str(exc), 500)


@mob_bp.put("/accounts/<account_id>")
def update_account(account_id: str):
	body = request.get_json(silent=True) or {}
	tenant_id = body.pop("tenant_id", "default")
	try:
		record = _run(_svc.update_account(account_id, tenant_id=tenant_id, **body))
		return jsonify(record), 200
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("update_account error: %s", exc)
		return _err(str(exc), 500)


@mob_bp.delete("/accounts/<account_id>")
def delete_account(account_id: str):
	tenant_id = request.args.get("tenant_id", "default")
	try:
		record = _run(_svc.delete_account(account_id, tenant_id=tenant_id))
		return jsonify(record), 200
	except KeyError as exc:
		return _err(str(exc), 404)
	except PermissionError as exc:
		return _err(str(exc), 403)
	except Exception as exc:
		_log.error("delete_account error: %s", exc)
		return _err(str(exc), 500)


# ── Balance ───────────────────────────────────────────────────────────────────

@mob_bp.post("/accounts/<account_number>/balance")
def get_balance(account_number: str):
	body = request.get_json(silent=True) or {}
	pin = body.get("pin", "")
	tenant_id = body.get("tenant_id", "default")
	if not pin:
		return _err("pin_required")
	try:
		result = _run(_svc.get_balance(account_number, pin=pin, tenant_id=tenant_id))
		return jsonify(result), 200
	except PermissionError as exc:
		return _err(str(exc), 403)
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("get_balance error: %s", exc)
		return _err(str(exc), 500)


@mob_bp.post("/accounts/<account_number>/deposit")
def deposit(account_number: str):
	body = request.get_json(silent=True) or {}
	tenant_id = body.get("tenant_id", "default")
	try:
		amount = Decimal(str(body["amount"]))
	except (KeyError, InvalidOperation):
		return _err("invalid_or_missing_amount")
	try:
		result = _run(_svc.deposit(account_number, amount=amount, narration=body.get("narration", ""), tenant_id=tenant_id))
		return jsonify(result), 200
	except Exception as exc:
		_log.error("deposit error: %s", exc)
		return _err(str(exc), 500)


# ── Mini-statement ────────────────────────────────────────────────────────────

@mob_bp.post("/accounts/<account_number>/statement")
def get_mini_statement(account_number: str):
	body = request.get_json(silent=True) or {}
	pin = body.get("pin", "")
	tenant_id = body.get("tenant_id", "default")
	rows = int(body.get("rows", 5))
	if not pin:
		return _err("pin_required")
	try:
		result = _run(_svc.get_mini_statement(account_number, pin=pin, rows=rows, tenant_id=tenant_id))
		return jsonify(result), 200
	except PermissionError as exc:
		return _err(str(exc), 403)
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("get_mini_statement error: %s", exc)
		return _err(str(exc), 500)


# ── Transfers ─────────────────────────────────────────────────────────────────

@mob_bp.get("/transfers")
def list_transfers():
	tenant_id = request.args.get("tenant_id", "default")
	status = request.args.get("status")
	try:
		items = _run(_svc.list_transfers(tenant_id=tenant_id, status=status))
		return jsonify({"items": items, "total": len(items)}), 200
	except Exception as exc:
		_log.error("list_transfers error: %s", exc)
		return _err(str(exc))


@mob_bp.post("/transfers")
def create_transfer():
	body = request.get_json(silent=True) or {}
	try:
		payload = MobTransferCreate(**body)
	except Exception as exc:
		return _err(f"validation_error: {exc}")
	try:
		record = _run(_svc.create_transfer(
			from_account=payload.from_account,
			to_account=payload.to_account,
			amount=payload.amount,
			pin=payload.pin,
			narration=payload.narration,
			currency=payload.currency,
			tenant_id=payload.tenant_id,
		))
		return jsonify(record), 201
	except PermissionError as exc:
		return _err(str(exc), 403)
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("create_transfer error: %s", exc)
		return _err(str(exc), 500)


@mob_bp.get("/transfers/<transfer_id>")
def get_transfer(transfer_id: str):
	tenant_id = request.args.get("tenant_id", "default")
	try:
		record = _run(_svc.get_transfer(transfer_id, tenant_id=tenant_id))
		return jsonify(record), 200
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("get_transfer error: %s", exc)
		return _err(str(exc), 500)


@mob_bp.post("/transfers/<transfer_id>/reverse")
def reverse_transfer(transfer_id: str):
	body = request.get_json(silent=True) or {}
	tenant_id = body.get("tenant_id", "default")
	reason = body.get("reason", "")
	if not reason:
		return _err("reason_required")
	try:
		record = _run(_svc.reverse_transfer(transfer_id, reason=reason, tenant_id=tenant_id))
		return jsonify(record), 200
	except PermissionError as exc:
		return _err(str(exc), 403)
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("reverse_transfer error: %s", exc)
		return _err(str(exc), 500)


# ── Standing orders ───────────────────────────────────────────────────────────

@mob_bp.get("/standing-orders")
def list_standing_orders():
	tenant_id = request.args.get("tenant_id", "default")
	account_number = request.args.get("account_number")
	try:
		items = _run(_svc.list_standing_orders(account_number=account_number, tenant_id=tenant_id))
		return jsonify({"items": items, "total": len(items)}), 200
	except Exception as exc:
		_log.error("list_standing_orders error: %s", exc)
		return _err(str(exc))


@mob_bp.post("/standing-orders")
def create_standing_order():
	body = request.get_json(silent=True) or {}
	try:
		payload = MobStandingOrderCreate(**body)
	except Exception as exc:
		return _err(f"validation_error: {exc}")
	try:
		record = _run(_svc.create_standing_order(
			from_account=payload.from_account,
			to_account=payload.to_account,
			amount=payload.amount,
			frequency=payload.frequency,
			start_date=payload.start_date,
			pin=payload.pin,
			end_date=payload.end_date,
			narration=payload.narration,
			tenant_id=payload.tenant_id,
		))
		return jsonify(record), 201
	except (PermissionError, ValueError) as exc:
		return _err(str(exc), 403)
	except Exception as exc:
		_log.error("create_standing_order error: %s", exc)
		return _err(str(exc), 500)


@mob_bp.get("/standing-orders/<order_id>")
def get_standing_order(order_id: str):
	tenant_id = request.args.get("tenant_id", "default")
	try:
		record = _run(_svc.get_standing_order(order_id, tenant_id=tenant_id))
		return jsonify(record), 200
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("get_standing_order error: %s", exc)
		return _err(str(exc), 500)


@mob_bp.put("/standing-orders/<order_id>")
def update_standing_order(order_id: str):
	body = request.get_json(silent=True) or {}
	tenant_id = body.pop("tenant_id", "default")
	try:
		record = _run(_svc.update_standing_order(order_id, tenant_id=tenant_id, **body))
		return jsonify(record), 200
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("update_standing_order error: %s", exc)
		return _err(str(exc), 500)


@mob_bp.delete("/standing-orders/<order_id>")
def delete_standing_order(order_id: str):
	tenant_id = request.args.get("tenant_id", "default")
	try:
		record = _run(_svc.delete_standing_order(order_id, tenant_id=tenant_id))
		return jsonify(record), 200
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("delete_standing_order error: %s", exc)
		return _err(str(exc), 500)


# ── PIN management ────────────────────────────────────────────────────────────

@mob_bp.post("/pin/change")
def change_pin():
	body = request.get_json(silent=True) or {}
	try:
		payload = MobPinChangeRequest(**body)
	except Exception as exc:
		return _err(f"validation_error: {exc}")
	try:
		result = _run(_svc.change_pin(
			account_number=payload.account_number,
			old_pin=payload.old_pin,
			new_pin=payload.new_pin,
			confirm_pin=payload.confirm_pin,
			tenant_id=payload.tenant_id,
		))
		return jsonify(result), 200
	except PermissionError as exc:
		return _err(str(exc), 403)
	except ValueError as exc:
		return _err(str(exc))
	except Exception as exc:
		_log.error("change_pin error: %s", exc)
		return _err(str(exc), 500)


@mob_bp.post("/pin/reset/otp")
def generate_pin_reset_otp():
	body = request.get_json(silent=True) or {}
	phone_number = body.get("phone_number", "")
	tenant_id = body.get("tenant_id", "default")
	if not phone_number:
		return _err("phone_number_required")
	try:
		result = _run(_svc.generate_pin_reset_otp(phone_number=phone_number, tenant_id=tenant_id))
		return jsonify(result), 200
	except KeyError as exc:
		return _err(str(exc), 404)
	except Exception as exc:
		_log.error("generate_pin_reset_otp error: %s", exc)
		return _err(str(exc), 500)


@mob_bp.post("/pin/reset")
def reset_pin():
	body = request.get_json(silent=True) or {}
	try:
		payload = MobPinResetRequest(**body)
	except Exception as exc:
		return _err(f"validation_error: {exc}")
	try:
		result = _run(_svc.reset_pin(
			phone_number=payload.phone_number,
			national_id=payload.national_id,
			new_pin=payload.new_pin,
			otp=payload.otp,
			tenant_id=payload.tenant_id,
		))
		return jsonify(result), 200
	except PermissionError as exc:
		return _err(str(exc), 403)
	except Exception as exc:
		_log.error("reset_pin error: %s", exc)
		return _err(str(exc), 500)


# ── USSD session ──────────────────────────────────────────────────────────────

@mob_bp.post("/ussd")
def handle_ussd():
	body = request.get_json(silent=True) or {}
	# Also accept form-encoded (telco gateway format)
	if not body:
		body = {
			"session_id": request.form.get("sessionId", ""),
			"phone_number": request.form.get("phoneNumber", ""),
			"service_code": request.form.get("serviceCode", ""),
			"input_text": request.form.get("text", ""),
		}
	try:
		payload = MobUssdSessionCreate(**body)
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
		# Return plain text for telco gateways
		if request.content_type and "form" in request.content_type:
			return result["response_text"], 200, {"Content-Type": "text/plain"}
		return jsonify(result), 200
	except Exception as exc:
		_log.error("handle_ussd error: %s", exc)
		return _err(str(exc), 500)


# ── Audit ─────────────────────────────────────────────────────────────────────

@mob_bp.get("/audit-events")
def get_audit_events():
	tenant_id = request.args.get("tenant_id", "default")
	event_type = request.args.get("event_type")
	try:
		events = _run(_svc.get_audit_events(tenant_id=tenant_id, event_type=event_type))
		return jsonify({"events": events, "total": len(events)}), 200
	except Exception as exc:
		_log.error("get_audit_events error: %s", exc)
		return _err(str(exc), 500)


@mob_bp.get("/statistics")
def get_statistics():
	tenant_id = request.args.get("tenant_id", "default")
	try:
		result = _run(_svc.get_tenant_statistics(tenant_id=tenant_id))
		return jsonify(result), 200
	except Exception as exc:
		_log.error("get_statistics error: %s", exc)
		return _err(str(exc), 500)
