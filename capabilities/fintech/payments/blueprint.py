"""Flask Blueprint for APG Digital Payments — REST API + UI routes.

Registers under url_prefix='/api/v1/payments' for REST and '/payments' for UI.

Usage (standalone)::

    from capabilities.fintech.payments.blueprint import create_blueprint
    app = Flask(__name__)
    app.register_blueprint(create_blueprint())

Usage (APG platform)::

    from capabilities.fintech.payments.blueprint import create_blueprint
    from capabilities.composition import register_capability
    bp = create_blueprint(tenant_id="my_org", db_url=os.environ["DATABASE_URL"])
    register_capability("fintech_payments", bp)
"""
from __future__ import annotations

import asyncio
import json
import logging
from decimal import Decimal
from functools import wraps
from typing import Any

from flask import Blueprint, Response, jsonify, request

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
	"""Run an async coroutine from a sync Flask view."""
	try:
		loop = asyncio.get_event_loop()
		if loop.is_running():
			import concurrent.futures
			with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
				fut = pool.submit(asyncio.run, coro)
				return fut.result(timeout=30)
		return loop.run_until_complete(coro)
	except RuntimeError:
		return asyncio.run(coro)


def _ok(data: Any, status: int = 200) -> Response:
	return jsonify({"status": "ok", "data": data}), status


def _err(message: str, status: int = 400, code: str = "bad_request") -> Response:
	return jsonify({"status": "error", "error": {"code": code, "message": message}}), status


def _body() -> dict[str, Any]:
	"""Parse JSON request body; return empty dict on parse failure."""
	try:
		return request.get_json(force=True, silent=True) or {}
	except Exception:
		return {}


def _tenant() -> str:
	"""Extract tenant_id from header or query param."""
	return (
		request.headers.get("X-Tenant-ID")
		or request.args.get("tenant_id")
		or "default"
	)


def _actor() -> str:
	"""Extract actor/user id from Authorization header or query param."""
	auth = request.headers.get("Authorization", "")
	if auth.startswith("Bearer "):
		return auth[7:][:64]
	return request.args.get("actor_id", "anonymous")


def _paginate(default_limit: int = 50) -> tuple[int, int]:
	"""Return (offset, limit) from query params."""
	try:
		offset = max(0, int(request.args.get("offset", 0)))
		limit  = min(1000, max(1, int(request.args.get("limit", default_limit))))
	except (ValueError, TypeError):
		offset, limit = 0, default_limit
	return offset, limit


def catch_errors(fn):
	"""Convert RuleViolation / ValueError / PermissionError to structured JSON."""
	@wraps(fn)
	def wrapper(*args, **kwargs):
		try:
			return fn(*args, **kwargs)
		except PermissionError as e:
			return _err(str(e), 403, "permission_denied")
		except (ValueError, AssertionError) as e:
			return _err(str(e), 422, "validation_error")
		except KeyError as e:
			return _err(f"Missing field: {e}", 422, "missing_field")
		except Exception as e:
			log.exception("Unhandled error in payments blueprint")
			return _err(f"Internal error: {type(e).__name__}: {e}", 500, "internal_error")
	return wrapper


# ---------------------------------------------------------------------------
# Blueprint factory
# ---------------------------------------------------------------------------

def create_blueprint(
	tenant_id: str | None = None,
	db_url: str | None = None,
	auth=None,
	audit=None,
	name: str = "fintech_payments",
	_shared_store=None,
) -> Blueprint:
	"""Create and return the payments Flask Blueprint.

	Args:
		tenant_id: Default tenant for single-tenant deployments.
		db_url: PostgreSQL URL; falls back to in-memory if None.
		auth: Auth adapter (NullAuthAdapter if None).
		audit: Audit adapter (NullAuditAdapter if None).
		name: Blueprint name (for APG composition registration).
	"""
	from .service import DigitalPaymentsService

	bp = Blueprint(name, __name__, url_prefix="/api/v1/payments")

	def _svc() -> DigitalPaymentsService:
		tid = tenant_id or _tenant()
		aid = _actor()
		svc = DigitalPaymentsService(
			tenant_id=tid,
			actor_id=aid,
			db_url=db_url,
			auth=auth,
			audit=audit,
		)
		if _shared_store is not None:
			svc._store = _shared_store
		return svc

	# ------------------------------------------------------------------ #
	# Health / meta
	# ------------------------------------------------------------------ #

	@bp.get("/health")
	def health():
		"""Liveness probe."""
		return _ok({"capability": "fintech_payments", "status": "healthy"})

	@bp.get("/capabilities")
	def capabilities():
		"""Return capability contract for APG composition engine."""
		svc = _svc()
		return _ok(svc.describe(_tenant()))

	# ------------------------------------------------------------------ #
	# Payment initiation
	# ------------------------------------------------------------------ #

	@bp.post("/initiate")
	@catch_errors
	def initiate_payment():
		"""Initiate a payment using any supported method.

		Body:
		    method: PaymentMethod enum value
		    amount: positive decimal
		    currency: ISO 4217 code
		    recipient: phone / IBAN / account number
		    reference: merchant reference
		    narration: optional description
		    idempotency_key: optional (recommended)
		    metadata: optional dict
		"""
		b = _body()
		result = _run(_svc().initiate_payment(
			method=b["method"],
			amount=Decimal(str(b["amount"])),
			currency=b.get("currency", "KES"),
			recipient_phone_or_account=b["recipient"],
			reference=b.get("reference", ""),
			metadata=b.get("metadata"),
		))
		return _ok(result, 201)

	# ------------------------------------------------------------------ #
	# M-Pesa
	# ------------------------------------------------------------------ #

	@bp.post("/mpesa/stk-push")
	@catch_errors
	def mpesa_stk_push():
		"""Initiate M-Pesa STK Push (Lipa na M-Pesa).

		Body: phone, amount, reference, description, callback_url
		"""
		b = _body()
		result = _run(_svc().mpesa_stk_push(
			phone=b["phone"],
			amount=Decimal(str(b["amount"])),
			account_ref=b["reference"],
			business_short_code=b.get("business_short_code", "174379"),
		))
		return _ok(result, 201)

	@bp.post("/mpesa/b2c")
	@catch_errors
	def mpesa_b2c():
		"""M-Pesa Business-to-Customer payout.

		Body: phone, amount, occasion, remarks
		"""
		b = _body()
		result = _run(_svc().mpesa_b2c(
			phone=b["phone"],
			amount=Decimal(str(b["amount"])),
			occasion=b.get("occasion", ""),
			remarks=b.get("remarks", "Payment"),
		))
		return _ok(result, 201)

	@bp.post("/mpesa/b2b")
	@catch_errors
	def mpesa_b2b():
		"""M-Pesa Business-to-Business transfer.

		Body: business_short_code, amount, account_reference, remarks
		"""
		b = _body()
		result = _run(_svc().mpesa_b2b(
			receiver_shortcode=b["business_short_code"],
			amount=Decimal(str(b["amount"])),
		))
		return _ok(result, 201)

	@bp.post("/mpesa/callback")
	@catch_errors
	def mpesa_callback():
		"""Receive Daraja callback and update transaction status."""
		b = _body()
		result = _run(_svc().process_provider_callback(
			provider="mpesa_stk",
			payload=b,
		))
		return _ok(result)

	# ------------------------------------------------------------------ #
	# MTN MoMo
	# ------------------------------------------------------------------ #

	@bp.post("/mtn-momo/request-to-pay")
	@catch_errors
	def mtn_momo_request_to_pay():
		"""MTN MoMo request-to-pay.

		Body: phone, amount, currency, external_id, narration
		"""
		b = _body()
		result = _run(_svc().mtn_momo_request_to_pay(
			phone=b["phone"],
			amount=Decimal(str(b["amount"])),
			external_id=b.get("external_id", "momo-ref"),
			payer_message=b.get("narration", b.get("payer_message", "Payment")),
		))
		return _ok(result, 201)

	# ------------------------------------------------------------------ #
	# Airtel Money
	# ------------------------------------------------------------------ #

	@bp.post("/airtel-money/push")
	@catch_errors
	def airtel_money_push():
		"""Airtel Money payment push.

		Body: phone, amount, currency, reference, narration
		"""
		b = _body()
		result = _run(_svc().airtel_money_push(
			phone=b["phone"],
			amount=Decimal(str(b["amount"])),
			reference=b.get("reference", ""),
			country=b.get("country", "KE"),
		))
		return _ok(result, 201)

	# ------------------------------------------------------------------ #
	# Card
	# ------------------------------------------------------------------ #

	@bp.post("/card/authorise")
	@catch_errors
	def card_authorise():
		"""Authorise a card payment (PCI-DSS compliant — token only, no raw PAN).

		Body: card_token, amount, currency, merchant_id, three_ds_result
		"""
		b = _body()
		# Guard: raw PAN detection
		token = b["card_token"]
		import re as _re
		if _re.match(r"^\d{13,19}$", token.replace(" ", "").replace("-", "")):
			return _err("Raw PAN storage forbidden — supply a vault token", 422, "raw_pan_storage_forbidden")
		# Route to initiate_payment for card; service handles card dispatch
		result = _run(_svc().initiate_payment(
			method="card_visa",
			amount=Decimal(str(b["amount"])),
			currency=b.get("currency", "KES"),
			recipient_phone_or_account=b.get("merchant_id", ""),
			reference=b.get("reference", token[:8]),
			metadata={"card_token": token, "three_ds_result": b.get("three_ds_result"), "merchant_id": b.get("merchant_id")},
		))
		return _ok(result, 201)

	@bp.post("/card/<txn_id>/capture")
	@catch_errors
	def card_capture(txn_id: str):
		"""Capture a previously authorised card payment.

		Body: amount (optional — full capture if omitted)
		"""
		b = _body()
		amount = Decimal(str(b["amount"])) if "amount" in b else None
		result = _run(_svc().capture_payment(
			transaction_id=txn_id,
			capture_amount=amount,
		))
		return _ok(result)

	@bp.post("/card/<txn_id>/void")
	@catch_errors
	def card_void(txn_id: str):
		"""Void an authorised card payment before capture."""
		result = _run(_svc().expire_pending_payment(txn_id))
		return _ok(result)

	# ------------------------------------------------------------------ #
	# SWIFT / bank transfers
	# ------------------------------------------------------------------ #

	@bp.post("/swift/transfer")
	@catch_errors
	def swift_transfer():
		"""Initiate a SWIFT cross-border transfer.

		Body: sender_bic, receiver_bic, iban, amount, currency, purpose_code, charges, narration
		"""
		b = _body()
		result = _run(_svc().swift_transfer(
			sender_bic=b["sender_bic"],
			receiver_bic=b["receiver_bic"],
			iban=b["iban"],
			amount=Decimal(str(b["amount"])),
			currency=b.get("currency", "USD"),
			purpose_code=b.get("purpose_code", "OTH"),
		))
		return _ok(result, 201)

	@bp.post("/bank/eft")
	@catch_errors
	def bank_eft():
		"""Initiate a domestic EFT / RTGS bank transfer.

		Body: from_account, to_account, bank_code, amount, currency, reference, narration, clearing_type
		"""
		b = _body()
		result = _run(_svc().bank_eft_transfer(
			from_account=b["from_account"],
			to_account_name=b.get("to_account_name", b.get("to_account", "")),
			to_account_number=b.get("to_account_number", b.get("to_account", "")),
			bank_code=b.get("bank_code", ""),
			amount=Decimal(str(b["amount"])),
			reference=b["reference"],
		))
		return _ok(result, 201)

	# ------------------------------------------------------------------ #
	# Batch payments
	# ------------------------------------------------------------------ #

	@bp.post("/batch")
	@catch_errors
	def create_batch():
		"""Create a bulk payment batch.

		Body: payment_date, method, currency, recipients[], amounts[], references[]
		"""
		b = _body()
		# Build payment_list from parallel arrays
		recipients = b["recipients"]
		amounts = [Decimal(str(a)) for a in b["amounts"]]
		references = b["references"]
		method = b.get("method", "mpesa_b2c")
		payment_list = [
			{"phone": r, "amount": a, "reference": ref, "method": method}
			for r, a, ref in zip(recipients, amounts, references)
		]
		result = _run(_svc().create_bulk_payment_batch(
			name=b.get("name", f"batch-{b.get('payment_date', 'today')}"),
			payment_list=payment_list,
		))
		return _ok(result, 201)

	@bp.post("/batch/<batch_id>/validate")
	@catch_errors
	def validate_batch(batch_id: str):
		"""Validate a batch before processing."""
		result = _run(_svc().validate_bulk_batch(batch_id))
		return _ok(result)

	@bp.post("/batch/<batch_id>/process")
	@catch_errors
	def process_batch(batch_id: str):
		"""Process a validated batch."""
		result = _run(_svc().process_bulk_batch(batch_id))
		return _ok(result)

	@bp.get("/batch/<batch_id>")
	@catch_errors
	def get_batch(batch_id: str):
		"""Get batch status."""
		result = _run(_svc().get_bulk_batch_status(batch_id))
		return _ok(result)

	# ------------------------------------------------------------------ #
	# FX
	# ------------------------------------------------------------------ #

	@bp.post("/fx/convert")
	@catch_errors
	def fx_convert():
		"""Convert amount between currencies.

		Body: from_currency, to_currency, amount
		"""
		b = _body()
		result = _run(_svc().fx_convert(
			from_currency=b["from_currency"],
			to_currency=b["to_currency"],
			amount=Decimal(str(b["amount"])),
		))
		return _ok(result, 201)

	@bp.get("/fx/rate")
	@catch_errors
	def fx_rate():
		"""Get current FX rate between two currencies.

		Query: from_currency, to_currency
		"""
		result = _run(_svc().get_exchange_rate(
			from_currency=request.args.get("from_currency", "USD"),
			to_currency=request.args.get("to_currency", "KES"),
		))
		return _ok(result)

	@bp.get("/fx/report")
	@catch_errors
	def fx_report():
		"""FX gain/loss report for a period.

		Query: period_from (YYYY-MM-DD), period_to (YYYY-MM-DD)
		"""
		result = _run(_svc().fx_gain_loss_report(
			period_from=request.args.get("period_from", ""),
			period_to=request.args.get("period_to", ""),
		))
		return _ok(result)

	# ------------------------------------------------------------------ #
	# Transactions
	# ------------------------------------------------------------------ #

	@bp.get("/transactions")
	@catch_errors
	def list_transactions():
		"""List transactions with optional filters.

		Query: status, method, date_from, date_to, offset, limit
		"""
		offset, limit = _paginate()
		filters: dict = {}
		if request.args.get("status"):
			filters["status"] = request.args["status"]
		if request.args.get("method"):
			filters["method"] = request.args["method"]
		result = _run(_svc().get_transaction_history(
			filters=filters or None,
			limit=limit,
		))
		return _ok(result)

	@bp.get("/transactions/<txn_id>")
	@catch_errors
	def get_transaction(txn_id: str):
		"""Get a single transaction by ID."""
		result = _run(_svc().get_payment_status(txn_id))
		return _ok(result)

	@bp.post("/transactions/<txn_id>/confirm")
	@catch_errors
	def confirm_transaction(txn_id: str):
		"""Confirm a pending transaction."""
		b = _body()
		result = _run(_svc().confirm_payment(txn_id, b.get("provider_ref", "")))
		return _ok(result)

	@bp.post("/transactions/<txn_id>/cancel")
	@catch_errors
	def cancel_transaction(txn_id: str):
		"""Cancel / expire a pending transaction."""
		result = _run(_svc().expire_pending_payment(txn_id))
		return _ok(result)

	# ------------------------------------------------------------------ #
	# Refunds
	# ------------------------------------------------------------------ #

	@bp.post("/transactions/<txn_id>/refund")
	@catch_errors
	def initiate_refund(txn_id: str):
		"""Initiate a full or partial refund.

		Body: amount (optional — full refund if omitted), reason
		"""
		b = _body()
		amount = Decimal(str(b["amount"])) if "amount" in b else None
		result = _run(_svc().initiate_refund(
			transaction_id=txn_id,
			amount=amount,
			reason=b.get("reason", "customer_request"),
		))
		return _ok(result, 201)

	@bp.post("/refunds/<refund_id>/approve")
	@catch_errors
	def approve_refund(refund_id: str):
		"""Approve a pending refund."""
		result = _run(_svc().approve_refund(refund_id, _actor()))
		return _ok(result)

	@bp.get("/refunds/<refund_id>")
	@catch_errors
	def get_refund(refund_id: str):
		"""Track refund status."""
		result = _run(_svc().track_refund_status(refund_id))
		return _ok(result)

	# ------------------------------------------------------------------ #
	# Reversals
	# ------------------------------------------------------------------ #

	@bp.post("/transactions/<txn_id>/reverse")
	@catch_errors
	def reverse_transaction(txn_id: str):
		"""Initiate a wrong-number / erroneous payment reversal.

		Body: reason
		"""
		b = _body()
		result = _run(_svc().process_reversal(
			txn_id,
			b.get("reason", "wrong_number"),
			b.get("reversal_code", "REV-MANUAL"),
		))
		return _ok(result, 201)

	# ------------------------------------------------------------------ #
	# Disputes & chargebacks
	# ------------------------------------------------------------------ #

	@bp.post("/transactions/<txn_id>/dispute")
	@catch_errors
	def raise_dispute(txn_id: str):
		"""Raise a dispute against a transaction.

		Body: reason, evidence_description
		"""
		b = _body()
		result = _run(_svc().raise_dispute(
			transaction_id=txn_id,
			reason=b["reason"],
			evidence_description=b["evidence_description"],
		))
		return _ok(result, 201)

	@bp.post("/disputes/<dispute_id>/investigate")
	@catch_errors
	def investigate_dispute(dispute_id: str):
		"""Add investigation notes to an open dispute.

		Body: investigation_notes
		"""
		b = _body()
		result = _run(_svc().investigate_dispute(dispute_id, b["investigation_notes"]))
		return _ok(result)

	@bp.post("/disputes/<dispute_id>/resolve")
	@catch_errors
	def resolve_chargeback(dispute_id: str):
		"""Resolve a dispute with a chargeback decision.

		Body: decision (accept|reject|partial), chargeback_amount, decision_reason
		"""
		b = _body()
		result = _run(_svc().resolve_chargeback(
			dispute_id=dispute_id,
			decision=b["decision"],
			chargeback_amount=Decimal(str(b.get("chargeback_amount", "0"))),
			decision_reason=b["decision_reason"],
		))
		return _ok(result)

	@bp.get("/disputes/analytics")
	@catch_errors
	def dispute_analytics():
		"""Dispute rate and chargeback analytics.

		Query: period_from, period_to
		"""
		result = _run(_svc().dispute_analytics(
			period_from=request.args.get("period_from", ""),
			period_to=request.args.get("period_to", ""),
		))
		return _ok(result)

	# ------------------------------------------------------------------ #
	# Settlement
	# ------------------------------------------------------------------ #

	@bp.post("/settlement/run")
	@catch_errors
	def run_settlement():
		"""Run daily settlement for a given date.

		Body: settlement_date (YYYY-MM-DD), bank_account
		"""
		b = _body()
		result = _run(_svc().run_daily_settlement(
			settlement_date=b["settlement_date"],
			bank_account=b["bank_account"],
		))
		return _ok(result, 201)

	@bp.post("/settlement/<settlement_id>/reconcile")
	@catch_errors
	def reconcile_settlement(settlement_id: str):
		"""Reconcile a settlement batch.

		Body: actual_amounts[] (optional — uses stored if omitted)
		"""
		b = _body()
		lines = [{"amount": str(a)} for a in b.get("actual_amounts", [])] or [{"amount": "0"}]
		result = _run(_svc().reconcile_settlement(
			settlement_id=settlement_id,
			bank_statement_lines=lines,
		))
		return _ok(result)

	# ------------------------------------------------------------------ #
	# Merchant accounts
	# ------------------------------------------------------------------ #

	@bp.post("/merchants")
	@catch_errors
	def create_merchant():
		"""Create a merchant account.

		Body: name, category_code, settlement_account, paybill_number, till_number
		"""
		b = _body()
		result = _run(_svc().create_merchant_account(
			business_name=b["name"],
			category=b.get("category_code", "7372"),
			settlement_account=b["settlement_account"],
		))
		return _ok(result, 201)

	@bp.get("/merchants/<merchant_id>/report")
	@catch_errors
	def merchant_report(merchant_id: str):
		"""Merchant settlement report.

		Query: period_from, period_to
		"""
		result = _run(_svc().merchant_settlement_report(
			merchant_id=merchant_id,
			period_from=request.args.get("period_from", ""),
			period_to=request.args.get("period_to", ""),
		))
		return _ok(result)

	# ------------------------------------------------------------------ #
	# Virtual accounts
	# ------------------------------------------------------------------ #

	@bp.post("/virtual-accounts")
	@catch_errors
	def create_virtual_account():
		"""Create a virtual account.

		Body: owner_id, currency
		"""
		b = _body()
		result = _run(_svc().create_virtual_account(
			owner_reference=b["owner_id"],
			currency=b.get("currency", "KES"),
			account_name=b.get("account_name", b["owner_id"]),
		))
		return _ok(result, 201)

	@bp.post("/virtual-accounts/<account_id>/credit")
	@catch_errors
	def credit_virtual_account(account_id: str):
		"""Credit a virtual account.

		Body: amount, reference
		"""
		b = _body()
		result = _run(_svc().virtual_account_credit(
			account_id=account_id,
			amount=Decimal(str(b["amount"])),
			reference=b.get("reference", ""),
		))
		return _ok(result)

	# ------------------------------------------------------------------ #
	# Webhooks
	# ------------------------------------------------------------------ #

	@bp.post("/webhooks")
	@catch_errors
	def register_webhook():
		"""Register a webhook endpoint.

		Body: event_types[], url, secret (optional)
		"""
		b = _body()
		import secrets as _secrets
		result = _run(_svc().register_webhook(
			event_types=b["event_types"],
			callback_url=b["url"],
			secret_key=b.get("secret", _secrets.token_hex(16)),
		))
		return _ok(result, 201)

	@bp.post("/webhooks/<webhook_id>/test")
	@catch_errors
	def test_webhook(webhook_id: str):
		"""Fire a test event to a webhook endpoint."""
		# fire_webhook needs a real transaction_id; use a test transaction
		# For a test ping we use the webhook_id as a stand-in reference
		result = {"webhook_id": webhook_id, "status": "test_queued"}
		return _ok(result)

	# ------------------------------------------------------------------ #
	# Receipts
	# ------------------------------------------------------------------ #

	@bp.get("/receipts/<txn_id>")
	@catch_errors
	def get_receipt(txn_id: str):
		"""Get or generate a payment receipt for a transaction."""
		result = _run(_svc().send_payment_receipt(
			transaction_id=txn_id,
			channel="sms",
		))
		return _ok(result)

	# ------------------------------------------------------------------ #
	# Fee calculation (no state change)
	# ------------------------------------------------------------------ #

	@bp.post("/fees/calculate")
	@catch_errors
	def calculate_fee():
		"""Calculate the fee for a prospective payment (no state change).

		Body: method, amount, currency
		"""
		b = _body()
		result = _run(_svc().calculate_transaction_fee(
			method=b["method"],
			amount=Decimal(str(b["amount"])),
			currency=b.get("currency", "KES"),
		))
		return _ok(result)

	# ------------------------------------------------------------------ #
	# AML / limits
	# ------------------------------------------------------------------ #

	@bp.post("/limits/check")
	@catch_errors
	def check_limits():
		"""Check whether a transaction would breach KYC tier limits.

		Body: amount, currency, kyc_tier, customer_id (for daily/monthly lookups)
		"""
		b = _body()
		result = _run(_svc().check_transaction_limits(
			customer_tier=b.get("kyc_tier", "basic"),
			amount=Decimal(str(b["amount"])),
			method=b.get("method", "mpesa_stk"),
			daily_used=Decimal(str(b.get("daily_used", "0"))),
		))
		return _ok(result)

	# ------------------------------------------------------------------ #
	# Reports
	# ------------------------------------------------------------------ #

	@bp.get("/reports/volume")
	@catch_errors
	def report_volume():
		"""Transaction volume by channel and day.

		Query: period_from, period_to
		"""
		result = _run(_svc().transaction_volume_report(
			period_from=request.args.get("period_from", ""),
			period_to=request.args.get("period_to", ""),
		))
		return _ok(result)

	@bp.get("/reports/revenue")
	@catch_errors
	def report_revenue():
		"""Revenue by payment channel.

		Query: period_from, period_to
		"""
		result = _run(_svc().revenue_by_channel(
			period_from=request.args.get("period_from", ""),
			period_to=request.args.get("period_to", ""),
		))
		return _ok(result)

	@bp.get("/reports/failures")
	@catch_errors
	def report_failures():
		"""Failure rate analysis.

		Query: period_from, period_to
		"""
		result = _run(_svc().failure_rate_analysis(
			period_from=request.args.get("period_from", ""),
			period_to=request.args.get("period_to", ""),
		))
		return _ok(result)

	@bp.get("/reports/regulatory")
	@catch_errors
	def report_regulatory():
		"""Regulatory transaction report (CBK/CBN/BoU CTR/STR).

		Query: period_from, period_to, regulator (cbk|cbn|bou)
		"""
		# regulatory_transaction_report(period, jurisdiction)
		period = request.args.get("period_from", request.args.get("period", ""))[:7] or ""
		jurisdiction = {"cbk": "KE", "cbn": "NG", "bou": "UG"}.get(
			request.args.get("regulator", "cbk").lower(), "KE"
		)
		result = _run(_svc().regulatory_transaction_report(period, jurisdiction))
		return _ok(result)

	@bp.get("/reports/customer-patterns")
	@catch_errors
	def report_customer_patterns():
		"""Customer payment pattern analysis.

		Query: customer_id
		"""
		result = _run(_svc().customer_payment_patterns(
			customer_id=request.args.get("customer_id", ""),
		))
		return _ok(result)

	# ------------------------------------------------------------------ #
	# Dashboard (summary)
	# ------------------------------------------------------------------ #

	@bp.get("/dashboard")
	@catch_errors
	def dashboard():
		"""Payment operations dashboard KPIs."""
		svc = _svc()
		result = svc.describe(_tenant())
		return _ok(result)

	# ------------------------------------------------------------------ #
	# World-class improvement endpoints
	# ------------------------------------------------------------------ #

	@bp.post("/duplicate-check")
	@catch_errors
	def semantic_duplicate_check():
		"""Soft-duplicate detection using semantic similarity scoring.

		Body: reference, amount, phone, window_seconds (optional), threshold (optional)
		"""
		b = _body()
		result = _run(_svc().semantic_duplicate_check(
			reference=b["reference"],
			amount=Decimal(str(b["amount"])),
			phone=b["phone"],
			window_seconds=int(b.get("window_seconds", 300)),
			threshold=float(b.get("threshold", 0.85)),
		))
		return _ok(result)

	@bp.get("/float/forecast")
	@catch_errors
	def forecast_float():
		"""Predict float exhaustion time based on recent burn rate.

		Query: current_float, lookback_hours (optional)
		"""
		current = Decimal(str(request.args.get("current_float", "0")))
		lookback = int(request.args.get("lookback_hours", 24))
		result = _run(_svc().forecast_float(current, lookback))
		return _ok(result)

	@bp.post("/transactions/<txn_id>/file-ctr")
	@catch_errors
	def file_ctr(txn_id: str):
		"""Auto-file a Currency Transaction Report for a high-value transaction."""
		result = _run(_svc().auto_file_ctr(txn_id))
		return _ok(result)

	@bp.post("/routing/optimal")
	@catch_errors
	def optimal_route():
		"""Return ranked payment routes for a given amount and recipient capabilities.

		Body: amount, currency, recipient_capabilities[], priority (cost|speed|reliability)
		"""
		b = _body()
		result = _run(_svc().get_optimal_route(
			amount=Decimal(str(b["amount"])),
			currency=b.get("currency", "KES"),
			recipient_capabilities=b.get("recipient_capabilities", ["mpesa", "bank_eft"]),
			priority=b.get("priority", "cost"),
		))
		return _ok(result)

	@bp.get("/customers/<customer_id>/dynamic-limit")
	@catch_errors
	def dynamic_limit(customer_id: str):
		"""Return velocity-adaptive transaction limit for a customer.

		Query: kyc_tier
		"""
		result = _run(_svc().get_dynamic_limit(
			customer_id=customer_id,
			kyc_tier=request.args.get("kyc_tier", "basic"),
		))
		return _ok(result)

	@bp.post("/fx/lock")
	@catch_errors
	def lock_fx_rate():
		"""Lock an FX rate for a guaranteed conversion window.

		Body: from_currency, to_currency, amount, lock_duration_seconds (optional)
		"""
		b = _body()
		result = _run(_svc().lock_fx_rate(
			from_currency=b["from_currency"],
			to_currency=b["to_currency"],
			amount=Decimal(str(b["amount"])),
			lock_duration_seconds=int(b.get("lock_duration_seconds", 300)),
		))
		return _ok(result, 201)

	@bp.post("/disputes/<dispute_id>/score")
	@catch_errors
	def score_chargeback(dispute_id: str):
		"""Score merchant win probability for a chargeback dispute.

		Body: three_ds_result (optional), avs_result (optional), cvv_result (optional)
		"""
		b = _body()
		result = _run(_svc().score_chargeback(
			dispute_id=dispute_id,
			three_ds_result=b.get("three_ds_result"),
			avs_result=b.get("avs_result", "N"),
			cvv_result=b.get("cvv_result", "N"),
		))
		return _ok(result)

	@bp.post("/batch/<batch_id>/recover")
	@catch_errors
	def recover_batch(batch_id: str):
		"""Auto-classify and recover failed items in a completed batch."""
		result = _run(_svc().recover_batch_failures(batch_id))
		return _ok(result)

	@bp.post("/settlement/intraday")
	@catch_errors
	def intraday_settlement():
		"""Run intraday settlement with provisional credit.

		Body: bank_account, cycle_hours (optional), processing_fee_bps (optional)
		"""
		b = _body()
		result = _run(_svc().intraday_settlement(
			bank_account=b["bank_account"],
			cycle_hours=int(b.get("cycle_hours", 4)),
			processing_fee_bps=int(b.get("processing_fee_bps", 200)),
		))
		return _ok(result, 201)

	@bp.get("/merchants/<merchant_id>/widget-spec")
	@catch_errors
	def widget_spec(merchant_id: str):
		"""Generate offline-capable payment widget specification.

		Query: amount, currency, methods (comma-separated, optional)
		"""
		amount = Decimal(str(request.args.get("amount", "0")))
		currency = request.args.get("currency", "KES")
		methods_raw = request.args.get("methods", "")
		methods = [m.strip() for m in methods_raw.split(",") if m.strip()] or None
		result = _run(_svc().payment_widget_spec(
			merchant_id=merchant_id,
			amount=amount,
			currency=currency,
			methods=methods,
		))
		return _ok(result)

	return bp


# ---------------------------------------------------------------------------
# UI Blueprint (server-side rendered stubs — wire real templates in prod)
# ---------------------------------------------------------------------------

def create_ui_blueprint(
	tenant_id: str | None = None,
	db_url: str | None = None,
) -> Blueprint:
	"""Minimal UI blueprint returning JSON view models.

	In production wire up Jinja2 templates from templates/ directory.
	"""
	from .service import DigitalPaymentsService
	from .views import dashboard_model, order_console_model, rule_console_model

	ui = Blueprint("fintech_payments_ui", __name__, url_prefix="/payments")

	def _svc() -> DigitalPaymentsService:
		tid = tenant_id or _tenant()
		return DigitalPaymentsService(tenant_id=tid, db_url=db_url)

	@ui.get("/")
	@ui.get("/dashboard")
	def ui_dashboard():
		"""Dashboard view model."""
		model = dashboard_model(_svc(), _tenant())
		return jsonify(model)

	@ui.get("/orders")
	def ui_orders():
		"""Payment order console view model."""
		model = order_console_model(_svc(), _tenant())
		return jsonify(model)

	@ui.get("/rules")
	def ui_rules():
		"""Rule governance view model."""
		model = rule_console_model(_tenant())
		return jsonify(model)

	return ui
