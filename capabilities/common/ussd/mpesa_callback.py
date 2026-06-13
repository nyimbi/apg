"""
MPESA Callback Handler — C2B confirmation/validation and STK Push callbacks.

Handles the three Safaricom MPESA callback types:
  - C2B Confirmation  : merchant receives payment notification
  - C2B Validation    : merchant accepts or rejects payment before confirmation
  - STK Push Callback : result of initiating a push-to-phone payment request

Each handler validates the payload, emits a NATS event, and returns the
appropriate response structure Safaricom expects.

NATS events emitted:
  ussd.payment_confirmed  — on C2B confirmation
  ussd.stk_result         — on STK push result

'Us' Pydantic model prefix throughout.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

try:
	from uuid6 import uuid7
	def uuid7str() -> str:
		return str(uuid7())
except ImportError:
	import uuid
	def uuid7str() -> str:  # type: ignore[misc]
		return str(uuid.uuid4())

_log = logging.getLogger(__name__)

# Safaricom response codes
_MPESA_ACCEPT = {"ResultCode": "0", "ResultDesc": "Accepted"}
_MPESA_REJECT = {"ResultCode": "C2B00011", "ResultDesc": "Rejected"}


# ── Pydantic models ──────────────────────────────────────────────────────────

class UsMpesaC2BPayload(BaseModel):
	"""Incoming C2B callback payload from Safaricom (both validation and confirmation)."""
	model_config = ConfigDict(extra="allow", validate_by_name=True, validate_by_alias=True)

	TransactionType: str = ""
	TransID: str = ""
	TransTime: str = ""
	TransAmount: str = "0"
	BusinessShortCode: str = ""
	BillRefNumber: str = ""
	InvoiceNumber: str = ""
	OrgAccountBalance: str = ""
	ThirdPartyTransID: str = ""
	MSISDN: str = ""
	FirstName: str = ""
	MiddleName: str = ""
	LastName: str = ""


class UsMpesaStkPayload(BaseModel):
	"""STK Push callback body from Safaricom."""
	model_config = ConfigDict(extra="allow", validate_by_name=True, validate_by_alias=True)

	MerchantRequestID: str = ""
	CheckoutRequestID: str = ""
	ResultCode: int = 0
	ResultDesc: str = ""
	# CallbackMetadata is present only on success (ResultCode == 0)
	CallbackMetadata: dict[str, Any] | None = None


class UsPaymentEvent(BaseModel):
	"""Normalised payment event emitted to NATS."""
	model_config = ConfigDict(extra="forbid", validate_by_name=True, validate_by_alias=True)

	event_id: str = Field(default_factory=uuid7str)
	event_type: str          # c2b_confirmed | c2b_rejected | stk_success | stk_failed
	transaction_id: str
	phone: str
	amount: str
	reference: str
	short_code: str
	timestamp: str           # ISO-8601
	raw: dict[str, Any] = Field(default_factory=dict)


# ── Callback handler ──────────────────────────────────────────────────────────

class MpesaCallbackHandler:
	"""
	Stateless MPESA callback processor.

	Inject a NATS client (duck-typed: async publish(subject, payload: bytes))
	and optional validation logic via the ``validate_c2b`` hook.

	validate_c2b hook signature::

		async def validate_c2b(payload: UsMpesaC2BPayload) -> bool: ...

	Returning False causes the validation endpoint to reject the transaction.
	"""

	def __init__(
		self,
		nats_client: Any | None = None,
		validate_c2b: Any | None = None,
		accepted_short_codes: set[str] | None = None,
	) -> None:
		self._nats = nats_client
		self._validate_c2b = validate_c2b
		self._accepted_short_codes: set[str] = accepted_short_codes or set()
		self._events: list[UsPaymentEvent] = []  # in-process audit buffer

	# ── C2B Validation ────────────────────────────────────────────────────────

	async def handle_c2b_validation(
		self, payload: dict[str, Any]
	) -> dict[str, str]:
		"""
		Validate an incoming C2B payment before it is confirmed.

		Safaricom calls this endpoint first; the merchant has a short window to
		accept or reject.  Return MPESA_ACCEPT to allow; MPESA_REJECT to deny.

		Args:
			payload: Raw JSON dict from Safaricom validation webhook.

		Returns:
			``{"ResultCode": "0", "ResultDesc": "Accepted"}`` or reject equivalent.
		"""
		try:
			parsed = UsMpesaC2BPayload.model_validate(payload)
		except Exception as exc:
			_log.warning("c2b_validation parse error: %s", exc)
			return _MPESA_REJECT

		# Short-code allow-list check
		if self._accepted_short_codes and parsed.BusinessShortCode not in self._accepted_short_codes:
			_log.warning(
				"c2b_validation rejected: unknown short_code=%s txn=%s",
				parsed.BusinessShortCode, parsed.TransID,
			)
			return _MPESA_REJECT

		# Caller-supplied business validation
		if self._validate_c2b is not None:
			try:
				accepted: bool = await self._validate_c2b(parsed)
				if not accepted:
					_log.info("c2b_validation rejected by hook: txn=%s ref=%s", parsed.TransID, parsed.BillRefNumber)
					return _MPESA_REJECT
			except Exception as exc:
				_log.error("validate_c2b hook error txn=%s: %s", parsed.TransID, exc)
				return _MPESA_REJECT

		_log.info(
			"c2b_validation accepted: txn=%s amount=%s msisdn=%s ref=%s",
			parsed.TransID, parsed.TransAmount, parsed.MSISDN, parsed.BillRefNumber,
		)
		return _MPESA_ACCEPT

	# ── C2B Confirmation ──────────────────────────────────────────────────────

	async def handle_c2b_confirmation(self, payload: dict[str, Any]) -> None:
		"""
		Process a confirmed C2B payment notification.

		Emits: ussd.payment_confirmed

		Args:
			payload: Raw JSON dict from Safaricom confirmation webhook.
		"""
		try:
			parsed = UsMpesaC2BPayload.model_validate(payload)
		except Exception as exc:
			_log.error("c2b_confirmation parse error: %s", exc)
			return

		event = UsPaymentEvent(
			event_type="c2b_confirmed",
			transaction_id=parsed.TransID,
			phone=parsed.MSISDN,
			amount=parsed.TransAmount,
			reference=parsed.BillRefNumber or parsed.InvoiceNumber,
			short_code=parsed.BusinessShortCode,
			timestamp=self._iso_now(),
			raw=dict(payload),
		)
		self._events.append(event)
		_log.info(
			"c2b_confirmation: txn=%s amount=%s phone=%s ref=%s",
			event.transaction_id, event.amount, event.phone, event.reference,
		)
		await self._publish("ussd.payment_confirmed", event.model_dump())

	# ── STK Push Callback ─────────────────────────────────────────────────────

	async def handle_stk_callback(self, payload: dict[str, Any]) -> None:
		"""
		Process an STK Push callback result.

		Safaricom wraps the actual result inside ``Body.stkCallback``.  This
		method unwraps it, normalises to UsPaymentEvent, and emits NATS.

		Emits: ussd.stk_result

		Args:
			payload: Raw JSON dict from Safaricom STK Push callback.
		"""
		# Safaricom nests: {"Body": {"stkCallback": {...}}}
		stk_body = (
			payload.get("Body", payload)
				.get("stkCallback", payload.get("stkCallback", payload))
		)

		try:
			parsed = UsMpesaStkPayload.model_validate(stk_body)
		except Exception as exc:
			_log.error("stk_callback parse error: %s", exc)
			return

		success = parsed.ResultCode == 0
		event_type = "stk_success" if success else "stk_failed"

		# Extract amount and phone from CallbackMetadata on success
		amount = ""
		phone = ""
		if success and parsed.CallbackMetadata:
			items = parsed.CallbackMetadata.get("Item", [])
			for item in items:
				name = item.get("Name", "")
				value = str(item.get("Value", ""))
				if name == "Amount":
					amount = value
				elif name == "PhoneNumber":
					phone = value

		event = UsPaymentEvent(
			event_type=event_type,
			transaction_id=parsed.CheckoutRequestID,
			phone=phone,
			amount=amount,
			reference=parsed.MerchantRequestID,
			short_code="",
			timestamp=self._iso_now(),
			raw=dict(stk_body),
		)
		self._events.append(event)
		_log.info(
			"stk_callback: result=%s checkout=%s merchant=%s amount=%s",
			parsed.ResultCode, parsed.CheckoutRequestID,
			parsed.MerchantRequestID, amount,
		)
		await self._publish("ussd.stk_result", event.model_dump())

	# ── Audit ─────────────────────────────────────────────────────────────────

	def get_events(self, event_type: str | None = None, limit: int = 100) -> list[UsPaymentEvent]:
		"""Return buffered payment events, optionally filtered by type."""
		results = list(self._events)
		if event_type:
			results = [e for e in results if e.event_type == event_type]
		return results[-limit:]

	# ── Internal ──────────────────────────────────────────────────────────────

	def _iso_now(self) -> str:
		return datetime.now(timezone.utc).isoformat(timespec="seconds")

	async def _publish(self, subject: str, payload: dict[str, Any]) -> None:
		if self._nats is None:
			_log.debug("NATS event %s: %s", subject, payload)
			return
		try:
			import json
			await self._nats.publish(subject, json.dumps(payload, default=str).encode())
		except Exception as exc:
			_log.warning("NATS publish failed subject=%s: %s", subject, exc)
