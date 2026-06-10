"""Payment USSD App service — bill pay, merchant payment, airtime top-up,
utility pay, send money confirmation."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from copy import deepcopy
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string

_log = logging.getLogger(__name__)

SUPPORTED_TELCOS = {"safaricom", "airtel", "telkom", "faiba"}
SUPPORTED_UTILITY_CODES = {"kplc_prepaid", "kplc_postpaid", "nairobi_water", "mombasa_water", "kisumu_water", "nwsc"}
SUPPORTED_BILLER_CATEGORIES = {"utility", "insurance", "tax", "school", "government", "telco", "water", "internet", "media"}
SUPPORTED_CURRENCIES = {"KES", "USD", "EUR", "GBP", "UGX", "TZS", "RWF"}
MAX_AIRTIME_AMOUNT = Decimal("10000")
MIN_AIRTIME_AMOUNT = Decimal("5")
MAX_BILL_AMOUNT = Decimal("9999999")
SEND_MONEY_CONFIRMATION_THRESHOLD = Decimal("10000")  # amounts above this require explicit confirmation
MAX_PIN_ATTEMPTS = 3


def _now() -> str:
	return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _record_id(prefix: str, explicit: str | None = None) -> str:
	return explicit or f"{prefix}-{uuid4().hex[:12]}"


def _receipt_number() -> str:
	import random, string
	return "".join(random.choices(string.ascii_uppercase + string.digits, k=10))


def _hash_pin(pin: str) -> str:
	return hashlib.sha256(pin.encode()).hexdigest()


# Built-in biller registry (bootstrap data)
_DEFAULT_BILLERS: list[dict[str, Any]] = [
	{"biller_code": "KPLC_PRE", "biller_name": "Kenya Power Prepaid", "category": "utility", "paybill_number": "888880"},
	{"biller_code": "KPLC_POST", "biller_name": "Kenya Power Postpaid", "category": "utility", "paybill_number": "888882"},
	{"biller_code": "NWC", "biller_name": "Nairobi Water", "category": "water", "paybill_number": "888861"},
	{"biller_code": "KRA", "biller_name": "Kenya Revenue Authority", "category": "tax", "paybill_number": "572572"},
	{"biller_code": "NHIF", "biller_name": "National Hospital Insurance Fund", "category": "insurance", "paybill_number": "200222"},
	{"biller_code": "NSSF", "biller_name": "National Social Security Fund", "category": "insurance", "paybill_number": "333200"},
	{"biller_code": "DStv", "biller_name": "DStv Africa", "category": "media", "paybill_number": "444200"},
	{"biller_code": "ZUKU", "biller_name": "Zuku Internet", "category": "internet", "paybill_number": "100600"},
	{"biller_code": "SAFARICOM_POSTPAID", "biller_name": "Safaricom Postpaid", "category": "telco", "paybill_number": "100200"},
]

# Built-in merchant registry (bootstrap data)
_DEFAULT_MERCHANTS: dict[str, str] = {
	"174379": "Carrefour Kenya",
	"522533": "Naivas Supermarket",
	"247247": "Shell Kenya",
	"200000": "KCB Bank",
	"300632": "Equity Bank",
}


class PayUssdService:
	"""Async service for USSD payment operations."""

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id
		# In-memory stores
		self.bill_payments: dict[str, dict[str, Any]] = {}
		self.merchant_payments: dict[str, dict[str, Any]] = {}
		self.airtime_topups: dict[str, dict[str, Any]] = {}
		self.utility_payments: dict[str, dict[str, Any]] = {}
		self.send_money_txns: dict[str, dict[str, Any]] = {}
		self.billers: dict[str, dict[str, Any]] = {}
		self.ussd_sessions: dict[str, dict[str, Any]] = {}
		self.pin_store: dict[str, str] = {}  # phone_number -> pin_hash
		self.pin_attempts: dict[str, int] = {}  # phone_number -> count
		self._audit_events: list[dict[str, Any]] = []
		# Seed built-in billers
		self._seed_billers()

	# ── Internal helpers ──────────────────────────────────────────────────────

	def _tenant(self, tenant_id: str | None = None) -> str:
		value = tenant_id or self.tenant_id
		guard_tenant_id(value)
		return value

	def _emit(self, tenant_id: str, event_type: str, record_id: str, record_type: str,
			  phone_number: str | None = None, amount: str | None = None, metadata: dict[str, Any] | None = None) -> None:
		self._audit_events.append({
			"id": _record_id("evt"),
			"tenant_id": tenant_id,
			"event_type": event_type,
			"record_id": record_id,
			"record_type": record_type,
			"phone_number": phone_number,
			"amount": amount,
			"metadata": metadata or {},
			"emitted_at": _now(),
		})

	def _validate_phone(self, phone: str) -> str:
		cleaned = re.sub(r"[^0-9]", "", phone)
		if cleaned.startswith("254") and len(cleaned) == 12:
			return cleaned
		if cleaned.startswith("0") and len(cleaned) == 10:
			return "254" + cleaned[1:]
		if cleaned.startswith("7") and len(cleaned) == 9:
			return "254" + cleaned
		raise ValueError(f"invalid_phone_number: {phone}")

	def _check_pin(self, phone_number: str) -> None:
		"""Verify PIN is registered for phone (stub — in production, delegates to MobUssdService)."""
		stored = self.pin_store.get(phone_number)
		if not stored:
			_log.debug("pin_not_registered_for_phone=%s, skipping verification", phone_number)

	def _verify_pin(self, phone_number: str, pin: str) -> bool:
		"""Verify PIN from local store; returns True if not registered (guest mode)."""
		stored = self.pin_store.get(phone_number)
		if stored is None:
			return True  # Guest mode: no PIN registered yet
		attempts = self.pin_attempts.get(phone_number, 0)
		if attempts >= MAX_PIN_ATTEMPTS:
			raise PermissionError("phone_locked_too_many_pin_attempts")
		if stored != _hash_pin(pin):
			self.pin_attempts[phone_number] = attempts + 1
			remaining = MAX_PIN_ATTEMPTS - self.pin_attempts[phone_number]
			raise PermissionError(f"invalid_pin: {remaining} attempts remaining")
		self.pin_attempts[phone_number] = 0
		return True

	def _lookup_biller(self, biller_code: str, tenant_id: str) -> dict[str, Any] | None:
		for b in self.billers.values():
			if b["biller_code"] == biller_code and (b["tenant_id"] == tenant_id or b.get("built_in")):
				return b
		return None

	def _lookup_merchant(self, till: str) -> str:
		return _DEFAULT_MERCHANTS.get(till, f"Merchant {till}")

	def _seed_billers(self) -> None:
		for b in _DEFAULT_BILLERS:
			bid = _record_id("biller")
			self.billers[bid] = {
				"id": bid,
				"type": "pay_biller",
				"biller_code": b["biller_code"],
				"biller_name": b["biller_name"],
				"category": b["category"],
				"paybill_number": b["paybill_number"],
				"account_mask": b.get("account_mask", ""),
				"min_amount": Decimal("1"),
				"max_amount": Decimal("9999999"),
				"status": "active",
				"tenant_id": "system",
				"built_in": True,
				"created_at": _now(),
			}

	def _build_ussd_menu(self, level: int, context: dict[str, Any]) -> tuple[str, bool]:
		"""Build USSD menu text and continuation flag."""
		if level == 0:
			return (
				"CON Welcome to PayUSSD\n1. Pay Bill\n2. Pay Merchant\n"
				"3. Buy Airtime\n4. Pay Utility\n5. Send Money\n0. Exit",
				True,
			)
		if level == 1:
			choice = context.get("choice", "")
			if choice == "1":
				return ("CON Pay Bill\nEnter Paybill number:", True)
			if choice == "2":
				return ("CON Pay Merchant\nEnter Till number:", True)
			if choice == "3":
				return ("CON Buy Airtime\nEnter phone number:", True)
			if choice == "4":
				return ("CON Pay Utility\nEnter utility code\n(KPLC_PRE/KPLC_POST/NWC):", True)
			if choice == "5":
				return ("CON Send Money\nEnter recipient phone:", True)
		if level == 2:
			choice = context.get("choice", "")
			if choice == "5" and context.get("pending_amount"):
				amt = context["pending_amount"]
				to = context.get("to_phone", "")
				return (f"CON Confirm send KES {amt} to {to}\n1. Confirm\n2. Cancel", True)
		return ("END Invalid option. Please try again.", False)

	# ── PIN registration (for payment USSD standalone) ─────────────────────────

	async def register_phone_pin(self, phone_number: str, pin: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Register a PIN for a phone number for payment authorization."""
		tenant = self._tenant(tenant_id)
		normalized = self._validate_phone(phone_number)
		if not pin.isdigit() or not (4 <= len(pin) <= 6):
			raise ValueError("pin_must_be_4_to_6_digits")
		self.pin_store[normalized] = _hash_pin(pin)
		self.pin_attempts[normalized] = 0
		self._emit(tenant, "pay_pin_registered", normalized, "pay_phone", phone_number=normalized)
		return {"phone_number": normalized, "status": "pin_registered", "registered_at": _now()}

	# ── Biller management ─────────────────────────────────────────────────────

	async def create_biller(
		self,
		biller_code: str,
		biller_name: str,
		category: str,
		paybill_number: str,
		account_mask: str = "",
		min_amount: Decimal = Decimal("1"),
		max_amount: Decimal = Decimal("9999999"),
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Register a new biller."""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(biller_code, "biller_code")
		guard_non_empty_string(biller_name, "biller_name")
		if category not in SUPPORTED_BILLER_CATEGORIES:
			raise ValueError(f"unsupported_biller_category: {category}")
		if self._lookup_biller(biller_code, tenant):
			raise ValueError(f"biller_code_already_exists: {biller_code}")
		biller_id = _record_id("pay-biller")
		record = {
			"id": biller_id,
			"type": "pay_biller",
			"biller_code": biller_code,
			"biller_name": biller_name,
			"category": category,
			"paybill_number": paybill_number,
			"account_mask": account_mask,
			"min_amount": min_amount,
			"max_amount": max_amount,
			"status": "active",
			"tenant_id": tenant,
			"built_in": False,
			"created_at": _now(),
		}
		self.billers[biller_id] = record
		self._emit(tenant, "pay_biller_created", biller_id, "pay_biller")
		_log.info("pay_biller_created code=%s tenant=%s", biller_code, tenant)
		return deepcopy(record)

	async def get_biller(self, biller_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Retrieve a biller by ID."""
		tenant = self._tenant(tenant_id)
		record = self.billers.get(biller_id)
		if not record or (record["tenant_id"] != tenant and not record.get("built_in")):
			raise KeyError(f"biller_not_found: {biller_id}")
		return deepcopy(record)

	async def list_billers(self, category: str | None = None, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""List all active billers (built-in + tenant-specific)."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(b) for b in self.billers.values()
				 if b["status"] == "active" and (b["tenant_id"] == tenant or b.get("built_in"))]
		if category:
			items = [b for b in items if b["category"] == category]
		return items

	async def update_biller(self, biller_id: str, biller_name: str | None = None, status: str | None = None, tenant_id: str | None = None) -> dict[str, Any]:
		"""Update a biller record."""
		tenant = self._tenant(tenant_id)
		record = self.billers.get(biller_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"biller_not_found: {biller_id}")
		if biller_name:
			record["biller_name"] = biller_name
		if status:
			record["status"] = status
		record["updated_at"] = _now()
		self._emit(tenant, "pay_biller_updated", biller_id, "pay_biller")
		return deepcopy(record)

	async def delete_biller(self, biller_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Deactivate a biller."""
		tenant = self._tenant(tenant_id)
		record = self.billers.get(biller_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"biller_not_found: {biller_id}")
		if record.get("built_in"):
			raise PermissionError("cannot_delete_built_in_biller")
		record["status"] = "inactive"
		record["deactivated_at"] = _now()
		self._emit(tenant, "pay_biller_deactivated", biller_id, "pay_biller")
		return deepcopy(record)

	# ── Bill payment ──────────────────────────────────────────────────────────

	async def pay_bill(
		self,
		phone_number: str,
		biller_code: str,
		account_reference: str,
		amount: Decimal,
		pin: str,
		narration: str = "",
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Pay a bill via paybill number."""
		tenant = self._tenant(tenant_id)
		normalized = self._validate_phone(phone_number)
		self._verify_pin(normalized, pin)
		biller = self._lookup_biller(biller_code, tenant)
		if not biller:
			raise KeyError(f"biller_not_found: {biller_code}")
		if biller["status"] != "active":
			raise PermissionError(f"biller_{biller['status']}")
		if amount < biller["min_amount"] or amount > biller["max_amount"]:
			raise ValueError(f"amount_outside_biller_limits: {biller['min_amount']} - {biller['max_amount']}")
		guard_non_empty_string(account_reference, "account_reference")
		pay_id = _record_id("pay-bill")
		record = {
			"id": pay_id,
			"type": "pay_bill_payment",
			"phone_number": normalized,
			"biller_code": biller_code,
			"biller_name": biller["biller_name"],
			"paybill_number": biller["paybill_number"],
			"account_reference": account_reference,
			"amount": str(amount),
			"currency": "KES",
			"narration": narration or f"Bill payment - {biller['biller_name']}",
			"receipt_number": _receipt_number(),
			"status": "completed",
			"tenant_id": tenant,
			"created_at": _now(),
			"completed_at": _now(),
		}
		self.bill_payments[pay_id] = record
		self._emit(tenant, "pay_bill_paid", pay_id, "pay_bill_payment",
				   phone_number=normalized, amount=str(amount),
				   metadata={"biller_code": biller_code, "account_reference": account_reference})
		_log.info("pay_bill_paid biller=%s phone=%s amount=%s tenant=%s", biller_code, normalized, amount, tenant)
		return deepcopy(record)

	async def get_bill_payment(self, payment_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Retrieve a bill payment record."""
		tenant = self._tenant(tenant_id)
		record = self.bill_payments.get(payment_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"bill_payment_not_found: {payment_id}")
		return deepcopy(record)

	async def list_bill_payments(self, phone_number: str | None = None, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""List bill payments for a tenant."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.bill_payments.values() if r["tenant_id"] == tenant]
		if phone_number:
			normalized = self._validate_phone(phone_number)
			items = [r for r in items if r["phone_number"] == normalized]
		return items

	async def reverse_bill_payment(self, payment_id: str, reason: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Reverse a completed bill payment."""
		tenant = self._tenant(tenant_id)
		record = self.bill_payments.get(payment_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"bill_payment_not_found: {payment_id}")
		if record["status"] != "completed":
			raise PermissionError("only_completed_payments_can_be_reversed")
		record["status"] = "reversed"
		record["reversal_reason"] = reason
		record["reversed_at"] = _now()
		self._emit(tenant, "pay_bill_reversed", payment_id, "pay_bill_payment",
				   phone_number=record["phone_number"], amount=record["amount"])
		return deepcopy(record)

	# ── Merchant payment ──────────────────────────────────────────────────────

	async def pay_merchant(
		self,
		phone_number: str,
		merchant_till: str,
		amount: Decimal,
		pin: str,
		narration: str = "",
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Pay a merchant via till/buy goods number."""
		tenant = self._tenant(tenant_id)
		normalized = self._validate_phone(phone_number)
		self._verify_pin(normalized, pin)
		if amount <= Decimal("0"):
			raise ValueError("merchant_payment_amount_must_be_positive")
		guard_non_empty_string(merchant_till, "merchant_till")
		merchant_name = self._lookup_merchant(merchant_till)
		pay_id = _record_id("pay-mcht")
		record = {
			"id": pay_id,
			"type": "pay_merchant_payment",
			"phone_number": normalized,
			"merchant_till": merchant_till,
			"merchant_name": merchant_name,
			"amount": str(amount),
			"currency": "KES",
			"narration": narration or f"Merchant payment - {merchant_name}",
			"receipt_number": _receipt_number(),
			"status": "completed",
			"tenant_id": tenant,
			"created_at": _now(),
		}
		self.merchant_payments[pay_id] = record
		self._emit(tenant, "pay_merchant_paid", pay_id, "pay_merchant_payment",
				   phone_number=normalized, amount=str(amount),
				   metadata={"merchant_till": merchant_till})
		_log.info("pay_merchant_paid till=%s phone=%s amount=%s tenant=%s", merchant_till, normalized, amount, tenant)
		return deepcopy(record)

	async def get_merchant_payment(self, payment_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Retrieve a merchant payment record."""
		tenant = self._tenant(tenant_id)
		record = self.merchant_payments.get(payment_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"merchant_payment_not_found: {payment_id}")
		return deepcopy(record)

	async def list_merchant_payments(self, phone_number: str | None = None, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""List merchant payments for a tenant."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.merchant_payments.values() if r["tenant_id"] == tenant]
		if phone_number:
			normalized = self._validate_phone(phone_number)
			items = [r for r in items if r["phone_number"] == normalized]
		return items

	# ── Airtime top-up ────────────────────────────────────────────────────────

	async def buy_airtime(
		self,
		phone_number: str,
		recipient_phone: str,
		amount: Decimal,
		pin: str,
		telco: str = "safaricom",
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Purchase airtime for a phone number."""
		tenant = self._tenant(tenant_id)
		normalized_buyer = self._validate_phone(phone_number)
		normalized_recipient = self._validate_phone(recipient_phone)
		self._verify_pin(normalized_buyer, pin)
		if telco not in SUPPORTED_TELCOS:
			raise ValueError(f"unsupported_telco: {telco}")
		if amount < MIN_AIRTIME_AMOUNT:
			raise ValueError(f"airtime_minimum_is_{MIN_AIRTIME_AMOUNT}")
		if amount > MAX_AIRTIME_AMOUNT:
			raise ValueError(f"airtime_maximum_is_{MAX_AIRTIME_AMOUNT}")
		topup_id = _record_id("pay-air")
		record = {
			"id": topup_id,
			"type": "pay_airtime_topup",
			"phone_number": normalized_buyer,
			"recipient_phone": normalized_recipient,
			"telco": telco,
			"amount": str(amount),
			"currency": "KES",
			"receipt_number": _receipt_number(),
			"status": "completed",
			"tenant_id": tenant,
			"created_at": _now(),
		}
		self.airtime_topups[topup_id] = record
		self._emit(tenant, "pay_airtime_purchased", topup_id, "pay_airtime_topup",
				   phone_number=normalized_buyer, amount=str(amount),
				   metadata={"recipient_phone": normalized_recipient, "telco": telco})
		_log.info("pay_airtime_purchased buyer=%s recipient=%s amount=%s telco=%s tenant=%s",
				  normalized_buyer, normalized_recipient, amount, telco, tenant)
		return deepcopy(record)

	async def get_airtime_topup(self, topup_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Retrieve an airtime top-up record."""
		tenant = self._tenant(tenant_id)
		record = self.airtime_topups.get(topup_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"airtime_topup_not_found: {topup_id}")
		return deepcopy(record)

	async def list_airtime_topups(self, phone_number: str | None = None, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""List airtime top-ups for a tenant."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.airtime_topups.values() if r["tenant_id"] == tenant]
		if phone_number:
			normalized = self._validate_phone(phone_number)
			items = [r for r in items if r["phone_number"] == normalized]
		return items

	# ── Utility payment ───────────────────────────────────────────────────────

	async def pay_utility(
		self,
		phone_number: str,
		utility_code: str,
		meter_number: str,
		amount: Decimal,
		pin: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Pay a utility bill (electricity, water) and receive token if applicable."""
		tenant = self._tenant(tenant_id)
		normalized = self._validate_phone(phone_number)
		self._verify_pin(normalized, pin)
		if utility_code not in SUPPORTED_UTILITY_CODES:
			raise ValueError(f"unsupported_utility_code: {utility_code}")
		guard_non_empty_string(meter_number, "meter_number")
		if amount <= Decimal("0"):
			raise ValueError("utility_payment_amount_must_be_positive")
		# Compute units for prepaid electricity (crude simulation)
		units_purchased = None
		token = None
		utility_name_map = {
			"kplc_prepaid": "Kenya Power Prepaid",
			"kplc_postpaid": "Kenya Power Postpaid",
			"nairobi_water": "Nairobi Water",
			"mombasa_water": "Mombasa Water",
			"kisumu_water": "Kisumu Water",
			"nwsc": "National Water & Sewerage Corp",
		}
		utility_name = utility_name_map.get(utility_code, utility_code)
		if utility_code == "kplc_prepaid":
			# Rough tariff: KES 20/unit (Band A residential)
			units = float(amount) / 20.0
			units_purchased = f"{units:.2f} kWh"
			import random
			token = "-".join([str(random.randint(1000, 9999)) for _ in range(5)])
		pay_id = _record_id("pay-util")
		record = {
			"id": pay_id,
			"type": "pay_utility_payment",
			"phone_number": normalized,
			"utility_code": utility_code,
			"utility_name": utility_name,
			"meter_number": meter_number,
			"amount": str(amount),
			"currency": "KES",
			"units_purchased": units_purchased,
			"token": token,
			"receipt_number": _receipt_number(),
			"status": "completed",
			"tenant_id": tenant,
			"created_at": _now(),
		}
		self.utility_payments[pay_id] = record
		self._emit(tenant, "pay_utility_paid", pay_id, "pay_utility_payment",
				   phone_number=normalized, amount=str(amount),
				   metadata={"utility_code": utility_code, "meter_number": meter_number})
		_log.info("pay_utility_paid code=%s meter=%s amount=%s tenant=%s", utility_code, meter_number, amount, tenant)
		return deepcopy(record)

	async def get_utility_payment(self, payment_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Retrieve a utility payment record."""
		tenant = self._tenant(tenant_id)
		record = self.utility_payments.get(payment_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"utility_payment_not_found: {payment_id}")
		return deepcopy(record)

	async def list_utility_payments(self, phone_number: str | None = None, utility_code: str | None = None, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""List utility payments, optionally filtered by phone or utility code."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.utility_payments.values() if r["tenant_id"] == tenant]
		if phone_number:
			normalized = self._validate_phone(phone_number)
			items = [r for r in items if r["phone_number"] == normalized]
		if utility_code:
			items = [r for r in items if r["utility_code"] == utility_code]
		return items

	# ── Send money ────────────────────────────────────────────────────────────

	async def initiate_send_money(
		self,
		from_phone: str,
		to_phone: str,
		amount: Decimal,
		pin: str,
		narration: str = "",
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Initiate a send-money transaction; large amounts require confirmation."""
		tenant = self._tenant(tenant_id)
		normalized_from = self._validate_phone(from_phone)
		normalized_to = self._validate_phone(to_phone)
		self._verify_pin(normalized_from, pin)
		if amount <= Decimal("0"):
			raise ValueError("send_money_amount_must_be_positive")
		if normalized_from == normalized_to:
			raise ValueError("cannot_send_money_to_self")
		requires_confirmation = amount >= SEND_MONEY_CONFIRMATION_THRESHOLD
		txn_id = _record_id("pay-send")
		status = "pending_confirmation" if requires_confirmation else "completed"
		record = {
			"id": txn_id,
			"type": "pay_send_money",
			"from_phone": normalized_from,
			"to_phone": normalized_to,
			"amount": str(amount),
			"currency": "KES",
			"narration": narration or f"Send money to {normalized_to}",
			"receipt_number": _receipt_number() if not requires_confirmation else None,
			"status": status,
			"requires_confirmation": requires_confirmation,
			"tenant_id": tenant,
			"created_at": _now(),
			"confirmed_at": _now() if not requires_confirmation else None,
		}
		self.send_money_txns[txn_id] = record
		event_type = "pay_send_money_pending" if requires_confirmation else "pay_send_money_completed"
		self._emit(tenant, event_type, txn_id, "pay_send_money",
				   phone_number=normalized_from, amount=str(amount),
				   metadata={"to_phone": normalized_to, "requires_confirmation": requires_confirmation})
		if not requires_confirmation:
			_log.info("pay_send_money_completed from=%s to=%s amount=%s tenant=%s",
					  normalized_from, normalized_to, amount, tenant)
		else:
			_log.info("pay_send_money_pending from=%s to=%s amount=%s (awaiting confirmation) tenant=%s",
					  normalized_from, normalized_to, amount, tenant)
		return deepcopy(record)

	async def confirm_send_money(self, transaction_id: str, pin: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Confirm a pending send-money transaction."""
		tenant = self._tenant(tenant_id)
		record = self.send_money_txns.get(transaction_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"send_money_transaction_not_found: {transaction_id}")
		if record["status"] != "pending_confirmation":
			raise PermissionError("transaction_not_pending_confirmation")
		self._verify_pin(record["from_phone"], pin)
		record["status"] = "completed"
		record["receipt_number"] = _receipt_number()
		record["confirmed_at"] = _now()
		self._emit(tenant, "pay_send_money_confirmed", transaction_id, "pay_send_money",
				   phone_number=record["from_phone"], amount=record["amount"])
		_log.info("pay_send_money_confirmed id=%s from=%s to=%s amount=%s tenant=%s",
				  transaction_id, record["from_phone"], record["to_phone"], record["amount"], tenant)
		return deepcopy(record)

	async def cancel_send_money(self, transaction_id: str, reason: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Cancel a pending send-money transaction."""
		tenant = self._tenant(tenant_id)
		record = self.send_money_txns.get(transaction_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"send_money_transaction_not_found: {transaction_id}")
		if record["status"] not in {"pending_confirmation", "completed"}:
			raise PermissionError(f"cannot_cancel_{record['status']}_transaction")
		record["status"] = "cancelled"
		record["cancellation_reason"] = reason
		record["cancelled_at"] = _now()
		self._emit(tenant, "pay_send_money_cancelled", transaction_id, "pay_send_money",
				   phone_number=record["from_phone"])
		return deepcopy(record)

	async def get_send_money_transaction(self, transaction_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Retrieve a send-money transaction."""
		tenant = self._tenant(tenant_id)
		record = self.send_money_txns.get(transaction_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"send_money_transaction_not_found: {transaction_id}")
		return deepcopy(record)

	async def list_send_money_transactions(self, phone_number: str | None = None, status: str | None = None, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""List send-money transactions for a tenant."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(r) for r in self.send_money_txns.values() if r["tenant_id"] == tenant]
		if phone_number:
			normalized = self._validate_phone(phone_number)
			items = [r for r in items if r["from_phone"] == normalized or r["to_phone"] == normalized]
		if status:
			items = [r for r in items if r["status"] == status]
		return items

	# ── USSD session ──────────────────────────────────────────────────────────

	async def handle_ussd_request(
		self,
		session_id: str,
		phone_number: str,
		service_code: str,
		input_text: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Process a USSD payment request and return the appropriate menu."""
		tenant = self._tenant(tenant_id)
		normalized = self._validate_phone(phone_number)
		session = self.ussd_sessions.get(session_id, {
			"session_id": session_id,
			"phone_number": normalized,
			"service_code": service_code,
			"menu_level": 0,
			"context": {},
			"inputs": [],
			"tenant_id": tenant,
			"created_at": _now(),
		})
		session["last_activity"] = _now()
		inputs = [x for x in input_text.split("*") if x] if input_text else []
		level = len(inputs)
		context: dict[str, Any] = {}
		pending_transaction_id = None
		if level >= 1:
			context["choice"] = inputs[0]
			# Send money: 2-step confirmation
			if inputs[0] == "5" and level >= 3:
				to_phone = inputs[1] if len(inputs) > 1 else ""
				raw_amount = inputs[2] if len(inputs) > 2 else "0"
				try:
					amt = Decimal(raw_amount)
					context["to_phone"] = to_phone
					context["pending_amount"] = str(amt)
					if amt >= SEND_MONEY_CONFIRMATION_THRESHOLD and level == 3:
						# Pending confirmation step
						pending_transaction_id = f"pending-{session_id}"
				except Exception:
					pass
		response_text, continues = self._build_ussd_menu(min(level, 2) if level > 1 else min(level, 1), context)
		session["menu_level"] = level
		session["inputs"] = inputs
		session["context"] = context
		session["pending_transaction_id"] = pending_transaction_id
		if not continues:
			session["ended_at"] = _now()
		self.ussd_sessions[session_id] = session
		self._emit(tenant, "pay_ussd_request", session_id, "pay_ussd_session", phone_number=normalized,
				   metadata={"level": level})
		return {
			"session_id": session_id,
			"phone_number": normalized,
			"menu_level": level,
			"response_text": response_text,
			"continues": continues,
			"pending_transaction_id": pending_transaction_id,
			"tenant_id": tenant,
			"created_at": session["created_at"],
			"last_activity": session["last_activity"],
		}

	async def get_ussd_session(self, session_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Retrieve a USSD session."""
		tenant = self._tenant(tenant_id)
		session = self.ussd_sessions.get(session_id)
		if not session or session["tenant_id"] != tenant:
			raise KeyError(f"ussd_session_not_found: {session_id}")
		return deepcopy(session)

	async def list_ussd_sessions(self, phone_number: str | None = None, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""List USSD sessions for a tenant."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(s) for s in self.ussd_sessions.values() if s["tenant_id"] == tenant]
		if phone_number:
			normalized = self._validate_phone(phone_number)
			items = [s for s in items if s["phone_number"] == normalized]
		return items

	# ── Payment history & analytics ───────────────────────────────────────────

	async def get_payment_history(self, phone_number: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Return consolidated payment history across all payment types for a phone."""
		tenant = self._tenant(tenant_id)
		normalized = self._validate_phone(phone_number)
		results = await asyncio.gather(
			self.list_bill_payments(phone_number=normalized, tenant_id=tenant),
			self.list_merchant_payments(phone_number=normalized, tenant_id=tenant),
			self.list_airtime_topups(phone_number=normalized, tenant_id=tenant),
			self.list_utility_payments(phone_number=normalized, tenant_id=tenant),
			self.list_send_money_transactions(phone_number=normalized, tenant_id=tenant),
			return_exceptions=True,
		)
		bills = results[0] if not isinstance(results[0], Exception) else []
		merchants = results[1] if not isinstance(results[1], Exception) else []
		airtime = results[2] if not isinstance(results[2], Exception) else []
		utilities = results[3] if not isinstance(results[3], Exception) else []
		sends = results[4] if not isinstance(results[4], Exception) else []
		total_bill = sum(Decimal(r["amount"]) for r in (bills or []))
		total_merchant = sum(Decimal(r["amount"]) for r in (merchants or []))
		total_airtime = sum(Decimal(r["amount"]) for r in (airtime or []))
		total_utility = sum(Decimal(r["amount"]) for r in (utilities or []))
		total_send = sum(Decimal(r["amount"]) for r in (sends or []) if r.get("status") == "completed")
		return {
			"phone_number": normalized,
			"bill_payments": bills,
			"merchant_payments": merchants,
			"airtime_topups": airtime,
			"utility_payments": utilities,
			"send_money_transactions": sends,
			"summary": {
				"bill_total": str(total_bill),
				"merchant_total": str(total_merchant),
				"airtime_total": str(total_airtime),
				"utility_total": str(total_utility),
				"send_money_total": str(total_send),
				"grand_total": str(total_bill + total_merchant + total_airtime + total_utility + total_send),
			},
			"generated_at": _now(),
		}

	async def get_tenant_statistics(self, tenant_id: str | None = None) -> dict[str, Any]:
		"""Return aggregate statistics for a tenant."""
		tenant = self._tenant(tenant_id)
		bills = [r for r in self.bill_payments.values() if r["tenant_id"] == tenant]
		merchants = [r for r in self.merchant_payments.values() if r["tenant_id"] == tenant]
		airtime = [r for r in self.airtime_topups.values() if r["tenant_id"] == tenant]
		utilities = [r for r in self.utility_payments.values() if r["tenant_id"] == tenant]
		sends = [r for r in self.send_money_txns.values() if r["tenant_id"] == tenant]
		return {
			"tenant_id": tenant,
			"bill_payment_count": len(bills),
			"merchant_payment_count": len(merchants),
			"airtime_topup_count": len(airtime),
			"utility_payment_count": len(utilities),
			"send_money_count": len(sends),
			"total_bill_volume": str(sum(Decimal(r["amount"]) for r in bills)),
			"total_merchant_volume": str(sum(Decimal(r["amount"]) for r in merchants)),
			"total_airtime_volume": str(sum(Decimal(r["amount"]) for r in airtime)),
			"total_utility_volume": str(sum(Decimal(r["amount"]) for r in utilities)),
			"total_send_money_volume": str(sum(Decimal(r["amount"]) for r in sends if r.get("status") == "completed")),
			"pending_confirmations": sum(1 for r in sends if r.get("status") == "pending_confirmation"),
			"audit_event_count": len([e for e in self._audit_events if e["tenant_id"] == tenant]),
			"generated_at": _now(),
		}

	async def get_daily_volume(self, tenant_id: str | None = None, date: str | None = None) -> dict[str, Any]:
		"""Return daily payment volume for a given date (defaults to today)."""
		tenant = self._tenant(tenant_id)
		target_date = date or _now()[:10]
		bills = [r for r in self.bill_payments.values() if r["tenant_id"] == tenant and r["created_at"][:10] == target_date]
		merchants = [r for r in self.merchant_payments.values() if r["tenant_id"] == tenant and r["created_at"][:10] == target_date]
		airtime = [r for r in self.airtime_topups.values() if r["tenant_id"] == tenant and r["created_at"][:10] == target_date]
		utilities = [r for r in self.utility_payments.values() if r["tenant_id"] == tenant and r["created_at"][:10] == target_date]
		sends = [r for r in self.send_money_txns.values() if r["tenant_id"] == tenant and r["created_at"][:10] == target_date and r["status"] == "completed"]
		return {
			"tenant_id": tenant,
			"date": target_date,
			"bill_payments": len(bills),
			"merchant_payments": len(merchants),
			"airtime_topups": len(airtime),
			"utility_payments": len(utilities),
			"send_money": len(sends),
			"total_transactions": len(bills) + len(merchants) + len(airtime) + len(utilities) + len(sends),
			"total_volume": str(
				sum(Decimal(r["amount"]) for r in bills + merchants + airtime + utilities + sends)
			),
			"generated_at": _now(),
		}

	async def search_payments(
		self,
		phone_number: str | None = None,
		payment_type: str | None = None,
		date_from: str | None = None,
		date_to: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Search payments across all types with optional filters."""
		tenant = self._tenant(tenant_id)
		all_payments: list[dict[str, Any]] = []
		if not payment_type or payment_type == "bill":
			all_payments += [dict(r, payment_type="bill") for r in self.bill_payments.values() if r["tenant_id"] == tenant]
		if not payment_type or payment_type == "merchant":
			all_payments += [dict(r, payment_type="merchant") for r in self.merchant_payments.values() if r["tenant_id"] == tenant]
		if not payment_type or payment_type == "airtime":
			all_payments += [dict(r, payment_type="airtime") for r in self.airtime_topups.values() if r["tenant_id"] == tenant]
		if not payment_type or payment_type == "utility":
			all_payments += [dict(r, payment_type="utility") for r in self.utility_payments.values() if r["tenant_id"] == tenant]
		if not payment_type or payment_type == "send_money":
			all_payments += [dict(r, payment_type="send_money") for r in self.send_money_txns.values() if r["tenant_id"] == tenant]
		if phone_number:
			normalized = self._validate_phone(phone_number)
			all_payments = [r for r in all_payments if r.get("phone_number") == normalized or r.get("from_phone") == normalized]
		if date_from:
			all_payments = [r for r in all_payments if r["created_at"][:10] >= date_from]
		if date_to:
			all_payments = [r for r in all_payments if r["created_at"][:10] <= date_to]
		all_payments.sort(key=lambda r: r["created_at"], reverse=True)
		return {"results": all_payments, "total": len(all_payments), "generated_at": _now()}

	# ── Utility ───────────────────────────────────────────────────────────────

	async def health_check(self) -> dict[str, Any]:
		"""Return service health status."""
		return {
			"service": "fintech_ussd_pay",
			"status": "healthy",
			"biller_count": len(self.billers),
			"bill_payment_count": len(self.bill_payments),
			"merchant_payment_count": len(self.merchant_payments),
			"airtime_topup_count": len(self.airtime_topups),
			"utility_payment_count": len(self.utility_payments),
			"send_money_count": len(self.send_money_txns),
			"pending_confirmations": sum(1 for r in self.send_money_txns.values() if r.get("status") == "pending_confirmation"),
			"checked_at": _now(),
		}

	async def get_audit_events(self, tenant_id: str | None = None, event_type: str | None = None) -> list[dict[str, Any]]:
		"""Return audit events for a tenant."""
		tenant = self._tenant(tenant_id)
		events = [deepcopy(e) for e in self._audit_events if e["tenant_id"] == tenant]
		if event_type:
			events = [e for e in events if e["event_type"] == event_type]
		return events

	async def describe(self) -> dict[str, Any]:
		"""Return capability description and metadata."""
		return {
			"capability_id": "fintech_ussd_pay",
			"name": "Payment USSD App",
			"description": "USSD payments: bill pay, merchant payment, airtime top-up, utility pay, send money confirmation",
			"version": "1.0.0",
			"domain": "fintech",
			"features": [
				"bill_payment",
				"merchant_payment",
				"airtime_topup",
				"utility_payment",
				"send_money_with_confirmation",
				"biller_registry",
				"payment_history",
				"ussd_session_handling",
				"daily_volume_reporting",
				"payment_search",
			],
			"supported_telcos": list(SUPPORTED_TELCOS),
			"supported_utility_codes": list(SUPPORTED_UTILITY_CODES),
			"supported_biller_categories": list(SUPPORTED_BILLER_CATEGORIES),
			"send_money_confirmation_threshold": str(SEND_MONEY_CONFIRMATION_THRESHOLD),
			"max_airtime_amount": str(MAX_AIRTIME_AMOUNT),
			"built_in_billers": len([b for b in self.billers.values() if b.get("built_in")]),
		}
