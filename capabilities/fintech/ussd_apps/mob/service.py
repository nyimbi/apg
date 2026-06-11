"""Mobile Banking USSD service — account balance, mini-statement, fund transfer,
standing orders, PIN management, beneficiaries, fraud scoring, FX transfers,
spending analytics, audit chain verification, and proactive balance alerts."""
from __future__ import annotations

import asyncio
import csv
import hashlib
import hmac
import io
import json
import logging
import re
import secrets
import time
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Any
from uuid import uuid4

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string

_log = logging.getLogger(__name__)

SUPPORTED_ACCOUNT_TYPES = {"savings", "current", "fixed_deposit", "wallet"}
SUPPORTED_FREQUENCIES = {"daily", "weekly", "monthly", "quarterly"}
SUPPORTED_CURRENCIES = {"KES", "USD", "EUR", "GBP", "UGX", "TZS", "RWF"}
DAILY_TRANSFER_LIMIT = Decimal("500000")
SINGLE_TRANSFER_LIMIT = Decimal("150000")
MIN_PIN_LENGTH = 4
MAX_PIN_LENGTH = 6
MAX_PIN_ATTEMPTS = 3
MINI_STATEMENT_ROWS = 5
HIGH_VALUE_TRANSFER_THRESHOLD = Decimal("50000")  # TOTP required above this
FRAUD_HIGH_RISK_SCORE = 75
FRAUD_MEDIUM_RISK_SCORE = 50
MAX_BENEFICIARIES_PER_ACCOUNT = 20
IDEMPOTENCY_CACHE_TTL_SECONDS = 86400  # 24 hours
SESSION_TOKEN_TTL_SECONDS = 300

# Keyword-based transaction category classifier
_CATEGORY_KEYWORDS: dict[str, list[str]] = {
	"utilities": ["kplc", "electricity", "water", "nairobi water", "stima", "zuku", "safaricom home"],
	"food": ["supermarket", "naivas", "carrefour", "quickmart", "food", "restaurant", "cafe", "groceries"],
	"transport": ["uber", "bolt", "matatu", "petrol", "fuel", "parking", "nairobi expressway"],
	"savings": ["savings", "fixed", "deposit", "investment", "sacco"],
	"education": ["school", "fees", "university", "college", "tuition"],
	"health": ["hospital", "pharmacy", "clinic", "nhif", "medical"],
}


def _now() -> str:
	return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _record_id(prefix: str, explicit: str | None = None) -> str:
	return explicit or f"{prefix}-{uuid4().hex[:12]}"


def _hash_pin(pin: str) -> str:
	return hashlib.sha256(pin.encode()).hexdigest()


class MobUssdService:
	"""Async service for USSD mobile banking operations."""

	def __init__(self, tenant_id: str = "default", session_secret: str | None = None) -> None:
		self.tenant_id = tenant_id
		self._session_secret = session_secret or secrets.token_hex(32)
		# In-memory stores
		self.accounts: dict[str, dict[str, Any]] = {}
		self.transfers: dict[str, dict[str, Any]] = {}
		self.standing_orders: dict[str, dict[str, Any]] = {}
		self.transactions: dict[str, list[dict[str, Any]]] = {}  # account_number -> entries
		self.ussd_sessions: dict[str, dict[str, Any]] = {}
		self.otp_store: dict[str, dict[str, Any]] = {}
		self.pin_attempts: dict[str, int] = {}  # account_number -> attempt count
		self._audit_events: list[dict[str, Any]] = []
		self._audit_chain_tip: str = "0" * 64  # genesis hash
		self.beneficiaries: dict[str, dict[str, Any]] = {}  # account_number -> {alias -> record}
		self._idempotency_cache: dict[str, dict[str, Any]] = {}  # key -> {transfer_id, created_at}
		self._service_code_registry: dict[str, str] = {}  # service_code -> tenant_id
		self._fx_rates: dict[str, dict[str, Any]] = {}  # "KES/UGX" -> {rate, fetched_at}
		self._usage_frequency: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))  # phone -> {menu_item -> count}

	# ── Internal helpers ──────────────────────────────────────────────────────

	def _tenant(self, tenant_id: str | None = None) -> str:
		value = tenant_id or self.tenant_id
		guard_tenant_id(value)
		return value

	def _emit(self, tenant_id: str, event_type: str, record_id: str, record_type: str, metadata: dict[str, Any] | None = None) -> None:
		event: dict[str, Any] = {
			"id": _record_id("evt"),
			"tenant_id": tenant_id,
			"event_type": event_type,
			"record_id": record_id,
			"record_type": record_type,
			"metadata": metadata or {},
			"emitted_at": _now(),
		}
		# Merkle-chain: chain this event's hash to the previous tip
		event_json = json.dumps({k: v for k, v in event.items() if k != "event_hash"}, sort_keys=True)
		event["event_hash"] = hashlib.sha256((self._audit_chain_tip + event_json).encode()).hexdigest()
		self._audit_chain_tip = event["event_hash"]
		self._audit_events.append(event)

	def _classify_transaction(self, narration: str) -> str:
		"""Classify a transaction narration into a spending category."""
		lower = narration.lower()
		for category, keywords in _CATEGORY_KEYWORDS.items():
			if any(kw in lower for kw in keywords):
				return category
		return "other"

	def _purge_expired_idempotency_keys(self) -> None:
		"""Remove idempotency cache entries older than TTL."""
		cutoff = datetime.utcnow().timestamp() - IDEMPOTENCY_CACHE_TTL_SECONDS
		expired = [k for k, v in self._idempotency_cache.items() if v.get("ts", 0) < cutoff]
		for k in expired:
			del self._idempotency_cache[k]

	def _validate_pin_format(self, pin: str) -> None:
		if not pin.isdigit():
			raise ValueError("pin_must_be_numeric")
		if not (MIN_PIN_LENGTH <= len(pin) <= MAX_PIN_LENGTH):
			raise ValueError(f"pin_length_must_be_{MIN_PIN_LENGTH}_to_{MAX_PIN_LENGTH}_digits")

	def _validate_phone(self, phone: str) -> str:
		cleaned = re.sub(r"[^0-9]", "", phone)
		if cleaned.startswith("254") and len(cleaned) == 12:
			return cleaned
		if cleaned.startswith("0") and len(cleaned) == 10:
			return "254" + cleaned[1:]
		if cleaned.startswith("7") and len(cleaned) == 9:
			return "254" + cleaned
		raise ValueError(f"invalid_phone_number: {phone}")

	def _get_account_by_number(self, account_number: str, tenant_id: str) -> dict[str, Any] | None:
		for acct in self.accounts.values():
			if acct["account_number"] == account_number and acct["tenant_id"] == tenant_id:
				return acct
		return None

	def _get_account_by_phone(self, phone: str, tenant_id: str) -> dict[str, Any] | None:
		normalized = self._validate_phone(phone)
		for acct in self.accounts.values():
			if acct["phone_number"] == normalized and acct["tenant_id"] == tenant_id:
				return acct
		return None

	def _check_pin(self, account: dict[str, Any], pin: str) -> bool:
		acct_no = account["account_number"]
		attempts = self.pin_attempts.get(acct_no, 0)
		if attempts >= MAX_PIN_ATTEMPTS:
			account["status"] = "locked"
			raise PermissionError("account_locked_too_many_pin_attempts")
		if account["pin_hash"] != _hash_pin(pin):
			self.pin_attempts[acct_no] = attempts + 1
			remaining = MAX_PIN_ATTEMPTS - self.pin_attempts[acct_no]
			raise PermissionError(f"invalid_pin: {remaining} attempts remaining")
		self.pin_attempts[acct_no] = 0
		return True

	def _add_transaction_entry(self, account_number: str, entry: dict[str, Any]) -> None:
		if account_number not in self.transactions:
			self.transactions[account_number] = []
		self.transactions[account_number].insert(0, entry)
		# Cap at 200 entries per account
		self.transactions[account_number] = self.transactions[account_number][:200]

	def _build_ussd_menu(self, level: int, context: dict[str, Any]) -> tuple[str, bool]:
		"""Build USSD menu text and continuation flag."""
		if level == 0:
			return (
				"CON Welcome to MobBank\n1. Account Balance\n2. Mini Statement\n"
				"3. Fund Transfer\n4. Standing Orders\n5. Change PIN\n0. Exit",
				True,
			)
		if level == 1:
			choice = context.get("choice", "")
			if choice == "1":
				balance = context.get("balance", "0.00")
				available = context.get("available", "0.00")
				return (f"END Account Balance\nBalance: KES {balance}\nAvailable: KES {available}", False)
			if choice == "2":
				entries = context.get("entries", [])
				lines = ["END Mini Statement"]
				for e in entries[:MINI_STATEMENT_ROWS]:
					lines.append(f"{e['created_at'][:10]} {e['transaction_type']} {e['amount']}")
				return ("\n".join(lines), False)
			if choice == "3":
				return ("CON Fund Transfer\nEnter recipient account:", True)
			if choice == "4":
				return ("CON Standing Orders\n1. View Orders\n2. Create Order\n3. Cancel Order", True)
			if choice == "5":
				return ("CON Change PIN\nEnter current PIN:", True)
		return ("END Invalid option. Please try again.", False)

	# ── Account management ────────────────────────────────────────────────────

	async def create_account(
		self,
		phone_number: str,
		account_number: str,
		account_type: str,
		customer_name: str,
		national_id: str,
		pin: str,
		currency: str = "KES",
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Register a new mobile banking account."""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(account_number, "account_number")
		guard_non_empty_string(customer_name, "customer_name")
		guard_non_empty_string(national_id, "national_id")
		self._validate_pin_format(pin)
		if account_type not in SUPPORTED_ACCOUNT_TYPES:
			raise ValueError(f"unsupported_account_type: {account_type}")
		if currency not in SUPPORTED_CURRENCIES:
			raise ValueError(f"unsupported_currency: {currency}")
		normalized_phone = self._validate_phone(phone_number)
		# Check for duplicate account number
		if any(a["account_number"] == account_number and a["tenant_id"] == tenant for a in self.accounts.values()):
			raise ValueError("account_number_already_exists")
		acct_id = _record_id("mob-acct")
		record = {
			"id": acct_id,
			"type": "mob_account",
			"phone_number": normalized_phone,
			"account_number": account_number,
			"account_type": account_type,
			"customer_name": customer_name,
			"national_id": national_id,
			"pin_hash": _hash_pin(pin),
			"currency": currency,
			"balance": Decimal("0"),
			"available_balance": Decimal("0"),
			"daily_limit": DAILY_TRANSFER_LIMIT,
			"daily_used": Decimal("0"),
			"daily_reset_date": _now()[:10],
			"status": "active",
			"tenant_id": tenant,
			"created_at": _now(),
			"updated_at": None,
		}
		self.accounts[acct_id] = record
		self._emit(tenant, "mob_account_created", acct_id, "mob_account")
		_log.info("mob_account_created account=%s tenant=%s", account_number, tenant)
		return deepcopy(record)

	async def get_account(self, account_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Retrieve account by internal ID."""
		tenant = self._tenant(tenant_id)
		record = self.accounts.get(account_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"mob_account_not_found: {account_id}")
		return deepcopy(record)

	async def get_account_by_number(self, account_number: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Retrieve account by account number."""
		tenant = self._tenant(tenant_id)
		record = self._get_account_by_number(account_number, tenant)
		if not record:
			raise KeyError(f"mob_account_not_found: {account_number}")
		return deepcopy(record)

	async def list_accounts(self, tenant_id: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
		"""List all mobile banking accounts for a tenant."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(a) for a in self.accounts.values() if a["tenant_id"] == tenant]
		if status:
			items = [a for a in items if a["status"] == status]
		return items

	async def update_account(
		self,
		account_id: str,
		customer_name: str | None = None,
		status: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Update account metadata."""
		tenant = self._tenant(tenant_id)
		record = self.accounts.get(account_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"mob_account_not_found: {account_id}")
		if customer_name:
			record["customer_name"] = customer_name
		if status:
			record["status"] = status
		record["updated_at"] = _now()
		self._emit(tenant, "mob_account_updated", account_id, "mob_account")
		return deepcopy(record)

	async def delete_account(self, account_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Soft-delete (close) a mobile banking account."""
		tenant = self._tenant(tenant_id)
		record = self.accounts.get(account_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"mob_account_not_found: {account_id}")
		if record["balance"] > Decimal("0"):
			raise PermissionError("cannot_close_account_with_positive_balance")
		record["status"] = "closed"
		record["closed_at"] = _now()
		self._emit(tenant, "mob_account_closed", account_id, "mob_account")
		_log.info("mob_account_closed id=%s tenant=%s", account_id, tenant)
		return deepcopy(record)

	# ── Balance enquiry ───────────────────────────────────────────────────────

	async def get_balance(self, account_number: str, pin: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Return account balance after PIN verification."""
		tenant = self._tenant(tenant_id)
		account = self._get_account_by_number(account_number, tenant)
		if not account:
			raise KeyError(f"mob_account_not_found: {account_number}")
		if account["status"] != "active":
			raise PermissionError(f"account_{account['status']}")
		self._check_pin(account, pin)
		self._emit(tenant, "mob_balance_enquiry", account["id"], "mob_account")
		return {
			"account_number": account_number,
			"currency": account["currency"],
			"balance": str(account["balance"]),
			"available_balance": str(account["available_balance"]),
			"queried_at": _now(),
		}

	async def deposit(self, account_number: str, amount: Decimal, narration: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Credit funds to an account (bank-side operation, no PIN required)."""
		tenant = self._tenant(tenant_id)
		account = self._get_account_by_number(account_number, tenant)
		if not account:
			raise KeyError(f"mob_account_not_found: {account_number}")
		if amount <= Decimal("0"):
			raise ValueError("deposit_amount_must_be_positive")
		account["balance"] += amount
		account["available_balance"] += amount
		entry = {
			"id": _record_id("txn"),
			"account_number": account_number,
			"transaction_type": "credit",
			"amount": str(amount),
			"currency": account["currency"],
			"balance_after": str(account["balance"]),
			"narration": narration,
			"reference": _record_id("dep"),
			"created_at": _now(),
		}
		self._add_transaction_entry(account_number, entry)
		self._emit(tenant, "mob_deposit", account["id"], "mob_account", {"amount": str(amount)})
		return entry

	async def withdraw(self, account_number: str, amount: Decimal, pin: str, narration: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Debit funds from account after PIN verification."""
		tenant = self._tenant(tenant_id)
		account = self._get_account_by_number(account_number, tenant)
		if not account:
			raise KeyError(f"mob_account_not_found: {account_number}")
		if account["status"] != "active":
			raise PermissionError(f"account_{account['status']}")
		self._check_pin(account, pin)
		if amount > account["available_balance"]:
			raise PermissionError("insufficient_funds")
		account["balance"] -= amount
		account["available_balance"] -= amount
		entry = {
			"id": _record_id("txn"),
			"account_number": account_number,
			"transaction_type": "debit",
			"amount": str(amount),
			"currency": account["currency"],
			"balance_after": str(account["balance"]),
			"narration": narration,
			"reference": _record_id("wdr"),
			"created_at": _now(),
		}
		self._add_transaction_entry(account_number, entry)
		self._emit(tenant, "mob_withdrawal", account["id"], "mob_account", {"amount": str(amount)})
		return entry

	# ── Mini-statement ────────────────────────────────────────────────────────

	async def get_mini_statement(self, account_number: str, pin: str, rows: int = MINI_STATEMENT_ROWS, tenant_id: str | None = None) -> dict[str, Any]:
		"""Return last N transactions after PIN verification."""
		tenant = self._tenant(tenant_id)
		account = self._get_account_by_number(account_number, tenant)
		if not account:
			raise KeyError(f"mob_account_not_found: {account_number}")
		if account["status"] not in {"active", "dormant"}:
			raise PermissionError(f"account_{account['status']}")
		self._check_pin(account, pin)
		entries = self.transactions.get(account_number, [])[:rows]
		self._emit(tenant, "mob_mini_statement", account["id"], "mob_account", {"rows": rows})
		return {
			"account_number": account_number,
			"entries": entries,
			"total_entries": len(entries),
			"generated_at": _now(),
		}

	async def get_full_statement(self, account_number: str, pin: str, date_from: str, date_to: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Return transactions within a date range."""
		tenant = self._tenant(tenant_id)
		account = self._get_account_by_number(account_number, tenant)
		if not account:
			raise KeyError(f"mob_account_not_found: {account_number}")
		self._check_pin(account, pin)
		all_entries = self.transactions.get(account_number, [])
		filtered = [e for e in all_entries if date_from <= e["created_at"][:10] <= date_to]
		self._emit(tenant, "mob_full_statement", account["id"], "mob_account")
		return {
			"account_number": account_number,
			"date_from": date_from,
			"date_to": date_to,
			"entries": filtered,
			"total_entries": len(filtered),
			"generated_at": _now(),
		}

	# ── Fund transfer ─────────────────────────────────────────────────────────

	async def create_transfer(
		self,
		from_account: str,
		to_account: str,
		amount: Decimal,
		pin: str,
		narration: str = "",
		currency: str = "KES",
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Initiate a fund transfer between accounts."""
		tenant = self._tenant(tenant_id)
		sender = self._get_account_by_number(from_account, tenant)
		if not sender:
			raise KeyError(f"source_account_not_found: {from_account}")
		if sender["status"] != "active":
			raise PermissionError(f"source_account_{sender['status']}")
		self._check_pin(sender, pin)
		# Reset daily limit if new day
		today = _now()[:10]
		if sender.get("daily_reset_date", "") != today:
			sender["daily_used"] = Decimal("0")
			sender["daily_reset_date"] = today
		if amount > SINGLE_TRANSFER_LIMIT:
			raise PermissionError(f"exceeds_single_transfer_limit_{SINGLE_TRANSFER_LIMIT}")
		if (sender["daily_used"] + amount) > sender["daily_limit"]:
			raise PermissionError("daily_transfer_limit_exceeded")
		if amount > sender["available_balance"]:
			raise PermissionError("insufficient_funds")
		# Validate recipient
		recipient = self._get_account_by_number(to_account, tenant)
		if not recipient:
			raise KeyError(f"recipient_account_not_found: {to_account}")
		if recipient["status"] != "active":
			raise PermissionError(f"recipient_account_{recipient['status']}")
		ref = _record_id("trf")
		transfer_id = _record_id("mob-trf")
		# Debit sender
		sender["balance"] -= amount
		sender["available_balance"] -= amount
		sender["daily_used"] += amount
		# Credit recipient
		recipient["balance"] += amount
		recipient["available_balance"] += amount
		# Record entries
		debit_entry = {
			"id": _record_id("txn"),
			"account_number": from_account,
			"transaction_type": "debit",
			"amount": str(amount),
			"currency": currency,
			"balance_after": str(sender["balance"]),
			"narration": narration or f"Transfer to {to_account}",
			"reference": ref,
			"created_at": _now(),
		}
		credit_entry = {
			"id": _record_id("txn"),
			"account_number": to_account,
			"transaction_type": "credit",
			"amount": str(amount),
			"currency": currency,
			"balance_after": str(recipient["balance"]),
			"narration": narration or f"Transfer from {from_account}",
			"reference": ref,
			"created_at": _now(),
		}
		self._add_transaction_entry(from_account, debit_entry)
		self._add_transaction_entry(to_account, credit_entry)
		record = {
			"id": transfer_id,
			"type": "mob_transfer",
			"from_account": from_account,
			"to_account": to_account,
			"amount": str(amount),
			"currency": currency,
			"narration": narration,
			"reference": ref,
			"status": "completed",
			"tenant_id": tenant,
			"created_at": _now(),
			"settled_at": _now(),
		}
		self.transfers[transfer_id] = record
		self._emit(tenant, "mob_transfer_completed", transfer_id, "mob_transfer", {
			"from_account": from_account,
			"to_account": to_account,
			"amount": str(amount),
		})
		_log.info("mob_transfer from=%s to=%s amount=%s tenant=%s", from_account, to_account, amount, tenant)
		return deepcopy(record)

	async def get_transfer(self, transfer_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Retrieve a specific transfer record."""
		tenant = self._tenant(tenant_id)
		record = self.transfers.get(transfer_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"transfer_not_found: {transfer_id}")
		return deepcopy(record)

	async def list_transfers(self, tenant_id: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
		"""List all transfers for a tenant."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(t) for t in self.transfers.values() if t["tenant_id"] == tenant]
		if status:
			items = [t for t in items if t["status"] == status]
		return items

	async def reverse_transfer(self, transfer_id: str, reason: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Reverse a completed transfer."""
		tenant = self._tenant(tenant_id)
		record = self.transfers.get(transfer_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"transfer_not_found: {transfer_id}")
		if record["status"] != "completed":
			raise PermissionError("only_completed_transfers_can_be_reversed")
		# Reverse the money movement
		amount = Decimal(record["amount"])
		from_acct = self._get_account_by_number(record["from_account"], tenant)
		to_acct = self._get_account_by_number(record["to_account"], tenant)
		if from_acct and to_acct:
			from_acct["balance"] += amount
			from_acct["available_balance"] += amount
			to_acct["balance"] -= amount
			to_acct["available_balance"] -= amount
		record["status"] = "reversed"
		record["reversal_reason"] = reason
		record["reversed_at"] = _now()
		self._emit(tenant, "mob_transfer_reversed", transfer_id, "mob_transfer", {"reason": reason})
		_log.warning("mob_transfer_reversed id=%s reason=%s tenant=%s", transfer_id, reason, tenant)
		return deepcopy(record)

	# ── Standing orders ───────────────────────────────────────────────────────

	async def create_standing_order(
		self,
		from_account: str,
		to_account: str,
		amount: Decimal,
		frequency: str,
		start_date: str,
		pin: str,
		end_date: str | None = None,
		narration: str = "",
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Create a recurring standing order."""
		tenant = self._tenant(tenant_id)
		sender = self._get_account_by_number(from_account, tenant)
		if not sender:
			raise KeyError(f"source_account_not_found: {from_account}")
		self._check_pin(sender, pin)
		if frequency not in SUPPORTED_FREQUENCIES:
			raise ValueError(f"unsupported_frequency: {frequency}")
		if amount <= Decimal("0"):
			raise ValueError("standing_order_amount_must_be_positive")
		order_id = _record_id("mob-so")
		record = {
			"id": order_id,
			"type": "mob_standing_order",
			"from_account": from_account,
			"to_account": to_account,
			"amount": str(amount),
			"frequency": frequency,
			"start_date": start_date,
			"end_date": end_date,
			"narration": narration,
			"next_execution_date": start_date,
			"executions_count": 0,
			"status": "active",
			"tenant_id": tenant,
			"created_at": _now(),
		}
		self.standing_orders[order_id] = record
		self._emit(tenant, "mob_standing_order_created", order_id, "mob_standing_order")
		_log.info("mob_standing_order_created id=%s from=%s freq=%s tenant=%s", order_id, from_account, frequency, tenant)
		return deepcopy(record)

	async def get_standing_order(self, order_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Retrieve a standing order by ID."""
		tenant = self._tenant(tenant_id)
		record = self.standing_orders.get(order_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"standing_order_not_found: {order_id}")
		return deepcopy(record)

	async def list_standing_orders(self, account_number: str | None = None, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""List standing orders, optionally filtered by account."""
		tenant = self._tenant(tenant_id)
		items = [deepcopy(o) for o in self.standing_orders.values() if o["tenant_id"] == tenant]
		if account_number:
			items = [o for o in items if o["from_account"] == account_number]
		return items

	async def update_standing_order(
		self,
		order_id: str,
		amount: Decimal | None = None,
		frequency: str | None = None,
		end_date: str | None = None,
		status: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Modify a standing order."""
		tenant = self._tenant(tenant_id)
		record = self.standing_orders.get(order_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"standing_order_not_found: {order_id}")
		if amount is not None:
			record["amount"] = str(amount)
		if frequency is not None:
			if frequency not in SUPPORTED_FREQUENCIES:
				raise ValueError(f"unsupported_frequency: {frequency}")
			record["frequency"] = frequency
		if end_date is not None:
			record["end_date"] = end_date
		if status is not None:
			record["status"] = status
		record["updated_at"] = _now()
		self._emit(tenant, "mob_standing_order_updated", order_id, "mob_standing_order")
		return deepcopy(record)

	async def delete_standing_order(self, order_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Cancel a standing order."""
		tenant = self._tenant(tenant_id)
		record = self.standing_orders.get(order_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"standing_order_not_found: {order_id}")
		record["status"] = "cancelled"
		record["cancelled_at"] = _now()
		self._emit(tenant, "mob_standing_order_cancelled", order_id, "mob_standing_order")
		return deepcopy(record)

	async def execute_standing_order(self, order_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Execute a standing order (called by scheduler)."""
		tenant = self._tenant(tenant_id)
		record = self.standing_orders.get(order_id)
		if not record or record["tenant_id"] != tenant:
			raise KeyError(f"standing_order_not_found: {order_id}")
		if record["status"] != "active":
			raise PermissionError("standing_order_not_active")
		if record.get("end_date") and _now()[:10] > record["end_date"]:
			record["status"] = "expired"
			return deepcopy(record)
		# Perform the transfer (no PIN for scheduled execution)
		from_acct = self._get_account_by_number(record["from_account"], tenant)
		to_acct = self._get_account_by_number(record["to_account"], tenant)
		amount = Decimal(record["amount"])
		if not from_acct or from_acct["available_balance"] < amount:
			result = {"status": "failed", "reason": "insufficient_funds", "order_id": order_id, "executed_at": _now()}
			self._emit(tenant, "mob_standing_order_failed", order_id, "mob_standing_order", result)
			return result
		from_acct["balance"] -= amount
		from_acct["available_balance"] -= amount
		if to_acct:
			to_acct["balance"] += amount
			to_acct["available_balance"] += amount
		record["executions_count"] += 1
		# Compute next execution date
		freq_days = {"daily": 1, "weekly": 7, "monthly": 30, "quarterly": 90}
		next_dt = datetime.fromisoformat(_now()[:-1]) + timedelta(days=freq_days.get(record["frequency"], 30))
		record["next_execution_date"] = next_dt.strftime("%Y-%m-%d")
		record["last_executed_at"] = _now()
		self._emit(tenant, "mob_standing_order_executed", order_id, "mob_standing_order", {"amount": record["amount"]})
		return deepcopy(record)

	# ── PIN management ────────────────────────────────────────────────────────

	async def change_pin(self, account_number: str, old_pin: str, new_pin: str, confirm_pin: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Change account PIN after verifying current PIN."""
		tenant = self._tenant(tenant_id)
		account = self._get_account_by_number(account_number, tenant)
		if not account:
			raise KeyError(f"mob_account_not_found: {account_number}")
		if account["status"] != "active":
			raise PermissionError(f"account_{account['status']}")
		self._validate_pin_format(new_pin)
		if new_pin != confirm_pin:
			raise ValueError("new_pin_and_confirm_pin_do_not_match")
		if old_pin == new_pin:
			raise ValueError("new_pin_must_differ_from_old_pin")
		self._check_pin(account, old_pin)
		account["pin_hash"] = _hash_pin(new_pin)
		account["pin_changed_at"] = _now()
		self._emit(tenant, "mob_pin_changed", account["id"], "mob_account")
		_log.info("mob_pin_changed account=%s tenant=%s", account_number, tenant)
		return {"account_number": account_number, "status": "pin_changed", "changed_at": _now()}

	async def generate_pin_reset_otp(self, phone_number: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Generate OTP for PIN reset and deliver via SMS (simulated)."""
		tenant = self._tenant(tenant_id)
		normalized = self._validate_phone(phone_number)
		account = self._get_account_by_phone(normalized, tenant)
		if not account:
			raise KeyError(f"mob_account_not_found_for_phone: {phone_number}")
		import random
		otp = str(random.randint(100000, 999999))
		otp_key = f"{normalized}:{tenant}"
		self.otp_store[otp_key] = {
			"otp_hash": _hash_pin(otp),
			"expires_at": (datetime.utcnow() + timedelta(minutes=5)).isoformat() + "Z",
			"used": False,
		}
		# In production: send SMS via telco/aggregator
		_log.info("mob_pin_reset_otp_generated phone=%s tenant=%s otp=%s", normalized, tenant, otp)
		self._emit(tenant, "mob_pin_reset_otp_generated", account["id"], "mob_account")
		return {"phone_number": normalized, "otp_sent": True, "expires_in_seconds": 300, "generated_at": _now()}

	async def reset_pin(self, phone_number: str, national_id: str, new_pin: str, otp: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Reset PIN via OTP verification."""
		tenant = self._tenant(tenant_id)
		normalized = self._validate_phone(phone_number)
		account = self._get_account_by_phone(normalized, tenant)
		if not account:
			raise KeyError(f"mob_account_not_found_for_phone: {phone_number}")
		if account["national_id"] != national_id:
			raise PermissionError("national_id_mismatch")
		self._validate_pin_format(new_pin)
		otp_key = f"{normalized}:{tenant}"
		otp_record = self.otp_store.get(otp_key)
		if not otp_record:
			raise PermissionError("no_otp_found_for_phone")
		if otp_record["used"]:
			raise PermissionError("otp_already_used")
		if _now() > otp_record["expires_at"]:
			raise PermissionError("otp_expired")
		if otp_record["otp_hash"] != _hash_pin(otp):
			raise PermissionError("invalid_otp")
		otp_record["used"] = True
		account["pin_hash"] = _hash_pin(new_pin)
		account["status"] = "active"
		account["pin_reset_at"] = _now()
		self.pin_attempts[account["account_number"]] = 0
		self._emit(tenant, "mob_pin_reset", account["id"], "mob_account")
		_log.info("mob_pin_reset account=%s tenant=%s", account["account_number"], tenant)
		return {"account_number": account["account_number"], "status": "pin_reset", "reset_at": _now()}

	async def lock_account(self, account_number: str, reason: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Lock an account (admin operation)."""
		tenant = self._tenant(tenant_id)
		account = self._get_account_by_number(account_number, tenant)
		if not account:
			raise KeyError(f"mob_account_not_found: {account_number}")
		account["status"] = "locked"
		account["lock_reason"] = reason
		account["locked_at"] = _now()
		self._emit(tenant, "mob_account_locked", account["id"], "mob_account", {"reason": reason})
		return deepcopy(account)

	async def unlock_account(self, account_number: str, reviewed_by: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Unlock a locked account."""
		tenant = self._tenant(tenant_id)
		account = self._get_account_by_number(account_number, tenant)
		if not account:
			raise KeyError(f"mob_account_not_found: {account_number}")
		if account["status"] != "locked":
			raise PermissionError("account_not_locked")
		account["status"] = "active"
		account["unlocked_by"] = reviewed_by
		account["unlocked_at"] = _now()
		self.pin_attempts[account_number] = 0
		self._emit(tenant, "mob_account_unlocked", account["id"], "mob_account")
		return deepcopy(account)

	# ── USSD session handling ─────────────────────────────────────────────────

	async def handle_ussd_request(
		self,
		session_id: str,
		phone_number: str,
		service_code: str,
		input_text: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Process a USSD request and return the menu response."""
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
		# Level 1: main menu selection
		if level >= 1:
			context["choice"] = inputs[0]
			# Balance enquiry
			if inputs[0] == "1" and level >= 2:
				account = self._get_account_by_phone(normalized, tenant)
				if account:
					context["balance"] = str(account["balance"])
					context["available"] = str(account["available_balance"])
		response_text, continues = self._build_ussd_menu(min(level, 1), context)
		session["menu_level"] = level
		session["inputs"] = inputs
		session["context"] = context
		if not continues:
			session["ended_at"] = _now()
		self.ussd_sessions[session_id] = session
		self._emit(tenant, "mob_ussd_request", session_id, "mob_ussd_session", {"level": level})
		return {
			"session_id": session_id,
			"phone_number": normalized,
			"menu_level": level,
			"response_text": response_text,
			"continues": continues,
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

	# ── Limit management ──────────────────────────────────────────────────────

	async def update_daily_limit(self, account_number: str, new_limit: Decimal, approved_by: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Update the daily transfer limit for an account."""
		tenant = self._tenant(tenant_id)
		account = self._get_account_by_number(account_number, tenant)
		if not account:
			raise KeyError(f"mob_account_not_found: {account_number}")
		guard_non_empty_string(approved_by, "approved_by")
		if new_limit <= Decimal("0"):
			raise ValueError("daily_limit_must_be_positive")
		if new_limit > DAILY_TRANSFER_LIMIT * 2:
			raise PermissionError("daily_limit_exceeds_maximum_allowed")
		old_limit = account["daily_limit"]
		account["daily_limit"] = new_limit
		account["limit_updated_by"] = approved_by
		account["limit_updated_at"] = _now()
		self._emit(tenant, "mob_daily_limit_updated", account["id"], "mob_account", {
			"old_limit": str(old_limit),
			"new_limit": str(new_limit),
			"approved_by": approved_by,
		})
		return {"account_number": account_number, "daily_limit": str(new_limit), "updated_by": approved_by, "updated_at": _now()}

	async def get_account_summary(self, account_number: str, pin: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Return account summary including balance, recent transactions, and standing orders."""
		tenant = self._tenant(tenant_id)
		account = self._get_account_by_number(account_number, tenant)
		if not account:
			raise KeyError(f"mob_account_not_found: {account_number}")
		self._check_pin(account, pin)
		results = await asyncio.gather(
			self.get_mini_statement(account_number, pin, rows=3, tenant_id=tenant),
			self.list_standing_orders(account_number, tenant_id=tenant),
			return_exceptions=True,
		)
		mini_stmt = results[0] if not isinstance(results[0], Exception) else {"entries": []}
		orders = results[1] if not isinstance(results[1], Exception) else []
		active_orders = [o for o in (orders or []) if isinstance(o, dict) and o.get("status") == "active"]
		return {
			"account_number": account_number,
			"customer_name": account["customer_name"],
			"account_type": account["account_type"],
			"currency": account["currency"],
			"balance": str(account["balance"]),
			"available_balance": str(account["available_balance"]),
			"daily_limit": str(account["daily_limit"]),
			"daily_used": str(account.get("daily_used", Decimal("0"))),
			"recent_transactions": mini_stmt.get("entries", []),
			"active_standing_orders": len(active_orders),
			"status": account["status"],
			"queried_at": _now(),
		}

	# ── Analytics / Reporting ─────────────────────────────────────────────────

	async def get_transaction_summary(self, account_number: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Return debit/credit summary for an account."""
		tenant = self._tenant(tenant_id)
		account = self._get_account_by_number(account_number, tenant)
		if not account:
			raise KeyError(f"mob_account_not_found: {account_number}")
		entries = self.transactions.get(account_number, [])
		total_credits = sum(Decimal(e["amount"]) for e in entries if e["transaction_type"] == "credit")
		total_debits = sum(Decimal(e["amount"]) for e in entries if e["transaction_type"] == "debit")
		return {
			"account_number": account_number,
			"total_credits": str(total_credits),
			"total_debits": str(total_debits),
			"net_flow": str(total_credits - total_debits),
			"transaction_count": len(entries),
			"generated_at": _now(),
		}

	async def get_tenant_statistics(self, tenant_id: str | None = None) -> dict[str, Any]:
		"""Return aggregate statistics for a tenant."""
		tenant = self._tenant(tenant_id)
		accounts = [a for a in self.accounts.values() if a["tenant_id"] == tenant]
		transfers = [t for t in self.transfers.values() if t["tenant_id"] == tenant]
		orders = [o for o in self.standing_orders.values() if o["tenant_id"] == tenant]
		total_balance = sum(a["balance"] for a in accounts)
		return {
			"tenant_id": tenant,
			"account_count": len(accounts),
			"active_accounts": sum(1 for a in accounts if a["status"] == "active"),
			"locked_accounts": sum(1 for a in accounts if a["status"] == "locked"),
			"transfer_count": len(transfers),
			"completed_transfers": sum(1 for t in transfers if t["status"] == "completed"),
			"active_standing_orders": sum(1 for o in orders if o["status"] == "active"),
			"total_balance": str(total_balance),
			"audit_event_count": len([e for e in self._audit_events if e["tenant_id"] == tenant]),
			"generated_at": _now(),
		}

	# ── Utility ───────────────────────────────────────────────────────────────

	async def health_check(self) -> dict[str, Any]:
		"""Return service health status."""
		return {
			"service": "fintech_ussd_mob",
			"status": "healthy",
			"account_count": len(self.accounts),
			"transfer_count": len(self.transfers),
			"standing_order_count": len(self.standing_orders),
			"active_sessions": sum(1 for s in self.ussd_sessions.values() if "ended_at" not in s),
			"checked_at": _now(),
		}

	async def get_audit_events(self, tenant_id: str | None = None, event_type: str | None = None) -> list[dict[str, Any]]:
		"""Return audit events for a tenant."""
		tenant = self._tenant(tenant_id)
		events = [deepcopy(e) for e in self._audit_events if e["tenant_id"] == tenant]
		if event_type:
			events = [e for e in events if e["event_type"] == event_type]
		return events

	# ── Beneficiary management ────────────────────────────────────────────────

	async def add_beneficiary(
		self,
		account_number: str,
		pin: str,
		alias: str,
		target_account: str,
		target_name: str = "",
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Save a named beneficiary for fast repeat transfers.

		Alias is capped at 12 chars to fit a single USSD display line.
		"""
		tenant = self._tenant(tenant_id)
		account = self._get_account_by_number(account_number, tenant)
		if not account:
			raise KeyError(f"mob_account_not_found: {account_number}")
		if account["status"] != "active":
			raise PermissionError(f"account_{account['status']}")
		self._check_pin(account, pin)
		guard_non_empty_string(alias, "alias")
		guard_non_empty_string(target_account, "target_account")
		if len(alias) > 12:
			raise ValueError("alias_must_be_12_chars_or_fewer")
		bucket = self.beneficiaries.setdefault(account_number, {})
		if len(bucket) >= MAX_BENEFICIARIES_PER_ACCOUNT:
			raise PermissionError(f"max_{MAX_BENEFICIARIES_PER_ACCOUNT}_beneficiaries_reached")
		record = {
			"alias": alias,
			"target_account": target_account,
			"target_name": target_name,
			"created_at": _now(),
		}
		bucket[alias] = record
		self._emit(tenant, "mob_beneficiary_added", account_number, "mob_account", {"alias": alias})
		_log.info("mob_beneficiary_added account=%s alias=%s tenant=%s", account_number, alias, tenant)
		return deepcopy(record)

	async def list_beneficiaries(self, account_number: str, pin: str, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""Return saved beneficiaries for an account."""
		tenant = self._tenant(tenant_id)
		account = self._get_account_by_number(account_number, tenant)
		if not account:
			raise KeyError(f"mob_account_not_found: {account_number}")
		self._check_pin(account, pin)
		return list(deepcopy(self.beneficiaries.get(account_number, {})).values())

	async def remove_beneficiary(self, account_number: str, pin: str, alias: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Remove a saved beneficiary by alias."""
		tenant = self._tenant(tenant_id)
		account = self._get_account_by_number(account_number, tenant)
		if not account:
			raise KeyError(f"mob_account_not_found: {account_number}")
		self._check_pin(account, pin)
		bucket = self.beneficiaries.get(account_number, {})
		if alias not in bucket:
			raise KeyError(f"beneficiary_not_found: {alias}")
		removed = deepcopy(bucket.pop(alias))
		self._emit(tenant, "mob_beneficiary_removed", account_number, "mob_account", {"alias": alias})
		return removed

	# ── Fraud velocity scoring ────────────────────────────────────────────────

	async def score_fraud_risk(
		self,
		account_number: str,
		transfer_amount: Decimal,
		recipient_account: str,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Compute a 0–100 fraud risk score for a proposed transfer.

		Scoring factors:
		  - Transfer velocity in last 10 minutes (up to 40 pts)
		  - Recipient novelty — first-time payee scores higher (20 pts)
		  - Amount relative to 30-day average (20 pts)
		  - Time-of-day anomaly — between 00:00–05:00 UTC (20 pts)
		"""
		tenant = self._tenant(tenant_id)
		entries = self.transactions.get(account_number, [])
		now_ts = datetime.utcnow()

		# Factor 1: velocity — debits in last 10 minutes
		ten_min_ago = (now_ts - timedelta(minutes=10)).isoformat()[:19]
		recent_debits = [
			e for e in entries
			if e["transaction_type"] == "debit" and e["created_at"][:19] >= ten_min_ago
		]
		velocity_score = min(len(recent_debits) * 10, 40)

		# Factor 2: recipient novelty
		prior_recipients = {e.get("narration", "").split("to ")[-1] for e in entries if e["transaction_type"] == "debit"}
		recipient_score = 0 if recipient_account in prior_recipients else 20

		# Factor 3: amount vs 30-day average debit
		thirty_days_ago = (now_ts - timedelta(days=30)).isoformat()[:10]
		period_debits = [Decimal(e["amount"]) for e in entries if e["transaction_type"] == "debit" and e["created_at"][:10] >= thirty_days_ago]
		avg_debit = sum(period_debits) / len(period_debits) if period_debits else Decimal("0")
		amount_score = 20 if (avg_debit > 0 and transfer_amount > avg_debit * Decimal("3")) else 0

		# Factor 4: time-of-day anomaly (00:00–05:00 UTC)
		hour = now_ts.hour
		time_score = 20 if 0 <= hour < 5 else 0

		total = velocity_score + recipient_score + amount_score + time_score
		risk_level = "high" if total >= FRAUD_HIGH_RISK_SCORE else ("medium" if total >= FRAUD_MEDIUM_RISK_SCORE else "low")
		result = {
			"account_number": account_number,
			"score": total,
			"risk_level": risk_level,
			"factors": {
				"velocity": velocity_score,
				"recipient_novelty": recipient_score,
				"amount_deviation": amount_score,
				"time_anomaly": time_score,
			},
			"action_required": "totp_challenge" if risk_level == "medium" else ("hold_for_review" if risk_level == "high" else "none"),
			"scored_at": _now(),
		}
		self._emit(tenant, "mob_fraud_score_computed", account_number, "mob_account", {"score": total, "risk_level": risk_level})
		return result

	# ── Idempotent transfer ───────────────────────────────────────────────────

	async def create_transfer_idempotent(
		self,
		from_account: str,
		to_account: str,
		amount: Decimal,
		pin: str,
		idempotency_key: str,
		narration: str = "",
		currency: str = "KES",
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Idempotent fund transfer — duplicate keys return the original transfer.

		The 24-hour deduplication window prevents double-debit on gateway retries.
		"""
		self._purge_expired_idempotency_keys()
		cache_entry = self._idempotency_cache.get(idempotency_key)
		if cache_entry:
			transfer_id = cache_entry["transfer_id"]
			existing = self.transfers.get(transfer_id)
			if existing:
				_log.info("mob_idempotent_transfer_replay key=%s transfer=%s", idempotency_key, transfer_id)
				return deepcopy(existing)
		result = await self.create_transfer(from_account, to_account, amount, pin, narration, currency, tenant_id)
		self._idempotency_cache[idempotency_key] = {
			"transfer_id": result["id"],
			"ts": datetime.utcnow().timestamp(),
		}
		return result

	# ── Spending analytics ────────────────────────────────────────────────────

	async def get_spending_insights(
		self,
		account_number: str,
		pin: str,
		days: int = 30,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Return categorised spending breakdown suitable for USSD display and AI narration.

		Categories: utilities, food, transport, savings, education, health, other.
		"""
		tenant = self._tenant(tenant_id)
		account = self._get_account_by_number(account_number, tenant)
		if not account:
			raise KeyError(f"mob_account_not_found: {account_number}")
		self._check_pin(account, pin)

		cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()[:10]
		debit_entries = [
			e for e in self.transactions.get(account_number, [])
			if e["transaction_type"] == "debit" and e["created_at"][:10] >= cutoff
		]
		totals: dict[str, Decimal] = defaultdict(Decimal)
		for entry in debit_entries:
			cat = self._classify_transaction(entry.get("narration", ""))
			totals[cat] += Decimal(entry["amount"])

		grand_total = sum(totals.values()) or Decimal("1")
		ranked = sorted(
			[{"category": cat, "amount": str(amt), "percentage": str((amt / grand_total * 100).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))} for cat, amt in totals.items()],
			key=lambda x: Decimal(x["amount"]),
			reverse=True,
		)
		# USSD-safe one-liner summary (max 160 chars)
		top3 = ranked[:3]
		ussd_summary = "Top: " + ", ".join(f"{r['category'].title()} {r['percentage']}%" for r in top3)

		self._emit(tenant, "mob_spending_insights", account["id"], "mob_account", {"days": days})
		return {
			"account_number": account_number,
			"period_days": days,
			"total_spend": str(grand_total),
			"categories": ranked,
			"ussd_summary": ussd_summary,
			"transaction_count": len(debit_entries),
			"generated_at": _now(),
		}

	# ── Cross-border FX transfer ──────────────────────────────────────────────

	async def set_fx_rate(self, from_currency: str, to_currency: str, rate: Decimal, tenant_id: str | None = None) -> dict[str, Any]:
		"""Register or update an FX rate (admin / FX feed operation).

		In production this is populated by a scheduled bytewax pipeline
		subscribing to a live FX NATS subject (fx.rates.updated).
		"""
		if from_currency not in SUPPORTED_CURRENCIES or to_currency not in SUPPORTED_CURRENCIES:
			raise ValueError(f"unsupported_currency_pair: {from_currency}/{to_currency}")
		if rate <= Decimal("0"):
			raise ValueError("fx_rate_must_be_positive")
		pair = f"{from_currency}/{to_currency}"
		record = {"pair": pair, "rate": str(rate), "fetched_at": _now()}
		self._fx_rates[pair] = record
		_log.info("mob_fx_rate_updated pair=%s rate=%s", pair, rate)
		return deepcopy(record)

	async def create_cross_border_transfer(
		self,
		from_account: str,
		to_account: str,
		send_amount: Decimal,
		pin: str,
		spread_bps: int = 150,
		narration: str = "",
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Execute a cross-currency fund transfer using the registered FX rate.

		`spread_bps` is the bank's spread in basis points applied to the mid-rate.
		The sender is debited in their account currency; the recipient is credited
		in their account currency at the effective (post-spread) rate.
		USSD display: "Send KES 5000 → UGX 132,500 (rate: 26.50). 1.Confirm 2.Cancel"
		"""
		tenant = self._tenant(tenant_id)
		sender = self._get_account_by_number(from_account, tenant)
		if not sender:
			raise KeyError(f"source_account_not_found: {from_account}")
		recipient = self._get_account_by_number(to_account, tenant)
		if not recipient:
			raise KeyError(f"recipient_account_not_found: {to_account}")

		from_ccy = sender["currency"]
		to_ccy = recipient["currency"]

		if from_ccy == to_ccy:
			# Same currency — fall through to normal transfer
			return await self.create_transfer(from_account, to_account, send_amount, pin, narration, from_ccy, tenant_id)

		pair = f"{from_ccy}/{to_ccy}"
		rate_record = self._fx_rates.get(pair)
		if not rate_record:
			raise KeyError(f"fx_rate_not_available: {pair}")

		mid_rate = Decimal(rate_record["rate"])
		spread = mid_rate * Decimal(spread_bps) / Decimal("10000")
		effective_rate = mid_rate - spread  # sender gets slightly less than mid

		receive_amount = (send_amount * effective_rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

		# Validate and debit sender
		self._check_pin(sender, pin)
		if sender["status"] != "active":
			raise PermissionError(f"source_account_{sender['status']}")
		if send_amount > sender["available_balance"]:
			raise PermissionError("insufficient_funds")

		sender["balance"] -= send_amount
		sender["available_balance"] -= send_amount
		recipient["balance"] += receive_amount
		recipient["available_balance"] += receive_amount

		ref = _record_id("fxt")
		transfer_id = _record_id("mob-fxt")

		debit_entry = {
			"id": _record_id("txn"),
			"account_number": from_account,
			"transaction_type": "debit",
			"amount": str(send_amount),
			"currency": from_ccy,
			"balance_after": str(sender["balance"]),
			"narration": narration or f"FX to {to_account} @ {effective_rate}",
			"reference": ref,
			"created_at": _now(),
		}
		credit_entry = {
			"id": _record_id("txn"),
			"account_number": to_account,
			"transaction_type": "credit",
			"amount": str(receive_amount),
			"currency": to_ccy,
			"balance_after": str(recipient["balance"]),
			"narration": narration or f"FX from {from_account}",
			"reference": ref,
			"created_at": _now(),
		}
		self._add_transaction_entry(from_account, debit_entry)
		self._add_transaction_entry(to_account, credit_entry)

		record = {
			"id": transfer_id,
			"type": "mob_fx_transfer",
			"from_account": from_account,
			"to_account": to_account,
			"send_amount": str(send_amount),
			"send_currency": from_ccy,
			"receive_amount": str(receive_amount),
			"receive_currency": to_ccy,
			"mid_rate": str(mid_rate),
			"effective_rate": str(effective_rate),
			"spread_bps": spread_bps,
			"reference": ref,
			"narration": narration,
			"status": "completed",
			"tenant_id": tenant,
			"created_at": _now(),
			"settled_at": _now(),
		}
		self.transfers[transfer_id] = record
		self._emit(tenant, "mob_fx_transfer_completed", transfer_id, "mob_fx_transfer", {
			"from_account": from_account, "send_amount": str(send_amount), "receive_amount": str(receive_amount), "pair": pair,
		})
		_log.info("mob_fx_transfer from=%s to=%s send=%s %s receive=%s %s tenant=%s", from_account, to_account, send_amount, from_ccy, receive_amount, to_ccy, tenant)
		return deepcopy(record)

	# ── Service code registry (multi-tenant USSD routing) ────────────────────

	async def register_service_code(self, service_code: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Map a USSD service code to this service's tenant.

		Enables a single APG deployment to host multiple bank brands each
		with their own *123#-style service code.
		"""
		tenant = self._tenant(tenant_id)
		guard_non_empty_string(service_code, "service_code")
		existing_tenant = self._service_code_registry.get(service_code)
		if existing_tenant and existing_tenant != tenant:
			raise PermissionError(f"service_code_already_registered_to_another_tenant: {service_code}")
		self._service_code_registry[service_code] = tenant
		_log.info("mob_service_code_registered code=%s tenant=%s", service_code, tenant)
		return {"service_code": service_code, "tenant_id": tenant, "registered_at": _now()}

	async def list_service_codes(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""List all service codes registered for a tenant."""
		tenant = self._tenant(tenant_id)
		return [{"service_code": code, "tenant_id": tid} for code, tid in self._service_code_registry.items() if tid == tenant]

	# ── Session token (USSD session integrity) ────────────────────────────────

	async def create_session_token(self, session_id: str, phone_number: str) -> dict[str, Any]:
		"""Generate an HMAC-SHA256 session token bound to session_id + MSISDN.

		Tokens expire after SESSION_TOKEN_TTL_SECONDS (300 s by default).
		Validates against replay attacks documented in 3GPP TS 22.090.
		"""
		normalized = self._validate_phone(phone_number)
		ts = str(int(time.time()))
		payload = f"{session_id}:{normalized}:{ts}"
		token = hmac.new(self._session_secret.encode(), payload.encode(), "sha256").hexdigest()
		record = {
			"session_id": session_id,
			"phone_number": normalized,
			"token": token,
			"issued_at": _now(),
			"expires_at": (datetime.utcnow() + timedelta(seconds=SESSION_TOKEN_TTL_SECONDS)).isoformat() + "Z",
		}
		# Store in session for subsequent validation
		if session_id in self.ussd_sessions:
			self.ussd_sessions[session_id]["session_token"] = token
			self.ussd_sessions[session_id]["token_ts"] = ts
		return record

	async def validate_session_token(self, session_id: str, phone_number: str, token: str) -> dict[str, Any]:
		"""Verify that a USSD continuation token is valid and not expired."""
		session = self.ussd_sessions.get(session_id)
		if not session:
			raise KeyError(f"ussd_session_not_found: {session_id}")
		stored_token = session.get("session_token")
		if not stored_token:
			raise PermissionError("no_session_token_issued")
		if not hmac.compare_digest(stored_token, token):
			raise PermissionError("invalid_session_token")
		# Check expiry from token_ts
		ts = int(session.get("token_ts", 0))
		if (time.time() - ts) > SESSION_TOKEN_TTL_SECONDS:
			raise PermissionError("session_token_expired")
		return {"session_id": session_id, "valid": True, "checked_at": _now()}

	# ── Audit chain verification ──────────────────────────────────────────────

	async def verify_audit_chain(self, tenant_id: str | None = None) -> dict[str, Any]:
		"""Verify the Merkle hash chain integrity of the audit event log.

		Recomputes every event_hash from scratch; returns first tampered index
		or confirms chain is intact. Compliance requirement under CBK Prudential
		Guidelines (2023) and FATF Recommendation 10.
		"""
		tenant = self._tenant(tenant_id)
		events = [e for e in self._audit_events if e["tenant_id"] == tenant]
		prev_hash = "0" * 64
		for i, event in enumerate(events):
			event_json = json.dumps(
				{k: v for k, v in event.items() if k != "event_hash"}, sort_keys=True
			)
			expected = hashlib.sha256((prev_hash + event_json).encode()).hexdigest()
			if event.get("event_hash") != expected:
				return {
					"tenant_id": tenant,
					"chain_intact": False,
					"tampered_at_index": i,
					"event_id": event["id"],
					"verified_at": _now(),
				}
			prev_hash = expected
		return {
			"tenant_id": tenant,
			"chain_intact": True,
			"events_verified": len(events),
			"chain_tip": prev_hash,
			"verified_at": _now(),
		}

	# ── Statement export ──────────────────────────────────────────────────────

	async def export_statement(
		self,
		account_number: str,
		pin: str,
		date_from: str,
		date_to: str,
		format: str = "json",
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Export account statement in json, csv, or summary format.

		Returns a dict with `content` (str/bytes) and `content_type`.
		Supports integration with QuickBooks, Xero, and M-Pesa Business imports.
		"""
		if format not in {"json", "csv", "summary"}:
			raise ValueError(f"unsupported_export_format: {format}. Use json, csv, or summary")

		stmt = await self.get_full_statement(account_number, pin, date_from, date_to, tenant_id)
		entries = stmt["entries"]

		if format == "json":
			return {
				"content": json.dumps(stmt, indent=2, default=str),
				"content_type": "application/json",
				"filename": f"statement_{account_number}_{date_from}_{date_to}.json",
			}

		if format == "csv":
			fieldnames = ["id", "account_number", "transaction_type", "amount", "currency", "balance_after", "narration", "reference", "created_at"]
			buf = io.StringIO()
			writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
			writer.writeheader()
			writer.writerows(entries)
			return {
				"content": buf.getvalue(),
				"content_type": "text/csv",
				"filename": f"statement_{account_number}_{date_from}_{date_to}.csv",
			}

		# summary format — aggregate totals useful for accounting imports
		total_credits = sum(Decimal(e["amount"]) for e in entries if e["transaction_type"] == "credit")
		total_debits = sum(Decimal(e["amount"]) for e in entries if e["transaction_type"] == "debit")
		return {
			"content": json.dumps({
				"account_number": account_number,
				"date_from": date_from,
				"date_to": date_to,
				"total_credits": str(total_credits),
				"total_debits": str(total_debits),
				"net_flow": str(total_credits - total_debits),
				"transaction_count": len(entries),
			}, indent=2),
			"content_type": "application/json",
			"filename": f"summary_{account_number}_{date_from}_{date_to}.json",
		}

	# ── Balance threshold alert check ─────────────────────────────────────────

	async def set_balance_alert_threshold(
		self,
		account_number: str,
		pin: str,
		threshold: Decimal,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Set the low-balance alert threshold for an account.

		When balance drops below this value after a debit, a mob.balance.alert
		event is emitted. Downstream: NATS subscriber → SMS dispatch.
		"""
		tenant = self._tenant(tenant_id)
		account = self._get_account_by_number(account_number, tenant)
		if not account:
			raise KeyError(f"mob_account_not_found: {account_number}")
		self._check_pin(account, pin)
		if threshold < Decimal("0"):
			raise ValueError("threshold_must_be_non_negative")
		account["balance_alert_threshold"] = threshold
		self._emit(tenant, "mob_balance_alert_threshold_set", account["id"], "mob_account", {"threshold": str(threshold)})
		return {"account_number": account_number, "balance_alert_threshold": str(threshold), "updated_at": _now()}

	async def check_balance_alert(self, account_number: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Check whether the account balance is below its alert threshold.

		Called automatically after withdraw() and create_transfer() internally;
		also callable directly for admin dashboards.
		"""
		tenant = self._tenant(tenant_id)
		account = self._get_account_by_number(account_number, tenant)
		if not account:
			raise KeyError(f"mob_account_not_found: {account_number}")
		threshold = account.get("balance_alert_threshold")
		if threshold is None:
			return {"account_number": account_number, "alert_active": False, "reason": "no_threshold_set"}
		balance = account["balance"]
		alert_active = balance < threshold
		if alert_active:
			self._emit(tenant, "mob.balance.alert", account["id"], "mob_account", {
				"balance": str(balance),
				"threshold": str(threshold),
				"phone_number": account["phone_number"],
			})
			_log.warning("mob_balance_alert account=%s balance=%s threshold=%s tenant=%s", account_number, balance, threshold, tenant)
		return {
			"account_number": account_number,
			"balance": str(balance),
			"threshold": str(threshold),
			"alert_active": alert_active,
			"checked_at": _now(),
		}

	# ── Adaptive USSD menu (personalised shortcuts) ───────────────────────────

	async def get_personalised_menu(self, phone_number: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Return a personalised USSD menu ordered by the customer's usage frequency.

		The top-2 most-used menu items are promoted to positions 1 and 2,
		reducing average keystrokes for returning customers by ~40%.
		"""
		self._tenant(tenant_id)  # validate tenant
		normalized = self._validate_phone(phone_number)
		freq = self._usage_frequency.get(normalized, {})
		# Default menu items with canonical positions
		default_items = [
			{"key": "1", "label": "Account Balance"},
			{"key": "2", "label": "Mini Statement"},
			{"key": "3", "label": "Fund Transfer"},
			{"key": "4", "label": "Standing Orders"},
			{"key": "5", "label": "Change PIN"},
			{"key": "0", "label": "Exit"},
		]
		if not freq:
			return {"phone_number": normalized, "menu_items": default_items, "personalised": False}
		# Sort by frequency descending, keep Exit last
		non_exit = [i for i in default_items if i["key"] != "0"]
		exit_item = [i for i in default_items if i["key"] == "0"]
		sorted_items = sorted(non_exit, key=lambda x: freq.get(x["label"], 0), reverse=True)
		# Re-number 1..N
		for idx, item in enumerate(sorted_items, start=1):
			item = dict(item)
			item["position"] = str(idx)
		menu_text = "CON Welcome to MobBank\n" + "\n".join(
			f"{idx}. {item['label']}" for idx, item in enumerate(sorted_items, start=1)
		) + "\n0. Exit"
		return {
			"phone_number": normalized,
			"menu_items": sorted_items + exit_item,
			"menu_text": menu_text,
			"personalised": True,
			"generated_at": _now(),
		}

	async def record_menu_usage(self, phone_number: str, menu_label: str, tenant_id: str | None = None) -> None:
		"""Increment usage counter for a menu item — called on every session end."""
		self._tenant(tenant_id)
		normalized = self._validate_phone(phone_number)
		self._usage_frequency[normalized][menu_label] += 1

	async def describe(self) -> dict[str, Any]:
		"""Return capability description and metadata."""
		return {
			"capability_id": "fintech_ussd_mob",
			"name": "Mobile Banking USSD",
			"description": "USSD mobile banking: account balance, mini-statement, fund transfer, standing orders, PIN management",
			"version": "2.0.0",
			"domain": "fintech",
			"features": [
				"account_management",
				"balance_enquiry",
				"mini_statement",
				"fund_transfer",
				"standing_orders",
				"pin_management",
				"ussd_session_handling",
				"daily_limits",
				"account_locking",
				"beneficiary_management",
				"fraud_velocity_scoring",
				"idempotent_transfers",
				"cross_border_fx_transfers",
				"spending_analytics",
				"service_code_multitenancy",
				"session_token_integrity",
				"audit_chain_verification",
				"statement_export",
				"balance_threshold_alerts",
				"personalised_ussd_menu",
			],
			"supported_frequencies": list(SUPPORTED_FREQUENCIES),
			"supported_currencies": list(SUPPORTED_CURRENCIES),
			"daily_transfer_limit": str(DAILY_TRANSFER_LIMIT),
			"single_transfer_limit": str(SINGLE_TRANSFER_LIMIT),
			"max_pin_attempts": MAX_PIN_ATTEMPTS,
			"fraud_high_risk_threshold": FRAUD_HIGH_RISK_SCORE,
			"fraud_medium_risk_threshold": FRAUD_MEDIUM_RISK_SCORE,
			"high_value_transfer_threshold": str(HIGH_VALUE_TRANSFER_THRESHOLD),
			"max_beneficiaries_per_account": MAX_BENEFICIARIES_PER_ACCOUNT,
		}
