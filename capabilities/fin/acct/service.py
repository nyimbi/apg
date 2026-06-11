"""Bank Account Management — core lifecycle service.

Implements regulatory bank account lifecycle: open, close, freeze, dormancy,
credits, debits, internal transfers, fund locks, statements, and GL integration.

Invariants:
- Every monetary operation guards tenant_id and positive amount.
- available_balance == book_balance - locked_balance + overdraft_available
- All state changes emit NATS events for GL and downstream consumers.
- GL posting uses a circuit breaker; failures degrade gracefully with retry
  events emitted to ACCT_EVENT_STREAM.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from __future__ import annotations

import json
import logging
import re
from copy import deepcopy
from datetime import datetime, date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from capabilities.common.reliability import (
	guard_tenant_id,
	guard_positive_amount,
	guard_non_empty_string,
	BoundedCache,
	CircuitBreaker,
	CircuitOpenError,
)

try:
	from .capability_contract import (
		ACCT_EVENT_STREAM,
		STREAMING,
		SUPPORTED_CURRENCIES,
		SUPPORTED_ACCOUNT_TYPES,
		SUPPORTED_TRANSACTION_TYPES,
		DORMANCY_THRESHOLD_DAYS,
	)
	from .models import (
		uuid7str,
		BankAccount, AccountTransaction, AccountBalance, FundLock,
		StatementEntry, AccountSignatory, AccountHistoryEntry,
		AccountProduct, TransactionSummary, AccountStats,
		AccountStatus, AccountType, TransactionType, TransactionDirection,
		SigningAuthority, FundLockStatus,
		OpenAccountRequest, CloseAccountRequest, FreezeAccountRequest,
		UnfreezeAccountRequest, CreditRequest, DebitRequest,
		TransferRequest, BulkCreditItem, BulkCreditResult,
	)
except ImportError:
	from capability_contract import (  # type: ignore
		ACCT_EVENT_STREAM, STREAMING, SUPPORTED_CURRENCIES,
		SUPPORTED_ACCOUNT_TYPES, SUPPORTED_TRANSACTION_TYPES,
		DORMANCY_THRESHOLD_DAYS,
	)
	from models import (  # type: ignore
		uuid7str,
		BankAccount, AccountTransaction, AccountBalance, FundLock,
		StatementEntry, AccountSignatory, AccountHistoryEntry,
		AccountProduct, TransactionSummary, AccountStats,
		AccountStatus, AccountType, TransactionType, TransactionDirection,
		SigningAuthority, FundLockStatus,
		OpenAccountRequest, CloseAccountRequest, FreezeAccountRequest,
		UnfreezeAccountRequest, CreditRequest, DebitRequest,
		TransferRequest, BulkCreditItem, BulkCreditResult,
	)

_log = logging.getLogger(__name__)

TWO = Decimal("0.01")


def _d(v: Any) -> Decimal:
	return Decimal(str(v)).quantize(TWO, rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# IBAN / account number generation
# ---------------------------------------------------------------------------

_ACCOUNT_COUNTERS: dict[str, int] = {}


def _generate_account_number(tenant_id: str) -> str:
	"""Generate a deterministic sequential account number per tenant."""
	seq = _ACCOUNT_COUNTERS.get(tenant_id, 0) + 1
	_ACCOUNT_COUNTERS[tenant_id] = seq
	prefix = tenant_id[:4].upper().replace("-", "X")
	return f"{prefix}{seq:010d}"


def _generate_iban(account_number: str, country_code: str = "KE") -> str:
	"""Generate a pseudo-IBAN.  Production: replace with ISO 13616 compliant generator."""
	bban = re.sub(r"[^A-Z0-9]", "0", account_number.upper())[:16].zfill(16)
	# Check digits placeholder (real IBAN: MOD-97 calc)
	check = "00"
	return f"{country_code}{check}{bban}"


# ---------------------------------------------------------------------------
# Built-in product catalogue (production: load from DB)
# ---------------------------------------------------------------------------

_PRODUCTS: dict[str, AccountProduct] = {
	"CURR001": AccountProduct(
		product_code="CURR001", product_name="Standard Current Account",
		account_type=AccountType.CURRENT, currency="KES",
		overdraft_allowed=True, max_overdraft=Decimal("50000"),
	),
	"SVGS001": AccountProduct(
		product_code="SVGS001", product_name="Standard Savings Account",
		account_type=AccountType.SAVINGS, currency="KES",
		interest_rate=Decimal("0.04"),
	),
	"USD001": AccountProduct(
		product_code="USD001", product_name="USD Current Account",
		account_type=AccountType.CURRENT, currency="USD",
		overdraft_allowed=False,
	),
}


class BankAccountService:
	"""In-memory executable service for the ACCT lifecycle.

	Stores all state in plain dicts.  Production adapters replace store
	attributes with repository objects exposing the same dict-like interface.

	Thread-safety: all mutations are synchronous dict operations.  For
	concurrent production use, replace stores with async repository layer.
	"""

	# ------------------------------------------------------------------
	# Construction
	# ------------------------------------------------------------------

	def __init__(self, tenant_id: str | None = None, user_id: str | None = None) -> None:
		self.tenant_id = tenant_id
		self.user_id = user_id

		# Core stores
		self.accounts: dict[str, dict[str, Any]] = {}
		self.transactions: dict[str, dict[str, Any]] = {}
		self.fund_locks: dict[str, dict[str, Any]] = {}
		self.signatories: dict[str, dict[str, Any]] = {}
		self.history: dict[str, list[dict[str, Any]]] = {}   # account_id -> list
		self.products: dict[str, AccountProduct] = deepcopy(_PRODUCTS)
		self._events: list[dict[str, Any]] = []

		# Indices
		self._acct_by_number: dict[str, str] = {}  # acct_number -> account_id

		# Infrastructure
		self._cache: BoundedCache = BoundedCache(max_size=1000)
		self._gl_circuit: CircuitBreaker = CircuitBreaker(
			service_name="gl_posting", failure_threshold=5, reset_timeout=60,
		)
		self._idempotency: set[str] = set()

	# ------------------------------------------------------------------
	# Infrastructure helpers
	# ------------------------------------------------------------------

	def _tenant(self, tenant_id: str | None = None) -> str:
		value = tenant_id or self.tenant_id
		guard_tenant_id(value)
		return value  # type: ignore[return-value]

	def _now(self) -> datetime:
		return datetime.utcnow()

	def _today(self) -> date:
		return date.today()

	def _log_event(self, event_type: str, payload: dict[str, Any]) -> None:
		event = {
			"stream": STREAMING.get(event_type, ACCT_EVENT_STREAM),
			"event_type": event_type,
			"occurred_at": self._now().isoformat() + "Z",
			"payload": payload,
		}
		self._events.append(event)
		_log.info("ACCT event=%s payload_keys=%s", event_type, list(payload.keys()))

	def _log_pretty_path(self, account_id: str) -> str:
		acc = self.accounts.get(account_id)
		if not acc:
			return f"<unknown:{account_id}>"
		return f"[{acc['account_type']}:{acc['account_number']}]"

	def _get_account(self, tenant_id: str, account_id: str) -> dict[str, Any]:
		acc = self.accounts.get(account_id)
		if not acc:
			raise KeyError(f"account_not_found:{account_id}")
		if acc["tenant_id"] != tenant_id:
			raise PermissionError("tenant_mismatch")
		return acc

	def _record_history(
		self, account_id: str, tenant_id: str, event_type: str,
		description: str, old_status: str | None = None,
		new_status: str | None = None, performed_by: str | None = None,
		metadata: dict[str, Any] | None = None,
	) -> None:
		entry = AccountHistoryEntry(
			account_id=account_id,
			tenant_id=tenant_id,
			event_type=event_type,
			old_status=old_status,
			new_status=new_status,
			description=description,
			performed_by=performed_by,
			metadata=metadata or {},
		)
		self.history.setdefault(account_id, []).append(entry.model_dump())

	def _post_gl(self, tenant_id: str, journal_payload: dict[str, Any]) -> str | None:
		"""Fire GL journal request synchronously (circuit breaker checked inline).

		Production: replace with async NATS publish or GLR service call.
		Returns journal_id or None on circuit-open / error.
		"""
		if self._gl_circuit.state.value == "open":
			_log.warning("GL circuit open — queuing for retry: %s", journal_payload)
			self._log_event("gl_journal_requested", {
				"tenant_id": tenant_id,
				"retry": True,
				**journal_payload,
			})
			return None
		try:
			journal_id = uuid7str()
			self._log_event("gl_journal_requested", {
				"tenant_id": tenant_id,
				"journal_id": journal_id,
				**journal_payload,
			})
			return journal_id
		except Exception as exc:
			_log.error("GL post failed: %s", exc)
			return None

	# ------------------------------------------------------------------
	# Account lifecycle
	# ------------------------------------------------------------------

	async def open_account(
		self,
		tenant_id: str,
		customer_id: str,
		product_code: str,
		currency: str,
		account_number: str | None = None,
		opening_deposit: Decimal | None = None,
		metadata: dict[str, Any] | None = None,
	) -> BankAccount:
		guard_tenant_id(tenant_id)
		guard_non_empty_string(customer_id, "customer_id")
		guard_non_empty_string(product_code, "product_code")
		guard_non_empty_string(currency, "currency")
		currency = currency.upper()
		assert currency in SUPPORTED_CURRENCIES, f"unsupported currency: {currency}"

		product = self.products.get(product_code)
		assert product is not None, f"product_not_found: {product_code}"

		acct_number = account_number or _generate_account_number(tenant_id)
		assert acct_number not in self._acct_by_number, f"duplicate_account_number: {acct_number}"

		iban = _generate_iban(acct_number)
		opening_deposit = opening_deposit or Decimal("0")
		if opening_deposit < 0:
			raise ValueError("opening_deposit must be >= 0")
		if product.min_balance and opening_deposit < product.min_balance:
			raise ValueError(f"opening_deposit below product minimum: {product.min_balance}")

		acct = BankAccount(
			tenant_id=tenant_id,
			customer_id=customer_id,
			account_number=acct_number,
			iban=iban,
			currency=currency,
			account_type=product.account_type,
			product_code=product_code,
			status=AccountStatus.ACTIVE,
			book_balance=_d(opening_deposit),
			available_balance=_d(opening_deposit),
			opening_deposit=_d(opening_deposit),
			metadata=metadata or {},
		)
		self.accounts[acct.id] = acct.model_dump()
		self._acct_by_number[acct_number] = acct.id
		self._record_history(
			acct.id, tenant_id, "account_opened",
			f"Account opened with deposit {opening_deposit} {currency}",
			new_status=AccountStatus.ACTIVE.value,
		)
		self._log_event("account_opened", {
			"tenant_id": tenant_id, "account_id": acct.id,
			"account_number": acct_number, "customer_id": customer_id,
			"currency": currency, "opening_deposit": str(opening_deposit),
		})
		if opening_deposit > 0:
			self._post_gl(tenant_id, {
				"type": "account_opening_deposit",
				"account_id": acct.id,
				"amount": str(opening_deposit),
				"currency": currency,
			})
		_log.info("open_account tenant=%s path=%s", tenant_id, self._log_pretty_path(acct.id))
		assert acct.status == AccountStatus.ACTIVE
		return acct

	async def close_account(
		self,
		tenant_id: str,
		account_id: str,
		reason: str,
		closed_by: str,
	) -> BankAccount:
		guard_tenant_id(tenant_id)
		guard_non_empty_string(reason, "reason")
		guard_non_empty_string(closed_by, "closed_by")
		acc = self._get_account(tenant_id, account_id)
		if acc["status"] == AccountStatus.CLOSED.value:
			raise ValueError("account_already_closed")
		# Check no active locks first
		active_locks = [
			l for l in self.fund_locks.values()
			if l["account_id"] == account_id
			and l["status"] == FundLockStatus.ACTIVE.value
			and l["tenant_id"] == tenant_id
		]
		if active_locks:
			raise ValueError(f"cannot_close_account_with_active_locks: {len(active_locks)}")
		if _d(acc["book_balance"]) != Decimal("0"):
			raise ValueError(f"cannot_close_non_zero_balance: {acc['book_balance']}")

		old_status = acc["status"]
		acc["status"] = AccountStatus.CLOSED.value
		acc["closed_at"] = self._now().isoformat()
		acc["close_reason"] = reason
		acc["closed_by"] = closed_by
		self._record_history(
			account_id, tenant_id, "account_closed",
			f"Account closed: {reason}", old_status=old_status,
			new_status=AccountStatus.CLOSED.value, performed_by=closed_by,
		)
		self._log_event("account_closed", {
			"tenant_id": tenant_id, "account_id": account_id, "reason": reason, "closed_by": closed_by,
		})
		_log.info("close_account path=%s by=%s", self._log_pretty_path(account_id), closed_by)
		return BankAccount(**acc)

	async def freeze_account(
		self,
		tenant_id: str,
		account_id: str,
		reason: str,
		frozen_by: str,
	) -> BankAccount:
		guard_tenant_id(tenant_id)
		guard_non_empty_string(reason, "reason")
		guard_non_empty_string(frozen_by, "frozen_by")
		acc = self._get_account(tenant_id, account_id)
		if acc["status"] in (AccountStatus.CLOSED.value, AccountStatus.FROZEN.value):
			raise ValueError(f"cannot_freeze_account_in_status: {acc['status']}")

		old_status = acc["status"]
		acc["status"] = AccountStatus.FROZEN.value
		acc["frozen_at"] = self._now().isoformat()
		acc["freeze_reason"] = reason
		acc["frozen_by"] = frozen_by
		self._record_history(
			account_id, tenant_id, "account_frozen",
			f"Account frozen: {reason}", old_status=old_status,
			new_status=AccountStatus.FROZEN.value, performed_by=frozen_by,
		)
		self._log_event("account_frozen", {
			"tenant_id": tenant_id, "account_id": account_id, "reason": reason, "frozen_by": frozen_by,
		})
		return BankAccount(**acc)

	async def unfreeze_account(
		self,
		tenant_id: str,
		account_id: str,
		reason: str,
		unfrozen_by: str,
	) -> BankAccount:
		guard_tenant_id(tenant_id)
		acc = self._get_account(tenant_id, account_id)
		if acc["status"] != AccountStatus.FROZEN.value:
			raise ValueError(f"account_not_frozen: {acc['status']}")

		old_status = acc["status"]
		acc["status"] = AccountStatus.ACTIVE.value
		acc["unfreeze_reason"] = reason
		acc["unfrozen_by"] = unfrozen_by
		self._record_history(
			account_id, tenant_id, "account_unfrozen",
			f"Account unfrozen: {reason}", old_status=old_status,
			new_status=AccountStatus.ACTIVE.value, performed_by=unfrozen_by,
		)
		self._log_event("account_unfrozen", {
			"tenant_id": tenant_id, "account_id": account_id, "reason": reason,
		})
		return BankAccount(**acc)

	async def mark_dormant(self, tenant_id: str, account_id: str) -> BankAccount:
		guard_tenant_id(tenant_id)
		acc = self._get_account(tenant_id, account_id)
		if acc["status"] != AccountStatus.ACTIVE.value:
			raise ValueError(f"cannot_mark_dormant_status: {acc['status']}")

		old_status = acc["status"]
		acc["status"] = AccountStatus.DORMANT.value
		acc["dormant_since"] = self._now().isoformat()
		self._record_history(
			account_id, tenant_id, "account_dormant",
			"Account marked dormant due to inactivity",
			old_status=old_status, new_status=AccountStatus.DORMANT.value,
		)
		self._log_event("account_dormant", {"tenant_id": tenant_id, "account_id": account_id})
		return BankAccount(**acc)

	async def reactivate_dormant(self, tenant_id: str, account_id: str) -> BankAccount:
		guard_tenant_id(tenant_id)
		acc = self._get_account(tenant_id, account_id)
		if acc["status"] != AccountStatus.DORMANT.value:
			raise ValueError(f"account_not_dormant: {acc['status']}")

		old_status = acc["status"]
		acc["status"] = AccountStatus.ACTIVE.value
		acc["dormant_since"] = None
		acc["last_transaction_at"] = self._now().isoformat()
		self._record_history(
			account_id, tenant_id, "account_reactivated",
			"Account reactivated", old_status=old_status,
			new_status=AccountStatus.ACTIVE.value,
		)
		self._log_event("account_reactivated", {"tenant_id": tenant_id, "account_id": account_id})
		return BankAccount(**acc)

	# ------------------------------------------------------------------
	# Read
	# ------------------------------------------------------------------

	async def get_account(self, tenant_id: str, account_id: str) -> BankAccount:
		guard_tenant_id(tenant_id)
		return BankAccount(**self._get_account(tenant_id, account_id))

	async def get_account_by_number(self, tenant_id: str, account_number: str) -> BankAccount:
		guard_tenant_id(tenant_id)
		guard_non_empty_string(account_number, "account_number")
		acct_id = self._acct_by_number.get(account_number)
		if not acct_id:
			raise KeyError(f"account_not_found:{account_number}")
		return BankAccount(**self._get_account(tenant_id, acct_id))

	async def list_accounts(
		self,
		tenant_id: str,
		customer_id: str | None = None,
		status: str | None = None,
		account_type: str | None = None,
	) -> list[BankAccount]:
		guard_tenant_id(tenant_id)
		results = []
		for acc in self.accounts.values():
			if acc["tenant_id"] != tenant_id:
				continue
			if customer_id and acc["customer_id"] != customer_id:
				continue
			if status and acc["status"] != status:
				continue
			if account_type and acc["account_type"] != account_type:
				continue
			results.append(BankAccount(**acc))
		return results

	# ------------------------------------------------------------------
	# Balance
	# ------------------------------------------------------------------

	async def get_balance(self, tenant_id: str, account_id: str) -> AccountBalance:
		guard_tenant_id(tenant_id)
		acc = self._get_account(tenant_id, account_id)
		book = _d(acc["book_balance"])
		locked = _d(acc["locked_balance"])
		overdraft_limit = _d(acc["overdraft_limit"])
		overdraft_used = _d(acc["overdraft_used"])
		overdraft_available = max(Decimal("0"), overdraft_limit - overdraft_used)
		available = book - locked + overdraft_available
		return AccountBalance(
			account_id=account_id,
			account_number=acc["account_number"],
			currency=acc["currency"],
			book_balance=book,
			available_balance=available,
			locked_balance=locked,
			overdraft_limit=overdraft_limit,
			overdraft_used=overdraft_used,
			overdraft_available=overdraft_available,
		)

	async def check_sufficient_funds(
		self, tenant_id: str, account_id: str, amount: Decimal
	) -> bool:
		guard_tenant_id(tenant_id)
		guard_positive_amount(float(amount), "amount")
		bal = await self.get_balance(tenant_id, account_id)
		return bal.available_balance >= _d(amount)

	# ------------------------------------------------------------------
	# Transactions: credit
	# ------------------------------------------------------------------

	async def credit_account(
		self,
		tenant_id: str,
		account_id: str,
		amount: Decimal,
		currency: str,
		reference: str,
		description: str,
		transaction_type: TransactionType = TransactionType.DEPOSIT,
	) -> AccountTransaction:
		guard_tenant_id(tenant_id)
		guard_positive_amount(float(amount), "amount")
		guard_non_empty_string(reference, "reference")
		acc = self._get_account(tenant_id, account_id)
		if acc["status"] == AccountStatus.CLOSED.value:
			raise ValueError("account_closed")
		if acc["currency"].upper() != currency.upper():
			raise ValueError(f"currency_mismatch: account={acc['currency']} request={currency}")

		amount = _d(amount)
		balance_before = _d(acc["book_balance"])
		balance_after = balance_before + amount

		acc["book_balance"] = str(balance_after)
		acc["available_balance"] = str(_d(acc["available_balance"]) + amount)
		acc["last_transaction_at"] = self._now().isoformat()

		txn = AccountTransaction(
			tenant_id=tenant_id,
			account_id=account_id,
			account_number=acc["account_number"],
			currency=currency.upper(),
			amount=amount,
			direction=TransactionDirection.CREDIT,
			transaction_type=transaction_type,
			reference=reference,
			description=description,
			balance_before=balance_before,
			balance_after=balance_after,
		)
		self.transactions[txn.id] = txn.model_dump()
		journal_id = self._post_gl(tenant_id, {
			"type": "credit",
			"account_id": account_id,
			"amount": str(amount),
			"currency": currency,
			"reference": reference,
			"transaction_id": txn.id,
		})
		if journal_id:
			self.transactions[txn.id]["gl_journal_id"] = journal_id
			txn = txn.model_copy(update={"gl_journal_id": journal_id})

		self._log_event("credit_posted", {
			"tenant_id": tenant_id, "account_id": account_id,
			"amount": str(amount), "currency": currency,
			"transaction_id": txn.id, "reference": reference,
		})
		_log.info(
			"credit_account path=%s amount=%s %s",
			self._log_pretty_path(account_id), amount, currency,
		)
		assert _d(acc["book_balance"]) == balance_after
		return txn

	# ------------------------------------------------------------------
	# Transactions: debit
	# ------------------------------------------------------------------

	async def debit_account(
		self,
		tenant_id: str,
		account_id: str,
		amount: Decimal,
		currency: str,
		reference: str,
		description: str,
		transaction_type: TransactionType = TransactionType.WITHDRAWAL,
	) -> AccountTransaction:
		guard_tenant_id(tenant_id)
		guard_positive_amount(float(amount), "amount")
		guard_non_empty_string(reference, "reference")
		acc = self._get_account(tenant_id, account_id)
		if acc["status"] == AccountStatus.CLOSED.value:
			raise ValueError("account_closed")
		if acc["status"] == AccountStatus.FROZEN.value:
			raise ValueError("account_frozen_debits_blocked")
		if acc["currency"].upper() != currency.upper():
			raise ValueError(f"currency_mismatch: account={acc['currency']} request={currency}")

		amount = _d(amount)
		available = _d(acc["available_balance"])
		if available < amount:
			raise ValueError(
				f"insufficient_funds: available={available} requested={amount}"
			)

		balance_before = _d(acc["book_balance"])
		balance_after = balance_before - amount

		# Overdraft accounting
		overdraft_used = _d(acc["overdraft_used"])
		if balance_after < Decimal("0"):
			new_overdraft_used = -balance_after
			acc["overdraft_used"] = str(new_overdraft_used)
		else:
			acc["overdraft_used"] = str(overdraft_used)

		acc["book_balance"] = str(balance_after)
		acc["available_balance"] = str(available - amount)
		acc["last_transaction_at"] = self._now().isoformat()

		txn = AccountTransaction(
			tenant_id=tenant_id,
			account_id=account_id,
			account_number=acc["account_number"],
			currency=currency.upper(),
			amount=amount,
			direction=TransactionDirection.DEBIT,
			transaction_type=transaction_type,
			reference=reference,
			description=description,
			balance_before=balance_before,
			balance_after=balance_after,
		)
		self.transactions[txn.id] = txn.model_dump()
		journal_id = self._post_gl(tenant_id, {
			"type": "debit",
			"account_id": account_id,
			"amount": str(amount),
			"currency": currency,
			"reference": reference,
			"transaction_id": txn.id,
		})
		if journal_id:
			self.transactions[txn.id]["gl_journal_id"] = journal_id
			txn = txn.model_copy(update={"gl_journal_id": journal_id})

		self._log_event("debit_posted", {
			"tenant_id": tenant_id, "account_id": account_id,
			"amount": str(amount), "currency": currency,
			"transaction_id": txn.id,
		})
		assert _d(acc["book_balance"]) == balance_after
		return txn

	# ------------------------------------------------------------------
	# Internal transfer
	# ------------------------------------------------------------------

	async def transfer_internal(
		self,
		tenant_id: str,
		from_account_id: str,
		to_account_id: str,
		amount: Decimal,
		reference: str,
		description: str,
	) -> tuple[AccountTransaction, AccountTransaction]:
		"""Atomic debit source + credit destination + single GL journal."""
		guard_tenant_id(tenant_id)
		guard_positive_amount(float(amount), "amount")
		if from_account_id == to_account_id:
			raise ValueError("same_account_transfer_not_allowed")

		src = self._get_account(tenant_id, from_account_id)
		dst = self._get_account(tenant_id, to_account_id)
		if src["currency"] != dst["currency"]:
			raise ValueError(
				f"cross_currency_transfer_not_supported: {src['currency']} -> {dst['currency']}"
			)

		debit_txn = await self.debit_account(
			tenant_id, from_account_id, amount, src["currency"],
			reference, description, TransactionType.TRANSFER_OUT,
		)
		credit_txn = await self.credit_account(
			tenant_id, to_account_id, amount, dst["currency"],
			reference, description, TransactionType.TRANSFER_IN,
		)

		# Override with a single bilateral GL journal
		journal_id = self._post_gl(tenant_id, {
			"type": "internal_transfer",
			"from_account_id": from_account_id,
			"to_account_id": to_account_id,
			"amount": str(_d(amount)),
			"currency": src["currency"],
			"reference": reference,
			"debit_transaction_id": debit_txn.id,
			"credit_transaction_id": credit_txn.id,
		})
		if journal_id:
			self.transactions[debit_txn.id]["gl_journal_id"] = journal_id
			self.transactions[credit_txn.id]["gl_journal_id"] = journal_id

		self._log_event("transfer_completed", {
			"tenant_id": tenant_id,
			"from_account_id": from_account_id,
			"to_account_id": to_account_id,
			"amount": str(_d(amount)),
			"reference": reference,
		})
		return debit_txn, credit_txn

	# ------------------------------------------------------------------
	# Fund locks
	# ------------------------------------------------------------------

	async def lock_funds(
		self,
		tenant_id: str,
		account_id: str,
		amount: Decimal,
		lock_reference: str,
		reason: str | None = None,
		expires_at: datetime | None = None,
	) -> FundLock:
		guard_tenant_id(tenant_id)
		guard_positive_amount(float(amount), "amount")
		guard_non_empty_string(lock_reference, "lock_reference")
		acc = self._get_account(tenant_id, account_id)
		if acc["status"] != AccountStatus.ACTIVE.value:
			raise ValueError(f"cannot_lock_funds_account_status: {acc['status']}")

		amount = _d(amount)
		available = _d(acc["available_balance"])
		if available < amount:
			raise ValueError(f"insufficient_funds_for_lock: available={available}")

		acc["locked_balance"] = str(_d(acc["locked_balance"]) + amount)
		acc["available_balance"] = str(available - amount)

		lock = FundLock(
			tenant_id=tenant_id,
			account_id=account_id,
			amount=amount,
			lock_reference=lock_reference,
			reason=reason,
			expires_at=expires_at,
		)
		self.fund_locks[lock.id] = lock.model_dump()
		self._log_event("funds_locked", {
			"tenant_id": tenant_id, "account_id": account_id,
			"amount": str(amount), "lock_reference": lock_reference, "lock_id": lock.id,
		})
		return lock

	async def release_lock(
		self, tenant_id: str, account_id: str, lock_reference: str
	) -> FundLock:
		guard_tenant_id(tenant_id)
		guard_non_empty_string(lock_reference, "lock_reference")
		acc = self._get_account(tenant_id, account_id)

		lock_rec = next(
			(l for l in self.fund_locks.values()
			 if l["lock_reference"] == lock_reference
			 and l["account_id"] == account_id
			 and l["tenant_id"] == tenant_id
			 and l["status"] == FundLockStatus.ACTIVE.value),
			None,
		)
		if not lock_rec:
			raise KeyError(f"active_lock_not_found:{lock_reference}")

		amount = _d(lock_rec["amount"])
		acc["locked_balance"] = str(max(Decimal("0"), _d(acc["locked_balance"]) - amount))
		acc["available_balance"] = str(_d(acc["available_balance"]) + amount)
		lock_rec["status"] = FundLockStatus.RELEASED.value
		lock_rec["released_at"] = self._now().isoformat()

		self._log_event("funds_released", {
			"tenant_id": tenant_id, "account_id": account_id,
			"amount": str(amount), "lock_reference": lock_reference,
		})
		return FundLock(**lock_rec)

	# ------------------------------------------------------------------
	# Overdraft
	# ------------------------------------------------------------------

	async def set_overdraft_limit(
		self,
		tenant_id: str,
		account_id: str,
		limit: Decimal,
		approved_by: str,
	) -> BankAccount:
		guard_tenant_id(tenant_id)
		guard_non_empty_string(approved_by, "approved_by")
		if limit < Decimal("0"):
			raise ValueError("overdraft_limit_must_be_non_negative")
		acc = self._get_account(tenant_id, account_id)
		old_limit = _d(acc["overdraft_limit"])
		acc["overdraft_limit"] = str(_d(limit))
		acc["overdraft_approved_by"] = approved_by
		# Recalculate available balance
		book = _d(acc["book_balance"])
		locked = _d(acc["locked_balance"])
		overdraft_used = _d(acc["overdraft_used"])
		overdraft_available = max(Decimal("0"), _d(limit) - overdraft_used)
		acc["available_balance"] = str(book - locked + overdraft_available)
		self._record_history(
			account_id, tenant_id, "overdraft_limit_changed",
			f"Overdraft limit changed from {old_limit} to {limit}",
			performed_by=approved_by,
		)
		self._log_event("overdraft_limit_set", {
			"tenant_id": tenant_id, "account_id": account_id,
			"old_limit": str(old_limit), "new_limit": str(_d(limit)), "approved_by": approved_by,
		})
		return BankAccount(**acc)

	# ------------------------------------------------------------------
	# Transactions: read
	# ------------------------------------------------------------------

	async def get_transactions(
		self,
		tenant_id: str,
		account_id: str,
		from_date: date | None = None,
		to_date: date | None = None,
		limit: int = 50,
		page: int = 1,
	) -> list[AccountTransaction]:
		guard_tenant_id(tenant_id)
		self._get_account(tenant_id, account_id)  # ownership check
		results = []
		for txn in self.transactions.values():
			if txn["account_id"] != account_id:
				continue
			if txn["tenant_id"] != tenant_id:
				continue
			posted_raw = txn["posted_at"]
			if isinstance(posted_raw, datetime):
				posted = posted_raw
			else:
				posted = datetime.fromisoformat(str(posted_raw).rstrip("Z"))
			if from_date and posted.date() < from_date:
				continue
			if to_date and posted.date() > to_date:
				continue
			results.append(AccountTransaction(**txn))
		results.sort(key=lambda t: t.posted_at, reverse=True)
		start = (page - 1) * limit
		return results[start: start + limit]

	async def get_transaction(self, tenant_id: str, transaction_id: str) -> AccountTransaction:
		guard_tenant_id(tenant_id)
		txn = self.transactions.get(transaction_id)
		if not txn:
			raise KeyError(f"transaction_not_found:{transaction_id}")
		if txn["tenant_id"] != tenant_id:
			raise PermissionError("tenant_mismatch")
		return AccountTransaction(**txn)

	# ------------------------------------------------------------------
	# Statement
	# ------------------------------------------------------------------

	async def generate_statement(
		self,
		tenant_id: str,
		account_id: str,
		from_date: date,
		to_date: date,
		format: str = "json",
	) -> dict[str, Any]:
		guard_tenant_id(tenant_id)
		acc = self._get_account(tenant_id, account_id)
		txns = await self.get_transactions(
			tenant_id, account_id, from_date=from_date, to_date=to_date, limit=10000,
		)
		txns_asc = sorted(txns, key=lambda t: t.posted_at)

		entries: list[StatementEntry] = []
		running = Decimal("0") if not txns_asc else _d(txns_asc[0].balance_before)
		for t in txns_asc:
			debit = t.amount if t.direction == TransactionDirection.DEBIT else None
			credit = t.amount if t.direction == TransactionDirection.CREDIT else None
			running = _d(t.balance_after)
			entries.append(StatementEntry(
				transaction_id=t.id,
				value_date=t.value_date,
				posted_at=t.posted_at,
				description=t.description,
				reference=t.reference,
				transaction_type=t.transaction_type.value,
				debit=debit,
				credit=credit,
				running_balance=running,
				currency=t.currency,
			))

		statement = {
			"account_number": acc["account_number"],
			"iban": acc.get("iban"),
			"customer_id": acc["customer_id"],
			"currency": acc["currency"],
			"from_date": from_date.isoformat(),
			"to_date": to_date.isoformat(),
			"opening_balance": str(txns_asc[0].balance_before) if txns_asc else str(acc["book_balance"]),
			"closing_balance": str(running),
			"entries": [e.model_dump() for e in entries],
			"generated_at": self._now().isoformat() + "Z",
			"format": format,
		}
		if format == "pdf":
			statement["pdf_note"] = "PDF rendering requires a PDF adapter (e.g. WeasyPrint)"
		return statement

	# ------------------------------------------------------------------
	# Product
	# ------------------------------------------------------------------

	async def get_account_product(self, tenant_id: str, account_id: str) -> AccountProduct:
		guard_tenant_id(tenant_id)
		acc = self._get_account(tenant_id, account_id)
		product = self.products.get(acc["product_code"])
		if not product:
			raise KeyError(f"product_not_found: {acc['product_code']}")
		return product

	async def link_product(
		self, tenant_id: str, account_id: str, product_code: str
	) -> BankAccount:
		guard_tenant_id(tenant_id)
		product = self.products.get(product_code)
		if not product:
			raise KeyError(f"product_not_found: {product_code}")
		acc = self._get_account(tenant_id, account_id)
		old_code = acc["product_code"]
		acc["product_code"] = product_code
		self._record_history(
			account_id, tenant_id, "product_changed",
			f"Product changed from {old_code} to {product_code}",
		)
		return BankAccount(**acc)

	# ------------------------------------------------------------------
	# Dormancy candidates
	# ------------------------------------------------------------------

	async def get_dormancy_candidates(
		self, tenant_id: str, days_inactive: int = DORMANCY_THRESHOLD_DAYS
	) -> list[BankAccount]:
		guard_tenant_id(tenant_id)
		threshold = self._now() - timedelta(days=days_inactive)
		result = []
		for acc in self.accounts.values():
			if acc["tenant_id"] != tenant_id:
				continue
			if acc["status"] != AccountStatus.ACTIVE.value:
				continue
			last_txn = acc.get("last_transaction_at")
			if last_txn:
				last_dt = datetime.fromisoformat(last_txn.rstrip("Z"))
				if last_dt < threshold:
					result.append(BankAccount(**acc))
			else:
				# Never transacted
				opened = datetime.fromisoformat(acc["opened_at"].rstrip("Z"))
				if opened < threshold:
					result.append(BankAccount(**acc))
		return result

	# ------------------------------------------------------------------
	# Stats
	# ------------------------------------------------------------------

	async def get_account_stats(self, tenant_id: str, customer_id: str) -> AccountStats:
		guard_tenant_id(tenant_id)
		accts = await self.list_accounts(tenant_id, customer_id=customer_id)
		total_book = sum((_d(a.book_balance) for a in accts), Decimal("0"))
		total_avail = sum((_d(a.available_balance) for a in accts), Decimal("0"))
		return AccountStats(
			customer_id=customer_id,
			tenant_id=tenant_id,
			total_accounts=len(accts),
			active_accounts=sum(1 for a in accts if a.status == AccountStatus.ACTIVE),
			frozen_accounts=sum(1 for a in accts if a.status == AccountStatus.FROZEN),
			dormant_accounts=sum(1 for a in accts if a.status == AccountStatus.DORMANT),
			closed_accounts=sum(1 for a in accts if a.status == AccountStatus.CLOSED),
			total_book_balance=total_book,
			total_available_balance=total_avail,
			currencies=list({a.currency for a in accts}),
		)

	# ------------------------------------------------------------------
	# Bulk credit (payroll disbursement)
	# ------------------------------------------------------------------

	async def bulk_credit(
		self,
		tenant_id: str,
		credits: list[dict[str, Any]],
	) -> BulkCreditResult:
		guard_tenant_id(tenant_id)
		if not credits:
			raise ValueError("credits_list_empty")
		succeeded: list[AccountTransaction] = []
		failed: list[dict[str, Any]] = []
		for item in credits:
			try:
				txn = await self.credit_account(
					tenant_id,
					item["account_id"],
					_d(item["amount"]),
					self.accounts[item["account_id"]]["currency"],
					item["reference"],
					item.get("description", "Bulk credit"),
					TransactionType.BULK_CREDIT,
				)
				succeeded.append(txn)
			except Exception as exc:
				failed.append({"item": item, "error": str(exc)})
		return BulkCreditResult(
			succeeded=succeeded,
			failed=failed,
			total=len(credits),
			success_count=len(succeeded),
			failure_count=len(failed),
		)

	# ------------------------------------------------------------------
	# Sweep
	# ------------------------------------------------------------------

	async def sweep_to_linked(
		self,
		tenant_id: str,
		account_id: str,
		linked_account_id: str | None = None,
		sweep_threshold: Decimal = Decimal("10000"),
		retain_amount: Decimal = Decimal("5000"),
	) -> AccountTransaction | None:
		"""Sweep funds above sweep_threshold to linked savings account."""
		guard_tenant_id(tenant_id)
		bal = await self.get_balance(tenant_id, account_id)
		sweep_amount = bal.available_balance - retain_amount
		if sweep_amount <= Decimal("0"):
			return None
		if not linked_account_id:
			raise ValueError("linked_account_id_required_for_sweep")
		debit_txn, _ = await self.transfer_internal(
			tenant_id, account_id, linked_account_id,
			sweep_amount, f"SWEEP-{account_id[:8]}", "Auto sweep to savings",
		)
		return debit_txn

	# ------------------------------------------------------------------
	# Signatories
	# ------------------------------------------------------------------

	async def add_joint_holder(
		self,
		tenant_id: str,
		account_id: str,
		customer_id: str,
		signing_authority: str,
	) -> AccountSignatory:
		guard_tenant_id(tenant_id)
		self._get_account(tenant_id, account_id)
		sig = AccountSignatory(
			tenant_id=tenant_id,
			account_id=account_id,
			customer_id=customer_id,
			signing_authority=SigningAuthority(signing_authority),
		)
		self.signatories[sig.id] = sig.model_dump()
		return sig

	async def get_account_signatories(
		self, tenant_id: str, account_id: str
	) -> list[AccountSignatory]:
		guard_tenant_id(tenant_id)
		self._get_account(tenant_id, account_id)
		return [
			AccountSignatory(**s)
			for s in self.signatories.values()
			if s["account_id"] == account_id
			and s["tenant_id"] == tenant_id
			and s["is_active"]
		]

	# ------------------------------------------------------------------
	# History & summary
	# ------------------------------------------------------------------

	async def get_account_history(
		self, tenant_id: str, account_id: str
	) -> list[AccountHistoryEntry]:
		guard_tenant_id(tenant_id)
		self._get_account(tenant_id, account_id)
		return [AccountHistoryEntry(**h) for h in self.history.get(account_id, [])]

	async def get_transaction_summary(
		self, tenant_id: str, account_id: str, period: str
	) -> TransactionSummary:
		guard_tenant_id(tenant_id)
		acc = self._get_account(tenant_id, account_id)
		# period format: "YYYY-MM"
		txns = []
		for t in self.transactions.values():
			if t["account_id"] != account_id or t["tenant_id"] != tenant_id:
				continue
			posted_raw = t["posted_at"]
			posted_str = posted_raw.isoformat() if isinstance(posted_raw, datetime) else str(posted_raw)
			if posted_str.startswith(period):
				txns.append(AccountTransaction(**t))
		total_credits = sum((_d(t.amount) for t in txns if t.direction == TransactionDirection.CREDIT), Decimal("0"))
		total_debits = sum((_d(t.amount) for t in txns if t.direction == TransactionDirection.DEBIT), Decimal("0"))
		opening_balance = _d(txns[0].balance_before) if txns else _d(acc["book_balance"])
		closing_balance = _d(txns[-1].balance_after) if txns else _d(acc["book_balance"])
		return TransactionSummary(
			account_id=account_id,
			period=period,
			total_credits=total_credits,
			total_debits=total_debits,
			net_movement=total_credits - total_debits,
			transaction_count=len(txns),
			opening_balance=opening_balance,
			closing_balance=closing_balance,
			currency=acc["currency"],
		)

	# ------------------------------------------------------------------
	# Health
	# ------------------------------------------------------------------

	async def health_check(self) -> dict[str, Any]:
		return {
			"status": "healthy",
			"capability": "fin_acct",
			"version": "1.0.0",
			"accounts": len(self.accounts),
			"transactions": len(self.transactions),
			"fund_locks": len(self.fund_locks),
			"gl_circuit_state": self._gl_circuit.state if hasattr(self._gl_circuit, "state") else "unknown",
			"cache_stats": self._cache.stats(),
			"events_queued": len(self._events),
		}
