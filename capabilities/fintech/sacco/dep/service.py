"""SACCO Deposits & Savings — full async service."""
from __future__ import annotations

import logging
from copy import deepcopy
from datetime import datetime, date
from decimal import Decimal, ROUND_HALF_UP
from typing import Any
from uuid import uuid4

_log = logging.getLogger(__name__)

CAPABILITY_ID = "fintech_sacco_dep"
PRODUCT_TYPES = {"regular", "fixed_deposit", "holiday", "junior", "institutional"}
POSTING_FREQUENCIES = {"daily", "monthly", "quarterly", "annually"}
PAYMENT_METHODS = {"cash", "mpesa", "bank_transfer", "cheque", "internal"}
ACCOUNT_STATUSES = {"active", "dormant", "frozen", "closed"}


class SaccoDepositsService:
	"""Async service for SACCO savings products, deposits, withdrawals, and interest accrual."""

	def __init__(self, tenant_id: str = "default") -> None:
		self.tenant_id = tenant_id
		self.products: dict[str, dict[str, Any]] = {}
		self.accounts: dict[str, dict[str, Any]] = {}
		self.transactions: dict[str, dict[str, Any]] = {}
		self.interest_postings: dict[str, dict[str, Any]] = {}
		self.minimum_balance_breaches: dict[str, dict[str, Any]] = {}
		self._audit_events: list[dict[str, Any]] = []
		self._account_counter: int = 0

	def _tenant(self, tenant_id: str | None = None) -> str:
		value = tenant_id or self.tenant_id
		if not value:
			raise PermissionError("tenant_context_required")
		return value

	def _record_id(self, prefix: str) -> str:
		return f"{prefix}-{uuid4().hex[:12]}"

	def _now(self) -> str:
		return datetime.utcnow().isoformat(timespec="seconds") + "Z"

	def _next_account_number(self, tenant_id: str) -> str:
		self._account_counter += 1
		return f"SAV-{tenant_id[:4].upper()}-{self._account_counter:08d}"

	def _emit(self, tenant_id: str, event_type: str, record: dict[str, Any]) -> None:
		self._audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"record_id": record.get("id", ""),
			"record_type": record.get("type", ""),
			"emitted_at": self._now(),
		})

	def _get_account(self, account_id: str, tenant_id: str) -> dict[str, Any]:
		acc = self.accounts.get(account_id)
		if not acc or acc["tenant_id"] != tenant_id:
			raise KeyError(f"account_not_found: {account_id}")
		return acc

	def _get_product(self, product_id: str, tenant_id: str) -> dict[str, Any]:
		p = self.products.get(product_id)
		if not p or p["tenant_id"] != tenant_id:
			raise KeyError(f"product_not_found: {product_id}")
		return p

	def _days_in_period(self, freq: str) -> int:
		return {"daily": 1, "monthly": 30, "quarterly": 91, "annually": 365}.get(freq, 30)

	# ── Health & Describe ─────────────────────────────────────────────────────

	async def health_check(self) -> dict[str, Any]:
		return {
			"service": CAPABILITY_ID,
			"status": "healthy",
			"product_count": len(self.products),
			"account_count": len(self.accounts),
			"active_accounts": sum(1 for a in self.accounts.values() if a.get("status") == "active"),
			"total_deposits": str(sum(a.get("balance", Decimal("0")) for a in self.accounts.values())),
			"checked_at": self._now(),
		}

	async def describe(self) -> dict[str, Any]:
		return {
			"capability_id": CAPABILITY_ID,
			"version": "1.0.0",
			"domain": "fintech",
			"description": "SACCO savings products, deposit taking, withdrawals, minimum balances, interest accrual",
			"product_types": list(PRODUCT_TYPES),
			"payment_methods": list(PAYMENT_METHODS),
		}

	async def get_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		t = self._tenant(tenant_id)
		return [deepcopy(e) for e in self._audit_events if e["tenant_id"] == t]

	# ── Savings Products ──────────────────────────────────────────────────────

	async def create_product(
		self,
		product_code: str,
		product_name: str,
		product_type: str,
		interest_rate_pa: float,
		tenant_id: str | None = None,
		min_balance: float = 0.0,
		min_opening_balance: float = 0.0,
		max_balance: float | None = None,
		lock_in_months: int = 0,
		interest_posting_frequency: str = "monthly",
		withdrawal_notice_days: int = 0,
		allow_overdraft: bool = False,
		tax_exempt: bool = False,
		description: str | None = None,
	) -> dict[str, Any]:
		"""Define a new savings product."""
		t = self._tenant(tenant_id)
		if product_type not in PRODUCT_TYPES:
			raise ValueError(f"invalid_product_type: {product_type}")
		if interest_posting_frequency not in POSTING_FREQUENCIES:
			raise ValueError(f"invalid_posting_frequency: {interest_posting_frequency}")
		for p in self.products.values():
			if p["tenant_id"] == t and p["product_code"] == product_code:
				raise ValueError(f"product_code_exists: {product_code}")
		product_id = self._record_id("prod")
		record: dict[str, Any] = {
			"id": product_id,
			"type": "sacco_savings_product",
			"tenant_id": t,
			"product_code": product_code,
			"product_name": product_name,
			"product_type": product_type,
			"interest_rate_pa": Decimal(str(interest_rate_pa)),
			"min_balance": Decimal(str(min_balance)),
			"min_opening_balance": Decimal(str(min_opening_balance)),
			"max_balance": Decimal(str(max_balance)) if max_balance is not None else None,
			"lock_in_months": lock_in_months,
			"interest_posting_frequency": interest_posting_frequency,
			"withdrawal_notice_days": withdrawal_notice_days,
			"allow_overdraft": allow_overdraft,
			"tax_exempt": tax_exempt,
			"description": description,
			"is_active": True,
			"created_at": self._now(),
		}
		self.products[product_id] = record
		self._emit(t, "savings_product_created", record)
		_log.info("Savings product created: %s tenant=%s", product_code, t)
		return deepcopy(record)

	async def update_product(
		self,
		product_id: str,
		tenant_id: str | None = None,
		interest_rate_pa: float | None = None,
		min_balance: float | None = None,
		max_balance: float | None = None,
		description: str | None = None,
		is_active: bool | None = None,
	) -> dict[str, Any]:
		"""Update a savings product configuration."""
		t = self._tenant(tenant_id)
		product = self._get_product(product_id, t)
		if interest_rate_pa is not None:
			product["interest_rate_pa"] = Decimal(str(interest_rate_pa))
		if min_balance is not None:
			product["min_balance"] = Decimal(str(min_balance))
		if max_balance is not None:
			product["max_balance"] = Decimal(str(max_balance))
		if description is not None:
			product["description"] = description
		if is_active is not None:
			product["is_active"] = is_active
		product["updated_at"] = self._now()
		self._emit(t, "savings_product_updated", product)
		return deepcopy(product)

	async def list_products(self, tenant_id: str | None = None, product_type: str | None = None, active_only: bool = True) -> list[dict[str, Any]]:
		"""List available savings products."""
		t = self._tenant(tenant_id)
		items = [deepcopy(p) for p in self.products.values() if p["tenant_id"] == t]
		if active_only:
			items = [p for p in items if p.get("is_active")]
		if product_type:
			items = [p for p in items if p.get("product_type") == product_type]
		return items

	async def get_product(self, product_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		t = self._tenant(tenant_id)
		return deepcopy(self._get_product(product_id, t))

	async def delete_product(self, product_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Deactivate a savings product (cannot delete if accounts exist)."""
		t = self._tenant(tenant_id)
		product = self._get_product(product_id, t)
		active_accounts = [a for a in self.accounts.values() if a["tenant_id"] == t and a["product_id"] == product_id and a["status"] == "active"]
		if active_accounts:
			raise ValueError(f"product_has_active_accounts: {len(active_accounts)}")
		product["is_active"] = False
		product["deactivated_at"] = self._now()
		self._emit(t, "savings_product_deactivated", product)
		return deepcopy(product)

	# ── Savings Accounts ──────────────────────────────────────────────────────

	async def open_account(
		self,
		member_id: str,
		product_id: str,
		tenant_id: str | None = None,
		opening_balance: float = 0.0,
		currency: str = "KES",
		account_name: str | None = None,
		maturity_date: str | None = None,
	) -> dict[str, Any]:
		"""Open a new savings account for a member."""
		t = self._tenant(tenant_id)
		product = self._get_product(product_id, t)
		if not product.get("is_active"):
			raise ValueError("product_not_active")
		opening = Decimal(str(opening_balance))
		if opening < product["min_opening_balance"]:
			raise ValueError(f"opening_balance_below_minimum: {product['min_opening_balance']}")
		account_number = self._next_account_number(t)
		acc_id = self._record_id("dep")
		record: dict[str, Any] = {
			"id": acc_id,
			"type": "sacco_savings_account",
			"tenant_id": t,
			"account_number": account_number,
			"member_id": member_id,
			"product_id": product_id,
			"product_code": product.get("product_code"),
			"product_name": product.get("product_name"),
			"account_name": account_name or f"{product.get('product_name')} Account",
			"balance": opening,
			"available_balance": opening,
			"accrued_interest": Decimal("0"),
			"currency": currency,
			"maturity_date": maturity_date,
			"status": "active",
			"created_at": self._now(),
			"updated_at": self._now(),
		}
		self.accounts[acc_id] = record
		if opening > 0:
			# record opening deposit transaction
			txn: dict[str, Any] = {
				"id": self._record_id("txn"),
				"type": "sacco_deposit",
				"tenant_id": t,
				"account_id": acc_id,
				"member_id": member_id,
				"amount": opening,
				"balance_after": opening,
				"narration": "Opening deposit",
				"payment_method": "cash",
				"status": "completed",
				"created_at": self._now(),
			}
			self.transactions[txn["id"]] = txn
		self._emit(t, "savings_account_opened", record)
		_log.info("Savings account opened: %s member=%s", account_number, member_id)
		return deepcopy(record)

	async def get_account(self, account_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		t = self._tenant(tenant_id)
		return deepcopy(self._get_account(account_id, t))

	async def update_account(self, account_id: str, tenant_id: str | None = None, account_name: str | None = None, status: str | None = None) -> dict[str, Any]:
		t = self._tenant(tenant_id)
		acc = self._get_account(account_id, t)
		if account_name is not None:
			acc["account_name"] = account_name
		if status is not None:
			if status not in ACCOUNT_STATUSES:
				raise ValueError(f"invalid_status: {status}")
			acc["status"] = status
		acc["updated_at"] = self._now()
		self._emit(t, "savings_account_updated", acc)
		return deepcopy(acc)

	async def close_account(self, account_id: str, closed_by: str, reason: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Close a savings account (balance must be zero or will be returned)."""
		t = self._tenant(tenant_id)
		acc = self._get_account(account_id, t)
		if acc["status"] == "closed":
			raise ValueError("account_already_closed")
		acc["status"] = "closed"
		acc["closed_by"] = closed_by
		acc["closure_reason"] = reason
		acc["closing_balance"] = acc["balance"]
		acc["balance"] = Decimal("0")
		acc["available_balance"] = Decimal("0")
		acc["closed_at"] = self._now()
		acc["updated_at"] = self._now()
		self._emit(t, "savings_account_closed", acc)
		return deepcopy(acc)

	async def list_accounts(
		self,
		tenant_id: str | None = None,
		member_id: str | None = None,
		product_id: str | None = None,
		status: str | None = None,
	) -> list[dict[str, Any]]:
		"""List savings accounts with optional filters."""
		t = self._tenant(tenant_id)
		items = [deepcopy(a) for a in self.accounts.values() if a["tenant_id"] == t]
		if member_id:
			items = [a for a in items if a["member_id"] == member_id]
		if product_id:
			items = [a for a in items if a["product_id"] == product_id]
		if status:
			items = [a for a in items if a["status"] == status]
		return items

	# ── Deposits ──────────────────────────────────────────────────────────────

	async def deposit(
		self,
		account_id: str,
		amount: float,
		payment_reference: str,
		recorded_by: str,
		tenant_id: str | None = None,
		payment_method: str = "cash",
		narration: str | None = None,
	) -> dict[str, Any]:
		"""Record a deposit into a savings account."""
		t = self._tenant(tenant_id)
		acc = self._get_account(account_id, t)
		if acc["status"] not in {"active"}:
			raise ValueError(f"cannot_deposit_to_account_in_status: {acc['status']}")
		if amount <= 0:
			raise ValueError("amount_must_be_positive")
		if payment_method not in PAYMENT_METHODS:
			raise ValueError(f"invalid_payment_method: {payment_method}")
		product = self._get_product(acc["product_id"], t)
		deposit_amount = Decimal(str(amount))
		new_balance = acc["balance"] + deposit_amount
		if product.get("max_balance") and new_balance > product["max_balance"]:
			raise ValueError(f"deposit_would_exceed_max_balance: {product['max_balance']}")
		txn_id = self._record_id("txn")
		txn: dict[str, Any] = {
			"id": txn_id,
			"type": "sacco_deposit",
			"tenant_id": t,
			"account_id": account_id,
			"account_number": acc.get("account_number"),
			"member_id": acc.get("member_id"),
			"amount": deposit_amount,
			"balance_before": acc["balance"],
			"balance_after": new_balance,
			"payment_reference": payment_reference,
			"payment_method": payment_method,
			"narration": narration or "Deposit",
			"recorded_by": recorded_by,
			"status": "completed",
			"created_at": self._now(),
		}
		self.transactions[txn_id] = txn
		acc["balance"] = new_balance
		acc["available_balance"] = new_balance
		acc["updated_at"] = self._now()
		self._emit(t, "deposit_made", txn)
		return deepcopy(txn)

	# ── Withdrawals ───────────────────────────────────────────────────────────

	async def withdraw(
		self,
		account_id: str,
		amount: float,
		approved_by: str,
		tenant_id: str | None = None,
		payment_method: str = "cash",
		narration: str | None = None,
		payment_reference: str | None = None,
	) -> dict[str, Any]:
		"""Process a withdrawal from a savings account."""
		t = self._tenant(tenant_id)
		acc = self._get_account(account_id, t)
		if acc["status"] not in {"active"}:
			raise ValueError(f"cannot_withdraw_from_account_in_status: {acc['status']}")
		product = self._get_product(acc["product_id"], t)
		withdrawal_amount = Decimal(str(amount))
		if withdrawal_amount <= 0:
			raise ValueError("amount_must_be_positive")
		balance_after = acc["balance"] - withdrawal_amount
		if balance_after < product.get("min_balance", Decimal("0")) and not product.get("allow_overdraft"):
			raise ValueError(f"withdrawal_would_breach_minimum_balance: {product.get('min_balance')}")
		txn_id = self._record_id("wtx")
		txn: dict[str, Any] = {
			"id": txn_id,
			"type": "sacco_withdrawal",
			"tenant_id": t,
			"account_id": account_id,
			"account_number": acc.get("account_number"),
			"member_id": acc.get("member_id"),
			"amount": withdrawal_amount,
			"balance_before": acc["balance"],
			"balance_after": balance_after,
			"payment_method": payment_method,
			"payment_reference": payment_reference,
			"narration": narration or "Withdrawal",
			"approved_by": approved_by,
			"status": "completed",
			"created_at": self._now(),
		}
		self.transactions[txn_id] = txn
		acc["balance"] = balance_after
		acc["available_balance"] = balance_after
		acc["updated_at"] = self._now()
		# Check for minimum balance breach
		if balance_after < product.get("min_balance", Decimal("0")):
			breach_id = self._record_id("brc")
			breach: dict[str, Any] = {
				"id": breach_id,
				"type": "sacco_min_balance_breach",
				"tenant_id": t,
				"account_id": account_id,
				"balance": balance_after,
				"min_balance": product.get("min_balance"),
				"created_at": self._now(),
			}
			self.minimum_balance_breaches[breach_id] = breach
		self._emit(t, "withdrawal_processed", txn)
		return deepcopy(txn)

	async def list_transactions(
		self,
		tenant_id: str | None = None,
		account_id: str | None = None,
		member_id: str | None = None,
		txn_type: str | None = None,
		from_date: str | None = None,
		to_date: str | None = None,
	) -> list[dict[str, Any]]:
		"""List transactions with optional filters."""
		t = self._tenant(tenant_id)
		items = [deepcopy(x) for x in self.transactions.values() if x["tenant_id"] == t]
		if account_id:
			items = [x for x in items if x["account_id"] == account_id]
		if member_id:
			items = [x for x in items if x.get("member_id") == member_id]
		if txn_type:
			items = [x for x in items if x["type"] == txn_type]
		if from_date:
			items = [x for x in items if x["created_at"][:10] >= from_date]
		if to_date:
			items = [x for x in items if x["created_at"][:10] <= to_date]
		return items

	# ── Interest Accrual ──────────────────────────────────────────────────────

	async def accrue_interest(
		self,
		period_start: str,
		period_end: str,
		posting_date: str,
		run_by: str,
		tenant_id: str | None = None,
		account_ids: list[str] | None = None,
	) -> dict[str, Any]:
		"""Run interest accrual and post to accounts."""
		t = self._tenant(tenant_id)
		accounts_to_process = [
			a for a in self.accounts.values()
			if a["tenant_id"] == t and a["status"] == "active"
			and (account_ids is None or a["id"] in account_ids)
		]
		posted_count = 0
		total_interest = Decimal("0")
		postings: list[dict[str, Any]] = []
		# Compute days in period
		try:
			d1 = date.fromisoformat(period_start)
			d2 = date.fromisoformat(period_end)
			days = (d2 - d1).days
		except Exception:
			days = 30

		for acc in accounts_to_process:
			product = self.products.get(acc["product_id"])
			if not product:
				continue
			rate = product.get("interest_rate_pa", Decimal("0"))
			daily_rate = rate / Decimal("36500")
			interest = (acc["balance"] * daily_rate * Decimal(str(days))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
			if interest <= 0:
				continue
			withholding_tax = Decimal("0") if product.get("tax_exempt") else (interest * Decimal("0.15")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
			net_interest = interest - withholding_tax
			posting_id = self._record_id("int")
			posting: dict[str, Any] = {
				"id": posting_id,
				"type": "sacco_interest_posting",
				"tenant_id": t,
				"account_id": acc["id"],
				"account_number": acc.get("account_number"),
				"member_id": acc.get("member_id"),
				"period_start": period_start,
				"period_end": period_end,
				"posting_date": posting_date,
				"days": days,
				"rate_pa": str(rate),
				"gross_interest": str(interest),
				"withholding_tax": str(withholding_tax),
				"net_interest": str(net_interest),
				"balance_after": str(acc["balance"] + net_interest),
				"run_by": run_by,
				"status": "posted",
				"created_at": self._now(),
			}
			self.interest_postings[posting_id] = posting
			acc["balance"] += net_interest
			acc["available_balance"] = acc["balance"]
			acc["accrued_interest"] = acc.get("accrued_interest", Decimal("0")) + net_interest
			acc["updated_at"] = self._now()
			total_interest += net_interest
			posted_count += 1
			postings.append(deepcopy(posting))
		run_record: dict[str, Any] = {
			"type": "sacco_interest_run",
			"tenant_id": t,
			"period_start": period_start,
			"period_end": period_end,
			"posting_date": posting_date,
			"accounts_processed": posted_count,
			"total_interest_posted": str(total_interest),
			"run_by": run_by,
			"status": "completed",
			"created_at": self._now(),
		}
		self._emit(t, "interest_accrued", run_record)
		return {**run_record, "postings": postings}

	async def list_interest_postings(self, account_id: str | None = None, tenant_id: str | None = None) -> list[dict[str, Any]]:
		"""List interest postings."""
		t = self._tenant(tenant_id)
		items = [deepcopy(p) for p in self.interest_postings.values() if p["tenant_id"] == t]
		if account_id:
			items = [p for p in items if p["account_id"] == account_id]
		return items

	# ── Balance & Statement ───────────────────────────────────────────────────

	async def get_account_statement(
		self,
		account_id: str,
		from_date: str | None = None,
		to_date: str | None = None,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Generate account statement for a period."""
		t = self._tenant(tenant_id)
		acc = self._get_account(account_id, t)
		txns = [
			x for x in self.transactions.values()
			if x["tenant_id"] == t and x["account_id"] == account_id
			and (from_date is None or x["created_at"][:10] >= from_date)
			and (to_date is None or x["created_at"][:10] <= to_date)
		]
		interest_txns = [
			p for p in self.interest_postings.values()
			if p["tenant_id"] == t and p["account_id"] == account_id
			and (from_date is None or p["created_at"][:10] >= from_date)
			and (to_date is None or p["created_at"][:10] <= to_date)
		]
		return {
			"account_id": account_id,
			"account_number": acc.get("account_number"),
			"member_id": acc.get("member_id"),
			"current_balance": str(acc["balance"]),
			"available_balance": str(acc.get("available_balance", acc["balance"])),
			"currency": acc.get("currency", "KES"),
			"from_date": from_date,
			"to_date": to_date,
			"transactions": txns,
			"interest_postings": interest_txns,
			"total_deposits": str(sum(x["amount"] for x in txns if x["type"] == "sacco_deposit")),
			"total_withdrawals": str(sum(x["amount"] for x in txns if x["type"] == "sacco_withdrawal")),
			"generated_at": self._now(),
		}

	async def check_minimum_balance(self, account_id: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Check if an account meets its product's minimum balance requirement."""
		t = self._tenant(tenant_id)
		acc = self._get_account(account_id, t)
		product = self._get_product(acc["product_id"], t)
		min_bal = product.get("min_balance", Decimal("0"))
		current = acc["balance"]
		return {
			"account_id": account_id,
			"current_balance": str(current),
			"minimum_balance": str(min_bal),
			"compliant": current >= min_bal,
			"shortfall": str(max(Decimal("0"), min_bal - current)),
			"checked_at": self._now(),
		}

	async def list_minimum_balance_breaches(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		t = self._tenant(tenant_id)
		return [deepcopy(b) for b in self.minimum_balance_breaches.values() if b["tenant_id"] == t]

	# ── Dormancy Management ───────────────────────────────────────────────────

	async def mark_dormant(self, account_id: str, reason: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Flag an account as dormant due to inactivity."""
		t = self._tenant(tenant_id)
		acc = self._get_account(account_id, t)
		if acc["status"] != "active":
			raise ValueError(f"cannot_mark_dormant_from_status: {acc['status']}")
		acc["status"] = "dormant"
		acc["dormancy_reason"] = reason
		acc["marked_dormant_at"] = self._now()
		acc["updated_at"] = self._now()
		self._emit(t, "account_marked_dormant", acc)
		return deepcopy(acc)

	async def reactivate_account(self, account_id: str, reactivated_by: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Reactivate a dormant or frozen account."""
		t = self._tenant(tenant_id)
		acc = self._get_account(account_id, t)
		if acc["status"] not in {"dormant", "frozen"}:
			raise ValueError(f"cannot_reactivate_from_status: {acc['status']}")
		acc["status"] = "active"
		acc["reactivated_by"] = reactivated_by
		acc["reactivated_at"] = self._now()
		acc["updated_at"] = self._now()
		self._emit(t, "account_reactivated", acc)
		return deepcopy(acc)

	# ── Portfolio Summary ─────────────────────────────────────────────────────

	async def portfolio_summary(self, tenant_id: str | None = None) -> dict[str, Any]:
		"""Aggregate savings portfolio statistics."""
		t = self._tenant(tenant_id)
		accounts = [a for a in self.accounts.values() if a["tenant_id"] == t]
		by_status: dict[str, int] = {}
		by_product: dict[str, Any] = {}
		total_balance = Decimal("0")
		for acc in accounts:
			by_status[acc.get("status", "unknown")] = by_status.get(acc.get("status", "unknown"), 0) + 1
			pid = acc.get("product_id", "unknown")
			if pid not in by_product:
				by_product[pid] = {"count": 0, "balance": Decimal("0")}
			by_product[pid]["count"] += 1
			by_product[pid]["balance"] += acc.get("balance", Decimal("0"))
			total_balance += acc.get("balance", Decimal("0"))
		return {
			"tenant_id": t,
			"total_accounts": len(accounts),
			"total_balance": str(total_balance),
			"by_status": by_status,
			"by_product": {k: {"count": v["count"], "balance": str(v["balance"])} for k, v in by_product.items()},
			"min_balance_breaches": len(self.minimum_balance_breaches),
			"generated_at": self._now(),
		}

	async def export_accounts(self, tenant_id: str | None = None, fmt: str = "json") -> dict[str, Any]:
		t = self._tenant(tenant_id)
		assert fmt in {"json", "csv", "excel"}, "fmt must be json|csv|excel"
		count = sum(1 for a in self.accounts.values() if a["tenant_id"] == t)
		return {
			"tenant_id": t,
			"format": fmt,
			"record_count": count,
			"export_reference": f"dep-accounts-{t}-{self._now()[:10]}.{fmt}",
			"generated_at": self._now(),
		}

	async def bulk_deposit(self, tenant_id: str | None = None, deposits: list[dict[str, Any]] | None = None) -> dict[str, Any]:
		"""Process multiple deposits in one call."""
		t = self._tenant(tenant_id)
		results, errors = [], []
		for dep in (deposits or []):
			try:
				rec = await self.deposit(
					account_id=dep["account_id"],
					amount=dep["amount"],
					payment_reference=dep["payment_reference"],
					recorded_by=dep["recorded_by"],
					tenant_id=t,
					payment_method=dep.get("payment_method", "cash"),
					narration=dep.get("narration"),
				)
				results.append(rec)
			except Exception as exc:
				_log.error("bulk_deposit error: %s", exc)
				errors.append({"input": dep, "error": str(exc)})
		return {"processed": len(results), "failed": len(errors), "results": results, "errors": errors}

	async def freeze_account(self, account_id: str, reason: str, frozen_by: str, tenant_id: str | None = None) -> dict[str, Any]:
		"""Freeze an account (regulatory or internal hold)."""
		t = self._tenant(tenant_id)
		acc = self._get_account(account_id, t)
		if acc["status"] == "frozen":
			raise ValueError("account_already_frozen")
		acc["status"] = "frozen"
		acc["freeze_reason"] = reason
		acc["frozen_by"] = frozen_by
		acc["frozen_at"] = self._now()
		acc["updated_at"] = self._now()
		self._emit(t, "account_frozen", acc)
		return deepcopy(acc)


# Alias
DepositsService = SaccoDepositsService
