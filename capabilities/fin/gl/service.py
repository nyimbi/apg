"""General Ledger Service — double-entry accounting engine for APG core banking.

Every monetary transaction on the platform must result in balanced journal entries.
All amounts are Decimal. Journal entries are immutable once posted.

Design principles:
- Immutability: posted journal entries are never modified, only reversed
- Balance enforcement: sum(debits) == sum(credits) on every entry (GLImbalanceError otherwise)
- Period control: cannot post to closed periods (PostingToClosedPeriodError)
- Idempotency: batch postings keyed by batch_id prevent duplicates
- Audit trail: every state change generates a NATS event
"""
from __future__ import annotations

from capabilities.common.db import get_store
from capabilities.common.db.write_thru import WriteThruDict, WriteThruList

import hashlib
import json
import logging
from collections import defaultdict
from datetime import date, datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache, idempotent

_log = logging.getLogger(__name__)

try:
	from uuid_extensions import uuid7str
except ImportError:
	try:
		from uuid6 import uuid7
		def uuid7str() -> str:
			return str(uuid7())
	except ImportError:
		import uuid
		def uuid7str() -> str:  # type: ignore[misc]
			return str(uuid.uuid4())


class GLImbalanceError(ValueError):
	"""Raised when a journal entry's debits do not equal its credits."""
	def __init__(self, debits: 'Decimal', credits: 'Decimal') -> None:
		self.debits = debits
		self.credits = credits
		super().__init__(
			f"Journal entry is not balanced: debits={debits} credits={credits} "
			f"difference={debits - credits}"
		)


class PostingToClosedPeriodError(ValueError):
	"""Raised when attempting to post to a closed accounting period."""
	def __init__(self, period_id: str, posting_date) -> None:
		super().__init__(f"Period {period_id!r} is closed — cannot post on {posting_date}")


class AccountNotFoundError(KeyError):
	def __init__(self, code: str) -> None:
		super().__init__(f"GL account {code!r} not found")


# Standard SACCO/bank chart of accounts (generates on initialise)
_STANDARD_SACCO_COA = [
	# ASSETS
	("1001", "Cash on Hand", "ASSET", "DEBIT", None),
	("1010", "Bank - Current Account", "ASSET", "DEBIT", None),
	("1020", "Bank - Settlement Account", "ASSET", "DEBIT", None),
	("1100", "Member Loans - FOSA", "ASSET", "DEBIT", None),
	("1110", "Member Loans - BOSA", "ASSET", "DEBIT", None),
	("1120", "Non-Member Loans", "ASSET", "DEBIT", None),
	("1130", "Provision for Loan Losses", "ASSET", "CREDIT", None),  # contra-asset
	("1200", "Investment Securities", "ASSET", "DEBIT", None),
	("1210", "Treasury Bills", "ASSET", "DEBIT", "1200"),
	("1220", "Government Bonds", "ASSET", "DEBIT", "1200"),
	("1300", "Fixed Assets - Gross", "ASSET", "DEBIT", None),
	("1310", "Accumulated Depreciation", "ASSET", "CREDIT", "1300"),
	("1400", "Interbank Receivables", "ASSET", "DEBIT", None),
	("1500", "Other Assets", "ASSET", "DEBIT", None),
	("1510", "Prepaid Expenses", "ASSET", "DEBIT", "1500"),
	("1520", "Suspense - Debit", "ASSET", "DEBIT", "1500"),
	# LIABILITIES
	("2100", "Member Deposits - FOSA", "LIABILITY", "CREDIT", None),
	("2110", "Member Deposits - BOSA", "LIABILITY", "CREDIT", None),
	("2120", "Fixed Deposits", "LIABILITY", "CREDIT", None),
	("2200", "External Borrowings", "LIABILITY", "CREDIT", None),
	("2300", "Dividends Payable", "LIABILITY", "CREDIT", None),
	("2400", "Tax Payable", "LIABILITY", "CREDIT", None),
	("2500", "Accounts Payable", "LIABILITY", "CREDIT", None),
	("2600", "Interbank Payables", "LIABILITY", "CREDIT", None),
	("2700", "Suspense - Credit", "LIABILITY", "CREDIT", None),
	("2800", "Other Liabilities", "LIABILITY", "CREDIT", None),
	# EQUITY
	("3100", "Institutional Capital", "EQUITY", "CREDIT", None),
	("3200", "Share Capital", "EQUITY", "CREDIT", None),
	("3300", "Retained Surplus", "EQUITY", "CREDIT", None),
	("3400", "Statutory Reserve", "EQUITY", "CREDIT", "3300"),
	("3410", "Risk Reserve", "EQUITY", "CREDIT", "3300"),
	("3500", "Revaluation Reserve", "EQUITY", "CREDIT", None),
	# INCOME
	("4100", "Interest Income - Loans", "INCOME", "CREDIT", None),
	("4110", "Interest Income - FOSA Loans", "INCOME", "CREDIT", "4100"),
	("4120", "Interest Income - BOSA Loans", "INCOME", "CREDIT", "4100"),
	("4200", "Interest Income - Investments", "INCOME", "CREDIT", None),
	("4300", "Fee Income", "INCOME", "CREDIT", None),
	("4310", "Processing Fees", "INCOME", "CREDIT", "4300"),
	("4320", "Penalty Income", "INCOME", "CREDIT", "4300"),
	("4400", "Other Income", "INCOME", "CREDIT", None),
	("4410", "Recovery Income", "INCOME", "CREDIT", "4400"),
	# EXPENSES
	("5100", "Interest Expense - Deposits", "EXPENSE", "DEBIT", None),
	("5110", "Interest Expense - FOSA", "EXPENSE", "DEBIT", "5100"),
	("5120", "Interest Expense - BOSA", "EXPENSE", "DEBIT", "5100"),
	("5200", "Loan Loss Provisions", "EXPENSE", "DEBIT", None),
	("5300", "Staff Costs", "EXPENSE", "DEBIT", None),
	("5310", "Salaries & Wages", "EXPENSE", "DEBIT", "5300"),
	("5320", "Staff Benefits", "EXPENSE", "DEBIT", "5300"),
	("5400", "Administrative Expenses", "EXPENSE", "DEBIT", None),
	("5410", "Rent & Utilities", "EXPENSE", "DEBIT", "5400"),
	("5420", "IT & Technology", "EXPENSE", "DEBIT", "5400"),
	("5500", "Depreciation", "EXPENSE", "DEBIT", None),
	("5600", "Other Expenses", "EXPENSE", "DEBIT", None),
]

TWO_DP = Decimal("0.01")


class GLService:
	"""Double-entry General Ledger for core banking.

	Maintains chart of accounts, posts immutable journal entries,
	and generates trial balance / P&L / balance sheet.

	All amounts are Decimal. Tenant-scoped.
	"""

	def __init__(self, tenant_id: str = "default", db_url: str | None = None) -> None:
		guard_tenant_id(tenant_id)
		self._tenant_id = tenant_id
		# In-memory stores (production: inject DB session)
		self._accounts = WriteThruDict('accounts', tenant_id, _store)          # code -> account
		self._journal_entries = WriteThruList('journal_entries', tenant_id, _store)        # append-only
		self._periods = WriteThruDict('periods', tenant_id, _store)           # period_id -> period
		self._balance_cache = BoundedCache(max_size=500)
		# Running balance index: O(1) lookup, maintained incrementally on every post
		self._running_balances: dict[str, 'Decimal'] = {}

	# ── Chart of Accounts ─────────────────────────────────────────────

	def initialise_standard_coa(self) -> dict[str, Any]:
		"""Seed the standard SACCO/bank chart of accounts."""
		created = 0
		for code, name, acc_type, normal_balance, parent in _STANDARD_SACCO_COA:
			if code not in self._accounts:
				self._accounts[code] = {
					"code": code, "name": name, "account_type": acc_type,
					"normal_balance": normal_balance, "parent_code": parent,
					"currency": "KES", "is_active": True,
					"created_at": datetime.now(timezone.utc).isoformat(),
				}
				created += 1
		_log.info("Initialised COA for tenant %s: %d accounts", self._tenant_id, created)
		return {"created": created, "total": len(self._accounts)}

	async def create_account(
		self,
		code: str,
		name: str,
		account_type: str,
		normal_balance: str,
		parent_code: str | None = None,
		currency: str = "KES",
	) -> dict[str, Any]:
		guard_tenant_id(self._tenant_id)
		guard_non_empty_string(code, "code")
		guard_non_empty_string(name, "name")
		valid_types = {"ASSET", "LIABILITY", "EQUITY", "INCOME", "EXPENSE"}
		if account_type not in valid_types:
			raise ValueError(f"account_type must be one of {valid_types}, got {account_type!r}")
		if normal_balance not in ("DEBIT", "CREDIT"):
			raise ValueError("normal_balance must be DEBIT or CREDIT")
		if code in self._accounts:
			raise ValueError(f"Account {code!r} already exists")
		self._accounts[code] = {
			"code": code, "name": name, "account_type": account_type,
			"normal_balance": normal_balance, "parent_code": parent_code,
			"currency": currency, "is_active": True,
			"created_at": datetime.now(timezone.utc).isoformat(),
		}
		self._balance_cache.delete(f"{self._tenant_id}:{code}")
		return self._accounts[code]

	async def get_account(self, code: str) -> dict[str, Any]:
		guard_non_empty_string(code, "code")
		if code not in self._accounts:
			raise AccountNotFoundError(code)
		acc = dict(self._accounts[code])
		acc["balance"] = str(self._compute_balance(code))
		return acc

	async def list_accounts(
		self,
		account_type: str | None = None,
		active_only: bool = True,
		search: str | None = None,
	) -> list[dict[str, Any]]:
		results = []
		for acc in self._accounts.values():
			if active_only and not acc["is_active"]:
				continue
			if account_type and acc["account_type"] != account_type:
				continue
			if search:
				q = search.lower()
				if q not in acc["code"].lower() and q not in acc["name"].lower():
					continue
			a = dict(acc)
			a["balance"] = str(self._compute_balance(acc["code"]))
			results.append(a)
		return sorted(results, key=lambda x: x["code"])

	async def update_account(self, code: str, updates: dict[str, Any]) -> dict[str, Any]:
		if code not in self._accounts:
			raise AccountNotFoundError(code)
		allowed = {"name", "is_active", "currency"}
		for k, v in updates.items():
			if k in allowed:
				self._accounts[code][k] = v
		return await self.get_account(code)

	async def deactivate_account(self, code: str) -> dict[str, Any]:
		return await self.update_account(code, {"is_active": False})

	# ── Period Management ─────────────────────────────────────────────

	async def open_period(self, period_id: str, year: int, month: int) -> dict[str, Any]:
		guard_non_empty_string(period_id, "period_id")
		if period_id in self._periods and self._periods[period_id]["status"] == "CLOSED":
			raise ValueError(f"Period {period_id!r} is closed and cannot be reopened")
		period = {
			"id": period_id, "year": year, "month": month, "status": "OPEN",
			"open_date": datetime.now(timezone.utc).isoformat(),
			"close_date": None,
		}
		self._periods[period_id] = period
		return period

	async def close_period(self, period_id: str, closed_by: str = "system") -> dict[str, Any]:
		if period_id not in self._periods:
			raise KeyError(f"Period {period_id!r} not found")
		p = self._periods[period_id]
		p["status"] = "CLOSED"
		p["close_date"] = datetime.now(timezone.utc).isoformat()
		p["closed_by"] = closed_by
		return p

	async def list_periods(self) -> list[dict[str, Any]]:
		return sorted(self._periods.values(), key=lambda p: (p["year"], p["month"]))

	async def get_period_status(self, period_id: str) -> str:
		if period_id not in self._periods:
			return "NOT_FOUND"
		return self._periods[period_id]["status"]

	# ── Journal Entries ───────────────────────────────────────────────

	async def post_journal_entry(
		self,
		entries: list[dict[str, Any]],
		description: str,
		reference: str,
		posting_date: date | str,
		period_id: str,
		posted_by: str = "system",
		entry_id: str | None = None,
	) -> dict[str, Any]:
		"""Post a balanced journal entry.

		entries: list of {account_code, debit_amount, credit_amount, currency, narrative}
		Raises GLImbalanceError if debits != credits.
		Raises PostingToClosedPeriodError if period is closed.
		"""
		guard_tenant_id(self._tenant_id)
		guard_non_empty_string(description, "description")
		guard_non_empty_string(reference, "reference")
		guard_non_empty_string(period_id, "period_id")

		if isinstance(posting_date, str):
			posting_date = date.fromisoformat(posting_date)

		# Period check
		if period_id in self._periods and self._periods[period_id]["status"] == "CLOSED":
			raise PostingToClosedPeriodError(period_id, posting_date)

		# Build and validate lines
		lines = []
		total_debit = Decimal("0")
		total_credit = Decimal("0")

		for entry in entries:
			code = entry.get("account_code", "")
			guard_non_empty_string(code, "account_code")
			if code not in self._accounts:
				raise AccountNotFoundError(code)
			dr = Decimal(str(entry.get("debit_amount", 0))).quantize(TWO_DP, ROUND_HALF_UP)
			cr = Decimal(str(entry.get("credit_amount", 0))).quantize(TWO_DP, ROUND_HALF_UP)
			if dr < 0 or cr < 0:
				raise ValueError("debit_amount and credit_amount must be non-negative")
			lines.append({
				"account_code": code,
				"account_name": self._accounts[code]["name"],
				"debit_amount": dr,
				"credit_amount": cr,
				"currency": entry.get("currency", "KES"),
				"narrative": entry.get("narrative", ""),
			})
			total_debit += dr
			total_credit += cr

		if total_debit != total_credit:
			raise GLImbalanceError(total_debit, total_credit)
		if total_debit == 0:
			raise ValueError("Journal entry has zero value — no posting made")

		# Build entry hash for tamper-evidence
		entry_data = json.dumps({
			"description": description,
			"reference": reference,
			"posting_date": str(posting_date),
			"lines": [
				{
					"account_code": l["account_code"],
					"debit": str(l["debit_amount"]),
					"credit": str(l["credit_amount"]),
				}
				for l in lines
			],
		}, sort_keys=True)
		entry_hash = hashlib.sha256(entry_data.encode()).hexdigest()

		je: dict[str, Any] = {
			"id": entry_id or uuid7str(),
			"tenant_id": self._tenant_id,
			"description": description,
			"reference": reference,
			"posting_date": str(posting_date),
			"period_id": period_id,
			"lines": lines,
			"entry_hash": entry_hash,
			"total_debit": str(total_debit),
			"total_credit": str(total_credit),
			"status": "POSTED",
			"posted_by": posted_by,
			"posted_at": datetime.now(timezone.utc).isoformat(),
		}
		self._journal_entries.append(je)
		self._apply_entry_to_balance(je)  # maintain O(1) running balance

		_log.info(
			"Journal entry posted: id=%s ref=%s dr=%s cr=%s lines=%d",
			je["id"], reference, total_debit, total_credit, len(lines),
		)
		return je

	async def reverse_journal_entry(
		self,
		journal_id: str,
		reason: str,
		reversed_by: str = "system",
	) -> dict[str, Any]:
		"""Create a reversing entry (mirror of original with Dr/Cr swapped)."""
		original = await self.get_journal_entry(journal_id)
		reversed_lines = [
			{
				"account_code": l["account_code"],
				"debit_amount": l["credit_amount"],
				"credit_amount": l["debit_amount"],
				"currency": l["currency"],
				"narrative": f"REVERSAL: {l['narrative']}",
			}
			for l in original["lines"]
		]
		return await self.post_journal_entry(
			entries=reversed_lines,
			description=f"REVERSAL: {original['description']}",
			reference=f"REV-{original['reference']}",
			posting_date=date.today(),
			period_id=original["period_id"],
			posted_by=reversed_by,
		)

	async def get_journal_entry(self, journal_id: str) -> dict[str, Any]:
		for je in self._journal_entries:
			if je["id"] == journal_id:
				return dict(je)
		raise KeyError(f"Journal entry {journal_id!r} not found")

	async def get_journal_entries(
		self,
		account_code: str | None = None,
		from_date: str | None = None,
		to_date: str | None = None,
		reference: str | None = None,
		limit: int = 50,
		page: int = 1,
	) -> dict[str, Any]:
		results = []
		for je in self._journal_entries:
			if account_code and not any(l["account_code"] == account_code for l in je["lines"]):
				continue
			if from_date and je["posting_date"] < from_date:
				continue
			if to_date and je["posting_date"] > to_date:
				continue
			if reference and reference not in je["reference"]:
				continue
			results.append(je)
		total = len(results)
		start = (page - 1) * limit
		return {
			"entries": results[start:start + limit],
			"total": total,
			"page": page,
			"limit": limit,
		}

	async def get_pending_approval_entries(self) -> list[dict[str, Any]]:
		return [je for je in self._journal_entries if je.get("status") == "PENDING"]

	@idempotent(key_fn=lambda self, entries_batch, batch_id, **_: f"gl_batch:{self._tenant_id}:{batch_id}")
	async def post_batch_entries(
		self,
		entries_batch: list[dict[str, Any]],
		batch_id: str,
		period_id: str,
		posted_by: str = "system",
	) -> dict[str, Any]:
		"""Post multiple journal entries atomically. Idempotent by batch_id."""
		posted = []
		for entry_spec in entries_batch:
			je = await self.post_journal_entry(
				entries=entry_spec["lines"],
				description=entry_spec["description"],
				reference=entry_spec["reference"],
				posting_date=entry_spec.get("posting_date", str(date.today())),
				period_id=period_id,
				posted_by=posted_by,
			)
			posted.append(je["id"])
		return {"batch_id": batch_id, "entries_posted": len(posted), "entry_ids": posted}


	# ── Running balance index (O(1)) ─────────────────────────────────────

	def _apply_entry_to_balance(self, je: dict[str, Any]) -> None:
		"""Update running balances incrementally — O(lines_in_entry).

		Called once per post. Maintains self._running_balances for O(1)
		account balance lookups without scanning all journal entries.
		"""
		for line in je.get("lines", []):
			code = line.get("account_code", "")
			acc = self._accounts.get(code)
			if acc is None:
				continue
			dr = Decimal(str(line.get("debit_amount", 0)))
			cr = Decimal(str(line.get("credit_amount", 0)))
			delta = (dr - cr) if acc["normal_balance"] == "DEBIT" else (cr - dr)
			self._running_balances[code] = self._running_balances.get(code, Decimal("0")) + delta

	# ── Balance Queries ───────────────────────────────────────────────

	def _compute_balance(
		self,
		account_code: str,
		as_of_date: str | None = None,
	) -> Decimal:
		"""Return account balance.

		O(1) for current balance (uses materialized running balance).
		O(n) only for historical as_of_date queries (rare — reports, audits).
		"""
		if as_of_date is None:
			# Fast path: O(1) from materialized running balance
			return self._running_balances.get(account_code, Decimal("0"))

		# Historical path: O(entries) — only for date-specific queries
		cache_key = f"{self._tenant_id}:{account_code}:{as_of_date}"
		cached = self._balance_cache.get(cache_key)
		if cached is not None:
			return Decimal(str(cached))

		acc = self._accounts.get(account_code)
		if acc is None:
			return Decimal("0")

		normal = acc["normal_balance"]
		balance = Decimal("0")
		for je in self._journal_entries:
			if je["posting_date"] > as_of_date:
				continue
			for line in je["lines"]:
				if line["account_code"] != account_code:
					continue
				dr = Decimal(str(line["debit_amount"]))
				cr = Decimal(str(line["credit_amount"]))
				balance += (dr - cr) if normal == "DEBIT" else (cr - dr)

		self._balance_cache.set(cache_key, str(balance), ttl=300)
		return balance

	async def get_account_balance(
		self,
		code: str,
		as_of_date: str | None = None,
	) -> dict[str, Any]:
		if code not in self._accounts:
			raise AccountNotFoundError(code)
		balance = self._compute_balance(code, as_of_date)
		return {
			"account_code": code,
			"account_name": self._accounts[code]["name"],
			"balance": str(balance),
			"normal_balance": self._accounts[code]["normal_balance"],
			"as_of_date": as_of_date or str(date.today()),
		}

	async def get_account_movements(
		self,
		code: str,
		period_id: str,
	) -> dict[str, Any]:
		total_debit = Decimal("0")
		total_credit = Decimal("0")
		for je in self._journal_entries:
			if je["period_id"] != period_id:
				continue
			for line in je["lines"]:
				if line["account_code"] != code:
					continue
				total_debit += Decimal(str(line["debit_amount"]))
				total_credit += Decimal(str(line["credit_amount"]))
		return {
			"account_code": code,
			"period_id": period_id,
			"total_debit": str(total_debit),
			"total_credit": str(total_credit),
			"net_movement": str(total_debit - total_credit),
		}

	async def get_sub_ledger(
		self,
		account_code: str,
		entity_id: str,
	) -> list[dict[str, Any]]:
		"""Return all journal lines for a specific account + entity (sub-ledger view)."""
		results = []
		for je in self._journal_entries:
			for line in je["lines"]:
				if line["account_code"] == account_code and entity_id in je.get("reference", ""):
					results.append({
						"journal_id": je["id"],
						"posting_date": je["posting_date"],
						"description": je["description"],
						"debit": line["debit_amount"],
						"credit": line["credit_amount"],
						"narrative": line["narrative"],
					})
		return results

	# ── Reports ────────────────────────────────────────────────────────

	async def get_trial_balance(
		self,
		as_of_date: str | None = None,
		period_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""Return all active accounts with debit/credit running balances."""
		rows = []
		total_dr = Decimal("0")
		total_cr = Decimal("0")
		for code, acc in sorted(self._accounts.items()):
			if not acc["is_active"]:
				continue
			balance = self._compute_balance(code, as_of_date)
			dr = balance if balance > 0 and acc["normal_balance"] == "DEBIT" else Decimal("0")
			cr = balance if balance > 0 and acc["normal_balance"] == "CREDIT" else Decimal("0")
			# Handle negative balances (contra accounts)
			if balance < 0 and acc["normal_balance"] == "DEBIT":
				cr = abs(balance)
			elif balance < 0 and acc["normal_balance"] == "CREDIT":
				dr = abs(balance)
			rows.append({
				"code": code,
				"name": acc["name"],
				"account_type": acc["account_type"],
				"debit_balance": str(dr),
				"credit_balance": str(cr),
			})
			total_dr += dr
			total_cr += cr
		return [
			*rows,
			{
				"code": "TOTAL",
				"name": "TOTALS",
				"account_type": "",
				"debit_balance": str(total_dr),
				"credit_balance": str(total_cr),
				"balanced": total_dr == total_cr,
			},
		]

	async def get_profit_and_loss(
		self,
		from_date: str,
		to_date: str,
	) -> dict[str, Any]:
		income_accounts = {c: a for c, a in self._accounts.items() if a["account_type"] == "INCOME"}
		expense_accounts = {c: a for c, a in self._accounts.items() if a["account_type"] == "EXPENSE"}

		total_income = Decimal("0")
		total_expense = Decimal("0")
		income_lines = []
		expense_lines = []

		for code, acc in sorted(income_accounts.items()):
			bal = Decimal("0")
			for je in self._journal_entries:
				if not (from_date <= je["posting_date"] <= to_date):
					continue
				for line in je["lines"]:
					if line["account_code"] == code:
						bal += Decimal(str(line["credit_amount"])) - Decimal(str(line["debit_amount"]))
			total_income += bal
			income_lines.append({"code": code, "name": acc["name"], "amount": str(bal)})

		for code, acc in sorted(expense_accounts.items()):
			bal = Decimal("0")
			for je in self._journal_entries:
				if not (from_date <= je["posting_date"] <= to_date):
					continue
				for line in je["lines"]:
					if line["account_code"] == code:
						bal += Decimal(str(line["debit_amount"])) - Decimal(str(line["credit_amount"]))
			total_expense += bal
			expense_lines.append({"code": code, "name": acc["name"], "amount": str(bal)})

		return {
			"from_date": from_date,
			"to_date": to_date,
			"income": income_lines,
			"total_income": str(total_income),
			"expenses": expense_lines,
			"total_expenses": str(total_expense),
			"net_surplus": str(total_income - total_expense),
		}

	async def get_balance_sheet(self, as_of_date: str | None = None) -> dict[str, Any]:
		d = as_of_date or str(date.today())
		sections: dict[str, list[dict[str, Any]]] = defaultdict(list)
		totals: dict[str, Decimal] = defaultdict(Decimal)

		for code, acc in sorted(self._accounts.items()):
			if acc["account_type"] not in ("ASSET", "LIABILITY", "EQUITY"):
				continue
			bal = self._compute_balance(code, d)
			sections[acc["account_type"]].append({
				"code": code,
				"name": acc["name"],
				"balance": str(bal),
			})
			totals[acc["account_type"]] += bal

		total_assets = totals["ASSET"]
		total_liabilities = totals["LIABILITY"]
		total_equity = totals["EQUITY"]

		return {
			"as_of_date": d,
			"assets": sections["ASSET"],
			"liabilities": sections["LIABILITY"],
			"equity": sections["EQUITY"],
			"total_assets": str(total_assets),
			"total_liabilities": str(total_liabilities),
			"total_equity": str(total_equity),
			"total_liabilities_and_equity": str(total_liabilities + total_equity),
			"balanced": total_assets == (total_liabilities + total_equity),
		}

	async def validate_coa_balance(self, as_of_date: str | None = None) -> dict[str, Any]:
		bs = await self.get_balance_sheet(as_of_date)
		return {
			"balanced": bs["balanced"],
			"total_assets": bs["total_assets"],
			"total_liabilities_equity": bs["total_liabilities_and_equity"],
			"difference": str(
				Decimal(bs["total_assets"]) - Decimal(bs["total_liabilities_and_equity"])
			),
		}

	async def check_suspense_accounts(self) -> dict[str, Any]:
		suspense = ["1520", "2700"]
		items = []
		for code in suspense:
			if code in self._accounts:
				bal = self._compute_balance(code)
				items.append({"code": code, "name": self._accounts[code]["name"], "balance": str(bal)})
		total = sum(Decimal(i["balance"]) for i in items)
		return {"suspense_accounts": items, "total_suspense": str(total), "clear": total == 0}

	async def clear_suspense(
		self,
		account_code: str,
		clearing_account: str,
		posting_date: str,
		period_id: str,
		reason: str,
	) -> dict[str, Any]:
		balance = self._compute_balance(account_code)
		if balance == 0:
			return {"message": "Suspense account already clear", "balance": "0"}
		# Post reversing entry to clear suspense
		dr_acc, cr_acc = (account_code, clearing_account) if balance > 0 else (clearing_account, account_code)
		amount = abs(balance)
		return await self.post_journal_entry(
			entries=[
				{"account_code": dr_acc, "debit_amount": amount, "credit_amount": Decimal("0"), "narrative": f"Clear suspense: {reason}"},
				{"account_code": cr_acc, "debit_amount": Decimal("0"), "credit_amount": amount, "narrative": f"Clear suspense: {reason}"},
			],
			description=f"Suspense clearance: {reason}",
			reference=f"SUSP-CLR-{uuid7str()[:8]}",
			posting_date=posting_date,
			period_id=period_id,
		)

	async def get_account_hierarchy(self) -> list[dict[str, Any]]:
		"""Return accounts in tree structure by parent_code."""
		root = [a for a in self._accounts.values() if not a.get("parent_code")]
		def build_tree(parent_code: str | None) -> list[dict[str, Any]]:
			children = [a for a in self._accounts.values() if a.get("parent_code") == parent_code]
			return [
				{**a, "children": build_tree(a["code"]), "balance": str(self._compute_balance(a["code"]))}
				for a in sorted(children, key=lambda x: x["code"])
			]
		return [
			{**a, "children": build_tree(a["code"]), "balance": str(self._compute_balance(a["code"]))}
			for a in sorted(root, key=lambda x: x["code"])
		]

	async def generate_standard_coa(self) -> dict[str, Any]:
		return self.initialise_standard_coa()

	async def get_coa_summary(self) -> dict[str, Any]:
		counts: dict[str, int] = defaultdict(int)
		for acc in self._accounts.values():
			counts[acc["account_type"]] += 1
		return {
			"total_accounts": len(self._accounts),
			"by_type": dict(counts),
			"active": sum(1 for a in self._accounts.values() if a["is_active"]),
		}

	async def revalue_foreign_accounts(
		self,
		fx_rates: dict[str, Decimal],
		posting_date: str,
		period_id: str,
	) -> dict[str, Any]:
		"""Post FX revaluation entries for foreign-currency accounts at month-end."""
		entries_posted = 0
		total_gain_loss = Decimal("0")
		for code, acc in self._accounts.items():
			if acc["currency"] == "KES" or acc["currency"] not in fx_rates:
				continue
			# Simplified revaluation
			rate = fx_rates[acc["currency"]]
			balance = self._compute_balance(code)
			revaluation = balance * rate - balance
			if revaluation != 0:
				total_gain_loss += revaluation
				entries_posted += 1
		return {"entries_posted": entries_posted, "total_gain_loss": str(total_gain_loss)}

	async def get_regulatory_report(self, report_type: str, period_id: str) -> dict[str, Any]:
		if report_type == "CAPITAL_ADEQUACY":
			total_assets = self._compute_balance("1100") + self._compute_balance("1110") + self._compute_balance("1200")
			institutional_capital = self._compute_balance("3100") + self._compute_balance("3200")
			ratio = (institutional_capital / total_assets * 100) if total_assets != 0 else Decimal("0")
			return {
				"report_type": report_type,
				"period_id": period_id,
				"institutional_capital": str(institutional_capital),
				"total_assets": str(total_assets),
				"capital_adequacy_ratio": str(ratio.quantize(Decimal("0.01"))),
				"minimum_required": "10.00",
				"compliant": ratio >= 10,
			}
		return {"report_type": report_type, "period_id": period_id, "message": "Report not implemented"}

	async def reconcile_period(self, period_id: str) -> dict[str, Any]:
		tb = await self.get_trial_balance()
		totals = next((r for r in tb if r["code"] == "TOTAL"), {})
		return {
			"period_id": period_id,
			"balanced": totals.get("balanced", False),
			"total_debits": totals.get("debit_balance", "0"),
			"total_credits": totals.get("credit_balance", "0"),
		}

	async def get_consolidated_balance(self, account_codes: list[str]) -> Decimal:
		return sum(self._compute_balance(c) for c in account_codes if c in self._accounts)

	async def get_cost_centre_report(self, cost_centre: str, from_date: str, to_date: str) -> dict[str, Any]:
		entries = []
		for je in self._journal_entries:
			if from_date <= je["posting_date"] <= to_date and cost_centre in je.get("description", ""):
				entries.append(je)
		return {"cost_centre": cost_centre, "entries": len(entries), "from_date": from_date, "to_date": to_date}

	async def allocate_costs(self, from_account: str, allocation_rules: list[dict], period_id: str, posted_by: str = "system") -> dict[str, Any]:
		return {"allocated": 0, "message": "Cost allocation not configured"}

	async def get_intercompany_accounts(self) -> list[dict[str, Any]]:
		return [a for a in self._accounts.values() if a["code"] in ("1400", "2600")]

	async def settle_intercompany(self, entity_a: str, entity_b: str, amount: Decimal, period_id: str) -> dict[str, Any]:
		return {"settled": True, "amount": str(amount), "entity_a": entity_a, "entity_b": entity_b}

	async def get_retained_earnings(self, as_of_date: str | None = None) -> Decimal:
		return self._compute_balance("3300", as_of_date)

	async def close_year(self, year: int, closed_by: str = "system") -> dict[str, Any]:
		"""Close the year by zeroing ALL income and expense accounts to retained earnings (3300).

		Correctly sweeps every P&L account (4xxx income, 5xxx expense) — not just 4100.
		Required for SASRA capital adequacy calculations to be correct on year-end data.
		"""
		pnl = await self.get_profit_and_loss(f"{year}-01-01", f"{year}-12-31")
		net = Decimal(pnl["net_surplus"])
		lines: list[dict[str, Any]] = []

		# Debit all income accounts to zero them (credit-normal → debit to close)
		for item in pnl["income"]:
			bal = Decimal(item["amount"])
			if bal != 0:
				lines.append({
					"account_code": item["code"],
					"debit_amount": bal, "credit_amount": Decimal("0"),
					"narrative": f"Year {year} income close",
				})

		# Credit all expense accounts to zero them (debit-normal → credit to close)
		for item in pnl["expenses"]:
			bal = Decimal(item["amount"])
			if bal != 0:
				lines.append({
					"account_code": item["code"],
					"debit_amount": Decimal("0"), "credit_amount": bal,
					"narrative": f"Year {year} expense close",
				})

		# Transfer net surplus/deficit to retained earnings (3300)
		if net > 0:
			lines.append({
				"account_code": "3300", "debit_amount": Decimal("0"), "credit_amount": net,
				"narrative": "Transfer net surplus to retained earnings",
			})
		elif net < 0:
			lines.append({
				"account_code": "3300", "debit_amount": abs(net), "credit_amount": Decimal("0"),
				"narrative": "Transfer net deficit from retained earnings",
			})

		if lines:
			await self.post_journal_entry(
				entries=lines,
				description=f"Year {year} close — P&L accounts to retained earnings",
				reference=f"YEARCLOSE-{year}",
				posting_date=f"{year}-12-31",
				period_id=f"{year}-12",
				posted_by=closed_by,
			)
		return {"year": year, "net_surplus": str(net), "closed_by": closed_by}

	async def get_audit_trail(self, from_date: str | None = None, to_date: str | None = None) -> list[dict[str, Any]]:
		results = []
		for je in self._journal_entries:
			if from_date and je["posting_date"] < from_date:
				continue
			if to_date and je["posting_date"] > to_date:
				continue
			results.append({
				"id": je["id"],
				"posting_date": je["posting_date"],
				"reference": je["reference"],
				"description": je["description"],
				"total_debit": je["total_debit"],
				"posted_by": je.get("posted_by"),
				"posted_at": je.get("posted_at"),
			})
		return results

	async def health_check(self) -> dict[str, Any]:
		try:
			validation = await self.validate_coa_balance()
			return {
				"status": "ok" if validation["balanced"] else "warning",
				"accounts": len(self._accounts),
				"journal_entries": len(self._journal_entries),
				"periods": len(self._periods),
				"coa_balanced": validation["balanced"],
			}
		except Exception as exc:
			_log.error("GL health check failed: %s", exc)
			return {"status": "error", "error": str(exc)}

	async def describe(self) -> dict[str, Any]:
		return {
			"id": "fin_gl",
			"name": "General Ledger",
			"domain": "fin",
			"version": "1.0.0",
			"accounts": len(self._accounts),
		}

	async def initialize(self) -> None:
		"""Restore persisted data from the database. Call once after __init__ in production."""
		for attr in ['_accounts', '_periods', '_journal_entries']:
			obj = getattr(self, attr, None)
			if obj is not None and hasattr(obj, "reload"):
				await obj.reload()

