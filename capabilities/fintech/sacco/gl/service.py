"""SACCO General Ledger — full async service.

Implements the ICPAK chart of accounts standard adapted for SACCOs per SASRA
requirements. All monetary values use Python Decimal for accuracy.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from copy import deepcopy
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from .models import (
	STANDARD_COA,
	AccountCategory,
	AccountingPeriod,
	BalanceSheet,
	GLAccount,
	GLSummary,
	IncomeStatement,
	JournalEntry,
	JournalLine,
	NormalBalance,
	ReconciliationResult,
	TrialBalanceRow,
	uuid7str,
)

_log = logging.getLogger(__name__)

CAPABILITY_ID = "fintech_sacco_gl"

# Account codes as strings for quick reference
_CASH = "1001"
_BANK = "1010"
_LOANS_FOSA = "1100"
_LOANS_BOSA = "1110"
_PROVISION = "1125"
_DEPOSITS_FOSA = "2100"
_DEPOSITS_BOSA = "2110"
_DIVIDENDS_PAYABLE = "2300"
_INSTITUTIONAL_CAP = "3100"
_SHARE_CAPITAL = "3200"
_RETAINED_SURPLUS = "3300"
_RESERVES = "3400"
_INTEREST_INCOME = "4100"
_FEE_INCOME = "4300"
_PENALTY_INCOME = "4350"
_INTEREST_EXPENSE = "5100"
_LOAN_LOSS_PROV_EXP = "5200"


def guard_tenant_id(tenant_id: str | None) -> str:
	if not tenant_id or not tenant_id.strip():
		raise PermissionError("tenant_id_required")
	return tenant_id.strip()


def _now() -> str:
	return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _today() -> str:
	return date.today().isoformat()


def _period_key(value_date: str) -> str:
	"""Extract YYYY-MM from an ISO date string."""
	return value_date[:7]


class SACCOGLService:
	"""Async service implementing the SACCO General Ledger.

	Uses in-memory stores (dict) keyed by tenant_id for multi-tenancy.
	Swap out for async DB calls in production by replacing the _store_* / _fetch_*
	methods.
	"""

	def __init__(self) -> None:
		# {tenant_id: {account_code: GLAccount}}
		self._accounts: dict[str, dict[str, GLAccount]] = defaultdict(dict)
		# {tenant_id: [JournalEntry]}
		self._journals: dict[str, list[JournalEntry]] = defaultdict(list)
		# {tenant_id: {period_key: AccountingPeriod}}
		self._periods: dict[str, dict[str, AccountingPeriod]] = defaultdict(dict)
		# subsidiary ledgers for reconciliation: {tenant_id: {account_code: Decimal}}
		self._subsidiary: dict[str, dict[str, Decimal]] = defaultdict(lambda: defaultdict(Decimal))

	# ── Helpers ────────────────────────────────────────────────────────────────

	def _log_entry(self, tenant_id: str, ref: str, txn_type: str, amount: Decimal) -> None:
		_log.debug("[%s] %s | %s | %s", tenant_id, ref, txn_type, amount)

	def _log_balance(self, tenant_id: str, code: str, balance: Decimal) -> None:
		_log.debug("[%s] balance %s → %s", tenant_id, code, balance)

	def _log_period(self, tenant_id: str, period: str, action: str) -> None:
		_log.info("[%s] period %s %s", tenant_id, period, action)

	def _account(self, tenant_id: str, code: str) -> GLAccount:
		acc = self._accounts[tenant_id].get(code)
		if acc is None:
			raise KeyError(f"account_not_found: {code}")
		return acc

	def _period_open(self, tenant_id: str, period_key: str) -> bool:
		p = self._periods[tenant_id].get(period_key)
		return p is None or p.status == "open"

	def _assert_period_open(self, tenant_id: str, value_date: str) -> None:
		pk = _period_key(value_date)
		if not self._period_open(tenant_id, pk):
			raise ValueError(f"period_closed: {pk}")

	def _post_lines(
		self,
		tenant_id: str,
		lines: list[JournalLine],
		reference: str,
		transaction_type: str,
		value_date: str,
		posted_by: str,
		narration: str = "",
	) -> JournalEntry:
		"""Core double-entry posting engine."""
		total_debit = sum(l.debit for l in lines)
		total_credit = sum(l.credit for l in lines)
		if total_debit.quantize(Decimal("0.01")) != total_credit.quantize(Decimal("0.01")):
			raise ValueError(
				f"double_entry_imbalance: debit={total_debit} credit={total_credit}"
			)

		self._assert_period_open(tenant_id, value_date)

		entry = JournalEntry(
			tenant_id=tenant_id,
			reference=reference,
			transaction_type=transaction_type,
			value_date=value_date,
			posted_at=_now(),
			posted_by=posted_by,
			narration=narration,
			lines=lines,
			total_debit=total_debit,
			total_credit=total_credit,
			period_key=_period_key(value_date),
		)

		# Update running balances
		for line in lines:
			acc = self._account(tenant_id, line.account_code)
			if acc.normal_balance == NormalBalance.DEBIT:
				acc.balance += line.debit - line.credit
			else:
				acc.balance += line.credit - line.debit
			self._log_balance(tenant_id, line.account_code, acc.balance)

		self._journals[tenant_id].append(entry)
		self._log_entry(tenant_id, reference, transaction_type, total_debit)
		return entry

	def _ref(self, txn_type: str) -> str:
		return f"{txn_type.upper()}-{uuid7str()[:8]}"

	def _vdate(self, value_date: str | None) -> str:
		return value_date or _today()

	# ── Chart of Accounts ──────────────────────────────────────────────────────

	async def initialise_sacco_coa(self, tenant_id: str) -> dict[str, Any]:
		"""Create the standard SACCO chart of accounts for a tenant.

		Idempotent — safe to call multiple times; will not overwrite existing accounts.
		"""
		tenant_id = guard_tenant_id(tenant_id)
		existing = self._accounts[tenant_id]
		created = []
		skipped = []

		for defn in STANDARD_COA:
			code = defn["code"]
			if code in existing:
				skipped.append(code)
				continue
			acc = GLAccount(
				tenant_id=tenant_id,
				code=code,
				name=defn["name"],
				category=AccountCategory(defn["category"]),
				normal_balance=NormalBalance(defn["normal_balance"]),
				description=defn.get("description", ""),
				created_at=_now(),
				updated_at=_now(),
			)
			existing[code] = acc
			created.append(code)

		# Ensure first open period exists for today
		pk = _period_key(_today())
		if pk not in self._periods[tenant_id]:
			self._periods[tenant_id][pk] = AccountingPeriod(
				tenant_id=tenant_id,
				year=int(pk[:4]),
				month=int(pk[5:7]),
				status="open",
				opened_at=_now(),
			)

		_log.info("[%s] COA initialised: created=%d skipped=%d", tenant_id, len(created), len(skipped))
		return {
			"tenant_id": tenant_id,
			"created": created,
			"skipped": skipped,
			"total_accounts": len(existing),
		}

	# ── Generic Posting ────────────────────────────────────────────────────────

	async def post_transaction(
		self,
		tenant_id: str,
		transaction_type: str,
		entries: list[dict[str, Any]],
		reference: str,
		value_date: str,
		posted_by: str,
		narration: str = "",
	) -> dict[str, Any]:
		"""Generic double-entry transaction posting.

		entries: list of {account_code, debit, credit, narrative}
		"""
		tenant_id = guard_tenant_id(tenant_id)
		lines = [
			JournalLine(
				account_code=e["account_code"],
				debit=Decimal(str(e.get("debit", 0))),
				credit=Decimal(str(e.get("credit", 0))),
				narrative=e.get("narrative", ""),
			)
			for e in entries
		]
		entry = self._post_lines(
			tenant_id, lines, reference, transaction_type, value_date, posted_by, narration
		)
		return {
			"id": entry.id,
			"reference": entry.reference,
			"transaction_type": entry.transaction_type,
			"total_debit": str(entry.total_debit),
			"posted_at": entry.posted_at,
		}

	# ── Standard Transaction Types ─────────────────────────────────────────────

	async def post_member_deposit(
		self,
		tenant_id: str,
		member_id: str,
		account_type: str,
		amount: Decimal,
		channel: str = "cash",
		value_date: str | None = None,
		posted_by: str = "system",
	) -> dict[str, Any]:
		"""DR Cash/Bank → CR Member Deposits (FOSA or BOSA)."""
		tenant_id = guard_tenant_id(tenant_id)
		assert amount > Decimal("0"), "amount must be positive"
		vd = self._vdate(value_date)

		debit_code = _BANK if channel in {"mpesa", "bank_transfer", "cheque"} else _CASH
		credit_code = _DEPOSITS_FOSA if account_type.upper() == "FOSA" else _DEPOSITS_BOSA

		lines = [
			JournalLine(account_code=debit_code, debit=amount, narrative=f"Deposit from {member_id}"),
			JournalLine(account_code=credit_code, credit=amount, narrative=f"Member deposit {member_id}"),
		]
		# Update subsidiary
		self._subsidiary[tenant_id][credit_code] += amount
		entry = self._post_lines(
			tenant_id, lines, self._ref("DEP"), "member_deposit", vd, posted_by,
			f"Deposit {account_type} member={member_id} channel={channel}"
		)
		return {"id": entry.id, "reference": entry.reference, "transaction_type": entry.transaction_type,
			"total_debit": str(entry.total_debit), "posted_at": entry.posted_at, "member_id": member_id}

	async def post_loan_disbursement(
		self,
		tenant_id: str,
		loan_id: str,
		amount: Decimal,
		loan_type: str = "BOSA",
		disbursement_channel: str = "savings_account",
		value_date: str | None = None,
		posted_by: str = "system",
	) -> dict[str, Any]:
		"""DR Member Loans → CR Member Deposits/Cash."""
		tenant_id = guard_tenant_id(tenant_id)
		assert amount > Decimal("0"), "amount must be positive"
		vd = self._vdate(value_date)

		loan_code = _LOANS_FOSA if loan_type.upper() == "FOSA" else _LOANS_BOSA
		credit_code = (
			_BANK if disbursement_channel == "bank_transfer"
			else _CASH if disbursement_channel == "cash"
			else _DEPOSITS_BOSA
		)
		lines = [
			JournalLine(account_code=loan_code, debit=amount, narrative=f"Loan disbursement {loan_id}"),
			JournalLine(account_code=credit_code, credit=amount, narrative=f"Loan {loan_id} payment out"),
		]
		self._subsidiary[tenant_id][loan_code] += amount
		entry = self._post_lines(
			tenant_id, lines, self._ref("DISB"), "loan_disbursement", vd, posted_by,
			f"Loan disbursement loan={loan_id} type={loan_type}"
		)
		return {"id": entry.id, "reference": entry.reference, "transaction_type": entry.transaction_type,
			"total_debit": str(entry.total_debit), "posted_at": entry.posted_at, "loan_id": loan_id}

	async def post_loan_repayment(
		self,
		tenant_id: str,
		loan_id: str,
		principal: Decimal,
		interest: Decimal,
		penalty: Decimal = Decimal("0"),
		payment_channel: str = "cash",
		value_date: str | None = None,
		posted_by: str = "system",
	) -> dict[str, Any]:
		"""DR Cash/Deposits → CR Loans (principal) + Interest Income + Penalty Income."""
		tenant_id = guard_tenant_id(tenant_id)
		total = principal + interest + penalty
		assert total > Decimal("0"), "total repayment must be positive"
		vd = self._vdate(value_date)

		debit_code = _BANK if payment_channel in {"mpesa", "bank_transfer"} else _CASH
		loan_code = _LOANS_BOSA  # default; caller should specify via loan_id lookup

		lines: list[JournalLine] = [
			JournalLine(account_code=debit_code, debit=total, narrative=f"Repayment {loan_id}"),
			JournalLine(account_code=loan_code, credit=principal, narrative=f"Principal {loan_id}"),
		]
		if interest > Decimal("0"):
			lines.append(JournalLine(account_code=_INTEREST_INCOME, credit=interest, narrative=f"Interest {loan_id}"))
		if penalty > Decimal("0"):
			lines.append(JournalLine(account_code=_PENALTY_INCOME, credit=penalty, narrative=f"Penalty {loan_id}"))

		self._subsidiary[tenant_id][loan_code] -= principal
		entry = self._post_lines(
			tenant_id, lines, self._ref("REPM"), "loan_repayment", vd, posted_by,
			f"Repayment loan={loan_id} principal={principal} interest={interest} penalty={penalty}"
		)
		return {"id": entry.id, "reference": entry.reference, "transaction_type": entry.transaction_type,
			"total_debit": str(entry.total_debit), "posted_at": entry.posted_at, "loan_id": loan_id}

	async def post_interest_earned(
		self,
		tenant_id: str,
		account_id: str,
		amount: Decimal,
		period: str,
		account_type: str = "BOSA",
		value_date: str | None = None,
		posted_by: str = "system",
	) -> dict[str, Any]:
		"""DR Interest Expense → CR Member Deposits (interest credited to members)."""
		tenant_id = guard_tenant_id(tenant_id)
		assert amount > Decimal("0"), "amount must be positive"
		vd = self._vdate(value_date)

		credit_code = _DEPOSITS_FOSA if account_type.upper() == "FOSA" else _DEPOSITS_BOSA
		lines = [
			JournalLine(account_code=_INTEREST_EXPENSE, debit=amount, narrative=f"Interest on {account_id} {period}"),
			JournalLine(account_code=credit_code, credit=amount, narrative=f"Interest credited {account_id}"),
		]
		self._subsidiary[tenant_id][credit_code] += amount
		entry = self._post_lines(
			tenant_id, lines, self._ref("INT"), "interest_earned", vd, posted_by,
			f"Interest earned account={account_id} period={period}"
		)
		return {"id": entry.id, "reference": entry.reference, "transaction_type": entry.transaction_type,
			"total_debit": str(entry.total_debit), "posted_at": entry.posted_at,
			"account_id": account_id, "period": period}

	async def post_dividend(
		self,
		tenant_id: str,
		member_id: str,
		amount: Decimal,
		year: int,
		pay_to_deposits: bool = False,
		value_date: str | None = None,
		posted_by: str = "system",
	) -> dict[str, Any]:
		"""DR Retained Surplus → CR Dividends Payable (then optionally → CR Member Deposits)."""
		tenant_id = guard_tenant_id(tenant_id)
		assert amount > Decimal("0"), "amount must be positive"
		vd = self._vdate(value_date)

		# Declaration entry
		decl_lines = [
			JournalLine(account_code=_RETAINED_SURPLUS, debit=amount, narrative=f"Dividend {year} member={member_id}"),
			JournalLine(account_code=_DIVIDENDS_PAYABLE, credit=amount, narrative=f"Dividend payable {member_id}"),
		]
		decl = self._post_lines(
			tenant_id, decl_lines, self._ref("DIVD"), "dividend_declaration", vd, posted_by,
			f"Dividend year={year} member={member_id}"
		)

		if pay_to_deposits:
			pay_lines = [
				JournalLine(account_code=_DIVIDENDS_PAYABLE, debit=amount, narrative=f"Pay dividend {member_id}"),
				JournalLine(account_code=_DEPOSITS_BOSA, credit=amount, narrative=f"Dividend to deposits {member_id}"),
			]
			self._post_lines(
				tenant_id, pay_lines, self._ref("DIVP"), "dividend_payment", vd, posted_by,
				f"Dividend payment year={year} member={member_id}"
			)
			self._subsidiary[tenant_id][_DEPOSITS_BOSA] += amount

		return {"declaration_id": decl.id, "member_id": member_id, "amount": str(amount), "year": year}

	async def post_share_purchase(
		self,
		tenant_id: str,
		member_id: str,
		amount: Decimal,
		channel: str = "cash",
		value_date: str | None = None,
		posted_by: str = "system",
	) -> dict[str, Any]:
		"""DR Cash/Member Deposits → CR Share Capital."""
		tenant_id = guard_tenant_id(tenant_id)
		assert amount > Decimal("0"), "amount must be positive"
		vd = self._vdate(value_date)

		debit_code = _BANK if channel in {"mpesa", "bank_transfer"} else _CASH
		lines = [
			JournalLine(account_code=debit_code, debit=amount, narrative=f"Share purchase {member_id}"),
			JournalLine(account_code=_SHARE_CAPITAL, credit=amount, narrative=f"Shares issued {member_id}"),
		]
		entry = self._post_lines(
			tenant_id, lines, self._ref("SHRP"), "share_purchase", vd, posted_by,
			f"Share purchase member={member_id}"
		)
		return {"id": entry.id, "reference": entry.reference, "transaction_type": entry.transaction_type,
			"total_debit": str(entry.total_debit), "posted_at": entry.posted_at, "member_id": member_id}

	async def post_withdrawal(
		self,
		tenant_id: str,
		member_id: str,
		amount: Decimal,
		account_type: str = "FOSA",
		channel: str = "cash",
		value_date: str | None = None,
		posted_by: str = "system",
	) -> dict[str, Any]:
		"""DR Member Deposits → CR Cash/Bank."""
		tenant_id = guard_tenant_id(tenant_id)
		assert amount > Decimal("0"), "amount must be positive"
		vd = self._vdate(value_date)

		debit_code = _DEPOSITS_FOSA if account_type.upper() == "FOSA" else _DEPOSITS_BOSA
		credit_code = _BANK if channel in {"mpesa", "bank_transfer"} else _CASH
		lines = [
			JournalLine(account_code=debit_code, debit=amount, narrative=f"Withdrawal {member_id}"),
			JournalLine(account_code=credit_code, credit=amount, narrative=f"Cash out {member_id}"),
		]
		self._subsidiary[tenant_id][debit_code] -= amount
		entry = self._post_lines(
			tenant_id, lines, self._ref("WITH"), "withdrawal", vd, posted_by,
			f"Withdrawal member={member_id} type={account_type} channel={channel}"
		)
		return {"id": entry.id, "reference": entry.reference, "transaction_type": entry.transaction_type,
			"total_debit": str(entry.total_debit), "posted_at": entry.posted_at, "member_id": member_id}

	async def post_provision(
		self,
		tenant_id: str,
		loan_id: str,
		provision_amount: Decimal,
		value_date: str | None = None,
		posted_by: str = "system",
	) -> dict[str, Any]:
		"""DR Loan Loss Provisions (expense) → CR Provision for Loan Losses (contra-asset)."""
		tenant_id = guard_tenant_id(tenant_id)
		assert provision_amount > Decimal("0"), "provision_amount must be positive"
		vd = self._vdate(value_date)

		lines = [
			JournalLine(account_code=_LOAN_LOSS_PROV_EXP, debit=provision_amount, narrative=f"Provision loan={loan_id}"),
			JournalLine(account_code=_PROVISION, credit=provision_amount, narrative=f"Provision balance loan={loan_id}"),
		]
		entry = self._post_lines(
			tenant_id, lines, self._ref("PROV"), "provision", vd, posted_by,
			f"Loan loss provision loan={loan_id}"
		)
		return {"id": entry.id, "reference": entry.reference, "transaction_type": entry.transaction_type,
			"total_debit": str(entry.total_debit), "posted_at": entry.posted_at, "loan_id": loan_id}

	async def post_write_off(
		self,
		tenant_id: str,
		loan_id: str,
		amount: Decimal,
		loan_type: str = "BOSA",
		value_date: str | None = None,
		posted_by: str = "system",
	) -> dict[str, Any]:
		"""DR Provision for Loan Losses (contra-asset) → CR Member Loans."""
		tenant_id = guard_tenant_id(tenant_id)
		assert amount > Decimal("0"), "amount must be positive"
		vd = self._vdate(value_date)

		loan_code = _LOANS_FOSA if loan_type.upper() == "FOSA" else _LOANS_BOSA
		lines = [
			JournalLine(account_code=_PROVISION, debit=amount, narrative=f"Write-off loan={loan_id}"),
			JournalLine(account_code=loan_code, credit=amount, narrative=f"Loan written off {loan_id}"),
		]
		self._subsidiary[tenant_id][loan_code] -= amount
		entry = self._post_lines(
			tenant_id, lines, self._ref("WOFF"), "write_off", vd, posted_by,
			f"Loan write-off loan={loan_id} amount={amount}"
		)
		return {"id": entry.id, "reference": entry.reference, "transaction_type": entry.transaction_type,
			"total_debit": str(entry.total_debit), "posted_at": entry.posted_at, "loan_id": loan_id}

	# ── Queries ────────────────────────────────────────────────────────────────

	async def get_account_balance(
		self,
		tenant_id: str,
		account_code: str,
		as_of_date: str | None = None,
	) -> Decimal:
		"""Running balance from journal entries, optionally filtered to as_of_date."""
		tenant_id = guard_tenant_id(tenant_id)
		self._account(tenant_id, account_code)  # existence check

		if as_of_date is None:
			return self._accounts[tenant_id][account_code].balance

		# Recompute from journal lines up to as_of_date
		acc_def = self._accounts[tenant_id][account_code]
		balance = Decimal("0")
		for entry in self._journals[tenant_id]:
			if entry.value_date <= as_of_date:
				for line in entry.lines:
					if line.account_code == account_code:
						if acc_def.normal_balance == NormalBalance.DEBIT:
							balance += line.debit - line.credit
						else:
							balance += line.credit - line.debit
		return balance

	async def get_trial_balance(
		self, tenant_id: str, as_of_date: str
	) -> list[dict[str, Any]]:
		"""Standard trial balance as of a date."""
		tenant_id = guard_tenant_id(tenant_id)
		rows: list[dict[str, Any]] = []

		# Aggregate debits and credits per account up to as_of_date
		debit_totals: dict[str, Decimal] = defaultdict(Decimal)
		credit_totals: dict[str, Decimal] = defaultdict(Decimal)
		for entry in self._journals[tenant_id]:
			if entry.value_date <= as_of_date:
				for line in entry.lines:
					debit_totals[line.account_code] += line.debit
					credit_totals[line.account_code] += line.credit

		for code, acc in sorted(self._accounts[tenant_id].items()):
			d = debit_totals[code]
			c = credit_totals[code]
			net = d - c
			rows.append({
				"code": code,
				"name": acc.name,
				"category": acc.category.value,
				"normal_balance": acc.normal_balance.value,
				"debit": d,
				"credit": c,
				"net": net,
			})
		return rows

	async def get_balance_sheet(
		self, tenant_id: str, as_of_date: str
	) -> BalanceSheet:
		"""SASRA-compliant balance sheet."""
		tenant_id = guard_tenant_id(tenant_id)
		rows = await self.get_trial_balance(tenant_id, as_of_date)

		assets: dict[str, Decimal] = {}
		liabilities: dict[str, Decimal] = {}
		equity: dict[str, Decimal] = {}

		for row in rows:
			cat = row["category"]
			normal = row["normal_balance"]
			# Balance = net if normal is debit, else -net
			balance = row["net"] if normal == "debit" else -row["net"]
			if cat == "asset":
				assets[row["name"]] = balance
			elif cat == "liability":
				liabilities[row["name"]] = balance
			elif cat == "equity":
				equity[row["name"]] = balance

		# Income/expense feed into retained surplus (simplified closing)
		# For a balance sheet snapshot we include P&L effect in equity
		income_rows = [r for r in rows if r["category"] == "income"]
		expense_rows = [r for r in rows if r["category"] == "expense"]
		period_surplus = sum(-r["net"] for r in income_rows) - sum(r["net"] for r in expense_rows)
		if period_surplus != Decimal("0"):
			equity["Period Surplus/(Deficit)"] = period_surplus

		total_assets = sum(assets.values())
		total_liabilities = sum(liabilities.values())
		total_equity = sum(equity.values())
		total_l_e = total_liabilities + total_equity

		return BalanceSheet(
			as_of_date=as_of_date,
			tenant_id=tenant_id,
			assets=assets,
			liabilities=liabilities,
			equity=equity,
			total_assets=total_assets,
			total_liabilities=total_liabilities,
			total_equity=total_equity,
			total_liabilities_equity=total_l_e,
			is_balanced=abs(total_assets - total_l_e) < Decimal("0.01"),
		)

	async def get_income_statement(
		self, tenant_id: str, from_date: str, to_date: str
	) -> IncomeStatement:
		"""P&L for a period."""
		tenant_id = guard_tenant_id(tenant_id)
		debit_totals: dict[str, Decimal] = defaultdict(Decimal)
		credit_totals: dict[str, Decimal] = defaultdict(Decimal)
		for entry in self._journals[tenant_id]:
			if from_date <= entry.value_date <= to_date:
				for line in entry.lines:
					debit_totals[line.account_code] += line.debit
					credit_totals[line.account_code] += line.credit

		income: dict[str, Decimal] = {}
		expenses: dict[str, Decimal] = {}
		for code, acc in self._accounts[tenant_id].items():
			if acc.category == AccountCategory.INCOME:
				income[acc.name] = credit_totals[code] - debit_totals[code]
			elif acc.category == AccountCategory.EXPENSE:
				expenses[acc.name] = debit_totals[code] - credit_totals[code]

		total_income = sum(income.values())
		total_expenses = sum(expenses.values())
		return IncomeStatement(
			from_date=from_date,
			to_date=to_date,
			tenant_id=tenant_id,
			income=income,
			expenses=expenses,
			total_income=total_income,
			total_expenses=total_expenses,
			surplus_deficit=total_income - total_expenses,
		)

	async def get_journal_entries(
		self,
		tenant_id: str,
		from_date: str,
		to_date: str,
		account_code: str | None = None,
		transaction_type: str | None = None,
		limit: int = 50,
	) -> list[dict[str, Any]]:
		"""Filter journal entries with optional account/type filters."""
		tenant_id = guard_tenant_id(tenant_id)
		results = []
		for entry in self._journals[tenant_id]:
			if not (from_date <= entry.value_date <= to_date):
				continue
			if transaction_type and entry.transaction_type != transaction_type:
				continue
			if account_code:
				codes = {l.account_code for l in entry.lines}
				if account_code not in codes:
					continue
			results.append({
				"id": entry.id,
				"reference": entry.reference,
				"transaction_type": entry.transaction_type,
				"value_date": entry.value_date,
				"posted_at": entry.posted_at,
				"posted_by": entry.posted_by,
				"narration": entry.narration,
				"total_debit": str(entry.total_debit),
				"total_credit": str(entry.total_credit),
				"lines": [
					{
						"account_code": l.account_code,
						"debit": str(l.debit),
						"credit": str(l.credit),
						"narrative": l.narrative,
					}
					for l in entry.lines
				],
			})
			if len(results) >= limit:
				break
		return results

	async def get_gl_summary(self, tenant_id: str, period: str) -> GLSummary:
		"""Key management metrics for a period (YYYY-MM)."""
		tenant_id = guard_tenant_id(tenant_id)
		# Use last day of period as snapshot date
		year, month = int(period[:4]), int(period[5:7])
		import calendar
		last_day = calendar.monthrange(year, month)[1]
		as_of = f"{period}-{last_day:02d}"

		rows = await self.get_trial_balance(tenant_id, as_of)
		row_map = {r["code"]: r for r in rows}

		def net(code: str) -> Decimal:
			r = row_map.get(code)
			if not r:
				return Decimal("0")
			return r["net"] if r["normal_balance"] == "debit" else -r["net"]

		loans_fosa = net(_LOANS_FOSA)
		loans_bosa = net(_LOANS_BOSA)
		provision = -net(_PROVISION)  # contra-asset, credit balance → subtract
		gross_loans = loans_fosa + loans_bosa
		net_loans = gross_loans - provision

		# Income/expense for the period
		from_date = f"{period}-01"
		stmt = await self.get_income_statement(tenant_id, from_date, as_of)

		# Build total assets (debit-normal assets only, net of contra-assets)
		total_assets = sum(
			(-r["net"] if r["normal_balance"] == "credit" else r["net"])
			for r in rows if r["category"] == "asset"
		)
		total_equity = sum(-r["net"] for r in rows if r["category"] == "equity")
		total_equity += stmt.surplus_deficit  # include current period P&L

		capital_ratio = (
			(total_equity / total_assets * 100).quantize(Decimal("0.01"), ROUND_HALF_UP)
			if total_assets else Decimal("0")
		)
		npa_ratio = (
			(provision / gross_loans * 100).quantize(Decimal("0.01"), ROUND_HALF_UP)
			if gross_loans else Decimal("0")
		)
		deposit_base = abs(net(_DEPOSITS_FOSA)) + abs(net(_DEPOSITS_BOSA))

		return GLSummary(
			tenant_id=tenant_id,
			period=period,
			total_assets=total_assets,
			loan_book_gross=gross_loans,
			loan_book_net=net_loans,
			deposit_base=deposit_base,
			share_capital=abs(net(_SHARE_CAPITAL)),
			total_equity=total_equity,
			capital_ratio_pct=capital_ratio,
			npa_ratio_pct=npa_ratio,
			total_income=stmt.total_income,
			total_expenses=stmt.total_expenses,
			surplus_deficit=stmt.surplus_deficit,
			journal_entry_count=len(self._journals[tenant_id]),
		)

	# ── Double-Entry Validation ────────────────────────────────────────────────

	async def validate_double_entry(
		self, tenant_id: str, as_of_date: str | None = None
	) -> dict[str, Any]:
		"""Verify that every posted journal entry balances."""
		tenant_id = guard_tenant_id(tenant_id)
		total_debit = Decimal("0")
		total_credit = Decimal("0")
		unbalanced = []

		for entry in self._journals[tenant_id]:
			if as_of_date and entry.value_date > as_of_date:
				continue
			if entry.total_debit != entry.total_credit:
				unbalanced.append({
					"id": entry.id,
					"reference": entry.reference,
					"debit": str(entry.total_debit),
					"credit": str(entry.total_credit),
				})
			total_debit += entry.total_debit
			total_credit += entry.total_credit

		diff = (total_debit - total_credit).quantize(Decimal("0.01"))
		return {
			"balanced": diff == Decimal("0") and not unbalanced,
			"difference": str(diff),
			"total_debit": str(total_debit),
			"total_credit": str(total_credit),
			"unbalanced_entries": unbalanced,
			"entry_count": len(self._journals[tenant_id]),
		}

	# ── Period Management ──────────────────────────────────────────────────────

	async def open_period(
		self, tenant_id: str, year: int, month: int
	) -> dict[str, Any]:
		tenant_id = guard_tenant_id(tenant_id)
		pk = f"{year}-{month:02d}"
		existing = self._periods[tenant_id].get(pk)
		if existing and existing.status == "open":
			return {"period": pk, "status": "already_open"}

		period = AccountingPeriod(
			tenant_id=tenant_id, year=year, month=month,
			status="open", opened_at=_now()
		)
		self._periods[tenant_id][pk] = period
		self._log_period(tenant_id, pk, "opened")
		return {"period": pk, "status": "open", "opened_at": period.opened_at}

	async def close_period(
		self, tenant_id: str, year: int, month: int, closed_by: str
	) -> dict[str, Any]:
		tenant_id = guard_tenant_id(tenant_id)
		pk = f"{year}-{month:02d}"
		period = self._periods[tenant_id].get(pk)
		if period is None:
			raise KeyError(f"period_not_found: {pk}")
		if period.status == "closed":
			return {"period": pk, "status": "already_closed"}

		# Validate balance before closing
		last_day = __import__("calendar").monthrange(year, month)[1]
		validation = await self.validate_double_entry(tenant_id, f"{pk}-{last_day:02d}")
		if not validation["balanced"]:
			raise ValueError(f"cannot_close_unbalanced_period: diff={validation['difference']}")

		period.status = "closed"
		period.closed_at = _now()
		period.closed_by = closed_by
		self._log_period(tenant_id, pk, "closed")
		return {"period": pk, "status": "closed", "closed_at": period.closed_at, "closed_by": closed_by}

	async def get_period_status(
		self, tenant_id: str, year: int, month: int
	) -> dict[str, Any]:
		tenant_id = guard_tenant_id(tenant_id)
		pk = f"{year}-{month:02d}"
		period = self._periods[tenant_id].get(pk)
		if period is None:
			return {"period": pk, "status": "not_found"}
		return period.model_dump()

	# ── Subsidiary Ledger Reconciliation ───────────────────────────────────────

	async def reconcile_subsidiary_ledgers(
		self, tenant_id: str, as_of_date: str
	) -> ReconciliationResult:
		"""Compare GL balances against subsidiary totals."""
		tenant_id = guard_tenant_id(tenant_id)
		gl_deps_fosa = await self.get_account_balance(tenant_id, _DEPOSITS_FOSA, as_of_date)
		gl_deps_bosa = await self.get_account_balance(tenant_id, _DEPOSITS_BOSA, as_of_date)
		gl_loans_fosa = await self.get_account_balance(tenant_id, _LOANS_FOSA, as_of_date)
		gl_loans_bosa = await self.get_account_balance(tenant_id, _LOANS_BOSA, as_of_date)

		sub = self._subsidiary[tenant_id]
		sub_deps_fosa = sub.get(_DEPOSITS_FOSA, Decimal("0"))
		sub_deps_bosa = sub.get(_DEPOSITS_BOSA, Decimal("0"))
		sub_loans_fosa = sub.get(_LOANS_FOSA, Decimal("0"))
		sub_loans_bosa = sub.get(_LOANS_BOSA, Decimal("0"))

		# GL deposits are credit-normal; balance stored as credit net
		gl_total_deps = abs(gl_deps_fosa) + abs(gl_deps_bosa)
		sub_total_deps = sub_deps_fosa + sub_deps_bosa
		gl_total_loans = gl_loans_fosa + gl_loans_bosa
		sub_total_loans = sub_loans_fosa + sub_loans_bosa

		differences = []
		if abs(gl_total_deps - sub_total_deps) > Decimal("0.01"):
			differences.append({
				"item": "member_deposits",
				"gl": str(gl_total_deps),
				"subsidiary": str(sub_total_deps),
				"difference": str(gl_total_deps - sub_total_deps),
			})
		if abs(gl_total_loans - sub_total_loans) > Decimal("0.01"):
			differences.append({
				"item": "member_loans",
				"gl": str(gl_total_loans),
				"subsidiary": str(sub_total_loans),
				"difference": str(gl_total_loans - sub_total_loans),
			})

		items = [
			{"account": "Deposits FOSA", "gl": str(abs(gl_deps_fosa)), "sub": str(sub_deps_fosa)},
			{"account": "Deposits BOSA", "gl": str(abs(gl_deps_bosa)), "sub": str(sub_deps_bosa)},
			{"account": "Loans FOSA", "gl": str(gl_loans_fosa), "sub": str(sub_loans_fosa)},
			{"account": "Loans BOSA", "gl": str(gl_loans_bosa), "sub": str(sub_loans_bosa)},
		]

		return ReconciliationResult(
			tenant_id=tenant_id,
			as_of_date=as_of_date,
			reconciled=len(differences) == 0,
			items=items,
			differences=differences,
			gl_total_deposits=gl_total_deps,
			subsidiary_total_deposits=sub_total_deps,
			gl_total_loans=gl_total_loans,
			subsidiary_total_loans=sub_total_loans,
		)

	# ── Health ─────────────────────────────────────────────────────────────────

	async def health_check(self) -> dict[str, Any]:
		return {
			"capability": CAPABILITY_ID,
			"status": "healthy",
			"tenants_loaded": len(self._accounts),
			"checked_at": _now(),
		}
