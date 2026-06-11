"""Loan Management System — service layer.

Post-origination loan lifecycle: disbursement → repayment → arrears →
restructuring → moratorium → write-off → recovery → closure.

All arithmetic uses Decimal.  Idempotent batch operations.
Plugs into APG via domain adapters; runs standalone with null adapters.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""
from __future__ import annotations

import logging
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

try:
	from .models import (
		AmortisationMethod, ArrearsPosition, CBK_DPD_THRESHOLDS, CBK_PROVISION_RATES,
		ClosureReason, DemandNoticeType, GLEntry, Installment, Loan, LoanClassification,
		LoanProvision, LoanStatus, Moratorium, MoratoriumType, PenaltyType, PaymentMethod,
		PortfolioQuality, Recovery, Repayment, Restructure, RestructureType,
		StatementLine, WriteOff, uuid7str,
	)
	from .domain.adapters import (
		AuthAdapter, NullAuthAdapter, AuditAdapter, NullAuditAdapter,
		NotifyAdapter, NullNotifyAdapter, GLAdapter, NullGLAdapter,
		LoanRepository, InMemoryLoanRepository, ScheduleRepository,
		InMemoryScheduleRepository, RepaymentRepository, InMemoryRepaymentRepository,
		InMemoryGLEntryStore, InMemoryEventStore,
	)
except ImportError:
	from models import (  # type: ignore[no-redef]
		AmortisationMethod, ArrearsPosition, CBK_DPD_THRESHOLDS, CBK_PROVISION_RATES,
		ClosureReason, DemandNoticeType, GLEntry, Installment, Loan, LoanClassification,
		LoanProvision, LoanStatus, Moratorium, MoratoriumType, PenaltyType, PaymentMethod,
		PortfolioQuality, Recovery, Repayment, Restructure, RestructureType,
		StatementLine, WriteOff, uuid7str,
	)
	from domain.adapters import (  # type: ignore[no-redef]
		AuthAdapter, NullAuthAdapter, AuditAdapter, NullAuditAdapter,
		NotifyAdapter, NullNotifyAdapter, GLAdapter, NullGLAdapter,
		LoanRepository, InMemoryLoanRepository, ScheduleRepository,
		InMemoryScheduleRepository, RepaymentRepository, InMemoryRepaymentRepository,
		InMemoryGLEntryStore, InMemoryEventStore,
	)

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

ZERO = Decimal("0")
ONE = Decimal("1")
DP2 = Decimal("0.01")
DP4 = Decimal("0.0001")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now_iso() -> str:
	return datetime.now(timezone.utc).isoformat(timespec="seconds") + "Z"


def _d(v: Any) -> Decimal:
	if isinstance(v, Decimal):
		return v
	return Decimal(str(v))


def _r2(v: Decimal) -> Decimal:
	return v.quantize(DP2, rounding=ROUND_HALF_UP)


def _r4(v: Decimal) -> Decimal:
	return v.quantize(DP4, rounding=ROUND_HALF_UP)


def _guard_tenant_id(tenant_id: str) -> None:
	if not tenant_id or not tenant_id.strip():
		raise ValueError("tenant_id is required")


def _guard_str(value: str, name: str) -> None:
	if not value or not value.strip():
		raise ValueError(f"{name} is required")


def _log_pretty_path(label: str, tenant_id: str, resource: str) -> str:
	return f"[lms][{tenant_id}] {label}: {resource}"


def _add_months(dt: date, months: int) -> date:
	"""Add N calendar months to a date, clamping to month end."""
	m = dt.month - 1 + months
	year = dt.year + m // 12
	month = m % 12 + 1
	import calendar
	day = min(dt.day, calendar.monthrange(year, month)[1])
	return date(year, month, day)


def _classify_by_dpd(days_past_due: int) -> LoanClassification:
	for threshold, classification in CBK_DPD_THRESHOLDS:
		if days_past_due >= threshold:
			return classification
	return LoanClassification.PERFORMING


# ── Amortisation calculators ─────────────────────────────────────────────────

def _schedule_reducing_balance(
	principal: Decimal,
	annual_rate: Decimal,
	tenor_months: int,
	first_payment_date: date,
) -> list[dict[str, Any]]:
	"""Monthly reducing balance — equal principal instalments, diminishing interest."""
	monthly_principal = _r4(principal / Decimal(tenor_months))
	monthly_rate = _r4(annual_rate / Decimal("12"))
	balance = principal
	rows: list[dict[str, Any]] = []
	for i in range(1, tenor_months + 1):
		due_date = _add_months(first_payment_date, i - 1)
		interest = _r4(balance * monthly_rate)
		# Last instalment absorbs rounding residue
		if i == tenor_months:
			monthly_principal = _r4(balance)
		total = _r4(monthly_principal + interest)
		balance = _r4(balance - monthly_principal)
		rows.append({
			"installment_no": i,
			"due_date": due_date,
			"principal": monthly_principal,
			"interest": interest,
			"total": total,
			"balance": balance,
		})
	return rows


def _schedule_flat_rate(
	principal: Decimal,
	annual_rate: Decimal,
	tenor_months: int,
	first_payment_date: date,
) -> list[dict[str, Any]]:
	"""Flat rate — interest calculated on original principal for all periods."""
	monthly_principal = _r4(principal / Decimal(tenor_months))
	monthly_interest = _r4(principal * annual_rate / Decimal("12"))
	total = _r4(monthly_principal + monthly_interest)
	balance = principal
	rows: list[dict[str, Any]] = []
	for i in range(1, tenor_months + 1):
		due_date = _add_months(first_payment_date, i - 1)
		p = monthly_principal if i < tenor_months else _r4(balance)
		bal_after = _r4(balance - p)
		rows.append({
			"installment_no": i,
			"due_date": due_date,
			"principal": p,
			"interest": monthly_interest,
			"total": _r4(p + monthly_interest),
			"balance": bal_after,
		})
		balance = bal_after
	return rows


def _schedule_french_annuity(
	principal: Decimal,
	annual_rate: Decimal,
	tenor_months: int,
	first_payment_date: date,
) -> list[dict[str, Any]]:
	"""French annuity (PMT) — constant total payment, declining principal / rising interest split."""
	r = annual_rate / Decimal("12")
	if r == ZERO:
		# Zero-rate edge case
		return _schedule_reducing_balance(principal, ZERO, tenor_months, first_payment_date)
	# PMT = P * r(1+r)^n / ((1+r)^n - 1)
	rn = (ONE + r) ** tenor_months
	pmt = _r4(principal * r * rn / (rn - ONE))
	balance = principal
	rows: list[dict[str, Any]] = []
	for i in range(1, tenor_months + 1):
		due_date = _add_months(first_payment_date, i - 1)
		interest = _r4(balance * r)
		p = _r4(pmt - interest)
		if i == tenor_months:
			p = _r4(balance)
		bal_after = _r4(balance - p)
		rows.append({
			"installment_no": i,
			"due_date": due_date,
			"principal": p,
			"interest": interest,
			"total": _r4(p + interest),
			"balance": bal_after,
		})
		balance = bal_after
	return rows


def _schedule_bullet(
	principal: Decimal,
	annual_rate: Decimal,
	tenor_months: int,
	first_payment_date: date,
) -> list[dict[str, Any]]:
	"""Bullet — interest paid monthly, principal at final instalment."""
	monthly_interest = _r4(principal * annual_rate / Decimal("12"))
	rows: list[dict[str, Any]] = []
	for i in range(1, tenor_months + 1):
		due_date = _add_months(first_payment_date, i - 1)
		p = principal if i == tenor_months else ZERO
		bal_after = ZERO if i == tenor_months else principal
		rows.append({
			"installment_no": i,
			"due_date": due_date,
			"principal": p,
			"interest": monthly_interest,
			"total": _r4(p + monthly_interest),
			"balance": bal_after,
		})
	return rows


def _schedule_interest_only(
	principal: Decimal,
	annual_rate: Decimal,
	tenor_months: int,
	first_payment_date: date,
) -> list[dict[str, Any]]:
	"""Interest only — same as bullet (alias for clarity)."""
	return _schedule_bullet(principal, annual_rate, tenor_months, first_payment_date)


_SCHEDULE_FN = {
	AmortisationMethod.REDUCING_BALANCE: _schedule_reducing_balance,
	AmortisationMethod.FLAT_RATE:        _schedule_flat_rate,
	AmortisationMethod.FRENCH_ANNUITY:   _schedule_french_annuity,
	AmortisationMethod.BULLET:           _schedule_bullet,
	AmortisationMethod.INTEREST_ONLY:    _schedule_interest_only,
}


# ── Service ───────────────────────────────────────────────────────────────────

class LoanManagementService:
	"""Post-origination loan lifecycle engine.

	Instantiate with null adapters for standalone/test use, or inject
	production adapters for platform integration.
	"""

	def __init__(
		self,
		*,
		auth: AuthAdapter | None = None,
		audit: AuditAdapter | None = None,
		notify: NotifyAdapter | None = None,
		gl: GLAdapter | None = None,
		loans: LoanRepository | None = None,
		schedules: ScheduleRepository | None = None,
		repayments: RepaymentRepository | None = None,
		gl_entries: InMemoryGLEntryStore | None = None,
		events: InMemoryEventStore | None = None,
	) -> None:
		self._auth = auth or NullAuthAdapter()
		self._audit = audit or NullAuditAdapter()
		self._notify = notify or NullNotifyAdapter()
		self._gl = gl or NullGLAdapter()
		self._loans = loans or InMemoryLoanRepository()
		self._schedules = schedules or InMemoryScheduleRepository()
		self._repayments = repayments or InMemoryRepaymentRepository()
		self._gl_entries = gl_entries or InMemoryGLEntryStore()
		self._events = events or InMemoryEventStore()

	# ── Internal helpers ──────────────────────────────────────────────────────

	async def _load_loan(self, tenant_id: str, loan_id: str) -> Loan:
		raw = await self._loans.get(loan_id, tenant_id)
		if raw is None:
			raise KeyError(f"Loan {loan_id!r} not found for tenant {tenant_id!r}")
		return Loan.model_validate(raw)

	async def _save_loan(self, loan: Loan) -> None:
		loan.updated_at = _now_iso()
		await self._loans.save(loan.model_dump())

	async def _post_gl(
		self,
		tenant_id: str,
		loan_id: str,
		entry_type: str,
		description: str,
		dr_account: str,
		cr_account: str,
		amount: Decimal,
		posting_date: date,
		ref: str | None = None,
		currency: str = "KES",
	) -> str:
		lines = [
			{"account": dr_account, "side": "DR", "amount": str(amount)},
			{"account": cr_account, "side": "CR", "amount": str(amount)},
		]
		gl_id = await self._gl.post_journal(
			tenant_id=tenant_id,
			journal_type="LMS",
			description=description,
			lines=lines,
			ref=ref,
			posting_date=posting_date.isoformat(),
		)
		entry = GLEntry(
			tenant_id=tenant_id,
			loan_id=loan_id,
			entry_type=entry_type,
			description=description,
			dr_account=dr_account,
			cr_account=cr_account,
			amount=amount,
			currency=currency,
			posting_date=posting_date,
			ref=ref or gl_id,
		)
		entry.id = gl_id  # type: ignore[assignment]
		await self._gl_entries.save(entry.model_dump())
		log.info(_log_pretty_path(f"GL posted [{entry_type}]", tenant_id, gl_id))
		return gl_id

	# ── Core lifecycle ────────────────────────────────────────────────────────

	async def disburse_loan(
		self,
		tenant_id: str,
		loan_id: str,
		disbursement_date: date,
		account_id: str,
		amount: Decimal,
		disbursement_ref: str,
	) -> dict[str, Any]:
		"""Disburse a loan: create schedule, post GL, update status."""
		_guard_tenant_id(tenant_id)
		_guard_str(loan_id, "loan_id")
		_guard_str(account_id, "account_id")
		assert amount > ZERO, "disbursement amount must be positive"

		loan = await self._load_loan(tenant_id, loan_id)
		if loan.status != LoanStatus.PENDING_DISBURSEMENT:
			raise ValueError(f"Loan {loan_id} is {loan.status.value}, expected PENDING_DISBURSEMENT")

		loan.disbursement_date = disbursement_date
		loan.disbursed_amount = _r2(amount)
		loan.outstanding_balance = _r2(amount)
		loan.account_id = account_id
		loan.disbursement_ref = disbursement_ref
		loan.status = LoanStatus.ACTIVE
		loan.first_payment_date = loan.first_payment_date or _add_months(disbursement_date, 1)
		loan.maturity_date = _add_months(loan.first_payment_date, loan.tenor_months - 1)

		schedule = await self.generate_amortisation_schedule(
			loan_id=loan_id,
			principal=amount,
			rate=loan.rate,
			tenor_months=loan.tenor_months,
			method=loan.method,
			first_payment_date=loan.first_payment_date,
		)

		gl_id = await self._post_gl(
			tenant_id=tenant_id,
			loan_id=loan_id,
			entry_type="disbursement",
			description=f"Loan disbursement {disbursement_ref}",
			dr_account="1200",  # Loans Receivable
			cr_account="2100",  # Customer Account
			amount=amount,
			posting_date=disbursement_date,
			ref=disbursement_ref,
		)

		await self._save_loan(loan)
		await self._audit.log_event(
			"loan_disbursed", "system", tenant_id, loan_id,
			{"amount": str(amount), "ref": disbursement_ref},
		)
		log.info(_log_pretty_path("disbursed", tenant_id, loan_id))

		return {
			"loan_id": loan_id,
			"schedule": schedule,
			"gl_entry_id": gl_id,
			"disbursed_amount": str(amount),
		}

	async def generate_amortisation_schedule(
		self,
		loan_id: str,
		principal: Decimal,
		rate: Decimal,
		tenor_months: int,
		method: AmortisationMethod,
		first_payment_date: date,
	) -> list[dict[str, Any]]:
		"""Generate and persist the amortisation schedule."""
		assert principal > ZERO, "principal must be positive"
		assert rate > ZERO, "rate must be positive"
		assert tenor_months > 0, "tenor_months must be positive"

		fn = _SCHEDULE_FN[method]
		rows = fn(principal, rate, tenor_months, first_payment_date)

		installments = [
			Installment(
				installment_no=r["installment_no"],
				due_date=r["due_date"],
				principal=r["principal"],
				interest=r["interest"],
				total=r["total"],
				balance=r["balance"],
			).model_dump()
			for r in rows
		]
		await self._schedules.save_installments(loan_id, installments)
		log.info(_log_pretty_path(f"schedule [{method.value}, {tenor_months}m]", "—", loan_id))
		return installments

	async def record_repayment(
		self,
		tenant_id: str,
		loan_id: str,
		amount: Decimal,
		payment_date: date,
		payment_ref: str,
		payment_method: PaymentMethod,
	) -> dict[str, Any]:
		"""Record a repayment: waterfall allocation, GL posting, schedule update."""
		_guard_tenant_id(tenant_id)
		assert amount > ZERO, "repayment amount must be positive"

		loan = await self._load_loan(tenant_id, loan_id)
		if loan.status in (LoanStatus.CLOSED, LoanStatus.WRITTEN_OFF):
			raise ValueError(f"Loan {loan_id} is {loan.status.value} — cannot accept repayments")

		remaining = amount

		# ── Waterfall: penalties → fees → interest → principal ────────────────
		penalty_cleared = ZERO
		fees_cleared    = ZERO
		interest_cleared = ZERO
		principal_cleared = ZERO

		# 1. Penalties
		if loan.total_penalties > ZERO and remaining > ZERO:
			take = min(remaining, loan.total_penalties)
			loan.total_penalties  -= take
			penalty_cleared       += take
			remaining             -= take

		# 2. Fees
		if loan.total_fees > ZERO and remaining > ZERO:
			take = min(remaining, loan.total_fees)
			loan.total_fees   -= take
			fees_cleared      += take
			remaining         -= take

		# 3. Interest — pull from schedule
		installments = await self._schedules.get_installments(loan_id)
		for inst in installments:
			if inst.get("status") in ("paid",):
				continue
			interest_due = _d(inst.get("interest", 0)) - _d(inst.get("paid_interest") or 0)
			if interest_due > ZERO and remaining > ZERO:
				take = min(remaining, interest_due)
				inst["paid_interest"] = str(_d(inst.get("paid_interest") or 0) + take)
				loan.total_interest_paid += take
				interest_cleared         += take
				remaining                -= take

		# 4. Principal
		for inst in installments:
			if inst.get("status") in ("paid",):
				continue
			principal_due = _d(inst.get("principal", 0)) - _d(inst.get("paid_principal") or 0)
			if principal_due > ZERO and remaining > ZERO:
				take = min(remaining, principal_due)
				inst["paid_principal"] = str(_d(inst.get("paid_principal") or 0) + take)
				loan.outstanding_balance     -= take
				loan.total_principal_paid    += take
				principal_cleared            += take
				remaining                    -= take
				# Mark instalment status
				total_paid = _d(inst.get("paid_principal") or 0) + _d(inst.get("paid_interest") or 0)
				if total_paid >= _d(inst.get("total", 0)):
					inst["status"] = "paid"
					inst["paid_date"] = payment_date.isoformat()
					inst["paid_amount"] = str(total_paid)

		await self._schedules.save_installments(loan_id, installments)

		# Count cleared installments
		installments_cleared = sum(1 for i in installments if i.get("status") == "paid")
		next_pending = next((i for i in installments if i.get("status") != "paid"), None)
		next_due_date = next_pending["due_date"] if next_pending else None

		# Post GL
		if principal_cleared > ZERO or interest_cleared > ZERO:
			total_gl = principal_cleared + interest_cleared + penalty_cleared + fees_cleared
			gl_id = await self._post_gl(
				tenant_id=tenant_id,
				loan_id=loan_id,
				entry_type="repayment",
				description=f"Repayment {payment_ref}",
				dr_account="2100",  # Customer Account (debit = reduces liability)
				cr_account="1200",  # Loans Receivable
				amount=total_gl - remaining,  # net allocated
				posting_date=payment_date,
				ref=payment_ref,
			)
		else:
			gl_id = None

		# Check full closure
		if loan.outstanding_balance <= ZERO and loan.total_penalties <= ZERO and loan.total_fees <= ZERO:
			loan.status = LoanStatus.CLOSED
			loan.closure_date = payment_date
			loan.closure_reason = ClosureReason.FULLY_PAID
			loan.days_past_due = 0

		repayment = Repayment(
			loan_id=loan_id,
			tenant_id=tenant_id,
			amount=amount,
			payment_date=payment_date,
			payment_ref=payment_ref,
			payment_method=payment_method,
			penalty_cleared=penalty_cleared,
			fees_cleared=fees_cleared,
			interest_cleared=interest_cleared,
			principal_cleared=principal_cleared,
			unallocated=remaining,
			gl_entry_id=gl_id,
		)
		await self._repayments.save(repayment.model_dump())
		await self._save_loan(loan)
		await self._audit.log_event(
			"repayment_recorded", "system", tenant_id, loan_id,
			{"amount": str(amount), "ref": payment_ref},
		)

		return {
			"allocated": {
				"penalty": str(penalty_cleared),
				"fees": str(fees_cleared),
				"interest": str(interest_cleared),
				"principal": str(principal_cleared),
				"unallocated": str(remaining),
			},
			"remaining_balance": str(loan.outstanding_balance),
			"installments_cleared": installments_cleared,
			"next_due_date": next_due_date,
			"gl_entry_id": gl_id,
		}

	async def calculate_arrears(
		self,
		tenant_id: str,
		loan_id: str,
		as_of_date: date,
	) -> ArrearsPosition:
		"""Calculate arrears position as of a given date."""
		_guard_tenant_id(tenant_id)

		loan = await self._load_loan(tenant_id, loan_id)
		installments = await self._schedules.get_installments(loan_id)

		overdue = [
			i for i in installments
			if i.get("status") not in ("paid",)
			and date.fromisoformat(str(i["due_date"])) <= as_of_date
		]

		amount_in_arrears = sum(
			_d(i["total"]) - _d(i.get("paid_amount") or 0)
			for i in overdue
		)
		installments_missed = len(overdue)
		days_past_due = 0
		if overdue:
			earliest_due = min(date.fromisoformat(str(i["due_date"])) for i in overdue)
			days_past_due = (as_of_date - earliest_due).days

		classification = _classify_by_dpd(days_past_due)
		npa_status = days_past_due >= 90

		# Accrue penalty on overdue principal
		penalty_accrued = ZERO
		if days_past_due > 0 and loan.daily_penalty_rate > ZERO:
			overdue_principal = sum(
				_d(i["principal"]) - _d(i.get("paid_principal") or 0)
				for i in overdue
			)
			penalty_accrued = _r2(overdue_principal * loan.daily_penalty_rate * Decimal(days_past_due))

		# Update loan classification + DPD
		loan.days_past_due = days_past_due
		loan.classification = classification
		if days_past_due > 0 and loan.status == LoanStatus.ACTIVE:
			loan.status = LoanStatus.IN_ARREARS
		if npa_status and loan.status not in (LoanStatus.WRITTEN_OFF, LoanStatus.CLOSED):
			loan.status = LoanStatus.NPA
		await self._save_loan(loan)

		return ArrearsPosition(
			loan_id=loan_id,
			tenant_id=tenant_id,
			as_of_date=as_of_date,
			days_past_due=days_past_due,
			amount_in_arrears=_r2(Decimal(str(amount_in_arrears))),
			installments_missed=installments_missed,
			penalty_accrued=penalty_accrued,
			total_overdue=_r2(Decimal(str(amount_in_arrears)) + penalty_accrued),
			npa_status=npa_status,
			classification=classification,
		)

	async def apply_penalty(
		self,
		tenant_id: str,
		loan_id: str,
		penalty_type: PenaltyType,
		as_of_date: date,
	) -> Decimal:
		"""Calculate and apply penalty charge to the loan."""
		_guard_tenant_id(tenant_id)

		loan = await self._load_loan(tenant_id, loan_id)
		if loan.status in (LoanStatus.CLOSED, LoanStatus.WRITTEN_OFF):
			return ZERO

		arrears = await self.calculate_arrears(tenant_id, loan_id, as_of_date)
		if arrears.days_past_due == 0:
			return ZERO

		if penalty_type == PenaltyType.LATE_FEE:
			penalty = loan.late_fee_amount
		else:  # DAILY_PENALTY
			overdue_principal = arrears.amount_in_arrears
			penalty = _r2(overdue_principal * loan.daily_penalty_rate * Decimal(arrears.days_past_due))

		loan.total_penalties += penalty
		await self._save_loan(loan)
		await self._audit.log_event(
			"penalty_applied", "system", tenant_id, loan_id,
			{"penalty_type": penalty_type.value, "amount": str(penalty)},
		)
		return penalty

	async def restructure_loan(
		self,
		tenant_id: str,
		loan_id: str,
		restructure_type: RestructureType,
		new_terms: dict[str, Any],
		effective_date: date,
		approved_by: str,
	) -> dict[str, Any]:
		"""Restructure a loan: update terms, regenerate schedule, post GL."""
		_guard_tenant_id(tenant_id)
		_guard_str(approved_by, "approved_by")

		loan = await self._load_loan(tenant_id, loan_id)

		old_balance = loan.outstanding_balance
		old_rate    = loan.rate
		old_tenor   = loan.tenor_months

		if restructure_type == RestructureType.EXTEND_TENOR:
			extension = int(new_terms.get("additional_months", 12))
			loan.tenor_months += extension
			loan.maturity_date = _add_months(effective_date, loan.tenor_months)

		elif restructure_type == RestructureType.REDUCE_RATE:
			loan.rate = _d(new_terms["new_rate"])

		elif restructure_type == RestructureType.CAPITALISE_ARREARS:
			arrears_pos = await self.calculate_arrears(tenant_id, loan_id, effective_date)
			capitalised = arrears_pos.amount_in_arrears + arrears_pos.penalty_accrued
			loan.outstanding_balance  += capitalised
			loan.total_penalties = ZERO
			loan.principal             = loan.outstanding_balance

		elif restructure_type == RestructureType.CONVERT_TO_TERM:
			loan.method = AmortisationMethod(new_terms.get("method", AmortisationMethod.REDUCING_BALANCE.value))
			if "tenor_months" in new_terms:
				loan.tenor_months = int(new_terms["tenor_months"])

		new_schedule = await self.generate_amortisation_schedule(
			loan_id=loan_id,
			principal=loan.outstanding_balance,
			rate=loan.rate,
			tenor_months=loan.tenor_months,
			method=loan.method,
			first_payment_date=_add_months(effective_date, 1),
		)

		gl_id = await self._post_gl(
			tenant_id=tenant_id,
			loan_id=loan_id,
			entry_type="restructure",
			description=f"Loan restructure [{restructure_type.value}] approved by {approved_by}",
			dr_account="1200",
			cr_account="3900",  # Restructure suspense
			amount=loan.outstanding_balance,
			posting_date=effective_date,
		)

		loan.status = LoanStatus.RESTRUCTURED

		restructure = Restructure(
			loan_id=loan_id,
			tenant_id=tenant_id,
			restructure_type=restructure_type,
			new_terms=new_terms,
			effective_date=effective_date,
			approved_by=approved_by,
			gl_entry_id=gl_id,
		)
		rec = restructure.model_dump()
		rec["_type"] = "restructure"
		await self._events.save(rec)
		await self._save_loan(loan)
		await self._audit.log_event(
			"loan_restructured", approved_by, tenant_id, loan_id,
			{"type": restructure_type.value, "old_balance": str(old_balance)},
		)

		return {
			"loan_id": loan_id,
			"restructure_id": restructure.id,
			"restructure_type": restructure_type.value,
			"new_outstanding_balance": str(loan.outstanding_balance),
			"new_tenor_months": loan.tenor_months,
			"new_rate": str(loan.rate),
			"gl_entry_id": gl_id,
			"new_schedule": new_schedule,
		}

	async def grant_moratorium(
		self,
		tenant_id: str,
		loan_id: str,
		from_date: date,
		to_date: date,
		moratorium_type: MoratoriumType,
		reason: str,
		approved_by: str,
		interest_accrues: bool = True,
	) -> dict[str, Any]:
		"""Grant a payment holiday.  Extends maturity by the moratorium period."""
		_guard_tenant_id(tenant_id)
		assert from_date < to_date, "from_date must precede to_date"

		loan = await self._load_loan(tenant_id, loan_id)

		moratorium_days = (to_date - from_date).days
		extension_months = max(1, moratorium_days // 30)

		# Extend tenor by the moratorium period
		loan.tenor_months  += extension_months
		loan.maturity_date = _add_months(to_date, loan.tenor_months)
		loan.status        = LoanStatus.MORATORIUM

		# Regenerate schedule from the day after moratorium ends
		new_first_payment = _add_months(to_date, 1)
		new_schedule = await self.generate_amortisation_schedule(
			loan_id=loan_id,
			principal=loan.outstanding_balance,
			rate=loan.rate,
			tenor_months=loan.tenor_months,
			method=loan.method,
			first_payment_date=new_first_payment,
		)

		mora = Moratorium(
			loan_id=loan_id,
			tenant_id=tenant_id,
			from_date=from_date,
			to_date=to_date,
			moratorium_type=moratorium_type,
			interest_accrues=interest_accrues,
			reason=reason,
			approved_by=approved_by,
		)
		rec = mora.model_dump()
		rec["_type"] = "moratorium"
		await self._events.save(rec)
		await self._save_loan(loan)
		await self._audit.log_event(
			"moratorium_granted", approved_by, tenant_id, loan_id,
			{"from": from_date.isoformat(), "to": to_date.isoformat(), "type": moratorium_type.value},
		)

		return {
			"loan_id": loan_id,
			"moratorium_id": mora.id,
			"from_date": from_date.isoformat(),
			"to_date": to_date.isoformat(),
			"interest_accrues": interest_accrues,
			"new_maturity_date": loan.maturity_date.isoformat() if loan.maturity_date else None,
			"new_schedule": new_schedule,
		}

	async def write_off_loan(
		self,
		tenant_id: str,
		loan_id: str,
		write_off_date: date,
		reason: str,
		approved_by: str,
		write_off_amount: Decimal,
	) -> dict[str, Any]:
		"""Write off a loan.  GL: DR Provision / CR Loans Receivable."""
		_guard_tenant_id(tenant_id)
		_guard_str(approved_by, "approved_by")
		assert write_off_amount > ZERO

		loan = await self._load_loan(tenant_id, loan_id)
		if loan.status == LoanStatus.CLOSED:
			raise ValueError("Cannot write off a closed loan")

		gl_id = await self._post_gl(
			tenant_id=tenant_id,
			loan_id=loan_id,
			entry_type="write_off",
			description=f"Loan write-off: {reason}",
			dr_account="5100",  # Provision for Loan Losses
			cr_account="1200",  # Loans Receivable
			amount=write_off_amount,
			posting_date=write_off_date,
		)

		loan.write_off_amount = write_off_amount
		loan.status           = LoanStatus.WRITTEN_OFF
		# Loan stays on books for recovery tracking

		wo = WriteOff(
			loan_id=loan_id,
			tenant_id=tenant_id,
			write_off_date=write_off_date,
			reason=reason,
			approved_by=approved_by,
			write_off_amount=write_off_amount,
			gl_entry_id=gl_id,
		)
		rec = wo.model_dump()
		rec["_type"] = "write_off"
		await self._events.save(rec)
		await self._save_loan(loan)
		await self._audit.log_event(
			"loan_written_off", approved_by, tenant_id, loan_id,
			{"amount": str(write_off_amount), "reason": reason},
		)

		return {
			"loan_id": loan_id,
			"write_off_id": wo.id,
			"write_off_amount": str(write_off_amount),
			"gl_entry_id": gl_id,
			"status": LoanStatus.WRITTEN_OFF.value,
		}

	async def record_recovery(
		self,
		tenant_id: str,
		loan_id: str,
		amount: Decimal,
		recovery_date: date,
		method: str,
	) -> dict[str, Any]:
		"""Post-write-off recovery: DR Cash / CR Recovery Income."""
		_guard_tenant_id(tenant_id)
		assert amount > ZERO

		loan = await self._load_loan(tenant_id, loan_id)
		if loan.status != LoanStatus.WRITTEN_OFF:
			raise ValueError(f"Loan {loan_id} is not written off")

		gl_id = await self._post_gl(
			tenant_id=tenant_id,
			loan_id=loan_id,
			entry_type="recovery",
			description=f"Recovery via {method}",
			dr_account="1000",  # Cash
			cr_account="4200",  # Recovery Income
			amount=amount,
			posting_date=recovery_date,
		)

		loan.recovered_amount += amount
		if loan.recovered_amount >= loan.write_off_amount:
			loan.status = LoanStatus.RECOVERED

		rec_obj = Recovery(
			loan_id=loan_id,
			tenant_id=tenant_id,
			amount=amount,
			recovery_date=recovery_date,
			method=method,
			gl_entry_id=gl_id,
		)
		ev = rec_obj.model_dump()
		ev["_type"] = "recovery"
		await self._events.save(ev)
		await self._save_loan(loan)
		await self._audit.log_event(
			"recovery_recorded", "system", tenant_id, loan_id,
			{"amount": str(amount), "method": method},
		)

		return {
			"loan_id": loan_id,
			"recovery_id": rec_obj.id,
			"recovered_amount": str(amount),
			"total_recovered": str(loan.recovered_amount),
			"gl_entry_id": gl_id,
		}

	# ── Query / reporting ─────────────────────────────────────────────────────

	async def get_loan(self, tenant_id: str, loan_id: str) -> dict[str, Any]:
		_guard_tenant_id(tenant_id)
		loan = await self._load_loan(tenant_id, loan_id)
		return loan.model_dump()

	async def get_loan_schedule(self, tenant_id: str, loan_id: str) -> list[dict[str, Any]]:
		_guard_tenant_id(tenant_id)
		await self._load_loan(tenant_id, loan_id)  # auth / existence check
		return await self._schedules.get_installments(loan_id)

	async def get_loan_statement(
		self,
		tenant_id: str,
		loan_id: str,
		from_date: date,
		to_date: date,
	) -> list[dict[str, Any]]:
		_guard_tenant_id(tenant_id)
		loan = await self._load_loan(tenant_id, loan_id)
		repayments = await self._repayments.list_by_loan(loan_id, tenant_id)
		gl_entries = await self._gl_entries.list_by_loan(loan_id)

		lines: list[dict[str, Any]] = []

		# Opening disbursement
		if loan.disbursement_date and from_date <= loan.disbursement_date <= to_date:
			lines.append(StatementLine(
				date=loan.disbursement_date,
				description=f"Loan disbursement (ref: {loan.disbursement_ref})",
				debit=loan.disbursed_amount,
				credit=ZERO,
				balance=loan.disbursed_amount,
				ref=loan.disbursement_ref,
			).model_dump())

		# Repayments
		running_balance = loan.disbursed_amount
		for r in sorted(repayments, key=lambda x: x["payment_date"]):
			pdate = date.fromisoformat(str(r["payment_date"]))
			if not (from_date <= pdate <= to_date):
				continue
			allocated = _d(r.get("principal_cleared", 0)) + _d(r.get("interest_cleared", 0))
			running_balance -= _d(r.get("principal_cleared", 0))
			lines.append(StatementLine(
				date=pdate,
				description=f"Repayment ({r.get('payment_method', '')})",
				debit=ZERO,
				credit=allocated,
				balance=running_balance,
				ref=r.get("payment_ref"),
			).model_dump())

		lines.sort(key=lambda x: str(x["date"]))
		return lines

	async def list_loans(
		self,
		tenant_id: str,
		customer_id: str | None = None,
		status: str | None = None,
		days_past_due_min: int | None = None,
	) -> list[dict[str, Any]]:
		_guard_tenant_id(tenant_id)
		return await self._loans.list_by_tenant(
			tenant_id=tenant_id,
			customer_id=customer_id,
			status=status,
			days_past_due_min=days_past_due_min,
		)

	async def get_portfolio_quality(
		self,
		tenant_id: str,
		as_of_date: date,
	) -> PortfolioQuality:
		_guard_tenant_id(tenant_id)
		loans_raw = await self._loans.list_by_tenant(tenant_id)
		active = [l for l in loans_raw if l.get("status") not in ("closed",)]

		total_portfolio = sum(_d(l.get("outstanding_balance", 0)) for l in active)
		npl_amount      = sum(_d(l.get("outstanding_balance", 0)) for l in active if l.get("days_past_due", 0) >= 90)
		par_30_amount   = sum(_d(l.get("outstanding_balance", 0)) for l in active if l.get("days_past_due", 0) > 30)
		par_90_amount   = sum(_d(l.get("outstanding_balance", 0)) for l in active if l.get("days_past_due", 0) > 90)
		written_off     = sum(_d(l.get("write_off_amount", 0)) for l in loans_raw)
		recovered       = sum(_d(l.get("recovered_amount", 0)) for l in loans_raw)

		# Provisions
		provisions = await self._events.list_by_tenant(tenant_id)
		total_provisions = sum(
			_d(p.get("posted_provision", 0))
			for p in provisions if p.get("_type") == "provision"
		)

		def _ratio(n: Decimal, d: Decimal) -> Decimal:
			return _r4(n / d) if d > ZERO else ZERO

		by_class: dict[str, dict[str, Any]] = {}
		for cls in LoanClassification:
			subset = [l for l in active if l.get("classification") == cls.value]
			bal = sum(_d(l.get("outstanding_balance", 0)) for l in subset)
			by_class[cls.value] = {"count": len(subset), "balance": str(bal)}

		return PortfolioQuality(
			tenant_id=tenant_id,
			as_of_date=as_of_date,
			total_loans=len(active),
			total_portfolio=_r2(Decimal(str(total_portfolio))),
			npl_amount=_r2(Decimal(str(npl_amount))),
			npl_ratio=_ratio(Decimal(str(npl_amount)), Decimal(str(total_portfolio))),
			par_30_amount=_r2(Decimal(str(par_30_amount))),
			par_30_ratio=_ratio(Decimal(str(par_30_amount)), Decimal(str(total_portfolio))),
			par_90_amount=_r2(Decimal(str(par_90_amount))),
			par_90_ratio=_ratio(Decimal(str(par_90_amount)), Decimal(str(total_portfolio))),
			total_provisions=_r2(Decimal(str(total_provisions))),
			provision_coverage=_ratio(Decimal(str(total_provisions)), Decimal(str(npl_amount))),
			written_off_amount=_r2(Decimal(str(written_off))),
			recovered_amount=_r2(Decimal(str(recovered))),
			by_classification=by_class,
		)

	async def classify_loan(self, tenant_id: str, loan_id: str) -> LoanClassification:
		"""Return CBK/Basel classification for a loan based on current DPD."""
		_guard_tenant_id(tenant_id)
		loan = await self._load_loan(tenant_id, loan_id)
		return _classify_by_dpd(loan.days_past_due)

	async def calculate_required_provision(self, tenant_id: str, loan_id: str) -> Decimal:
		"""Return required provision per CBK provisioning matrix."""
		_guard_tenant_id(tenant_id)
		loan = await self._load_loan(tenant_id, loan_id)
		classification = _classify_by_dpd(loan.days_past_due)
		rate = CBK_PROVISION_RATES[classification]
		return _r2(loan.outstanding_balance * rate)

	async def post_provision_entry(
		self,
		tenant_id: str,
		loan_id: str,
		provision_amount: Decimal,
		posting_date: date,
	) -> dict[str, Any]:
		_guard_tenant_id(tenant_id)
		assert provision_amount >= ZERO

		loan = await self._load_loan(tenant_id, loan_id)
		classification = _classify_by_dpd(loan.days_past_due)
		rate = CBK_PROVISION_RATES[classification]

		gl_id = await self._post_gl(
			tenant_id=tenant_id,
			loan_id=loan_id,
			entry_type="provision",
			description=f"Provision [{classification.value}] {rate*100:.0f}%",
			dr_account="6100",  # Provision Expense
			cr_account="1290",  # Allowance for Loan Losses (contra-asset)
			amount=provision_amount,
			posting_date=posting_date,
		)

		prov = LoanProvision(
			loan_id=loan_id,
			tenant_id=tenant_id,
			classification=classification,
			outstanding_balance=loan.outstanding_balance,
			provision_rate=rate,
			required_provision=_r2(loan.outstanding_balance * rate),
			posted_provision=provision_amount,
			posting_date=posting_date,
			gl_entry_id=gl_id,
		)
		rec = prov.model_dump()
		rec["_type"] = "provision"
		await self._events.save(rec)
		return {"provision_id": prov.id, "gl_entry_id": gl_id, "posted_provision": str(provision_amount)}

	async def get_provision_report(
		self,
		tenant_id: str,
		as_of_date: date,
	) -> dict[str, Any]:
		_guard_tenant_id(tenant_id)
		events = await self._events.list_by_tenant(tenant_id)
		provisions = [e for e in events if e.get("_type") == "provision"]
		loans_raw  = await self._loans.list_by_tenant(tenant_id)

		rows = []
		for loan in loans_raw:
			if loan.get("status") in ("closed",):
				continue
			l_id = loan["id"]
			classification = _classify_by_dpd(loan.get("days_past_due", 0))
			rate = CBK_PROVISION_RATES[classification]
			outstanding = _d(loan.get("outstanding_balance", 0))
			required = _r2(outstanding * rate)
			posted_entries = [p for p in provisions if p.get("loan_id") == l_id]
			posted = _r2(sum(_d(p.get("posted_provision", 0)) for p in posted_entries))
			rows.append({
				"loan_id": l_id,
				"customer_id": loan.get("customer_id"),
				"outstanding_balance": str(outstanding),
				"classification": classification.value,
				"provision_rate": str(rate),
				"required_provision": str(required),
				"posted_provision": str(posted),
				"shortfall": str(max(ZERO, required - posted)),
			})

		total_required = sum(_d(r["required_provision"]) for r in rows)
		total_posted   = sum(_d(r["posted_provision"]) for r in rows)
		return {
			"tenant_id": tenant_id,
			"as_of_date": as_of_date.isoformat(),
			"total_required": str(_r2(Decimal(str(total_required)))),
			"total_posted":   str(_r2(Decimal(str(total_posted)))),
			"total_shortfall": str(_r2(Decimal(str(max(ZERO, Decimal(str(total_required)) - Decimal(str(total_posted))))))),
			"rows": rows,
		}

	# ── Collections / notices ─────────────────────────────────────────────────

	async def send_demand_notice(
		self,
		tenant_id: str,
		loan_id: str,
		notice_type: DemandNoticeType,
	) -> dict[str, Any]:
		_guard_tenant_id(tenant_id)
		loan = await self._load_loan(tenant_id, loan_id)

		subjects = {
			DemandNoticeType.REMINDER:      "Loan Payment Reminder",
			DemandNoticeType.FORMAL_DEMAND:  "Formal Demand Notice",
			DemandNoticeType.LEGAL:          "Legal Notice — Final Demand",
		}
		body = (
			f"Dear Customer {loan.customer_id},\n"
			f"Your loan {loan_id} has an outstanding balance of {loan.outstanding_balance} {loan.currency}.\n"
			f"Days past due: {loan.days_past_due}.\n"
			f"Please settle immediately to avoid further action."
		)
		await self._notify.send(
			recipient=loan.customer_id,
			channel="email",
			subject=subjects[notice_type],
			body=body,
			metadata={"loan_id": loan_id, "notice_type": notice_type.value},
		)

		notice_id = uuid7str()
		notice_rec = {
			"_type": "demand_notice",
			"id": notice_id,
			"loan_id": loan_id,
			"tenant_id": tenant_id,
			"notice_type": notice_type.value,
			"sent_at": _now_iso(),
		}
		await self._events.save(notice_rec)

		loan.last_notice_type = notice_type
		loan.last_notice_date = _now_iso()
		await self._save_loan(loan)

		return {"notice_id": notice_id, "notice_type": notice_type.value, "sent_at": notice_rec["sent_at"]}

	async def refer_to_collections(
		self,
		tenant_id: str,
		loan_id: str,
		referred_by: str,
		notes: str,
	) -> dict[str, Any]:
		_guard_tenant_id(tenant_id)
		loan = await self._load_loan(tenant_id, loan_id)
		loan.referred_to_collections = True
		loan.collections_referred_at  = _now_iso()
		loan.collections_referred_by  = referred_by
		loan.collections_notes        = notes
		await self._save_loan(loan)
		await self._audit.log_event(
			"referred_to_collections", referred_by, tenant_id, loan_id,
			{"notes": notes},
		)
		return {
			"loan_id": loan_id,
			"referred_by": referred_by,
			"referred_at": loan.collections_referred_at,
		}

	async def close_loan(
		self,
		tenant_id: str,
		loan_id: str,
		closure_date: date,
		reason: ClosureReason,
	) -> dict[str, Any]:
		_guard_tenant_id(tenant_id)
		loan = await self._load_loan(tenant_id, loan_id)
		loan.status        = LoanStatus.CLOSED
		loan.closure_date  = closure_date
		loan.closure_reason = reason
		await self._save_loan(loan)
		await self._audit.log_event(
			"loan_closed", "system", tenant_id, loan_id,
			{"reason": reason.value, "closure_date": closure_date.isoformat()},
		)
		return {"loan_id": loan_id, "status": "closed", "closure_reason": reason.value}

	async def get_early_settlement_amount(
		self,
		tenant_id: str,
		loan_id: str,
		settlement_date: date,
	) -> dict[str, Any]:
		"""Calculate early settlement: outstanding + any penalties, minus interest rebate."""
		_guard_tenant_id(tenant_id)
		loan = await self._load_loan(tenant_id, loan_id)
		installments = await self._schedules.get_installments(loan_id)

		future_interest = sum(
			_d(i["interest"])
			for i in installments
			if i.get("status") != "paid"
			and date.fromisoformat(str(i["due_date"])) > settlement_date
		)
		# 50% rebate on future unearned interest (configurable)
		rebate = _r2(Decimal(str(future_interest)) * Decimal("0.5"))
		outstanding = loan.outstanding_balance + loan.total_penalties + loan.total_fees
		settlement_amount = _r2(outstanding - rebate)

		return {
			"outstanding": str(outstanding),
			"future_interest": str(_r2(Decimal(str(future_interest)))),
			"rebate": str(rebate),
			"settlement_amount": str(max(ZERO, settlement_amount)),
		}

	async def reprice_loan(
		self,
		tenant_id: str,
		loan_id: str,
		new_rate: Decimal,
		effective_date: date,
		approved_by: str,
	) -> dict[str, Any]:
		"""Reprice (change interest rate) and regenerate schedule."""
		_guard_tenant_id(tenant_id)
		_guard_str(approved_by, "approved_by")
		assert new_rate > ZERO

		loan = await self._load_loan(tenant_id, loan_id)
		old_rate  = loan.rate
		loan.rate = new_rate

		remaining_installments = [
			i for i in await self._schedules.get_installments(loan_id)
			if i.get("status") != "paid"
		]
		remaining_tenor = len(remaining_installments)

		new_schedule = await self.generate_amortisation_schedule(
			loan_id=loan_id,
			principal=loan.outstanding_balance,
			rate=new_rate,
			tenor_months=remaining_tenor,
			method=loan.method,
			first_payment_date=_add_months(effective_date, 1),
		)

		await self._save_loan(loan)
		ev = {
			"_type": "reprice",
			"loan_id": loan_id,
			"tenant_id": tenant_id,
			"old_rate": str(old_rate),
			"new_rate": str(new_rate),
			"effective_date": effective_date.isoformat(),
			"approved_by": approved_by,
		}
		await self._events.save(ev)
		await self._audit.log_event(
			"loan_repriced", approved_by, tenant_id, loan_id,
			{"old_rate": str(old_rate), "new_rate": str(new_rate)},
		)

		return {
			"loan_id": loan_id,
			"old_rate": str(old_rate),
			"new_rate": str(new_rate),
			"effective_date": effective_date.isoformat(),
			"new_schedule": new_schedule,
		}

	async def batch_calculate_arrears(
		self,
		tenant_id: str,
		as_of_date: date,
	) -> dict[str, Any]:
		"""Nightly arrears run — idempotent.  Updates all active loans."""
		_guard_tenant_id(tenant_id)
		loans_raw = await self._loans.list_by_tenant(tenant_id)
		processed = 0
		errors = 0
		npa_count = 0

		for loan_raw in loans_raw:
			if loan_raw.get("status") in ("closed", "written_off", "recovered"):
				continue
			try:
				arrears = await self.calculate_arrears(tenant_id, loan_raw["id"], as_of_date)
				if arrears.npa_status:
					npa_count += 1
				processed += 1
			except Exception as exc:
				log.warning(_log_pretty_path("batch_arrears_error", tenant_id, loan_raw["id"]), exc_info=exc)
				errors += 1

		log.info(_log_pretty_path(f"batch_arrears processed={processed} errors={errors}", tenant_id, as_of_date.isoformat()))
		return {
			"tenant_id": tenant_id,
			"as_of_date": as_of_date.isoformat(),
			"processed": processed,
			"errors": errors,
			"npa_count": npa_count,
		}

	def health_check(self) -> dict[str, Any]:
		return {
			"service": "fin_lms",
			"status": "healthy",
			"version": "1.1.0",
			"adapters": {
				"auth": type(self._auth).__name__,
				"audit": type(self._audit).__name__,
				"notify": type(self._notify).__name__,
				"gl": type(self._gl).__name__,
			},
		}

	# ── Improvement I1: Partial prepayment with configurable strategy ─────────

	async def prepay_with_options(
		self,
		tenant_id: str,
		loan_id: str,
		amount: Decimal,
		prepay_date: date,
		payment_ref: str,
		strategy: str = "reduce_tenor",
	) -> dict[str, Any]:
		"""Partial or full prepayment with explicit strategy for schedule rebuilding.

		strategy:
		  "reduce_tenor"      — advance future principal, shorten remaining term
		  "reduce_instalment" — keep tenor, recalculate lower PMT
		  "advance_next"      — apply surplus to next installment(s) only
		"""
		guard_tenant_id(tenant_id)
		assert amount > ZERO, "prepayment amount must be positive"
		assert strategy in ("reduce_tenor", "reduce_instalment", "advance_next"), \
			f"unknown strategy {strategy!r}"

		loan = await self._load_loan(tenant_id, loan_id)
		if loan.status in (LoanStatus.CLOSED, LoanStatus.WRITTEN_OFF):
			raise ValueError(f"Loan {loan_id} is {loan.status.value} — cannot prepay")

		# Apply waterfall first (handles penalties/fees/interest portion)
		waterfall_result = await self.record_repayment(
			tenant_id=tenant_id,
			loan_id=loan_id,
			amount=amount,
			payment_date=prepay_date,
			payment_ref=payment_ref,
			payment_method=PaymentMethod.BANK_TRANSFER,
		)
		# Re-load after waterfall
		loan = await self._load_loan(tenant_id, loan_id)

		if loan.status == LoanStatus.CLOSED:
			log.info(_log_pretty_path("prepay_full_closure", tenant_id, loan_id))
			return {**waterfall_result, "strategy": strategy, "new_schedule": [], "closed": True}

		installments = [
			i for i in await self._schedules.get_installments(loan_id)
			if i.get("status") != "paid"
		]
		remaining_tenor = len(installments)

		if strategy == "reduce_tenor" or strategy == "advance_next":
			# Re-amortise remaining balance over current remaining tenor
			new_schedule = await self.generate_amortisation_schedule(
				loan_id=loan_id,
				principal=loan.outstanding_balance,
				rate=loan.rate,
				tenor_months=max(1, remaining_tenor),
				method=loan.method,
				first_payment_date=_add_months(prepay_date, 1),
			)
		else:  # reduce_instalment — same tenor but lower PMT
			new_schedule = await self.generate_amortisation_schedule(
				loan_id=loan_id,
				principal=loan.outstanding_balance,
				rate=loan.rate,
				tenor_months=max(1, remaining_tenor),
				method=AmortisationMethod.FRENCH_ANNUITY,
				first_payment_date=_add_months(prepay_date, 1),
			)

		await self._audit.log_event(
			"prepayment_with_strategy", "system", tenant_id, loan_id,
			{"amount": str(amount), "strategy": strategy, "ref": payment_ref},
		)
		log.info(_log_pretty_path(f"prepay strategy={strategy}", tenant_id, loan_id))

		return {
			"loan_id": loan_id,
			"prepaid_amount": str(amount),
			"strategy": strategy,
			"remaining_balance": str(loan.outstanding_balance),
			"new_schedule": new_schedule,
			"waterfall": waterfall_result["allocated"],
		}

	# ── Improvement I2: Daily interest accrual engine ────────────────────────

	async def accrue_daily_interest(
		self,
		tenant_id: str,
		as_of_date: date,
	) -> dict[str, Any]:
		"""IFRS 9 daily interest accrual batch.

		For every active loan, computes `balance × EIR / 365` for each day since
		the loan's `last_accrual_date` and posts:
		  DR Accrued Interest Receivable (1210)
		  CR Interest Income              (4100)

		Idempotent: skips loans where `last_accrual_date >= as_of_date`.
		"""
		guard_tenant_id(tenant_id)
		loans_raw = await self._loans.list_by_tenant(tenant_id)

		accrued_count = 0
		total_accrued = ZERO
		errors = 0

		for lr in loans_raw:
			if lr.get("status") in ("closed", "written_off", "recovered"):
				continue
			try:
				loan = await self._load_loan(tenant_id, lr["id"])
				last_accrual_raw = lr.get("last_accrual_date")
				if last_accrual_raw:
					last_accrual = date.fromisoformat(str(last_accrual_raw))
				else:
					last_accrual = loan.disbursement_date or as_of_date
				if last_accrual >= as_of_date:
					continue  # already accrued up to date

				days = (as_of_date - last_accrual).days
				if days <= 0:
					continue

				daily_rate = _r4(loan.rate / Decimal("365"))
				accrual_amount = _r2(loan.outstanding_balance * daily_rate * Decimal(str(days)))
				if accrual_amount <= ZERO:
					continue

				gl_id = await self._post_gl(
					tenant_id=tenant_id,
					loan_id=loan.id,
					entry_type="interest_accrual",
					description=f"Daily interest accrual {last_accrual.isoformat()} to {as_of_date.isoformat()}",
					dr_account="1210",  # Accrued Interest Receivable
					cr_account="4100",  # Interest Income
					amount=accrual_amount,
					posting_date=as_of_date,
				)

				# Persist last_accrual_date on the loan record
				loan_dict = loan.model_dump()
				loan_dict["last_accrual_date"] = as_of_date.isoformat()
				loan_dict["accrued_interest"] = str(
					_d(loan_dict.get("accrued_interest") or 0) + accrual_amount
				)
				await self._loans.save(loan_dict)

				total_accrued += accrual_amount
				accrued_count += 1
				log.info(_log_pretty_path(
					f"accrual days={days} amount={accrual_amount}", tenant_id, loan.id
				))
			except Exception as exc:
				log.warning(_log_pretty_path("accrual_error", tenant_id, lr["id"]), exc_info=exc)
				errors += 1

		log.info(_log_pretty_path(
			f"accrue_daily_interest processed={accrued_count} errors={errors} total={total_accrued}",
			tenant_id, as_of_date.isoformat(),
		))
		return {
			"tenant_id": tenant_id,
			"as_of_date": as_of_date.isoformat(),
			"accrued_count": accrued_count,
			"total_accrued": str(_r2(total_accrued)),
			"errors": errors,
		}

	# ── Improvement I3: EIR / XIRR calculation ───────────────────────────────

	async def calculate_eir(
		self,
		tenant_id: str,
		loan_id: str,
		origination_fees: Decimal = ZERO,
		transaction_costs: Decimal = ZERO,
	) -> dict[str, Any]:
		"""Calculate Effective Interest Rate (EIR) per IFRS 9 amortised cost.

		Solves for the rate r that satisfies:
		  net_proceeds = sum(CF_t / (1+r)^t)
		where net_proceeds = principal - origination_fees - transaction_costs
		and CF_t are the scheduled installment cashflows.

		Uses Newton-Raphson iteration (max 200 iterations, tol=1e-8).
		"""
		guard_tenant_id(tenant_id)
		loan = await self._load_loan(tenant_id, loan_id)
		installments = await self._schedules.get_installments(loan_id)

		net_proceeds = loan.principal - origination_fees - transaction_costs
		assert net_proceeds > ZERO, "net_proceeds must be positive after fees"

		cashflows: list[Decimal] = [_d(i["total"]) for i in installments]
		n = len(cashflows)
		if n == 0:
			raise ValueError(f"No schedule found for loan {loan_id}")

		# Newton-Raphson solver on monthly rate r_m
		r_m = loan.rate / Decimal("12")  # initial guess = contractual monthly rate
		tol = Decimal("1e-8")
		for _iteration in range(200):
			pv = ZERO
			dpv = ZERO
			for t, cf in enumerate(cashflows, start=1):
				discount = (ONE + r_m) ** t
				pv  += cf / discount
				dpv -= Decimal(str(t)) * cf / ((ONE + r_m) ** (t + 1))
			f  = pv - net_proceeds
			if dpv == ZERO:
				break
			r_m_new = r_m - f / dpv
			if abs(r_m_new - r_m) < tol:
				r_m = r_m_new
				break
			r_m = r_m_new

		eir_annual = _r4((ONE + r_m) ** 12 - ONE)
		total_interest = sum(cashflows) - loan.principal
		total_cost_of_credit = total_interest + origination_fees + transaction_costs

		ev = {
			"_type": "eir_calculation",
			"loan_id": loan_id,
			"tenant_id": tenant_id,
			"eir": str(eir_annual),
			"origination_fees": str(origination_fees),
			"transaction_costs": str(transaction_costs),
			"total_cost_of_credit": str(_r2(total_cost_of_credit)),
			"calculated_at": _now_iso(),
		}
		await self._events.save(ev)
		log.info(_log_pretty_path(f"EIR={eir_annual:.4%}", tenant_id, loan_id))

		return {
			"loan_id": loan_id,
			"contractual_rate": str(loan.rate),
			"eir_annual": str(eir_annual),
			"net_proceeds": str(_r2(net_proceeds)),
			"total_interest": str(_r2(total_interest)),
			"origination_fees": str(origination_fees),
			"transaction_costs": str(transaction_costs),
			"total_cost_of_credit": str(_r2(total_cost_of_credit)),
		}

	# ── Improvement I4: IFRS 9 ECL stage bucketing ──────────────────────────

	async def compute_ecl_provision(
		self,
		tenant_id: str,
		loan_id: str,
		pd: Decimal,
		lgd: Decimal,
		ead: Decimal | None = None,
		stage: int = 1,
	) -> dict[str, Any]:
		"""Compute IFRS 9 Expected Credit Loss (ECL) provision.

		Stage 1 → 12-month ECL = PD_12m × LGD × EAD
		Stage 2/3 → Lifetime ECL: sum PD_t × LGD × EAD discounted at EIR over
		            remaining cashflows (simplified: use remaining installments).

		pd  — Probability of Default (annual, decimal e.g. 0.05 = 5%)
		lgd — Loss Given Default (decimal e.g. 0.40 = 40%)
		ead — Exposure at Default; defaults to outstanding_balance
		stage — IFRS 9 stage (1, 2, or 3)
		"""
		guard_tenant_id(tenant_id)
		assert stage in (1, 2, 3), "stage must be 1, 2, or 3"
		assert ZERO <= pd <= ONE, "pd must be between 0 and 1"
		assert ZERO <= lgd <= ONE, "lgd must be between 0 and 1"

		loan = await self._load_loan(tenant_id, loan_id)
		if ead is None:
			ead = loan.outstanding_balance

		installments = [
			i for i in await self._schedules.get_installments(loan_id)
			if i.get("status") != "paid"
		]

		if stage == 1:
			# 12-month PD only
			pd_12m = ONE - (ONE - pd) ** (Decimal("1") / Decimal("12"))  # monthly PD
			ecl = _r2(pd_12m * Decimal("12") * lgd * ead)
		else:
			# Lifetime ECL — sum over remaining period with monthly PD
			pd_monthly = ONE - (ONE - pd) ** (Decimal("1") / Decimal("12"))
			eir_monthly = loan.rate / Decimal("12")
			ecl = ZERO
			survival = ONE  # probability of not having defaulted yet
			for t, inst in enumerate(installments, start=1):
				cf_ead = _d(inst.get("balance", ead))
				default_prob_t = survival * pd_monthly
				discount = (ONE + eir_monthly) ** t
				ecl += _r4(default_prob_t * lgd * cf_ead / discount)
				survival *= (ONE - pd_monthly)
			ecl = _r2(ecl)

		stage_label = {1: "12-month ECL", 2: "Lifetime ECL (Stage 2)", 3: "Lifetime ECL (Stage 3)"}[stage]

		ev = {
			"_type": "ecl_provision",
			"loan_id": loan_id,
			"tenant_id": tenant_id,
			"stage": stage,
			"pd": str(pd),
			"lgd": str(lgd),
			"ead": str(ead),
			"ecl": str(ecl),
			"calculated_at": _now_iso(),
		}
		await self._events.save(ev)
		log.info(_log_pretty_path(f"ECL stage={stage} ecl={ecl}", tenant_id, loan_id))

		return {
			"loan_id": loan_id,
			"stage": stage,
			"stage_label": stage_label,
			"pd": str(pd),
			"lgd": str(lgd),
			"ead": str(ead),
			"ecl": str(ecl),
			"outstanding_balance": str(loan.outstanding_balance),
		}

	# ── Improvement I7: Loan top-up / additional drawdown ────────────────────

	async def topup_loan(
		self,
		tenant_id: str,
		loan_id: str,
		additional_amount: Decimal,
		topup_date: date,
		approved_by: str,
		approved_limit: Decimal | None = None,
		disbursement_ref: str | None = None,
	) -> dict[str, Any]:
		"""Additional drawdown on an existing facility (top-up / revolving credit).

		Validates additional_amount against approved_limit if supplied,
		adds to outstanding_balance, regenerates schedule, posts GL.
		"""
		guard_tenant_id(tenant_id)
		_guard_str(approved_by, "approved_by")
		assert additional_amount > ZERO, "top-up amount must be positive"

		loan = await self._load_loan(tenant_id, loan_id)
		if loan.status in (LoanStatus.CLOSED, LoanStatus.WRITTEN_OFF):
			raise ValueError(f"Loan {loan_id} is {loan.status.value} — cannot top-up")

		if approved_limit is not None:
			available = approved_limit - loan.outstanding_balance
			if additional_amount > available:
				raise ValueError(
					f"Top-up {additional_amount} exceeds available facility headroom {available}"
				)

		topup_ref = disbursement_ref or f"TOPUP-{uuid7str()[:8].upper()}"
		old_balance = loan.outstanding_balance
		loan.outstanding_balance = _r2(old_balance + additional_amount)
		loan.principal            = _r2(loan.principal + additional_amount)

		installments = [
			i for i in await self._schedules.get_installments(loan_id)
			if i.get("status") != "paid"
		]
		remaining_tenor = max(1, len(installments))

		new_schedule = await self.generate_amortisation_schedule(
			loan_id=loan_id,
			principal=loan.outstanding_balance,
			rate=loan.rate,
			tenor_months=remaining_tenor,
			method=loan.method,
			first_payment_date=_add_months(topup_date, 1),
		)

		gl_id = await self._post_gl(
			tenant_id=tenant_id,
			loan_id=loan_id,
			entry_type="topup_disbursement",
			description=f"Loan top-up {topup_ref} approved by {approved_by}",
			dr_account="1200",  # Loans Receivable
			cr_account="2100",  # Customer Account
			amount=additional_amount,
			posting_date=topup_date,
			ref=topup_ref,
		)

		ev = {
			"_type": "topup",
			"loan_id": loan_id,
			"tenant_id": tenant_id,
			"additional_amount": str(additional_amount),
			"old_balance": str(old_balance),
			"new_balance": str(loan.outstanding_balance),
			"topup_date": topup_date.isoformat(),
			"approved_by": approved_by,
			"gl_entry_id": gl_id,
		}
		await self._events.save(ev)
		await self._save_loan(loan)
		await self._audit.log_event(
			"loan_topup", approved_by, tenant_id, loan_id,
			{"additional_amount": str(additional_amount), "ref": topup_ref},
		)
		log.info(_log_pretty_path(
			f"topup +{additional_amount} new_balance={loan.outstanding_balance}", tenant_id, loan_id
		))

		return {
			"loan_id": loan_id,
			"topup_ref": topup_ref,
			"additional_amount": str(additional_amount),
			"new_outstanding_balance": str(loan.outstanding_balance),
			"gl_entry_id": gl_id,
			"new_schedule": new_schedule,
		}

	# ── Improvement I8: Collateral registration and coverage ─────────────────

	async def register_collateral(
		self,
		tenant_id: str,
		loan_id: str,
		collateral_type: str,
		market_value: Decimal,
		fsv: Decimal,
		haircut_rate: Decimal,
		valuation_date: date,
		description: str = "",
	) -> dict[str, Any]:
		"""Register collateral securing a loan.

		collateral_type — e.g. "land_title", "motor_vehicle", "shares", "cash_deposit"
		market_value    — current open-market value
		fsv             — Forced Sale Value (typically 70–80% of market)
		haircut_rate    — additional regulatory haircut (e.g. 0.20 for shares)
		valuation_date  — date of last formal valuation

		Net collateral value = fsv × (1 - haircut_rate)
		"""
		guard_tenant_id(tenant_id)
		_guard_str(collateral_type, "collateral_type")
		assert market_value > ZERO, "market_value must be positive"
		assert ZERO <= fsv <= market_value, "fsv must be <= market_value"
		assert ZERO <= haircut_rate < ONE, "haircut_rate must be in [0, 1)"

		# Verify loan exists
		await self._load_loan(tenant_id, loan_id)

		net_collateral_value = _r2(fsv * (ONE - haircut_rate))
		collateral_id = uuid7str()

		rec = {
			"_type": "collateral",
			"id": collateral_id,
			"loan_id": loan_id,
			"tenant_id": tenant_id,
			"collateral_type": collateral_type,
			"description": description,
			"market_value": str(market_value),
			"fsv": str(fsv),
			"haircut_rate": str(haircut_rate),
			"net_collateral_value": str(net_collateral_value),
			"valuation_date": valuation_date.isoformat(),
			"created_at": _now_iso(),
		}
		await self._events.save(rec)
		await self._audit.log_event(
			"collateral_registered", "system", tenant_id, loan_id,
			{"collateral_type": collateral_type, "net_value": str(net_collateral_value)},
		)
		log.info(_log_pretty_path(
			f"collateral {collateral_type} net={net_collateral_value}", tenant_id, loan_id
		))

		return {
			"collateral_id": collateral_id,
			"loan_id": loan_id,
			"collateral_type": collateral_type,
			"market_value": str(market_value),
			"fsv": str(fsv),
			"net_collateral_value": str(net_collateral_value),
		}

	async def get_collateral_coverage(
		self,
		tenant_id: str,
		loan_id: str,
	) -> dict[str, Any]:
		"""Compute collateral coverage ratio and net-of-collateral exposure.

		Net exposure = max(0, outstanding_balance - total_net_collateral_value)
		Coverage ratio = total_net_collateral_value / outstanding_balance
		"""
		guard_tenant_id(tenant_id)
		loan = await self._load_loan(tenant_id, loan_id)
		events = await self._events.list_by_tenant(tenant_id)

		collaterals = [
			e for e in events
			if e.get("_type") == "collateral" and e.get("loan_id") == loan_id
		]
		total_ncv = _r2(sum(_d(c.get("net_collateral_value", 0)) for c in collaterals))
		outstanding = loan.outstanding_balance

		def _ratio(n: Decimal, d: Decimal) -> Decimal:
			return _r4(n / d) if d > ZERO else ZERO

		net_exposure = _r2(max(ZERO, outstanding - total_ncv))
		coverage_ratio = _ratio(total_ncv, outstanding)

		return {
			"loan_id": loan_id,
			"outstanding_balance": str(outstanding),
			"total_net_collateral_value": str(total_ncv),
			"coverage_ratio": str(coverage_ratio),
			"net_exposure": str(net_exposure),
			"collateral_count": len(collaterals),
			"collaterals": [
				{
					"id": c["id"],
					"type": c["collateral_type"],
					"market_value": c["market_value"],
					"net_collateral_value": c["net_collateral_value"],
					"valuation_date": c["valuation_date"],
				}
				for c in collaterals
			],
		}

	# ── Improvement I9: Collections escalation ladder ────────────────────────

	async def run_collections_escalation(
		self,
		tenant_id: str,
		as_of_date: date,
		policy: dict[int, str] | None = None,
	) -> dict[str, Any]:
		"""Automated collections escalation ladder for all in-arrears loans.

		Default DPD → action mapping (override via `policy`):
		  DPD  5 → REMINDER
		  DPD 15 → FORMAL_DEMAND
		  DPD 30 → LEGAL (formal legal notice)
		  DPD 60 → refer_to_collections
		  DPD 90 → NPA (no additional notice, already classified)

		Idempotent: compares loan's `last_notice_type` against required stage
		and only escalates if the new stage is strictly higher.
		"""
		guard_tenant_id(tenant_id)

		_default_policy: dict[int, str] = {
			5:  "REMINDER",
			15: "FORMAL_DEMAND",
			30: "LEGAL",
			60: "COLLECTIONS_REFERRAL",
			90: "NPA",
		}
		active_policy = policy or _default_policy
		escalation_order = ["REMINDER", "FORMAL_DEMAND", "LEGAL", "COLLECTIONS_REFERRAL", "NPA"]

		loans_raw = await self._loans.list_by_tenant(tenant_id)
		escalated = 0
		skipped   = 0
		errors    = 0

		for lr in loans_raw:
			if lr.get("status") in ("closed", "written_off", "recovered"):
				continue
			dpd = int(lr.get("days_past_due") or 0)
			if dpd == 0:
				continue

			# Determine required action for this DPD
			required_action: str | None = None
			for threshold in sorted(active_policy.keys(), reverse=True):
				if dpd >= threshold:
					required_action = active_policy[threshold]
					break
			if required_action is None:
				skipped += 1
				continue

			# Check current escalation level
			current_notice = lr.get("last_notice_type")
			current_level = escalation_order.index(current_notice) if current_notice in escalation_order else -1
			required_level = escalation_order.index(required_action) if required_action in escalation_order else -1

			if required_level <= current_level:
				skipped += 1
				continue

			try:
				loan_id = lr["id"]
				if required_action == "COLLECTIONS_REFERRAL":
					await self.refer_to_collections(
						tenant_id=tenant_id,
						loan_id=loan_id,
						referred_by="collections_escalation_bot",
						notes=f"Auto-escalated at DPD {dpd} on {as_of_date.isoformat()}",
					)
				elif required_action != "NPA":
					nt_map = {
						"REMINDER": DemandNoticeType.REMINDER,
						"FORMAL_DEMAND": DemandNoticeType.FORMAL_DEMAND,
						"LEGAL": DemandNoticeType.LEGAL,
					}
					await self.send_demand_notice(
						tenant_id=tenant_id,
						loan_id=loan_id,
						notice_type=nt_map[required_action],
					)
				escalated += 1
				log.info(_log_pretty_path(
					f"escalated dpd={dpd} action={required_action}", tenant_id, loan_id
				))
			except Exception as exc:
				log.warning(_log_pretty_path("escalation_error", tenant_id, lr["id"]), exc_info=exc)
				errors += 1

		log.info(_log_pretty_path(
			f"collections_escalation escalated={escalated} skipped={skipped} errors={errors}",
			tenant_id, as_of_date.isoformat(),
		))
		return {
			"tenant_id": tenant_id,
			"as_of_date": as_of_date.isoformat(),
			"escalated": escalated,
			"skipped": skipped,
			"errors": errors,
		}

	# ── Improvement I10: Fee schedule engine ─────────────────────────────────

	async def apply_fee(
		self,
		tenant_id: str,
		loan_id: str,
		fee_type: str,
		amount: Decimal,
		due_date: date,
		description: str = "",
		defer_to_eir: bool = False,
	) -> dict[str, Any]:
		"""Apply a structured fee to a loan.

		fee_type values: ORIGINATION, PROCESSING, ANNUAL_FACILITY, EXIT, INSURANCE
		defer_to_eir — if True, the fee is deferred (IFRS 9 integral) and amortised
		               over the loan life; if False, it is expensed immediately.

		GL for immediate fee:
		  DR Customer Account (2100) / CR Fee Income (4300)
		GL for deferred fee:
		  DR Deferred Fee Asset (1250) / CR Fee Income (4300)  [amortised over tenor]
		"""
		guard_tenant_id(tenant_id)
		_guard_str(fee_type, "fee_type")
		assert amount > ZERO, "fee amount must be positive"

		loan = await self._load_loan(tenant_id, loan_id)
		fee_id = uuid7str()

		dr_account = "1250" if defer_to_eir else "2100"
		cr_account = "4300"  # Fee Income

		gl_id = await self._post_gl(
			tenant_id=tenant_id,
			loan_id=loan_id,
			entry_type="fee",
			description=description or f"{fee_type} fee",
			dr_account=dr_account,
			cr_account=cr_account,
			amount=amount,
			posting_date=due_date,
		)

		fee_rec = {
			"_type": "fee",
			"id": fee_id,
			"loan_id": loan_id,
			"tenant_id": tenant_id,
			"fee_type": fee_type,
			"amount": str(amount),
			"due_date": due_date.isoformat(),
			"deferred": defer_to_eir,
			"amortised_amount": "0",
			"description": description,
			"gl_entry_id": gl_id,
			"created_at": _now_iso(),
		}
		await self._events.save(fee_rec)

		# Add to loan's total_fees for waterfall clearing
		if not defer_to_eir:
			loan.total_fees += amount
		await self._save_loan(loan)

		await self._audit.log_event(
			"fee_applied", "system", tenant_id, loan_id,
			{"fee_type": fee_type, "amount": str(amount), "deferred": defer_to_eir},
		)
		log.info(_log_pretty_path(f"fee {fee_type}={amount} deferred={defer_to_eir}", tenant_id, loan_id))

		return {
			"fee_id": fee_id,
			"loan_id": loan_id,
			"fee_type": fee_type,
			"amount": str(amount),
			"deferred": defer_to_eir,
			"gl_entry_id": gl_id,
		}

	# ── Improvement I13: Regulatory reporting pack ───────────────────────────

	async def generate_cbk_loan_register(
		self,
		tenant_id: str,
		reporting_date: date,
	) -> dict[str, Any]:
		"""Generate CBK Form CBK-LR1 loan register for statutory submission.

		Each row contains the mandatory CBK fields:
		  loan_id, customer_id, product_code, outstanding_balance,
		  classification, days_past_due, provision_rate, required_provision,
		  currency, disbursement_date, maturity_date, status
		"""
		guard_tenant_id(tenant_id)
		loans_raw = await self._loans.list_by_tenant(tenant_id)
		events = await self._events.list_by_tenant(tenant_id)

		rows: list[dict[str, Any]] = []
		total_portfolio = ZERO
		total_provisions_required = ZERO

		for lr in loans_raw:
			outstanding = _d(lr.get("outstanding_balance", 0))
			dpd         = int(lr.get("days_past_due") or 0)
			classification = _classify_by_dpd(dpd)
			prov_rate   = CBK_PROVISION_RATES[classification]
			required    = _r2(outstanding * prov_rate)

			# Posted provision for this loan
			loan_provisions = [
				e for e in events
				if e.get("_type") == "provision" and e.get("loan_id") == lr["id"]
			]
			posted = _r2(sum(_d(p.get("posted_provision", 0)) for p in loan_provisions))

			row = {
				"loan_id":             lr["id"],
				"customer_id":         lr.get("customer_id"),
				"product_code":        lr.get("product_code"),
				"currency":            lr.get("currency", "KES"),
				"outstanding_balance": str(outstanding),
				"days_past_due":       dpd,
				"classification":      classification.value,
				"provision_rate":      str(prov_rate),
				"required_provision":  str(required),
				"posted_provision":    str(posted),
				"provision_shortfall": str(max(ZERO, required - posted)),
				"status":              lr.get("status"),
				"disbursement_date":   str(lr.get("disbursement_date") or ""),
				"maturity_date":       str(lr.get("maturity_date") or ""),
				"npa_flag":            1 if dpd >= 90 else 0,
			}
			rows.append(row)
			total_portfolio          += outstanding
			total_provisions_required += required

		# Data quality checks
		dq_issues: list[str] = []
		for r in rows:
			if not r["customer_id"]:
				dq_issues.append(f"loan {r['loan_id']}: missing customer_id")
			if not r["disbursement_date"]:
				dq_issues.append(f"loan {r['loan_id']}: missing disbursement_date")

		log.info(_log_pretty_path(
			f"CBK_LR1 loans={len(rows)} total={_r2(total_portfolio)}",
			tenant_id, reporting_date.isoformat(),
		))

		return {
			"form":              "CBK-LR1",
			"tenant_id":         tenant_id,
			"reporting_date":    reporting_date.isoformat(),
			"total_loans":       len(rows),
			"total_portfolio":   str(_r2(total_portfolio)),
			"total_provisions_required": str(_r2(total_provisions_required)),
			"data_quality_issues": dq_issues,
			"rows":              rows,
		}

	# ── Improvement I14: Multi-currency FX revaluation ───────────────────────

	async def revalue_fx_loan(
		self,
		tenant_id: str,
		loan_id: str,
		spot_rate: Decimal,
		revaluation_date: date,
		base_currency: str = "KES",
	) -> dict[str, Any]:
		"""Revalue a foreign-currency loan to the base currency at spot rate.

		DR/CR Loans Receivable Translated (1200)  by delta
		CR/DR FX Translation Reserve (3100)        by delta

		If kes_equivalent increases (currency appreciated): DR 1200 / CR 3100 (FX gain)
		If kes_equivalent decreases (currency depreciated): DR 3100 / CR 1200 (FX loss)
		"""
		guard_tenant_id(tenant_id)
		assert spot_rate > ZERO, "spot_rate must be positive"

		loan = await self._load_loan(tenant_id, loan_id)
		if loan.currency == base_currency:
			return {
				"loan_id": loan_id,
				"message": f"Loan is already denominated in {base_currency} — no revaluation required",
				"fx_gain_loss": "0",
			}

		loan_dict = loan.model_dump()
		prior_rate = _d(loan_dict.get("last_revaluation_rate") or spot_rate)
		prior_kes  = _r2(loan.outstanding_balance * prior_rate)
		new_kes    = _r2(loan.outstanding_balance * spot_rate)
		delta      = _r2(new_kes - prior_kes)

		if delta != ZERO:
			gain = delta > ZERO
			gl_id = await self._post_gl(
				tenant_id=tenant_id,
				loan_id=loan_id,
				entry_type="fx_revaluation",
				description=(
					f"FX revaluation {loan.currency}/{base_currency} "
					f"@ {spot_rate} on {revaluation_date.isoformat()}"
				),
				dr_account="1200" if gain else "3100",   # Loans Receivable or FX Reserve
				cr_account="3100" if gain else "1200",   # FX Reserve or Loans Receivable
				amount=abs(delta),
				posting_date=revaluation_date,
				currency=loan.currency,
			)
		else:
			gl_id = None

		# Persist revaluation state on the loan record
		loan_dict["last_revaluation_rate"] = str(spot_rate)
		loan_dict["last_revaluation_date"] = revaluation_date.isoformat()
		loan_dict["kes_equivalent"]        = str(new_kes)
		await self._loans.save(loan_dict)

		await self._audit.log_event(
			"fx_revaluation", "system", tenant_id, loan_id,
			{"spot_rate": str(spot_rate), "delta_kes": str(delta)},
		)
		log.info(_log_pretty_path(
			f"FX revalue {loan.currency}→{base_currency} rate={spot_rate} delta={delta}",
			tenant_id, loan_id,
		))

		return {
			"loan_id":          loan_id,
			"loan_currency":    loan.currency,
			"base_currency":    base_currency,
			"spot_rate":        str(spot_rate),
			"prior_rate":       str(prior_rate),
			"outstanding_ccy":  str(loan.outstanding_balance),
			"prior_kes":        str(prior_kes),
			"new_kes":          str(new_kes),
			"fx_gain_loss":     str(delta),
			"gain_or_loss":     "gain" if delta > ZERO else ("loss" if delta < ZERO else "nil"),
			"gl_entry_id":      gl_id,
		}
