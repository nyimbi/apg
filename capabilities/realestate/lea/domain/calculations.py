"""Financial and domain calculations for Lease Management (IFRS 16 / ASC 842).

All functions are pure (no I/O, no side-effects).  Inputs and outputs use
:class:`decimal.Decimal` for monetary precision.

Key formulae
------------
Lease Liability (initial)      PV = Σ  Pₜ / (1 + r)ᵗ   for t = 1…n
ROU Asset (initial)            = PV + IDC + prepaid payments − lease incentives + ARO
Interest expense (period t)    = opening_balance × periodic_rate
Principal reduction (period t) = payment − interest
Depreciation (period t)        = carrying_amount / remaining_periods  (straight-line)
CPI-adjusted payment           = base_payment × (current_CPI / base_CPI)
Modification gain/loss         = new_liability − (old_liability × partial_proportion)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from typing import Any


# ── Helpers ───────────────────────────────────────────────────────────────────

_CENT = Decimal("0.01")
_PLACES6 = Decimal("0.000001")


def _q(value: Decimal, places: Decimal = _CENT) -> Decimal:
	"""Quantise to a given number of decimal places."""
	return value.quantize(places, rounding=ROUND_HALF_UP)


def _periodic_rate(annual_rate_pct: Decimal, periods_per_year: int) -> Decimal:
	"""Convert annual % rate to periodic decimal rate.

	Args:
		annual_rate_pct: Rate as percentage (e.g. Decimal("5.5") for 5.5 % p.a.)
		periods_per_year: Number of payment periods per year (12 for monthly).

	Returns:
		Periodic rate as a decimal fraction.
	"""
	return (annual_rate_pct / Decimal("100")) / Decimal(str(periods_per_year))


def _months_between(start: date, end: date) -> int:
	"""Whole months from start (inclusive) to end (exclusive)."""
	return max(1, (end.year - start.year) * 12 + (end.month - start.month))


def _periods_per_year(payment_frequency: str) -> int:
	"""Map payment frequency string to number of periods per year."""
	_map = {
		"monthly": 12,
		"quarterly": 4,
		"semi_annual": 2,
		"annual": 1,
		"in_advance": 12,
		"in_arrears": 12,
	}
	return _map.get(payment_frequency, 12)


def _advance_month(d: date, n: int) -> date:
	"""Return date advanced by n calendar months, clamped to month end."""
	import calendar
	month = d.month + n
	year = d.year + (month - 1) // 12
	month = (month - 1) % 12 + 1
	last_day = calendar.monthrange(year, month)[1]
	return d.replace(year=year, month=month, day=min(d.day, last_day))


# ── Core PV Engine ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AmortisationRow:
	"""Single row in a lease liability amortisation schedule."""
	period: int
	due_date: date | None
	opening_balance: Decimal
	payment: Decimal
	interest: Decimal
	principal: Decimal
	closing_balance: Decimal
	cumulative_interest: Decimal


def calculate_lease_liability(
	payment: Decimal,
	n_periods: int,
	annual_rate_pct: Decimal,
	periods_per_year: int = 12,
	in_advance: bool = False,
) -> Decimal:
	"""Compute the present value of a level annuity.

	Args:
		payment: Periodic payment amount (Decimal).
		n_periods: Total number of periods.
		annual_rate_pct: IBR or implicit rate as percentage p.a.
		periods_per_year: Payment periods per year.
		in_advance: True if payments are made at the start of each period
		            (annuity-due); False for payments at end (annuity-immediate).

	Returns:
		Present value (lease liability at commencement).

	Examples:
		>>> calculate_lease_liability(Decimal("10000"), 60, Decimal("5.0"))
		Decimal('529126.27')
	"""
	assert payment >= 0, "payment must be non-negative"
	assert n_periods > 0, "n_periods must be positive"
	assert 0 < annual_rate_pct < 100, "rate must be between 0 and 100"

	r = _periodic_rate(annual_rate_pct, periods_per_year)
	if r == 0:
		return _q(payment * Decimal(str(n_periods)))

	if in_advance:
		# annuity-due: multiply annuity-immediate by (1 + r)
		pv = payment * (1 - (1 + r) ** (-n_periods)) / r * (1 + r)
	else:
		pv = payment * (1 - (1 + r) ** (-n_periods)) / r

	return _q(Decimal(str(float(pv))))


def build_amortisation_schedule(
	opening_balance: Decimal,
	payment: Decimal,
	annual_rate_pct: Decimal,
	n_periods: int,
	periods_per_year: int = 12,
	start_date: date | None = None,
) -> list[AmortisationRow]:
	"""Build the full lease liability amortisation schedule.

	Each row shows interest accrual and principal reduction for one period.
	The schedule exactly zeroes the balance at period n_periods (rounding
	adjustment applied on the final period).

	Args:
		opening_balance: Lease liability at commencement (PV).
		payment: Constant periodic payment.
		annual_rate_pct: IBR / implicit rate as % p.a.
		n_periods: Total periods.
		periods_per_year: Periods per year for rate conversion.
		start_date: Optional commencement date; used to compute due_date per row.

	Returns:
		List of :class:`AmortisationRow`.
	"""
	assert opening_balance > 0, "opening_balance must be positive"
	assert payment > 0, "payment must be positive"

	r = _periodic_rate(annual_rate_pct, periods_per_year)
	rows: list[AmortisationRow] = []
	balance = opening_balance
	cumulative_interest = Decimal("0")

	for i in range(1, n_periods + 1):
		interest = _q(balance * r)
		# On the final period, sweep any rounding residual into principal
		if i == n_periods:
			principal = balance
			adjusted_payment = balance + interest
		else:
			principal = _q(payment - interest)
			adjusted_payment = payment

		closing = _q(max(balance - principal, Decimal("0")))
		cumulative_interest += interest

		due = _advance_month(start_date, i - 1) if start_date is not None else None

		rows.append(AmortisationRow(
			period=i,
			due_date=due,
			opening_balance=balance,
			payment=_q(adjusted_payment),
			interest=interest,
			principal=principal,
			closing_balance=closing,
			cumulative_interest=_q(cumulative_interest),
		))
		balance = closing

	return rows


# ── ROU Asset ─────────────────────────────────────────────────────────────────

def calculate_rou_asset(
	lease_liability: Decimal,
	initial_direct_costs: Decimal = Decimal("0"),
	prepaid_payments: Decimal = Decimal("0"),
	lease_incentives: Decimal = Decimal("0"),
	dismantling_costs: Decimal = Decimal("0"),
) -> Decimal:
	"""Compute the initial measurement of the right-of-use asset.

	IFRS 16.24:
		ROU = Lease Liability
		    + Initial direct costs
		    + Prepaid lease payments
		    − Lease incentives received
		    + Estimated dismantling / restoration costs (ARO)

	Args:
		lease_liability: Initial lease liability (PV of payments).
		initial_direct_costs: Incremental costs of obtaining the lease.
		prepaid_payments: Lease payments made at or before commencement.
		lease_incentives: Incentives received from the lessor (rent-free etc.).
		dismantling_costs: Asset retirement obligation / make-good.

	Returns:
		ROU asset at initial recognition.
	"""
	assert lease_liability >= 0
	rou = lease_liability + initial_direct_costs + prepaid_payments - lease_incentives + dismantling_costs
	return _q(max(rou, Decimal("0")))


def calculate_depreciation_charge(
	rou_asset: Decimal,
	useful_life_periods: int,
	accumulated_depreciation: Decimal = Decimal("0"),
	impairment: Decimal = Decimal("0"),
) -> Decimal:
	"""Straight-line depreciation charge for one period.

	Args:
		rou_asset: Initial ROU asset amount.
		useful_life_periods: Total depreciation periods (= lease term, normally).
		accumulated_depreciation: Depreciation accumulated so far.
		impairment: Any impairment loss already recognised.

	Returns:
		Depreciation charge for the current period.
	"""
	assert useful_life_periods > 0
	carrying = rou_asset - accumulated_depreciation - impairment
	if carrying <= 0:
		return Decimal("0")
	# Remaining periods = total - periods already elapsed
	if rou_asset > 0:
		period_charge = _q(rou_asset / Decimal(str(useful_life_periods)), _PLACES6)
		periods_elapsed = int((accumulated_depreciation / period_charge).to_integral_value()) if period_charge > 0 else 0
	else:
		periods_elapsed = 0
	remaining = max(1, useful_life_periods - periods_elapsed)
	return _q(carrying / Decimal(str(remaining)))


def amortise_rou_asset_period(
	rou_initial: Decimal,
	useful_life_months: int,
	accumulated_depreciation: Decimal,
	impairment_loss: Decimal = Decimal("0"),
) -> tuple[Decimal, Decimal, bool]:
	"""Advance ROU asset amortisation by one period.

	Returns:
		(depreciation_charge, new_carrying_amount, fully_amortised)
	"""
	charge = calculate_depreciation_charge(rou_initial, useful_life_months, accumulated_depreciation, impairment_loss)
	new_accumulated = accumulated_depreciation + charge
	carrying = _q(max(rou_initial - new_accumulated - impairment_loss, Decimal("0")))
	fully_amortised = carrying <= 0
	return charge, carrying, fully_amortised


# ── Interest Expense ──────────────────────────────────────────────────────────

def calculate_interest_expense(
	opening_balance: Decimal,
	annual_rate_pct: Decimal,
	periods_per_year: int = 12,
) -> Decimal:
	"""Interest expense for one period using the effective interest method.

	Args:
		opening_balance: Lease liability at the start of the period.
		annual_rate_pct: Effective interest rate as % p.a.
		periods_per_year: Payment periods per year.

	Returns:
		Interest charge for the period.
	"""
	r = _periodic_rate(annual_rate_pct, periods_per_year)
	return _q(opening_balance * r)


# ── CPI / Variable Payments ───────────────────────────────────────────────────

def apply_cpi_escalation(
	base_payment: Decimal,
	base_cpi: Decimal,
	current_cpi: Decimal,
	cap_rate: Decimal | None = None,
	floor_rate: Decimal | None = None,
) -> Decimal:
	"""Adjust a lease payment for CPI indexation.

	Under IFRS 16.42, when payments are indexed to CPI, the lessee remeasures
	the lease liability using the revised payments whenever the CPI changes.

	Args:
		base_payment: Payment at the reference CPI.
		base_cpi: CPI index value at commencement / last reset.
		current_cpi: Current CPI index value.
		cap_rate: Maximum percentage increase (collar ceiling).
		floor_rate: Minimum percentage increase (collar floor).

	Returns:
		Adjusted payment amount.
	"""
	assert base_cpi > 0, "base_cpi must be positive"
	change_ratio = current_cpi / base_cpi
	adjusted = base_payment * change_ratio

	# Apply collar
	if cap_rate is not None:
		max_payment = base_payment * (1 + cap_rate / Decimal("100"))
		adjusted = min(adjusted, max_payment)
	if floor_rate is not None:
		min_payment = base_payment * (1 + floor_rate / Decimal("100"))
		adjusted = max(adjusted, min_payment)

	return _q(adjusted)


def apply_fixed_escalation(
	payment: Decimal,
	annual_rate_pct: Decimal,
	periods_elapsed: int,
	periods_per_year: int = 12,
) -> Decimal:
	"""Compound a fixed-percentage escalation over elapsed periods.

	Args:
		payment: Base payment.
		annual_rate_pct: Annual escalation rate as %.
		periods_elapsed: Number of periods since last escalation.
		periods_per_year: Periods per year for rate conversion.

	Returns:
		Escalated payment.
	"""
	r = _periodic_rate(annual_rate_pct, periods_per_year)
	return _q(payment * (1 + r) ** periods_elapsed)


# ── Lease Modification ────────────────────────────────────────────────────────

def remeasure_lease_liability(
	new_payment: Decimal,
	remaining_periods: int,
	revised_rate_pct: Decimal,
	periods_per_year: int = 12,
) -> Decimal:
	"""Remeasure the lease liability at a modification date.

	IFRS 16.45 / .46: Use the revised discount rate (or IBR at modification date
	if the rate was previously undetermined).

	Args:
		new_payment: Revised periodic payment from modification date.
		remaining_periods: Remaining lease periods after modification.
		revised_rate_pct: Discount rate to use for remeasurement (% p.a.).
		periods_per_year: Periods per year.

	Returns:
		Remeasured lease liability.
	"""
	return calculate_lease_liability(new_payment, remaining_periods, revised_rate_pct, periods_per_year)


def calculate_partial_surrender_adjustment(
	current_liability: Decimal,
	current_rou: Decimal,
	surrendered_proportion: Decimal,
) -> tuple[Decimal, Decimal, Decimal]:
	"""Calculate gain/loss and adjusted balances for a partial surrender.

	IFRS 16.46(b): When the lessee surrenders part of the ROU asset, the
	carrying amounts of both the liability and ROU asset are reduced proportionally
	and a gain/loss is recognised.

	Args:
		current_liability: Current lease liability carrying amount.
		current_rou: Current ROU asset carrying amount.
		surrendered_proportion: Fraction of the space surrendered (0 < x < 1).

	Returns:
		(new_liability, new_rou, gain_loss)
		gain_loss > 0 → gain, < 0 → loss.
	"""
	assert 0 < surrendered_proportion < 1, "surrendered_proportion must be between 0 and 1"
	liability_reduction = _q(current_liability * surrendered_proportion)
	rou_reduction = _q(current_rou * surrendered_proportion)
	gain_loss = _q(liability_reduction - rou_reduction)
	return (
		_q(current_liability - liability_reduction),
		_q(current_rou - rou_reduction),
		gain_loss,
	)


def calculate_scope_increase_new_lease(
	additional_payment: Decimal,
	additional_periods: int,
	ibr_pct: Decimal,
	periods_per_year: int = 12,
) -> tuple[Decimal, Decimal]:
	"""Calculate initial measurements for a new lease arising from scope increase.

	IFRS 16.44: A modification that increases scope at a price commensurate with
	standalone value is treated as a separate new lease.

	Returns:
		(new_lease_liability, new_rou_asset) — both equal at initial recognition
		(assuming no IDC or incentives for the new component).
	"""
	new_liability = calculate_lease_liability(additional_payment, additional_periods, ibr_pct, periods_per_year)
	new_rou = new_liability  # IFRS 16.24 — no additional items assumed
	return new_liability, new_rou


# ── Lease Term ────────────────────────────────────────────────────────────────

def calculate_lease_term_months(
	commencement: date,
	contractual_end: date,
	renewal_options_months: int = 0,
	reasonably_certain_renewals: bool = False,
	termination_option_months: int = 0,
	reasonably_certain_not_to_terminate: bool = True,
) -> int:
	"""Determine the accounting lease term per IFRS 16.19.

	The lease term includes:
	- Non-cancellable period
	- Optional renewals where exercise is reasonably certain
	- Less optional termination periods where non-exercise is reasonably certain

	Args:
		commencement: Lease commencement date.
		contractual_end: End of non-cancellable period.
		renewal_options_months: Duration of available renewal option(s) in months.
		reasonably_certain_renewals: Whether renewal exercise is reasonably certain.
		termination_option_months: Months removed if termination option exercised.
		reasonably_certain_not_to_terminate: Whether lessee will not terminate early.

	Returns:
		Accounting lease term in months.
	"""
	base_months = _months_between(commencement, contractual_end)
	term = base_months
	if reasonably_certain_renewals:
		term += renewal_options_months
	if not reasonably_certain_not_to_terminate:
		term = max(0, term - termination_option_months)
	return term


# ── Exemption Tests ───────────────────────────────────────────────────────────

def is_short_term_exempt(lease_term_months: int) -> bool:
	"""IFRS 16.B34: Short-term exemption applies when lease term ≤ 12 months."""
	return lease_term_months <= 12


def is_low_value_exempt(
	fair_value_new_usd: Decimal,
	threshold_usd: Decimal = Decimal("5000"),
) -> bool:
	"""IFRS 16.5(b): Low-value exemption applies when the underlying asset's
	fair value when new is below the threshold (typically USD 5,000).

	Args:
		fair_value_new_usd: Fair value of the underlying asset when new, in USD.
		threshold_usd: Threshold (default USD 5,000 per IASB guidance).
	"""
	return fair_value_new_usd <= threshold_usd


# ── Sublease Income ───────────────────────────────────────────────────────────

def calculate_sublease_income(
	sublease_payment: Decimal,
	n_periods: int,
) -> Decimal:
	"""Total undiscounted sublease income over the sublease term.

	Args:
		sublease_payment: Periodic sublease receipt.
		n_periods: Remaining sublease periods.

	Returns:
		Total undiscounted sublease income.
	"""
	return _q(sublease_payment * Decimal(str(n_periods)))


def calculate_net_investment_sublease(
	sublease_payment: Decimal,
	n_periods: int,
	implicit_rate_pct: Decimal,
	periods_per_year: int = 12,
	residual_value: Decimal = Decimal("0"),
) -> Decimal:
	"""Net investment in finance sublease (lessor perspective, IFRS 16.67).

	Args:
		sublease_payment: Periodic lease payment from sublessee.
		n_periods: Remaining periods.
		implicit_rate_pct: Rate implicit in the sublease.
		periods_per_year: Periods per year.
		residual_value: Unguaranteed residual value.

	Returns:
		Net investment in the sublease (PV of payments + residual).
	"""
	pv_payments = calculate_lease_liability(sublease_payment, n_periods, implicit_rate_pct, periods_per_year)
	r = _periodic_rate(implicit_rate_pct, periods_per_year)
	pv_residual = _q(residual_value / (1 + r) ** n_periods)
	return _q(pv_payments + pv_residual)


# ── Portfolio Analytics ───────────────────────────────────────────────────────

def calculate_weighted_average_ibr(
	leases: list[dict[str, Any]],
) -> Decimal:
	"""Compute the liability-weighted average IBR across a portfolio.

	Args:
		leases: List of dicts with keys 'lease_liability' and 'ibr_pct'.

	Returns:
		Weighted average IBR as % p.a.
	"""
	total_liability = sum(Decimal(str(l["lease_liability"])) for l in leases)
	if total_liability == 0:
		return Decimal("0")
	weighted_sum = sum(
		Decimal(str(l["lease_liability"])) * Decimal(str(l["ibr_pct"]))
		for l in leases
	)
	return _q(weighted_sum / total_liability)


def calculate_maturity_analysis(
	schedule_rows: list[AmortisationRow],
	as_at: date,
) -> dict[str, Decimal]:
	"""Produce the undiscounted maturity analysis required by IFRS 16.58.

	Buckets: within 1 year, 1–5 years, beyond 5 years.

	Args:
		schedule_rows: Full amortisation schedule.
		as_at: Balance sheet date.

	Returns:
		Dict mapping band label to undiscounted payment total.
	"""
	bands: dict[str, Decimal] = {
		"within_1_year": Decimal("0"),
		"1_to_5_years": Decimal("0"),
		"beyond_5_years": Decimal("0"),
	}
	for row in schedule_rows:
		if row.due_date is None or row.due_date <= as_at:
			continue
		years_out = (row.due_date - as_at).days / 365.25
		if years_out <= 1:
			bands["within_1_year"] += row.payment
		elif years_out <= 5:
			bands["1_to_5_years"] += row.payment
		else:
			bands["beyond_5_years"] += row.payment
	return {k: _q(v) for k, v in bands.items()}


# ── Journal Entry Generation ──────────────────────────────────────────────────

@dataclass(frozen=True)
class JournalLine:
	"""A single debit or credit line in a journal entry."""
	account: str
	debit: Decimal
	credit: Decimal
	description: str


def generate_commencement_journals(
	lease_id: str,
	rou_asset: Decimal,
	lease_liability: Decimal,
	initial_direct_costs: Decimal = Decimal("0"),
	prepaid_payments: Decimal = Decimal("0"),
	lease_incentives: Decimal = Decimal("0"),
	dismantling_costs: Decimal = Decimal("0"),
	currency: str = "KES",
) -> list[JournalLine]:
	"""Generate journal entries at lease commencement (IFRS 16.22–24).

	Dr  Right-of-use asset
	Cr  Lease liability
	Dr  Right-of-use asset (IDC & prepaid)
	Cr  Cash / Prepayments (IDC & prepaid)
	Cr  Right-of-use asset (incentives received)
	Dr  Right-of-use asset (ARO)
	Cr  Provision for dismantling
	"""
	ZERO = Decimal("0")
	lines: list[JournalLine] = []

	# Core recognition
	lines.append(JournalLine("right_of_use_asset", rou_asset, ZERO, f"IFRS16 commencement — ROU asset [{lease_id}]"))
	lines.append(JournalLine("lease_liability", ZERO, lease_liability, f"IFRS16 commencement — lease liability [{lease_id}]"))

	if initial_direct_costs > 0:
		lines.append(JournalLine("right_of_use_asset", initial_direct_costs, ZERO, "Add: initial direct costs"))
		lines.append(JournalLine("cash_or_accruals", ZERO, initial_direct_costs, "Initial direct costs paid"))

	if prepaid_payments > 0:
		lines.append(JournalLine("right_of_use_asset", prepaid_payments, ZERO, "Add: prepaid lease payments"))
		lines.append(JournalLine("prepayments", ZERO, prepaid_payments, "Reclassify prepaid lease payments"))

	if lease_incentives > 0:
		lines.append(JournalLine("right_of_use_asset", ZERO, lease_incentives, "Less: lease incentives received"))
		lines.append(JournalLine("lease_incentive_liability", lease_incentives, ZERO, "Reclassify lessor incentive"))

	if dismantling_costs > 0:
		lines.append(JournalLine("right_of_use_asset", dismantling_costs, ZERO, "Add: ARO / make-good provision"))
		lines.append(JournalLine("provision_for_dismantling", ZERO, dismantling_costs, "Provision for dismantling"))

	return lines


def generate_period_journals(
	lease_id: str,
	period: str,
	depreciation: Decimal,
	interest: Decimal,
	payment: Decimal,
	principal: Decimal,
) -> list[JournalLine]:
	"""Generate period-end journals for an IFRS 16 lease.

	Dr  Depreciation expense
	Cr  Accumulated depreciation — ROU asset

	Dr  Interest expense
	Cr  Lease liability (accrued interest)

	Dr  Lease liability (payment)
	Cr  Cash
	"""
	ZERO = Decimal("0")
	ref = f"[{lease_id}] {period}"
	return [
		JournalLine("depreciation_expense", depreciation, ZERO, f"ROU asset depreciation {ref}"),
		JournalLine("accumulated_depreciation_rou", ZERO, depreciation, f"ROU asset depreciation {ref}"),
		JournalLine("interest_expense", interest, ZERO, f"Finance cost on lease liability {ref}"),
		JournalLine("lease_liability", ZERO, interest, f"Accrue interest on lease liability {ref}"),
		JournalLine("lease_liability", payment, ZERO, f"Lease payment — reduces liability {ref}"),
		JournalLine("cash_at_bank", ZERO, payment, f"Cash payment {ref}"),
	]


# ── IFRS 16 Disclosure Computation ───────────────────────────────────────────

@dataclass
class Ifrs16PeriodSummary:
	"""Aggregated IFRS 16 figures for a single reporting period."""
	total_rou_assets: Decimal = Decimal("0")
	total_lease_liabilities: Decimal = Decimal("0")
	current_lease_liabilities: Decimal = Decimal("0")
	non_current_lease_liabilities: Decimal = Decimal("0")
	depreciation_charge: Decimal = Decimal("0")
	interest_expense: Decimal = Decimal("0")
	short_term_lease_expense: Decimal = Decimal("0")
	low_value_lease_expense: Decimal = Decimal("0")
	variable_lease_expense: Decimal = Decimal("0")
	total_cash_outflow: Decimal = Decimal("0")
	maturity_analysis: dict[str, Decimal] | None = None
	weighted_average_ibr: Decimal = Decimal("0")
	lease_count: int = 0


def compute_current_portion(
	schedule_rows: list[AmortisationRow],
	as_at: date,
) -> tuple[Decimal, Decimal]:
	"""Split lease liability into current and non-current portions.

	Args:
		schedule_rows: Remaining amortisation schedule rows (future only).
		as_at: Balance sheet date.

	Returns:
		(current_portion, non_current_portion) — undiscounted principal split.
	"""
	one_year_out = date(as_at.year + 1, as_at.month, as_at.day)
	current = Decimal("0")
	non_current = Decimal("0")
	for row in schedule_rows:
		if row.due_date is None or row.due_date <= as_at:
			continue
		if row.due_date <= one_year_out:
			current += row.principal
		else:
			non_current += row.principal
	return _q(current), _q(non_current)
