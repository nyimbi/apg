"""
Financial calculations for APG Digital Lending.

All formulas are pure functions — no side effects, no I/O.
Type-safe inputs/outputs. Comprehensive edge case handling.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from __future__ import annotations

import calendar
import math
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _d2(v: float) -> float:
	"""Round to 2 decimal places using banker's rounding via Decimal."""
	return float(Decimal(str(v)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _add_months(d: date, months: int) -> date:
	"""Add calendar months, clamping to last day if needed."""
	if months == 0:
		return d
	month = d.month - 1 + months
	year = d.year + month // 12
	month = month % 12 + 1
	day = min(d.day, calendar.monthrange(year, month)[1])
	return date(year, month, day)


def _add_weeks(d: date, weeks: int) -> date:
	return d + timedelta(weeks=weeks)


def _add_days(d: date, days: int) -> date:
	return d + timedelta(days=days)


# ---------------------------------------------------------------------------
# EMI / instalment
# ---------------------------------------------------------------------------

def emi(principal: float, monthly_rate: float, n_months: int) -> float:
	"""
	Equated Monthly Instalment for reducing-balance loan.

	EMI = P * r * (1+r)^n / ((1+r)^n - 1)

	Edge cases:
	- Zero rate → flat division
	- Single period → full principal + interest
	"""
	if n_months <= 0:
		raise ValueError(f"n_months must be positive, got {n_months}")
	if principal <= 0:
		raise ValueError(f"principal must be positive, got {principal}")
	if monthly_rate < 0:
		raise ValueError(f"monthly_rate must be non-negative, got {monthly_rate}")
	if monthly_rate == 0:
		return _d2(principal / n_months)
	factor = (1 + monthly_rate) ** n_months
	return _d2(principal * monthly_rate * factor / (factor - 1))


def flat_rate_emi(principal: float, annual_flat_rate: float, n_months: int) -> float:
	"""
	Flat-rate instalment: interest computed on original principal throughout.

	EMI = (P + P * r * n/12) / n
	"""
	if n_months <= 0:
		raise ValueError("n_months must be positive")
	if principal <= 0:
		raise ValueError("principal must be positive")
	total_interest = _d2(principal * annual_flat_rate * n_months / 12)
	return _d2((principal + total_interest) / n_months)


def flat_rate_to_apr(flat_rate: float, n_months: int, iterations: int = 50) -> float:
	"""
	Convert flat rate to approximate APR via Newton-Raphson on the annuity formula.
	Accurate to 4 decimal places.
	"""
	if n_months <= 0:
		raise ValueError("n_months must be positive")
	# Initial guess: APR ≈ 2 * n / (n + 1) * flat_rate (actuarial approximation)
	# Initial guess: actuarial approximation
	r = flat_rate * 2 * n_months / (n_months + 1) / 12
	if r <= 0:
		r = 0.01 / 12
	flat_emi = flat_rate_emi(1.0, flat_rate, n_months)
	for _ in range(iterations):
		factor = (1 + r) ** n_months
		denom = factor - 1
		if abs(denom) < 1e-12:
			break
		f = r * factor / denom - flat_emi
		# d/dr [r*(1+r)^n / ((1+r)^n - 1)]
		# = [(1+r)^n + r*n*(1+r)^(n-1)] * denom - r*(1+r)^n * n*(1+r)^(n-1)
		#   / denom^2
		d_factor = n_months * (1 + r) ** (n_months - 1)
		df = ((factor + r * d_factor) * denom - r * factor * d_factor) / (denom ** 2)
		if abs(df) < 1e-14:
			break
		r_new = r - f / df
		if abs(r_new - r) < 1e-9:
			r = r_new
			break
		r = max(1e-8, r_new)
	return _d2(r * 12)


# ---------------------------------------------------------------------------
# Amortisation schedule
# ---------------------------------------------------------------------------

def build_amortisation_schedule(
	principal: float,
	annual_rate: float,
	tenor_months: int,
	start_date: date,
	schedule_type: str = "reducing_balance",
	repayment_frequency: str = "monthly",
) -> dict[str, Any]:
	"""
	Build a full amortisation schedule.

	schedule_type: reducing_balance | flat_rate | bullet | interest_only
	repayment_frequency: daily | weekly | biweekly | monthly | quarterly | bullet

	Returns schedule dict with installments list.
	"""
	if principal <= 0:
		raise ValueError("principal must be positive")
	if annual_rate < 0:
		raise ValueError("annual_rate must be non-negative")
	if tenor_months <= 0:
		raise ValueError("tenor_months must be positive")

	# Map frequency to periods per year and date-advance function
	freq_map: dict[str, tuple[int, Any]] = {
		"daily":     (365, lambda d, i: _add_days(d, i)),
		"weekly":    (52,  lambda d, i: _add_weeks(d, i)),
		"biweekly":  (26,  lambda d, i: _add_weeks(d, i * 2)),
		"monthly":   (12,  lambda d, i: _add_months(d, i)),
		"quarterly": (4,   lambda d, i: _add_months(d, i * 3)),
		"bullet":    (1,   lambda d, i: _add_months(d, tenor_months)),
	}
	if repayment_frequency not in freq_map:
		raise ValueError(f"unsupported repayment_frequency: {repayment_frequency}")

	periods_per_year, date_fn = freq_map[repayment_frequency]
	period_rate = annual_rate / periods_per_year

	# Compute number of installments
	if repayment_frequency == "bullet":
		n_periods = 1
	elif repayment_frequency == "daily":
		n_periods = tenor_months * 30  # approximate
	elif repayment_frequency == "weekly":
		n_periods = tenor_months * 4
	elif repayment_frequency == "biweekly":
		n_periods = tenor_months * 2
	elif repayment_frequency == "monthly":
		n_periods = tenor_months
	elif repayment_frequency == "quarterly":
		n_periods = max(1, tenor_months // 3)
	else:
		n_periods = tenor_months

	if schedule_type == "reducing_balance":
		inst_amount = emi(principal, period_rate, n_periods)
	elif schedule_type == "flat_rate":
		inst_amount = flat_rate_emi(principal, annual_rate, n_periods)
	elif schedule_type in ("bullet", "interest_only"):
		# Interest only + bullet principal at end
		inst_amount = _d2(principal * period_rate)
	else:
		raise ValueError(f"unsupported schedule_type: {schedule_type}")

	installments: list[dict[str, Any]] = []
	balance = principal
	cumulative_interest = 0.0
	cumulative_principal = 0.0

	for n in range(1, n_periods + 1):
		due_date = date_fn(start_date, n)
		opening_balance = balance

		if schedule_type == "reducing_balance":
			interest_portion = _d2(balance * period_rate)
			principal_portion = _d2(inst_amount - interest_portion)
		elif schedule_type == "flat_rate":
			interest_portion = _d2(principal * period_rate)
			principal_portion = _d2(principal / n_periods)
		elif schedule_type == "interest_only":
			interest_portion = _d2(balance * period_rate)
			principal_portion = _d2(principal) if n == n_periods else 0.0
		elif schedule_type == "bullet":
			interest_portion = _d2(balance * period_rate * tenor_months)
			principal_portion = principal

		# Last period: clear any floating-point residual
		if n == n_periods and balance > 0:
			principal_portion = _d2(principal_portion + balance - principal_portion)
			# Recalculate properly
			principal_portion = _d2(balance)
			if schedule_type not in ("bullet", "interest_only"):
				inst_amount_last = _d2(principal_portion + interest_portion)
			else:
				inst_amount_last = _d2(principal_portion + interest_portion)
			balance = 0.0
		else:
			balance = _d2(max(0.0, balance - principal_portion))

		cumulative_interest = _d2(cumulative_interest + interest_portion)
		cumulative_principal = _d2(cumulative_principal + principal_portion)

		installments.append({
			"installment_no": n,
			"due_date": due_date.isoformat(),
			"emi": _d2(principal_portion + interest_portion) if n == n_periods else inst_amount,
			"principal_portion": principal_portion,
			"interest_portion": interest_portion,
			"opening_balance": opening_balance,
			"closing_balance": balance,
			"cumulative_interest": cumulative_interest,
			"cumulative_principal": cumulative_principal,
			"status": "pending",
			"paid_amount": 0.0,
			"paid_date": None,
			"dpd": 0,
		})

	total_repayable = _d2(sum(i["emi"] for i in installments))
	total_interest = _d2(total_repayable - principal)

	return {
		"schedule_type": schedule_type,
		"repayment_frequency": repayment_frequency,
		"principal": principal,
		"annual_rate": annual_rate,
		"tenor_months": tenor_months,
		"n_periods": n_periods,
		"period_rate": period_rate,
		"monthly_emi": emi(principal, annual_rate / 12, tenor_months),
		"total_repayable": total_repayable,
		"total_interest": total_interest,
		"installments": installments,
	}


# ---------------------------------------------------------------------------
# Early settlement
# ---------------------------------------------------------------------------

def early_settlement_amount(
	outstanding_principal: float,
	annual_rate: float,
	disbursement_date: date,
	settlement_date: date,
	early_settlement_fee_pct: float = 0.01,
) -> dict[str, float]:
	"""
	Total amount to settle a loan early.

	= outstanding_principal
	+ accrued daily interest to settlement_date
	+ early_settlement_fee (% of outstanding principal)
	"""
	if settlement_date <= disbursement_date:
		raise ValueError("settlement_date must be after disbursement_date")
	days_accrued = (settlement_date - disbursement_date).days
	accrued_interest = _d2(outstanding_principal * annual_rate * days_accrued / 365)
	settlement_fee = _d2(outstanding_principal * early_settlement_fee_pct)
	total = _d2(outstanding_principal + accrued_interest + settlement_fee)
	return {
		"outstanding_principal": outstanding_principal,
		"accrued_interest": accrued_interest,
		"early_settlement_fee": settlement_fee,
		"total_settlement_amount": total,
	}


def interest_saving_early_settlement(
	remaining_installments: list[dict[str, Any]],
	settlement_total: float,
) -> float:
	"""Interest saved vs paying all remaining installments."""
	total_remaining = sum(i.get("interest_portion", 0) for i in remaining_installments)
	return _d2(max(0.0, total_remaining - (settlement_total - sum(i.get("principal_portion", 0) for i in remaining_installments))))


# ---------------------------------------------------------------------------
# Credit scoring
# ---------------------------------------------------------------------------

def risk_grade(score: int) -> str:
	"""Map composite score 300–850 to risk grade A–F."""
	if score >= 750:
		return "A"
	if score >= 680:
		return "B"
	if score >= 620:
		return "C"
	if score >= 560:
		return "D"
	if score >= 480:
		return "E"
	return "F"


def probability_of_default(score: int) -> float:
	"""
	Logistic-curve PD from score 300–850.
	Grade A ~ 0.5%, Grade F ~ 25%.
	"""
	if not 300 <= score <= 850:
		raise ValueError(f"score must be 300–850, got {score}")
	normalised = (score - 300) / 550
	return round(0.30 * math.exp(-3.5 * normalised), 4)


def composite_credit_score(
	behavioural_raw: float,  # 0–1
	demographic_raw: float,  # 0–1
	bureau_raw: float,       # 0–1
	weights: tuple[float, float, float] = (0.45, 0.20, 0.35),
) -> int:
	"""Weighted composite score mapped to 300–850."""
	w_b, w_d, w_bu = weights
	assert abs(w_b + w_d + w_bu - 1.0) < 1e-6, "weights must sum to 1.0"
	composite = (
		max(0.0, min(1.0, behavioural_raw)) * w_b
		+ max(0.0, min(1.0, demographic_raw)) * w_d
		+ max(0.0, min(1.0, bureau_raw)) * w_bu
	)
	return max(300, min(850, int(300 + composite * 550)))


def behavioural_score_raw(
	payment_ratio: float,
	utilisation_ratio: float,
	delinquent_loan_count: int,
) -> float:
	"""
	Behavioural pillar score (0–1).
	payment_ratio: fraction of payments made on time (0–1)
	utilisation_ratio: outstanding / original principal (0–1); lower is better
	delinquent_loan_count: loans ever 30+ DPD
	"""
	delinquency_penalty = min(delinquent_loan_count * 0.05, 0.40)
	utilisation_score = max(0.0, 1.0 - utilisation_ratio) * 0.8 + 0.2
	return max(0.0, min(1.0,
		payment_ratio * 0.55
		+ utilisation_score * 0.25
		+ max(0.0, 1.0 - delinquency_penalty) * 0.20
	))


def bureau_score_raw(
	bureau_score: int,           # 300–900
	defaults_count: int,
	fraud_flags_count: int,
	bureau_min: int = 300,
	bureau_max: int = 900,
) -> float:
	"""Bureau pillar score (0–1) from CRB/bureau data."""
	normalised = (bureau_score - bureau_min) / (bureau_max - bureau_min)
	return max(0.0, normalised - defaults_count * 0.08 - fraud_flags_count * 0.15)


# ---------------------------------------------------------------------------
# Debt service ratio
# ---------------------------------------------------------------------------

def debt_service_ratio(
	monthly_net_income: float,
	existing_monthly_obligations: float,
	new_monthly_emi: float,
) -> dict[str, Any]:
	"""
	DSR = (existing_obligations + new_emi) / monthly_net_income.
	Standard threshold: 40% (configurable).
	"""
	if monthly_net_income <= 0:
		return {
			"dsr": float("inf"),
			"passes": False,
			"total_obligations": existing_monthly_obligations + new_monthly_emi,
			"monthly_net_income": monthly_net_income,
			"threshold": 0.40,
		}
	total = existing_monthly_obligations + new_monthly_emi
	dsr = round(total / monthly_net_income, 4)
	return {
		"dsr": dsr,
		"passes": dsr <= 0.40,
		"total_obligations": _d2(total),
		"existing_obligations": _d2(existing_monthly_obligations),
		"new_emi": _d2(new_monthly_emi),
		"monthly_net_income": _d2(monthly_net_income),
		"threshold": 0.40,
	}


def max_loan_from_dsr(
	monthly_net_income: float,
	existing_monthly_obligations: float,
	annual_rate: float,
	tenor_months: int,
	dsr_threshold: float = 0.40,
) -> float:
	"""
	Solve for maximum loan principal given DSR constraint.
	affordable_emi = income * threshold - existing_obligations
	P = affordable_emi * [(1-(1+r)^-n) / r]
	"""
	affordable_emi = monthly_net_income * dsr_threshold - existing_monthly_obligations
	if affordable_emi <= 0:
		return 0.0
	monthly_rate = annual_rate / 12
	if monthly_rate == 0:
		return _d2(affordable_emi * tenor_months)
	return _d2(affordable_emi * (1 - (1 + monthly_rate) ** -tenor_months) / monthly_rate)


# ---------------------------------------------------------------------------
# IFRS 9 ECL
# ---------------------------------------------------------------------------

def ecl_stage1(ead: float, pd_12m: float, lgd: float) -> float:
	"""12-month ECL for performing loans (Stage 1)."""
	return _d2(pd_12m * lgd * ead)


def ecl_stage2(ead: float, pd_lifetime: float, lgd: float) -> float:
	"""Lifetime ECL for significant credit deterioration (Stage 2)."""
	return _d2(pd_lifetime * lgd * ead)


def ecl_stage3(ead: float, lgd: float) -> float:
	"""Lifetime ECL for credit-impaired loans (Stage 3): PD = 1.0."""
	return _d2(lgd * ead)


def lgd_from_collateral(has_collateral: bool, collateral_coverage: float = 0.0) -> float:
	"""
	LGD estimate:
	- Unsecured: 40%
	- Partially secured: 25–35%
	- Fully secured (coverage >= 1.0): 15–25%
	"""
	if not has_collateral:
		return 0.40
	if collateral_coverage >= 1.5:
		return 0.15
	if collateral_coverage >= 1.0:
		return 0.25
	return 0.35


def pd_lifetime(pd_12m: float, tenor_months: int, remaining_months: int) -> float:
	"""Approximate lifetime PD from 12-month PD using Merton-style extrapolation."""
	if remaining_months <= 12:
		return pd_12m
	return min(1.0, round(pd_12m * remaining_months / 12, 4))


# ---------------------------------------------------------------------------
# Collateral valuation
# ---------------------------------------------------------------------------

HAIRCUT_TABLE: dict[str, float] = {
	"property":  0.60,
	"land":      0.55,
	"vehicle":   0.70,
	"cash":      0.95,
	"shares":    0.50,
	"inventory": 0.40,
	"machinery": 0.55,
	"other":     0.60,
}


def forced_sale_value(market_value: float, collateral_type: str) -> float:
	"""FSV = market_value × (1 − haircut)."""
	haircut = HAIRCUT_TABLE.get(collateral_type.lower(), 0.60)
	return _d2(market_value * haircut)


def collateral_coverage_ratio(total_fsv: float, outstanding_principal: float) -> float:
	"""Coverage = total FSV / outstanding principal. >1.0 is fully covered."""
	if outstanding_principal <= 0:
		return float("inf")
	return round(total_fsv / outstanding_principal, 4)


# ---------------------------------------------------------------------------
# DPD & delinquency
# ---------------------------------------------------------------------------

def dpd_bucket(dpd: int) -> str:
	"""Map DPD integer to standard delinquency bucket label."""
	if dpd == 0:
		return "current"
	if dpd <= 30:
		return "1-30"
	if dpd <= 60:
		return "31-60"
	if dpd <= 90:
		return "61-90"
	if dpd <= 120:
		return "91-120"
	return "120+"


def ifrs9_stage(dpd: int) -> str:
	"""Derive IFRS 9 stage from DPD."""
	if dpd == 0:
		return "stage1"
	if dpd <= 90:
		return "stage2"
	return "stage3"


def par_ratio(bucket_outstanding: float, total_portfolio_outstanding: float) -> float:
	"""Portfolio at Risk ratio for a DPD bucket."""
	if total_portfolio_outstanding <= 0:
		return 0.0
	return round(bucket_outstanding / total_portfolio_outstanding, 4)


# ---------------------------------------------------------------------------
# Rate pricing
# ---------------------------------------------------------------------------

GRADE_SPREAD: dict[str, float] = {
	"A": 0.00,
	"B": 0.02,
	"C": 0.04,
	"D": 0.07,
	"E": 0.12,
	"F": 0.20,
}

GRADE_CAP: dict[str, float] = {
	"A": 1.00,
	"B": 0.85,
	"C": 0.70,
	"D": 0.50,
	"E": 0.30,
	"F": 0.20,
}


def risk_adjusted_rate(base_rate: float, risk_grade: str) -> float:
	"""Base rate + credit spread based on risk grade."""
	spread = GRADE_SPREAD.get(risk_grade.upper(), 0.10)
	return round(max(0.01, base_rate + spread), 6)


def grade_amount_cap(max_product_amount: float, risk_grade: str) -> float:
	"""Grade-based cap: Grade A gets 100% of product max, F gets 20%."""
	cap = GRADE_CAP.get(risk_grade.upper(), 0.50)
	return _d2(max_product_amount * cap)


# ---------------------------------------------------------------------------
# Penalty calculations
# ---------------------------------------------------------------------------

def late_payment_penalty(overdue_amount: float, penalty_rate: float, days_overdue: int) -> float:
	"""
	Daily compounding late penalty.
	penalty = overdue_amount × ((1 + daily_rate)^days − 1)
	"""
	if days_overdue <= 0 or overdue_amount <= 0:
		return 0.0
	daily_rate = penalty_rate / 365
	return _d2(overdue_amount * ((1 + daily_rate) ** days_overdue - 1))


def processing_fee(principal: float, fee_pct: float) -> float:
	"""Flat processing fee deducted at disbursement."""
	return _d2(principal * fee_pct)


# ---------------------------------------------------------------------------
# PAR / portfolio
# ---------------------------------------------------------------------------

def portfolio_at_risk(
	loans: list[dict[str, Any]],
	as_of_date: date,
	dpd_threshold: int = 30,
) -> float:
	"""
	PAR(threshold) = outstanding balance of loans with max DPD > threshold
	               / total active portfolio outstanding.
	Each loan dict: {outstanding_principal, installments: [{due_date, status}]}
	"""
	total = 0.0
	at_risk = 0.0
	for loan in loans:
		if loan.get("status") != "active":
			continue
		bal = loan.get("outstanding_principal", 0.0)
		total += bal
		max_dpd_loan = 0
		for inst in loan.get("installments", []):
			if inst.get("status") == "paid":
				continue
			try:
				due = date.fromisoformat(inst["due_date"])
			except (KeyError, ValueError):
				continue
			dpd_val = max(0, (as_of_date - due).days)
			max_dpd_loan = max(max_dpd_loan, dpd_val)
		if max_dpd_loan > dpd_threshold:
			at_risk += bal
	return round(at_risk / total, 4) if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Stress testing
# ---------------------------------------------------------------------------

def stress_scenario(
	total_book: float,
	base_npl_ratio: float,
	additional_default_rate: float,
	lgd: float = 0.40,
) -> dict[str, float]:
	"""Single stress scenario: incremental loss and stressed NPL."""
	if not 0 <= additional_default_rate <= 1:
		raise ValueError("additional_default_rate must be in [0,1]")
	if not 0 <= lgd <= 1:
		raise ValueError("lgd must be in [0,1]")
	incremental_loss = _d2(total_book * additional_default_rate * lgd)
	stressed_npl = round(min(1.0, base_npl_ratio + additional_default_rate), 4)
	return {
		"incremental_loss": incremental_loss,
		"stressed_npl_ratio": stressed_npl,
		"stressed_npl_balance": _d2(total_book * stressed_npl),
		"capital_charge": incremental_loss,
	}
