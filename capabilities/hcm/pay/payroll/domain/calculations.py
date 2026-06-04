"""
Payroll domain calculations — tax, statutory deductions, overtime,
proration, leave encashment, final settlement.

All functions are pure (no I/O). Monetary inputs/outputs are Decimal.
Country-specific logic is isolated into dedicated functions; the dispatcher
`calculate_paye` and `calculate_statutory_deductions` route by Country enum.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

D = Decimal
TWO = D("0.01")


def _d(v: float | int | str | Decimal) -> Decimal:
	return D(str(v))


def _round(v: Decimal) -> Decimal:
	return v.quantize(TWO, rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Tax band data structures
# ---------------------------------------------------------------------------

@dataclass
class TaxBand:
	"""One bracket in a progressive tax table."""
	min_income: Decimal
	max_income: Decimal | None   # None = unbounded
	rate: Decimal                # 0.10 means 10%


@dataclass
class PayeResult:
	"""Output of a PAYE calculation."""
	country: str
	gross_income: Decimal
	taxable_income: Decimal
	personal_relief: Decimal
	insurance_relief: Decimal
	mortgage_relief: Decimal
	other_relief: Decimal
	gross_tax: Decimal
	tax_relief_total: Decimal
	paye_amount: Decimal
	bands_applied: list[dict] = field(default_factory=list)


@dataclass
class StatutoryResult:
	"""Output of statutory deduction calculation for one contribution type."""
	deduction_type: str
	basis: Decimal
	employee_amount: Decimal
	employer_amount: Decimal
	rate_used: Decimal | None
	cap_applied: bool
	notes: str = ""


# ---------------------------------------------------------------------------
# Generic progressive tax engine
# ---------------------------------------------------------------------------

def _apply_progressive_bands(
	taxable: Decimal,
	bands: list[TaxBand],
) -> tuple[Decimal, list[dict]]:
	"""Apply graduated tax bands; return (gross_tax, band_detail)."""
	gross_tax = D("0")
	band_detail: list[dict] = []
	remaining = taxable

	for band in sorted(bands, key=lambda b: b.min_income):
		if remaining <= 0:
			break
		if band.max_income is not None:
			band_size = band.max_income - band.min_income
		else:
			band_size = remaining
		taxable_in_band = min(remaining, band_size)
		tax_in_band = _round(taxable_in_band * band.rate)
		gross_tax += tax_in_band
		band_detail.append({
			"band_min": band.min_income,
			"band_max": band.max_income,
			"rate": band.rate,
			"taxable_in_band": taxable_in_band,
			"tax_in_band": tax_in_band,
		})
		remaining -= taxable_in_band

	return _round(gross_tax), band_detail


# ===========================================================================
# KENYA PAYE (KRA)
# Rates effective FY2023/24 onwards (Finance Act 2023)
# Monthly bands, KES
# ===========================================================================

_KE_BANDS_MONTHLY: list[TaxBand] = [
	TaxBand(D("0"),      D("24000"),  D("0.10")),
	TaxBand(D("24000"),  D("32333"),  D("0.25")),
	TaxBand(D("32333"),  D("500000"), D("0.30")),
	TaxBand(D("500000"), D("800000"), D("0.325")),
	TaxBand(D("800000"), None,        D("0.35")),
]

_KE_PERSONAL_RELIEF_MONTHLY = D("2400")     # KES/month
_KE_INSURANCE_RELIEF_MAX_MONTHLY = D("5000")  # 15% of premiums, max KES 5,000/mo


def calculate_paye_ke(
	gross_monthly: Decimal,
	*,
	non_taxable_allowances: Decimal = D("0"),
	pension_employee: Decimal = D("0"),
	insurance_premiums: Decimal = D("0"),
	mortgage_interest_annual: Decimal = D("0"),
	is_resident: bool = True,
) -> PayeResult:
	"""
	Kenya PAYE (monthly).

	Taxable income = gross - non_taxable_allowances - pension_deduction
	Personal relief: KES 2,400/month (residents only)
	Insurance relief: 15% of premiums paid, capped at KES 5,000/month
	Mortgage relief: 25% of mortgage interest, max KES 25,000/month
	"""
	# Pension deduction capped at KES 20,000/month (Finance Act 2021)
	pension_cap = D("20000")
	pension_deduction = min(pension_employee, pension_cap)

	taxable_income = _round(gross_monthly - non_taxable_allowances - pension_deduction)
	taxable_income = max(D("0"), taxable_income)

	gross_tax, bands = _apply_progressive_bands(taxable_income, _KE_BANDS_MONTHLY)

	personal_relief = _KE_PERSONAL_RELIEF_MONTHLY if is_resident else D("0")
	insurance_relief = min(
		_round(insurance_premiums * D("0.15")),
		_KE_INSURANCE_RELIEF_MAX_MONTHLY,
	) if insurance_premiums > 0 else D("0")
	mortgage_relief_monthly = _round(mortgage_interest_annual / 12 * D("0.25"))
	mortgage_relief = min(mortgage_relief_monthly, D("25000"))

	total_relief = personal_relief + insurance_relief + mortgage_relief
	paye = _round(max(D("0"), gross_tax - total_relief))

	return PayeResult(
		country="KE",
		gross_income=gross_monthly,
		taxable_income=taxable_income,
		personal_relief=personal_relief,
		insurance_relief=insurance_relief,
		mortgage_relief=mortgage_relief,
		other_relief=D("0"),
		gross_tax=gross_tax,
		tax_relief_total=total_relief,
		paye_amount=paye,
		bands_applied=bands,
	)


# ===========================================================================
# TANZANIA PAYE (TRA)
# Monthly bands, TZS (Finance Act 2023)
# ===========================================================================

_TZ_BANDS_MONTHLY: list[TaxBand] = [
	TaxBand(D("0"),       D("270000"),  D("0.00")),
	TaxBand(D("270000"),  D("520000"),  D("0.08")),
	TaxBand(D("520000"),  D("760000"),  D("0.20")),
	TaxBand(D("760000"),  D("1000000"), D("0.25")),
	TaxBand(D("1000000"), None,         D("0.30")),
]


def calculate_paye_tz(
	gross_monthly: Decimal,
	*,
	pension_employee: Decimal = D("0"),
) -> PayeResult:
	taxable_income = _round(max(D("0"), gross_monthly - pension_employee))
	gross_tax, bands = _apply_progressive_bands(taxable_income, _TZ_BANDS_MONTHLY)
	return PayeResult(
		country="TZ",
		gross_income=gross_monthly,
		taxable_income=taxable_income,
		personal_relief=D("0"),
		insurance_relief=D("0"),
		mortgage_relief=D("0"),
		other_relief=D("0"),
		gross_tax=gross_tax,
		tax_relief_total=D("0"),
		paye_amount=gross_tax,
		bands_applied=bands,
	)


# ===========================================================================
# UGANDA PAYE (URA)
# Monthly bands, UGX (FY2023/24)
# ===========================================================================

_UG_BANDS_MONTHLY: list[TaxBand] = [
	TaxBand(D("0"),        D("235000"),  D("0.00")),
	TaxBand(D("235000"),   D("335000"),  D("0.10")),
	TaxBand(D("335000"),   D("410000"),  D("0.20")),
	TaxBand(D("410000"),   None,         D("0.30")),
]


def calculate_paye_ug(gross_monthly: Decimal) -> PayeResult:
	# 10% surcharge if income > UGX 10M/month
	surcharge = D("0")
	gross_tax, bands = _apply_progressive_bands(gross_monthly, _UG_BANDS_MONTHLY)
	if gross_monthly > D("10000000"):
		surcharge = _round(gross_tax * D("0.10"))
	paye = gross_tax + surcharge
	return PayeResult(
		country="UG",
		gross_income=gross_monthly,
		taxable_income=gross_monthly,
		personal_relief=D("0"),
		insurance_relief=D("0"),
		mortgage_relief=D("0"),
		other_relief=surcharge * -1,   # show surcharge as negative relief for clarity
		gross_tax=gross_tax,
		tax_relief_total=D("0"),
		paye_amount=paye,
		bands_applied=bands,
	)


# ===========================================================================
# GHANA PAYE (GRA)
# Monthly bands, GHS (2024)
# ===========================================================================

_GH_BANDS_MONTHLY: list[TaxBand] = [
	TaxBand(D("0"),     D("402"),    D("0.00")),
	TaxBand(D("402"),   D("510"),    D("0.05")),
	TaxBand(D("510"),   D("880"),    D("0.10")),
	TaxBand(D("880"),   D("2585"),   D("0.175")),
	TaxBand(D("2585"),  D("5290"),   D("0.25")),
	TaxBand(D("5290"),  D("10580"),  D("0.30")),
	TaxBand(D("10580"), None,        D("0.35")),
]


def calculate_paye_gh(gross_monthly: Decimal, *, ssnit_employee: Decimal = D("0")) -> PayeResult:
	taxable = _round(max(D("0"), gross_monthly - ssnit_employee))
	gross_tax, bands = _apply_progressive_bands(taxable, _GH_BANDS_MONTHLY)
	return PayeResult(
		country="GH",
		gross_income=gross_monthly,
		taxable_income=taxable,
		personal_relief=D("0"),
		insurance_relief=D("0"),
		mortgage_relief=D("0"),
		other_relief=D("0"),
		gross_tax=gross_tax,
		tax_relief_total=D("0"),
		paye_amount=gross_tax,
		bands_applied=bands,
	)


# ===========================================================================
# NIGERIA PAYE (FIRS)
# Annual bands, NGN (2023); divide by 12 for monthly context
# Taxable = gross - pension - NHF - NHIS - life insurance
# ===========================================================================

_NG_BANDS_ANNUAL: list[TaxBand] = [
	TaxBand(D("0"),         D("300000"),   D("0.07")),
	TaxBand(D("300000"),    D("600000"),   D("0.11")),
	TaxBand(D("600000"),    D("1100000"),  D("0.15")),
	TaxBand(D("1100000"),   D("1600000"),  D("0.19")),
	TaxBand(D("1600000"),   D("3200000"),  D("0.21")),
	TaxBand(D("3200000"),   None,          D("0.24")),
]


def calculate_paye_ng(
	gross_annual: Decimal,
	*,
	pension_employee: Decimal = D("0"),
	nhf: Decimal = D("0"),
	nhis: Decimal = D("0"),
	life_insurance: Decimal = D("0"),
) -> PayeResult:
	"""Nigeria PAYE — annual inputs/outputs in NGN."""
	# Consolidated relief: higher of NGN 200,000 or 1% of gross
	cra = max(D("200000"), _round(gross_annual * D("0.01"))) + _round(gross_annual * D("0.20"))
	deductions = pension_employee + nhf + nhis + life_insurance + cra
	taxable = _round(max(D("0"), gross_annual - deductions))
	gross_tax, bands = _apply_progressive_bands(taxable, _NG_BANDS_ANNUAL)
	return PayeResult(
		country="NG",
		gross_income=gross_annual,
		taxable_income=taxable,
		personal_relief=cra,
		insurance_relief=life_insurance,
		mortgage_relief=D("0"),
		other_relief=pension_employee + nhf + nhis,
		gross_tax=gross_tax,
		tax_relief_total=cra + life_insurance + pension_employee + nhf + nhis,
		paye_amount=gross_tax,
		bands_applied=bands,
	)


# ===========================================================================
# SOUTH AFRICA PAYE (SARS)
# Annual bands, ZAR (2024/25)
# ===========================================================================

_ZA_BANDS_ANNUAL: list[TaxBand] = [
	TaxBand(D("0"),        D("237100"),   D("0.18")),
	TaxBand(D("237100"),   D("370500"),   D("0.26")),
	TaxBand(D("370500"),   D("512800"),   D("0.31")),
	TaxBand(D("512800"),   D("673000"),   D("0.36")),
	TaxBand(D("673000"),   D("857900"),   D("0.39")),
	TaxBand(D("857900"),   D("1817000"),  D("0.41")),
	TaxBand(D("1817000"),  None,          D("0.45")),
]

_ZA_PRIMARY_REBATE = D("17235")   # annual
_ZA_SECONDARY_REBATE = D("9444")  # age 65–74
_ZA_TERTIARY_REBATE = D("3145")   # age 75+


def calculate_paye_za(
	gross_annual: Decimal,
	*,
	age: int = 30,
	pension_employee: Decimal = D("0"),
	ra_contributions: Decimal = D("0"),
) -> PayeResult:
	"""South Africa income tax — annual figures in ZAR."""
	# Retirement annuity/pension: capped at 27.5% of remuneration, max ZAR 350,000
	retirement_deduction = min(
		_round((gross_annual) * D("0.275")),
		D("350000"),
		pension_employee + ra_contributions,
	)
	taxable = _round(max(D("0"), gross_annual - retirement_deduction))
	gross_tax, bands = _apply_progressive_bands(taxable, _ZA_BANDS_ANNUAL)

	rebate = _ZA_PRIMARY_REBATE
	if age >= 65:
		rebate += _ZA_SECONDARY_REBATE
	if age >= 75:
		rebate += _ZA_TERTIARY_REBATE

	paye = _round(max(D("0"), gross_tax - rebate))
	return PayeResult(
		country="ZA",
		gross_income=gross_annual,
		taxable_income=taxable,
		personal_relief=rebate,
		insurance_relief=D("0"),
		mortgage_relief=D("0"),
		other_relief=retirement_deduction,
		gross_tax=gross_tax,
		tax_relief_total=rebate + retirement_deduction,
		paye_amount=paye,
		bands_applied=bands,
	)


# ===========================================================================
# ZAMBIA PAYE (ZRA)
# Monthly bands, ZMW (2024)
# ===========================================================================

_ZM_BANDS_MONTHLY: list[TaxBand] = [
	TaxBand(D("0"),      D("4800"),   D("0.00")),
	TaxBand(D("4800"),   D("9200"),   D("0.20")),
	TaxBand(D("9200"),   D("13800"),  D("0.30")),
	TaxBand(D("13800"),  None,        D("0.375")),
]


def calculate_paye_zm(gross_monthly: Decimal, *, napsa_employee: Decimal = D("0")) -> PayeResult:
	taxable = _round(max(D("0"), gross_monthly - napsa_employee))
	gross_tax, bands = _apply_progressive_bands(taxable, _ZM_BANDS_MONTHLY)
	return PayeResult(
		country="ZM",
		gross_income=gross_monthly,
		taxable_income=taxable,
		personal_relief=D("0"),
		insurance_relief=D("0"),
		mortgage_relief=D("0"),
		other_relief=D("0"),
		gross_tax=gross_tax,
		tax_relief_total=D("0"),
		paye_amount=gross_tax,
		bands_applied=bands,
	)


# ===========================================================================
# RWANDA PAYE (RRA)
# Monthly bands, RWF (2024)
# ===========================================================================

_RW_BANDS_MONTHLY: list[TaxBand] = [
	TaxBand(D("0"),      D("30000"),  D("0.00")),
	TaxBand(D("30000"),  D("100000"), D("0.20")),
	TaxBand(D("100000"), None,        D("0.30")),
]


def calculate_paye_rw(gross_monthly: Decimal) -> PayeResult:
	gross_tax, bands = _apply_progressive_bands(gross_monthly, _RW_BANDS_MONTHLY)
	return PayeResult(
		country="RW",
		gross_income=gross_monthly,
		taxable_income=gross_monthly,
		personal_relief=D("0"),
		insurance_relief=D("0"),
		mortgage_relief=D("0"),
		other_relief=D("0"),
		gross_tax=gross_tax,
		tax_relief_total=D("0"),
		paye_amount=gross_tax,
		bands_applied=bands,
	)


# ===========================================================================
# PAYE dispatcher
# ===========================================================================

def calculate_paye(
	country: str,
	gross: Decimal,
	**kwargs,
) -> PayeResult:
	"""
	Route PAYE calculation to the correct country engine.

	kwargs are forwarded verbatim to the country function — see individual
	calculate_paye_<cc> functions for their parameters.
	"""
	_map = {
		"KE": calculate_paye_ke,
		"TZ": calculate_paye_tz,
		"UG": calculate_paye_ug,
		"GH": calculate_paye_gh,
		"NG": calculate_paye_ng,
		"ZA": calculate_paye_za,
		"ZM": calculate_paye_zm,
		"RW": calculate_paye_rw,
	}
	fn = _map.get(country.upper())
	if fn is None:
		# Fallback: flat 30% (conservative; flag as estimate)
		taxable = gross
		tax = _round(taxable * D("0.30"))
		return PayeResult(
			country=country,
			gross_income=gross,
			taxable_income=taxable,
			personal_relief=D("0"),
			insurance_relief=D("0"),
			mortgage_relief=D("0"),
			other_relief=D("0"),
			gross_tax=tax,
			tax_relief_total=D("0"),
			paye_amount=tax,
			bands_applied=[{"note": "estimated_flat_30pct"}],
		)
	return fn(gross, **kwargs)


# ===========================================================================
# STATUTORY DEDUCTIONS
# ===========================================================================

# ---------------------------------------------------------------------------
# Kenya NSSF (NSSF Act 2013, new contribution model)
# Tier I: 6% of pensionable wages up to KES 6,000 lower earnings limit
# Tier II: 6% on wages between lower earnings limit and upper earnings limit (KES 18,000)
# Employee 6%, Employer 6%
# ---------------------------------------------------------------------------

_KE_NSSF_LOWER_LIMIT = D("6000")    # KES/month
_KE_NSSF_UPPER_LIMIT = D("18000")   # KES/month
_KE_NSSF_RATE = D("0.06")


def calculate_nssf_ke(pensionable_wages: Decimal) -> StatutoryResult:
	capped = min(pensionable_wages, _KE_NSSF_UPPER_LIMIT)
	employee = _round(capped * _KE_NSSF_RATE)
	employer = employee  # equal contributions
	cap_applied = pensionable_wages > _KE_NSSF_UPPER_LIMIT
	return StatutoryResult(
		deduction_type="nssf",
		basis=pensionable_wages,
		employee_amount=employee,
		employer_amount=employer,
		rate_used=_KE_NSSF_RATE,
		cap_applied=cap_applied,
		notes=f"Capped at KES {_KE_NSSF_UPPER_LIMIT}" if cap_applied else "",
	)


# ---------------------------------------------------------------------------
# Kenya NHIF / SHA (Social Health Insurance — Finance Act 2023 SHIF)
# SHIF: 2.75% of gross salary, no cap, effective Oct 2023
# Legacy NHIF: banded; we implement the new SHI model
# ---------------------------------------------------------------------------

_KE_SHIF_RATE = D("0.0275")
_KE_SHIF_MIN = D("300")        # minimum monthly contribution


def calculate_nhif_ke(gross_salary: Decimal) -> StatutoryResult:
	employee = max(_round(gross_salary * _KE_SHIF_RATE), _KE_SHIF_MIN)
	return StatutoryResult(
		deduction_type="nhif_shi",
		basis=gross_salary,
		employee_amount=employee,
		employer_amount=D("0"),
		rate_used=_KE_SHIF_RATE,
		cap_applied=False,
		notes="Social Health Insurance (SHIF) @ 2.75%",
	)


# ---------------------------------------------------------------------------
# Kenya NITA (National Industrial Training Authority) — employer only
# KES 50 per employee per month
# ---------------------------------------------------------------------------

def calculate_nita_ke() -> StatutoryResult:
	return StatutoryResult(
		deduction_type="nita",
		basis=D("0"),
		employee_amount=D("0"),
		employer_amount=D("50"),
		rate_used=None,
		cap_applied=False,
		notes="NITA levy KES 50/employee/month",
	)


# ---------------------------------------------------------------------------
# Tanzania NSSF (10% employee + 10% employer on gross, capped)
# ---------------------------------------------------------------------------

_TZ_NSSF_RATE_EMPLOYEE = D("0.10")
_TZ_NSSF_RATE_EMPLOYER = D("0.10")
_TZ_NSSF_WAGE_CAP = D("5000000")  # TZS / month


def calculate_nssf_tz(gross_wages: Decimal) -> StatutoryResult:
	capped = min(gross_wages, _TZ_NSSF_WAGE_CAP)
	employee = _round(capped * _TZ_NSSF_RATE_EMPLOYEE)
	employer = _round(capped * _TZ_NSSF_RATE_EMPLOYER)
	return StatutoryResult(
		deduction_type="nssf",
		basis=gross_wages,
		employee_amount=employee,
		employer_amount=employer,
		rate_used=_TZ_NSSF_RATE_EMPLOYEE,
		cap_applied=gross_wages > _TZ_NSSF_WAGE_CAP,
	)


# Tanzania SDL (4% of gross wages — employer only)
_TZ_SDL_RATE = D("0.04")


def calculate_sdl_tz(gross_wages: Decimal) -> StatutoryResult:
	employer = _round(gross_wages * _TZ_SDL_RATE)
	return StatutoryResult(
		deduction_type="sdl",
		basis=gross_wages,
		employee_amount=D("0"),
		employer_amount=employer,
		rate_used=_TZ_SDL_RATE,
		cap_applied=False,
		notes="Skills Development Levy 4% (employer)",
	)


# Tanzania WCF (0.5% of gross wages — employer only, capped at TZS 3,000/month)
_TZ_WCF_RATE = D("0.005")
_TZ_WCF_CAP = D("3000")


def calculate_wcf_tz(gross_wages: Decimal) -> StatutoryResult:
	raw = _round(gross_wages * _TZ_WCF_RATE)
	employer = min(raw, _TZ_WCF_CAP)
	return StatutoryResult(
		deduction_type="wcf",
		basis=gross_wages,
		employee_amount=D("0"),
		employer_amount=employer,
		rate_used=_TZ_WCF_RATE,
		cap_applied=raw > _TZ_WCF_CAP,
		notes="Workers Compensation Fund",
	)


# ---------------------------------------------------------------------------
# Ghana SSNIT — 5.5% employee, 13% employer (2023)
# ---------------------------------------------------------------------------

_GH_SSNIT_RATE_EMPLOYEE = D("0.055")
_GH_SSNIT_RATE_EMPLOYER = D("0.13")


def calculate_ssnit_gh(gross_wages: Decimal) -> StatutoryResult:
	employee = _round(gross_wages * _GH_SSNIT_RATE_EMPLOYEE)
	employer = _round(gross_wages * _GH_SSNIT_RATE_EMPLOYER)
	return StatutoryResult(
		deduction_type="ssnit",
		basis=gross_wages,
		employee_amount=employee,
		employer_amount=employer,
		rate_used=_GH_SSNIT_RATE_EMPLOYEE,
		cap_applied=False,
	)


# ---------------------------------------------------------------------------
# Nigeria PenCom — 8% employee, 10% employer (Pension Reform Act 2014)
# ---------------------------------------------------------------------------

_NG_PENCOM_RATE_EMPLOYEE = D("0.08")
_NG_PENCOM_RATE_EMPLOYER = D("0.10")


def calculate_pencom_ng(pensionable_emoluments: Decimal) -> StatutoryResult:
	employee = _round(pensionable_emoluments * _NG_PENCOM_RATE_EMPLOYEE)
	employer = _round(pensionable_emoluments * _NG_PENCOM_RATE_EMPLOYER)
	return StatutoryResult(
		deduction_type="pencom",
		basis=pensionable_emoluments,
		employee_amount=employee,
		employer_amount=employer,
		rate_used=_NG_PENCOM_RATE_EMPLOYEE,
		cap_applied=False,
	)


# ---------------------------------------------------------------------------
# Zambia NAPSA — 5% employee, 5% employer; capped at 5% of national monthly avg wage
# Assumption: cap is ZMW 1,221.80 per side (2024 NAPSA ceiling)
# ---------------------------------------------------------------------------

_ZM_NAPSA_RATE = D("0.05")
_ZM_NAPSA_CAP_EMPLOYEE = D("1221.80")
_ZM_NAPSA_CAP_EMPLOYER = D("1221.80")


def calculate_napsa_zm(gross_wages: Decimal) -> StatutoryResult:
	raw_employee = _round(gross_wages * _ZM_NAPSA_RATE)
	employee = min(raw_employee, _ZM_NAPSA_CAP_EMPLOYEE)
	employer = min(raw_employee, _ZM_NAPSA_CAP_EMPLOYER)
	return StatutoryResult(
		deduction_type="napsa",
		basis=gross_wages,
		employee_amount=employee,
		employer_amount=employer,
		rate_used=_ZM_NAPSA_RATE,
		cap_applied=raw_employee > _ZM_NAPSA_CAP_EMPLOYEE,
	)


# ---------------------------------------------------------------------------
# South Africa UIF (Unemployment Insurance Fund)
# 1% employee + 1% employer; capped at ZAR 177.12/month each side (2024)
# ---------------------------------------------------------------------------

_ZA_UIF_RATE = D("0.01")
_ZA_UIF_CAP = D("177.12")


def calculate_uif_za(gross_wages: Decimal) -> StatutoryResult:
	raw = _round(gross_wages * _ZA_UIF_RATE)
	employee = min(raw, _ZA_UIF_CAP)
	employer = min(raw, _ZA_UIF_CAP)
	return StatutoryResult(
		deduction_type="nssf",   # UIF maps to nssf slot
		basis=gross_wages,
		employee_amount=employee,
		employer_amount=employer,
		rate_used=_ZA_UIF_RATE,
		cap_applied=raw > _ZA_UIF_CAP,
		notes="UIF 1% + 1%",
	)


# ---------------------------------------------------------------------------
# Statutory dispatcher
# ---------------------------------------------------------------------------

def calculate_statutory_deductions(
	country: str,
	gross_wages: Decimal,
	pensionable_wages: Decimal | None = None,
) -> list[StatutoryResult]:
	"""
	Return all statutory deductions for a country.
	pensionable_wages defaults to gross_wages when not supplied.
	"""
	pw = pensionable_wages if pensionable_wages is not None else gross_wages
	c = country.upper()

	if c == "KE":
		return [
			calculate_nssf_ke(pw),
			calculate_nhif_ke(gross_wages),
			calculate_nita_ke(),
		]
	if c == "TZ":
		return [
			calculate_nssf_tz(gross_wages),
			calculate_sdl_tz(gross_wages),
			calculate_wcf_tz(gross_wages),
		]
	if c == "GH":
		return [calculate_ssnit_gh(gross_wages)]
	if c == "NG":
		return [calculate_pencom_ng(pw)]
	if c == "ZM":
		return [calculate_napsa_zm(gross_wages)]
	if c == "ZA":
		return [calculate_uif_za(gross_wages)]
	# UG, RW, ZW, ET — return empty; callers handle country-specific schemes
	return []


# ===========================================================================
# PRORATION
# ===========================================================================

def prorate_salary(
	monthly_salary: Decimal,
	working_days_in_month: int,
	days_worked: int,
) -> Decimal:
	"""
	Pro-rate salary for partial-month hire or termination.
	Uses calendar working-days basis.
	"""
	if working_days_in_month <= 0:
		return D("0")
	days_worked = max(0, min(days_worked, working_days_in_month))
	return _round(monthly_salary * D(days_worked) / D(working_days_in_month))


def working_days(start: date, end: date, exclude_weekends: bool = True) -> int:
	"""Count working days between start and end inclusive."""
	total = 0
	current = start
	from datetime import timedelta
	while current <= end:
		if exclude_weekends and current.weekday() >= 5:
			current += timedelta(days=1)
			continue
		total += 1
		current += timedelta(days=1)
	return total


# ===========================================================================
# OVERTIME
# ===========================================================================

def calculate_overtime_amount(
	basic_monthly_salary: Decimal,
	working_hours_per_month: int,
	overtime_hours: Decimal,
	multiplier: Decimal = D("1.5"),
) -> Decimal:
	"""
	OT amount = (basic / working_hours_per_month) * multiplier * overtime_hours
	"""
	if working_hours_per_month <= 0:
		return D("0")
	hourly_rate = _round(basic_monthly_salary / D(working_hours_per_month))
	return _round(hourly_rate * multiplier * overtime_hours)


# ===========================================================================
# LEAVE ENCASHMENT
# ===========================================================================

def encash_leave(
	daily_rate: Decimal,
	leave_days: Decimal,
) -> Decimal:
	"""Encash unused leave at the employee's daily rate."""
	return _round(daily_rate * leave_days)


def daily_rate(monthly_salary: Decimal, working_days_per_month: int = 22) -> Decimal:
	return _round(monthly_salary / D(working_days_per_month))


# ===========================================================================
# TERMINAL BENEFITS — Kenya defaults
# ===========================================================================

def calculate_notice_pay(
	monthly_salary: Decimal,
	notice_days_owed: int,
	working_days_per_month: int = 22,
) -> Decimal:
	"""Notice pay = pro-rated salary for unserved notice days."""
	return _round(monthly_salary / D(working_days_per_month) * D(notice_days_owed))


def calculate_severance_pay_ke(
	monthly_salary: Decimal,
	completed_years: int,
) -> Decimal:
	"""
	Kenya Employment Act 2007, s.35: 15 days per year of service.
	15/26 working days in a month.
	"""
	return _round(monthly_salary / D("26") * D("15") * D(completed_years))


def calculate_gratuity(
	monthly_salary: Decimal,
	years_of_service: Decimal,
	rate: Decimal = D("0.25"),   # 25% of annual salary per year (common contractual)
) -> Decimal:
	"""Contractual gratuity (not statutory in Kenya but common in contracts)."""
	annual = monthly_salary * D("12")
	return _round(annual * years_of_service * rate)


# ===========================================================================
# EXPATRIATE — Days Rule (183-day test)
# ===========================================================================

class ExpatTaxResult(NamedTuple):
	is_tax_resident: bool
	days_in_country: int
	taxable_income: Decimal
	effective_rate: Decimal
	estimated_tax: Decimal


def assess_expat_tax(
	gross_income: Decimal,
	days_in_country: int,
	home_country_tax_rate: Decimal = D("0.30"),
	host_country_paye: Decimal = D("0"),
	has_tax_equalisation: bool = False,
) -> ExpatTaxResult:
	"""
	Simple 183-day test for tax residency.
	Tax equalisation: employee pays as if in home country; employer bears the difference.
	"""
	is_resident = days_in_country >= 183
	taxable = gross_income if is_resident else D("0")
	if has_tax_equalisation:
		# Hypothetical tax at home country rate
		hypo_tax = _round(gross_income * home_country_tax_rate)
		effective = home_country_tax_rate
		est_tax = hypo_tax
	else:
		est_tax = host_country_paye if is_resident else D("0")
		effective = _round(est_tax / gross_income) if gross_income > 0 else D("0")
	return ExpatTaxResult(
		is_tax_resident=is_resident,
		days_in_country=days_in_country,
		taxable_income=taxable,
		effective_rate=effective,
		estimated_tax=est_tax,
	)


# ===========================================================================
# NET PAY
# ===========================================================================

def calculate_net_pay(
	gross_earnings: Decimal,
	total_deductions: Decimal,
	total_tax: Decimal,
) -> Decimal:
	return _round(max(D("0"), gross_earnings - total_deductions - total_tax))


# ===========================================================================
# VARIANCE CHECK
# ===========================================================================

def compute_variance_pct(current: Decimal, previous: Decimal) -> Decimal | None:
	if previous == 0:
		return None
	return _round(((current - previous) / previous) * D("100"))
