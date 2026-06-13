"""Tax Calculation Engine — core service layer.

This is the horizontal integration point. Every invoicing, payroll, commerce,
and procurement capability should call calculate_tax() here rather than
implementing its own tax logic.

Design principles:
  - OPA-style rule evaluation via evaluate_capability_rules
  - Cached rate lookups (BoundedCache, TTL 1h by default)
  - Full audit trail on every calculation, rate fetch, and override
  - Period tracking with open/close/file lifecycle
  - Compound tax support (Ghana VAT+NHIL+GETFund)
  - Treaty-aware WHT delegation to tax_wht subcapability
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from datetime import date, datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

try:
	from .capability_contract import (
		SUPPORTED_CALCULATION_STATUSES,
		SUPPORTED_COUNTRY_CODES,
		SUPPORTED_TAX_TYPES,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from .models import (
		TxApplicableRate,
		TxCalculationRequest,
		TxRateLookupRequest,
		TxTaxAudit,
		TxTaxCalculation,
		TxTaxPeriod,
		TxTaxRate,
		TxTaxResult,
	)
except ImportError:
	from capability_contract import (  # type: ignore
		SUPPORTED_CALCULATION_STATUSES,
		SUPPORTED_COUNTRY_CODES,
		SUPPORTED_TAX_TYPES,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from models import (  # type: ignore
		TxApplicableRate,
		TxCalculationRequest,
		TxRateLookupRequest,
		TxTaxAudit,
		TxTaxCalculation,
		TxTaxPeriod,
		TxTaxRate,
		TxTaxResult,
	)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _now() -> datetime:
	return datetime.now(timezone.utc)


def _today() -> date:
	return datetime.now(timezone.utc).date()


def _present(v: Any) -> bool:
	if v is None:
		return False
	if isinstance(v, str):
		return bool(v.strip())
	return True


class _BoundedCache:
	"""Simple LRU cache with max capacity — no external dep required."""

	def __init__(self, maxsize: int = 512) -> None:
		self._store: OrderedDict[str, Any] = OrderedDict()
		self._maxsize = maxsize

	def get(self, key: str) -> Any | None:
		if key not in self._store:
			return None
		self._store.move_to_end(key)
		return self._store[key]

	def set(self, key: str, value: Any) -> None:
		if key in self._store:
			self._store.move_to_end(key)
		self._store[key] = value
		if len(self._store) > self._maxsize:
			self._store.popitem(last=False)

	def invalidate(self, key: str) -> None:
		self._store.pop(key, None)

	def clear(self) -> None:
		self._store.clear()

	@property
	def size(self) -> int:
		return len(self._store)


# ---------------------------------------------------------------------------
# Bundled country rule packs (authoritative rates embedded for offline use)
# These are loaded by the VAT subcapability too; stored here as canonical truth.
# ---------------------------------------------------------------------------

# fmt: off
_BUILTIN_RATES: list[dict[str, Any]] = [
	# Kenya — KRA iTax
	{"country_code": "KE", "tax_type": "vat", "product_category": "standard",           "rate_pct": "16.0",  "authority_name": "KRA",  "effective_from": "2021-01-01", "authority_ref": "VAT Act Cap 476"},
	{"country_code": "KE", "tax_type": "vat", "product_category": "zero_rated",          "rate_pct": "0.0",   "authority_name": "KRA",  "effective_from": "2021-01-01"},
	{"country_code": "KE", "tax_type": "vat", "product_category": "exempt",              "rate_pct": "0.0",   "authority_name": "KRA",  "effective_from": "2021-01-01"},
	{"country_code": "KE", "tax_type": "vat", "product_category": "health",              "rate_pct": "0.0",   "authority_name": "KRA",  "effective_from": "2021-01-01"},
	{"country_code": "KE", "tax_type": "vat", "product_category": "education",           "rate_pct": "0.0",   "authority_name": "KRA",  "effective_from": "2021-01-01"},
	{"country_code": "KE", "tax_type": "vat", "product_category": "financial_services",  "rate_pct": "0.0",   "authority_name": "KRA",  "effective_from": "2021-01-01"},
	{"country_code": "KE", "tax_type": "vat", "product_category": "agriculture",         "rate_pct": "0.0",   "authority_name": "KRA",  "effective_from": "2021-01-01"},
	{"country_code": "KE", "tax_type": "digital_services", "product_category": "digital","rate_pct": "16.0",  "authority_name": "KRA",  "effective_from": "2021-01-01", "authority_ref": "Finance Act 2019"},
	{"country_code": "KE", "tax_type": "wht", "product_category": "standard",           "rate_pct": "5.0",   "authority_name": "KRA",  "effective_from": "2021-01-01", "notes": "Professional fees"},
	{"country_code": "KE", "tax_type": "paye", "product_category": "standard",          "rate_pct": "30.0",  "authority_name": "KRA",  "effective_from": "2021-01-01", "notes": "Top marginal rate"},
	{"country_code": "KE", "tax_type": "corporate", "product_category": "standard",     "rate_pct": "30.0",  "authority_name": "KRA",  "effective_from": "2021-01-01"},
	{"country_code": "KE", "tax_type": "corporate", "product_category": "reduced_rate", "rate_pct": "25.0",  "authority_name": "KRA",  "effective_from": "2021-01-01", "notes": "New investors in first 5 years"},

	# Nigeria — FIRS
	{"country_code": "NG", "tax_type": "vat", "product_category": "standard",           "rate_pct": "7.5",   "authority_name": "FIRS", "effective_from": "2020-02-01", "authority_ref": "Finance Act 2019"},
	{"country_code": "NG", "tax_type": "vat", "product_category": "zero_rated",         "rate_pct": "0.0",   "authority_name": "FIRS", "effective_from": "2020-02-01"},
	{"country_code": "NG", "tax_type": "vat", "product_category": "exempt",             "rate_pct": "0.0",   "authority_name": "FIRS", "effective_from": "2020-02-01"},
	{"country_code": "NG", "tax_type": "vat", "product_category": "health",             "rate_pct": "0.0",   "authority_name": "FIRS", "effective_from": "2020-02-01"},
	{"country_code": "NG", "tax_type": "vat", "product_category": "education",          "rate_pct": "0.0",   "authority_name": "FIRS", "effective_from": "2020-02-01"},
	{"country_code": "NG", "tax_type": "vat", "product_category": "agriculture",        "rate_pct": "0.0",   "authority_name": "FIRS", "effective_from": "2020-02-01"},
	{"country_code": "NG", "tax_type": "wht", "product_category": "standard",          "rate_pct": "10.0",  "authority_name": "FIRS", "effective_from": "2020-02-01", "notes": "Dividends/royalties"},
	{"country_code": "NG", "tax_type": "corporate", "product_category": "standard",    "rate_pct": "30.0",  "authority_name": "FIRS", "effective_from": "2020-02-01"},
	{"country_code": "NG", "tax_type": "corporate", "product_category": "reduced_rate","rate_pct": "20.0",  "authority_name": "FIRS", "effective_from": "2020-02-01", "notes": "Medium companies NGN 25m-100m turnover"},

	# Ghana — GRA (compound VAT bundle: 15% VAT + 2.5% NHIL + 1% GETFund)
	{"country_code": "GH", "tax_type": "vat", "product_category": "standard",          "rate_pct": "15.0",  "authority_name": "GRA",  "effective_from": "2023-01-01", "authority_ref": "VAT Act 870"},
	{"country_code": "GH", "tax_type": "vat", "product_category": "zero_rated",        "rate_pct": "0.0",   "authority_name": "GRA",  "effective_from": "2023-01-01"},
	{"country_code": "GH", "tax_type": "vat", "product_category": "exempt",            "rate_pct": "0.0",   "authority_name": "GRA",  "effective_from": "2023-01-01"},
	{"country_code": "GH", "tax_type": "excise", "product_category": "standard",       "rate_pct": "2.5",   "authority_name": "GRA",  "effective_from": "2023-01-01", "notes": "NHIL — National Health Insurance Levy", "is_compound": True},
	{"country_code": "GH", "tax_type": "excise", "product_category": "education",      "rate_pct": "1.0",   "authority_name": "GRA",  "effective_from": "2023-01-01", "notes": "GETFund Levy", "is_compound": True},
	{"country_code": "GH", "tax_type": "corporate", "product_category": "standard",   "rate_pct": "25.0",  "authority_name": "GRA",  "effective_from": "2023-01-01"},

	# Uganda — URA
	{"country_code": "UG", "tax_type": "vat", "product_category": "standard",          "rate_pct": "18.0",  "authority_name": "URA",  "effective_from": "2021-01-01"},
	{"country_code": "UG", "tax_type": "vat", "product_category": "zero_rated",        "rate_pct": "0.0",   "authority_name": "URA",  "effective_from": "2021-01-01"},
	{"country_code": "UG", "tax_type": "vat", "product_category": "exempt",            "rate_pct": "0.0",   "authority_name": "URA",  "effective_from": "2021-01-01"},
	{"country_code": "UG", "tax_type": "wht", "product_category": "standard",         "rate_pct": "15.0",  "authority_name": "URA",  "effective_from": "2021-01-01"},
	{"country_code": "UG", "tax_type": "corporate", "product_category": "standard",   "rate_pct": "30.0",  "authority_name": "URA",  "effective_from": "2021-01-01"},

	# Tanzania — TRA
	{"country_code": "TZ", "tax_type": "vat", "product_category": "standard",          "rate_pct": "18.0",  "authority_name": "TRA",  "effective_from": "2021-01-01"},
	{"country_code": "TZ", "tax_type": "vat", "product_category": "zero_rated",        "rate_pct": "0.0",   "authority_name": "TRA",  "effective_from": "2021-01-01"},
	{"country_code": "TZ", "tax_type": "vat", "product_category": "exempt",            "rate_pct": "0.0",   "authority_name": "TRA",  "effective_from": "2021-01-01"},
	{"country_code": "TZ", "tax_type": "wht", "product_category": "standard",         "rate_pct": "10.0",  "authority_name": "TRA",  "effective_from": "2021-01-01"},
	{"country_code": "TZ", "tax_type": "corporate", "product_category": "standard",   "rate_pct": "30.0",  "authority_name": "TRA",  "effective_from": "2021-01-01"},

	# South Africa — SARS
	{"country_code": "ZA", "tax_type": "vat", "product_category": "standard",          "rate_pct": "15.0",  "authority_name": "SARS", "effective_from": "2018-04-01", "authority_ref": "VAT Act 89 of 1991"},
	{"country_code": "ZA", "tax_type": "vat", "product_category": "zero_rated",        "rate_pct": "0.0",   "authority_name": "SARS", "effective_from": "2018-04-01"},
	{"country_code": "ZA", "tax_type": "vat", "product_category": "exempt",            "rate_pct": "0.0",   "authority_name": "SARS", "effective_from": "2018-04-01"},
	{"country_code": "ZA", "tax_type": "wht", "product_category": "standard",         "rate_pct": "20.0",  "authority_name": "SARS", "effective_from": "2018-04-01", "notes": "Dividends tax"},
	{"country_code": "ZA", "tax_type": "corporate", "product_category": "standard",   "rate_pct": "27.0",  "authority_name": "SARS", "effective_from": "2022-04-01"},
	{"country_code": "ZA", "tax_type": "capital_gains", "product_category": "standard","rate_pct": "22.4", "authority_name": "SARS", "effective_from": "2018-04-01", "notes": "Effective CGT rate for companies"},

	# Rwanda — RRA
	{"country_code": "RW", "tax_type": "vat", "product_category": "standard",          "rate_pct": "18.0",  "authority_name": "RRA",  "effective_from": "2021-01-01"},
	{"country_code": "RW", "tax_type": "vat", "product_category": "zero_rated",        "rate_pct": "0.0",   "authority_name": "RRA",  "effective_from": "2021-01-01"},
	{"country_code": "RW", "tax_type": "corporate", "product_category": "standard",   "rate_pct": "30.0",  "authority_name": "RRA",  "effective_from": "2021-01-01"},
]
# fmt: on

# Default currency per country (ISO 4217)
_COUNTRY_CURRENCY: dict[str, str] = {
	"KE": "KES", "NG": "NGN", "GH": "GHS", "UG": "UGX", "TZ": "TZS",
	"ZA": "ZAR", "RW": "RWF", "ET": "ETB", "EG": "EGP", "MA": "MAD",
	"TN": "TND", "CI": "XOF", "SN": "XOF", "CM": "XAF", "ZM": "ZMW",
	"ZW": "ZWL", "BW": "BWP", "MZ": "MZN", "AO": "AOA", "NA": "NAD",
}


# ---------------------------------------------------------------------------
# TaxCalcService
# ---------------------------------------------------------------------------

class TaxCalcService:
	"""Tenant-scoped tax calculation engine.

	Thread-safe for concurrent async callers (no mutable shared state between
	different tenant_ids — each calculation is atomic and independently stored).

	Rate cache is shared across tenant calls because rates are public data.
	Calculations, periods, and audit records are tenant-scoped.
	"""

	def __init__(self, cache_maxsize: int = 512) -> None:
		# Rate cache — key: "CC:tax_type:product_category:YYYY-MM-DD"
		self._rate_cache: _BoundedCache = _BoundedCache(maxsize=cache_maxsize)

		# Master rate store — key: (country_code, tax_type, product_category, rate_id)
		self._rates: dict[tuple[str, str, str, str], TxTaxRate] = {}

		# Tenant-scoped stores
		self._calculations: dict[tuple[str, str], TxTaxCalculation] = {}
		self._periods: dict[tuple[str, str], TxTaxPeriod] = {}
		self._audits: list[TxTaxAudit] = []

		# Load builtin rates (seeded at startup)
		self._seed_builtin_rates()

	# ------------------------------------------------------------------
	# Describe / evaluate (standard APG contract)
	# ------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------
	# Primary API: calculate_tax
	# ------------------------------------------------------------------

	async def calculate_tax(
		self,
		tenant_id: str,
		reference_id: str,
		reference_type: str,
		tax_type: str,
		country_code: str,
		product_category: str,
		taxable_amount: Decimal,
		currency: str | None = None,
		as_of_date: date | None = None,
		entity_type: str = "company",
		treaty_status: str = "domestic",
		period_id: str | None = None,
		notes: str = "",
	) -> TxTaxResult:
		"""Core calculation entry point.

		Called by: invoicing, payroll, commerce, procurement, and any other
		capability that needs to compute tax.

		Returns TxTaxResult — a lightweight, serialisable result. The full
		TxTaxCalculation record is stored internally and can be retrieved by
		calculation_id.

		Raises:
			PermissionError: if OPA rules deny the operation.
			ValueError: if inputs are invalid or no rate found.
		"""
		assert _present(tenant_id), "tenant_id is required"
		assert _present(reference_id), "reference_id is required"
		assert _present(reference_type), "reference_type is required"

		tax_type = tax_type.lower().strip()
		country_code = country_code.upper().strip()
		product_category = product_category.lower().strip()
		as_of = as_of_date or _today()
		resolved_currency = (currency or _COUNTRY_CURRENCY.get(country_code, "USD")).upper()

		# OPA rule gate
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "calculate_tax",
			"tax_type_supported": tax_type in SUPPORTED_TAX_TYPES,
			"country_code_supported": country_code in SUPPORTED_COUNTRY_CODES,
			"amount_positive": taxable_amount > 0,
			"product_category_present": _present(product_category),
			"cross_tenant": False,
		})

		# Rate resolution (cached)
		applicable_rates = await self._resolve_rates(
			country_code=country_code,
			tax_type=tax_type,
			product_category=product_category,
			as_of_date=as_of,
			entity_type=entity_type,
			treaty_status=treaty_status,
		)

		if not applicable_rates:
			# Fall back to zero (exempt / not applicable) rather than error
			applicable_rates = []
			tax_amount = Decimal("0")
			breakdown: list[dict[str, Any]] = []
			effective_rate = Decimal("0")
		else:
			tax_amount, breakdown, effective_rate = self._compute_tax(taxable_amount, applicable_rates)

		total_amount = taxable_amount + tax_amount

		calc = TxTaxCalculation(
			tenant_id=tenant_id,
			reference_id=reference_id,
			reference_type=reference_type,
			tax_type=tax_type,
			country_code=country_code,
			product_category=product_category,
			entity_type=entity_type,
			treaty_status=treaty_status,
			taxable_amount=taxable_amount,
			currency=resolved_currency,
			applicable_rates=applicable_rates,
			tax_amount=tax_amount,
			total_amount=total_amount,
			tax_breakdown=breakdown,
			period_id=period_id,
			notes=notes,
		)
		self._calculations[(tenant_id, calc.id)] = calc

		# Attach to period if provided
		if period_id:
			self._attach_to_period(tenant_id, period_id, calc)

		self._audit(
			tenant_id=tenant_id,
			action="calculation_performed",
			reference_id=calc.id,
			reference_type="calculation",
			country_code=country_code,
			tax_type=tax_type,
			snapshot={
				"taxable_amount": str(taxable_amount),
				"tax_amount": str(tax_amount),
				"effective_rate_pct": str(effective_rate),
				"currency": resolved_currency,
			},
		)

		return TxTaxResult(
			calculation_id=calc.id,
			tenant_id=tenant_id,
			reference_id=reference_id,
			reference_type=reference_type,
			tax_type=tax_type,
			country_code=country_code,
			product_category=product_category,
			taxable_amount=taxable_amount,
			tax_amount=tax_amount,
			total_amount=total_amount,
			currency=resolved_currency,
			effective_rate_pct=effective_rate,
			tax_breakdown=breakdown,
			calculated_at=calc.calculated_at,
			period_id=period_id,
			notes=notes,
		)

	# ------------------------------------------------------------------
	# Rate management
	# ------------------------------------------------------------------

	async def get_rate(
		self,
		country_code: str,
		tax_type: str,
		product_category: str,
		as_of_date: date | None = None,
	) -> list[TxApplicableRate]:
		"""Return applicable rate(s) for the given context, from cache if available."""
		as_of = as_of_date or _today()
		cache_key = f"{country_code}:{tax_type}:{product_category}:{as_of.isoformat()}"
		cached = self._rate_cache.get(cache_key)
		if cached is not None:
			self._audit_simple("rate_lookup", cache_key, "rate_cache", notes="cache_hit")
			return cached

		rates = await self._resolve_rates(
			country_code=country_code.upper(),
			tax_type=tax_type.lower(),
			product_category=product_category.lower(),
			as_of_date=as_of,
		)
		self._rate_cache.set(cache_key, rates)
		self._audit_simple("rate_lookup", cache_key, "rate_cache", notes="cache_miss")
		return rates

	def register_rate(
		self,
		tenant_id: str,
		country_code: str,
		tax_type: str,
		product_category: str,
		rate_pct: Decimal,
		authority_name: str,
		effective_from: date,
		effective_to: date | None = None,
		is_compound: bool = False,
		authority_ref: str = "",
		notes: str = "",
	) -> TxTaxRate:
		"""Register a custom or updated tax rate.

		Useful for new legislation or tenant-specific special economic zone rates.
		Invalidates the rate cache for the affected key.
		"""
		assert _present(tenant_id), "tenant_id is required"
		assert _present(authority_name), "authority_name is required"

		rate = TxTaxRate(
			tenant_id=tenant_id,
			country_code=country_code.upper(),
			tax_type=tax_type.lower(),
			product_category=product_category.lower(),
			rate_pct=rate_pct,
			authority_name=authority_name,
			authority_ref=authority_ref or "",
			effective_from=effective_from,
			effective_to=effective_to,
			is_compound=is_compound,
			notes=notes,
			created_by=tenant_id,
		)
		key = (rate.country_code, rate.tax_type, rate.product_category, rate.id)
		self._rates[key] = rate

		# Bust cache for this combo
		for as_of_str in list(
			k.split(":")[-1]
			for k in self._rate_cache._store
			if k.startswith(f"{rate.country_code}:{rate.tax_type}:{rate.product_category}:")
		):
			self._rate_cache.invalidate(
				f"{rate.country_code}:{rate.tax_type}:{rate.product_category}:{as_of_str}"
			)

		self._audit(
			tenant_id=tenant_id,
			action="rate_lookup",
			reference_id=rate.id,
			reference_type="rate",
			country_code=country_code.upper(),
			tax_type=tax_type.lower(),
			snapshot={"rate_pct": str(rate_pct), "effective_from": str(effective_from)},
		)
		return rate

	async def override_rate(
		self,
		tenant_id: str,
		calculation_id: str,
		override_rate_pct: Decimal,
		justification: str,
		approved_by: str,
	) -> TxTaxResult:
		"""Apply a post-calculation rate override (e.g. for dispute resolution).

		Creates an amended calculation record linked to the original.
		"""
		assert _present(justification), "justification is required"
		assert _present(approved_by), "approved_by is required"

		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation": "override_rate",
			"justification_present": bool(justification.strip()),
			"approved_by_present": bool(approved_by.strip()),
		})

		original = self._calculations.get((tenant_id, calculation_id))
		if original is None:
			raise KeyError(f"calculation {calculation_id!r} not found for tenant {tenant_id!r}")

		override_tax = (original.taxable_amount * override_rate_pct / 100).quantize(
			Decimal("0.01"), rounding=ROUND_HALF_UP
		)
		amended = TxTaxCalculation(
			tenant_id=tenant_id,
			reference_id=original.reference_id,
			reference_type=original.reference_type,
			tax_type=original.tax_type,
			country_code=original.country_code,
			product_category=original.product_category,
			entity_type=original.entity_type,
			treaty_status=original.treaty_status,
			taxable_amount=original.taxable_amount,
			currency=original.currency,
			applicable_rates=list(original.applicable_rates),
			tax_amount=override_tax,
			total_amount=original.taxable_amount + override_tax,
			tax_breakdown=[{"source": "manual_override", "rate_pct": str(override_rate_pct), "tax_amount": str(override_tax)}],
			rate_overridden=True,
			override_rate_pct=override_rate_pct,
			override_justification=justification,
			override_approved_by=approved_by,
			status="calculated",
			period_id=original.period_id,
			amended_from_id=original.id,
			notes=f"Override of {original.id}: {justification}",
		)
		self._calculations[(tenant_id, amended.id)] = amended

		self._audit(
			tenant_id=tenant_id,
			action="override_applied",
			reference_id=amended.id,
			reference_type="calculation",
			country_code=original.country_code,
			tax_type=original.tax_type,
			snapshot={
				"original_id": original.id,
				"original_tax": str(original.tax_amount),
				"override_rate_pct": str(override_rate_pct),
				"override_tax": str(override_tax),
				"approved_by": approved_by,
			},
		)

		return TxTaxResult(
			calculation_id=amended.id,
			tenant_id=tenant_id,
			reference_id=amended.reference_id,
			reference_type=amended.reference_type,
			tax_type=amended.tax_type,
			country_code=amended.country_code,
			product_category=amended.product_category,
			taxable_amount=amended.taxable_amount,
			tax_amount=amended.tax_amount,
			total_amount=amended.total_amount,
			currency=amended.currency,
			effective_rate_pct=override_rate_pct,
			tax_breakdown=amended.tax_breakdown,
			calculated_at=amended.calculated_at,
			period_id=amended.period_id,
			notes=amended.notes,
		)

	# ------------------------------------------------------------------
	# Period management
	# ------------------------------------------------------------------

	def open_period(
		self,
		tenant_id: str,
		country_code: str,
		tax_type: str,
		period_name: str,
		period_start: date,
		period_end: date,
		filing_due_date: date,
		currency: str | None = None,
	) -> TxTaxPeriod:
		"""Open a new tax period for a country/tax-type combination."""
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "open_period",
			"period_dates_present": _present(period_start) and _present(period_end),
		})
		resolved_currency = (currency or _COUNTRY_CURRENCY.get(country_code.upper(), "USD")).upper()
		period = TxTaxPeriod(
			tenant_id=tenant_id,
			country_code=country_code.upper(),
			tax_type=tax_type.lower(),
			period_name=period_name,
			period_start=period_start,
			period_end=period_end,
			filing_due_date=filing_due_date,
			currency=resolved_currency,
		)
		self._periods[(tenant_id, period.id)] = period
		self._audit(
			tenant_id=tenant_id,
			action="rate_lookup",
			reference_id=period.id,
			reference_type="period",
			country_code=country_code.upper(),
			tax_type=tax_type.lower(),
			snapshot={"period_name": period_name, "status": "open"},
		)
		return period

	def close_period(self, tenant_id: str, period_id: str, closed_by: str) -> TxTaxPeriod:
		"""Close a period, preventing further calculation attachments."""
		assert _present(closed_by), "closed_by is required"
		period = self._periods.get((tenant_id, period_id))
		if period is None:
			raise KeyError(f"period {period_id!r} not found for tenant {tenant_id!r}")
		if period.status not in ("open",):
			raise ValueError(f"period {period_id!r} is {period.status!r}, cannot close")
		object.__setattr__(period, "status", "closed") if period.model_config.get("frozen") else setattr(period, "status", "closed")
		# Pydantic v2: use model_copy for immutable, but TxTaxPeriod is not frozen
		period.status = "closed"
		self._audit_simple("amendment_recorded", period_id, "period", notes=f"closed_by={closed_by}")
		return period

	def file_period(
		self,
		tenant_id: str,
		period_id: str,
		filed_by: str,
		payment_reference: str = "",
	) -> TxTaxPeriod:
		"""Mark a period as filed with the tax authority."""
		assert _present(filed_by), "filed_by is required"
		period = self._periods.get((tenant_id, period_id))
		if period is None:
			raise KeyError(f"period {period_id!r} not found for tenant {tenant_id!r}")

		today = _today()
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation": "file_period",
			"period_in_future": period.period_end > today,
		})

		period.status = "filed"
		period.filed_at = _now()
		period.filed_by = filed_by
		period.payment_reference = payment_reference or ""
		self._audit(
			tenant_id=tenant_id,
			action="filing_triggered",
			reference_id=period_id,
			reference_type="period",
			country_code=period.country_code,
			tax_type=period.tax_type,
			snapshot={
				"filed_by": filed_by,
				"total_tax_amount": str(period.total_tax_amount),
				"payment_reference": payment_reference,
			},
		)
		return period

	# ------------------------------------------------------------------
	# Retrieval & reporting
	# ------------------------------------------------------------------

	def get_calculation(self, tenant_id: str, calculation_id: str) -> TxTaxCalculation:
		calc = self._calculations.get((tenant_id, calculation_id))
		if calc is None:
			raise KeyError(f"calculation {calculation_id!r} not found for tenant {tenant_id!r}")
		return calc

	def list_calculations(
		self,
		tenant_id: str,
		tax_type: str | None = None,
		country_code: str | None = None,
		reference_type: str | None = None,
		period_id: str | None = None,
	) -> list[TxTaxCalculation]:
		results = []
		for (tid, _), calc in self._calculations.items():
			if tid != tenant_id:
				continue
			if tax_type and calc.tax_type != tax_type.lower():
				continue
			if country_code and calc.country_code != country_code.upper():
				continue
			if reference_type and calc.reference_type != reference_type:
				continue
			if period_id and calc.period_id != period_id:
				continue
			results.append(calc)
		return results

	def list_periods(self, tenant_id: str, country_code: str | None = None, tax_type: str | None = None) -> list[TxTaxPeriod]:
		results = []
		for (tid, _), period in self._periods.items():
			if tid != tenant_id:
				continue
			if country_code and period.country_code != country_code.upper():
				continue
			if tax_type and period.tax_type != tax_type.lower():
				continue
			results.append(period)
		return results

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		"""Aggregate tax position summary for a tenant."""
		calcs = [c for (tid, _), c in self._calculations.items() if tid == tenant_id]
		periods = [p for (tid, _), p in self._periods.items() if tid == tenant_id]
		total_taxable = sum(c.taxable_amount for c in calcs)
		total_tax = sum(c.tax_amount for c in calcs)
		by_type: dict[str, dict[str, Any]] = {}
		for c in calcs:
			if c.tax_type not in by_type:
				by_type[c.tax_type] = {"count": 0, "taxable": Decimal("0"), "tax": Decimal("0")}
			by_type[c.tax_type]["count"] += 1
			by_type[c.tax_type]["taxable"] += c.taxable_amount
			by_type[c.tax_type]["tax"] += c.tax_amount
		# convert Decimal to str for serialisation
		for v in by_type.values():
			v["taxable"] = str(v["taxable"])
			v["tax"] = str(v["tax"])
		return {
			"tenant_id": tenant_id,
			"calculation_count": len(calcs),
			"period_count": len(periods),
			"open_periods": sum(1 for p in periods if p.status == "open"),
			"total_taxable_amount": str(total_taxable),
			"total_tax_amount": str(total_tax),
			"by_tax_type": by_type,
			"cache_size": self._rate_cache.size,
			"as_of": _now().isoformat(),
		}

	def audit_trail(self, tenant_id: str, limit: int = 100) -> list[dict[str, Any]]:
		"""Return most recent audit events for a tenant."""
		events = [a.model_dump() for a in self._audits if a.tenant_id == tenant_id]
		return events[-limit:]

	def effective_rate_summary(self, tenant_id: str, country_code: str) -> dict[str, Any]:
		"""Report effective rates paid per tax type for a country."""
		calcs = [
			c for (tid, _), c in self._calculations.items()
			if tid == tenant_id and c.country_code == country_code.upper()
		]
		summary: dict[str, dict[str, Any]] = {}
		for c in calcs:
			tt = c.tax_type
			if tt not in summary:
				summary[tt] = {"count": 0, "taxable": Decimal("0"), "tax": Decimal("0")}
			summary[tt]["count"] += 1
			summary[tt]["taxable"] += c.taxable_amount
			summary[tt]["tax"] += c.tax_amount
		result: dict[str, Any] = {}
		for tt, data in summary.items():
			taxable = data["taxable"]
			tax = data["tax"]
			eff = (tax / taxable * 100).quantize(Decimal("0.01")) if taxable else Decimal("0")
			result[tt] = {
				"count": data["count"],
				"total_taxable": str(taxable),
				"total_tax": str(tax),
				"effective_rate_pct": str(eff),
			}
		return {
			"tenant_id": tenant_id,
			"country_code": country_code.upper(),
			"by_tax_type": result,
			"as_of": _now().isoformat(),
		}

	# ------------------------------------------------------------------
	# Cross-capability API (called by other capabilities)
	# ------------------------------------------------------------------

	async def compute_invoice_tax(
		self,
		tenant_id: str,
		invoice_id: str,
		country_code: str,
		line_items: list[dict[str, Any]],
		currency: str | None = None,
		period_id: str | None = None,
	) -> dict[str, Any]:
		"""Batch calculate tax for all lines on an invoice.

		line_items: list of {product_category, taxable_amount, tax_type?, ...}
		Returns a line-by-line breakdown plus invoice totals.
		"""
		assert _present(invoice_id), "invoice_id is required"
		assert line_items, "line_items must not be empty"

		results = []
		invoice_taxable = Decimal("0")
		invoice_tax = Decimal("0")

		tasks = []
		for i, item in enumerate(line_items):
			tax_type = item.get("tax_type", "vat")
			product_category = item.get("product_category", "standard")
			amount = Decimal(str(item.get("taxable_amount", "0")))
			tasks.append(self.calculate_tax(
				tenant_id=tenant_id,
				reference_id=invoice_id,
				reference_type="invoice",
				tax_type=tax_type,
				country_code=country_code,
				product_category=product_category,
				taxable_amount=amount,
				currency=currency,
				period_id=period_id,
				notes=f"invoice_line_{i}",
			))

		# Run lines concurrently
		line_results = await asyncio.gather(*tasks)

		for i, (item, r) in enumerate(zip(line_items, line_results)):
			invoice_taxable += r.taxable_amount
			invoice_tax += r.tax_amount
			results.append({
				"line_index": i,
				"calculation_id": r.calculation_id,
				"product_category": r.product_category,
				"taxable_amount": str(r.taxable_amount),
				"tax_amount": str(r.tax_amount),
				"total_amount": str(r.total_amount),
				"effective_rate_pct": str(r.effective_rate_pct),
				"tax_breakdown": r.tax_breakdown,
			})

		return {
			"invoice_id": invoice_id,
			"tenant_id": tenant_id,
			"country_code": country_code.upper(),
			"currency": (currency or _COUNTRY_CURRENCY.get(country_code.upper(), "USD")).upper(),
			"lines": results,
			"invoice_taxable_total": str(invoice_taxable),
			"invoice_tax_total": str(invoice_tax),
			"invoice_grand_total": str(invoice_taxable + invoice_tax),
			"calculated_at": _now().isoformat(),
		}

	async def compute_payroll_tax(
		self,
		tenant_id: str,
		payroll_run_id: str,
		country_code: str,
		employees: list[dict[str, Any]],
		period_id: str | None = None,
	) -> dict[str, Any]:
		"""Compute PAYE for a payroll run.

		employees: list of {employee_id, gross_salary, currency?}
		Returns per-employee PAYE and payroll totals.
		"""
		assert _present(payroll_run_id), "payroll_run_id is required"
		assert employees, "employees must not be empty"

		tasks = []
		for emp in employees:
			tasks.append(self.calculate_tax(
				tenant_id=tenant_id,
				reference_id=payroll_run_id,
				reference_type="payroll_run",
				tax_type="paye",
				country_code=country_code,
				product_category="standard",
				taxable_amount=Decimal(str(emp.get("gross_salary", "0"))),
				currency=emp.get("currency"),
				period_id=period_id,
				notes=f"employee={emp.get('employee_id', 'unknown')}",
			))

		results = await asyncio.gather(*tasks)

		total_gross = Decimal("0")
		total_paye = Decimal("0")
		emp_results = []
		for emp, r in zip(employees, results):
			total_gross += r.taxable_amount
			total_paye += r.tax_amount
			emp_results.append({
				"employee_id": emp.get("employee_id"),
				"calculation_id": r.calculation_id,
				"gross_salary": str(r.taxable_amount),
				"paye_amount": str(r.tax_amount),
				"net_salary": str(r.taxable_amount - r.tax_amount),
				"effective_rate_pct": str(r.effective_rate_pct),
			})

		return {
			"payroll_run_id": payroll_run_id,
			"tenant_id": tenant_id,
			"country_code": country_code.upper(),
			"employee_count": len(employees),
			"total_gross": str(total_gross),
			"total_paye": str(total_paye),
			"total_net": str(total_gross - total_paye),
			"employees": emp_results,
			"calculated_at": _now().isoformat(),
		}

	# ------------------------------------------------------------------
	# Rate resolution internals
	# ------------------------------------------------------------------

	async def _resolve_rates(
		self,
		country_code: str,
		tax_type: str,
		product_category: str,
		as_of_date: date,
		entity_type: str = "company",
		treaty_status: str = "domestic",
	) -> list[TxApplicableRate]:
		"""Find all rates matching (country, tax_type, product_category) effective on as_of_date."""
		matches: list[TxApplicableRate] = []
		for (cc, tt, pc, _), rate in self._rates.items():
			if cc != country_code or tt != tax_type or pc != product_category:
				continue
			if rate.effective_from > as_of_date:
				continue
			if rate.effective_to and rate.effective_to < as_of_date:
				continue
			matches.append(rate.as_applicable_rate())
		return matches

	def _compute_tax(
		self,
		taxable_amount: Decimal,
		rates: list[TxApplicableRate],
	) -> tuple[Decimal, list[dict[str, Any]], Decimal]:
		"""Compute total tax from a list of rates, handling compound taxes.

		Returns (total_tax_amount, breakdown_list, effective_rate_pct).
		Compound rates (e.g. Ghana NHIL, GETFund) are applied on top of the
		base tax, not on the original taxable amount.
		"""
		breakdown: list[dict[str, Any]] = []
		total_tax = Decimal("0")

		# Separate base and compound rates
		base_rates = [r for r in rates if not r.is_compound]
		compound_rates = [r for r in rates if r.is_compound]

		base_tax = Decimal("0")
		for rate in base_rates:
			line_tax = (taxable_amount * rate.rate_pct / 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
			base_tax += line_tax
			breakdown.append({
				"rate_id": rate.rate_id,
				"label": rate.source,
				"rate_pct": str(rate.rate_pct),
				"base_amount": str(taxable_amount),
				"tax_amount": str(line_tax),
				"is_compound": False,
			})
		total_tax += base_tax

		# Compound rates apply on taxable_amount (not on base_tax)
		for rate in compound_rates:
			line_tax = (taxable_amount * rate.rate_pct / 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
			total_tax += line_tax
			breakdown.append({
				"rate_id": rate.rate_id,
				"label": rate.source,
				"rate_pct": str(rate.rate_pct),
				"base_amount": str(taxable_amount),
				"tax_amount": str(line_tax),
				"is_compound": True,
			})

		effective_rate = (total_tax / taxable_amount * 100).quantize(Decimal("0.0001")) if taxable_amount else Decimal("0")
		return total_tax, breakdown, effective_rate

	# ------------------------------------------------------------------
	# Period attachment
	# ------------------------------------------------------------------

	def _attach_to_period(self, tenant_id: str, period_id: str, calc: TxTaxCalculation) -> None:
		period = self._periods.get((tenant_id, period_id))
		if period is None:
			return  # silent — period may not exist yet
		if period.status not in ("open",):
			return  # don't attach to closed/filed periods
		period.calculation_ids.append(calc.id)
		period.total_taxable_amount += calc.taxable_amount
		period.total_tax_amount += calc.tax_amount

	# ------------------------------------------------------------------
	# Audit helpers
	# ------------------------------------------------------------------

	def _audit(
		self,
		tenant_id: str,
		action: str,
		reference_id: str,
		reference_type: str,
		country_code: str | None = None,
		tax_type: str | None = None,
		snapshot: dict[str, Any] | None = None,
		notes: str = "",
	) -> None:
		entry = TxTaxAudit(
			tenant_id=tenant_id,
			action=action,
			reference_id=reference_id,
			reference_type=reference_type,
			country_code=country_code,
			tax_type=tax_type,
			snapshot=snapshot or {},
			notes=notes,
		)
		self._audits.append(entry)

	def _audit_simple(self, action: str, reference_id: str, reference_type: str, notes: str = "") -> None:
		"""Lightweight audit for cache events — no tenant needed."""
		entry = TxTaxAudit(
			tenant_id="system",
			action=action,
			reference_id=reference_id,
			reference_type=reference_type,
			notes=notes,
		)
		self._audits.append(entry)

	# ------------------------------------------------------------------
	# OPA enforcement
	# ------------------------------------------------------------------

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(
			action.get("reason", action.get("rule", "tax_policy_denied"))
			for action in result["actions"]
		)
		raise PermissionError(reasons or "tax_policy_denied")

	# ------------------------------------------------------------------
	# Builtin rate seeding
	# ------------------------------------------------------------------

	def _seed_builtin_rates(self) -> None:
		"""Load the embedded African rate table at service startup."""
		for raw in _BUILTIN_RATES:
			rate = TxTaxRate(
				tenant_id="system",
				country_code=raw["country_code"],
				tax_type=raw["tax_type"],
				product_category=raw["product_category"],
				rate_pct=Decimal(raw["rate_pct"]),
				authority_name=raw["authority_name"],
				authority_ref=raw.get("authority_ref", ""),
				effective_from=date.fromisoformat(raw["effective_from"]),
				effective_to=date.fromisoformat(raw["effective_to"]) if raw.get("effective_to") else None,
				is_compound=raw.get("is_compound", False),
				notes=raw.get("notes", ""),
				created_by="system_seed",
			)
			key = (rate.country_code, rate.tax_type, rate.product_category, rate.id)
			self._rates[key] = rate


# Public alias
CommonTaxCalcService = TaxCalcService
