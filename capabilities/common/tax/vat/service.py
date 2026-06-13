"""VAT/GST Country Rule Pack service.

Provides:
  - get_vat_rate()  — rate lookup with Ghana compound-levy expansion
  - submit_return() — VAT return lifecycle
  - register_exemption() / is_exempt() — exemption registry
  - Country config store with filing-frequency and threshold data

Delegates actual calculation to TaxCalcService (calc subcapability).
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any

try:
	from .capability_contract import (
		SUPPORTED_COUNTRY_CODES,
		SUPPORTED_EXEMPTION_TYPES,
		SUPPORTED_RETURN_STATUSES,
		SUPPORTED_VAT_CATEGORIES,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from .models import TxVatCountryConfig, TxVatExemption, TxVatRate, TxVatReturn
except ImportError:
	from capability_contract import (  # type: ignore
		SUPPORTED_COUNTRY_CODES,
		SUPPORTED_EXEMPTION_TYPES,
		SUPPORTED_RETURN_STATUSES,
		SUPPORTED_VAT_CATEGORIES,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from models import TxVatCountryConfig, TxVatExemption, TxVatRate, TxVatReturn  # type: ignore


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


# ---------------------------------------------------------------------------
# Embedded country configurations (authoritative)
# ---------------------------------------------------------------------------

_COUNTRY_CONFIGS: list[dict[str, Any]] = [
	{
		"country_code": "KE",
		"authority_name": "KRA",
		"authority_website": "https://www.kra.go.ke",
		"registration_threshold_local": "5000000",  # KES 5m annual turnover
		"threshold_currency": "KES",
		"filing_frequency": "monthly",
		"standard_rate_pct": "16.0",
		"has_compound_levies": False,
		"digital_services_tax_pct": "16.0",
		"notes": "KRA iTax system. Monthly VAT returns due by 20th of following month.",
	},
	{
		"country_code": "NG",
		"authority_name": "FIRS",
		"authority_website": "https://www.firs.gov.ng",
		"registration_threshold_local": "25000000",  # NGN 25m
		"threshold_currency": "NGN",
		"filing_frequency": "monthly",
		"standard_rate_pct": "7.5",
		"has_compound_levies": False,
		"notes": "Finance Act 2019 reduced rate from 5% to 7.5%. Monthly returns due 21st.",
	},
	{
		"country_code": "GH",
		"authority_name": "GRA",
		"authority_website": "https://www.gra.gov.gh",
		"registration_threshold_local": "200000",  # GHS 200k
		"threshold_currency": "GHS",
		"filing_frequency": "monthly",
		"standard_rate_pct": "15.0",
		"has_compound_levies": True,
		"compound_levy_names": ["NHIL", "GETFund"],
		"notes": "VAT 15% + NHIL 2.5% + GETFund 1% = effective 18.5% on standard supplies.",
	},
	{
		"country_code": "UG",
		"authority_name": "URA",
		"authority_website": "https://www.ura.go.ug",
		"registration_threshold_local": "150000000",  # UGX 150m
		"threshold_currency": "UGX",
		"filing_frequency": "monthly",
		"standard_rate_pct": "18.0",
		"has_compound_levies": False,
		"notes": "URA e-Tax. Monthly returns due last working day of following month.",
	},
	{
		"country_code": "TZ",
		"authority_name": "TRA",
		"authority_website": "https://www.tra.go.tz",
		"registration_threshold_local": "100000000",  # TZS 100m
		"threshold_currency": "TZS",
		"filing_frequency": "monthly",
		"standard_rate_pct": "18.0",
		"has_compound_levies": False,
		"notes": "TRA. Monthly returns due 20th of following month.",
	},
	{
		"country_code": "ZA",
		"authority_name": "SARS",
		"authority_website": "https://www.sars.gov.za",
		"registration_threshold_local": "1000000",  # ZAR 1m
		"threshold_currency": "ZAR",
		"filing_frequency": "monthly",
		"standard_rate_pct": "15.0",
		"has_compound_levies": False,
		"notes": "SARS eFiling. Rate increased from 14% to 15% in April 2018.",
	},
	{
		"country_code": "RW",
		"authority_name": "RRA",
		"authority_website": "https://www.rra.gov.rw",
		"registration_threshold_local": "20000000",  # RWF 20m
		"threshold_currency": "RWF",
		"filing_frequency": "monthly",
		"standard_rate_pct": "18.0",
		"has_compound_levies": False,
	},
]


class VatService:
	"""Tenant-scoped VAT/GST rule pack service."""

	def __init__(self) -> None:
		# Rate store: (country_code, vat_category) -> list[TxVatRate]
		self._rates: dict[tuple[str, str], list[TxVatRate]] = {}
		# Country configs
		self._country_configs: dict[str, TxVatCountryConfig] = {}
		# Tenant-scoped returns: (tenant_id, return_id)
		self._returns: dict[tuple[str, str], TxVatReturn] = {}
		# Tenant-scoped exemptions: (tenant_id, exemption_id)
		self._exemptions: dict[tuple[str, str], TxVatExemption] = {}

		self._seed_country_configs()

	# ------------------------------------------------------------------
	# Describe / evaluate
	# ------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------
	# Primary API: get_vat_rate
	# ------------------------------------------------------------------

	def get_vat_rate(
		self,
		country_code: str,
		product_category: str,
		as_of_date: date | None = None,
	) -> list[TxVatRate]:
		"""Return active VAT rate(s) for country + category on as_of_date.

		Ghana returns multiple entries (VAT + NHIL + GETFund).
		Most countries return a single entry.
		"""
		as_of = as_of_date or _today()
		country_code = country_code.upper().strip()
		product_category = product_category.lower().strip()

		self._enforce({
			"tenant_context_present": True,
			"operation": "get_vat_rate",
			"country_code_supported": country_code in SUPPORTED_COUNTRY_CODES,
			"vat_category_supported": product_category in SUPPORTED_VAT_CATEGORIES,
		})

		key = (country_code, product_category)
		rates = self._rates.get(key, [])

		# Filter by effective date
		active = [
			r for r in rates
			if r.effective_from <= as_of and (r.effective_to is None or r.effective_to >= as_of)
		]

		# If no explicit entry, fall back to country standard rate for category
		if not active:
			config = self._country_configs.get(country_code)
			if config:
				# Return a synthetic rate record
				synthetic = TxVatRate(
					country_code=country_code,
					vat_category=product_category,
					rate_pct=config.standard_rate_pct if product_category == "standard" else Decimal("0"),
					authority_name=config.authority_name,
					effective_from=as_of,
					notes="synthetic_fallback",
				)
				return [synthetic]

		return active

	def get_effective_vat_rate_pct(
		self,
		country_code: str,
		product_category: str,
		as_of_date: date | None = None,
	) -> Decimal:
		"""Return the blended effective VAT rate (sum of all applicable entries)."""
		rates = self.get_vat_rate(country_code, product_category, as_of_date)
		return sum((r.rate_pct for r in rates), Decimal("0"))

	# ------------------------------------------------------------------
	# Rate registration (for custom or updated country rules)
	# ------------------------------------------------------------------

	def register_vat_rate(
		self,
		tenant_id: str,
		country_code: str,
		vat_category: str,
		rate_pct: Decimal,
		authority_name: str,
		effective_from: date,
		effective_to: date | None = None,
		is_levy: bool = False,
		levy_name: str = "",
		authority_ref: str = "",
		notes: str = "",
	) -> TxVatRate:
		"""Register or update a VAT rate for a country/category combination."""
		self._enforce({
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
		})
		rate = TxVatRate(
			country_code=country_code.upper(),
			vat_category=vat_category.lower(),
			rate_pct=rate_pct,
			authority_name=authority_name,
			authority_ref=authority_ref,
			effective_from=effective_from,
			effective_to=effective_to,
			is_levy=is_levy,
			levy_name=levy_name,
			notes=notes,
		)
		key = (rate.country_code, rate.vat_category)
		if key not in self._rates:
			self._rates[key] = []
		self._rates[key].append(rate)
		return rate

	# ------------------------------------------------------------------
	# VAT returns
	# ------------------------------------------------------------------

	def create_return(
		self,
		tenant_id: str,
		country_code: str,
		tax_period_id: str,
		period_name: str,
		period_start: date,
		period_end: date,
		filing_due_date: date,
		output_vat: Decimal,
		input_vat: Decimal,
		currency: str | None = None,
	) -> TxVatReturn:
		"""Create a draft VAT return for the period."""
		assert _present(tenant_id), "tenant_id is required"
		assert _present(tax_period_id), "tax_period_id is required"

		config = self._country_configs.get(country_code.upper())
		resolved_currency = currency or (config.threshold_currency if config and config.threshold_currency else "USD")

		vat_return = TxVatReturn(
			tenant_id=tenant_id,
			country_code=country_code.upper(),
			tax_period_id=tax_period_id,
			period_name=period_name,
			period_start=period_start,
			period_end=period_end,
			filing_due_date=filing_due_date,
			output_vat=output_vat,
			input_vat=input_vat,
			currency=resolved_currency,
			status="draft",
		)
		self._returns[(tenant_id, vat_return.id)] = vat_return
		return vat_return

	def submit_return(
		self,
		tenant_id: str,
		return_id: str,
		submitted_by: str,
	) -> TxVatReturn:
		"""Transition a draft return to submitted status."""
		assert _present(submitted_by), "submitted_by is required"
		vat_return = self._returns.get((tenant_id, return_id))
		if vat_return is None:
			raise KeyError(f"VAT return {return_id!r} not found for tenant {tenant_id!r}")

		self._enforce({
			"tenant_context_present": bool(tenant_id),
			"operation": "submit_return",
			"status_supported": "submitted" in SUPPORTED_RETURN_STATUSES,
			"period_in_future": vat_return.period_end > _today(),
		})

		vat_return.status = "submitted"
		vat_return.submitted_at = _now()
		vat_return.submitted_by = submitted_by
		return vat_return

	def accept_return(
		self,
		tenant_id: str,
		return_id: str,
		authority_reference: str,
	) -> TxVatReturn:
		"""Record acceptance by tax authority (e.g. after iTax acknowledgement)."""
		assert _present(authority_reference), "authority_reference is required"
		vat_return = self._returns.get((tenant_id, return_id))
		if vat_return is None:
			raise KeyError(f"VAT return {return_id!r} not found")
		vat_return.status = "accepted"
		vat_return.authority_reference = authority_reference
		return vat_return

	def list_returns(
		self,
		tenant_id: str,
		country_code: str | None = None,
		status: str | None = None,
	) -> list[TxVatReturn]:
		results = []
		for (tid, _), r in self._returns.items():
			if tid != tenant_id:
				continue
			if country_code and r.country_code != country_code.upper():
				continue
			if status and r.status != status.lower():
				continue
			results.append(r)
		return results

	# ------------------------------------------------------------------
	# Exemptions
	# ------------------------------------------------------------------

	def register_exemption(
		self,
		tenant_id: str,
		country_code: str,
		exemption_type: str,
		entity_reference: str,
		evidence_reference: str,
		granted_from: date,
		expires_at: date | None = None,
		authority_ref: str = "",
		notes: str = "",
	) -> TxVatExemption:
		"""Register a VAT exemption for a product, entity, or transaction type."""
		self._enforce({
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_exemption",
			"evidence_present": _present(evidence_reference),
		})
		exemption = TxVatExemption(
			tenant_id=tenant_id,
			country_code=country_code.upper(),
			exemption_type=exemption_type.lower(),
			entity_reference=entity_reference,
			evidence_reference=evidence_reference,
			granted_from=granted_from,
			expires_at=expires_at,
			authority_ref=authority_ref,
			notes=notes,
		)
		self._exemptions[(tenant_id, exemption.id)] = exemption
		return exemption

	def is_exempt(
		self,
		tenant_id: str,
		country_code: str,
		entity_reference: str,
		as_of_date: date | None = None,
	) -> bool:
		"""Check if an entity/product has an active VAT exemption."""
		as_of = as_of_date or _today()
		for (tid, _), ex in self._exemptions.items():
			if tid != tenant_id:
				continue
			if ex.country_code != country_code.upper():
				continue
			if ex.entity_reference != entity_reference:
				continue
			if ex.granted_from > as_of:
				continue
			if ex.expires_at and ex.expires_at < as_of:
				continue
			return True
		return False

	def list_exemptions(self, tenant_id: str, country_code: str | None = None) -> list[TxVatExemption]:
		results = []
		for (tid, _), ex in self._exemptions.items():
			if tid != tenant_id:
				continue
			if country_code and ex.country_code != country_code.upper():
				continue
			results.append(ex)
		return results

	# ------------------------------------------------------------------
	# Country configuration
	# ------------------------------------------------------------------

	def get_country_config(self, country_code: str) -> TxVatCountryConfig | None:
		return self._country_configs.get(country_code.upper())

	def list_country_configs(self) -> list[TxVatCountryConfig]:
		return list(self._country_configs.values())

	def country_vat_summary(self, country_code: str) -> dict[str, Any]:
		"""Structured summary of a country's VAT regime."""
		config = self._country_configs.get(country_code.upper())
		if config is None:
			return {"country_code": country_code.upper(), "supported": False}

		today = _today()
		rates_by_category: dict[str, Any] = {}
		for (cc, cat), rate_list in self._rates.items():
			if cc != country_code.upper():
				continue
			active = [r for r in rate_list if r.effective_from <= today and (r.effective_to is None or r.effective_to >= today)]
			if active:
				rates_by_category[cat] = [{"rate_pct": str(r.rate_pct), "is_levy": r.is_levy, "levy_name": r.levy_name} for r in active]

		return {
			"country_code": country_code.upper(),
			"supported": True,
			"authority": config.authority_name,
			"authority_website": config.authority_website,
			"standard_rate_pct": str(config.standard_rate_pct),
			"registration_threshold": str(config.registration_threshold_local) if config.registration_threshold_local else None,
			"threshold_currency": config.threshold_currency,
			"filing_frequency": config.filing_frequency,
			"has_compound_levies": config.has_compound_levies,
			"compound_levy_names": config.compound_levy_names,
			"digital_services_tax_pct": str(config.digital_services_tax_pct) if config.digital_services_tax_pct else None,
			"rates_by_category": rates_by_category,
			"notes": config.notes,
		}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		returns = [r for (tid, _), r in self._returns.items() if tid == tenant_id]
		exemptions = [e for (tid, _), e in self._exemptions.items() if tid == tenant_id]
		by_status: dict[str, int] = {}
		for r in returns:
			by_status[r.status] = by_status.get(r.status, 0) + 1
		return {
			"tenant_id": tenant_id,
			"return_count": len(returns),
			"returns_by_status": by_status,
			"exemption_count": len(exemptions),
			"active_exemptions": sum(1 for e in exemptions if e.is_active),
			"supported_countries": len(self._country_configs),
			"as_of": _now().isoformat(),
		}

	# ------------------------------------------------------------------
	# Internals
	# ------------------------------------------------------------------

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", action.get("rule", "vat_policy_denied")) for action in result["actions"])
		raise PermissionError(reasons or "vat_policy_denied")

	def _seed_country_configs(self) -> None:
		for raw in _COUNTRY_CONFIGS:
			config = TxVatCountryConfig(
				country_code=raw["country_code"],
				authority_name=raw["authority_name"],
				authority_website=raw.get("authority_website", ""),
				registration_threshold_local=Decimal(raw["registration_threshold_local"]) if raw.get("registration_threshold_local") else None,
				threshold_currency=raw.get("threshold_currency", ""),
				filing_frequency=raw.get("filing_frequency", "monthly"),
				standard_rate_pct=Decimal(raw["standard_rate_pct"]),
				has_compound_levies=raw.get("has_compound_levies", False),
				compound_levy_names=raw.get("compound_levy_names", []),
				digital_services_tax_pct=Decimal(raw["digital_services_tax_pct"]) if raw.get("digital_services_tax_pct") else None,
				notes=raw.get("notes", ""),
			)
			self._country_configs[config.country_code] = config


# Public alias
CommonTaxVatService = VatService
