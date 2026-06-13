"""Withholding Tax (WHT) service.

Provides:
  - get_wht_rate()         — treaty-aware rate lookup
  - record_payment()       — record a WHT-attracting payment
  - issue_certificate()    — generate and store a WHT certificate
  - aggregate_return()     — build a quarterly WHT return
  - submit_return()        — file the return with the authority
  - WHT dashboard/summary
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

try:
	from .capability_contract import (
		SUPPORTED_COUNTRY_CODES,
		SUPPORTED_PAYMENT_TYPES,
		SUPPORTED_TREATY_STATUSES,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from .models import TxWhtCertificate, TxWhtPayment, TxWhtRate, TxWhtReturn
except ImportError:
	from capability_contract import (  # type: ignore
		SUPPORTED_COUNTRY_CODES,
		SUPPORTED_PAYMENT_TYPES,
		SUPPORTED_TREATY_STATUSES,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from models import TxWhtCertificate, TxWhtPayment, TxWhtRate, TxWhtReturn  # type: ignore


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
# Embedded WHT rate table
# Kenya rates per Income Tax Act Cap 470
# ---------------------------------------------------------------------------

_BUILTIN_WHT_RATES: list[dict[str, Any]] = [
	# Kenya — KRA
	{"country_code": "KE", "payment_type": "professional_fees",   "treaty_status": "domestic",        "entity_type": "company",    "rate_pct": "5.0",  "authority_name": "KRA", "effective_from": "2016-01-01", "notes": "ITA s35(1)(a)"},
	{"country_code": "KE", "payment_type": "professional_fees",   "treaty_status": "domestic",        "entity_type": "individual", "rate_pct": "5.0",  "authority_name": "KRA", "effective_from": "2016-01-01"},
	{"country_code": "KE", "payment_type": "management_fees",     "treaty_status": "domestic",        "entity_type": "company",    "rate_pct": "5.0",  "authority_name": "KRA", "effective_from": "2016-01-01"},
	{"country_code": "KE", "payment_type": "rent",                "treaty_status": "domestic",        "entity_type": "company",    "rate_pct": "3.0",  "authority_name": "KRA", "effective_from": "2016-01-01"},
	{"country_code": "KE", "payment_type": "rent",                "treaty_status": "domestic",        "entity_type": "individual", "rate_pct": "3.0",  "authority_name": "KRA", "effective_from": "2016-01-01", "notes": "Residential rent"},
	{"country_code": "KE", "payment_type": "dividends",           "treaty_status": "domestic",        "entity_type": "company",    "rate_pct": "15.0", "authority_name": "KRA", "effective_from": "2016-01-01", "notes": "Unlisted companies"},
	{"country_code": "KE", "payment_type": "dividends",           "treaty_status": "domestic",        "entity_type": "individual", "rate_pct": "5.0",  "authority_name": "KRA", "effective_from": "2016-01-01"},
	{"country_code": "KE", "payment_type": "interest",            "treaty_status": "domestic",        "entity_type": "company",    "rate_pct": "15.0", "authority_name": "KRA", "effective_from": "2016-01-01"},
	{"country_code": "KE", "payment_type": "royalties",           "treaty_status": "domestic",        "entity_type": "company",    "rate_pct": "5.0",  "authority_name": "KRA", "effective_from": "2016-01-01"},
	{"country_code": "KE", "payment_type": "commissions",         "treaty_status": "domestic",        "entity_type": "company",    "rate_pct": "5.0",  "authority_name": "KRA", "effective_from": "2016-01-01"},
	{"country_code": "KE", "payment_type": "winnings",            "treaty_status": "domestic",        "entity_type": "individual", "rate_pct": "20.0", "authority_name": "KRA", "effective_from": "2016-01-01"},
	# Non-resident rates
	{"country_code": "KE", "payment_type": "dividends",           "treaty_status": "non_resident",    "entity_type": "company",    "rate_pct": "10.0", "authority_name": "KRA", "effective_from": "2016-01-01", "notes": "Non-resident dividend WHT"},
	{"country_code": "KE", "payment_type": "royalties",           "treaty_status": "non_resident",    "entity_type": "company",    "rate_pct": "20.0", "authority_name": "KRA", "effective_from": "2016-01-01"},
	{"country_code": "KE", "payment_type": "professional_fees",   "treaty_status": "non_resident",    "entity_type": "company",    "rate_pct": "20.0", "authority_name": "KRA", "effective_from": "2016-01-01"},

	# Nigeria — FIRS
	{"country_code": "NG", "payment_type": "dividends",           "treaty_status": "domestic",        "entity_type": "company",    "rate_pct": "10.0", "authority_name": "FIRS", "effective_from": "2020-02-01"},
	{"country_code": "NG", "payment_type": "interest",            "treaty_status": "domestic",        "entity_type": "company",    "rate_pct": "10.0", "authority_name": "FIRS", "effective_from": "2020-02-01"},
	{"country_code": "NG", "payment_type": "royalties",           "treaty_status": "domestic",        "entity_type": "company",    "rate_pct": "10.0", "authority_name": "FIRS", "effective_from": "2020-02-01"},
	{"country_code": "NG", "payment_type": "rent",                "treaty_status": "domestic",        "entity_type": "company",    "rate_pct": "10.0", "authority_name": "FIRS", "effective_from": "2020-02-01"},
	{"country_code": "NG", "payment_type": "professional_fees",   "treaty_status": "domestic",        "entity_type": "company",    "rate_pct": "5.0",  "authority_name": "FIRS", "effective_from": "2020-02-01", "notes": "Consultancy/management fees"},
	{"country_code": "NG", "payment_type": "construction",        "treaty_status": "domestic",        "entity_type": "company",    "rate_pct": "5.0",  "authority_name": "FIRS", "effective_from": "2020-02-01"},

	# Ghana — GRA
	{"country_code": "GH", "payment_type": "dividends",           "treaty_status": "domestic",        "entity_type": "company",    "rate_pct": "8.0",  "authority_name": "GRA", "effective_from": "2021-01-01"},
	{"country_code": "GH", "payment_type": "interest",            "treaty_status": "domestic",        "entity_type": "company",    "rate_pct": "8.0",  "authority_name": "GRA", "effective_from": "2021-01-01"},
	{"country_code": "GH", "payment_type": "royalties",           "treaty_status": "domestic",        "entity_type": "company",    "rate_pct": "15.0", "authority_name": "GRA", "effective_from": "2021-01-01"},
	{"country_code": "GH", "payment_type": "rent",                "treaty_status": "domestic",        "entity_type": "individual", "rate_pct": "8.0",  "authority_name": "GRA", "effective_from": "2021-01-01"},

	# Uganda — URA
	{"country_code": "UG", "payment_type": "dividends",           "treaty_status": "domestic",        "entity_type": "company",    "rate_pct": "15.0", "authority_name": "URA", "effective_from": "2021-01-01"},
	{"country_code": "UG", "payment_type": "interest",            "treaty_status": "domestic",        "entity_type": "company",    "rate_pct": "15.0", "authority_name": "URA", "effective_from": "2021-01-01"},
	{"country_code": "UG", "payment_type": "royalties",           "treaty_status": "domestic",        "entity_type": "company",    "rate_pct": "15.0", "authority_name": "URA", "effective_from": "2021-01-01"},
	{"country_code": "UG", "payment_type": "professional_fees",   "treaty_status": "domestic",        "entity_type": "company",    "rate_pct": "15.0", "authority_name": "URA", "effective_from": "2021-01-01"},

	# South Africa — SARS
	{"country_code": "ZA", "payment_type": "dividends",           "treaty_status": "domestic",        "entity_type": "company",    "rate_pct": "20.0", "authority_name": "SARS", "effective_from": "2017-03-01", "notes": "Dividends tax s64B"},
	{"country_code": "ZA", "payment_type": "dividends",           "treaty_status": "non_resident",    "entity_type": "company",    "rate_pct": "20.0", "authority_name": "SARS", "effective_from": "2017-03-01"},
	{"country_code": "ZA", "payment_type": "royalties",           "treaty_status": "non_resident",    "entity_type": "company",    "rate_pct": "15.0", "authority_name": "SARS", "effective_from": "2017-03-01"},
	{"country_code": "ZA", "payment_type": "interest",            "treaty_status": "non_resident",    "entity_type": "company",    "rate_pct": "15.0", "authority_name": "SARS", "effective_from": "2017-03-01"},
]

# Default currency per country
_COUNTRY_CURRENCY: dict[str, str] = {
	"KE": "KES", "NG": "NGN", "GH": "GHS", "UG": "UGX", "TZ": "TZS",
	"ZA": "ZAR", "RW": "RWF", "ET": "ETB", "EG": "EGP", "MA": "MAD",
}


class WhtService:
	"""Tenant-scoped Withholding Tax engine."""

	def __init__(self) -> None:
		# WHT rates: (country_code, payment_type, treaty_status, entity_type) -> list[TxWhtRate]
		self._rates: dict[tuple[str, str, str, str], list[TxWhtRate]] = {}
		# Tenant-scoped payments
		self._payments: dict[tuple[str, str], TxWhtPayment] = {}
		# Tenant-scoped certificates
		self._certificates: dict[tuple[str, str], TxWhtCertificate] = {}
		# Tenant-scoped returns
		self._returns: dict[tuple[str, str], TxWhtReturn] = {}
		# Certificate sequence number per tenant
		self._cert_seq: dict[str, int] = {}

		self._seed_builtin_rates()

	# ------------------------------------------------------------------
	# Describe / evaluate
	# ------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------
	# Rate lookup
	# ------------------------------------------------------------------

	def get_wht_rate(
		self,
		country_code: str,
		payment_type: str,
		treaty_status: str = "domestic",
		entity_type: str = "company",
		as_of_date: date | None = None,
		treaty_evidence_present: bool = True,
	) -> TxWhtRate | None:
		"""Return the applicable WHT rate for a country × payment_type × treaty_status.

		Returns None if no rate is configured (payment may be exempt).
		Raises PermissionError if treaty_reduced requested without evidence.
		"""
		country_code = country_code.upper().strip()
		payment_type = payment_type.lower().strip()
		treaty_status = treaty_status.lower().strip()
		entity_type = entity_type.lower().strip()
		as_of = as_of_date or _today()

		self._enforce({
			"tenant_context_present": True,
			"operation": "get_wht_rate",
			"country_code_supported": country_code in SUPPORTED_COUNTRY_CODES,
			"payment_type_supported": payment_type in SUPPORTED_PAYMENT_TYPES,
			"treaty_status": treaty_status,
			"treaty_evidence_present": treaty_evidence_present,
		})

		key = (country_code, payment_type, treaty_status, entity_type)
		rates = self._rates.get(key, [])
		active = [
			r for r in rates
			if r.effective_from <= as_of and (r.effective_to is None or r.effective_to >= as_of)
		]
		if not active:
			# Try fallback to company rate
			fallback_key = (country_code, payment_type, treaty_status, "company")
			if fallback_key != key:
				fallback = self._rates.get(fallback_key, [])
				active = [r for r in fallback if r.effective_from <= as_of and (r.effective_to is None or r.effective_to >= as_of)]
		return active[0] if active else None

	def register_wht_rate(
		self,
		tenant_id: str,
		country_code: str,
		payment_type: str,
		rate_pct: Decimal,
		authority_name: str,
		effective_from: date,
		treaty_status: str = "domestic",
		entity_type: str = "company",
		treaty_country_code: str | None = None,
		effective_to: date | None = None,
		authority_ref: str = "",
		notes: str = "",
	) -> TxWhtRate:
		"""Register a custom WHT rate (e.g. for a new double-tax treaty)."""
		assert _present(tenant_id), "tenant_id is required"
		rate = TxWhtRate(
			country_code=country_code.upper(),
			payment_type=payment_type.lower(),
			treaty_status=treaty_status.lower(),
			entity_type=entity_type.lower(),
			treaty_country_code=treaty_country_code,
			rate_pct=rate_pct,
			authority_name=authority_name,
			authority_ref=authority_ref,
			effective_from=effective_from,
			effective_to=effective_to,
			notes=notes,
		)
		key = (rate.country_code, rate.payment_type, rate.treaty_status, rate.entity_type)
		if key not in self._rates:
			self._rates[key] = []
		self._rates[key].append(rate)
		return rate

	# ------------------------------------------------------------------
	# Payment recording
	# ------------------------------------------------------------------

	def record_payment(
		self,
		tenant_id: str,
		country_code: str,
		payer_id: str,
		payee_id: str,
		payment_type: str,
		gross_amount: Decimal,
		payment_date: date,
		currency: str | None = None,
		treaty_status: str = "domestic",
		entity_type: str = "company",
		treaty_country_code: str | None = None,
		source_document_id: str = "",
		source_document_type: str = "",
		notes: str = "",
	) -> TxWhtPayment:
		"""Record a payment attracting WHT and compute the WHT amount."""
		assert _present(payer_id), "payer_id is required"
		assert _present(payee_id), "payee_id is required"
		assert gross_amount > 0, "gross_amount must be positive"

		resolved_currency = (currency or _COUNTRY_CURRENCY.get(country_code.upper(), "USD")).upper()

		rate_obj = self.get_wht_rate(
			country_code=country_code,
			payment_type=payment_type,
			treaty_status=treaty_status,
			entity_type=entity_type,
			as_of_date=payment_date,
		)

		if rate_obj:
			wht_rate_pct = rate_obj.rate_pct
		else:
			wht_rate_pct = Decimal("0")

		wht_amount = (gross_amount * wht_rate_pct / 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
		net_amount = gross_amount - wht_amount

		payment = TxWhtPayment(
			tenant_id=tenant_id,
			country_code=country_code.upper(),
			payer_id=payer_id,
			payee_id=payee_id,
			payment_type=payment_type.lower(),
			treaty_status=treaty_status.lower(),
			treaty_country_code=treaty_country_code,
			gross_amount=gross_amount,
			wht_rate_pct=wht_rate_pct,
			wht_amount=wht_amount,
			net_amount=net_amount,
			currency=resolved_currency,
			payment_date=payment_date,
			source_document_id=source_document_id,
			source_document_type=source_document_type,
			notes=notes,
		)
		self._payments[(tenant_id, payment.id)] = payment
		return payment

	# ------------------------------------------------------------------
	# Certificate management
	# ------------------------------------------------------------------

	def issue_certificate(
		self,
		tenant_id: str,
		payment_id: str,
		payer_name: str,
		payee_name: str,
		payee_tax_pin: str = "",
		issued_by: str = "system",
	) -> TxWhtCertificate:
		"""Issue a WHT certificate for a recorded payment."""
		assert _present(payer_name), "payer_name is required"
		assert _present(payee_name), "payee_name is required"

		payment = self._payments.get((tenant_id, payment_id))
		if payment is None:
			raise KeyError(f"payment {payment_id!r} not found for tenant {tenant_id!r}")

		self._enforce({
			"tenant_context_present": bool(tenant_id),
			"operation": "issue_certificate",
			"payment_proof_present": True,  # existence of payment record is proof
		})

		seq = self._cert_seq.get(tenant_id, 0) + 1
		self._cert_seq[tenant_id] = seq
		cert_number = f"WHT-{tenant_id[:8].upper()}-{payment.country_code}-{seq:06d}"

		cert = TxWhtCertificate(
			tenant_id=tenant_id,
			certificate_number=cert_number,
			country_code=payment.country_code,
			payer_id=payment.payer_id,
			payer_name=payer_name,
			payee_id=payment.payee_id,
			payee_name=payee_name,
			payee_tax_pin=payee_tax_pin,
			payment_type=payment.payment_type,
			gross_payment=payment.gross_amount,
			wht_rate_pct=payment.wht_rate_pct,
			wht_amount=payment.wht_amount,
			currency=payment.currency,
			payment_date=payment.payment_date,
			payment_reference=payment_id,
			status="issued",
			issued_by=issued_by,
		)
		self._certificates[(tenant_id, cert.id)] = cert

		# Link certificate back to payment
		payment.certificate_id = cert.id
		return cert

	def get_certificate(self, tenant_id: str, certificate_id: str) -> TxWhtCertificate:
		cert = self._certificates.get((tenant_id, certificate_id))
		if cert is None:
			raise KeyError(f"certificate {certificate_id!r} not found for tenant {tenant_id!r}")
		return cert

	def list_certificates(
		self,
		tenant_id: str,
		country_code: str | None = None,
		payee_id: str | None = None,
		status: str | None = None,
	) -> list[TxWhtCertificate]:
		results = []
		for (tid, _), cert in self._certificates.items():
			if tid != tenant_id:
				continue
			if country_code and cert.country_code != country_code.upper():
				continue
			if payee_id and cert.payee_id != payee_id:
				continue
			if status and cert.status != status.lower():
				continue
			results.append(cert)
		return results

	# ------------------------------------------------------------------
	# WHT returns
	# ------------------------------------------------------------------

	def aggregate_return(
		self,
		tenant_id: str,
		country_code: str,
		period_name: str,
		period_start: date,
		period_end: date,
		filing_due_date: date,
		currency: str | None = None,
	) -> TxWhtReturn:
		"""Build a WHT return by aggregating all payments in the period."""
		country_code = country_code.upper()
		resolved_currency = (currency or _COUNTRY_CURRENCY.get(country_code, "USD")).upper()

		# Collect payments in period for this country
		period_payments = [
			p for (tid, _), p in self._payments.items()
			if tid == tenant_id
			and p.country_code == country_code
			and period_start <= p.payment_date <= period_end
		]

		total_gross = sum(p.gross_amount for p in period_payments)
		total_wht = sum(p.wht_amount for p in period_payments)
		payment_ids = [p.id for p in period_payments]

		# Collect certificates linked to these payments
		cert_ids = [p.certificate_id for p in period_payments if p.certificate_id]

		wht_return = TxWhtReturn(
			tenant_id=tenant_id,
			country_code=country_code,
			period_name=period_name,
			period_start=period_start,
			period_end=period_end,
			filing_due_date=filing_due_date,
			total_gross_payments=total_gross,
			total_wht_amount=total_wht,
			currency=resolved_currency,
			payment_ids=payment_ids,
			certificate_ids=cert_ids,
			status="draft",
		)
		self._returns[(tenant_id, wht_return.id)] = wht_return

		# Link payments to this return
		for p in period_payments:
			p.return_id = wht_return.id

		# Link certificates to this return
		for cid in cert_ids:
			cert = self._certificates.get((tenant_id, cid))
			if cert:
				cert.wht_return_id = wht_return.id

		return wht_return

	def submit_return(
		self,
		tenant_id: str,
		return_id: str,
		submitted_by: str,
	) -> TxWhtReturn:
		"""Submit a WHT return to the tax authority."""
		assert _present(submitted_by), "submitted_by is required"
		wht_return = self._returns.get((tenant_id, return_id))
		if wht_return is None:
			raise KeyError(f"WHT return {return_id!r} not found for tenant {tenant_id!r}")

		self._enforce({
			"tenant_context_present": bool(tenant_id),
			"operation": "submit_return",
			"period_in_future": wht_return.period_end > _today(),
		})

		wht_return.status = "submitted"
		wht_return.submitted_at = _now()
		wht_return.submitted_by = submitted_by
		return wht_return

	def list_returns(
		self,
		tenant_id: str,
		country_code: str | None = None,
		status: str | None = None,
	) -> list[TxWhtReturn]:
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
	# Reporting
	# ------------------------------------------------------------------

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		payments = [p for (tid, _), p in self._payments.items() if tid == tenant_id]
		certs = [c for (tid, _), c in self._certificates.items() if tid == tenant_id]
		returns = [r for (tid, _), r in self._returns.items() if tid == tenant_id]

		total_gross = sum(p.gross_amount for p in payments)
		total_wht = sum(p.wht_amount for p in payments)

		by_payment_type: dict[str, dict[str, Any]] = {}
		for p in payments:
			pt = p.payment_type
			if pt not in by_payment_type:
				by_payment_type[pt] = {"count": 0, "gross": Decimal("0"), "wht": Decimal("0")}
			by_payment_type[pt]["count"] += 1
			by_payment_type[pt]["gross"] += p.gross_amount
			by_payment_type[pt]["wht"] += p.wht_amount

		return {
			"tenant_id": tenant_id,
			"payment_count": len(payments),
			"certificate_count": len(certs),
			"return_count": len(returns),
			"draft_returns": sum(1 for r in returns if r.status == "draft"),
			"submitted_returns": sum(1 for r in returns if r.status == "submitted"),
			"total_gross_payments": str(total_gross),
			"total_wht_withheld": str(total_wht),
			"by_payment_type": {
				pt: {"count": v["count"], "gross": str(v["gross"]), "wht": str(v["wht"])}
				for pt, v in by_payment_type.items()
			},
			"as_of": _now().isoformat(),
		}

	def payee_wht_summary(self, tenant_id: str, payee_id: str) -> dict[str, Any]:
		"""All WHT withheld from a specific payee — useful for reconciliation."""
		payments = [
			p for (tid, _), p in self._payments.items()
			if tid == tenant_id and p.payee_id == payee_id
		]
		certs = [
			c for (tid, _), c in self._certificates.items()
			if tid == tenant_id and c.payee_id == payee_id
		]
		total_gross = sum(p.gross_amount for p in payments)
		total_wht = sum(p.wht_amount for p in payments)
		return {
			"tenant_id": tenant_id,
			"payee_id": payee_id,
			"payment_count": len(payments),
			"certificate_count": len(certs),
			"total_gross": str(total_gross),
			"total_wht": str(total_wht),
			"certificates": [c.certificate_number for c in certs],
			"as_of": _now().isoformat(),
		}

	# ------------------------------------------------------------------
	# Internals
	# ------------------------------------------------------------------

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", action.get("rule", "wht_policy_denied")) for action in result["actions"])
		raise PermissionError(reasons or "wht_policy_denied")

	def _seed_builtin_rates(self) -> None:
		for raw in _BUILTIN_WHT_RATES:
			rate = TxWhtRate(
				country_code=raw["country_code"],
				payment_type=raw["payment_type"],
				treaty_status=raw.get("treaty_status", "domestic"),
				entity_type=raw.get("entity_type", "company"),
				rate_pct=Decimal(raw["rate_pct"]),
				authority_name=raw["authority_name"],
				authority_ref=raw.get("authority_ref", ""),
				effective_from=date.fromisoformat(raw["effective_from"]),
				effective_to=date.fromisoformat(raw["effective_to"]) if raw.get("effective_to") else None,
				notes=raw.get("notes", ""),
			)
			key = (rate.country_code, rate.payment_type, rate.treaty_status, rate.entity_type)
			if key not in self._rates:
				self._rates[key] = []
			self._rates[key].append(rate)


# Public alias
CommonTaxWhtService = WhtService
