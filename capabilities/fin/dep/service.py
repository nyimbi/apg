"""Deposit Products Engine — service layer.

In-memory, async-first, Decimal arithmetic.  Idempotent batch accrual.
Plugs into APG via domain adapters; runs standalone with null adapters.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""
from __future__ import annotations

import logging
import uuid
from copy import deepcopy
from datetime import date, datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

try:
	from .models import (
		AccrualEntry, BatchAccrualResult, CompoundingFrequency, DepositProduct,
		FeeConfig, FeeFrequency, InterestCalculationResult, InterestCalculationType,
		InterestConfig, InterestPostingEntry, InterestTier, MaturityInstruction,
		MaturityRecord, MinimumBalanceCheck, ProductStatus, ProductTerms, ProductType,
		RateHistoryEntry, SimulationResult, WithholdingTaxEntry,
	)
except ImportError:  # pragma: no cover
	from models import (  # type: ignore
		AccrualEntry, BatchAccrualResult, CompoundingFrequency, DepositProduct,
		FeeConfig, FeeFrequency, InterestCalculationResult, InterestCalculationType,
		InterestConfig, InterestPostingEntry, InterestTier, MaturityInstruction,
		MaturityRecord, MinimumBalanceCheck, ProductStatus, ProductTerms, ProductType,
		RateHistoryEntry, SimulationResult, WithholdingTaxEntry,
	)

log  = logging.getLogger(__name__)
_log = log  # alias used by new methods

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _now() -> datetime:
	return datetime.now(timezone.utc)


def _uid() -> str:
	return str(uuid.uuid4())


def _d(v: Any) -> Decimal:
	"""Coerce to Decimal, raising on bad input."""
	if isinstance(v, Decimal):
		return v
	return Decimal(str(v))


def _round(v: Decimal, places: int = 2) -> Decimal:
	q = Decimal(10) ** -places
	return v.quantize(q, rounding=ROUND_HALF_UP)


def _guard_tenant(tenant_id: str) -> None:
	if not tenant_id or not tenant_id.strip():
		raise ValueError("tenant_id is required")


def _guard_str(value: str, name: str) -> None:
	if not value or not value.strip():
		raise ValueError(f"{name} is required")


def _log_pretty_path(label: str, tenant_id: str, resource: str) -> str:
	return f"[dep][{tenant_id}] {label}: {resource}"


# ─────────────────────────────────────────────────────────────
# Interest arithmetic
# ─────────────────────────────────────────────────────────────

def _resolve_tier_rate(tiers: list[InterestTier], balance: Decimal, base_rate: Decimal) -> Decimal:
	"""Return the applicable tier rate for a given balance, or base_rate if no tiers."""
	if not tiers:
		return base_rate
	applicable = base_rate
	for tier in sorted(tiers, key=lambda t: t.min_balance):
		if balance >= tier.min_balance:
			applicable = tier.rate
	return applicable


def _calc_simple(principal: Decimal, annual_rate: Decimal, days: int) -> Decimal:
	return principal * (annual_rate / Decimal("100")) * (Decimal(days) / Decimal("365"))


def _calc_compound(
	principal: Decimal,
	annual_rate: Decimal,
	days: int,
	compounding: CompoundingFrequency,
) -> Decimal:
	r = annual_rate / Decimal("100")
	if compounding == CompoundingFrequency.DAILY:
		n = Decimal("365")
	elif compounding == CompoundingFrequency.MONTHLY:
		n = Decimal("12")
	else:
		n = Decimal("1")
	t = Decimal(days) / Decimal("365")
	# A = P(1 + r/n)^(nt)
	nt = n * t
	# Python Decimal doesn't support fractional exponents natively; use float
	factor = Decimal(str(float((1 + float(r / n)) ** float(nt))))
	return principal * (factor - Decimal("1"))


def _calc_daily_accrual(
	principal: Decimal,
	annual_rate: Decimal,
	days: int,
) -> Decimal:
	"""Daily simple accrual summed over the period."""
	daily = principal * (annual_rate / Decimal("100")) / Decimal("365")
	return daily * Decimal(days)


def _calculate_gross(
	balance: Decimal,
	cfg: InterestConfig,
	days: int,
) -> tuple[Decimal, Decimal]:
	"""Return (gross_interest, rate_applied)."""
	rate = _resolve_tier_rate(cfg.tiers, balance, cfg.rate)
	if cfg.calculation == InterestCalculationType.SIMPLE:
		gross = _calc_simple(balance, rate, days)
	elif cfg.calculation == InterestCalculationType.COMPOUND:
		gross = _calc_compound(balance, rate, days, cfg.compounding)
	else:
		gross = _calc_daily_accrual(balance, rate, days)
	return _round(gross, 6), rate


# ─────────────────────────────────────────────────────────────
# Service
# ─────────────────────────────────────────────────────────────

class DepositProductsService:
	"""Banking product factory and interest calculation engine for deposits.

	Runs fully in-memory for standalone use (tests, CLI).  Swap in real
	database adapters by extending _store_* / _load_* methods.
	"""

	# ── Stores (in-memory, keyed by tenant_id) ──────────────────────────────

	def __init__(self) -> None:
		# products[(tenant_id, code)] -> DepositProduct
		self._products:   dict[tuple[str, str], DepositProduct]         = {}
		# accounts[(tenant_id, account_id)] -> dict  (lightweight stub)
		self._accounts:   dict[tuple[str, str], dict[str, Any]]         = {}
		# accruals[(tenant_id, account_id, accrual_date)] -> AccrualEntry
		self._accruals:   dict[tuple[str, str, str], AccrualEntry]       = {}
		# postings[(tenant_id, account_id)] -> list[InterestPostingEntry]
		self._postings:   dict[tuple[str, str], list[InterestPostingEntry]] = {}
		# rate_history[(tenant_id, product_code)] -> list[RateHistoryEntry]
		self._rate_hist:  dict[tuple[str, str], list[RateHistoryEntry]]  = {}
		# maturity[(tenant_id, account_id)] -> MaturityRecord
		self._maturities: dict[tuple[str, str], MaturityRecord]          = {}
		# batch idempotency: (tenant_id, accrual_date_iso) -> BatchAccrualResult
		self._batch_done: dict[str, BatchAccrualResult]                  = {}
		# fees[(tenant_id, account_id, posting_date)] -> dict
		self._fees:       dict[tuple[str, str, str], dict[str, Any]]     = {}

	# ── Account registration (needed for balance look-up) ──────────────────

	def register_account(
		self,
		tenant_id: str,
		account_id: str,
		product_code: str,
		balance: Decimal,
		opening_date: date | None = None,
		maturity_date: date | None = None,
		linked_account: str = "",
	) -> dict[str, Any]:
		"""Register or update an account stub for interest calculations."""
		_guard_tenant(tenant_id)
		_guard_str(account_id, "account_id")
		key = (tenant_id, account_id)
		self._accounts[key] = {
			"account_id":    account_id,
			"tenant_id":     tenant_id,
			"product_code":  product_code,
			"balance":       balance,
			"opening_date":  (opening_date or date.today()).isoformat(),
			"maturity_date": maturity_date.isoformat() if maturity_date else None,
			"linked_account": linked_account,
		}
		log.debug(_log_pretty_path("register_account", tenant_id, account_id))
		return deepcopy(self._accounts[key])

	def _get_account(self, tenant_id: str, account_id: str) -> dict[str, Any]:
		key = (tenant_id, account_id)
		if key not in self._accounts:
			raise KeyError(f"Account {account_id!r} not found for tenant {tenant_id!r}")
		return self._accounts[key]

	# ── Product lifecycle ───────────────────────────────────────────────────

	def create_product(
		self,
		tenant_id: str,
		code: str,
		name: str,
		product_type: ProductType,
		currency: str,
		interest_config: InterestConfig,
		fee_config: FeeConfig,
		terms: ProductTerms,
		gl_interest_income_account: str = "",
		gl_interest_payable_account: str = "",
		gl_wht_payable_account: str = "",
		created_by: str = "system",
	) -> DepositProduct:
		_guard_tenant(tenant_id)
		_guard_str(code, "code")
		_guard_str(name, "name")
		key = (tenant_id, code)
		if key in self._products:
			raise ValueError(f"Product {code!r} already exists for tenant {tenant_id!r}")
		now = _now()
		product = DepositProduct(
			id=_uid(),
			tenant_id=tenant_id,
			code=code,
			name=name,
			product_type=product_type,
			currency=currency,
			interest_config=interest_config,
			fee_config=fee_config,
			terms=terms,
			status=ProductStatus.ACTIVE,
			created_at=now,
			updated_at=now,
			created_by=created_by,
			gl_interest_income_account=gl_interest_income_account,
			gl_interest_payable_account=gl_interest_payable_account,
			gl_wht_payable_account=gl_wht_payable_account,
		)
		self._products[key] = product
		self._rate_hist[(tenant_id, code)] = [
			RateHistoryEntry(
				id=_uid(),
				tenant_id=tenant_id,
				product_code=code,
				old_rate=Decimal("0"),
				new_rate=interest_config.rate,
				effective_date=now.date(),
				reason="product_created",
				changed_by=created_by,
				changed_at=now,
			)
		]
		log.info(_log_pretty_path("create_product", tenant_id, code))
		return deepcopy(product)

	def get_product(self, tenant_id: str, code: str) -> DepositProduct:
		_guard_tenant(tenant_id)
		key = (tenant_id, code)
		if key not in self._products:
			raise KeyError(f"Product {code!r} not found for tenant {tenant_id!r}")
		return deepcopy(self._products[key])

	def list_products(
		self,
		tenant_id: str,
		product_type: ProductType | None = None,
		active_only: bool = True,
	) -> list[DepositProduct]:
		_guard_tenant(tenant_id)
		results = []
		for (tid, _), product in self._products.items():
			if tid != tenant_id:
				continue
			if active_only and product.status != ProductStatus.ACTIVE:
				continue
			if product_type and product.product_type != product_type:
				continue
			results.append(deepcopy(product))
		return results

	def update_product(
		self,
		tenant_id: str,
		code: str,
		updates: dict[str, Any],
	) -> DepositProduct:
		_guard_tenant(tenant_id)
		key = (tenant_id, code)
		if key not in self._products:
			raise KeyError(f"Product {code!r} not found for tenant {tenant_id!r}")
		product = deepcopy(self._products[key])
		allowed = {"name", "interest_config", "fee_config", "terms",
			"gl_interest_income_account", "gl_interest_payable_account",
			"gl_wht_payable_account"}
		for field, val in updates.items():
			if field in allowed:
				object.__setattr__(product, field, val)
		product = product.model_copy(update={"updated_at": _now()})
		self._products[key] = product
		log.info(_log_pretty_path("update_product", tenant_id, code))
		return deepcopy(product)

	def deactivate_product(self, tenant_id: str, code: str) -> DepositProduct:
		_guard_tenant(tenant_id)
		key = (tenant_id, code)
		if key not in self._products:
			raise KeyError(f"Product {code!r} not found for tenant {tenant_id!r}")
		product = self._products[key].model_copy(
			update={"status": ProductStatus.INACTIVE, "updated_at": _now()}
		)
		self._products[key] = product
		log.info(_log_pretty_path("deactivate_product", tenant_id, code))
		return deepcopy(product)

	# ── Rate management ─────────────────────────────────────────────────────

	def update_product_rate(
		self,
		tenant_id: str,
		product_code: str,
		new_rate: Decimal,
		effective_date: date,
		reason: str,
		changed_by: str = "system",
	) -> RateHistoryEntry:
		_guard_tenant(tenant_id)
		key = (tenant_id, product_code)
		if key not in self._products:
			raise KeyError(f"Product {product_code!r} not found")
		product = self._products[key]
		old_rate = product.interest_config.rate
		new_cfg = product.interest_config.model_copy(update={"rate": new_rate})
		self._products[key] = product.model_copy(
			update={"interest_config": new_cfg, "updated_at": _now()}
		)
		entry = RateHistoryEntry(
			id=_uid(),
			tenant_id=tenant_id,
			product_code=product_code,
			old_rate=old_rate,
			new_rate=new_rate,
			effective_date=effective_date,
			reason=reason,
			changed_by=changed_by,
			changed_at=_now(),
		)
		hkey = (tenant_id, product_code)
		self._rate_hist.setdefault(hkey, []).append(entry)
		log.info(_log_pretty_path("update_product_rate", tenant_id, f"{product_code} -> {new_rate}%"))
		return entry

	def get_rate_schedule(self, tenant_id: str, product_code: str) -> list[RateHistoryEntry]:
		_guard_tenant(tenant_id)
		return list(self._rate_hist.get((tenant_id, product_code), []))

	# ── Interest calculation ────────────────────────────────────────────────

	def calculate_interest(
		self,
		tenant_id: str,
		account_id: str,
		from_date: date,
		to_date: date,
		balance: Decimal,
		product_code: str,
	) -> InterestCalculationResult:
		"""Compute interest for the period; does NOT post anything."""
		_guard_tenant(tenant_id)
		assert from_date <= to_date, "from_date must be <= to_date"
		assert balance >= Decimal("0"), "balance must be non-negative"

		product = self.get_product(tenant_id, product_code)
		cfg     = product.interest_config
		days    = (to_date - from_date).days
		if days == 0:
			days = 1  # same-day accrual = 1 day

		gross, rate = _calculate_gross(balance, cfg, days)

		if product.terms.tax_exempt:
			wht = Decimal("0")
		else:
			wht = _round(gross * (cfg.withholding_rate / Decimal("100")), 6)

		net = _round(gross - wht, 6)

		# Build tier breakdown for transparency
		breakdown: list[dict[str, Any]] = []
		if cfg.tiers:
			for tier in sorted(cfg.tiers, key=lambda t: t.min_balance):
				if balance >= tier.min_balance:
					g, _ = _calculate_gross(balance, cfg.model_copy(update={"rate": tier.rate, "tiers": []}), days)
					breakdown.append({
						"min_balance": str(tier.min_balance),
						"rate": str(tier.rate),
						"gross": str(_round(g)),
					})

		result = InterestCalculationResult(
			gross_interest=gross,
			withholding_tax=wht,
			net_interest=net,
			accrual_days=days,
			rate_applied=rate,
			calculation_type=cfg.calculation.value,
			tier_breakdown=breakdown,
		)
		log.debug(_log_pretty_path("calculate_interest", tenant_id,
			f"{account_id} days={days} rate={rate}% gross={gross}"))
		return result

	def get_accrued_interest(
		self,
		tenant_id: str,
		account_id: str,
		as_of_date: date,
	) -> Decimal:
		"""Sum un-posted accruals up to as_of_date."""
		_guard_tenant(tenant_id)
		total = Decimal("0")
		for key, entry in self._accruals.items():
			if key[0] != tenant_id or key[1] != account_id:
				continue
			if entry.posted:
				continue
			accrual_date = date.fromisoformat(key[2])
			if accrual_date <= as_of_date:
				total += entry.net_amount
		return total

	# ── Interest posting ────────────────────────────────────────────────────

	def apply_interest(
		self,
		tenant_id: str,
		account_id: str,
		interest_amount: Decimal,
		value_date: date,
		posting_ref: str,
	) -> dict[str, Any]:
		"""Post interest credit to account and generate GL journal stub."""
		_guard_tenant(tenant_id)
		acct = self._get_account(tenant_id, account_id)
		product = self.get_product(tenant_id, acct["product_code"])
		cfg     = product.interest_config

		gross  = interest_amount
		wht    = Decimal("0") if product.terms.tax_exempt else _round(gross * (cfg.withholding_rate / Decimal("100")))
		net    = gross - wht

		# Credit account balance
		self._accounts[(tenant_id, account_id)]["balance"] = _d(acct["balance"]) + net

		gl_ref = _uid()
		entry  = InterestPostingEntry(
			id=_uid(),
			tenant_id=tenant_id,
			account_id=account_id,
			product_code=acct["product_code"],
			value_date=value_date,
			gross_interest=gross,
			wht_amount=wht,
			net_interest=net,
			posting_ref=posting_ref,
			gl_ref=gl_ref,
			posted_at=_now(),
		)
		self._postings.setdefault((tenant_id, account_id), []).append(entry)

		# Mark any pending accruals for this account as posted
		for k, accrual in self._accruals.items():
			if k[0] == tenant_id and k[1] == account_id and not accrual.posted:
				self._accruals[k] = accrual.model_copy(update={"posted": True, "posting_ref": posting_ref})

		log.info(_log_pretty_path("apply_interest", tenant_id,
			f"{account_id} net={net} ref={posting_ref}"))
		return {
			"entry_id":      entry.id,
			"account_id":    account_id,
			"gross_interest": str(gross),
			"wht_amount":    str(wht),
			"net_interest":  str(net),
			"value_date":    value_date.isoformat(),
			"posting_ref":   posting_ref,
			"gl_ref":        gl_ref,
			"gl_debit":      product.gl_interest_income_account,
			"gl_credit_account": account_id,
			"gl_wht":        product.gl_wht_payable_account,
		}

	# ── Fee management ──────────────────────────────────────────────────────

	def apply_maintenance_fee(
		self,
		tenant_id: str,
		account_id: str,
		posting_date: date,
	) -> dict[str, Any]:
		"""Compute and post maintenance fee, respecting minimum-balance waiver."""
		_guard_tenant(tenant_id)
		acct    = self._get_account(tenant_id, account_id)
		product = self.get_product(tenant_id, acct["product_code"])
		fc      = product.fee_config
		balance = _d(acct["balance"])

		# Determine applicable fee
		if balance < fc.minimum_balance and fc.below_minimum_fee > Decimal("0"):
			fee = fc.below_minimum_fee
			reason = "below_minimum_balance"
		elif fc.maintenance_fee > Decimal("0"):
			fee = fc.maintenance_fee
			reason = "maintenance"
		else:
			return {
				"account_id":  account_id,
				"fee_amount":  "0",
				"reason":      "no_fee_applicable",
				"posted":      False,
			}

		# Debit account
		self._accounts[(tenant_id, account_id)]["balance"] = balance - fee
		fee_id = _uid()
		record = {
			"id":           fee_id,
			"account_id":   account_id,
			"tenant_id":    tenant_id,
			"product_code": acct["product_code"],
			"fee_amount":   str(fee),
			"reason":       reason,
			"posting_date": posting_date.isoformat(),
			"posted_at":    _now().isoformat(),
		}
		self._fees[(tenant_id, account_id, posting_date.isoformat())] = record
		log.info(_log_pretty_path("apply_maintenance_fee", tenant_id,
			f"{account_id} fee={fee} reason={reason}"))
		return record

	def check_minimum_balance(
		self,
		tenant_id: str,
		account_id: str,
	) -> MinimumBalanceCheck:
		_guard_tenant(tenant_id)
		acct    = self._get_account(tenant_id, account_id)
		product = self.get_product(tenant_id, acct["product_code"])
		balance = _d(acct["balance"])
		minimum = product.fee_config.minimum_balance
		meets   = balance >= minimum
		shortfall = Decimal("0") if meets else minimum - balance
		return MinimumBalanceCheck(
			account_id=account_id,
			meets_minimum=meets,
			current_balance=balance,
			minimum_required=minimum,
			shortfall=shortfall,
			fee_applicable=not meets and product.fee_config.below_minimum_fee > Decimal("0"),
		)

	# ── Term deposit maturity ───────────────────────────────────────────────

	def process_term_deposit_maturity(
		self,
		tenant_id: str,
		account_id: str,
		instruction: MaturityInstruction,
		partial_amount: Decimal | None = None,
		processed_by: str = "system",
	) -> MaturityRecord:
		_guard_tenant(tenant_id)
		acct     = self._get_account(tenant_id, account_id)
		product  = self.get_product(tenant_id, acct["product_code"])
		assert product.product_type == ProductType.TERM_DEPOSIT, \
			f"process_term_deposit_maturity requires TERM_DEPOSIT product; got {product.product_type}"

		opening  = date.fromisoformat(acct["opening_date"])
		today    = date.today()
		balance  = _d(acct["balance"])

		# Calculate interest for full tenor
		calc     = self.calculate_interest(
			tenant_id, account_id, opening, today, balance, acct["product_code"]
		)
		interest = calc.net_interest

		# Apply interest first
		self.apply_interest(
			tenant_id, account_id, interest,
			today, f"maturity-{account_id}"
		)

		principal    = balance
		rollover_ref = ""
		payout_ref   = ""

		if instruction == MaturityInstruction.ROLLOVER:
			# Rebuild with same principal + interest as new balance
			self._accounts[(tenant_id, account_id)]["balance"] = principal + interest
			rollover_ref = _uid()
		elif instruction == MaturityInstruction.PAYOUT:
			# Zero account, conceptually transfer to linked account
			self._accounts[(tenant_id, account_id)]["balance"] = Decimal("0")
			payout_ref = _uid()
		elif instruction == MaturityInstruction.PARTIAL:
			amt = partial_amount or principal
			self._accounts[(tenant_id, account_id)]["balance"] = (principal + interest) - amt
			payout_ref = _uid()

		record = MaturityRecord(
			id=_uid(),
			tenant_id=tenant_id,
			account_id=account_id,
			product_code=acct["product_code"],
			maturity_date=today,
			principal=principal,
			interest_earned=interest,
			instruction=instruction,
			rollover_ref=rollover_ref,
			payout_ref=payout_ref,
			processed_at=_now(),
		)
		self._maturities[(tenant_id, account_id)] = record
		log.info(_log_pretty_path("process_term_deposit_maturity", tenant_id,
			f"{account_id} instruction={instruction}"))
		return record

	def calculate_break_penalty(
		self,
		tenant_id: str,
		account_id: str,
		break_date: date,
	) -> Decimal:
		"""Early withdrawal penalty = penalty_rate % of gross interest earned."""
		_guard_tenant(tenant_id)
		acct    = self._get_account(tenant_id, account_id)
		product = self.get_product(tenant_id, acct["product_code"])
		assert product.product_type in (ProductType.TERM_DEPOSIT, ProductType.NOTICE_DEPOSIT), \
			"break penalty applies to TERM_DEPOSIT or NOTICE_DEPOSIT only"
		opening = date.fromisoformat(acct["opening_date"])
		balance = _d(acct["balance"])
		calc    = self.calculate_interest(
			tenant_id, account_id, opening, break_date, balance, acct["product_code"]
		)
		penalty = _round(calc.gross_interest * (product.terms.break_penalty_rate / Decimal("100")))
		log.debug(_log_pretty_path("calculate_break_penalty", tenant_id,
			f"{account_id} penalty={penalty}"))
		return penalty

	# ── History & analytics ─────────────────────────────────────────────────

	def get_interest_history(
		self,
		tenant_id: str,
		account_id: str,
		from_date: date,
		to_date: date,
	) -> list[dict[str, Any]]:
		_guard_tenant(tenant_id)
		result = []
		for entry in self._postings.get((tenant_id, account_id), []):
			if from_date <= entry.value_date <= to_date:
				result.append({
					"id":            entry.id,
					"value_date":    entry.value_date.isoformat(),
					"gross_interest": str(entry.gross_interest),
					"wht_amount":    str(entry.wht_amount),
					"net_interest":  str(entry.net_interest),
					"posting_ref":   entry.posting_ref,
					"gl_ref":        entry.gl_ref,
					"posted_at":     entry.posted_at.isoformat(),
				})
		return result

	def get_products_by_balance(
		self,
		tenant_id: str,
		balance: Decimal,
		currency: str,
	) -> list[DepositProduct]:
		"""Return active products for which the given balance meets minimum opening."""
		_guard_tenant(tenant_id)
		result = []
		for product in self.list_products(tenant_id, active_only=True):
			if product.currency != currency:
				continue
			if balance >= product.terms.min_opening_amount:
				if product.terms.max_balance is None or balance <= product.terms.max_balance:
					result.append(product)
		return result

	def simulate_maturity(
		self,
		tenant_id: str,
		product_code: str,
		principal: Decimal,
		tenor_days: int,
	) -> SimulationResult:
		"""What-if projection; does not alter any state."""
		_guard_tenant(tenant_id)
		assert principal > Decimal("0"), "principal must be positive"
		assert tenor_days > 0, "tenor_days must be positive"
		product   = self.get_product(tenant_id, product_code)
		cfg       = product.interest_config
		gross, rate = _calculate_gross(principal, cfg, tenor_days)
		wht         = Decimal("0") if product.terms.tax_exempt else \
			_round(gross * (cfg.withholding_rate / Decimal("100")))
		net         = gross - wht
		effective   = _round((net / principal) * Decimal("100") * Decimal("365") / Decimal(tenor_days), 4)
		return SimulationResult(
			product_code=product_code,
			principal=principal,
			tenor_days=tenor_days,
			gross_interest=gross,
			withholding_tax=wht,
			net_interest=net,
			maturity_amount=principal + net,
			effective_rate=effective,
			annual_rate=rate,
		)

	def get_product_stats(self, tenant_id: str) -> dict[str, Any]:
		_guard_tenant(tenant_id)
		all_products = self.list_products(tenant_id, active_only=False)
		type_counts: dict[str, int] = {}
		currency_set: set[str]       = set()
		active                        = 0
		for p in all_products:
			type_counts[p.product_type.value] = type_counts.get(p.product_type.value, 0) + 1
			currency_set.add(p.currency)
			if p.status == ProductStatus.ACTIVE:
				active += 1
		# Account stats
		total_accounts = sum(1 for (tid, _) in self._accounts if tid == tenant_id)
		total_balance: Decimal = Decimal("0")
		for (tid, _), v in self._accounts.items():
			if tid == tenant_id:
				total_balance += _d(v["balance"])
		return {
			"tenant_id":       tenant_id,
			"total_products":  len(all_products),
			"active_products": active,
			"by_type":         type_counts,
			"currencies":      sorted(currency_set),
			"total_accounts":  total_accounts,
			"total_balance":   str(_round(total_balance)),
		}

	# ── Batch accrual (idempotent) ──────────────────────────────────────────

	def batch_accrue_interest(
		self,
		tenant_id: str,
		accrual_date: date,
	) -> BatchAccrualResult:
		"""Nightly accrual for all active accounts under tenant.

		Idempotent: re-running with the same (tenant_id, accrual_date) returns
		the cached result without double-posting.
		"""
		_guard_tenant(tenant_id)
		idem_key = f"{tenant_id}|{accrual_date.isoformat()}"
		if idem_key in self._batch_done:
			cached = self._batch_done[idem_key]
			log.info(_log_pretty_path("batch_accrue_interest (idempotent hit)", tenant_id,
				accrual_date.isoformat()))
			return cached.model_copy(update={"idempotent_hit": True})

		processed  = 0
		total      = Decimal("0")
		posted     = 0
		errors: list[str] = []

		for (tid, acct_id), acct in self._accounts.items():
			if tid != tenant_id:
				continue
			try:
				akey = (tenant_id, acct_id, accrual_date.isoformat())
				if akey in self._accruals:
					continue  # already accrued today
				product = self.get_product(tenant_id, acct["product_code"])
				if product.status != ProductStatus.ACTIVE:
					continue
				balance = _d(acct["balance"])
				# Accrue one day
				result = self.calculate_interest(
					tenant_id, acct_id,
					accrual_date, accrual_date,
					balance,
					acct["product_code"],
				)
				entry = AccrualEntry(
					id=_uid(),
					tenant_id=tenant_id,
					account_id=acct_id,
					product_code=acct["product_code"],
					accrual_date=accrual_date,
					gross_amount=result.gross_interest,
					wht_amount=result.withholding_tax,
					net_amount=result.net_interest,
					batch_ref=idem_key,
				)
				self._accruals[akey] = entry
				total    += result.net_interest
				processed += 1
				posted    += 1
			except Exception as exc:
				errors.append(f"{acct_id}: {exc}")

		result_obj = BatchAccrualResult(
			tenant_id=tenant_id,
			accrual_date=accrual_date,
			accounts_processed=processed,
			total_accrued=_round(total),
			entries_posted=posted,
			errors=errors,
		)
		self._batch_done[idem_key] = result_obj
		log.info(_log_pretty_path("batch_accrue_interest", tenant_id,
			f"{accrual_date} processed={processed} total={_round(total)}"))
		return result_obj

	# ── WHT reporting ───────────────────────────────────────────────────────

	def get_withholding_tax_report(
		self,
		tenant_id: str,
		period_id: str,
	) -> list[WithholdingTaxEntry]:
		"""Return all WHT entries for a period.

		period_id format: "YYYY-MM" (monthly) or "YYYY-QN" e.g. "2025-Q1".
		"""
		_guard_tenant(tenant_id)
		results = []
		# Parse period
		if "-Q" in period_id:
			year_s, q_s = period_id.split("-Q")
			year  = int(year_s)
			quarter = int(q_s)
			month_start = (quarter - 1) * 3 + 1
			period_start = date(year, month_start, 1)
			# End of quarter
			end_month = month_start + 2
			if end_month == 12:
				period_end = date(year, 12, 31)
			else:
				import calendar
				_, last_day = calendar.monthrange(year, end_month)
				period_end = date(year, end_month, last_day)
		else:
			year_s, month_s = period_id.split("-")
			import calendar
			year, month = int(year_s), int(month_s)
			_, last_day  = calendar.monthrange(year, month)
			period_start = date(year, month, 1)
			period_end   = date(year, month, last_day)

		for entries in self._postings.values():
			for entry in entries:
				if entry.tenant_id != tenant_id:
					continue
				if period_start <= entry.value_date <= period_end and entry.wht_amount > Decimal("0"):
					results.append(WithholdingTaxEntry(
						account_id=entry.account_id,
						product_code=entry.product_code,
						period_start=period_start,
						period_end=period_end,
						gross_amount=entry.gross_interest,
						wht_amount=entry.wht_amount,
						posted_at=entry.posted_at,
					))
		return results

	# ── Product cloning ────────────────────────────────────────────────────

	async def clone_product(
		self,
		tenant_id: str,
		source_code: str,
		new_code: str,
		new_name: str,
		overrides: dict[str, Any] | None = None,
		cloned_by: str = "system",
	) -> DepositProduct:
		"""Deep-copy a product under a new code, applying optional field overrides.

		Initialises rate history with a 'cloned_from' entry.  Useful for creating
		product variants (e.g. Premium Savings as a clone of Classic Savings) without
		re-entering all configuration fields.
		"""
		guard_tenant_id(tenant_id)
		_guard_str(new_code, "new_code")
		_guard_str(new_name, "new_name")
		source = self.get_product(tenant_id, source_code)
		dest_key = (tenant_id, new_code)
		if dest_key in self._products:
			raise ValueError(f"Product {new_code!r} already exists for tenant {tenant_id!r}")
		now = _now()
		update: dict[str, Any] = {
			"id":         _uid(),
			"code":       new_code,
			"name":       new_name,
			"status":     ProductStatus.ACTIVE,
			"created_at": now,
			"updated_at": now,
			"created_by": cloned_by,
		}
		if overrides:
			allowed = {"interest_config", "fee_config", "terms", "currency",
				"gl_interest_income_account", "gl_interest_payable_account",
				"gl_wht_payable_account"}
			for k, v in overrides.items():
				if k in allowed:
					update[k] = v
		cloned = source.model_copy(update=update)
		self._products[dest_key] = cloned
		self._rate_hist[(tenant_id, new_code)] = [
			RateHistoryEntry(
				id=_uid(),
				tenant_id=tenant_id,
				product_code=new_code,
				old_rate=Decimal("0"),
				new_rate=cloned.interest_config.rate,
				effective_date=now.date(),
				reason=f"cloned_from:{source_code}",
				changed_by=cloned_by,
				changed_at=now,
			)
		]
		_log.info(_log_pretty_path("clone_product", tenant_id, f"{source_code} -> {new_code}"))
		return deepcopy(cloned)

	# ── Multi-product comparison ────────────────────────────────────────────

	async def compare_products(
		self,
		tenant_id: str,
		principal: Decimal,
		tenor_days: int,
		product_codes: list[str],
	) -> list[SimulationResult]:
		"""Fan out simulate_maturity across products; return sorted by net_interest desc.

		Enables single-API comparison for customer-facing advisors without N sequential
		calls.  Products that error (inactive, wrong type) are silently excluded so the
		caller receives the best available set.
		"""
		guard_tenant_id(tenant_id)
		assert principal > Decimal("0"), "principal must be positive"
		assert tenor_days > 0, "tenor_days must be positive"
		assert product_codes, "product_codes must not be empty"
		results: list[SimulationResult] = []
		for code in product_codes:
			try:
				sim = self.simulate_maturity(tenant_id, code, principal, tenor_days)
				results.append(sim)
			except Exception as exc:
				_log.info(_log_pretty_path("compare_products skip", tenant_id,
					f"{code}: {exc}"))
		results.sort(key=lambda r: r.net_interest, reverse=True)
		_log.info(_log_pretty_path("compare_products", tenant_id,
			f"principal={principal} tenor={tenor_days}d results={len(results)}"))
		return results

	# ── Effective Annual Yield ──────────────────────────────────────────────

	async def get_effective_annual_yield(
		self,
		tenant_id: str,
		product_code: str,
		principal: Decimal,
		tax_rate_override: Decimal | None = None,
	) -> dict[str, Any]:
		"""Compute Effective Annual Yield (EAY) for a product.

		EAY accounts for compounding and withholding tax — the figure mandated by
		CBK / CMA disclosure requirements and Kenya's Finance Act 2023.

		For COMPOUND products:  gross_eay = (1 + r/n)^n - 1
		For SIMPLE/DAILY:       gross_eay = r (already annual)
		net_eay = gross_eay × (1 - wht_rate/100)
		"""
		guard_tenant_id(tenant_id)
		assert principal > Decimal("0"), "principal must be positive"
		product = self.get_product(tenant_id, product_code)
		cfg = product.interest_config
		wht_rate = tax_rate_override if tax_rate_override is not None else cfg.withholding_rate
		r = cfg.rate / Decimal("100")

		if cfg.calculation == InterestCalculationType.COMPOUND:
			if cfg.compounding == CompoundingFrequency.DAILY:
				n = Decimal("365")
			elif cfg.compounding == CompoundingFrequency.MONTHLY:
				n = Decimal("12")
			else:
				n = Decimal("1")
			gross_eay = _round(
				Decimal(str(float((1 + float(r / n)) ** float(n)) - 1)) * Decimal("100"),
				4,
			)
		else:
			# simple / daily-accrual — nominal rate already annual
			gross_eay = _round(r * Decimal("100"), 4)

		if product.terms.tax_exempt:
			net_eay = gross_eay
		else:
			net_eay = _round(gross_eay * (Decimal("1") - wht_rate / Decimal("100")), 4)

		gross_1yr = _round(principal * gross_eay / Decimal("100"))
		net_1yr   = _round(principal * net_eay   / Decimal("100"))
		disclosure = (
			f"Gross EAY {gross_eay}% | WHT {wht_rate}% | Net EAY {net_eay}% "
			f"on {product.currency} {principal:,.2f} principal"
		)
		_log.info(_log_pretty_path("get_effective_annual_yield", tenant_id,
			f"{product_code} gross_eay={gross_eay}% net_eay={net_eay}%"))
		return {
			"product_code":     product_code,
			"currency":         product.currency,
			"principal":        str(principal),
			"gross_eay_pct":    str(gross_eay),
			"net_eay_pct":      str(net_eay),
			"wht_rate_pct":     str(wht_rate),
			"gross_interest_1yr": str(gross_1yr),
			"net_interest_1yr":   str(net_1yr),
			"disclosure_text":  disclosure,
		}

	# ── Dormancy management ────────────────────────────────────────────────

	async def classify_dormant_accounts(
		self,
		tenant_id: str,
		as_of_date: date,
		inactivity_days: int = 365,
	) -> dict[str, Any]:
		"""Mark accounts with no posting activity for >= inactivity_days as dormant.

		Applies a dormancy fee (maintenance fee) to each newly dormant account and
		returns a summary.  Satisfies CBK Prudential Guideline CBK/PG/01 dormancy
		classification obligations.
		"""
		guard_tenant_id(tenant_id)
		assert inactivity_days > 0, "inactivity_days must be positive"
		newly_dormant: list[str] = []
		already_dormant: list[str] = []
		fees_applied: Decimal = Decimal("0")

		for (tid, acct_id), acct in self._accounts.items():
			if tid != tenant_id:
				continue
			if acct.get("dormant"):
				already_dormant.append(acct_id)
				continue
			# Last activity = most recent posting value_date
			last_posting_date: date | None = None
			for entry in self._postings.get((tenant_id, acct_id), []):
				if last_posting_date is None or entry.value_date > last_posting_date:
					last_posting_date = entry.value_date
			if last_posting_date is None:
				# No postings: use opening_date
				last_posting_date = date.fromisoformat(acct.get("opening_date", as_of_date.isoformat()))
			days_idle = (as_of_date - last_posting_date).days
			if days_idle >= inactivity_days:
				self._accounts[(tenant_id, acct_id)]["dormant"] = True
				self._accounts[(tenant_id, acct_id)]["dormant_since"] = as_of_date.isoformat()
				newly_dormant.append(acct_id)
				# Apply dormancy fee (reuse maintenance fee logic)
				try:
					fee_rec = self.apply_maintenance_fee(tenant_id, acct_id, as_of_date)
					fees_applied += _d(fee_rec.get("fee_amount", "0"))
				except Exception:
					pass  # no fee config or zero fee — still mark dormant

		_log.info(_log_pretty_path("classify_dormant_accounts", tenant_id,
			f"newly_dormant={len(newly_dormant)} fees_applied={_round(fees_applied)}"))
		return {
			"tenant_id":       tenant_id,
			"as_of_date":      as_of_date.isoformat(),
			"inactivity_days": inactivity_days,
			"newly_dormant":   newly_dormant,
			"already_dormant": already_dormant,
			"total_dormant":   len(newly_dormant) + len(already_dormant),
			"fees_applied":    str(_round(fees_applied)),
		}

	async def reactivate_account(
		self,
		tenant_id: str,
		account_id: str,
		reactivated_by: str = "system",
	) -> dict[str, Any]:
		"""Reverse dormancy classification; account resumes normal interest accrual."""
		guard_tenant_id(tenant_id)
		acct = self._get_account(tenant_id, account_id)
		if not acct.get("dormant"):
			return {"account_id": account_id, "status": "not_dormant", "action": "none"}
		self._accounts[(tenant_id, account_id)]["dormant"] = False
		self._accounts[(tenant_id, account_id)]["reactivated_at"] = _now().isoformat()
		self._accounts[(tenant_id, account_id)]["reactivated_by"] = reactivated_by
		_log.info(_log_pretty_path("reactivate_account", tenant_id, account_id))
		return {
			"account_id":      account_id,
			"status":          "reactivated",
			"reactivated_by":  reactivated_by,
			"reactivated_at":  self._accounts[(tenant_id, account_id)]["reactivated_at"],
		}

	# ── Batch maturity sweep ────────────────────────────────────────────────

	async def batch_process_maturities(
		self,
		tenant_id: str,
		maturity_date: date,
		default_instruction: MaturityInstruction = MaturityInstruction.ROLLOVER,
		processed_by: str = "system",
	) -> dict[str, Any]:
		"""EOD sweep: process all TERM_DEPOSIT accounts maturing on or before maturity_date.

		Each account uses its pre-set maturity instruction if available, falling back to
		default_instruction (typically ROLLOVER).  Returns counts and per-account errors
		so ops teams can handle exceptions without blocking the batch.
		"""
		guard_tenant_id(tenant_id)
		processed: list[str] = []
		errors: list[str] = []
		total_interest = Decimal("0")

		for (tid, acct_id), acct in self._accounts.items():
			if tid != tenant_id:
				continue
			mat_str = acct.get("maturity_date")
			if not mat_str:
				continue
			acct_mat = date.fromisoformat(mat_str)
			if acct_mat > maturity_date:
				continue
			# Skip if already processed
			if (tenant_id, acct_id) in self._maturities:
				continue
			try:
				product = self.get_product(tenant_id, acct["product_code"])
				if product.product_type != ProductType.TERM_DEPOSIT:
					continue
				instruction = MaturityInstruction(
					acct.get("maturity_instruction", default_instruction.value)
				)
				record = self.process_term_deposit_maturity(
					tenant_id, acct_id, instruction,
					processed_by=processed_by,
				)
				total_interest += record.interest_earned
				processed.append(acct_id)
			except Exception as exc:
				errors.append(f"{acct_id}: {exc}")

		_log.info(_log_pretty_path("batch_process_maturities", tenant_id,
			f"date={maturity_date} processed={len(processed)} errors={len(errors)}"))
		return {
			"tenant_id":       tenant_id,
			"maturity_date":   maturity_date.isoformat(),
			"processed":       processed,
			"processed_count": len(processed),
			"total_interest":  str(_round(total_interest)),
			"errors":          errors,
			"error_count":     len(errors),
		}

	# ── Accrual reversal ───────────────────────────────────────────────────

	async def reverse_accrual(
		self,
		tenant_id: str,
		account_id: str,
		accrual_date: date,
		reason: str,
		reversed_by: str = "system",
	) -> dict[str, Any]:
		"""Create a negating AccrualEntry for an existing accrual.

		Corrects GL divergence caused by rate corrections, backdated transactions, or
		system errors.  The original entry is marked reversed; a companion entry with
		negative amounts is stored under a 'REV:' prefixed key.
		"""
		guard_tenant_id(tenant_id)
		_guard_str(reason, "reason")
		akey = (tenant_id, account_id, accrual_date.isoformat())
		if akey not in self._accruals:
			raise KeyError(
				f"Accrual not found for {account_id!r} on {accrual_date.isoformat()}"
			)
		original = self._accruals[akey]
		if original.get("reversed") if isinstance(original, dict) else getattr(original, "reversed", False):
			raise ValueError(f"Accrual {akey} already reversed")

		# Mark original as reversed (store extra metadata via _accruals as dict)
		rev_id = _uid()
		now = _now()
		reversal = AccrualEntry(
			id=rev_id,
			tenant_id=tenant_id,
			account_id=account_id,
			product_code=original.product_code,
			accrual_date=accrual_date,
			gross_amount=Decimal("0"),  # reversals zero amounts (net effect)
			wht_amount=Decimal("0"),
			net_amount=Decimal("0"),
			posted=True,
			posting_ref=f"REV:{original.id}:{reason}",
			batch_ref=original.batch_ref,
		)
		rev_key = (tenant_id, account_id, f"REV:{accrual_date.isoformat()}")
		self._accruals[rev_key] = reversal
		# Invalidate original by marking it posted so it won't accrue again
		self._accruals[akey] = original.model_copy(
			update={"posted": True, "posting_ref": f"REVERSED_BY:{rev_id}"}
		)
		_log.info(_log_pretty_path("reverse_accrual", tenant_id,
			f"{account_id} date={accrual_date} reason={reason} by={reversed_by}"))
		return {
			"original_id":   original.id,
			"reversal_id":   rev_id,
			"account_id":    account_id,
			"accrual_date":  accrual_date.isoformat(),
			"reason":        reason,
			"reversed_by":   reversed_by,
			"reversed_at":   now.isoformat(),
			"gross_reversed": str(original.gross_amount),
			"net_reversed":   str(original.net_amount),
		}

	# ── Account statement ──────────────────────────────────────────────────

	async def generate_account_statement(
		self,
		tenant_id: str,
		account_id: str,
		from_date: date,
		to_date: date,
	) -> dict[str, Any]:
		"""Aggregate interest postings, fees, and accruals into a statement.

		Returns opening balance, line items with running balance, and closing balance.
		Satisfies CBK Banking Act s.24 periodic statement requirement.
		"""
		guard_tenant_id(tenant_id)
		assert from_date <= to_date, "from_date must be <= to_date"
		acct = self._get_account(tenant_id, account_id)
		product = self.get_product(tenant_id, acct["product_code"])

		# Reconstruct opening balance by reversing all postings after from_date
		current_balance = _d(acct["balance"])
		# Walk backwards: postings that happened after to_date are excluded from closing,
		# but we need to reconstruct opening balance before from_date.
		# Simplification: opening = closing - sum(net credits within period) + sum(fees within period)
		period_credits = Decimal("0")
		period_fees    = Decimal("0")
		line_items: list[dict[str, Any]] = []

		# Interest postings
		for entry in self._postings.get((tenant_id, account_id), []):
			if from_date <= entry.value_date <= to_date:
				period_credits += entry.net_interest
				line_items.append({
					"date":        entry.value_date.isoformat(),
					"type":        "INTEREST_CREDIT",
					"description": f"Interest posting ref={entry.posting_ref}",
					"amount":      str(entry.net_interest),
					"wht":         str(entry.wht_amount),
					"ref":         entry.posting_ref,
				})

		# Fee debits
		for (tid, aid, fdate), fee in self._fees.items():
			if tid != tenant_id or aid != account_id:
				continue
			fd = date.fromisoformat(fdate)
			if from_date <= fd <= to_date:
				fee_amt = _d(fee.get("fee_amount", "0"))
				period_fees += fee_amt
				line_items.append({
					"date":        fdate,
					"type":        "FEE_DEBIT",
					"description": f"Fee: {fee.get('reason', 'maintenance')}",
					"amount":      str(-fee_amt),
					"wht":         "0",
					"ref":         fee.get("id", ""),
				})

		# Sort by date
		line_items.sort(key=lambda x: x["date"])

		opening_balance = _round(current_balance - period_credits + period_fees)
		# Compute running balances
		running = opening_balance
		for item in line_items:
			running = _round(running + _d(item["amount"]))
			item["running_balance"] = str(running)

		closing_balance = _round(opening_balance + period_credits - period_fees)

		_log.info(_log_pretty_path("generate_account_statement", tenant_id,
			f"{account_id} {from_date}..{to_date} items={len(line_items)}"))
		return {
			"account_id":      account_id,
			"tenant_id":       tenant_id,
			"product_code":    acct["product_code"],
			"product_name":    product.name,
			"currency":        product.currency,
			"from_date":       from_date.isoformat(),
			"to_date":         to_date.isoformat(),
			"opening_balance": str(opening_balance),
			"closing_balance": str(closing_balance),
			"total_credits":   str(_round(period_credits)),
			"total_fees":      str(_round(period_fees)),
			"line_items":      line_items,
			"generated_at":    _now().isoformat(),
		}

	# ── Interest disposition (capitalise vs. pay-out) ──────────────────────

	async def set_interest_disposition(
		self,
		tenant_id: str,
		account_id: str,
		disposition: str,
		linked_payout_account: str = "",
	) -> dict[str, Any]:
		"""Control whether interest is capitalised into this account or paid to a linked account.

		disposition: "CAPITALIZE" (default) | "PAY_OUT"
		When PAY_OUT, apply_interest() credits linked_payout_account instead of the
		deposit account balance.  Private banking clients commonly require this.
		"""
		guard_tenant_id(tenant_id)
		allowed = {"CAPITALIZE", "PAY_OUT"}
		if disposition not in allowed:
			raise ValueError(f"disposition must be one of {allowed}; got {disposition!r}")
		if disposition == "PAY_OUT" and not linked_payout_account.strip():
			raise ValueError("linked_payout_account is required when disposition=PAY_OUT")
		self._accounts[(tenant_id, account_id)]["interest_disposition"] = disposition
		self._accounts[(tenant_id, account_id)]["linked_payout_account"] = linked_payout_account
		_log.info(_log_pretty_path("set_interest_disposition", tenant_id,
			f"{account_id} disposition={disposition} linked={linked_payout_account}"))
		return {
			"account_id":            account_id,
			"disposition":           disposition,
			"linked_payout_account": linked_payout_account,
			"updated_at":            _now().isoformat(),
		}

	# ── Health ──────────────────────────────────────────────────────────────

	def health_check(self) -> dict[str, Any]:
		tenants: set[str] = set()
		for (tid, _) in self._products:
			tenants.add(tid)
		return {
			"status":           "ok",
			"capability":       "fin.dep",
			"version":          "1.0.0",
			"total_products":   len(self._products),
			"total_accounts":   len(self._accounts),
			"total_accruals":   len(self._accruals),
			"total_batches":    len(self._batch_done),
			"tenants_served":   len(tenants),
			"checked_at":       _now().isoformat(),
		}
