"""Deterministic domain rules for Digital Payments.

These rules are the single source of truth for all governance decisions.
All functions are pure (no I/O), raising RuleViolation on any violation.
"""
from __future__ import annotations

import re as _re
from decimal import Decimal as _D
from typing import Any


class RuleViolation(Exception):
	"""Raised when a business rule is violated."""

	def __init__(self, rule_name: str, reason: str, required_action: str = "") -> None:
		self.rule_name = rule_name
		self.reason = reason
		self.required_action = required_action
		super().__init__(f"Rule '{rule_name}' violated: {reason}")


# ─────────────────────────────────────────────────────────────
# Multi-tenancy
# ─────────────────────────────────────────────────────────────

def assert_tenant_context(context: dict[str, Any]) -> None:
	"""All operations require a tenant context."""
	if not context.get("tenant_id"):
		raise RuleViolation("tenant_context_required", "tenant_id is required", "attach_tenant_context")


def assert_write_policy(context: dict[str, Any]) -> None:
	"""Write operations require an attached policy."""
	if context.get("operation_type") == "write" and not context.get("policy_attached"):
		raise RuleViolation("write_requires_policy", "write operations require an attached policy", "attach_policy")


def assert_no_cross_tenant_access(actor_tenant: str, resource_tenant: str) -> None:
	"""Cross-tenant access is always denied."""
	if actor_tenant != resource_tenant:
		raise RuleViolation(
			"cross_tenant_access_denied",
			"cross-tenant access is not permitted",
			"use_own_tenant_resources",
		)


# ─────────────────────────────────────────────────────────────
# Amount guards
# ─────────────────────────────────────────────────────────────

def assert_positive_amount(amount: Any, field: str = "amount") -> None:
	"""Amount must be strictly positive."""
	try:
		val = _D(str(amount))
	except Exception:
		raise RuleViolation("invalid_amount", f"{field} must be a valid number", "provide_valid_amount")
	if val <= 0:
		raise RuleViolation("non_positive_amount", f"{field} must be greater than zero", "provide_positive_amount")


def assert_amount_precision(amount: Any, max_decimal_places: int = 2, field: str = "amount") -> None:
	"""Amount must not exceed the specified decimal precision."""
	try:
		val = _D(str(amount))
		_sign, _digits, exponent = val.as_tuple()
		actual_places = max(0, -exponent)
		if actual_places > max_decimal_places:
			raise RuleViolation(
				"amount_precision_exceeded",
				f"{field} exceeds {max_decimal_places} decimal places",
				"round_amount_to_valid_precision",
			)
	except RuleViolation:
		raise
	except Exception:
		raise RuleViolation("invalid_amount_precision", f"{field} must be a valid decimal", "provide_valid_amount")


def assert_supported_currency(currency: str, supported_currencies: list[str] | None = None) -> None:
	"""Currency must be in the supported set."""
	defaults = ["KES", "UGX", "TZS", "RWF", "GHS", "NGN", "ZAR", "USD", "EUR", "GBP", "XOF", "XAF"]
	allowed = supported_currencies or defaults
	if currency not in allowed:
		raise RuleViolation("unsupported_currency", f"currency {currency!r} is not supported", "use_supported_currency")


# ─────────────────────────────────────────────────────────────
# KYC / transaction limits  (CBK Prudential Guidelines)
# ─────────────────────────────────────────────────────────────

# Per-transaction limits
_KYC_PER_TXN = {
	"basic":    _D("150000"),
	"standard": _D("500000"),
	"full_kyc": _D("1000000"),
	"enhanced": _D("999999999"),
}

# Daily limits
_KYC_DAILY = {
	"basic":    _D("300000"),
	"standard": _D("1000000"),
	"full_kyc": _D("5000000"),
	"enhanced": _D("999999999"),
}

# Monthly limits
_KYC_MONTHLY = {
	"basic":    _D("3000000"),
	"standard": _D("10000000"),
	"full_kyc": _D("50000000"),
	"enhanced": _D("999999999"),
}


def assert_kyc_per_txn_limit(amount: Any, tier: str = "basic") -> None:
	"""Single transaction must not exceed the KYC tier per-transaction limit."""
	limit = _KYC_PER_TXN.get(tier, _D("150000"))
	if _D(str(amount)) > limit:
		raise RuleViolation(
			"kyc_per_txn_limit_exceeded",
			f"amount exceeds per-transaction limit of {limit} for tier {tier!r}",
			"upgrade_kyc_tier",
		)


def assert_kyc_daily_limit(daily_used: Any, new_amount: Any, tier: str = "basic") -> None:
	"""Cumulative daily spend must not exceed the KYC tier daily limit."""
	limit = _KYC_DAILY.get(tier, _D("300000"))
	if _D(str(daily_used)) + _D(str(new_amount)) > limit:
		raise RuleViolation(
			"kyc_daily_limit_exceeded",
			f"daily limit of {limit} exceeded for tier {tier!r}",
			"wait_for_next_day",
		)


def assert_kyc_monthly_limit(monthly_used: Any, new_amount: Any, tier: str = "basic") -> None:
	"""Cumulative monthly spend must not exceed the KYC tier monthly limit."""
	limit = _KYC_MONTHLY.get(tier, _D("3000000"))
	if _D(str(monthly_used)) + _D(str(new_amount)) > limit:
		raise RuleViolation(
			"kyc_monthly_limit_exceeded",
			f"monthly limit of {limit} exceeded for tier {tier!r}",
			"upgrade_kyc_tier",
		)


# ─────────────────────────────────────────────────────────────
# Duplicate / idempotency
# ─────────────────────────────────────────────────────────────

def assert_no_duplicate(reference: str, existing_refs: set[str] | list[str], window_desc: str = "5 minutes") -> None:
	"""Reference must not already exist in the dedup window.

	Empty references are ignored (not tracked).
	"""
	if not reference:
		return
	if reference in existing_refs:
		raise RuleViolation(
			"duplicate_payment_detected",
			f"payment with reference {reference!r} already exists within {window_desc}",
			"use_unique_reference",
		)


def assert_retry_window(retry_count: int, max_retries: int) -> None:
	"""Retry count must be strictly less than the maximum allowed retries."""
	if retry_count >= max_retries:
		raise RuleViolation(
			"max_retries_exceeded",
			f"retry count {retry_count} has reached maximum of {max_retries}",
			"create_new_payment",
		)


# ─────────────────────────────────────────────────────────────
# M-Pesa rules
# ─────────────────────────────────────────────────────────────

_MPESA_MIN = _D("1")
_MPESA_MAX = _D("300000")   # CBK KES 300k single-transaction ceiling for STK Push


def assert_mpesa_amount(amount: Any) -> None:
	"""M-Pesa STK Push amount must be KES 1–300,000."""
	val = _D(str(amount))
	if val < _MPESA_MIN:
		raise RuleViolation(
			"mpesa_amount_below_minimum",
			f"M-Pesa amount {val} is below the minimum of {_MPESA_MIN}",
			"increase_amount",
		)
	if val > _MPESA_MAX:
		raise RuleViolation(
			"mpesa_amount_above_maximum",
			f"M-Pesa amount {val} exceeds the maximum of {_MPESA_MAX}",
			"reduce_amount_or_split",
		)


def assert_mpesa_phone(phone: str) -> None:
	"""M-Pesa phone must be in international E.164 format starting with 254."""
	digits = "".join(c for c in phone if c.isdigit())
	if not digits.startswith("254") or len(digits) != 12:
		raise RuleViolation(
			"mpesa_invalid_phone",
			f"M-Pesa phone {phone!r} must be in 254XXXXXXXXX format (12 digits starting 254)",
			"normalise_phone_to_e164",
		)


def assert_mpesa_float_sufficient(float_balance: Any, amount: Any) -> None:
	"""Agent/wallet float must be sufficient to cover the disbursement."""
	if _D(str(float_balance)) < _D(str(amount)):
		raise RuleViolation(
			"mpesa_insufficient_float",
			"M-Pesa agent float is insufficient for this disbursement",
			"top_up_float",
		)


def assert_mpesa_reference_length(reference: str) -> None:
	"""M-Pesa account reference must be 1–12 characters."""
	if not (1 <= len(reference) <= 12):
		raise RuleViolation(
			"invalid_mpesa_reference",
			f"M-Pesa account reference must be 1–12 characters, got {len(reference)}",
			"shorten_reference",
		)


# ─────────────────────────────────────────────────────────────
# Mobile money (MTN / Airtel / Tigo)
# ─────────────────────────────────────────────────────────────

def assert_momo_amount(amount: Any, currency: str = "KES") -> None:
	"""Mobile money amount must be within provider-specific range."""
	limits: dict[str, tuple[_D, _D]] = {
		"KES": (_D("1"),   _D("500000")),
		"UGX": (_D("500"), _D("5000000")),
		"GHS": (_D("1"),   _D("5000")),
		"RWF": (_D("100"), _D("1000000")),
	}
	lo, hi = limits.get(currency, (_D("1"), _D("1000000")))
	val = _D(str(amount))
	if not (lo <= val <= hi):
		raise RuleViolation(
			"momo_amount_out_of_range",
			f"Mobile money amount must be between {lo} and {hi} {currency}",
			"adjust_amount",
		)


# ─────────────────────────────────────────────────────────────
# Card rules (PCI-DSS)
# ─────────────────────────────────────────────────────────────

def assert_card_token_not_pan(value: str) -> None:
	"""Reject raw card PANs — only vault tokens are permitted."""
	# Strip common delimiters, check if what remains looks like a PAN
	cleaned = "".join(c for c in value if c.isdigit())
	if len(cleaned) in (13, 14, 15, 16, 19) and cleaned.isdigit() and len(value) == len(cleaned):
		raise RuleViolation(
			"raw_pan_storage_forbidden",
			"raw card PAN must not be stored — supply a vault token",
			"tokenise_card_before_storage",
		)


def assert_card_cvv_not_stored(value: Any) -> None:
	"""CVV must never be stored — only post-auth result codes are permitted.

	Allowed: None, or a single-character result code (M/N/P/U).
	Rejected: any numeric string of 3–4 digits.
	"""
	if value is None:
		return
	s = str(value).strip()
	if s.isdigit() and 2 <= len(s) <= 4:
		raise RuleViolation(
			"cvv_storage_forbidden",
			"CVV must never be stored after authorisation (PCI-DSS requirement 3.2)",
			"discard_cvv_after_auth",
		)


_3DS_THRESHOLD = _D("10000")   # KES 10,000 — 3DS required above this


def assert_3ds_result(eci: str | None, amount: Any) -> None:
	"""3DS authentication is required for high-value card transactions.

	Below KES 10,000 — 3DS is optional.
	At or above KES 10,000 — a valid 3DS ECI code (Y/A) is required.
	"""
	if _D(str(amount)) < _3DS_THRESHOLD:
		return
	if eci not in ("Y", "A"):
		raise RuleViolation(
			"3ds_required",
			f"3DS authentication (ECI Y or A) is required for amounts >= {_3DS_THRESHOLD} KES, got {eci!r}",
			"complete_3ds_authentication",
		)


# ─────────────────────────────────────────────────────────────
# SWIFT / IBAN
# ─────────────────────────────────────────────────────────────

def assert_swift_bic(bic: str) -> None:
	"""BIC/SWIFT code must match the ISO 9362 format (8 or 11 chars)."""
	if not _re.match(r"^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?$", bic.upper()):
		raise RuleViolation(
			"invalid_swift_bic",
			f"BIC/SWIFT code {bic!r} does not match ISO 9362 format",
			"provide_valid_bic",
		)


def assert_iban(iban: str) -> None:
	"""IBAN must match the ISO 13616 format."""
	stripped = iban.replace(" ", "").upper()
	if not (15 <= len(stripped) <= 34) or not _re.match(r"^[A-Z]{2}[0-9]{2}[A-Z0-9]+$", stripped):
		raise RuleViolation(
			"invalid_iban",
			f"IBAN {iban!r} does not match ISO 13616 format",
			"provide_valid_iban",
		)


def assert_swift_purpose_code(purpose_code: str) -> None:
	"""SWIFT purpose code must be a recognised 3-character code."""
	valid = {
		"OTH", "SAL", "SALA", "SUPP", "TRAD", "DIVI", "CHAR",
		"LOAR", "INTC", "CMDT", "DIV", "INT", "COM", "GDS", "SVC",
	}
	if purpose_code.upper() not in valid:
		raise RuleViolation(
			"invalid_swift_purpose_code",
			f"SWIFT purpose code {purpose_code!r} is not recognised",
			"use_valid_purpose_code",
		)


# ─────────────────────────────────────────────────────────────
# Refund / reversal rules
# ─────────────────────────────────────────────────────────────

def assert_refund_amount(refund_amount: Any, original_amount: Any) -> None:
	"""Refund amount must not exceed the original transaction amount."""
	if _D(str(refund_amount)) > _D(str(original_amount)):
		raise RuleViolation(
			"refund_exceeds_original",
			"refund amount cannot exceed original transaction amount",
			"reduce_refund_amount",
		)


def assert_refund_not_duplicate(
	new_refund_amount: Any,
	already_refunded: Any,
	original_amount: Any,
) -> None:
	"""Cumulative refunds must not exceed the original transaction amount."""
	cumulative = _D(str(already_refunded)) + _D(str(new_refund_amount))
	if cumulative > _D(str(original_amount)):
		raise RuleViolation(
			"cumulative_refund_exceeds_original",
			f"cumulative refund {cumulative} would exceed original amount {original_amount}",
			"reduce_refund_amount",
		)


def assert_reversal_window(original_created_at: str, window_hours: int = 24) -> None:
	"""Reversal must be initiated within the allowed window."""
	from datetime import datetime, timezone

	try:
		created = datetime.fromisoformat(original_created_at.replace("Z", "+00:00"))
		if created.tzinfo is None:
			created = created.replace(tzinfo=timezone.utc)
		age_hours = (datetime.now(timezone.utc) - created).total_seconds() / 3600
		if age_hours > window_hours:
			raise RuleViolation(
				"reversal_window_expired",
				f"reversal must be within {window_hours}h — transaction is {age_hours:.1f}h old",
				"contact_support_for_late_reversal",
			)
	except RuleViolation:
		raise
	except Exception:
		pass


def assert_refundable_status(status: str) -> None:
	"""Transaction must be in a refundable terminal status."""
	if status not in ("completed", "settled"):
		raise RuleViolation(
			"not_refundable",
			f"transaction in status {status!r} cannot be refunded",
			"wait_for_settlement",
		)


# ─────────────────────────────────────────────────────────────
# FX rules
# ─────────────────────────────────────────────────────────────

def assert_fx_rate_freshness(rate_age_seconds: float, max_age_seconds: int = 300) -> None:
	"""FX rate must have been fetched within the allowed window."""
	if rate_age_seconds > max_age_seconds:
		raise RuleViolation(
			"fx_rate_stale",
			f"FX rate is {rate_age_seconds:.0f}s old — must be refreshed every {max_age_seconds}s",
			"refresh_fx_rate",
		)


# ─────────────────────────────────────────────────────────────
# Settlement rules
# ─────────────────────────────────────────────────────────────

def assert_settlement_variance(
	variance_amount: Any,
	settlement_total: Any,
	threshold_bps: int = 10,
) -> None:
	"""Settlement variance must not exceed the allowed basis-point threshold.

	Args:
		variance_amount: Absolute variance (positive or negative).
		settlement_total: Total expected settlement amount.
		threshold_bps: Maximum variance in basis points (default 10 = 0.1%).
	"""
	total = _D(str(settlement_total))
	var = abs(_D(str(variance_amount)))
	if total > 0:
		actual_bps = (var / total * _D("10000")).quantize(_D("0.01"))
		if actual_bps > _D(str(threshold_bps)):
			raise RuleViolation(
				"settlement_variance_exceeded",
				f"settlement variance {actual_bps} bps exceeds threshold of {threshold_bps} bps",
				"investigate_settlement_break",
			)


# ─────────────────────────────────────────────────────────────
# AML / velocity rules
# ─────────────────────────────────────────────────────────────

_AML_CTR_KES = _D("1000000")    # CBK Currency Transaction Report threshold
_AML_CTR_NGN = _D("5000000")    # CBN
_AML_CTR_UGX = _D("20000000")   # BoU


def assert_aml_velocity(
	transactions_last_24h: int,
	amount_sum_last_24h: Any,
	current_amount: Any,
	currency: str = "KES",
) -> None:
	"""Flag structuring / velocity patterns that require AML review.

	Triggers on:
	  - >5 transactions in 24h AND cumulative amount > 80% of CTR threshold, OR
	  - Single transaction at or above CTR threshold.
	"""
	thresholds = {"KES": _AML_CTR_KES, "NGN": _AML_CTR_NGN, "UGX": _AML_CTR_UGX}
	ctr_threshold = thresholds.get(currency.upper(), _AML_CTR_KES)
	total = _D(str(amount_sum_last_24h)) + _D(str(current_amount))

	# Structuring: multiple transactions approaching threshold
	if transactions_last_24h > 5 and total >= (ctr_threshold * _D("0.8")):
		raise RuleViolation(
			"aml_velocity_threshold",
			f"{transactions_last_24h} transactions totalling {total} {currency} in 24h "
			f"approaches CTR threshold — AML review required",
			"submit_sar_for_review",
		)

	# Single large transaction
	if _D(str(current_amount)) >= ctr_threshold:
		raise RuleViolation(
			"aml_velocity_threshold",
			f"Single transaction of {current_amount} {currency} meets CTR threshold of {ctr_threshold}",
			"file_currency_transaction_report",
		)


# ─────────────────────────────────────────────────────────────
# Batch rules
# ─────────────────────────────────────────────────────────────

def assert_batch_size(batch_size: int, max_batch: int = 1000) -> None:
	"""Batch size must not exceed the maximum allowed items."""
	if batch_size > max_batch:
		raise RuleViolation(
			"batch_too_large",
			f"batch of {batch_size} exceeds maximum of {max_batch}",
			"split_into_smaller_batches",
		)


def assert_batch_lists_aligned(*lists: list) -> None:
	"""All batch input lists must have equal length."""
	lengths = [len(lst) for lst in lists]
	if len(set(lengths)) > 1:
		raise RuleViolation(
			"batch_lists_misaligned",
			f"batch lists have different lengths: {lengths}",
			"align_batch_list_lengths",
		)


# ─────────────────────────────────────────────────────────────
# Webhook rules
# ─────────────────────────────────────────────────────────────

def assert_webhook_url(url: str) -> None:
	"""Webhook URLs must use HTTPS."""
	if not url.startswith("https://"):
		raise RuleViolation(
			"webhook_url_must_use_https",
			"webhook URL must use HTTPS — plain HTTP is not permitted",
			"update_webhook_url_to_https",
		)


def assert_mcc_code(mcc: str) -> None:
	"""Merchant Category Code must be a 4-digit numeric string."""
	if not _re.match(r"^\d{4}$", str(mcc)):
		raise RuleViolation(
			"invalid_mcc",
			f"merchant category code {mcc!r} must be a 4-digit number",
			"provide_valid_mcc",
		)


def assert_valid_phone(phone: str, country_prefix: str = "254") -> None:
	"""Generic phone validation: 9–15 digits."""
	digits = "".join(c for c in phone if c.isdigit())
	if len(digits) < 9 or len(digits) > 15:
		raise RuleViolation(
			"invalid_phone",
			"phone number must be 9–15 digits",
			"provide_valid_phone_number",
		)


# ─────────────────────────────────────────────────────────────
# Regulatory obligation calculations
# ─────────────────────────────────────────────────────────────

def calculate_ctr_obligation(amount: Any, currency: str = "KES", threshold: Any = None) -> dict:
	"""Determine if a transaction requires a Currency Transaction Report.

	Thresholds:
	  KES → CBK: KES 1,000,000
	  NGN → CBN: NGN 5,000,000
	  UGX → BoU: UGX 20,000,000
	  Others  → generic central_bank threshold of USD 10,000 equivalent
	"""
	thresholds = {
		"KES": (_AML_CTR_KES, "CBK"),
		"NGN": (_AML_CTR_NGN, "CBN"),
		"UGX": (_AML_CTR_UGX, "BoU"),
	}
	if threshold is not None:
		limit = _D(str(threshold))
		regulator = "central_bank"
	else:
		limit, regulator = thresholds.get(currency.upper(), (_D("1300000"), "central_bank"))

	val = _D(str(amount))
	return {
		"requires_ctr": val >= limit,
		"amount": str(val),
		"currency": currency.upper(),
		"threshold": str(limit),
		"regulator": regulator,
		"report_to": regulator,
	}


# ─────────────────────────────────────────────────────────────
# Payment calculation helpers (delegated from calculations.py for
# convenience — these are re-exported here so test_rules.py can
# import everything from a single module)
# ─────────────────────────────────────────────────────────────

def calculate_mpesa_fee(amount: Any) -> _D:
	"""M-Pesa P2P fee schedule (Safaricom Kenya, 2025)."""
	bands = [
		(_D("100"),    _D("0")),
		(_D("500"),    _D("7")),
		(_D("1000"),   _D("13")),
		(_D("1500"),   _D("23")),
		(_D("2500"),   _D("33")),
		(_D("3500"),   _D("53")),
		(_D("5000"),   _D("57")),
		(_D("7500"),   _D("78")),
		(_D("10000"),  _D("90")),
		(_D("15000"),  _D("100")),
		(_D("20000"),  _D("105")),
		(_D("35000"),  _D("108")),
		(_D("50000"),  _D("108")),
		(_D("250000"), _D("108")),
		(_D("500000"), _D("108")),
	]
	val = _D(str(amount))
	for upper, fee in bands:
		if val <= upper:
			return fee
	return _D("108")


def calculate_vat_ke(amount: Any, vat_rate: Any = "0.16") -> _D:
	"""Kenya VAT at standard rate (default 16%)."""
	from decimal import ROUND_HALF_UP
	return (_D(str(amount)) * _D(str(vat_rate))).quantize(_D("0.01"), rounding=ROUND_HALF_UP)


def calculate_excise_ke(fee_amount: Any, excise_rate: Any = "0.20") -> _D:
	"""Kenya excise duty on financial services fees (Finance Act 2022: 20%)."""
	from decimal import ROUND_HALF_UP
	return (_D(str(fee_amount)) * _D(str(excise_rate))).quantize(_D("0.01"), rounding=ROUND_HALF_UP)


def calculate_total_charge(principal: Any, fee: Any, vat_on_fee: Any = "0", excise: Any = "0") -> _D:
	"""Total customer charge = principal + fee + VAT on fee + excise."""
	return _D(str(principal)) + _D(str(fee)) + _D(str(vat_on_fee)) + _D(str(excise))


def calculate_fx_amount(
	amount: Any,
	rate: Any,
	spread_bps: int = 0,
	direction: str = "buy",
) -> _D:
	"""Convert amount at rate with optional spread.

	Args:
		amount: Source amount.
		rate: Mid exchange rate (1 source unit = rate target units).
		spread_bps: Spread in basis points applied to mid-rate.
		direction: "buy" (customer buys target, gets worse rate) or "sell".

	Returns:
		Converted amount as Decimal, rounded to 2dp.
	"""
	from decimal import ROUND_HALF_UP
	mid = _D(str(rate))
	half_spread = _D(str(spread_bps)) / _D("20000")
	if direction == "buy":
		effective = mid * (1 - half_spread)
	else:
		effective = mid * (1 + half_spread)
	return (_D(str(amount)) * effective).quantize(_D("0.01"), rounding=ROUND_HALF_UP)


def calculate_settlement_net(
	gross_amount: Any,
	processing_fee_rate_bps: int = 200,
) -> _D:
	"""Net settlement after processing fee.

	Args:
		gross_amount: Total gross settlement amount.
		processing_fee_rate_bps: Processing fee in basis points (default 200 = 2%).
	"""
	from decimal import ROUND_HALF_UP
	gross = _D(str(gross_amount))
	fee = (gross * _D(str(processing_fee_rate_bps)) / _D("10000")).quantize(
		_D("0.01"), rounding=ROUND_HALF_UP
	)
	return (gross - fee).quantize(_D("0.01"), rounding=ROUND_HALF_UP)


def calculate_late_settlement_penalty(
	outstanding: Any,
	days_overdue: int,
	daily_rate: Any = "0.001",
) -> _D:
	"""Late settlement penalty = outstanding × daily_rate × days_overdue."""
	from decimal import ROUND_HALF_UP
	return (
		_D(str(outstanding)) * _D(str(daily_rate)) * _D(str(days_overdue))
	).quantize(_D("0.01"), rounding=ROUND_HALF_UP)
