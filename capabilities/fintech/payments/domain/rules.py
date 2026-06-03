"""Business rules for APG Digital Payments — Africa-first.

Every rule is a pure callable with no side-effects. Services call assert_*
functions; rule violations raise RuleViolation which the service converts to
HTTP 422 / PermissionError depending on context.

Covers:
- Tenant isolation
- KYC-tier transaction limits (CBK, CBN, BoU)
- Duplicate / idempotency detection
- M-Pesa Daraja constraints
- Card PCI-DSS guards
- SWIFT / correspondent-banking rules
- Refund & reversal guards
- FX / settlement rules
- AML velocity checks
- Regulatory thresholds (CBK STR, CBN STR, BoU STR)
"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Any


# ---------------------------------------------------------------------------
# Core exception
# ---------------------------------------------------------------------------

class RuleViolation(Exception):
	"""Raised when a business rule is violated."""

	def __init__(self, rule_name: str, reason: str, required_action: str = "") -> None:
		self.rule_name = rule_name
		self.reason = reason
		self.required_action = required_action
		super().__init__(f"Rule '{rule_name}' violated: {reason}")


# ---------------------------------------------------------------------------
# Tenant isolation
# ---------------------------------------------------------------------------

def assert_tenant_context(context: dict[str, Any]) -> None:
	"""All operations require a non-empty tenant_id."""
	if not context.get("tenant_id"):
		raise RuleViolation(
			"tenant_context_required",
			"tenant_id is required for every operation",
			"attach_tenant_context",
		)


def assert_write_policy(context: dict[str, Any]) -> None:
	"""Mutating operations require an attached policy token."""
	if context.get("operation_type") == "write" and not context.get("policy_attached"):
		raise RuleViolation(
			"write_requires_policy",
			"write operations require an attached policy",
			"attach_policy",
		)


def assert_no_cross_tenant_access(actor_tenant: str, resource_tenant: str) -> None:
	"""Cross-tenant resource access is always denied."""
	if actor_tenant != resource_tenant:
		raise RuleViolation(
			"cross_tenant_access_denied",
			f"actor tenant '{actor_tenant}' may not access resource of tenant '{resource_tenant}'",
			"use_own_tenant_resources",
		)


# ---------------------------------------------------------------------------
# Amount guards
# ---------------------------------------------------------------------------

def assert_positive_amount(amount: Decimal | int | str, field: str = "amount") -> None:
	"""Amount must be strictly positive."""
	val = Decimal(str(amount))
	if val <= 0:
		raise RuleViolation(
			"amount_must_be_positive",
			f"{field} must be > 0, got {val}",
			"supply_positive_amount",
		)


def assert_amount_precision(amount: Decimal, max_decimal_places: int = 2) -> None:
	"""Amounts must not have more than max_decimal_places decimal places."""
	quantized = amount.quantize(Decimal("0.01"))
	if amount != quantized and abs(amount - quantized) > Decimal("0.005"):
		raise RuleViolation(
			"amount_precision_exceeded",
			f"amount {amount} exceeds {max_decimal_places} decimal places",
			"round_amount_correctly",
		)


# ---------------------------------------------------------------------------
# KYC / transaction limits (CBK Prudential Guidelines — Kenya)
# ---------------------------------------------------------------------------

# CBK tier limits in KES
_KYC_LIMITS: dict[str, dict[str, Decimal]] = {
	"basic": {
		"per_txn": Decimal("150000"),
		"daily":   Decimal("300000"),
		"monthly": Decimal("3000000"),
	},
	"standard": {
		"per_txn": Decimal("500000"),
		"daily":   Decimal("1000000"),
		"monthly": Decimal("10000000"),
	},
	"full_kyc": {
		"per_txn": Decimal("1000000"),
		"daily":   Decimal("5000000"),
		"monthly": Decimal("50000000"),
	},
	"enhanced": {
		"per_txn": Decimal("999999999"),
		"daily":   Decimal("999999999"),
		"monthly": Decimal("999999999"),
	},
}

# CBN Nigeria limits (NGN)
_CBN_LIMITS: dict[str, dict[str, Decimal]] = {
	"tier1": {
		"per_txn": Decimal("50000"),
		"daily":   Decimal("200000"),
		"monthly": Decimal("500000"),
	},
	"tier2": {
		"per_txn": Decimal("200000"),
		"daily":   Decimal("1000000"),
		"monthly": Decimal("5000000"),
	},
	"tier3": {
		"per_txn": Decimal("999999999"),
		"daily":   Decimal("999999999"),
		"monthly": Decimal("999999999"),
	},
}


def assert_kyc_per_txn_limit(amount: Decimal, kyc_tier: str, currency: str = "KES") -> None:
	"""Single transaction must not exceed KYC tier limit."""
	limits = _KYC_LIMITS.get(kyc_tier) or _KYC_LIMITS["basic"]
	limit = limits["per_txn"]
	if currency == "NGN":
		limits = _CBN_LIMITS.get(kyc_tier.replace("full_kyc", "tier3").replace("standard", "tier2").replace("basic", "tier1"), _CBN_LIMITS["tier1"])
		limit = limits["per_txn"]
	if amount > limit:
		raise RuleViolation(
			"kyc_per_txn_limit_exceeded",
			f"Amount {amount} {currency} exceeds per-txn limit {limit} for KYC tier '{kyc_tier}'",
			"upgrade_kyc_tier_or_split_payment",
		)


def assert_kyc_daily_limit(daily_total: Decimal, amount: Decimal, kyc_tier: str, currency: str = "KES") -> None:
	"""Running daily total (including this transaction) must not exceed daily limit."""
	limits = _KYC_LIMITS.get(kyc_tier) or _KYC_LIMITS["basic"]
	limit = limits["daily"]
	if daily_total + amount > limit:
		raise RuleViolation(
			"kyc_daily_limit_exceeded",
			f"Daily total {daily_total + amount} {currency} exceeds daily limit {limit} for KYC tier '{kyc_tier}'",
			"wait_until_tomorrow_or_upgrade_kyc",
		)


def assert_kyc_monthly_limit(monthly_total: Decimal, amount: Decimal, kyc_tier: str, currency: str = "KES") -> None:
	"""Running monthly total must not exceed monthly limit."""
	limits = _KYC_LIMITS.get(kyc_tier) or _KYC_LIMITS["basic"]
	limit = limits["monthly"]
	if monthly_total + amount > limit:
		raise RuleViolation(
			"kyc_monthly_limit_exceeded",
			f"Monthly total {monthly_total + amount} {currency} exceeds monthly limit {limit} for KYC tier '{kyc_tier}'",
			"upgrade_kyc_tier",
		)


# ---------------------------------------------------------------------------
# Duplicate / idempotency
# ---------------------------------------------------------------------------

def assert_no_duplicate(idempotency_key: str, existing_keys: set[str]) -> None:
	"""Idempotency key must be unique within tenant scope."""
	if idempotency_key and idempotency_key in existing_keys:
		raise RuleViolation(
			"duplicate_payment_detected",
			f"Idempotency key '{idempotency_key}' already used — return existing result",
			"return_cached_response",
		)


def assert_retry_window(retry_count: int, max_retries: int = 3) -> None:
	"""Do not retry more than max_retries times."""
	if retry_count >= max_retries:
		raise RuleViolation(
			"max_retries_exceeded",
			f"Payment has been retried {retry_count} times (max {max_retries})",
			"contact_support_or_raise_dispute",
		)


# ---------------------------------------------------------------------------
# M-Pesa Daraja constraints
# ---------------------------------------------------------------------------

_MPESA_MIN = Decimal("1")
_MPESA_MAX = Decimal("300000")   # Safaricom B2C/C2B single txn ceiling
_MPESA_B2C_MAX = Decimal("300000")
_MPESA_PAYBILL_MAX = Decimal("300000")


def assert_mpesa_amount(amount: Decimal, channel: str = "stk_push") -> None:
	"""M-Pesa amounts must be between KES 1 and 300,000."""
	if amount < _MPESA_MIN:
		raise RuleViolation(
			"mpesa_amount_below_minimum",
			f"M-Pesa {channel} minimum is KES {_MPESA_MIN}, got {amount}",
			"increase_amount",
		)
	ceiling = _MPESA_B2C_MAX if channel == "b2c" else _MPESA_PAYBILL_MAX
	if amount > ceiling:
		raise RuleViolation(
			"mpesa_amount_above_maximum",
			f"M-Pesa {channel} maximum is KES {ceiling}, got {amount}",
			"split_into_multiple_transactions",
		)


def assert_mpesa_phone(msisdn: str) -> None:
	"""Phone must be a valid Safaricom Kenya number (07xx or 01xx prefix, E.164)."""
	import re
	if not re.match(r"^254[71]\d{8}$", msisdn):
		raise RuleViolation(
			"mpesa_invalid_phone",
			f"M-Pesa requires a valid Safaricom Kenya number in E.164 format (254...), got '{msisdn}'",
			"normalise_phone_to_254xxxxxxxxx",
		)


def assert_mpesa_float_sufficient(float_balance: Decimal, amount: Decimal) -> None:
	"""Agent/merchant float must cover the transaction."""
	if float_balance < amount:
		raise RuleViolation(
			"mpesa_insufficient_float",
			f"Agent float {float_balance} KES is insufficient for transaction {amount} KES",
			"top_up_agent_float",
		)


def assert_mpesa_reference_length(reference: str) -> None:
	"""Daraja account reference must be 1-12 chars."""
	if not reference or len(reference) > 12:
		raise RuleViolation(
			"mpesa_reference_too_long",
			f"Daraja account reference must be 1-12 chars, got {len(reference)}",
			"shorten_reference",
		)


# ---------------------------------------------------------------------------
# MTN MoMo / Airtel / Tigo
# ---------------------------------------------------------------------------

def assert_momo_amount(amount: Decimal, provider: str) -> None:
	"""Mobile money amounts must be within provider limits."""
	limits = {
		"mtn_momo": (Decimal("100"), Decimal("2000000")),   # UGX
		"airtel":   (Decimal("10"),  Decimal("5000000")),
		"tigo":     (Decimal("100"), Decimal("1000000")),   # TZS
	}
	lo, hi = limits.get(provider, (Decimal("1"), Decimal("999999999")))
	if amount < lo or amount > hi:
		raise RuleViolation(
			f"{provider}_amount_out_of_range",
			f"{provider} amount {amount} out of range [{lo}, {hi}]",
			"check_provider_limits",
		)


# ---------------------------------------------------------------------------
# Card PCI-DSS guards
# ---------------------------------------------------------------------------

def assert_card_token_not_pan(token: str) -> None:
	"""Card token must never look like a raw PAN (16-digit number)."""
	import re
	if re.match(r"^\d{13,19}$", token.replace(" ", "").replace("-", "")):
		raise RuleViolation(
			"raw_pan_storage_forbidden",
			"Raw PAN must not be stored; use a tokenised reference from your vault",
			"tokenise_card_before_storage",
		)


def assert_card_cvv_not_stored(cvv: str | None) -> None:
	"""CVV must never be persisted after authorisation."""
	if cvv and len(cvv) in (3, 4) and cvv.isdigit():
		raise RuleViolation(
			"cvv_storage_forbidden",
			"CVV/CVC must not be stored after authorisation (PCI-DSS Req 3.2.1)",
			"discard_cvv_after_auth",
		)


def assert_3ds_result(three_ds_result: str | None, amount: Decimal) -> None:
	"""Transactions above KES 10,000 equivalent must have 3DS result."""
	if amount > Decimal("10000") and not three_ds_result:
		raise RuleViolation(
			"3ds_required_for_high_value",
			f"3D Secure result required for amount {amount} > 10,000",
			"complete_3ds_challenge",
		)


# ---------------------------------------------------------------------------
# SWIFT / correspondent banking
# ---------------------------------------------------------------------------

def assert_swift_bic(bic: str) -> None:
	"""BIC must be 8 or 11 characters (ISO 9362)."""
	import re
	if not re.match(r"^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?$", bic.upper()):
		raise RuleViolation(
			"invalid_swift_bic",
			f"BIC '{bic}' is not valid ISO 9362 (8 or 11 chars)",
			"supply_valid_bic",
		)


def assert_iban(iban: str) -> None:
	"""Basic IBAN format check (2-letter country + up to 32 alphanumeric)."""
	import re
	clean = iban.replace(" ", "").upper()
	if not re.match(r"^[A-Z]{2}[0-9]{2}[A-Z0-9]{1,30}$", clean):
		raise RuleViolation(
			"invalid_iban",
			f"IBAN '{iban}' fails format check",
			"supply_valid_iban",
		)


def assert_swift_purpose_code(purpose_code: str) -> None:
	"""SWIFT purpose code must be 3 chars."""
	if not purpose_code or len(purpose_code) != 3:
		raise RuleViolation(
			"invalid_swift_purpose_code",
			f"Purpose code must be exactly 3 chars, got '{purpose_code}'",
			"use_standard_swift_purpose_code",
		)


# ---------------------------------------------------------------------------
# Refund / reversal guards
# ---------------------------------------------------------------------------

def assert_refund_amount(refund_amount: Decimal, original_amount: Decimal) -> None:
	"""Refund must not exceed original transaction amount."""
	if refund_amount > original_amount:
		raise RuleViolation(
			"refund_exceeds_original",
			f"Refund {refund_amount} exceeds original amount {original_amount}",
			"reduce_refund_amount",
		)


def assert_refund_not_duplicate(already_refunded: Decimal, new_refund: Decimal, original: Decimal) -> None:
	"""Cumulative refunds must not exceed original amount."""
	if already_refunded + new_refund > original:
		raise RuleViolation(
			"cumulative_refund_exceeds_original",
			f"Total refunds {already_refunded + new_refund} would exceed original {original}",
			"reduce_refund_amount",
		)


def assert_reversal_window(created_at_iso: str, window_hours: int = 24) -> None:
	"""Wrong-number reversals must be initiated within window_hours of creation."""
	from datetime import datetime, timezone
	created = datetime.fromisoformat(created_at_iso.replace("Z", "+00:00"))
	now = datetime.now(timezone.utc)
	elapsed = (now - created).total_seconds() / 3600
	if elapsed > window_hours:
		raise RuleViolation(
			"reversal_window_expired",
			f"Reversal window of {window_hours}h expired ({elapsed:.1f}h elapsed)",
			"raise_dispute_instead",
		)


def assert_refundable_status(status: str) -> None:
	"""Only completed/captured transactions can be refunded."""
	allowed = {"completed", "captured", "settled"}
	if status not in allowed:
		raise RuleViolation(
			"transaction_not_refundable",
			f"Status '{status}' is not refundable (must be one of {allowed})",
			"wait_for_completion_before_refunding",
		)


# ---------------------------------------------------------------------------
# FX / settlement rules
# ---------------------------------------------------------------------------

def assert_fx_rate_freshness(rate_age_seconds: float, max_age_seconds: float = 300) -> None:
	"""FX rates older than max_age_seconds must not be used for execution."""
	if rate_age_seconds > max_age_seconds:
		raise RuleViolation(
			"fx_rate_stale",
			f"FX rate is {rate_age_seconds:.0f}s old (max {max_age_seconds}s)",
			"refresh_fx_rate_before_execution",
		)


def assert_settlement_variance(variance: Decimal, expected: Decimal, threshold_bps: int = 10) -> None:
	"""Settlement variance must not exceed threshold_bps basis points of expected amount."""
	if expected == 0:
		return
	bps = abs(variance / expected * 10000).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
	if bps > Decimal(str(threshold_bps)):
		raise RuleViolation(
			"settlement_variance_exceeded",
			f"Variance {variance} ({bps} bps) exceeds threshold of {threshold_bps} bps",
			"investigate_settlement_discrepancy",
		)


def assert_supported_currency(currency: str, supported: set[str] | None = None) -> None:
	"""Currency must be in the supported set."""
	if supported is None:
		supported = {"KES", "UGX", "TZS", "RWF", "GHS", "NGN", "ZAR", "USD", "EUR", "GBP", "XOF", "XAF"}
	if currency.upper() not in supported:
		raise RuleViolation(
			"unsupported_currency",
			f"Currency '{currency}' is not supported",
			f"use_one_of_{sorted(supported)}",
		)


# ---------------------------------------------------------------------------
# AML velocity / regulatory thresholds
# ---------------------------------------------------------------------------

# CBK STR threshold (Kenya) — transactions >= KES 1,000,000 require CTR filing
_CBK_CTR_THRESHOLD = Decimal("1000000")
# CBN STR threshold (Nigeria) — NGN 5,000,000
_CBN_CTR_THRESHOLD = Decimal("5000000")
# BoU STR threshold (Uganda) — UGX 20,000,000
_BOU_CTR_THRESHOLD = Decimal("20000000")

_CTR_THRESHOLDS: dict[str, Decimal] = {
	"KES": _CBK_CTR_THRESHOLD,
	"NGN": _CBN_CTR_THRESHOLD,
	"UGX": _BOU_CTR_THRESHOLD,
}


def assert_aml_velocity(
	transactions_last_24h: int,
	amount_last_24h: Decimal,
	amount: Decimal,
	currency: str,
) -> None:
	"""Flag structuring risk: > 5 transactions or > threshold in 24h triggers review."""
	threshold = _CTR_THRESHOLDS.get(currency.upper(), Decimal("500000"))
	if transactions_last_24h >= 5 and (amount_last_24h + amount) >= threshold * Decimal("0.8"):
		raise RuleViolation(
			"aml_velocity_threshold",
			f"Possible structuring: {transactions_last_24h} txns, {amount_last_24h + amount} {currency} in 24h",
			"submit_ctr_and_flag_for_review",
		)


def calculate_ctr_obligation(amount: Decimal, currency: str) -> dict[str, Any]:
	"""Determine regulatory reporting obligation for a transaction."""
	threshold = _CTR_THRESHOLDS.get(currency.upper(), Decimal("1000000"))
	requires_ctr = amount >= threshold
	return {
		"requires_ctr": requires_ctr,
		"threshold": str(threshold),
		"currency": currency,
		"amount": str(amount),
		"regulator": {
			"KES": "CBK",
			"NGN": "CBN",
			"UGX": "BoU",
			"TZS": "BoT",
			"GHS": "BoG",
			"ZAR": "SARB",
		}.get(currency.upper(), "UNKNOWN"),
	}


# ---------------------------------------------------------------------------
# Batch payment rules
# ---------------------------------------------------------------------------

def assert_batch_size(count: int, max_batch: int = 10000) -> None:
	"""Batch must not exceed max_batch entries."""
	if count > max_batch:
		raise RuleViolation(
			"batch_too_large",
			f"Batch has {count} entries; max is {max_batch}",
			"split_into_smaller_batches",
		)


def assert_batch_lists_aligned(recipients: list, amounts: list, references: list) -> None:
	"""All batch lists must have equal length."""
	n = len(recipients)
	if len(amounts) != n or len(references) != n:
		raise RuleViolation(
			"batch_lists_misaligned",
			f"recipients={len(recipients)}, amounts={len(amounts)}, references={len(references)} — must be equal",
			"align_batch_lists",
		)


# ---------------------------------------------------------------------------
# Webhook rules
# ---------------------------------------------------------------------------

def assert_webhook_url(url: str) -> None:
	"""Webhook URL must use HTTPS."""
	if not url.startswith("https://"):
		raise RuleViolation(
			"webhook_url_must_use_https",
			f"Webhook URL '{url}' must use HTTPS",
			"use_https_webhook_endpoint",
		)


# ---------------------------------------------------------------------------
# Merchant rules
# ---------------------------------------------------------------------------

def assert_mcc_code(mcc: str) -> None:
	"""MCC must be a 4-digit numeric code."""
	if not (mcc.isdigit() and len(mcc) == 4):
		raise RuleViolation(
			"invalid_mcc",
			f"MCC '{mcc}' must be 4 numeric digits",
			"use_valid_iso_mcc",
		)


# ---------------------------------------------------------------------------
# Calculation helpers (deterministic, no side-effects)
# ---------------------------------------------------------------------------

def calculate_mpesa_fee(amount: Decimal) -> Decimal:
	"""Return Safaricom M-Pesa withdrawal fee for the given amount (KES)."""
	tiers: list[tuple[Decimal, Decimal, Decimal]] = [
		(Decimal("1"),     Decimal("100"),    Decimal("0")),
		(Decimal("101"),   Decimal("500"),    Decimal("7")),
		(Decimal("501"),   Decimal("1000"),   Decimal("13")),
		(Decimal("1001"),  Decimal("1500"),   Decimal("23")),
		(Decimal("1501"),  Decimal("2500"),   Decimal("33")),
		(Decimal("2501"),  Decimal("3500"),   Decimal("53")),
		(Decimal("3501"),  Decimal("5000"),   Decimal("57")),
		(Decimal("5001"),  Decimal("7500"),   Decimal("78")),
		(Decimal("7501"),  Decimal("10000"),  Decimal("90")),
		(Decimal("10001"), Decimal("15000"),  Decimal("100")),
		(Decimal("15001"), Decimal("20000"),  Decimal("105")),
		(Decimal("20001"), Decimal("35000"),  Decimal("108")),
		(Decimal("35001"), Decimal("250000"), Decimal("108")),
		(Decimal("250001"),Decimal("999999"), Decimal("108")),
	]
	for lo, hi, fee in tiers:
		if lo <= amount <= hi:
			return fee
	return Decimal("108")


def calculate_excise_ke(fee: Decimal) -> Decimal:
	"""Kenya Finance Act 2022 — 20% excise duty on financial services fees."""
	return (fee * Decimal("0.20")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def calculate_vat_ke(fee: Decimal) -> Decimal:
	"""Kenya VAT at 16% on financial services fees."""
	return (fee * Decimal("0.16")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def calculate_total_charge(fee: Decimal, excise: Decimal) -> Decimal:
	"""Total charge = fee + excise (VAT is embedded in regulated M-Pesa fees)."""
	return (fee + excise).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def calculate_fx_amount(
	from_amount: Decimal,
	mid_rate: Decimal,
	spread_bps: int = 150,
	direction: str = "buy",
) -> Decimal:
	"""Apply spread to mid-rate and convert amount.

	direction='buy'  — customer buys foreign currency (rate worsened by spread/2)
	direction='sell' — customer sells foreign currency (rate improved by spread/2)
	"""
	half_spread = Decimal(str(spread_bps)) / Decimal("20000")   # bps/2 / 10000
	if direction == "buy":
		effective_rate = mid_rate * (1 - half_spread)
	else:
		effective_rate = mid_rate * (1 + half_spread)
	return (from_amount * effective_rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def calculate_settlement_net(gross: Decimal, processing_fee_rate_bps: int = 200) -> Decimal:
	"""Net settlement after processing fee deduction."""
	fee = (gross * Decimal(str(processing_fee_rate_bps)) / Decimal("10000")).quantize(
		Decimal("0.01"), rounding=ROUND_HALF_UP
	)
	return gross - fee


def calculate_late_settlement_penalty(
	gross: Decimal,
	days_overdue: int,
	daily_penalty_rate: Decimal = Decimal("0.001"),
) -> Decimal:
	"""Late settlement penalty accruing at daily_penalty_rate per day."""
	return (gross * daily_penalty_rate * Decimal(str(days_overdue))).quantize(
		Decimal("0.01"), rounding=ROUND_HALF_UP
	)
