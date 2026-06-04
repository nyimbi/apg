"""Deterministic domain rules for Digital Payments.

These rules are evaluated by the capability rule engine and are the single
source of truth for all governance decisions within this capability.
"""
from __future__ import annotations
from typing import Any


class RuleViolation(Exception):
    """Raised when a business rule is violated."""
    def __init__(self, rule_name: str, reason: str, required_action: str = "") -> None:
        self.rule_name = rule_name
        self.reason = reason
        self.required_action = required_action
        super().__init__(f"Rule '{rule_name}' violated: {reason}")


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
        raise RuleViolation("cross_tenant_access_denied", "cross-tenant access is not permitted", "use_own_tenant_resources")


def assert_positive_amount(amount: Any, field: str = "amount") -> None:
	"""Amount must be a positive number."""
	from decimal import Decimal
	try:
		val = Decimal(str(amount))
	except Exception:
		raise RuleViolation("invalid_amount", f"{field} must be a valid number", "provide_valid_amount")
	if val <= 0:
		raise RuleViolation("non_positive_amount", f"{field} must be greater than zero", "provide_positive_amount")


def assert_amount_precision(amount: Any, max_decimal_places: int = 2, field: str = "amount") -> None:
	"""Amount must not exceed the specified decimal precision."""
	from decimal import Decimal
	try:
		val = Decimal(str(amount))
		sign, digits, exponent = val.as_tuple()
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
		raise RuleViolation("invalid_amount_precision", f"{field} must be a valid decimal number", "provide_valid_amount")


def assert_supported_currency(currency: str, supported_currencies: list[str] | None = None) -> None:
	"""Currency must be in the supported set."""
	defaults = ["KES", "UGX", "TZS", "RWF", "GHS", "NGN", "ZAR", "USD", "EUR", "GBP"]
	allowed = supported_currencies or defaults
	if currency not in allowed:
		raise RuleViolation("unsupported_currency", f"currency {currency!r} is not supported", "use_supported_currency")


def assert_valid_phone(phone: str, country_prefix: str = "254") -> None:
	"""Mobile money phone number must be valid."""
	digits = "".join(c for c in phone if c.isdigit())
	if len(digits) < 9 or len(digits) > 15:
		raise RuleViolation("invalid_phone", "phone number must be 9-15 digits", "provide_valid_phone_number")


# ─────────────────────────────────────────────────────────────
# Payment-channel-specific rule functions
# ─────────────────────────────────────────────────────────────
from decimal import Decimal as _D
import re as _re

def assert_kyc_per_txn_limit(amount: Any, tier: str = "basic") -> None:
	limits = {"basic": _D("300000"), "standard": _D("1000000"), "full_kyc": _D("5000000"), "enhanced": _D("999999999")}
	limit = limits.get(tier, _D("300000"))
	if _D(str(amount)) > limit:
		raise RuleViolation("kyc_per_txn_limit", f"amount exceeds per-transaction limit of {limit} for tier {tier!r}", "upgrade_kyc_tier")

def assert_kyc_daily_limit(daily_used: Any, new_amount: Any, tier: str = "basic") -> None:
	limits = {"basic": _D("300000"), "standard": _D("1000000"), "full_kyc": _D("5000000"), "enhanced": _D("999999999")}
	limit = limits.get(tier, _D("300000"))
	if _D(str(daily_used)) + _D(str(new_amount)) > limit:
		raise RuleViolation("kyc_daily_limit", f"daily limit of {limit} exceeded for tier {tier!r}", "wait_for_next_day")

def assert_kyc_monthly_limit(monthly_used: Any, new_amount: Any, tier: str = "basic") -> None:
	limits = {"basic": _D("3000000"), "standard": _D("10000000"), "full_kyc": _D("50000000"), "enhanced": _D("999999999")}
	limit = limits.get(tier, _D("3000000"))
	if _D(str(monthly_used)) + _D(str(new_amount)) > limit:
		raise RuleViolation("kyc_monthly_limit", f"monthly limit of {limit} exceeded for tier {tier!r}", "upgrade_kyc_tier")

def assert_no_duplicate(reference: str, existing_refs: list[str], window_desc: str = "5 minutes") -> None:
	if reference in existing_refs:
		raise RuleViolation("duplicate_payment", f"payment with reference {reference!r} already exists within {window_desc}", "use_unique_reference")

def assert_retry_window(original_created_at: str, max_minutes: int = 60) -> None:
	from datetime import datetime, timezone
	try:
		created = datetime.fromisoformat(original_created_at.replace("Z", "+00:00"))
		now = datetime.now(timezone.utc)
		if (now - created).total_seconds() > max_minutes * 60:
			raise RuleViolation("retry_window_expired", f"retry window of {max_minutes} minutes has expired", "create_new_payment")
	except (ValueError, AttributeError):
		pass

def assert_mpesa_amount(amount: Any) -> None:
	val = _D(str(amount))
	if val < _D("1") or val > _D("500000"):
		raise RuleViolation("mpesa_amount_range", "M-Pesa amount must be between KES 1 and KES 500,000", "adjust_amount")

def assert_mpesa_phone(phone: str) -> None:
	digits = "".join(c for c in phone if c.isdigit())
	if not (digits.startswith("254") or digits.startswith("07") or digits.startswith("01")):
		raise RuleViolation("invalid_mpesa_phone", "M-Pesa phone must be a valid Kenyan number (07XX, 01XX, 254XX)", "provide_valid_kenyan_phone")

def assert_mpesa_float_sufficient(float_balance: Any, amount: Any) -> None:
	if _D(str(float_balance)) < _D(str(amount)):
		raise RuleViolation("insufficient_float", "M-Pesa agent float is insufficient", "top_up_float")

def assert_mpesa_reference_length(reference: str) -> None:
	if not (1 <= len(reference) <= 20):
		raise RuleViolation("invalid_mpesa_reference", "M-Pesa account reference must be 1-20 characters", "shorten_reference")

def assert_momo_amount(amount: Any, currency: str = "KES") -> None:
	limits = {"KES": (_D("1"), _D("500000")), "UGX": (_D("500"), _D("5000000")),
	          "GHS": (_D("1"), _D("5000")), "RWF": (_D("100"), _D("1000000"))}
	lo, hi = limits.get(currency, (_D("1"), _D("1000000")))
	val = _D(str(amount))
	if not (lo <= val <= hi):
		raise RuleViolation("momo_amount_range", f"Mobile money amount must be between {lo} and {hi} {currency}", "adjust_amount")

def assert_card_token_not_pan(value: str) -> None:
	cleaned = "".join(c for c in value if c.isdigit())
	if len(cleaned) in (13, 14, 15, 16, 19) and cleaned.isdigit():
		raise RuleViolation("raw_pan_not_allowed", "raw card PAN must not be stored — use a token", "tokenise_card")

def assert_card_cvv_not_stored(has_stored_cvv: bool) -> None:
	if has_stored_cvv:
		raise RuleViolation("cvv_storage_prohibited", "CVV must never be stored after authorisation (PCI-DSS)", "remove_stored_cvv")

def assert_3ds_result(eci: str, status: str = "Y") -> None:
	if eci not in ("05", "06", "07") and status not in ("Y", "A"):
		raise RuleViolation("3ds_authentication_failed", f"3DS authentication result {eci!r}/{status!r} not accepted", "request_customer_authentication")

def assert_swift_bic(bic: str) -> None:
	if not _re.match(r"^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?$", bic.upper()):
		raise RuleViolation("invalid_bic", f"BIC/SWIFT code {bic!r} is not valid", "provide_valid_bic")

def assert_iban(iban: str) -> None:
	stripped = iban.replace(" ", "").upper()
	if not (15 <= len(stripped) <= 34) or not _re.match(r"^[A-Z]{2}[0-9]{2}[A-Z0-9]+$", stripped):
		raise RuleViolation("invalid_iban", f"IBAN {iban!r} format is invalid", "provide_valid_iban")

def assert_swift_purpose_code(purpose_code: str) -> None:
	valid = {"SALA", "SUPP", "TRAD", "DIVI", "CHAR", "LOAR", "INTC", "CMDT"}
	if purpose_code.upper() not in valid:
		raise RuleViolation("invalid_purpose_code", f"SWIFT purpose code {purpose_code!r} not recognised", "use_valid_purpose_code")

def assert_refund_amount(refund_amount: Any, original_amount: Any) -> None:
	if _D(str(refund_amount)) > _D(str(original_amount)):
		raise RuleViolation("refund_exceeds_original", "refund amount cannot exceed original transaction amount", "reduce_refund_amount")

def assert_refund_not_duplicate(refund_ref: str, existing_refund_refs: list[str]) -> None:
	if refund_ref in existing_refund_refs:
		raise RuleViolation("duplicate_refund", f"refund {refund_ref!r} has already been processed", "verify_refund_status")

def assert_reversal_window(original_created_at: str, max_hours: int = 24) -> None:
	from datetime import datetime, timezone
	try:
		created = datetime.fromisoformat(original_created_at.replace("Z", "+00:00"))
		age_h = (datetime.now(timezone.utc) - created).total_seconds() / 3600
		if age_h > max_hours:
			raise RuleViolation("reversal_window_expired", f"reversal must be within {max_hours}h — transaction is {age_h:.1f}h old", "contact_support_for_late_reversal")
	except RuleViolation:
		raise
	except Exception:
		pass

def assert_refundable_status(status: str) -> None:
	if status not in ("completed", "settled"):
		raise RuleViolation("not_refundable", f"transaction in status {status!r} cannot be refunded", "wait_for_settlement")

def assert_fx_rate_freshness(rate_age_seconds: int, max_age: int = 300) -> None:
	if rate_age_seconds > max_age:
		raise RuleViolation("stale_fx_rate", f"FX rate is {rate_age_seconds}s old — must be refreshed every {max_age}s", "refresh_fx_rate")

def assert_settlement_variance(expected: Any, actual: Any, tolerance_pct: float = 0.01) -> None:
	e, a = _D(str(expected)), _D(str(actual))
	if e > 0 and abs(a - e) / e > _D(str(tolerance_pct)):
		raise RuleViolation("settlement_variance", f"settlement variance {abs(a-e)/e:.2%} exceeds tolerance {tolerance_pct:.0%}", "investigate_settlement_break")

def assert_webhook_url(url: str) -> None:
	if not url.startswith("https://"):
		raise RuleViolation("insecure_webhook_url", "webhook URL must use HTTPS", "use_https_webhook_url")

def assert_mcc_code(mcc: str) -> None:
	if not (_re.match(r"^\d{4}$", str(mcc))):
		raise RuleViolation("invalid_mcc", f"merchant category code {mcc!r} must be a 4-digit number", "provide_valid_mcc")

def assert_aml_velocity(count_in_window: int, max_count: int = 10) -> None:
	if count_in_window >= max_count:
		raise RuleViolation("aml_velocity_breach", f"{count_in_window} transactions in window exceeds AML velocity limit of {max_count}", "hold_for_aml_review")

def assert_batch_size(batch_size: int, max_size: int = 1000) -> None:
	if batch_size > max_size:
		raise RuleViolation("batch_too_large", f"batch of {batch_size} exceeds maximum {max_size}", "split_into_smaller_batches")

def assert_batch_lists_aligned(*lists: list) -> None:
	lengths = [len(lst) for lst in lists]
	if len(set(lengths)) > 1:
		raise RuleViolation("batch_lists_misaligned", f"batch lists have different lengths: {lengths}", "align_batch_list_lengths")


# ─────────────────────────────────────────────────────────────
# Payment calculation utilities
# ─────────────────────────────────────────────────────────────

def calculate_mpesa_fee(amount: Any) -> _D:
	"""Kenyan M-Pesa P2P fee schedule (Safaricom 2024)."""
	bands = [
		(_D("100"), _D("0")),
		(_D("500"), _D("7")),
		(_D("1000"), _D("13")),
		(_D("1500"), _D("23")),
		(_D("2500"), _D("33")),
		(_D("3500"), _D("53")),
		(_D("5000"), _D("57")),
		(_D("7500"), _D("78")),
		(_D("10000"), _D("90")),
		(_D("15000"), _D("100")),
		(_D("20000"), _D("105")),
		(_D("35000"), _D("108")),
		(_D("50000"), _D("108")),
		(_D("250000"), _D("108")),
		(_D("500000"), _D("108")),
	]
	val = _D(str(amount))
	for upper, fee in bands:
		if val <= upper:
			return fee
	return _D("108")


def calculate_vat_ke(amount: Any, vat_rate: Any = "0.16") -> _D:
	"""Kenyan VAT at standard rate (default 16%)."""
	return (_D(str(amount)) * _D(str(vat_rate))).quantize(_D("0.01"))


def calculate_excise_ke(fee_amount: Any, excise_rate: Any = "0.20") -> _D:
	"""Kenyan excise duty on financial services fees (Finance Act 2022: 20%)."""
	return (_D(str(fee_amount)) * _D(str(excise_rate))).quantize(_D("0.01"))


def calculate_total_charge(principal: Any, fee: Any, vat_on_fee: Any = "0", excise: Any = "0") -> _D:
	"""Total customer charge = principal + fee + VAT on fee + excise."""
	return _D(str(principal)) + _D(str(fee)) + _D(str(vat_on_fee)) + _D(str(excise))


def calculate_fx_amount(amount: Any, rate: Any, spread_pct: Any = "0.015") -> dict[str, _D]:
	"""Convert amount using mid-rate with spread. Returns buy/sell/mid rates."""
	mid = _D(str(rate))
	spread = _D(str(spread_pct))
	buy = (mid * (1 - spread)).quantize(_D("0.0001"))
	sell = (mid * (1 + spread)).quantize(_D("0.0001"))
	converted = (_D(str(amount)) * mid).quantize(_D("0.01"))
	return {"converted_amount": converted, "buy_rate": buy, "sell_rate": sell, "mid_rate": mid}


def calculate_ctr_obligation(amount: Any, currency: str = "KES", threshold: Any = "1000000") -> dict:
	"""Determine if transaction requires Currency Transaction Report (CBK threshold: KES 1M)."""
	val = _D(str(amount))
	limit = _D(str(threshold))
	return {
		"requires_ctr": val >= limit,
		"amount": str(val),
		"currency": currency,
		"threshold": str(limit),
		"report_to": "CBK" if currency == "KES" else "central_bank",
	}


def calculate_settlement_net(gross_amount: Any, fees: Any, refunds: Any = "0", chargebacks: Any = "0") -> _D:
	"""Net settlement = gross - fees - refunds - chargebacks."""
	return (
		_D(str(gross_amount)) - _D(str(fees)) - _D(str(refunds)) - _D(str(chargebacks))
	).quantize(_D("0.01"))


def calculate_late_settlement_penalty(outstanding: Any, days_late: int, daily_rate: Any = "0.001") -> _D:
	"""Late settlement penalty = outstanding × daily_rate × days_late."""
	return (_D(str(outstanding)) * _D(str(daily_rate)) * _D(str(days_late))).quantize(_D("0.01"))
