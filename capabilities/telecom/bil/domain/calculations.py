"""Financial and domain calculations for Telecom Billing.

All formulas are pure functions — type-safe, deterministic, edge-case hardened.
No I/O, no side effects. Import freely from service or rating engine.

© 2025 Datacraft. All rights reserved.
"""
from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CENT = Decimal("0.01")
ZERO = Decimal("0")
ONE = Decimal("1")
HUNDRED = Decimal("100")

# Minimum billable time unit for voice (6-second pulse — East Africa standard)
VOICE_PULSE_SECONDS = 6

# KRA default VAT rate (Kenya)
DEFAULT_VAT_PCT = Decimal("16")


# ---------------------------------------------------------------------------
# Rounding
# ---------------------------------------------------------------------------

def round_currency(amount: Decimal, places: int = 2) -> Decimal:
	"""Round to currency precision using ROUND_HALF_UP (banker-safe)."""
	quantize_str = Decimal(10) ** -places
	return amount.quantize(quantize_str, rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Voice rating
# ---------------------------------------------------------------------------

def calculate_voice_charge(
	duration_seconds: int,
	rate_per_second: Decimal,
	minimum_charge: Decimal = ZERO,
	pulse_seconds: int = VOICE_PULSE_SECONDS,
) -> Decimal:
	"""Rate a voice call with pulse-based billing.

	duration_seconds=0 → minimum_charge applies if set.
	Default pulse is 6 seconds (common in East Africa).
	"""
	assert duration_seconds >= 0, "duration must be >= 0"
	assert rate_per_second >= ZERO, "rate must be >= 0"
	assert pulse_seconds > 0, "pulse must be > 0"

	if duration_seconds == 0:
		return round_currency(minimum_charge)

	# Ceiling division for pulse billing
	pulses = -(-duration_seconds // pulse_seconds)
	charge = Decimal(str(pulses * pulse_seconds)) * rate_per_second
	return round_currency(max(charge, minimum_charge))


def calculate_tiered_voice_charge(
	duration_seconds: int,
	tiers: list[dict[str, Any]],
	minimum_charge: Decimal = ZERO,
) -> Decimal:
	"""Rate voice using tiered tariff.

	tiers: [{"up_to_seconds": 60, "rate_per_second": "0.10"},
	        {"up_to_seconds": 300, "rate_per_second": "0.08"},
	        {"up_to_seconds": None, "rate_per_second": "0.06"}]
	"""
	assert duration_seconds >= 0
	if duration_seconds == 0:
		return round_currency(minimum_charge)

	remaining = duration_seconds
	total = ZERO
	prev_threshold = 0

	for tier in sorted(tiers, key=lambda t: t.get("up_to_seconds") or float("inf")):
		rate = Decimal(str(tier["rate_per_second"]))
		up_to = tier.get("up_to_seconds")
		tier_seconds = (up_to - prev_threshold) if up_to is not None else remaining
		billable = min(remaining, tier_seconds)
		total += Decimal(str(billable)) * rate
		remaining -= billable
		prev_threshold = up_to or 0
		if remaining <= 0:
			break

	return round_currency(max(total, minimum_charge))


def calculate_time_of_day_voice_charge(
	duration_seconds: int,
	rate_per_second: Decimal,
	off_peak_rate_per_second: Decimal,
	peak_start_hour: int,
	peak_end_hour: int,
	call_hour: int,
	minimum_charge: Decimal = ZERO,
	pulse_seconds: int = VOICE_PULSE_SECONDS,
) -> Decimal:
	"""Rate voice differentiating peak vs. off-peak hours."""
	assert 0 <= call_hour <= 23
	is_peak = peak_start_hour <= call_hour < peak_end_hour
	rate = rate_per_second if is_peak else off_peak_rate_per_second
	return calculate_voice_charge(duration_seconds, rate, minimum_charge, pulse_seconds)


# ---------------------------------------------------------------------------
# Data rating
# ---------------------------------------------------------------------------

def calculate_data_charge(
	data_bytes: int,
	rate_per_kb: Decimal,
	minimum_charge: Decimal = ZERO,
	rounding_kb: int = 1,
) -> Decimal:
	"""Rate data usage in KBs (ceiling-rounded per KB by default)."""
	assert data_bytes >= 0
	assert rate_per_kb >= ZERO

	if data_bytes == 0:
		return round_currency(minimum_charge)

	kb = -(-data_bytes // (1024 * rounding_kb)) * rounding_kb
	charge = Decimal(str(kb)) * rate_per_kb
	return round_currency(max(charge, minimum_charge))


def calculate_data_charge_gb(
	data_bytes: int,
	rate_per_gb: Decimal,
	minimum_charge: Decimal = ZERO,
) -> Decimal:
	"""Rate data at per-GB granularity (used for high-volume accounts)."""
	assert data_bytes >= 0
	gb = Decimal(str(data_bytes)) / Decimal("1073741824")
	charge = gb * rate_per_gb
	return round_currency(max(charge, minimum_charge))


def calculate_volume_data_charge(
	data_bytes: int,
	tiers: list[dict[str, Any]],
) -> Decimal:
	"""Volume-break data pricing.

	tiers: [{"up_to_gb": 1, "rate_per_gb": "100"},
	        {"up_to_gb": 10, "rate_per_gb": "80"},
	        {"up_to_gb": None, "rate_per_gb": "60"}]
	"""
	gb = Decimal(str(data_bytes)) / Decimal("1073741824")
	remaining_gb = gb
	total = ZERO
	prev = ZERO

	for tier in sorted(tiers, key=lambda t: t.get("up_to_gb") or float("inf")):
		rate = Decimal(str(tier["rate_per_gb"]))
		up_to = tier.get("up_to_gb")
		bucket_size = (Decimal(str(up_to)) - prev) if up_to is not None else remaining_gb
		billable = min(remaining_gb, bucket_size)
		total += billable * rate
		remaining_gb -= billable
		prev = Decimal(str(up_to)) if up_to is not None else prev
		if remaining_gb <= ZERO:
			break

	return round_currency(total)


# ---------------------------------------------------------------------------
# SMS rating
# ---------------------------------------------------------------------------

def calculate_sms_charge(
	sms_count: int,
	rate_per_sms: Decimal,
	minimum_charge: Decimal = ZERO,
) -> Decimal:
	"""Rate SMS messages."""
	assert sms_count >= 0
	assert rate_per_sms >= ZERO
	if sms_count == 0:
		return round_currency(minimum_charge)
	return round_currency(max(Decimal(str(sms_count)) * rate_per_sms, minimum_charge))


# ---------------------------------------------------------------------------
# Tax calculations
# ---------------------------------------------------------------------------

def calculate_tax(
	pre_tax_amount: Decimal,
	tax_rate_pct: Decimal,
) -> Decimal:
	"""Calculate tax amount from pre-tax value."""
	assert pre_tax_amount >= ZERO
	assert tax_rate_pct >= ZERO
	return round_currency(pre_tax_amount * tax_rate_pct / HUNDRED)


def calculate_tax_inclusive_split(
	gross_amount: Decimal,
	tax_rate_pct: Decimal,
) -> tuple[Decimal, Decimal]:
	"""Split a tax-inclusive amount into (net, tax).

	Returns (net_amount, tax_amount).
	"""
	assert gross_amount >= ZERO
	assert tax_rate_pct >= ZERO
	divisor = ONE + tax_rate_pct / HUNDRED
	net = round_currency(gross_amount / divisor)
	tax = round_currency(gross_amount - net)
	return net, tax


def calculate_multi_tax(
	pre_tax_amount: Decimal,
	tax_components: list[dict[str, Any]],
) -> dict[str, Decimal]:
	"""Calculate multiple stacked tax components.

	tax_components: [{"name": "vat", "rate_pct": "16"},
	                 {"name": "excise", "rate_pct": "2"},
	                 {"name": "usf", "rate_pct": "0.5"}]
	Returns {tax_name: amount, ..., "total_tax": total}
	"""
	result: dict[str, Decimal] = {}
	total = ZERO
	for component in tax_components:
		name = component["name"]
		rate = Decimal(str(component["rate_pct"]))
		tax = round_currency(pre_tax_amount * rate / HUNDRED)
		result[name] = tax
		total += tax
	result["total_tax"] = round_currency(total)
	return result


def calculate_jurisdiction_tax(
	pre_tax_amount: Decimal,
	jurisdiction: str,
) -> dict[str, Any]:
	"""Return tax breakdown for known jurisdictions.

	Supported: KE (Kenya), UG (Uganda), TZ (Tanzania), NG (Nigeria), ZA (South Africa).
	Falls back to 0% for unknown jurisdictions.
	"""
	JURISDICTION_TAX: dict[str, list[dict[str, Any]]] = {
		"KE": [
			{"name": "vat", "rate_pct": "16"},
			{"name": "excise_duty", "rate_pct": "15"},  # telecom excise
		],
		"UG": [
			{"name": "vat", "rate_pct": "18"},
			{"name": "ott_levy", "rate_pct": "0"},  # removed 2021
		],
		"TZ": [
			{"name": "vat", "rate_pct": "18"},
			{"name": "excise", "rate_pct": "17"},
		],
		"NG": [
			{"name": "vat", "rate_pct": "7.5"},
			{"name": "nca_levy", "rate_pct": "1"},
		],
		"ZA": [
			{"name": "vat", "rate_pct": "15"},
		],
	}
	components = JURISDICTION_TAX.get(jurisdiction.upper(), [])
	breakdown = calculate_multi_tax(pre_tax_amount, components)
	total_tax = breakdown.get("total_tax", ZERO)
	return {
		"jurisdiction": jurisdiction.upper(),
		"pre_tax_amount": pre_tax_amount,
		"components": breakdown,
		"total_tax": total_tax,
		"total_with_tax": round_currency(pre_tax_amount + total_tax),
	}


# ---------------------------------------------------------------------------
# Discount calculations
# ---------------------------------------------------------------------------

def apply_percentage_discount(amount: Decimal, discount_pct: Decimal) -> Decimal:
	"""Apply a percentage discount. Returns discounted amount (not the saving)."""
	assert ZERO <= discount_pct <= HUNDRED, f"discount_pct {discount_pct} out of range"
	saving = round_currency(amount * discount_pct / HUNDRED)
	return round_currency(amount - saving)


def apply_flat_discount(amount: Decimal, flat_discount: Decimal) -> Decimal:
	"""Subtract flat discount, floored at zero."""
	return round_currency(max(ZERO, amount - flat_discount))


def calculate_combined_discount(
	amount: Decimal,
	discount_pct: Decimal,
	flat_discount: Decimal,
	cascade: bool = True,
) -> dict[str, Decimal]:
	"""Apply percentage then flat discount (or in parallel).

	cascade=True: flat applied to post-percentage amount.
	cascade=False: both computed on original, summed.
	"""
	if cascade:
		after_pct = apply_percentage_discount(amount, discount_pct)
		final = apply_flat_discount(after_pct, flat_discount)
		pct_saving = amount - after_pct
		flat_saving = after_pct - final
	else:
		pct_saving = round_currency(amount * discount_pct / HUNDRED)
		flat_saving = min(flat_discount, amount)
		total_saving = min(pct_saving + flat_saving, amount)
		final = amount - total_saving
		pct_saving = round_currency(amount * discount_pct / HUNDRED)
		flat_saving = total_saving - pct_saving

	return {
		"original": amount,
		"pct_saving": pct_saving,
		"flat_saving": flat_saving,
		"total_saving": amount - final,
		"final_amount": final,
	}


def calculate_bundle_overage(
	consumed: Decimal,
	total: Decimal,
	overage_rate: Decimal,
	unit_size: Decimal = ONE,
) -> Decimal:
	"""Charge for consumption beyond bundle allowance."""
	assert consumed >= ZERO
	assert total >= ZERO
	overage_units = max(ZERO, consumed - total)
	if overage_units == ZERO:
		return ZERO
	overage_chunks = -(-int(overage_units) // int(unit_size))
	return round_currency(Decimal(str(overage_chunks)) * overage_rate)


# ---------------------------------------------------------------------------
# Roaming charges
# ---------------------------------------------------------------------------

def calculate_roaming_charge(
	duration_seconds: int,
	data_bytes: int,
	sms_count: int,
	zone_rates: dict[str, Any],
	zone: str,
	minimum_charge: Decimal = ZERO,
) -> dict[str, Decimal]:
	"""Calculate full roaming charge breakdown.

	zone_rates: {
	  "zone_a": {
	    "voice_rate_per_second": "0.15",
	    "data_rate_per_kb": "0.05",
	    "sms_rate": "3.00",
	    "surcharge_pct": "10"
	  }
	}
	Returns {"voice": .., "data": .., "sms": .., "surcharge": .., "total": ..}
	"""
	rates = zone_rates.get(zone, zone_rates.get("default", {}))
	voice = calculate_voice_charge(
		duration_seconds,
		Decimal(str(rates.get("voice_rate_per_second", "0"))),
	)
	data = calculate_data_charge(
		data_bytes,
		Decimal(str(rates.get("data_rate_per_kb", "0"))),
	)
	sms = calculate_sms_charge(
		sms_count,
		Decimal(str(rates.get("sms_rate", "0"))),
	)
	subtotal = voice + data + sms
	surcharge_pct = Decimal(str(rates.get("surcharge_pct", "0")))
	surcharge = round_currency(subtotal * surcharge_pct / HUNDRED)
	total = round_currency(max(subtotal + surcharge, minimum_charge))
	return {"voice": voice, "data": data, "sms": sms, "surcharge": surcharge, "total": total}


def calculate_tap_settlement_amount(
	originating_minutes: Decimal,
	terminating_minutes: Decimal,
	rate_per_minute: Decimal,
	currency: str = "KES",
) -> dict[str, Any]:
	"""Calculate bilateral TAP roaming settlement amounts."""
	receivable = round_currency(originating_minutes * rate_per_minute)
	payable = round_currency(terminating_minutes * rate_per_minute)
	net = round_currency(receivable - payable)
	return {
		"receivable": receivable,
		"payable": payable,
		"net": net,
		"net_direction": "receivable" if net >= ZERO else "payable",
		"currency": currency,
	}


# ---------------------------------------------------------------------------
# Interconnect settlement
# ---------------------------------------------------------------------------

def calculate_interconnect_net(
	receivable: Decimal,
	payable: Decimal,
) -> Decimal:
	"""Net interconnect position. Positive = net receivable."""
	return round_currency(receivable - payable)


def calculate_termination_charge(
	minutes: Decimal,
	rate_per_minute: Decimal,
) -> Decimal:
	"""Calculate termination (MTR/FTR) charges."""
	assert minutes >= ZERO
	return round_currency(minutes * rate_per_minute)


def calculate_transit_charge(
	minutes: Decimal,
	rate_per_minute: Decimal,
) -> Decimal:
	"""Calculate transit switching charges."""
	assert minutes >= ZERO
	return round_currency(minutes * rate_per_minute)


# ---------------------------------------------------------------------------
# Credit limit
# ---------------------------------------------------------------------------

def credit_utilisation_pct(current_usage: Decimal, hard_limit: Decimal) -> Decimal:
	"""Return percentage of credit limit consumed."""
	if hard_limit == ZERO:
		return ZERO
	return round_currency(current_usage / hard_limit * HUNDRED)


def headroom(current_usage: Decimal, hard_limit: Decimal) -> Decimal:
	"""Remaining credit before hard limit."""
	return round_currency(max(ZERO, hard_limit - current_usage))


def is_over_soft_limit(current_usage: Decimal, soft_limit: Decimal) -> bool:
	return current_usage >= soft_limit


def is_over_hard_limit(current_usage: Decimal, hard_limit: Decimal) -> bool:
	return current_usage >= hard_limit


# ---------------------------------------------------------------------------
# Invoice aggregation
# ---------------------------------------------------------------------------

def aggregate_invoice_totals(
	line_items: list[dict[str, Any]],
	discount_pct: Decimal = ZERO,
	flat_discount: Decimal = ZERO,
	tax_rate_pct: Decimal = DEFAULT_VAT_PCT,
) -> dict[str, Decimal]:
	"""Aggregate line items into invoice totals.

	Returns: subtotal, discount_amount, taxable_amount, tax_amount, total_amount
	"""
	subtotal = round_currency(sum(
		Decimal(str(item.get("amount", "0"))) for item in line_items
	))
	pct_saving = round_currency(subtotal * discount_pct / HUNDRED)
	post_pct = subtotal - pct_saving
	flat_saving = min(flat_discount, post_pct)
	taxable = round_currency(post_pct - flat_saving)
	tax = calculate_tax(taxable, tax_rate_pct)
	total = round_currency(taxable + tax)
	return {
		"subtotal": subtotal,
		"discount_amount": pct_saving + flat_saving,
		"taxable_amount": taxable,
		"tax_amount": tax,
		"total_amount": total,
	}


def calculate_late_payment_penalty(
	outstanding: Decimal,
	days_overdue: int,
	daily_penalty_rate_pct: Decimal = Decimal("0.1"),
	max_penalty_pct: Decimal = Decimal("10"),
) -> Decimal:
	"""Calculate late payment penalty (capped at max_penalty_pct of outstanding)."""
	assert outstanding >= ZERO
	assert days_overdue >= 0
	accrued_pct = Decimal(str(days_overdue)) * daily_penalty_rate_pct
	capped_pct = min(accrued_pct, max_penalty_pct)
	return round_currency(outstanding * capped_pct / HUNDRED)


# ---------------------------------------------------------------------------
# Revenue assurance
# ---------------------------------------------------------------------------

def leakage_rate_pct(
	billed_revenue: Decimal,
	estimated_revenue: Decimal,
) -> Decimal:
	"""Percentage of estimated revenue not billed (leakage indicator)."""
	if estimated_revenue == ZERO:
		return ZERO
	leakage = max(ZERO, estimated_revenue - billed_revenue)
	return round_currency(leakage / estimated_revenue * HUNDRED)


def collection_rate_pct(
	collected: Decimal,
	invoiced: Decimal,
) -> Decimal:
	"""Percentage of invoiced revenue actually collected."""
	if invoiced == ZERO:
		return ZERO
	return round_currency(collected / invoiced * HUNDRED)


def arpu(total_revenue: Decimal, active_accounts: int) -> Decimal:
	"""Average Revenue Per User."""
	if active_accounts == 0:
		return ZERO
	return round_currency(total_revenue / Decimal(str(active_accounts)))


def days_sales_outstanding(
	outstanding_receivables: Decimal,
	total_revenue: Decimal,
	period_days: int,
) -> Decimal:
	"""DSO = (outstanding / revenue) * period_days."""
	if total_revenue == ZERO or period_days == 0:
		return ZERO
	return round_currency(
		outstanding_receivables / total_revenue * Decimal(str(period_days))
	)


# ---------------------------------------------------------------------------
# Convergent billing
# ---------------------------------------------------------------------------

def calculate_convergent_bill(
	fixed_line_charges: list[dict[str, Any]],
	mobile_charges: list[dict[str, Any]],
	data_charges: list[dict[str, Any]],
	shared_discount_pct: Decimal = ZERO,
	tax_rate_pct: Decimal = DEFAULT_VAT_PCT,
) -> dict[str, Any]:
	"""Produce a single convergent bill combining fixed, mobile, and data.

	Each charge list item: {"description": str, "amount": str/Decimal}
	"""
	def _sum(items: list[dict[str, Any]]) -> Decimal:
		return round_currency(sum(Decimal(str(i.get("amount", "0"))) for i in items))

	fixed_total = _sum(fixed_line_charges)
	mobile_total = _sum(mobile_charges)
	data_total = _sum(data_charges)
	combined_subtotal = round_currency(fixed_total + mobile_total + data_total)

	discount = round_currency(combined_subtotal * shared_discount_pct / HUNDRED)
	taxable = round_currency(combined_subtotal - discount)
	tax = calculate_tax(taxable, tax_rate_pct)
	total = round_currency(taxable + tax)

	return {
		"fixed_line_total": fixed_total,
		"mobile_total": mobile_total,
		"data_total": data_total,
		"combined_subtotal": combined_subtotal,
		"shared_discount": discount,
		"taxable_amount": taxable,
		"tax_amount": tax,
		"total_amount": total,
	}


# ---------------------------------------------------------------------------
# Spend velocity (fraud detection helper)
# ---------------------------------------------------------------------------

def calculate_spend_velocity(
	charges: list[dict[str, Any]],
	window_minutes: int = 60,
) -> dict[str, Any]:
	"""Compute spend velocity metrics from a list of recent charge events.

	Each charge item: {"amount": str/Decimal, "charged_at": datetime-iso-str}
	Returns velocity stats for the most recent window_minutes.
	"""
	from datetime import datetime, timedelta

	now = datetime.utcnow()
	cutoff = now - timedelta(minutes=window_minutes)
	recent = []
	for c in charges:
		try:
			charged_at = datetime.fromisoformat(str(c.get("charged_at", "")))
			if charged_at >= cutoff:
				recent.append(Decimal(str(c.get("amount", "0"))))
		except (ValueError, TypeError):
			continue

	total_in_window = round_currency(sum(recent))
	count_in_window = len(recent)
	avg_charge = round_currency(total_in_window / Decimal(str(count_in_window))) if count_in_window else ZERO

	return {
		"window_minutes": window_minutes,
		"total_amount": total_in_window,
		"transaction_count": count_in_window,
		"average_per_transaction": avg_charge,
	}
