"""Financial calculations for APG Digital Payments — Africa-first.

All functions are pure (no I/O, no side-effects), type-safe, and handle
edge cases explicitly. Used by service.py and domain/rules.py.

Covers:
- M-Pesa fee schedule (Safaricom Kenya, 2025)
- MTN MoMo / Airtel / Tigo fee structures
- Kenya tax (excise, VAT, withholding)
- SWIFT charges (SHA/OUR/BEN)
- FX conversion with spread
- Settlement net calculation
- Chargeback fee estimation
- Partial settlement across clearing cycles
- Regulatory threshold detection
- Velocity scoring
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP, ROUND_CEILING
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MPESA_FEE_TIERS: list[tuple[Decimal, Decimal, Decimal]] = [
	(Decimal("1"),      Decimal("100"),    Decimal("0")),
	(Decimal("101"),    Decimal("500"),    Decimal("7")),
	(Decimal("501"),    Decimal("1000"),   Decimal("13")),
	(Decimal("1001"),   Decimal("1500"),   Decimal("23")),
	(Decimal("1501"),   Decimal("2500"),   Decimal("33")),
	(Decimal("2501"),   Decimal("3500"),   Decimal("53")),
	(Decimal("3501"),   Decimal("5000"),   Decimal("57")),
	(Decimal("5001"),   Decimal("7500"),   Decimal("78")),
	(Decimal("7501"),   Decimal("10000"),  Decimal("90")),
	(Decimal("10001"),  Decimal("15000"),  Decimal("100")),
	(Decimal("15001"),  Decimal("20000"),  Decimal("105")),
	(Decimal("20001"),  Decimal("35000"),  Decimal("108")),
	(Decimal("35001"),  Decimal("250000"), Decimal("108")),
	(Decimal("250001"), Decimal("999999"), Decimal("108")),
]

# Kenya: excise 20% on financial fees (Finance Act 2022), VAT 16%
KE_EXCISE_RATE  = Decimal("0.20")
KE_VAT_RATE     = Decimal("0.16")

# SWIFT correspondent banking fees
SWIFT_OUR_FEE   = Decimal("35")   # USD — all charges to sender
SWIFT_SHA_FEE   = Decimal("15")   # USD — shared
SWIFT_BEN_FEE   = Decimal("0")    # sender pays nothing, beneficiary pays

# MTN MoMo Uganda (UGX) fee schedule (simplified tiers)
MTN_MOMO_FEE_TIERS: list[tuple[Decimal, Decimal, Decimal]] = [
	(Decimal("500"),    Decimal("2500"),   Decimal("250")),
	(Decimal("2501"),   Decimal("5000"),   Decimal("350")),
	(Decimal("5001"),   Decimal("15000"),  Decimal("600")),
	(Decimal("15001"),  Decimal("45000"),  Decimal("800")),
	(Decimal("45001"),  Decimal("90000"),  Decimal("1000")),
	(Decimal("90001"),  Decimal("200000"), Decimal("1500")),
	(Decimal("200001"), Decimal("500000"), Decimal("2000")),
	(Decimal("500001"), Decimal("2000000"),Decimal("3000")),
]

# Airtel Money Kenya (KES) — similar to M-Pesa but slightly lower
AIRTEL_MONEY_FEE_TIERS: list[tuple[Decimal, Decimal, Decimal]] = [
	(Decimal("1"),      Decimal("100"),    Decimal("0")),
	(Decimal("101"),    Decimal("500"),    Decimal("5")),
	(Decimal("501"),    Decimal("1000"),   Decimal("10")),
	(Decimal("1001"),   Decimal("2500"),   Decimal("20")),
	(Decimal("2501"),   Decimal("5000"),   Decimal("40")),
	(Decimal("5001"),   Decimal("10000"),  Decimal("65")),
	(Decimal("10001"),  Decimal("20000"),  Decimal("85")),
	(Decimal("20001"),  Decimal("70000"),  Decimal("97")),
]

# FX mid-rates relative to KES (indicative interbank Q4 2025)
FX_MID_RATES: dict[str, Decimal] = {
	"KES": Decimal("1"),
	"UGX": Decimal("0.035"),
	"TZS": Decimal("0.046"),
	"RWF": Decimal("0.077"),
	"GHS": Decimal("9.8"),
	"NGN": Decimal("0.078"),
	"ZAR": Decimal("6.8"),
	"USD": Decimal("129.5"),
	"EUR": Decimal("141.2"),
	"GBP": Decimal("164.0"),
	"XOF": Decimal("0.215"),
	"XAF": Decimal("0.215"),
}


# ---------------------------------------------------------------------------
# Dataclasses for structured return values
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeeBreakdown:
	"""Detailed breakdown of a payment fee."""
	base_fee:       Decimal
	excise_duty:    Decimal
	vat:            Decimal
	total:          Decimal
	currency:       str
	tier_label:     str = ""


@dataclass(frozen=True)
class FXResult:
	"""Result of a foreign exchange conversion."""
	from_amount:    Decimal
	to_amount:      Decimal
	from_currency:  str
	to_currency:    str
	mid_rate:       Decimal
	effective_rate: Decimal
	spread_bps:     int
	fee:            Decimal = Decimal("0")


@dataclass(frozen=True)
class SettlementResult:
	"""Net settlement calculation for a batch."""
	gross_amount:        Decimal
	processing_fee:      Decimal
	processing_fee_rate_bps: int
	net_amount:          Decimal
	currency:            str


@dataclass(frozen=True)
class PartialSettlement:
	"""One leg of a multi-cycle partial settlement."""
	cycle:           int
	amount:          Decimal
	cumulative:      Decimal
	remaining:       Decimal
	currency:        str
	is_final:        bool


# ---------------------------------------------------------------------------
# M-Pesa fee calculations
# ---------------------------------------------------------------------------

def mpesa_fee(amount: Decimal, include_excise: bool = True) -> FeeBreakdown:
	"""Calculate M-Pesa withdrawal/transfer fee including Kenya excise.

	Args:
		amount: Transaction amount in KES.
		include_excise: Whether to include 20% excise duty (default True).

	Returns:
		FeeBreakdown with base_fee, excise_duty, vat (0 for M-Pesa), total.
	"""
	assert amount >= 0, "Amount must be non-negative"
	base = Decimal("0")
	tier_label = "free"
	for lo, hi, fee in MPESA_FEE_TIERS:
		if lo <= amount <= hi:
			base = fee
			tier_label = f"{lo}-{hi}"
			break
	else:
		base = Decimal("108")
		tier_label = "250001+"

	excise = (base * KE_EXCISE_RATE).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP) if include_excise else Decimal("0")
	vat = Decimal("0")   # M-Pesa fees are VAT-exempt (regulated)
	total = base + excise + vat
	return FeeBreakdown(
		base_fee=base,
		excise_duty=excise,
		vat=vat,
		total=total,
		currency="KES",
		tier_label=tier_label,
	)


def mpesa_send_fee(amount: Decimal) -> FeeBreakdown:
	"""M-Pesa send money fee (same schedule as withdrawal in 2025)."""
	return mpesa_fee(amount, include_excise=True)


# ---------------------------------------------------------------------------
# MTN MoMo fee calculations
# ---------------------------------------------------------------------------

def mtn_momo_fee(amount: Decimal, currency: str = "UGX") -> FeeBreakdown:
	"""Calculate MTN MoMo transfer fee.

	Args:
		amount: Transaction amount.
		currency: Currency code (default UGX for Uganda).

	Returns:
		FeeBreakdown.
	"""
	assert amount >= 0
	base = Decimal("0")
	tier_label = "free"
	for lo, hi, fee in MTN_MOMO_FEE_TIERS:
		if lo <= amount <= hi:
			base = fee
			tier_label = f"{lo}-{hi}"
			break
	else:
		if amount > 0:
			base = Decimal("3000")
			tier_label = "500001+"

	# Uganda applies 0.5% levy on MoMo (OTT tax abolished 2023, replaced by 0.5% on withdrawals)
	excise = (base * Decimal("0.005") * amount).quantize(Decimal("1"), rounding=ROUND_CEILING) if currency == "UGX" else Decimal("0")
	return FeeBreakdown(
		base_fee=base,
		excise_duty=Decimal("0"),
		vat=excise,
		total=base + excise,
		currency=currency,
		tier_label=tier_label,
	)


# ---------------------------------------------------------------------------
# Airtel Money fee calculations
# ---------------------------------------------------------------------------

def airtel_money_fee(amount: Decimal, currency: str = "KES") -> FeeBreakdown:
	"""Calculate Airtel Money transfer fee."""
	assert amount >= 0
	base = Decimal("0")
	tier_label = "free"
	for lo, hi, fee in AIRTEL_MONEY_FEE_TIERS:
		if lo <= amount <= hi:
			base = fee
			tier_label = f"{lo}-{hi}"
			break
	else:
		if amount > 0:
			base = Decimal("97")
			tier_label = "20001+"

	excise = (base * KE_EXCISE_RATE).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP) if currency == "KES" else Decimal("0")
	return FeeBreakdown(
		base_fee=base,
		excise_duty=excise,
		vat=Decimal("0"),
		total=base + excise,
		currency=currency,
		tier_label=tier_label,
	)


# ---------------------------------------------------------------------------
# Bank transfer fees
# ---------------------------------------------------------------------------

def bank_eft_fee(amount: Decimal, currency: str = "KES") -> FeeBreakdown:
	"""Kenya EFT/RTGS fee: KES 50 + 0.1% (capped at KES 5,000)."""
	assert amount >= 0
	raw = Decimal("50") + amount * Decimal("0.001")
	base = min(raw, Decimal("5000")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
	excise = (base * KE_EXCISE_RATE).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
	vat    = (base * KE_VAT_RATE).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
	return FeeBreakdown(
		base_fee=base,
		excise_duty=excise,
		vat=vat,
		total=base + excise + vat,
		currency=currency,
	)


def pesalink_fee(amount: Decimal) -> FeeBreakdown:
	"""PesaLink interbank transfer fee (Kenya Banker's Association, 2025)."""
	assert amount >= 0
	if amount <= Decimal("1000"):
		base = Decimal("12")
	elif amount <= Decimal("10000"):
		base = Decimal("35")
	elif amount <= Decimal("50000"):
		base = Decimal("55")
	elif amount <= Decimal("100000"):
		base = Decimal("85")
	else:
		base = Decimal("105")
	excise = (base * KE_EXCISE_RATE).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
	return FeeBreakdown(
		base_fee=base,
		excise_duty=excise,
		vat=Decimal("0"),
		total=base + excise,
		currency="KES",
	)


# ---------------------------------------------------------------------------
# SWIFT fee calculations
# ---------------------------------------------------------------------------

def swift_fee(charges: str = "SHA", currency: str = "USD") -> FeeBreakdown:
	"""SWIFT transfer fee based on charge arrangement.

	Args:
		charges: "SHA" (shared), "OUR" (sender pays all), "BEN" (beneficiary pays).
		currency: Currency of the SWIFT payment.

	Returns:
		FeeBreakdown in USD equivalent.
	"""
	base_map = {"SHA": SWIFT_SHA_FEE, "OUR": SWIFT_OUR_FEE, "BEN": SWIFT_BEN_FEE}
	base = base_map.get(charges.upper(), SWIFT_SHA_FEE)
	# SWIFT fees are in USD; no Kenya excise on international wire fees
	return FeeBreakdown(
		base_fee=base,
		excise_duty=Decimal("0"),
		vat=Decimal("0"),
		total=base,
		currency="USD",
		tier_label=f"charges={charges}",
	)


# ---------------------------------------------------------------------------
# FX calculations
# ---------------------------------------------------------------------------

def fx_convert(
	from_amount: Decimal,
	from_currency: str,
	to_currency: str,
	spread_bps: int = 150,
	direction: str = "buy",
	custom_mid_rate: Decimal | None = None,
) -> FXResult:
	"""Convert an amount from one currency to another with spread.

	Args:
		from_amount: Amount to convert.
		from_currency: Source currency ISO code.
		to_currency: Target currency ISO code.
		spread_bps: Spread in basis points (default 150 = 1.5%).
		direction: 'buy' (customer buys to_currency) or 'sell'.
		custom_mid_rate: Override the static mid-rate table (live rate injection).

	Returns:
		FXResult with from/to amounts, rates, and fee.
	"""
	assert from_amount >= 0, "from_amount must be non-negative"
	assert spread_bps >= 0, "spread_bps must be non-negative"

	if from_currency.upper() == to_currency.upper():
		return FXResult(
			from_amount=from_amount,
			to_amount=from_amount,
			from_currency=from_currency,
			to_currency=to_currency,
			mid_rate=Decimal("1"),
			effective_rate=Decimal("1"),
			spread_bps=0,
		)

	if custom_mid_rate is not None:
		mid = custom_mid_rate
	else:
		from_mid = FX_MID_RATES.get(from_currency.upper(), Decimal("1"))
		to_mid   = FX_MID_RATES.get(to_currency.upper(), Decimal("1"))
		mid = (from_mid / to_mid).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)

	half_spread = Decimal(str(spread_bps)) / Decimal("20000")
	if direction == "buy":
		effective = mid * (1 - half_spread)
	else:
		effective = mid * (1 + half_spread)
	effective = effective.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)

	to_amount = (from_amount * effective).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
	# Spread cost in from_currency terms
	mid_amount = (from_amount * mid).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
	spread_cost = abs(to_amount - mid_amount)

	return FXResult(
		from_amount=from_amount,
		to_amount=to_amount,
		from_currency=from_currency.upper(),
		to_currency=to_currency.upper(),
		mid_rate=mid,
		effective_rate=effective,
		spread_bps=spread_bps,
		fee=spread_cost,
	)


def fx_gain_loss(
	original_rate: Decimal,
	settlement_rate: Decimal,
	amount_original_currency: Decimal,
) -> Decimal:
	"""Calculate FX gain/loss on settlement.

	Positive = gain, Negative = loss (from payer perspective).
	"""
	assert amount_original_currency >= 0
	gain = (settlement_rate - original_rate) * amount_original_currency
	return gain.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Settlement calculations
# ---------------------------------------------------------------------------

def settlement_net(
	gross: Decimal,
	processing_fee_rate_bps: int = 200,
	currency: str = "KES",
) -> SettlementResult:
	"""Calculate net settlement after processing fee.

	Args:
		gross: Gross settlement amount.
		processing_fee_rate_bps: Processing fee in basis points (default 200 = 2%).
		currency: Currency code.

	Returns:
		SettlementResult.
	"""
	assert gross >= 0
	fee = (gross * Decimal(str(processing_fee_rate_bps)) / Decimal("10000")).quantize(
		Decimal("0.01"), rounding=ROUND_HALF_UP
	)
	net = gross - fee
	return SettlementResult(
		gross_amount=gross,
		processing_fee=fee,
		processing_fee_rate_bps=processing_fee_rate_bps,
		net_amount=net,
		currency=currency,
	)


def partial_settlement_schedule(
	total: Decimal,
	cycles: int,
	currency: str = "KES",
) -> list[PartialSettlement]:
	"""Split a settlement across multiple clearing cycles.

	Distributes evenly; any remainder goes to the final cycle.

	Args:
		total: Total amount to settle.
		cycles: Number of clearing cycles.
		currency: Currency code.

	Returns:
		List of PartialSettlement records.
	"""
	assert total >= 0
	assert cycles >= 1
	per_cycle = (total / Decimal(str(cycles))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
	results: list[PartialSettlement] = []
	cumulative = Decimal("0")
	for i in range(cycles):
		is_final = i == cycles - 1
		amount = (total - cumulative) if is_final else per_cycle
		cumulative += amount
		remaining = total - cumulative
		results.append(PartialSettlement(
			cycle=i + 1,
			amount=amount,
			cumulative=cumulative,
			remaining=remaining,
			currency=currency,
			is_final=is_final,
		))
	return results


def settlement_variance(expected: Decimal, actual: Decimal) -> tuple[Decimal, Decimal]:
	"""Return (absolute_variance, variance_bps).

	variance_bps = |variance| / expected * 10000
	"""
	if expected == 0:
		return Decimal("0"), Decimal("0")
	variance = actual - expected
	bps = (abs(variance) / expected * Decimal("10000")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
	return variance, bps


# ---------------------------------------------------------------------------
# Card calculations
# ---------------------------------------------------------------------------

def card_interchange_fee(amount: Decimal, card_type: str = "standard", currency: str = "KES") -> FeeBreakdown:
	"""Estimate interchange fee for card payments.

	Rates approximate Visa/Mastercard Kenya schedule (2025).
	"""
	assert amount >= 0
	rates = {
		"standard":    Decimal("0.0175"),  # 1.75%
		"premium":     Decimal("0.023"),   # 2.3%
		"corporate":   Decimal("0.025"),   # 2.5%
		"debit":       Decimal("0.012"),   # 1.2%
		"prepaid":     Decimal("0.015"),   # 1.5%
	}
	rate = rates.get(card_type.lower(), rates["standard"])
	base = (amount * rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
	excise = (base * KE_EXCISE_RATE).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
	return FeeBreakdown(
		base_fee=base,
		excise_duty=excise,
		vat=Decimal("0"),
		total=base + excise,
		currency=currency,
		tier_label=f"card_type={card_type} rate={rate}",
	)


def chargeback_fee(amount: Decimal, scheme: str = "visa", currency: str = "USD") -> Decimal:
	"""Estimate chargeback processing fee from card scheme.

	Visa/Mastercard: USD 20-25 per case regardless of transaction amount.
	"""
	assert amount >= 0
	fees = {"visa": Decimal("20"), "mastercard": Decimal("20"), "amex": Decimal("25")}
	return fees.get(scheme.lower(), Decimal("20"))


# ---------------------------------------------------------------------------
# Tax calculations
# ---------------------------------------------------------------------------

def ke_withholding_tax(amount: Decimal, rate: Decimal = Decimal("0.05")) -> Decimal:
	"""Kenya withholding tax on payments to resident contractors (5%)."""
	return (amount * rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def ke_vat(amount: Decimal) -> Decimal:
	"""Kenya VAT at 16%."""
	return (amount * KE_VAT_RATE).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def ke_excise(fee: Decimal) -> Decimal:
	"""Kenya excise duty on financial services at 20%."""
	return (fee * KE_EXCISE_RATE).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Velocity scoring
# ---------------------------------------------------------------------------

def velocity_score(
	txn_count_24h:   int,
	amount_sum_24h:  Decimal,
	unique_recipients_24h: int,
	failed_count_24h: int,
	avg_amount:      Decimal,
	current_amount:  Decimal,
) -> dict[str, Any]:
	"""Produce a 0-100 risk velocity score.

	Higher score = higher risk. Inputs are tenant-scoped 24h rolling windows.

	Scoring components:
	  - txn_count_24h: > 20 = suspicious
	  - amount_deviation: current vs avg (> 10x = suspicious)
	  - failed_count_24h: > 3 = suspicious
	  - unique_recipients: > 10 = potential money mule
	"""
	score = Decimal("0")
	flags: list[str] = []

	# Transaction count component (0-30)
	if txn_count_24h > 50:
		score += 30
		flags.append("excessive_velocity")
	elif txn_count_24h > 20:
		score += 15
		flags.append("high_velocity")
	elif txn_count_24h > 10:
		score += 5

	# Amount deviation (0-25)
	if avg_amount > 0:
		deviation = current_amount / avg_amount
		if deviation > 20:
			score += 25
			flags.append("extreme_amount_deviation")
		elif deviation > 10:
			score += 15
			flags.append("high_amount_deviation")
		elif deviation > 5:
			score += 5

	# Failed transactions (0-20)
	if failed_count_24h > 5:
		score += 20
		flags.append("excessive_failures")
	elif failed_count_24h > 3:
		score += 10
		flags.append("elevated_failures")

	# Fan-out (0-25)
	if unique_recipients_24h > 20:
		score += 25
		flags.append("fan_out_pattern")
	elif unique_recipients_24h > 10:
		score += 10
		flags.append("elevated_fan_out")

	score = min(score, Decimal("100")).quantize(Decimal("1"))
	level = "low"
	if score >= 70:
		level = "critical"
	elif score >= 50:
		level = "high"
	elif score >= 30:
		level = "medium"

	return {
		"score": int(score),
		"level": level,
		"flags": flags,
		"inputs": {
			"txn_count_24h": txn_count_24h,
			"amount_sum_24h": str(amount_sum_24h),
			"unique_recipients_24h": unique_recipients_24h,
			"failed_count_24h": failed_count_24h,
			"current_amount": str(current_amount),
		},
	}


# ---------------------------------------------------------------------------
# Reconciliation helpers
# ---------------------------------------------------------------------------

def reconcile_amounts(
	expected: list[Decimal],
	actual: list[Decimal],
) -> dict[str, Any]:
	"""Reconcile two lists of amounts, identifying matches, variances, and misses.

	Returns a summary with total_matched, total_variance, unmatched_expected,
	unmatched_actual.
	"""
	assert len(expected) == len(actual), "Lists must have equal length"
	matched = 0
	total_variance = Decimal("0")
	variances: list[dict[str, Any]] = []
	for i, (exp, act) in enumerate(zip(expected, actual)):
		var = act - exp
		if var == 0:
			matched += 1
		else:
			total_variance += var
			variances.append({"index": i, "expected": str(exp), "actual": str(act), "variance": str(var)})

	return {
		"total": len(expected),
		"matched": matched,
		"variance_count": len(variances),
		"total_variance": str(total_variance),
		"variances": variances,
		"reconciled": len(variances) == 0,
	}
