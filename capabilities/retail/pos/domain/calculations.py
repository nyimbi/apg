"""Financial and domain calculations for APG Point of Sale.

Pure functions. No I/O, no side effects.
All monetary values are floats rounded to 4 decimal places internally,
2 decimal places on output.
"""

from __future__ import annotations

from typing import Any


def round2(v: float) -> float:
	return round(v, 2)


def round4(v: float) -> float:
	return round(v, 4)


# ---------------------------------------------------------------------------
# Item-level calculations
# ---------------------------------------------------------------------------

def item_subtotal(unit_price: float, quantity: float) -> float:
	"""Gross item amount before discounts/tax."""
	return round4(unit_price * quantity)


def item_discount(
	unit_price: float,
	quantity: float,
	discount_pct: float = 0.0,
	discount_fixed: float = 0.0,
) -> float:
	"""Item-level discount. Percentage takes precedence over fixed if both given."""
	base = item_subtotal(unit_price, quantity)
	if discount_pct > 0:
		return round4(base * min(discount_pct / 100.0, 1.0))
	return round4(min(discount_fixed, base))


def item_taxable_amount(
	unit_price: float,
	quantity: float,
	discount: float,
	tax_inclusive: bool,
) -> float:
	"""Amount subject to tax after discount."""
	base = item_subtotal(unit_price, quantity) - discount
	return round4(max(base, 0.0))


def item_tax(taxable: float, tax_rate: float, tax_inclusive: bool = True) -> float:
	"""Extract (inclusive) or compute (exclusive) tax."""
	if tax_inclusive:
		return round4(taxable - taxable / (1 + tax_rate))
	return round4(taxable * tax_rate)


def item_net(taxable: float, tax: float, tax_inclusive: bool = True) -> float:
	"""Net (ex-tax) value of an item line."""
	if tax_inclusive:
		return round4(taxable - tax)
	return round4(taxable)


def item_line_total(unit_price: float, quantity: float, discount: float = 0.0) -> float:
	"""Final line total charged to customer (tax inclusive)."""
	return round4(item_subtotal(unit_price, quantity) - discount)


# ---------------------------------------------------------------------------
# Transaction-level calculations
# ---------------------------------------------------------------------------

def transaction_subtotal(line_totals: list[float]) -> float:
	return round4(sum(line_totals))


def transaction_discount(
	subtotal: float,
	header_discount_pct: float = 0.0,
	header_discount_fixed: float = 0.0,
	item_discounts_total: float = 0.0,
) -> float:
	"""Combined transaction-level discount."""
	header = 0.0
	if header_discount_pct > 0:
		header = round4(subtotal * min(header_discount_pct / 100.0, 1.0))
	elif header_discount_fixed > 0:
		header = round4(min(header_discount_fixed, subtotal))
	return round4(item_discounts_total + header)


def transaction_tax(item_taxes: list[float]) -> float:
	return round4(sum(item_taxes))


def transaction_grand_total(subtotal: float, extra_discount: float = 0.0) -> float:
	"""Grand total paid by customer (subtotal is already net of item discounts)."""
	return round2(max(subtotal - extra_discount, 0.0))


def change_due(grand_total: float, tendered: float) -> float:
	"""Change owed to customer. Positive = change. Negative = balance still owed."""
	return round2(tendered - grand_total)


def balance_due(grand_total: float, payments_total: float) -> float:
	"""Remaining amount to collect. 0 means fully paid."""
	return round2(max(grand_total - payments_total, 0.0))


# ---------------------------------------------------------------------------
# Multi-tender / split payment
# ---------------------------------------------------------------------------

def allocate_split_payments(
	grand_total: float,
	payments: list[dict[str, float]],
) -> dict[str, Any]:
	"""
	Validate and summarise a split-payment tender.

	payments: list of {"method": ..., "amount": ...}

	Returns:
		total_tendered, change_due, balance_remaining, fully_paid
	"""
	total = sum(p["amount"] for p in payments)
	chg = round2(max(total - grand_total, 0.0))
	bal = round2(max(grand_total - total, 0.0))
	return {
		"total_tendered": round2(total),
		"change_due": chg,
		"balance_remaining": bal,
		"fully_paid": bal == 0.0,
	}


# ---------------------------------------------------------------------------
# Loyalty calculations
# ---------------------------------------------------------------------------

def earn_points(purchase_amount: float, earn_rate: float) -> int:
	"""Points earned on a purchase."""
	return int(purchase_amount * earn_rate)


def redeem_value(points: int, redeem_rate: float) -> float:
	"""Currency value of redeemed points."""
	return round2(points * redeem_rate)


def points_after_earn(balance: int, earned: int) -> int:
	return balance + earned


def points_after_redeem(balance: int, redeemed: int) -> int:
	result = balance - redeemed
	assert result >= 0, "insufficient loyalty points"
	return result


# ---------------------------------------------------------------------------
# Cash management
# ---------------------------------------------------------------------------

def expected_cash_in_till(
	opening_float: float,
	cash_sales: float,
	cash_refunds: float,
	safe_drops: float,
	safe_pickups: float,
	petty_cash_out: float,
	petty_cash_in: float,
	till_loans: float,
	corrections: float,
) -> float:
	"""Expected physical cash in the till drawer."""
	return round2(
		opening_float
		+ cash_sales
		- cash_refunds
		- safe_drops
		- safe_pickups
		- petty_cash_out
		+ petty_cash_in
		+ till_loans
		+ corrections
	)


def cash_variance(expected: float, counted: float) -> float:
	"""Positive = overage, negative = shortage."""
	return round2(counted - expected)


def variance_percentage(expected: float, variance: float) -> float:
	"""Variance as a percentage of expected; 0 if expected is 0."""
	if expected == 0:
		return 0.0
	return round2(abs(variance) / expected * 100)


# ---------------------------------------------------------------------------
# Denomination counting
# ---------------------------------------------------------------------------

STANDARD_DENOMINATIONS_KES = [1000, 500, 200, 100, 50, 20, 10, 5, 1]


def count_denominations(denominations: dict[str, int]) -> float:
	"""Sum cash from a denomination count dict {"1000": 3, "500": 5, ...}."""
	total = 0.0
	for denom_str, count in denominations.items():
		try:
			total += float(denom_str) * count
		except (ValueError, TypeError):
			pass
	return round2(total)


def suggest_denominations(amount: float, denoms: list[int] | None = None) -> dict[str, int]:
	"""Greedy breakdown of amount into denominations (for change suggestion)."""
	if denoms is None:
		denoms = STANDARD_DENOMINATIONS_KES
	result: dict[str, int] = {}
	remaining = int(round(amount))
	for d in sorted(denoms, reverse=True):
		if remaining <= 0:
			break
		count = remaining // d
		if count > 0:
			result[str(d)] = count
			remaining -= count * d
	return result


# ---------------------------------------------------------------------------
# Tax calculation
# ---------------------------------------------------------------------------

def vat_inclusive_breakdown(
	inclusive_amount: float,
	vat_rate: float,
) -> dict[str, float]:
	"""Break down a VAT-inclusive amount into net and VAT components."""
	vat = round4(inclusive_amount - inclusive_amount / (1 + vat_rate))
	net = round4(inclusive_amount - vat)
	return {"net": round2(net), "vat": round2(vat), "gross": round2(inclusive_amount)}


def apply_tax_exemption(line_totals: list[dict[str, Any]]) -> list[dict[str, Any]]:
	"""Zero out tax amounts for a tax-exempt customer."""
	result = []
	for item in line_totals:
		item = dict(item)
		item["tax_amount"] = 0.0
		item["tax_rate"] = 0.0
		result.append(item)
	return result


# ---------------------------------------------------------------------------
# EOD / reporting calculations
# ---------------------------------------------------------------------------

def gross_margin(
	net_sales: float,
	cost_of_goods: float,
) -> dict[str, float]:
	"""Gross margin and percentage."""
	margin = round2(net_sales - cost_of_goods)
	pct = round2(margin / net_sales * 100) if net_sales > 0 else 0.0
	return {"gross_margin": margin, "gross_margin_pct": pct}


def average_transaction_value(total_sales: float, transaction_count: int) -> float:
	if transaction_count == 0:
		return 0.0
	return round2(total_sales / transaction_count)


def items_per_transaction(total_items: int, transaction_count: int) -> float:
	if transaction_count == 0:
		return 0.0
	return round2(total_items / transaction_count)


def hourly_sales_breakdown(
	transactions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
	"""Group transactions by hour of day."""
	hourly: dict[int, dict[str, Any]] = {}
	for txn in transactions:
		created = txn.get("created_at") or txn.get("posted_at")
		if isinstance(created, str):
			from datetime import datetime
			created = datetime.fromisoformat(created)
		if created is None:
			continue
		hour = created.hour
		if hour not in hourly:
			hourly[hour] = {"hour": hour, "count": 0, "total": 0.0}
		hourly[hour]["count"] += 1
		hourly[hour]["total"] = round2(hourly[hour]["total"] + txn.get("grand_total", 0.0))
	return sorted(hourly.values(), key=lambda x: x["hour"])


def top_selling_skus(
	sale_items: list[dict[str, Any]],
	top_n: int = 10,
) -> list[dict[str, Any]]:
	"""Return top N SKUs by revenue."""
	agg: dict[str, dict[str, Any]] = {}
	for item in sale_items:
		sku = item.get("sku", "")
		if sku not in agg:
			agg[sku] = {"sku": sku, "description": item.get("description", ""), "qty": 0.0, "revenue": 0.0}
		agg[sku]["qty"] += item.get("quantity", 0.0)
		agg[sku]["revenue"] = round2(agg[sku]["revenue"] + item.get("line_total", 0.0))
	return sorted(agg.values(), key=lambda x: x["revenue"], reverse=True)[:top_n]
