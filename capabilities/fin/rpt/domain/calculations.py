"""
APG Financial Reporting — Domain Calculations
© 2025 Datacraft. Author: Nyimbi Odero

All financial statement formulas, ratio calculations, and period comparison
utilities. Type-safe, edge-case-hardened, with Decimal precision where needed.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Any


_D = Decimal


def _dec(v: float | int | str | None) -> Decimal:
	if v is None:
		return _D("0")
	return _D(str(v))


def _round(d: Decimal, places: int = 4) -> float:
	quantizer = _D("0." + "0" * places) if places > 0 else _D("1")
	return float(d.quantize(quantizer, rounding=ROUND_HALF_UP))


# ── Income Statement ──────────────────────────────────────────────────────────

def gross_profit(revenue: float, cogs: float) -> float:
	return _round(_dec(revenue) - _dec(cogs))


def gross_margin_pct(revenue: float, cogs: float) -> float | None:
	r = _dec(revenue)
	if r == 0:
		return None
	return _round((_dec(revenue) - _dec(cogs)) / r * 100)


def operating_profit(gross_profit_val: float, opex: float) -> float:
	return _round(_dec(gross_profit_val) - _dec(opex))


def operating_margin_pct(operating_profit_val: float, revenue: float) -> float | None:
	r = _dec(revenue)
	if r == 0:
		return None
	return _round(_dec(operating_profit_val) / r * 100)


def ebitda(
	operating_profit_val: float,
	depreciation: float,
	amortization: float,
) -> float:
	return _round(_dec(operating_profit_val) + _dec(depreciation) + _dec(amortization))


def ebitda_margin_pct(ebitda_val: float, revenue: float) -> float | None:
	r = _dec(revenue)
	if r == 0:
		return None
	return _round(_dec(ebitda_val) / r * 100)


def net_income(operating_profit_val: float, interest: float, tax: float) -> float:
	return _round(_dec(operating_profit_val) - _dec(interest) - _dec(tax))


def net_margin_pct(net_income_val: float, revenue: float) -> float | None:
	r = _dec(revenue)
	if r == 0:
		return None
	return _round(_dec(net_income_val) / r * 100)


def eps(net_income_val: float, shares_outstanding: float) -> float | None:
	s = _dec(shares_outstanding)
	if s == 0:
		return None
	return _round(_dec(net_income_val) / s)


# ── Balance Sheet ─────────────────────────────────────────────────────────────

def total_equity(total_assets: float, total_liabilities: float) -> float:
	return _round(_dec(total_assets) - _dec(total_liabilities))


def current_ratio(current_assets: float, current_liabilities: float) -> float | None:
	cl = _dec(current_liabilities)
	if cl == 0:
		return None
	return _round(_dec(current_assets) / cl)


def quick_ratio(
	current_assets: float, inventory: float, current_liabilities: float
) -> float | None:
	cl = _dec(current_liabilities)
	if cl == 0:
		return None
	return _round((_dec(current_assets) - _dec(inventory)) / cl)


def cash_ratio(cash: float, current_liabilities: float) -> float | None:
	cl = _dec(current_liabilities)
	if cl == 0:
		return None
	return _round(_dec(cash) / cl)


def debt_to_equity_ratio(total_debt: float, total_equity_val: float) -> float | None:
	e = _dec(total_equity_val)
	if e == 0:
		return None
	return _round(_dec(total_debt) / e)


def debt_to_assets_ratio(total_debt: float, total_assets: float) -> float | None:
	a = _dec(total_assets)
	if a == 0:
		return None
	return _round(_dec(total_debt) / a)


def equity_multiplier(total_assets: float, total_equity_val: float) -> float | None:
	e = _dec(total_equity_val)
	if e == 0:
		return None
	return _round(_dec(total_assets) / e)


# ── Profitability & Return ────────────────────────────────────────────────────

def return_on_equity(net_income_val: float, total_equity_val: float) -> float | None:
	e = _dec(total_equity_val)
	if e == 0:
		return None
	return _round(_dec(net_income_val) / e * 100)


def return_on_assets(net_income_val: float, total_assets: float) -> float | None:
	a = _dec(total_assets)
	if a == 0:
		return None
	return _round(_dec(net_income_val) / a * 100)


def return_on_invested_capital(
	nopat: float, invested_capital: float
) -> float | None:
	ic = _dec(invested_capital)
	if ic == 0:
		return None
	return _round(_dec(nopat) / ic * 100)


def asset_turnover(revenue: float, total_assets: float) -> float | None:
	a = _dec(total_assets)
	if a == 0:
		return None
	return _round(_dec(revenue) / a)


# ── Cash Flow ─────────────────────────────────────────────────────────────────

def free_cash_flow(operating_cash_flow: float, capex: float) -> float:
	return _round(_dec(operating_cash_flow) - _dec(capex))


def cash_conversion_efficiency(
	operating_cash_flow: float, net_income_val: float
) -> float | None:
	ni = _dec(net_income_val)
	if ni == 0:
		return None
	return _round(_dec(operating_cash_flow) / ni)


# ── Working Capital ───────────────────────────────────────────────────────────

def working_capital(current_assets: float, current_liabilities: float) -> float:
	return _round(_dec(current_assets) - _dec(current_liabilities))


def days_sales_outstanding(
	accounts_receivable: float, revenue: float, period_days: int = 365
) -> float | None:
	r = _dec(revenue)
	if r == 0:
		return None
	return _round(_dec(accounts_receivable) / r * _dec(period_days))


def days_inventory_outstanding(
	inventory: float, cogs: float, period_days: int = 365
) -> float | None:
	c = _dec(cogs)
	if c == 0:
		return None
	return _round(_dec(inventory) / c * _dec(period_days))


def days_payable_outstanding(
	accounts_payable: float, cogs: float, period_days: int = 365
) -> float | None:
	c = _dec(cogs)
	if c == 0:
		return None
	return _round(_dec(accounts_payable) / c * _dec(period_days))


def cash_conversion_cycle(dso: float, dio: float, dpo: float) -> float:
	return _round(_dec(dso) + _dec(dio) - _dec(dpo))


# ── Coverage ratios ───────────────────────────────────────────────────────────

def interest_coverage_ratio(ebit: float, interest_expense: float) -> float | None:
	ie = _dec(interest_expense)
	if ie == 0:
		return None
	return _round(_dec(ebit) / ie)


def debt_service_coverage(
	net_operating_income: float, total_debt_service: float
) -> float | None:
	ds = _dec(total_debt_service)
	if ds == 0:
		return None
	return _round(_dec(net_operating_income) / ds)


# ── Variance analysis ─────────────────────────────────────────────────────────

def variance(actual: float, budget: float) -> float:
	return _round(_dec(actual) - _dec(budget))


def variance_pct(actual: float, budget: float) -> float | None:
	b = _dec(budget)
	if b == 0:
		return None
	return _round((_dec(actual) - b) / abs(b) * 100)


def period_change(current: float, prior: float) -> tuple[float, float | None]:
	"""Returns (absolute_change, percent_change)."""
	abs_change = _round(_dec(current) - _dec(prior))
	p = _dec(prior)
	pct = None if p == 0 else _round((_dec(current) - p) / abs(p) * 100)
	return abs_change, pct


# ── FX translation ────────────────────────────────────────────────────────────

def fx_translate(amount: float, rate: float) -> float:
	"""Translate amount from functional currency to reporting currency."""
	if rate <= 0:
		raise ValueError(f"exchange_rate must be positive, got {rate}")
	return _round(_dec(amount) / _dec(rate))


def fx_translate_average(amount: float, avg_rate: float) -> float:
	"""Translate income statement items using average rate."""
	if avg_rate <= 0:
		raise ValueError(f"avg_rate must be positive, got {avg_rate}")
	return _round(_dec(amount) / _dec(avg_rate))


def fx_translation_adjustment(
	closing_balance_translated: float,
	opening_balance_translated: float,
	income_translated: float,
) -> float:
	"""CTA = closing_equity_translated - opening_equity_translated - income_translated."""
	return _round(
		_dec(closing_balance_translated)
		- _dec(opening_balance_translated)
		- _dec(income_translated)
	)


# ── Segment reporting ─────────────────────────────────────────────────────────

def segment_margin_pct(segment_operating_profit: float, segment_revenue: float) -> float | None:
	r = _dec(segment_revenue)
	if r == 0:
		return None
	return _round(_dec(segment_operating_profit) / r * 100)


def segment_contribution_pct(
	segment_revenue: float, total_revenue: float
) -> float | None:
	t = _dec(total_revenue)
	if t == 0:
		return None
	return _round(_dec(segment_revenue) / t * 100)


def segment_asset_intensity(segment_assets: float, segment_revenue: float) -> float | None:
	r = _dec(segment_revenue)
	if r == 0:
		return None
	return _round(_dec(segment_assets) / r)


# ── Consolidation ─────────────────────────────────────────────────────────────

def minority_interest(
	subsidiary_equity: float, parent_ownership_pct: float
) -> float:
	"""Calculate non-controlling interest equity."""
	minority_pct = _D("100") - _dec(parent_ownership_pct)
	return _round(_dec(subsidiary_equity) * minority_pct / _D("100"))


def proportional_consolidation_amount(
	subsidiary_amount: float, ownership_pct: float
) -> float:
	return _round(_dec(subsidiary_amount) * _dec(ownership_pct) / _D("100"))


def goodwill(
	purchase_price: float,
	fair_value_net_assets: float,
	ownership_pct: float,
) -> float:
	"""Goodwill on acquisition = purchase_price - (FV_net_assets × ownership_pct%)."""
	acquired_share = _dec(fair_value_net_assets) * _dec(ownership_pct) / _D("100")
	return _round(_dec(purchase_price) - acquired_share)


# ── Full KPI suite ────────────────────────────────────────────────────────────

def compute_full_kpi_suite(data: dict[str, Any]) -> dict[str, float | None]:
	"""
	Compute the complete suite of KPIs from a flat financials dict.

	Expected keys (all optional — missing treated as 0):
	    revenue, cogs, opex, depreciation, amortization, interest_expense, tax_expense,
	    total_assets, total_liabilities, current_assets, current_liabilities,
	    inventory, cash, accounts_receivable, accounts_payable,
	    total_debt, shares_outstanding, capex,
	    operating_cash_flow, net_operating_income, total_debt_service
	"""
	get = lambda k: data.get(k) or 0.0  # noqa: E731

	rev      = get("revenue")
	cogs_val = get("cogs")
	opex_val = get("opex")
	dep      = get("depreciation")
	amort    = get("amortization")
	interest = get("interest_expense")
	tax      = get("tax_expense")
	t_assets = get("total_assets")
	t_liab   = get("total_liabilities")
	cur_a    = get("current_assets")
	cur_l    = get("current_liabilities")
	inv      = get("inventory")
	cash_val = get("cash")
	ar       = get("accounts_receivable")
	ap       = get("accounts_payable")
	t_debt   = get("total_debt")
	shares   = get("shares_outstanding")
	capex    = get("capex")
	ocf      = get("operating_cash_flow")
	noi      = get("net_operating_income")
	tds      = get("total_debt_service")

	gp   = gross_profit(rev, cogs_val)
	op   = operating_profit(gp, opex_val)
	ebit = op
	ebit_da = ebitda(op, dep, amort)
	ni   = net_income(op, interest, tax)
	t_eq = total_equity(t_assets, t_liab)
	wc   = working_capital(cur_a, cur_l)
	fcf  = free_cash_flow(ocf, capex)

	dso_val = days_sales_outstanding(ar, rev)
	dio_val = days_inventory_outstanding(inv, cogs_val)
	dpo_val = days_payable_outstanding(ap, cogs_val)
	ccc     = cash_conversion_cycle(dso_val or 0, dio_val or 0, dpo_val or 0)

	return {
		"gross_profit": gp,
		"gross_margin_pct": gross_margin_pct(rev, cogs_val),
		"ebitda": ebit_da,
		"ebitda_margin_pct": ebitda_margin_pct(ebit_da, rev),
		"operating_profit": op,
		"operating_margin_pct": operating_margin_pct(op, rev),
		"net_income": ni,
		"net_margin_pct": net_margin_pct(ni, rev),
		"eps": eps(ni, shares),
		"roe": return_on_equity(ni, t_eq),
		"roa": return_on_assets(ni, t_assets),
		"asset_turnover": asset_turnover(rev, t_assets),
		"equity_multiplier": equity_multiplier(t_assets, t_eq),
		"current_ratio": current_ratio(cur_a, cur_l),
		"quick_ratio": quick_ratio(cur_a, inv, cur_l),
		"cash_ratio": cash_ratio(cash_val, cur_l),
		"working_capital": wc,
		"debt_to_equity": debt_to_equity_ratio(t_debt, t_eq),
		"debt_to_assets": debt_to_assets_ratio(t_debt, t_assets),
		"interest_coverage": interest_coverage_ratio(ebit, interest),
		"debt_service_coverage": debt_service_coverage(noi, tds),
		"free_cash_flow": fcf,
		"cash_conversion_efficiency": cash_conversion_efficiency(ocf, ni),
		"dso": dso_val,
		"dio": dio_val,
		"dpo": dpo_val,
		"cash_conversion_cycle": ccc,
	}
