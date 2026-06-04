"""
APG Budgeting & Forecasting — Financial Calculations

Pure-function library.  No I/O, no side-effects, fully type-safe.
All monetary values use Decimal for precision.

© 2025 Datacraft. Author: Nyimbi Odero <nyimbi@gmail.com>
"""

from __future__ import annotations

import math
import statistics
from decimal import ROUND_HALF_UP, Decimal
from typing import Any


_CENT = Decimal("0.01")
_ZERO = Decimal("0")


# ---------------------------------------------------------------------------
# Rounding helpers
# ---------------------------------------------------------------------------

def round_currency(value: Decimal, places: int = 2) -> Decimal:
	"""Round to *places* decimal places using ROUND_HALF_UP (banker-safe)."""
	quantize_str = Decimal("0." + "0" * places) if places > 0 else Decimal("1")
	return value.quantize(quantize_str, rounding=ROUND_HALF_UP)


def round_pct(value: Decimal, places: int = 4) -> Decimal:
	return value.quantize(Decimal("0." + "0" * places), rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Variance
# ---------------------------------------------------------------------------

def calculate_variance(budget: Decimal, actual: Decimal) -> tuple[Decimal, Decimal]:
	"""
	Return (variance_amount, variance_pct).

	Favorable: actual < budget for expense; actual > budget for revenue.
	Sign convention: positive = budget exceeded (unfavorable for expense).
	"""
	variance_amount = actual - budget
	if budget == _ZERO:
		variance_pct = _ZERO
	else:
		variance_pct = round_pct((variance_amount / budget) * Decimal("100"))
	return round_currency(variance_amount), variance_pct


def variance_type(budget: Decimal, actual: Decimal, line_type: str) -> str:
	"""
	Classify variance as 'favorable', 'unfavorable', or 'neutral'.

	For revenue lines: actual > budget → favorable.
	For expense/capital lines: actual < budget → favorable.
	"""
	diff = actual - budget
	if diff == _ZERO:
		return "neutral"
	if line_type in ("revenue",):
		return "favorable" if diff > _ZERO else "unfavorable"
	return "favorable" if diff < _ZERO else "unfavorable"


def significance_level(variance_pct: Decimal) -> str:
	"""Map a variance percentage to a significance tier."""
	abs_pct = abs(variance_pct)
	if abs_pct >= Decimal("20"):
		return "critical"
	if abs_pct >= Decimal("10"):
		return "high"
	if abs_pct >= Decimal("5"):
		return "medium"
	if abs_pct >= Decimal("2"):
		return "low"
	return "minimal"


# ---------------------------------------------------------------------------
# Distribution
# ---------------------------------------------------------------------------

def distribute_equal(total: Decimal, periods: int) -> list[Decimal]:
	"""Spread *total* evenly across *periods*, remainder in last slot."""
	assert periods > 0, "periods must be > 0"
	base = round_currency(total / Decimal(periods))
	amounts = [base] * periods
	# Adjust last period for rounding difference
	amounts[-1] = round_currency(total - sum(amounts[:-1]))
	return amounts


def distribute_seasonal(total: Decimal, weights: list[float]) -> list[Decimal]:
	"""
	Distribute *total* proportionally to *weights*.

	weights need not sum to 1; they are normalised internally.
	"""
	assert weights, "weights cannot be empty"
	w_sum = sum(weights)
	assert w_sum > 0, "weights must not all be zero"
	factors = [w / w_sum for w in weights]
	amounts = [round_currency(total * Decimal(str(f))) for f in factors]
	amounts[-1] = round_currency(total - sum(amounts[:-1]))
	return amounts


def distribute_top_down(total: Decimal, department_weights: dict[str, float]) -> dict[str, Decimal]:
	"""
	Allocate a top-level total across departments by weight.

	Returns a mapping of department_key → allocated amount.
	"""
	assert department_weights, "department_weights required"
	w_sum = sum(department_weights.values())
	assert w_sum > 0
	result: dict[str, Decimal] = {}
	allocated = _ZERO
	keys = list(department_weights.keys())
	for key in keys[:-1]:
		amt = round_currency(total * Decimal(str(department_weights[key] / w_sum)))
		result[key] = amt
		allocated += amt
	result[keys[-1]] = round_currency(total - allocated)
	return result


def distribute_zero_based(
	line_justifications: list[dict[str, Any]],
) -> list[Decimal]:
	"""
	Zero-based: each line amount is taken directly from justified amount field.

	Expects each element to have a 'justified_amount' key (Decimal or float).
	"""
	return [round_currency(Decimal(str(item["justified_amount"]))) for item in line_justifications]


# ---------------------------------------------------------------------------
# Rolling forecast
# ---------------------------------------------------------------------------

def calculate_rolling_average(values: list[Decimal], window: int) -> list[Decimal]:
	"""
	Simple moving average of *values* with given *window*.

	Returns list of same length; first (window-1) elements use expanding window.
	"""
	result: list[Decimal] = []
	for i, _ in enumerate(values):
		start = max(0, i - window + 1)
		window_vals = values[start : i + 1]
		result.append(round_currency(sum(window_vals) / Decimal(len(window_vals))))
	return result


def exponential_smoothing(values: list[Decimal], alpha: float = 0.3) -> list[Decimal]:
	"""
	Single exponential smoothing.

	alpha: smoothing factor in (0, 1).  Larger alpha → more weight on recent.
	"""
	assert 0 < alpha < 1, "alpha must be in (0, 1)"
	if not values:
		return []
	smoothed: list[Decimal] = [values[0]]
	a = Decimal(str(alpha))
	for v in values[1:]:
		s = a * v + (Decimal("1") - a) * smoothed[-1]
		smoothed.append(round_currency(s))
	return smoothed


def double_exponential_smoothing(
	values: list[Decimal], alpha: float = 0.3, beta: float = 0.1
) -> list[Decimal]:
	"""
	Holt's double exponential smoothing (captures linear trend).
	"""
	assert 0 < alpha < 1 and 0 < beta < 1
	if len(values) < 2:
		return values[:]
	a = Decimal(str(alpha))
	b = Decimal(str(beta))
	level = values[0]
	trend = values[1] - values[0]
	result: list[Decimal] = [round_currency(level + trend)]
	for v in values[1:]:
		prev_level = level
		level = a * v + (Decimal("1") - a) * (level + trend)
		trend = b * (level - prev_level) + (Decimal("1") - b) * trend
		result.append(round_currency(level + trend))
	return result


def project_rolling(
	actuals: list[Decimal], periods: int, alpha: float = 0.3
) -> list[Decimal]:
	"""
	Project *periods* future values using exponential smoothing.

	Returns only the projected values (not the historical ones).
	"""
	smoothed = exponential_smoothing(actuals, alpha)
	last = smoothed[-1]
	# Use last velocity for trend
	trend = _ZERO
	if len(smoothed) >= 2:
		trend = smoothed[-1] - smoothed[-2]
	projections: list[Decimal] = []
	for i in range(1, periods + 1):
		projections.append(round_currency(last + trend * Decimal(i)))
	return projections


# ---------------------------------------------------------------------------
# Driver-based forecasting
# ---------------------------------------------------------------------------

def driver_based_forecast(
	base_value: Decimal,
	driver_changes: dict[str, float],
	driver_elasticities: dict[str, float],
) -> Decimal:
	"""
	Apply driver elasticities to project a new value.

	forecast = base * prod(1 + elasticity_i * change_i)
	driver_changes: {driver_name: fractional_change}  e.g. {"volume": 0.05}
	driver_elasticities: {driver_name: elasticity}    e.g. {"volume": 1.2}
	"""
	multiplier = Decimal("1")
	for driver, change in driver_changes.items():
		elasticity = driver_elasticities.get(driver, 1.0)
		multiplier *= Decimal("1") + Decimal(str(elasticity * change))
	return round_currency(base_value * multiplier)


def apply_seasonal_adjustment(base: Decimal, seasonality_factors: list[float]) -> list[Decimal]:
	"""
	Spread *base* annual amount across months using seasonality indices.

	seasonality_factors: 12 multiplicative indices (normalised to sum=12 for equal weight baseline).
	"""
	assert len(seasonality_factors) == 12
	total_weight = sum(seasonality_factors)
	result: list[Decimal] = []
	allocated = _ZERO
	for i, w in enumerate(seasonality_factors):
		if i == 11:
			result.append(round_currency(base - allocated))
		else:
			amt = round_currency(base * Decimal(str(w / total_weight)))
			result.append(amt)
			allocated += amt
	return result


# ---------------------------------------------------------------------------
# Scenario / What-if analysis
# ---------------------------------------------------------------------------

def scenario_delta(base_net: Decimal, scenario_adjustments: list[Decimal]) -> Decimal:
	"""Net impact of all adjustments on the base net amount."""
	return round_currency(base_net + sum(scenario_adjustments))


def scenario_delta_pct(base_net: Decimal, scenario_net: Decimal) -> Decimal:
	if base_net == _ZERO:
		return _ZERO
	return round_pct(((scenario_net - base_net) / abs(base_net)) * Decimal("100"))


def weighted_scenario_outcome(
	outcomes: list[Decimal], probabilities: list[float]
) -> Decimal:
	"""Expected value of multiple scenarios."""
	assert len(outcomes) == len(probabilities)
	ev = sum(o * Decimal(str(p)) for o, p in zip(outcomes, probabilities))
	return round_currency(ev)


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

def sensitivity(
	base_value: Decimal,
	driver_value: Decimal,
	perturbed_value: Decimal,
	perturbed_driver: Decimal,
) -> float:
	"""
	Arc elasticity: % change in output / % change in input.
	"""
	if driver_value == _ZERO or base_value == _ZERO:
		return 0.0
	pct_output = float((perturbed_value - base_value) / abs(base_value))
	pct_input = float((perturbed_driver - driver_value) / abs(driver_value))
	if pct_input == 0.0:
		return 0.0
	return pct_output / pct_input


def sensitivity_range(
	compute_fn: Any,  # callable(driver_value: Decimal) -> Decimal
	base_driver: Decimal,
	steps: list[float],  # fractional deltas e.g. [-0.2, -0.1, 0.1, 0.2]
) -> list[dict[str, Any]]:
	"""
	Evaluate *compute_fn* at each perturbed driver value.

	Returns list of {perturbation_pct, driver_value, output_value}.
	"""
	base_output: Decimal = compute_fn(base_driver)
	results: list[dict[str, Any]] = []
	for step in steps:
		perturbed = round_currency(base_driver * (Decimal("1") + Decimal(str(step))))
		output = compute_fn(perturbed)
		results.append({
			"perturbation_pct": step * 100,
			"driver_value": perturbed,
			"output_value": output,
			"delta": round_currency(output - base_output),
			"delta_pct": float(round_pct(((output - base_output) / abs(base_output)) * Decimal("100"))) if base_output else 0.0,
		})
	return results


# ---------------------------------------------------------------------------
# Accuracy metrics
# ---------------------------------------------------------------------------

def calculate_mape(actuals: list[float], forecasts: list[float]) -> float:
	"""Mean Absolute Percentage Error (excludes zero actuals)."""
	pairs = [(a, f) for a, f in zip(actuals, forecasts) if a != 0]
	if not pairs:
		return float("nan")
	return sum(abs((a - f) / a) for a, f in pairs) / len(pairs) * 100


def calculate_rmse(actuals: list[float], forecasts: list[float]) -> float:
	"""Root Mean Squared Error."""
	if not actuals:
		return float("nan")
	mse = sum((a - f) ** 2 for a, f in zip(actuals, forecasts)) / len(actuals)
	return math.sqrt(mse)


def calculate_mae(actuals: list[float], forecasts: list[float]) -> float:
	"""Mean Absolute Error."""
	if not actuals:
		return float("nan")
	return sum(abs(a - f) for a, f in zip(actuals, forecasts)) / len(actuals)


# ---------------------------------------------------------------------------
# Consolidation
# ---------------------------------------------------------------------------

def consolidate_budgets(
	budget_totals: list[dict[str, Decimal]],
) -> dict[str, Decimal]:
	"""
	Sum revenue, expense, net across multiple budget records.

	Each element: {'revenue': Decimal, 'expense': Decimal}.
	"""
	total_rev = sum(b.get("revenue", _ZERO) for b in budget_totals)
	total_exp = sum(b.get("expense", _ZERO) for b in budget_totals)
	return {
		"total_revenue": round_currency(total_rev),
		"total_expense": round_currency(total_exp),
		"net_amount": round_currency(total_rev - total_exp),
	}


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------

def confidence_interval(
	values: list[float], confidence: float = 0.95
) -> tuple[float, float]:
	"""
	Normal-approximation confidence interval for the mean of *values*.

	Returns (lower, upper).
	"""
	import scipy.stats as stats  # type: ignore[import]  # optional dep

	n = len(values)
	if n == 0:
		return (0.0, 0.0)
	mean = statistics.mean(values)
	if n == 1:
		return (mean, mean)
	se = statistics.stdev(values) / math.sqrt(n)
	h = stats.t.ppf((1 + confidence) / 2, df=n - 1) * se
	return (mean - h, mean + h)


def bootstrap_confidence_interval(
	values: list[float],
	confidence: float = 0.95,
	iterations: int = 1000,
) -> tuple[float, float]:
	"""
	Non-parametric bootstrap CI — does not require scipy.

	Suitable when distribution is non-normal.
	"""
	import random

	if not values:
		return (0.0, 0.0)
	samples = [
		statistics.mean(random.choices(values, k=len(values)))
		for _ in range(iterations)
	]
	samples.sort()
	lo_idx = int((1 - confidence) / 2 * iterations)
	hi_idx = int((1 + confidence) / 2 * iterations) - 1
	return (samples[lo_idx], samples[hi_idx])
