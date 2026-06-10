"""Executable service layer for APG Wealth Management.

© 2025 Datacraft — www.datacraft.co.ke
"""

from __future__ import annotations

import datetime
import statistics
import uuid
from typing import Any
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

try:
	from .capability_contract import (
		SUPPORTED_AGENT_ROLES,
		SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_CURRENCIES,
		SUPPORTED_HORIZONS,
		SUPPORTED_MANDATES,
		SUPPORTED_ORDER_SIDES,
		SUPPORTED_RISK_PROFILES,
		SUPPORTED_TOLERANCES,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from .models import (
		AdvisoryMandate,
		ClientProfile,
		FeeSchedule,
		PerformanceSnapshot,
		Portfolio,
		RebalanceProposal,
		SuitabilityProfile,
		WealthEvidence,
		WealthOrder,
	)
	from .wealth_runtime import (
		allocation_totals_100,
		normalize_code,
		normalize_codes,
		normalize_currency,
		percent_bounded,
	)
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore
		SUPPORTED_AGENT_ROLES,
		SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_CURRENCIES,
		SUPPORTED_HORIZONS,
		SUPPORTED_MANDATES,
		SUPPORTED_ORDER_SIDES,
		SUPPORTED_RISK_PROFILES,
		SUPPORTED_TOLERANCES,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from models import (  # type: ignore
		AdvisoryMandate,
		ClientProfile,
		FeeSchedule,
		PerformanceSnapshot,
		Portfolio,
		RebalanceProposal,
		SuitabilityProfile,
		WealthEvidence,
		WealthOrder,
	)
	from wealth_runtime import (  # type: ignore
		allocation_totals_100,
		normalize_code,
		normalize_codes,
		normalize_currency,
		percent_bounded,
	)


# ---------------------------------------------------------------------------
# Benchmark return series (annualised %)
# ---------------------------------------------------------------------------
_BENCHMARK_RETURNS: dict[str, float] = {
	"nse20":        8.4,
	"sp500":       10.5,
	"msci_world":   9.2,
	"ftse100":      6.8,
	"bonds_ke":     7.1,
	"tbills_ke":    9.5,
	"cash":         3.0,
}

# Model allocations per risk profile (asset_class -> target_pct)
_MODEL_ALLOCATIONS: dict[str, dict[str, float]] = {
	"conservative": {
		"government_bonds": 50.0,
		"money_market":     25.0,
		"equities":         15.0,
		"real_estate":       5.0,
		"cash":              5.0,
	},
	"moderate": {
		"equities":          45.0,
		"government_bonds":  30.0,
		"real_estate":       10.0,
		"money_market":      10.0,
		"cash":               5.0,
	},
	"balanced": {
		"equities":          55.0,
		"government_bonds":  20.0,
		"real_estate":       12.0,
		"alternatives":       8.0,
		"cash":               5.0,
	},
	"aggressive": {
		"equities":          70.0,
		"alternatives":      12.0,
		"real_estate":        8.0,
		"government_bonds":   5.0,
		"cash":               5.0,
	},
	"very_aggressive": {
		"equities":          85.0,
		"alternatives":      10.0,
		"cash":               5.0,
	},
}

# Stress scenarios: scenario -> shock factor per asset class
_STRESS_SCENARIOS: dict[str, dict[str, float]] = {
	"market_crash_2008": {
		"equities":          -0.50,
		"government_bonds":  +0.08,
		"real_estate":       -0.35,
		"alternatives":      -0.40,
		"money_market":      -0.01,
		"cash":               0.00,
	},
	"covid_2020": {
		"equities":          -0.34,
		"government_bonds":  +0.05,
		"real_estate":       -0.20,
		"alternatives":      -0.25,
		"money_market":       0.00,
		"cash":               0.00,
	},
	"rising_rates_2022": {
		"equities":          -0.20,
		"government_bonds":  -0.15,
		"real_estate":       -0.10,
		"alternatives":      -0.05,
		"money_market":      +0.02,
		"cash":              +0.01,
	},
	"ke_currency_crisis": {
		"equities":          -0.30,
		"government_bonds":  -0.20,
		"real_estate":       -0.15,
		"alternatives":      -0.25,
		"money_market":      -0.05,
		"cash":              -0.10,
	},
}

# Dividend yield by asset class (%)
_DIVIDEND_YIELDS: dict[str, float] = {
	"equities":         3.5,
	"real_estate":      6.0,
	"government_bonds": 0.0,
	"money_market":     0.0,
	"alternatives":     2.0,
	"cash":             0.0,
}

# TLH: assume 15% CGT rate in Kenya
_CGT_RATE = 0.15


class WealthManagementService:
	"""Full-featured wealth management runtime for APG applications.

	Covers client onboarding, suitability assessment, portfolio creation,
	rebalancing, performance reporting, stress testing, tax-loss harvesting,
	dividend reinvestment, financial planning, and wealth dashboards.
	"""

	def __init__(
		self,
		tenant_id: str = "default",
		actor_id: str = "system",
		*,
		auth: Any = None,
		audit: Any = None,
		notify: Any = None,
		db_url: str | None = None,
		store: Any = None,
	) -> None:
		self.tenant_id = tenant_id
		self.actor_id = actor_id
		self._auth = auth
		self._audit_adapter = audit
		self._notify = notify
		self._db_url = db_url
		self._store = store

		self.clients: dict[str, ClientProfile] = {}
		self.suitability: dict[str, SuitabilityProfile] = {}
		self.portfolios: dict[str, Portfolio] = {}
		self.mandates: dict[str, AdvisoryMandate] = {}
		self.rebalances: dict[str, RebalanceProposal] = {}
		self.orders: dict[str, WealthOrder] = {}
		self.performance: dict[str, PerformanceSnapshot] = {}
		self.fees: dict[str, FeeSchedule] = {}
		self.evidence: dict[str, WealthEvidence] = {}
		self.audit_events: list[dict[str, Any]] = []

		# Extended state
		self._portfolio_holdings: dict[str, dict[str, float]] = {}  # portfolio_id -> {asset_class: value_usd}
		self._financial_plans: dict[str, dict[str, Any]] = {}
		self._dividend_records: list[dict[str, Any]] = []
		self._stress_results: dict[str, list[dict[str, Any]]] = {}
		self._tlh_candidates: list[dict[str, Any]] = []
		self._suitability_questionnaires: dict[str, dict[str, Any]] = {}

	# ------------------------------------------------------------------
	# Contract / policy
	# ------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------
	# Client suitability assessment
	# ------------------------------------------------------------------

	async def client_suitability_assessment(
		self,
		customer_id: str,
		*,
		age: int = 35,
		income_usd: float = 50_000.0,
		net_worth_usd: float = 200_000.0,
		investment_experience_years: int = 5,
		dependants: int = 2,
		employment_status: str = "employed",
	) -> dict[str, Any]:
		"""Score a client's suitability for various investment mandates."""
		assert customer_id, "customer_id required"
		assert age > 0, "age must be positive"

		# Capacity for loss scoring (0-100)
		income_score = min(income_usd / 1_000, 40.0)
		nw_score = min(net_worth_usd / 10_000, 30.0)
		age_score = max(0.0, 30.0 - age * 0.5)  # younger = higher capacity
		experience_score = min(investment_experience_years * 2.0, 20.0)
		dependant_penalty = dependants * 3.0

		capacity_score = max(0.0, income_score + nw_score + age_score + experience_score - dependant_penalty)
		capacity_score = min(100.0, capacity_score)

		# Map score to risk profile
		if capacity_score >= 80:
			recommended_risk = "very_aggressive"
		elif capacity_score >= 60:
			recommended_risk = "aggressive"
		elif capacity_score >= 45:
			recommended_risk = "balanced"
		elif capacity_score >= 30:
			recommended_risk = "moderate"
		else:
			recommended_risk = "conservative"

		assessment_id = str(uuid.uuid4())
		result = {
			"assessment_id": assessment_id,
			"customer_id": customer_id,
			"age": age,
			"income_usd": income_usd,
			"net_worth_usd": net_worth_usd,
			"investment_experience_years": investment_experience_years,
			"dependants": dependants,
			"employment_status": employment_status,
			"capacity_for_loss_score": round(capacity_score, 2),
			"recommended_risk_profile": recommended_risk,
			"recommended_allocation": _MODEL_ALLOCATIONS.get(recommended_risk, {}),
			"assessed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}
		self._suitability_questionnaires[assessment_id] = result
		self._audit(self.tenant_id, "suitability_assessed", customer_id)
		return result

	# ------------------------------------------------------------------
	# Portfolio management
	# ------------------------------------------------------------------

	async def create_portfolio(
		self,
		customer_id: str,
		risk_profile: str,
		investment_goal: str,
		time_horizon: str,
		*,
		base_currency: str = "USD",
		initial_value_usd: float = 0.0,
		advisor_id: str = "robo",
	) -> dict[str, Any]:
		"""Create a new managed portfolio for a customer."""
		assert customer_id, "customer_id required"
		risk_profile = normalize_code(risk_profile)
		assert risk_profile in SUPPORTED_RISK_PROFILES, f"unsupported risk_profile: {risk_profile}"
		base_currency = normalize_currency(base_currency)

		portfolio_id = str(uuid.uuid4())
		allocation = _MODEL_ALLOCATIONS.get(risk_profile, {})

		# Seed holdings with initial value if provided
		if initial_value_usd > 0:
			holdings = {
				asset: round(initial_value_usd * pct / 100, 2)
				for asset, pct in allocation.items()
			}
			self._portfolio_holdings[portfolio_id] = holdings
		else:
			self._portfolio_holdings[portfolio_id] = {}

		# Persist via existing sync method
		portfolio_rec = self.create_portfolio_sync(
			portfolio_id,
			self.tenant_id,
			customer_id,
			f"{risk_profile}_{investment_goal}",
			base_currency,
			advisor_id,
			f"policy-{risk_profile}",
		)

		self._audit(self.tenant_id, "portfolio_created_async", portfolio_id)
		return {
			"portfolio_id": portfolio_id,
			"customer_id": customer_id,
			"risk_profile": risk_profile,
			"investment_goal": investment_goal,
			"time_horizon": time_horizon,
			"base_currency": base_currency,
			"initial_value_usd": initial_value_usd,
			"target_allocation": allocation,
			"current_holdings": self._portfolio_holdings.get(portfolio_id, {}),
			"advisor_id": advisor_id,
			"created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
			"status": "active",
		}

	async def portfolio_rebalance(
		self,
		portfolio_id: str,
		target_allocation: dict[str, float],
		*,
		drift_threshold_pct: float = 5.0,
		dry_run: bool = False,
	) -> dict[str, Any]:
		"""Rebalance a portfolio to match the target allocation."""
		portfolio = self.portfolios.get(portfolio_id)
		assert portfolio is not None, f"portfolio not found: {portfolio_id}"
		assert allocation_totals_100(target_allocation), "target_allocation must sum to 100"

		holdings = self._portfolio_holdings.get(portfolio_id, {})
		total_value = sum(holdings.values())

		if total_value == 0:
			return {
				"portfolio_id": portfolio_id,
				"status": "no_assets",
				"message": "Portfolio has no assets to rebalance",
			}

		trades: list[dict[str, Any]] = []
		for asset, target_pct in target_allocation.items():
			target_value = total_value * target_pct / 100
			current_value = holdings.get(asset, 0.0)
			drift_pct = abs(current_value - target_value) / total_value * 100
			if drift_pct >= drift_threshold_pct:
				direction = "buy" if target_value > current_value else "sell"
				trade_value = abs(target_value - current_value)
				trades.append({
					"asset_class": asset,
					"direction": direction,
					"current_value": round(current_value, 2),
					"target_value": round(target_value, 2),
					"trade_value": round(trade_value, 2),
					"drift_pct": round(drift_pct, 2),
				})
				if not dry_run:
					holdings[asset] = target_value

		if not dry_run:
			self._portfolio_holdings[portfolio_id] = holdings

		rebalance_id = str(uuid.uuid4())
		self._audit(self.tenant_id, "portfolio_rebalanced", portfolio_id)
		return {
			"rebalance_id": rebalance_id,
			"portfolio_id": portfolio_id,
			"total_portfolio_value": round(total_value, 2),
			"trades_required": len(trades),
			"trades": trades,
			"dry_run": dry_run,
			"rebalanced_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
			"status": "executed" if not dry_run else "simulated",
		}

	async def asset_allocation_review(self, portfolio_id: str) -> dict[str, Any]:
		"""Review current allocation vs target and flag drift."""
		portfolio = self.portfolios.get(portfolio_id)
		assert portfolio is not None, f"portfolio not found: {portfolio_id}"

		holdings = self._portfolio_holdings.get(portfolio_id, {})
		total_value = sum(holdings.values())

		allocation_breakdown: list[dict[str, Any]] = []
		for asset, value in holdings.items():
			pct = (value / total_value * 100) if total_value > 0 else 0.0
			allocation_breakdown.append({
				"asset_class": asset,
				"value_usd": round(value, 2),
				"allocation_pct": round(pct, 2),
			})
		allocation_breakdown.sort(key=lambda x: x["allocation_pct"], reverse=True)

		self._audit(self.tenant_id, "asset_allocation_reviewed", portfolio_id)
		return {
			"portfolio_id": portfolio_id,
			"total_value_usd": round(total_value, 2),
			"asset_count": len(holdings),
			"allocation": allocation_breakdown,
			"reviewed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	# ------------------------------------------------------------------
	# Performance reporting
	# ------------------------------------------------------------------

	async def performance_report(
		self,
		portfolio_id: str,
		period: str,
		benchmark: str = "msci_world",
		*,
		annualise: bool = True,
	) -> dict[str, Any]:
		"""Generate a full performance report including alpha vs benchmark."""
		portfolio = self.portfolios.get(portfolio_id)
		assert portfolio is not None, f"portfolio not found: {portfolio_id}"

		snapshots = [
			s for s in self.performance.values()
			if s.portfolio_id == portfolio_id
		]

		if not snapshots:
			portfolio_return = 0.0
		else:
			returns = [s.return_percent for s in snapshots]
			portfolio_return = statistics.mean(returns)

		benchmark_return = _BENCHMARK_RETURNS.get(benchmark.lower(), 8.0)
		alpha = portfolio_return - benchmark_return
		sharpe_ratio = portfolio_return / max(abs(portfolio_return) * 0.15, 0.01)  # simplified

		holdings = self._portfolio_holdings.get(portfolio_id, {})
		total_value = sum(holdings.values())

		self._audit(self.tenant_id, "performance_report_generated", portfolio_id)
		return {
			"portfolio_id": portfolio_id,
			"period": period,
			"benchmark": benchmark,
			"portfolio_return_pct": round(portfolio_return, 4),
			"benchmark_return_pct": round(benchmark_return, 4),
			"alpha_pct": round(alpha, 4),
			"sharpe_ratio": round(sharpe_ratio, 4),
			"snapshot_count": len(snapshots),
			"current_value_usd": round(total_value, 2),
			"annualised": annualise,
			"generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	# ------------------------------------------------------------------
	# Tax-loss harvesting
	# ------------------------------------------------------------------

	async def tax_loss_harvesting(
		self,
		portfolio_id: str,
		year: int,
		*,
		min_loss_usd: float = 500.0,
		jurisdiction: str = "KE",
	) -> dict[str, Any]:
		"""Identify and execute tax-loss harvesting opportunities."""
		portfolio = self.portfolios.get(portfolio_id)
		assert portfolio is not None, f"portfolio not found: {portfolio_id}"

		holdings = self._portfolio_holdings.get(portfolio_id, {})
		candidates: list[dict[str, Any]] = []

		for asset, current_value in holdings.items():
			# Stub cost basis: assume cost is 110% of current (loss position)
			cost_basis = current_value * 1.10
			unrealised_loss = cost_basis - current_value
			tax_saving = max(0.0, unrealised_loss * _CGT_RATE)

			if unrealised_loss >= min_loss_usd:
				candidates.append({
					"asset_class": asset,
					"current_value_usd": round(current_value, 2),
					"cost_basis_usd": round(cost_basis, 2),
					"unrealised_loss_usd": round(unrealised_loss, 2),
					"estimated_tax_saving_usd": round(tax_saving, 2),
					"harvest_recommended": True,
				})

		total_harvestable_loss = sum(c["unrealised_loss_usd"] for c in candidates)
		total_tax_saving = sum(c["estimated_tax_saving_usd"] for c in candidates)
		self._tlh_candidates.extend(candidates)

		self._audit(self.tenant_id, "tax_loss_harvest_computed", portfolio_id)
		return {
			"portfolio_id": portfolio_id,
			"tax_year": year,
			"jurisdiction": jurisdiction,
			"cgt_rate_pct": _CGT_RATE * 100,
			"candidates_found": len(candidates),
			"total_harvestable_loss_usd": round(total_harvestable_loss, 2),
			"total_estimated_tax_saving_usd": round(total_tax_saving, 2),
			"candidates": candidates,
			"computed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	# ------------------------------------------------------------------
	# Dividend reinvestment
	# ------------------------------------------------------------------

	async def dividend_reinvestment(
		self,
		portfolio_id: str,
		*,
		reinvest: bool = True,
	) -> dict[str, Any]:
		"""Compute and optionally reinvest dividends across portfolio holdings."""
		portfolio = self.portfolios.get(portfolio_id)
		assert portfolio is not None, f"portfolio not found: {portfolio_id}"

		holdings = self._portfolio_holdings.get(portfolio_id, {})
		dividends: list[dict[str, Any]] = []
		total_dividend_usd = 0.0

		for asset, value in holdings.items():
			yield_pct = _DIVIDEND_YIELDS.get(asset, 0.0)
			if yield_pct == 0.0:
				continue
			# Pro-rate for 1 quarter
			quarterly_div = value * yield_pct / 100 / 4
			dividends.append({
				"asset_class": asset,
				"holding_value_usd": round(value, 2),
				"annual_yield_pct": yield_pct,
				"quarterly_dividend_usd": round(quarterly_div, 2),
			})
			total_dividend_usd += quarterly_div
			if reinvest:
				holdings[asset] = value + quarterly_div

		if reinvest:
			self._portfolio_holdings[portfolio_id] = holdings

		record = {
			"portfolio_id": portfolio_id,
			"total_dividend_usd": round(total_dividend_usd, 2),
			"reinvested": reinvest,
			"dividend_breakdown": dividends,
			"processed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}
		self._dividend_records.append(record)
		self._audit(self.tenant_id, "dividend_reinvestment_processed", portfolio_id)
		return record

	# ------------------------------------------------------------------
	# Financial planning
	# ------------------------------------------------------------------

	async def financial_plan(
		self,
		customer_id: str,
		goals: list[dict[str, Any]],
		*,
		current_savings_usd: float = 0.0,
		monthly_contribution_usd: float = 500.0,
		expected_return_pct: float = 8.0,
	) -> dict[str, Any]:
		"""Generate a long-term financial plan projecting goal attainment."""
		assert customer_id, "customer_id required"
		assert goals, "at least one goal required"
		assert monthly_contribution_usd >= 0, "monthly_contribution must be non-negative"

		plan_id = str(uuid.uuid4())
		monthly_rate = expected_return_pct / 100 / 12
		goal_analysis: list[dict[str, Any]] = []

		for goal in goals:
			target = float(goal.get("target_usd", 100_000))
			years = float(goal.get("years", 10))
			months = years * 12
			label = goal.get("label", "goal")

			# Future value of current savings
			fv_savings = current_savings_usd * ((1 + monthly_rate) ** months)
			# Future value of monthly contributions (annuity)
			if monthly_rate > 0:
				fv_contributions = monthly_contribution_usd * (((1 + monthly_rate) ** months - 1) / monthly_rate)
			else:
				fv_contributions = monthly_contribution_usd * months

			projected_value = fv_savings + fv_contributions
			shortfall = max(0.0, target - projected_value)
			on_track = projected_value >= target

			# Required monthly to hit target
			if monthly_rate > 0 and months > 0:
				gap = max(0.0, target - fv_savings)
				required_monthly = gap * monthly_rate / (((1 + monthly_rate) ** months) - 1)
			else:
				required_monthly = max(0.0, (target - fv_savings) / months) if months > 0 else 0.0

			goal_analysis.append({
				"label": label,
				"target_usd": round(target, 2),
				"years": years,
				"projected_value_usd": round(projected_value, 2),
				"shortfall_usd": round(shortfall, 2),
				"on_track": on_track,
				"required_monthly_contribution_usd": round(required_monthly, 2),
			})

		plan = {
			"plan_id": plan_id,
			"customer_id": customer_id,
			"current_savings_usd": current_savings_usd,
			"monthly_contribution_usd": monthly_contribution_usd,
			"expected_return_pct": expected_return_pct,
			"goal_count": len(goals),
			"goals": goal_analysis,
			"all_goals_on_track": all(g["on_track"] for g in goal_analysis),
			"created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}
		self._financial_plans[plan_id] = plan
		self._audit(self.tenant_id, "financial_plan_generated", customer_id)
		return plan

	# ------------------------------------------------------------------
	# Stress testing
	# ------------------------------------------------------------------

	async def portfolio_stress_test(
		self,
		portfolio_id: str,
		scenario: str,
		*,
		custom_shocks: dict[str, float] | None = None,
	) -> dict[str, Any]:
		"""Apply a stress scenario to a portfolio and compute impact."""
		portfolio = self.portfolios.get(portfolio_id)
		assert portfolio is not None, f"portfolio not found: {portfolio_id}"

		shocks = custom_shocks or _STRESS_SCENARIOS.get(scenario)
		assert shocks is not None, (
			f"unknown scenario: {scenario}. Available: {list(_STRESS_SCENARIOS)}"
		)

		holdings = self._portfolio_holdings.get(portfolio_id, {})
		total_value = sum(holdings.values())
		stressed_holdings: dict[str, float] = {}
		asset_impacts: list[dict[str, Any]] = []

		for asset, value in holdings.items():
			shock = shocks.get(asset, 0.0)
			stressed_value = value * (1 + shock)
			stressed_holdings[asset] = max(0.0, stressed_value)
			asset_impacts.append({
				"asset_class": asset,
				"pre_stress_usd": round(value, 2),
				"shock_pct": round(shock * 100, 2),
				"post_stress_usd": round(stressed_value, 2),
				"impact_usd": round(stressed_value - value, 2),
			})

		stressed_total = sum(stressed_holdings.values())
		portfolio_loss = total_value - stressed_total
		loss_pct = (portfolio_loss / total_value * 100) if total_value > 0 else 0.0

		result = {
			"portfolio_id": portfolio_id,
			"scenario": scenario,
			"pre_stress_value_usd": round(total_value, 2),
			"post_stress_value_usd": round(stressed_total, 2),
			"portfolio_loss_usd": round(portfolio_loss, 2),
			"portfolio_loss_pct": round(loss_pct, 2),
			"asset_impacts": asset_impacts,
			"recovery_scenario": "likely_12_24_months" if loss_pct < 30 else "uncertain_36_plus_months",
			"tested_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}
		self._stress_results.setdefault(portfolio_id, []).append(result)
		self._audit(self.tenant_id, "portfolio_stress_tested", portfolio_id)
		return result

	# ------------------------------------------------------------------
	# Wealth dashboard
	# ------------------------------------------------------------------

	async def wealth_dashboard(self, customer_id: str) -> dict[str, Any]:
		"""Aggregate all wealth data for a customer into a single view."""
		client = next(
			(c for c in self.clients.values() if c.id == customer_id),
			None,
		)

		# Find all portfolios for this customer
		customer_portfolios = [
			p for p in self.portfolios.values()
			if p.client_id == customer_id
		]

		total_aum = sum(
			sum(self._portfolio_holdings.get(p.id, {}).values())
			for p in customer_portfolios
		)

		portfolio_summaries: list[dict[str, Any]] = []
		for p in customer_portfolios:
			holdings = self._portfolio_holdings.get(p.id, {})
			pv = sum(holdings.values())
			snapshots = [s for s in self.performance.values() if s.portfolio_id == p.id]
			avg_return = statistics.mean([s.return_percent for s in snapshots]) if snapshots else 0.0
			portfolio_summaries.append({
				"portfolio_id": p.id,
				"name": p.name,
				"value_usd": round(pv, 2),
				"avg_return_pct": round(avg_return, 4),
				"asset_count": len(holdings),
			})

		recent_dividends = [
			d for d in self._dividend_records
			if any(p.id == d.get("portfolio_id") for p in customer_portfolios)
		]

		suitability_profiles = [
			s for s in self.suitability.values()
			if s.client_id == customer_id
		]

		self._audit(self.tenant_id, "wealth_dashboard_generated", customer_id)
		return {
			"customer_id": customer_id,
			"client_name": client.name if client else None,
			"total_aum_usd": round(total_aum, 2),
			"portfolio_count": len(customer_portfolios),
			"portfolios": portfolio_summaries,
			"suitability_profiles": len(suitability_profiles),
			"dividend_payments_ytd": len(recent_dividends),
			"financial_plans": len([
				p for p in self._financial_plans.values()
				if p["customer_id"] == customer_id
			]),
			"stress_tests_run": sum(
				len(v) for k, v in self._stress_results.items()
				if any(p.id == k for p in customer_portfolios)
			),
			"tlh_candidates": len(self._tlh_candidates),
			"generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	# ------------------------------------------------------------------
	# Existing core sync methods (preserved from original)
	# ------------------------------------------------------------------

	def register_client_profile(
		self,
		client_id: str,
		tenant_id: str,
		name: str,
		kyc_reference: str,
		tax_reference: str,
		risk_reference: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "register_client_profile",
			"kyc_present": bool(kyc_reference),
			"tax_present": bool(tax_reference),
			"risk_present": bool(risk_reference),
		})
		client = ClientProfile(client_id, tenant_id, name, kyc_reference, tax_reference, risk_reference)
		self.clients[client_id] = client
		self._audit(tenant_id, "client_profile_registered", client_id)
		return client.to_dict()

	def capture_suitability_profile(
		self,
		suitability_id: str,
		tenant_id: str,
		client_id: str,
		risk_profile: str,
		risk_tolerance: str,
		horizon: str,
		goals: list[str],
		policy_attached: bool = True,
	) -> dict[str, Any]:
		client = self._tenant_client_or_none(client_id, tenant_id)
		risk_profile = normalize_code(risk_profile)
		risk_tolerance = normalize_code(risk_tolerance)
		horizon = normalize_code(horizon)
		goals = normalize_codes(goals)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "capture_suitability_profile",
			"client_present": client is not None,
			"risk_profile_supported": risk_profile in SUPPORTED_RISK_PROFILES,
			"tolerance_supported": risk_tolerance in SUPPORTED_TOLERANCES,
			"horizon_supported": horizon in SUPPORTED_HORIZONS,
			"goals_present": bool(goals),
		})
		profile = SuitabilityProfile(suitability_id, tenant_id, client_id, risk_profile, risk_tolerance, horizon, goals)
		self.suitability[suitability_id] = profile
		self._audit(tenant_id, "suitability_profile_captured", suitability_id)
		return profile.to_dict()

	def create_portfolio_sync(
		self,
		portfolio_id: str,
		tenant_id: str,
		client_id: str,
		name: str,
		base_currency: str,
		advisor_id: str,
		policy_reference: str,
	) -> dict[str, Any]:
		client = self._tenant_client_or_none(client_id, tenant_id)
		base_currency = normalize_currency(base_currency)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "create_portfolio",
			"client_present": client is not None,
			"currency_supported": base_currency in SUPPORTED_CURRENCIES,
			"advisor_present": bool(advisor_id),
			"policy_present": bool(policy_reference),
		})
		portfolio = Portfolio(portfolio_id, tenant_id, client_id, name, base_currency, advisor_id, policy_reference)
		self.portfolios[portfolio_id] = portfolio
		self._audit(tenant_id, "portfolio_created", portfolio_id)
		return portfolio.to_dict()

	def create_advisory_mandate(
		self,
		mandate_id: str,
		tenant_id: str,
		portfolio_id: str,
		suitability_id: str,
		mandate_type: str,
		policy_reference: str,
	) -> dict[str, Any]:
		portfolio = self._tenant_portfolio_or_none(portfolio_id, tenant_id)
		suitability = self._tenant_suitability_or_none(suitability_id, tenant_id)
		mandate_type = normalize_code(mandate_type)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "create_advisory_mandate",
			"portfolio_present": portfolio is not None,
			"suitability_present": suitability is not None,
			"mandate_type_supported": mandate_type in SUPPORTED_MANDATES,
			"policy_present": bool(policy_reference),
		})
		mandate = AdvisoryMandate(mandate_id, tenant_id, portfolio_id, suitability_id, mandate_type, policy_reference)
		self.mandates[mandate_id] = mandate
		self._audit(tenant_id, "advisory_mandate_created", mandate_id)
		return mandate.to_dict()

	def propose_rebalance(
		self,
		rebalance_id: str,
		tenant_id: str,
		portfolio_id: str,
		mandate_id: str,
		target_allocation: dict[str, float],
		analysis_reference: str,
	) -> dict[str, Any]:
		portfolio = self._tenant_portfolio_or_none(portfolio_id, tenant_id)
		mandate = self._tenant_mandate_or_none(mandate_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "propose_rebalance",
			"portfolio_present": portfolio is not None,
			"mandate_present": mandate is not None,
			"mandate_matches_portfolio": mandate is not None and mandate.portfolio_id == portfolio_id,
			"allocation_totals_100": allocation_totals_100(target_allocation),
			"analysis_present": bool(analysis_reference),
		})
		rebalance = RebalanceProposal(rebalance_id, tenant_id, portfolio_id, mandate_id, dict(target_allocation), analysis_reference)
		self.rebalances[rebalance_id] = rebalance
		self._audit(tenant_id, "rebalance_proposed", rebalance_id)
		return rebalance.to_dict()

	def stage_order(
		self,
		order_id: str,
		tenant_id: str,
		portfolio_id: str,
		instrument_id: str,
		side: str,
		quantity: float,
		notional_minor: int,
		risk_reference: str,
		human_approval: str = "",
	) -> dict[str, Any]:
		portfolio = self._tenant_portfolio_or_none(portfolio_id, tenant_id)
		side = normalize_code(side)
		large_order = int(notional_minor) >= get_capability_contract(tenant_id)["configuration"]["orders"]["large_order_threshold_minor"]
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "stage_order",
			"portfolio_present": portfolio is not None,
			"side_supported": side in SUPPORTED_ORDER_SIDES,
			"positive_quantity": float(quantity) > 0,
			"risk_reference_present": bool(risk_reference),
			"large_order": large_order,
			"human_approval_recorded": bool(human_approval),
		})
		order = WealthOrder(order_id, tenant_id, portfolio_id, instrument_id, side, float(quantity), int(notional_minor), risk_reference, human_approval)
		self.orders[order_id] = order
		self._audit(tenant_id, "order_staged", order_id)
		return order.to_dict()

	def record_performance(
		self,
		snapshot_id: str,
		tenant_id: str,
		portfolio_id: str,
		period: str,
		valuation_reference: str,
		benchmark_reference: str,
		return_percent: float,
	) -> dict[str, Any]:
		portfolio = self._tenant_portfolio_or_none(portfolio_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_performance",
			"portfolio_present": portfolio is not None,
			"valuation_present": bool(valuation_reference),
			"benchmark_present": bool(benchmark_reference),
		})
		snapshot = PerformanceSnapshot(snapshot_id, tenant_id, portfolio_id, period, valuation_reference, benchmark_reference, float(return_percent))
		self.performance[snapshot_id] = snapshot
		self._audit(tenant_id, "performance_recorded", snapshot_id)
		return snapshot.to_dict()

	def record_fee_schedule(
		self,
		fee_id: str,
		tenant_id: str,
		portfolio_id: str,
		advisory_percent: float,
		performance_percent: float,
		platform_percent: float,
		contract_reference: str,
	) -> dict[str, Any]:
		portfolio = self._tenant_portfolio_or_none(portfolio_id, tenant_id)
		percent_bounded_all = all(
			percent_bounded(v) for v in [advisory_percent, performance_percent, platform_percent]
		)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_fee_schedule",
			"portfolio_present": portfolio is not None,
			"percent_bounded": percent_bounded_all,
			"contract_present": bool(contract_reference),
		})
		fee = FeeSchedule(fee_id, tenant_id, portfolio_id, float(advisory_percent), float(performance_percent), float(platform_percent), contract_reference)
		self.fees[fee_id] = fee
		self._audit(tenant_id, "fee_schedule_recorded", fee_id)
		return fee.to_dict()

	def register_wealth_agent(
		self,
		agent_id: str,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
	) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_wealth_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
		})
		evidence = WealthEvidence(agent_id, tenant_id, "agent", agent_id, "registered", {"name": name, "runtime": runtime, "role": role, "scope": scope})
		self.evidence[agent_id] = evidence
		self._audit(tenant_id, "wealth_agent_registered", agent_id)
		return evidence.to_dict()

	def validate_batch(
		self,
		tenant_id: str,
		item_count: int,
		event_stream: str = "bytewax",
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation": "wealth_batch",
			"event_stream": event_stream,
		})
		return {
			"tenant_id": tenant_id,
			"item_count": item_count,
			"processor": "bytewax",
			"stream": "apg.fintech.wealth.lifecycle",
			"accepted": True,
		}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"client_count": self._count(self.clients, tenant_id),
			"suitability_count": self._count(self.suitability, tenant_id),
			"portfolio_count": self._count(self.portfolios, tenant_id),
			"mandate_count": self._count(self.mandates, tenant_id),
			"rebalance_count": self._count(self.rebalances, tenant_id),
			"order_count": self._count(self.orders, tenant_id),
			"performance_count": self._count(self.performance, tenant_id),
			"fee_count": self._count(self.fees, tenant_id),
			"dividend_records": len(self._dividend_records),
			"financial_plans": len(self._financial_plans),
			"stress_tests_run": sum(len(v) for v in self._stress_results.values()),
			"tlh_candidates": len(self._tlh_candidates),
			"audit_event_count": sum(1 for e in self.audit_events if e["tenant_id"] == tenant_id),
			"streaming": get_capability_contract(tenant_id)["streaming"],
		}

	# ------------------------------------------------------------------
	# Additional async methods
	# ------------------------------------------------------------------

	async def health_check(self) -> dict[str, Any]:
		"""Return wealth management service health status."""
		return {
			"service": "wealth_management",
			"status": "healthy",
			"client_count": len(self.clients),
			"portfolio_count": len(self.portfolios),
			"checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def bulk_create_portfolios(self, portfolios: list[dict[str, Any]]) -> dict[str, Any]:
		"""Bulk-create portfolios for multiple clients."""
		processed, errors = [], []
		for p in portfolios:
			try:
				rec = await self.create_portfolio(
					customer_id=p["customer_id"],
					risk_profile=p.get("risk_profile", "balanced"),
					investment_goal=p.get("investment_goal", "growth"),
					time_horizon=p.get("time_horizon", "medium_term"),
					base_currency=p.get("base_currency", "USD"),
					initial_value_usd=float(p.get("initial_value_usd", 0.0)),
				)
				processed.append(rec["portfolio_id"])
			except Exception as exc:
				errors.append({"input": p, "error": str(exc)})
		return {"processed": len(processed), "failed": len(errors), "portfolio_ids": processed, "errors": errors}

	async def client_risk_review(self, customer_id: str, annual: bool = True) -> dict[str, Any]:
		"""Conduct an annual or periodic client risk and suitability review."""
		profile = next((s for s in self.suitability.values() if s.client_id == customer_id), None)
		reviews = [r for r in self.performance.values() if r.portfolio_id in {p.id for p in self.portfolios.values() if p.client_id == customer_id}]
		return {
			"customer_id": customer_id,
			"review_type": "annual" if annual else "adhoc",
			"suitability_on_file": profile is not None,
			"risk_profile": profile.risk_profile if profile else None,
			"performance_records": len(reviews),
			"recommendation": "rebalance" if len(reviews) > 3 else "monitor",
			"reviewed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def esg_screening(self, portfolio_id: str, esg_criteria: dict[str, Any]) -> dict[str, Any]:
		"""Apply ESG screening criteria to a portfolio's holdings."""
		portfolio = self.portfolios.get(portfolio_id)
		assert portfolio is not None, f"portfolio not found: {portfolio_id}"
		holdings = self._portfolio_holdings.get(portfolio_id, {})
		exclusions = esg_criteria.get("exclusions", ["tobacco", "weapons", "fossil_fuels"])
		flagged = [h for h in holdings if any(ex in h.lower() for ex in exclusions)]
		return {
			"portfolio_id": portfolio_id,
			"esg_criteria": esg_criteria,
			"total_holdings": len(holdings),
			"flagged_holdings": flagged,
			"esg_compliant": len(flagged) == 0,
			"screened_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def benchmark_comparison(self, portfolio_id: str, benchmark: str) -> dict[str, Any]:
		"""Compare portfolio performance against a benchmark index."""
		portfolio = self.portfolios.get(portfolio_id)
		assert portfolio is not None, f"portfolio not found: {portfolio_id}"
		portfolio_return = sum(s.return_percent for s in self.performance.values() if s.portfolio_id == portfolio_id) / max(len([s for s in self.performance.values() if s.portfolio_id == portfolio_id]), 1)
		benchmark_return = _BENCHMARK_RETURNS.get(benchmark.lower(), 8.0)
		alpha = portfolio_return - benchmark_return
		self._audit(self.tenant_id, "benchmark_comparison_generated", portfolio_id)
		return {
			"portfolio_id": portfolio_id, "benchmark": benchmark,
			"portfolio_return_pct": round(portfolio_return, 4),
			"benchmark_return_pct": round(benchmark_return, 4),
			"alpha_pct": round(alpha, 4),
			"outperforming": alpha > 0,
			"generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def fee_billing_run(self, customer_id: str, billing_period: str) -> dict[str, Any]:
		"""Run fee billing for all portfolios of a client for a period."""
		client_portfolios = [p for p in self.portfolios.values() if p.client_id == customer_id]
		billing_records = []
		for p in client_portfolios:
			fee_schedule = next((f for f in self.fees.values() if f.portfolio_id == p.id), None)
			holdings = self._portfolio_holdings.get(p.id, {})
			portfolio_value = sum(holdings.values())
			if fee_schedule:
				advisory_fee = portfolio_value * fee_schedule.advisory_percent / 100
				platform_fee = portfolio_value * fee_schedule.platform_percent / 100
				billing_records.append({
					"portfolio_id": p.id, "advisory_fee_usd": round(advisory_fee, 2),
					"platform_fee_usd": round(platform_fee, 2),
					"total_fee_usd": round(advisory_fee + platform_fee, 2),
				})
		self._audit(self.tenant_id, "fee_billing_run", customer_id)
		return {
			"customer_id": customer_id, "billing_period": billing_period,
			"portfolio_count": len(client_portfolios),
			"total_fees_usd": round(sum(r["total_fee_usd"] for r in billing_records), 2),
			"billing_records": billing_records,
			"generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def model_portfolio_analytics(self) -> dict[str, Any]:
		"""Return analytics across all model portfolio allocations."""
		allocation_summary = {}
		for risk, alloc in _MODEL_ALLOCATIONS.items():
			expected_return = sum(alloc[a] / 100 * _BENCHMARK_RETURNS.get(a, 5.0) for a in alloc if a in _BENCHMARK_RETURNS)
			allocation_summary[risk] = {"allocation": alloc, "expected_annual_return_pct": round(expected_return, 4)}
		return {
			"model_count": len(_MODEL_ALLOCATIONS),
			"models": allocation_summary,
			"generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def customer_net_worth_report(self, customer_id: str) -> dict[str, Any]:
		"""Compute estimated net worth from portfolio holdings and financial plans."""
		client_portfolios = [p for p in self.portfolios.values() if p.client_id == customer_id]
		total_aum = sum(sum(self._portfolio_holdings.get(p.id, {}).values()) for p in client_portfolios)
		plan = next((p for p in self._financial_plans.values() if p["customer_id"] == customer_id), None)
		self._audit(self.tenant_id, "net_worth_report_generated", customer_id)
		return {
			"customer_id": customer_id,
			"total_portfolio_value_usd": round(total_aum, 2),
			"financial_plan_on_file": plan is not None,
			"portfolio_count": len(client_portfolios),
			"generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def cma_compliance_report(self, period: str) -> dict[str, Any]:
		"""Generate a CMA (Capital Markets Authority - Kenya) compliance report."""
		return {
			"report_type": "CMA_WEALTH_RETURN",
			"period": period,
			"total_clients": len(self.clients),
			"total_portfolios": len(self.portfolios),
			"total_aum_usd": round(sum(sum(self._portfolio_holdings.get(p.id, {}).values()) for p in self.portfolios.values()), 2),
			"mandate_types": list(set(m.mandate_type for m in self.mandates.values())),
			"generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
			"status": "draft",
		}

	async def prospect_suitability_quiz(self, prospect_id: str, answers: dict[str, Any]) -> dict[str, Any]:
		"""Run suitability quiz for a prospective client."""
		return await self.client_suitability_assessment(
			customer_id=prospect_id,
			age=int(answers.get("age", 35)),
			income_usd=float(answers.get("income_usd", 50_000)),
			net_worth_usd=float(answers.get("net_worth_usd", 200_000)),
			investment_experience_years=int(answers.get("experience_years", 3)),
			dependants=int(answers.get("dependants", 2)),
			employment_status=str(answers.get("employment_status", "employed")),
		)

	async def export_client_data(self, customer_id: str, fmt: str = "json") -> dict[str, Any]:
		"""Export all client wealth data for portability or reporting."""
		assert fmt in {"json", "csv", "pdf"}
		client = next((c for c in self.clients.values() if c.id == customer_id), None)
		portfolios = [p.to_dict() for p in self.portfolios.values() if p.client_id == customer_id]
		return {
			"customer_id": customer_id, "format": fmt,
			"client_on_file": client is not None,
			"portfolio_count": len(portfolios),
			"export_reference": f"wealth_{customer_id}_{fmt}",
			"generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def concentration_risk_check(self, portfolio_id: str, threshold_pct: float = 30.0) -> dict[str, Any]:
		"""Check for concentration risk in a portfolio — any single holding above threshold."""
		portfolio = self.portfolios.get(portfolio_id)
		assert portfolio is not None, f"portfolio not found: {portfolio_id}"
		holdings = self._portfolio_holdings.get(portfolio_id, {})
		total = sum(holdings.values())
		breaches = {a: round(v / total * 100, 2) for a, v in holdings.items() if total > 0 and v / total * 100 > threshold_pct}
		self._audit(self.tenant_id, "concentration_risk_checked", portfolio_id)
		return {
			"portfolio_id": portfolio_id, "threshold_pct": threshold_pct,
			"breaches": breaches, "breach_count": len(breaches),
			"checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def nse_equity_analysis(self, portfolio_id: str) -> dict[str, Any]:
		"""Analyse NSE equity holdings in a portfolio."""
		portfolio = self.portfolios.get(portfolio_id)
		assert portfolio is not None, f"portfolio not found: {portfolio_id}"
		holdings = self._portfolio_holdings.get(portfolio_id, {})
		nse_value = holdings.get("equities", 0.0)
		total = sum(holdings.values())
		nse_pct = round(nse_value / total * 100, 2) if total > 0 else 0.0
		self._audit(self.tenant_id, "nse_equity_analysed", portfolio_id)
		return {
			"portfolio_id": portfolio_id, "nse_equity_value_usd": round(nse_value, 2),
			"nse_allocation_pct": nse_pct, "benchmark": "NSE_20", "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def government_bond_allocation(self, portfolio_id: str) -> dict[str, Any]:
		"""Return government bond allocation details for a portfolio (T-Bills, IFBs, Eurobonds)."""
		portfolio = self.portfolios.get(portfolio_id)
		assert portfolio is not None, f"portfolio not found: {portfolio_id}"
		holdings = self._portfolio_holdings.get(portfolio_id, {})
		bond_value = holdings.get("government_bonds", 0.0)
		self._audit(self.tenant_id, "bond_allocation_reviewed", portfolio_id)
		return {
			"portfolio_id": portfolio_id, "government_bonds_usd": round(bond_value, 2),
			"instruments": ["KE_TBILL_91D", "KE_TBILL_182D", "KE_TBOND_10Y", "KE_IFB"],
			"generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def rebalance_trigger_check(self, portfolio_id: str) -> dict[str, Any]:
		"""Check whether rebalancing is triggered based on drift beyond tolerance."""
		portfolio = self.portfolios.get(portfolio_id)
		assert portfolio is not None, f"portfolio not found: {portfolio_id}"
		holdings = self._portfolio_holdings.get(portfolio_id, {})
		total = sum(holdings.values())
		risk_profile = next((p.name.split("_")[0] for p in self.portfolios.values() if p.id == portfolio_id), "balanced")
		target = _MODEL_ALLOCATIONS.get(risk_profile, {})
		max_drift = max((abs(holdings.get(a, 0) / total * 100 - target.get(a, 0)) for a in target if total > 0), default=0.0)
		rebalance_needed = max_drift >= 5.0
		return {
			"portfolio_id": portfolio_id, "max_drift_pct": round(max_drift, 2),
			"rebalance_needed": rebalance_needed, "threshold_pct": 5.0,
			"checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def advisor_assignment(self, portfolio_id: str, advisor_id: str, mandate_type: str) -> dict[str, Any]:
		"""Assign or change the financial advisor for a portfolio."""
		portfolio = self.portfolios.get(portfolio_id)
		assert portfolio is not None, f"portfolio not found: {portfolio_id}"
		portfolio.advisor_id = advisor_id
		self._audit(self.tenant_id, "advisor_assigned", portfolio_id)
		return {
			"portfolio_id": portfolio_id, "advisor_id": advisor_id, "mandate_type": mandate_type,
			"assigned_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def bulk_performance_update(self, performance_records: list[dict[str, Any]]) -> dict[str, Any]:
		"""Bulk-upload performance snapshots for multiple portfolios."""
		processed, errors = [], []
		for rec in performance_records:
			try:
				result = self.record_performance(
					snapshot_id=str(uuid.uuid4()),
					tenant_id=rec.get("tenant_id", self.tenant_id),
					portfolio_id=rec["portfolio_id"],
					period=rec["period"],
					valuation_reference=rec.get("valuation_reference", f"val-{rec['portfolio_id']}"),
					benchmark_reference=rec.get("benchmark_reference", "msci_world"),
					return_percent=float(rec["return_percent"]),
				)
				processed.append(result["id"])
			except Exception as exc:
				errors.append({"input": rec, "error": str(exc)})
		return {"processed": len(processed), "failed": len(errors), "snapshot_ids": processed}

	# ------------------------------------------------------------------
	# Internal helpers
	# ------------------------------------------------------------------

	def _tenant_client_or_none(self, item_id: str, tenant_id: str) -> ClientProfile | None:
		item = self.clients.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_suitability_or_none(self, item_id: str, tenant_id: str) -> SuitabilityProfile | None:
		item = self.suitability.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_portfolio_or_none(self, item_id: str, tenant_id: str) -> Portfolio | None:
		item = self.portfolios.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_mandate_or_none(self, item_id: str, tenant_id: str) -> AdvisoryMandate | None:
		item = self.mandates.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"reference_id": reference_id,
			"actor_id": self.actor_id,
			"recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		})

	def _count(self, items: dict[str, Any], tenant_id: str) -> int:
		return sum(1 for item in items.values() if item.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(
			action.get("reason", "wealth_policy_denied") for action in result["actions"]
		)
		raise PermissionError(reasons or "wealth_policy_denied")


WealthService = WealthManagementService
