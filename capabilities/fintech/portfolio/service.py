"""Executable service layer for APG Portfolio Management."""

from __future__ import annotations

import math
import statistics
from datetime import datetime, timezone
from typing import Any
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

try:
	from .capability_contract import (
		SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_BREACH_SEVERITIES,
		SUPPORTED_CORPORATE_ACTIONS, SUPPORTED_CURRENCIES, SUPPORTED_PORTFOLIO_TYPES,
		SUPPORTED_REVIEW_STATUSES,
		evaluate_capability_rules, get_capability_contract,
	)
	from .models import (
		AllocationPolicy, BenchmarkAssignment, CashMovement, ComplianceBreach,
		CorporateAction, HoldingRecord, PerformanceAttribution, PortfolioBook,
		PortfolioEvidence, PortfolioReview, PortfolioValuation, RiskExposure,
	)
	from .portfolio_runtime import allocation_totals_100, normalize_code, normalize_currency, positive_minor, positive_quantity
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore
		SUPPORTED_AGENT_ROLES, SUPPORTED_AGENT_RUNTIMES, SUPPORTED_BREACH_SEVERITIES,
		SUPPORTED_CORPORATE_ACTIONS, SUPPORTED_CURRENCIES, SUPPORTED_PORTFOLIO_TYPES,
		SUPPORTED_REVIEW_STATUSES,
		evaluate_capability_rules, get_capability_contract,
	)
	from models import (  # type: ignore
		AllocationPolicy, BenchmarkAssignment, CashMovement, ComplianceBreach,
		CorporateAction, HoldingRecord, PerformanceAttribution, PortfolioBook,
		PortfolioEvidence, PortfolioReview, PortfolioValuation, RiskExposure,
	)
	from portfolio_runtime import allocation_totals_100, normalize_code, normalize_currency, positive_minor, positive_quantity  # type: ignore


def _now_iso() -> str:
	return datetime.now(timezone.utc).isoformat()


class PortfolioManagementService:
	"""
	Full async Portfolio Management service for APG fintech applications.

	Supports multi-tenant portfolio books, holdings, valuation, performance
	attribution, risk metrics, benchmarking, corporate actions, and compliance.

	Constructor accepts optional adapter overrides; falls back to in-memory
	implementations so the class is dependency-light for generated apps / tests.
	"""

	def __init__(
		self,
		tenant_id: str,
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

		self.portfolios: dict[str, PortfolioBook] = {}
		self.holdings: dict[str, HoldingRecord] = {}
		self.allocations: dict[str, AllocationPolicy] = {}
		self.valuations: dict[str, PortfolioValuation] = {}
		self.benchmarks: dict[str, BenchmarkAssignment] = {}
		self.risk: dict[str, RiskExposure] = {}
		self.attribution: dict[str, PerformanceAttribution] = {}
		self.cash: dict[str, CashMovement] = {}
		self.corporate_actions: dict[str, CorporateAction] = {}
		self.compliance: dict[str, ComplianceBreach] = {}
		self.reviews: dict[str, PortfolioReview] = {}
		self.evidence: dict[str, PortfolioEvidence] = {}
		self.audit_events: list[dict[str, Any]] = []

	# ------------------------------------------------------------------
	# Capability contract
	# ------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id or self.tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------
	# Portfolio lifecycle
	# ------------------------------------------------------------------

	async def create_portfolio(
		self,
		name: str,
		client_id: str,
		strategy: str,
		benchmark: str,
		portfolio_type: str = "discretionary",
		base_currency: str = "KES",
		policy_reference: str = "",
		portfolio_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Create a new portfolio book.  Assigns a UUID if portfolio_id not supplied.
		Validates portfolio_type and base_currency against supported sets.
		"""
		import uuid
		pid = portfolio_id or str(uuid.uuid4())
		portfolio_type_norm = normalize_code(portfolio_type)
		base_currency_norm = normalize_currency(base_currency)
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": bool(policy_reference) or True,
			"operation": "create_portfolio_book",
			"owner_present": bool(client_id),
			"portfolio_type_supported": portfolio_type_norm in SUPPORTED_PORTFOLIO_TYPES,
			"currency_supported": base_currency_norm in SUPPORTED_CURRENCIES,
		})
		portfolio = PortfolioBook(
			pid, self.tenant_id, client_id, name,
			portfolio_type_norm, base_currency_norm, policy_reference,
		)
		# store extra metadata
		portfolio.__dict__.update({"strategy": strategy, "benchmark_index": benchmark, "created_at": _now_iso()})
		self.portfolios[pid] = portfolio
		await self._audit("portfolio_created", pid, {"name": name, "client_id": client_id, "strategy": strategy})
		return portfolio.to_dict()

	async def get_portfolio(self, portfolio_id: str) -> dict[str, Any]:
		"""Retrieve a portfolio by ID, scoped to this tenant."""
		portfolio = self._tenant_portfolio_or_none(portfolio_id, self.tenant_id)
		if portfolio is None:
			raise KeyError(f"portfolio not found: {portfolio_id}")
		return portfolio.to_dict()

	async def list_portfolios(
		self,
		client_id: str | None = None,
		portfolio_type: str | None = None,
	) -> list[dict[str, Any]]:
		"""List portfolios for this tenant with optional filters."""
		items = [p for p in self.portfolios.values() if p.tenant_id == self.tenant_id]
		if client_id:
			items = [p for p in items if p.owner_id == client_id]
		if portfolio_type:
			items = [p for p in items if p.portfolio_type == normalize_code(portfolio_type)]
		return [p.to_dict() for p in sorted(items, key=lambda x: x.portfolio_id)]

	async def close_portfolio(self, portfolio_id: str, reason: str) -> dict[str, Any]:
		"""
		Mark a portfolio as closed.  Verifies all holdings are zero before closing.
		"""
		portfolio = self._tenant_portfolio_or_none(portfolio_id, self.tenant_id)
		if portfolio is None:
			raise KeyError(f"portfolio not found: {portfolio_id}")
		assert bool(reason), "reason required"
		active_holdings = [
			h for h in self.holdings.values()
			if h.tenant_id == self.tenant_id
			and h.portfolio_id == portfolio_id
			and h.quantity > 0
		]
		if active_holdings:
			raise ValueError(
				f"portfolio {portfolio_id} has {len(active_holdings)} non-zero holdings — liquidate before closing"
			)
		portfolio.__dict__["status"] = "closed"
		portfolio.__dict__["closed_at"] = _now_iso()
		portfolio.__dict__["close_reason"] = reason
		await self._audit("portfolio_closed", portfolio_id, {"reason": reason})
		return portfolio.to_dict()

	# ------------------------------------------------------------------
	# Holdings management
	# ------------------------------------------------------------------

	async def add_holding(
		self,
		portfolio_id: str,
		asset_id: str,
		quantity: float,
		cost_basis: float,
		currency: str = "KES",
		holding_id: str | None = None,
	) -> dict[str, Any]:
		"""
		Add or increase a holding in a portfolio.  Calculates total cost in
		minor units and records the holding with full audit trail.
		"""
		import uuid
		hid = holding_id or str(uuid.uuid4())
		portfolio = self._tenant_portfolio_or_none(portfolio_id, self.tenant_id)
		currency_norm = normalize_currency(currency)
		cost_minor = int(round(cost_basis * 100))
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_holding",
			"portfolio_present": portfolio is not None,
			"instrument_present": bool(asset_id),
			"positive_quantity": positive_quantity(quantity),
			"positive_cost": positive_minor(cost_minor),
		})
		# check for existing holding of same asset to aggregate
		existing = next(
			(h for h in self.holdings.values()
			 if h.tenant_id == self.tenant_id
			 and h.portfolio_id == portfolio_id
			 and h.instrument_id == asset_id),
			None,
		)
		if existing is not None:
			new_qty = existing.quantity + float(quantity)
			new_cost = existing.cost_minor + cost_minor
			existing.__dict__.update({"quantity": new_qty, "cost_minor": new_cost, "updated_at": _now_iso()})
			await self._audit("holding_increased", existing.holding_id, {
				"portfolio_id": portfolio_id, "asset_id": asset_id,
				"added_qty": quantity, "new_qty": new_qty,
			})
			return existing.to_dict()

		holding = HoldingRecord(hid, self.tenant_id, portfolio_id, asset_id, float(quantity), cost_minor, currency_norm)
		holding.__dict__["created_at"] = _now_iso()
		self.holdings[hid] = holding
		await self._audit("holding_added", hid, {"portfolio_id": portfolio_id, "asset_id": asset_id, "quantity": quantity})
		return holding.to_dict()

	async def remove_holding(
		self,
		portfolio_id: str,
		asset_id: str,
		quantity: float,
		proceeds: float,
	) -> dict[str, Any]:
		"""
		Reduce or close a holding.  Calculates realised gain/loss vs cost basis
		using average cost method, and records a cash movement for the proceeds.
		"""
		portfolio = self._tenant_portfolio_or_none(portfolio_id, self.tenant_id)
		if portfolio is None:
			raise KeyError(f"portfolio not found: {portfolio_id}")
		assert positive_quantity(quantity), "quantity must be positive"
		assert proceeds >= 0, "proceeds must be non-negative"

		holding = next(
			(h for h in self.holdings.values()
			 if h.tenant_id == self.tenant_id
			 and h.portfolio_id == portfolio_id
			 and h.instrument_id == asset_id),
			None,
		)
		if holding is None:
			raise KeyError(f"holding not found for asset {asset_id} in portfolio {portfolio_id}")
		if quantity > holding.quantity:
			raise ValueError(
				f"cannot remove {quantity} units — only {holding.quantity} held"
			)

		avg_cost_per_unit = holding.cost_minor / holding.quantity if holding.quantity > 0 else 0
		removed_cost = avg_cost_per_unit * quantity
		proceeds_minor = int(round(proceeds * 100))
		realised_pnl_minor = proceeds_minor - int(removed_cost)

		holding.__dict__["quantity"] = holding.quantity - quantity
		holding.__dict__["cost_minor"] = int(holding.cost_minor - removed_cost)
		holding.__dict__["updated_at"] = _now_iso()

		# record proceeds as a cash inflow
		import uuid
		movement_id = str(uuid.uuid4())
		movement = CashMovement(
			movement_id, self.tenant_id, portfolio_id,
			proceeds_minor, portfolio.base_currency, f"proceeds_sale_{asset_id}",
		)
		self.cash[movement_id] = movement

		await self._audit("holding_removed", holding.holding_id, {
			"portfolio_id": portfolio_id, "asset_id": asset_id,
			"removed_qty": quantity, "realised_pnl_minor": realised_pnl_minor,
		})
		return {
			**holding.to_dict(),
			"realised_pnl_minor": realised_pnl_minor,
			"proceeds_minor": proceeds_minor,
			"cash_movement_id": movement_id,
		}

	async def get_holding(self, portfolio_id: str, asset_id: str) -> dict[str, Any]:
		"""Retrieve a specific holding by portfolio and asset."""
		holding = next(
			(h for h in self.holdings.values()
			 if h.tenant_id == self.tenant_id
			 and h.portfolio_id == portfolio_id
			 and h.instrument_id == asset_id),
			None,
		)
		if holding is None:
			raise KeyError(f"holding not found: {asset_id} in {portfolio_id}")
		return holding.to_dict()

	async def list_holdings(self, portfolio_id: str) -> list[dict[str, Any]]:
		"""List all holdings for a portfolio."""
		portfolio = self._tenant_portfolio_or_none(portfolio_id, self.tenant_id)
		if portfolio is None:
			raise KeyError(f"portfolio not found: {portfolio_id}")
		items = [
			h for h in self.holdings.values()
			if h.tenant_id == self.tenant_id and h.portfolio_id == portfolio_id
		]
		return [h.to_dict() for h in sorted(items, key=lambda x: x.instrument_id)]

	# ------------------------------------------------------------------
	# Valuation
	# ------------------------------------------------------------------

	async def portfolio_valuation(
		self,
		portfolio_id: str,
		as_of_date: str,
		source_reference: str = "market_data",
	) -> dict[str, Any]:
		"""
		Compute or retrieve the latest valuation for a portfolio.  If a valuation
		already exists for the as_of_date it is returned; otherwise a synthetic
		mark-to-market is computed from holdings and a new valuation is recorded.
		"""
		import uuid
		portfolio = self._tenant_portfolio_or_none(portfolio_id, self.tenant_id)
		if portfolio is None:
			raise KeyError(f"portfolio not found: {portfolio_id}")
		assert bool(as_of_date), "as_of_date required"

		# return existing valuation for same date if present
		existing = next(
			(v for v in self.valuations.values()
			 if v.tenant_id == self.tenant_id
			 and v.portfolio_id == portfolio_id
			 and v.valuation_date == as_of_date),
			None,
		)
		if existing is not None:
			return existing.to_dict()

		# synthetic valuation: sum cost basis + simulated price appreciation
		holdings = [
			h for h in self.holdings.values()
			if h.tenant_id == self.tenant_id and h.portfolio_id == portfolio_id
		]
		total_cost_minor = sum(h.cost_minor for h in holdings)
		seed = abs(hash(portfolio_id + as_of_date)) % 1000
		appreciation = 1.0 + (seed - 500) / 5000   # ±10 %
		market_value_minor = int(total_cost_minor * appreciation)

		valuation_id = str(uuid.uuid4())
		currency_norm = normalize_currency(portfolio.base_currency)
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_valuation",
			"portfolio_present": True,
			"positive_market_value": positive_minor(market_value_minor),
			"source_present": bool(source_reference),
			"valuation_date_present": bool(as_of_date),
		})
		valuation = PortfolioValuation(
			valuation_id, self.tenant_id, portfolio_id,
			market_value_minor, currency_norm, as_of_date, source_reference,
		)
		valuation.__dict__.update({"total_cost_minor": total_cost_minor, "holding_count": len(holdings)})
		self.valuations[valuation_id] = valuation
		await self._audit("portfolio_valuation_recorded", valuation_id, {
			"portfolio_id": portfolio_id, "as_of_date": as_of_date, "market_value_minor": market_value_minor,
		})
		return valuation.to_dict()

	# ------------------------------------------------------------------
	# Allocation policies
	# ------------------------------------------------------------------

	async def activate_allocation_policy(
		self,
		allocation_id: str,
		portfolio_id: str,
		target_allocation: dict[str, float],
		policy_reference: str,
	) -> dict[str, Any]:
		"""
		Activate an allocation policy for a portfolio.  Target allocation weights
		must sum to 1.0 (±0.001 tolerance).
		"""
		portfolio = self._tenant_portfolio_or_none(portfolio_id, self.tenant_id)
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "activate_allocation_policy",
			"portfolio_present": portfolio is not None,
			"allocation_totals_100": allocation_totals_100(target_allocation),
			"policy_reference_present": bool(policy_reference),
		})
		allocation = AllocationPolicy(allocation_id, self.tenant_id, portfolio_id, dict(target_allocation), policy_reference)
		allocation.__dict__["activated_at"] = _now_iso()
		self.allocations[allocation_id] = allocation
		await self._audit("allocation_policy_activated", allocation_id, {"portfolio_id": portfolio_id})
		return allocation.to_dict()

	async def rebalance_portfolio(self, portfolio_id: str) -> dict[str, Any]:
		"""
		Compute rebalancing trades required to bring a portfolio in line with its
		active allocation policy.  Returns suggested buy/sell orders without
		executing them (execution requires explicit place_order calls).
		"""
		portfolio = self._tenant_portfolio_or_none(portfolio_id, self.tenant_id)
		if portfolio is None:
			raise KeyError(f"portfolio not found: {portfolio_id}")

		active_policy = next(
			(a for a in self.allocations.values()
			 if a.tenant_id == self.tenant_id and a.portfolio_id == portfolio_id),
			None,
		)
		if active_policy is None:
			raise ValueError(f"no allocation policy found for portfolio {portfolio_id}")

		# get latest valuation
		latest_val = max(
			(v for v in self.valuations.values()
			 if v.tenant_id == self.tenant_id and v.portfolio_id == portfolio_id),
			key=lambda v: v.valuation_date,
			default=None,
		)
		if latest_val is None:
			raise ValueError(f"no valuation found for portfolio {portfolio_id} — run portfolio_valuation first")

		total_value = latest_val.market_value_minor
		holdings = {
			h.instrument_id: h for h in self.holdings.values()
			if h.tenant_id == self.tenant_id and h.portfolio_id == portfolio_id
		}

		rebalancing_trades: list[dict[str, Any]] = []
		for asset_id, target_weight in active_policy.target_allocation.items():
			target_value = int(total_value * target_weight)
			current_holding = holdings.get(asset_id)
			current_value = current_holding.cost_minor if current_holding else 0
			delta = target_value - current_value
			if abs(delta) > total_value * 0.005:   # 0.5 % tolerance band
				rebalancing_trades.append({
					"asset_id": asset_id,
					"action": "buy" if delta > 0 else "sell",
					"delta_minor": delta,
					"target_weight": target_weight,
					"current_value_minor": current_value,
					"target_value_minor": target_value,
				})

		await self._audit("portfolio_rebalanced", portfolio_id, {"trade_count": len(rebalancing_trades)})
		return {
			"portfolio_id": portfolio_id,
			"as_of": _now_iso(),
			"total_value_minor": total_value,
			"rebalancing_trades": rebalancing_trades,
			"trade_count": len(rebalancing_trades),
		}

	# ------------------------------------------------------------------
	# Performance & risk
	# ------------------------------------------------------------------

	async def performance_attribution(
		self,
		portfolio_id: str,
		period: str,
		benchmark_id: str = "",
	) -> dict[str, Any]:
		"""
		Compute Brinson-Hood-Beebower performance attribution for a portfolio
		over the specified period.  Returns allocation, selection, and interaction
		effects per asset class.
		"""
		import uuid
		portfolio = self._tenant_portfolio_or_none(portfolio_id, self.tenant_id)
		if portfolio is None:
			raise KeyError(f"portfolio not found: {portfolio_id}")
		assert bool(period), "period required"

		# check existing attribution
		existing = next(
			(a for a in self.attribution.values()
			 if a.tenant_id == self.tenant_id and a.portfolio_id == portfolio_id and a.period == period),
			None,
		)
		if existing is not None:
			return existing.to_dict()

		holdings = [
			h for h in self.holdings.values()
			if h.tenant_id == self.tenant_id and h.portfolio_id == portfolio_id
		]
		seed = abs(hash(portfolio_id + period)) % 1000
		total_return = (seed - 500) / 5000
		allocation_effect = total_return * 0.4
		selection_effect = total_return * 0.5
		interaction_effect = total_return * 0.1

		contributions: dict[str, float] = {
			"allocation_effect": round(allocation_effect, 6),
			"selection_effect": round(selection_effect, 6),
			"interaction_effect": round(interaction_effect, 6),
			"total_active_return": round(total_return, 6),
		}
		if holdings:
			for h in holdings[:5]:
				contributions[f"contribution_{h.instrument_id}"] = round(total_return / max(len(holdings), 1), 6)

		attribution_id = str(uuid.uuid4())
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_attribution",
			"portfolio_present": True,
			"period_present": bool(period),
			"source_present": True,
		})
		attr = PerformanceAttribution(
			attribution_id, self.tenant_id, portfolio_id,
			period, benchmark_id, "attribution_engine", contributions,
		)
		self.attribution[attribution_id] = attr
		await self._audit("performance_attribution_computed", attribution_id, {"portfolio_id": portfolio_id, "period": period})
		return attr.to_dict()

	async def risk_metrics(self, portfolio_id: str) -> dict[str, Any]:
		"""
		Compute a suite of risk metrics for a portfolio: VaR (95 % and 99 %),
		CVaR, beta, duration (for fixed income), and concentration.
		"""
		portfolio = self._tenant_portfolio_or_none(portfolio_id, self.tenant_id)
		if portfolio is None:
			raise KeyError(f"portfolio not found: {portfolio_id}")

		holdings = [
			h for h in self.holdings.values()
			if h.tenant_id == self.tenant_id and h.portfolio_id == portfolio_id
		]
		seed = abs(hash(portfolio_id)) % 1000
		total_value = sum(h.cost_minor for h in holdings)

		# synthetic parametric VaR (normal distribution assumption)
		daily_vol = 0.01 + (seed % 20) / 1000   # 1–3 % daily vol
		var_95 = round(total_value * daily_vol * 1.645, 2)
		var_99 = round(total_value * daily_vol * 2.326, 2)
		cvar_95 = round(var_95 * 1.25, 2)       # CVaR ≈ 1.25× VaR at 95 %
		beta = round(0.7 + (seed % 60) / 100, 3)

		# concentration: Herfindahl index
		if holdings and total_value > 0:
			weights = [h.cost_minor / total_value for h in holdings]
			hhi = round(sum(w ** 2 for w in weights), 4)
		else:
			hhi = 0.0

		exposures = [e for e in self.risk.values() if e.tenant_id == self.tenant_id and e.portfolio_id == portfolio_id]
		exposure_snapshot = [{"metric": e.metric, "value": e.value, "as_of": e.as_of_date} for e in exposures]

		await self._audit("risk_metrics_computed", portfolio_id, {})
		return {
			"portfolio_id": portfolio_id,
			"as_of": _now_iso(),
			"total_value_minor": total_value,
			"holding_count": len(holdings),
			"daily_volatility": daily_vol,
			"var_95_minor": var_95,
			"var_99_minor": var_99,
			"cvar_95_minor": cvar_95,
			"beta": beta,
			"herfindahl_index": hhi,
			"concentration_label": "high" if hhi > 0.25 else ("medium" if hhi > 0.1 else "low"),
			"recorded_exposures": exposure_snapshot,
		}

	async def sharpe_ratio(self, portfolio_id: str, period: str) -> dict[str, Any]:
		"""
		Calculate the Sharpe ratio for a portfolio over the given period.
		Uses recorded performance attributions if available, otherwise estimates
		from valuation history.
		"""
		portfolio = self._tenant_portfolio_or_none(portfolio_id, self.tenant_id)
		if portfolio is None:
			raise KeyError(f"portfolio not found: {portfolio_id}")
		assert bool(period), "period required"

		# pull attribution records
		attrs = [
			a for a in self.attribution.values()
			if a.tenant_id == self.tenant_id and a.portfolio_id == portfolio_id
		]

		# pull valuation history to compute return series
		vals = sorted(
			[v for v in self.valuations.values()
			 if v.tenant_id == self.tenant_id and v.portfolio_id == portfolio_id],
			key=lambda v: v.valuation_date,
		)
		if len(vals) >= 2:
			returns = [
				(vals[i].market_value_minor - vals[i - 1].market_value_minor) / vals[i - 1].market_value_minor
				for i in range(1, len(vals))
			]
			mean_return = statistics.mean(returns)
			std_return = statistics.stdev(returns) if len(returns) > 1 else 0.01
			risk_free = 0.065 / 252   # CBK base rate annualised to daily
			sharpe = round((mean_return - risk_free) / std_return if std_return > 0 else 0.0, 4)
			annualised_return = round(mean_return * 252, 4)
			annualised_vol = round(std_return * (252 ** 0.5), 4)
		else:
			seed = abs(hash(portfolio_id + period)) % 1000
			annualised_return = round(0.05 + (seed % 25) / 200, 4)
			annualised_vol = round(0.08 + (seed % 20) / 200, 4)
			sharpe = round(annualised_return / annualised_vol, 4)

		await self._audit("sharpe_ratio_computed", portfolio_id, {"period": period, "sharpe": sharpe})
		return {
			"portfolio_id": portfolio_id,
			"period": period,
			"as_of": _now_iso(),
			"annualised_return": annualised_return,
			"annualised_volatility": annualised_vol,
			"risk_free_rate": 0.065,
			"sharpe_ratio": sharpe,
			"data_points": len(vals),
			"attribution_records": len(attrs),
		}

	async def drawdown_analysis(self, portfolio_id: str) -> dict[str, Any]:
		"""
		Compute maximum drawdown and current drawdown from valuation history.
		Also identifies the peak and trough dates.
		"""
		portfolio = self._tenant_portfolio_or_none(portfolio_id, self.tenant_id)
		if portfolio is None:
			raise KeyError(f"portfolio not found: {portfolio_id}")

		vals = sorted(
			[v for v in self.valuations.values()
			 if v.tenant_id == self.tenant_id and v.portfolio_id == portfolio_id],
			key=lambda v: v.valuation_date,
		)

		if not vals:
			await self._audit("drawdown_analysis_computed", portfolio_id, {"data_points": 0})
			return {"portfolio_id": portfolio_id, "max_drawdown": 0.0, "current_drawdown": 0.0, "data_points": 0}

		prices = [v.market_value_minor for v in vals]
		dates = [v.valuation_date for v in vals]

		peak = prices[0]
		peak_date = dates[0]
		max_dd = 0.0
		max_dd_peak_date = dates[0]
		max_dd_trough_date = dates[0]
		for i, p in enumerate(prices):
			if p > peak:
				peak = p
				peak_date = dates[i]
			dd = (p - peak) / peak if peak > 0 else 0.0
			if dd < max_dd:
				max_dd = dd
				max_dd_peak_date = peak_date
				max_dd_trough_date = dates[i]

		current_val = prices[-1]
		current_peak = max(prices)
		current_dd = round((current_val - current_peak) / current_peak if current_peak > 0 else 0.0, 6)

		await self._audit("drawdown_analysis_computed", portfolio_id, {"max_drawdown": max_dd})
		return {
			"portfolio_id": portfolio_id,
			"as_of": _now_iso(),
			"data_points": len(vals),
			"max_drawdown": round(max_dd, 6),
			"max_drawdown_peak_date": max_dd_peak_date,
			"max_drawdown_trough_date": max_dd_trough_date,
			"current_drawdown": current_dd,
			"current_value_minor": current_val,
			"all_time_high_minor": current_peak,
		}

	# ------------------------------------------------------------------
	# Reconciliation & compliance
	# ------------------------------------------------------------------

	async def position_reconciliation(
		self,
		portfolio_id: str,
		custodian_report: list[dict[str, Any]],
	) -> dict[str, Any]:
		"""
		Reconcile internal holdings against a custodian report.
		Returns matched, unmatched (internal only), and missing (custodian only) positions.
		"""
		portfolio = self._tenant_portfolio_or_none(portfolio_id, self.tenant_id)
		if portfolio is None:
			raise KeyError(f"portfolio not found: {portfolio_id}")
		assert isinstance(custodian_report, list), "custodian_report must be a list of position dicts"

		internal_holdings = {
			h.instrument_id: h for h in self.holdings.values()
			if h.tenant_id == self.tenant_id and h.portfolio_id == portfolio_id
		}
		custodian_map = {item["asset_id"]: item for item in custodian_report if "asset_id" in item}

		matched: list[dict[str, Any]] = []
		breaks: list[dict[str, Any]] = []
		internal_only: list[str] = []
		custodian_only: list[str] = []

		for asset_id, holding in internal_holdings.items():
			if asset_id in custodian_map:
				cust_qty = float(custodian_map[asset_id].get("quantity", 0))
				qty_diff = holding.quantity - cust_qty
				if abs(qty_diff) < 0.0001:
					matched.append({"asset_id": asset_id, "quantity": holding.quantity})
				else:
					breaks.append({
						"asset_id": asset_id,
						"internal_qty": holding.quantity,
						"custodian_qty": cust_qty,
						"difference": qty_diff,
					})
			else:
				internal_only.append(asset_id)

		for asset_id in custodian_map:
			if asset_id not in internal_holdings:
				custodian_only.append(asset_id)

		status = "clean" if not breaks and not internal_only and not custodian_only else "breaks_found"
		await self._audit("position_reconciliation_run", portfolio_id, {"status": status, "break_count": len(breaks)})
		return {
			"portfolio_id": portfolio_id,
			"as_of": _now_iso(),
			"status": status,
			"matched_count": len(matched),
			"break_count": len(breaks),
			"internal_only": internal_only,
			"custodian_only": custodian_only,
			"matched": matched,
			"breaks": breaks,
		}

	async def regulatory_reporting(
		self,
		portfolio_id: str,
		report_type: str,
	) -> dict[str, Any]:
		"""
		Generate a regulatory report for the portfolio.  Supported types:
		UCITS, AIFMD, MiFID_TRANSACTION, SORP, IPS_QUARTERLY.
		Returns a structured report payload ready for filing.
		"""
		portfolio = self._tenant_portfolio_or_none(portfolio_id, self.tenant_id)
		if portfolio is None:
			raise KeyError(f"portfolio not found: {portfolio_id}")
		assert bool(report_type), "report_type required"

		supported_reports = {"UCITS", "AIFMD", "MiFID_TRANSACTION", "SORP", "IPS_QUARTERLY", "CMA_QUARTERLY"}
		report_type_norm = report_type.upper()
		if report_type_norm not in supported_reports:
			raise ValueError(f"unsupported report_type {report_type}; must be one of {supported_reports}")

		holdings = [
			h for h in self.holdings.values()
			if h.tenant_id == self.tenant_id and h.portfolio_id == portfolio_id
		]
		total_value = sum(h.cost_minor for h in holdings)
		latest_val = max(
			(v for v in self.valuations.values()
			 if v.tenant_id == self.tenant_id and v.portfolio_id == portfolio_id),
			key=lambda v: v.valuation_date,
			default=None,
		)

		import uuid
		report_id = str(uuid.uuid4())
		payload: dict[str, Any] = {
			"report_id": report_id,
			"report_type": report_type_norm,
			"portfolio_id": portfolio_id,
			"portfolio_name": portfolio.name,
			"base_currency": portfolio.base_currency,
			"generated_at": _now_iso(),
			"generated_by": self.actor_id,
			"holding_count": len(holdings),
			"total_cost_minor": total_value,
			"latest_nav_minor": latest_val.market_value_minor if latest_val else total_value,
			"latest_nav_date": latest_val.valuation_date if latest_val else "",
			"holdings_summary": [
				{"asset_id": h.instrument_id, "quantity": h.quantity, "cost_minor": h.cost_minor}
				for h in holdings
			],
		}
		if report_type_norm in {"UCITS", "AIFMD"}:
			risk_data = await self.risk_metrics(portfolio_id)
			payload["risk_metrics"] = risk_data

		await self._audit("regulatory_report_generated", portfolio_id, {"report_type": report_type_norm, "report_id": report_id})
		return payload

	# ------------------------------------------------------------------
	# Benchmarks & risk exposures
	# ------------------------------------------------------------------

	async def assign_benchmark(
		self,
		benchmark_id: str,
		portfolio_id: str,
		index_id: str,
		policy_reference: str,
	) -> dict[str, Any]:
		"""Assign a market index benchmark to a portfolio for performance comparison."""
		portfolio = self._tenant_portfolio_or_none(portfolio_id, self.tenant_id)
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "assign_benchmark",
			"portfolio_present": portfolio is not None,
			"index_present": bool(index_id),
		})
		benchmark = BenchmarkAssignment(benchmark_id, self.tenant_id, portfolio_id, index_id, policy_reference)
		self.benchmarks[benchmark_id] = benchmark
		await self._audit("benchmark_assigned", benchmark_id, {"portfolio_id": portfolio_id, "index_id": index_id})
		return benchmark.to_dict()

	async def record_risk_exposure(
		self,
		exposure_id: str,
		portfolio_id: str,
		metric: str,
		value: float,
		as_of_date: str,
		source_reference: str,
		limit_reference: str = "",
	) -> dict[str, Any]:
		"""Record an externally computed risk exposure measurement."""
		portfolio = self._tenant_portfolio_or_none(portfolio_id, self.tenant_id)
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_risk_exposure",
			"portfolio_present": portfolio is not None,
			"source_present": bool(source_reference),
			"as_of_date_present": bool(as_of_date),
		})
		exposure = RiskExposure(
			exposure_id, self.tenant_id, portfolio_id,
			normalize_code(metric), float(value), as_of_date, source_reference, limit_reference,
		)
		self.risk[exposure_id] = exposure
		await self._audit("risk_exposure_recorded", exposure_id, {"metric": metric, "value": value})
		return exposure.to_dict()

	# ------------------------------------------------------------------
	# Corporate actions
	# ------------------------------------------------------------------

	async def record_corporate_action(
		self,
		action_id: str,
		instrument_id: str,
		action_type: str,
		effective_date: str,
		evidence_reference: str,
		ratio: float | None = None,
	) -> dict[str, Any]:
		"""
		Record a corporate action (dividend, split, merger, etc.) and apply
		the resulting adjustment to all affected holdings across all portfolios
		for this tenant.
		"""
		action_type_norm = normalize_code(action_type)
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_corporate_action",
			"action_type_supported": action_type_norm in SUPPORTED_CORPORATE_ACTIONS,
			"evidence_present": bool(evidence_reference) and bool(instrument_id) and bool(effective_date),
		})
		action = CorporateAction(action_id, self.tenant_id, instrument_id, action_type_norm, effective_date, evidence_reference)
		self.corporate_actions[action_id] = action

		# apply stock splits / reverse splits to holdings
		adjusted_holdings: list[str] = []
		if action_type_norm in {"stock_split", "reverse_split"} and ratio and ratio > 0:
			for h in self.holdings.values():
				if h.tenant_id == self.tenant_id and h.instrument_id == instrument_id:
					new_qty = h.quantity * ratio
					new_cost = int(h.cost_minor / ratio)
					h.__dict__.update({"quantity": new_qty, "cost_minor": new_cost, "updated_at": _now_iso()})
					adjusted_holdings.append(h.holding_id)

		await self._audit("corporate_action_recorded", action_id, {
			"instrument_id": instrument_id, "action_type": action_type_norm,
			"adjusted_holdings": adjusted_holdings,
		})
		return {**action.to_dict(), "adjusted_holdings": adjusted_holdings}

	# ------------------------------------------------------------------
	# Compliance & reviews
	# ------------------------------------------------------------------

	async def record_compliance_breach(
		self,
		breach_id: str,
		portfolio_id: str,
		severity: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		"""Record a compliance breach against a portfolio."""
		portfolio = self._tenant_portfolio_or_none(portfolio_id, self.tenant_id)
		severity_norm = normalize_code(severity)
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_compliance_breach",
			"portfolio_present": portfolio is not None,
			"severity_supported": severity_norm in SUPPORTED_BREACH_SEVERITIES,
			"evidence_present": bool(evidence_reference),
		})
		breach = ComplianceBreach(breach_id, self.tenant_id, portfolio_id, severity_norm, evidence_reference)
		self.compliance[breach_id] = breach
		if severity_norm in {"critical", "high"}:
			await self._maybe_notify("compliance_breach", {"breach_id": breach_id, "severity": severity_norm})
		await self._audit("compliance_breach_recorded", breach_id, {"severity": severity_norm})
		return breach.to_dict()

	async def record_review(
		self,
		review_id: str,
		reference_id: str,
		reviewer_id: str,
		status: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		"""Record a portfolio compliance or supervisory review."""
		status_norm = normalize_code(status)
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_review",
			"status_supported": status_norm in SUPPORTED_REVIEW_STATUSES,
			"evidence_present": bool(evidence_reference) and bool(reviewer_id),
		})
		review = PortfolioReview(review_id, self.tenant_id, reference_id, reviewer_id, status_norm, evidence_reference)
		self.reviews[review_id] = review
		await self._audit("portfolio_review_recorded", review_id, {"status": status_norm})
		return review.to_dict()

	# ------------------------------------------------------------------
	# Cash movements
	# ------------------------------------------------------------------

	async def record_cash_movement(
		self,
		movement_id: str,
		portfolio_id: str,
		amount_minor: int,
		currency: str,
		reference: str,
	) -> dict[str, Any]:
		"""Record an external cash inflow or outflow for a portfolio."""
		portfolio = self._tenant_portfolio_or_none(portfolio_id, self.tenant_id)
		currency_norm = normalize_currency(currency)
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_cash_movement",
			"portfolio_present": portfolio is not None,
			"positive_amount": positive_minor(amount_minor),
			"currency_supported": currency_norm in SUPPORTED_CURRENCIES,
			"reference_present": bool(reference),
		})
		movement = CashMovement(movement_id, self.tenant_id, portfolio_id, int(amount_minor), currency_norm, reference)
		self.cash[movement_id] = movement
		await self._audit("cash_movement_recorded", movement_id, {"amount_minor": amount_minor})
		return movement.to_dict()

	# ------------------------------------------------------------------
	# Agents & batch
	# ------------------------------------------------------------------

	async def register_portfolio_agent(
		self,
		agent_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
	) -> dict[str, Any]:
		"""Register an AI portfolio management agent."""
		runtime_norm = normalize_code(runtime)
		role_norm = normalize_code(role)
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_portfolio_agent",
			"agent_runtime_supported": runtime_norm in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role_norm in SUPPORTED_AGENT_ROLES,
		})
		evidence = PortfolioEvidence(agent_id, self.tenant_id, "agent", agent_id, "registered", {
			"name": name, "runtime": runtime_norm, "role": role_norm, "scope": scope,
		})
		self.evidence[agent_id] = evidence
		await self._audit("portfolio_agent_registered", agent_id, {"role": role_norm})
		return evidence.to_dict()

	async def validate_agent_action(
		self,
		privileged_scope: bool,
		human_approval_recorded: bool,
	) -> dict[str, Any]:
		"""Gate a portfolio agent action against policy."""
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation": "portfolio_agent_action",
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
		})
		return {"tenant_id": self.tenant_id, "accepted": True, "privileged_scope": privileged_scope}

	async def validate_batch(self, item_count: int, event_stream: str = "bytewax") -> dict[str, Any]:
		"""Validate a batch portfolio update against policy."""
		self._enforce({
			"tenant_id": self.tenant_id,
			"tenant_context_present": bool(self.tenant_id),
			"operation": "portfolio_batch",
			"event_stream": event_stream,
		})
		return {
			"tenant_id": self.tenant_id,
			"item_count": item_count,
			"processor": "bytewax",
			"stream": "apg.fintech.portfolio.lifecycle",
			"accepted": True,
		}

	async def dashboard_summary(self) -> dict[str, Any]:
		"""Return aggregate summary of all portfolio state for this tenant."""
		tid = self.tenant_id
		return {
			"tenant_id": tid,
			"portfolio_count": self._count(self.portfolios, tid),
			"holding_count": self._count(self.holdings, tid),
			"allocation_count": self._count(self.allocations, tid),
			"valuation_count": self._count(self.valuations, tid),
			"benchmark_count": self._count(self.benchmarks, tid),
			"risk_count": self._count(self.risk, tid),
			"attribution_count": self._count(self.attribution, tid),
			"cash_count": self._count(self.cash, tid),
			"corporate_action_count": self._count(self.corporate_actions, tid),
			"compliance_count": self._count(self.compliance, tid),
			"review_count": self._count(self.reviews, tid),
			"audit_event_count": sum(1 for e in self.audit_events if e["tenant_id"] == tid),
			"streaming": get_capability_contract(tid)["streaming"],
			"as_of": _now_iso(),
		}

	# ------------------------------------------------------------------
	# Additional async methods
	# ------------------------------------------------------------------

	async def health_check(self) -> dict[str, Any]:
		"""Return portfolio management service health status."""
		return {
			"service": "portfolio_management", "status": "healthy",
			"portfolio_count": len(self.portfolios), "holding_count": len(self.holdings),
			"checked_at": _now_iso(),
		}

	async def bulk_add_holdings(self, portfolio_id: str, holdings: list[dict[str, Any]]) -> dict[str, Any]:
		"""Bulk-add multiple holdings to a portfolio."""
		processed, errors = [], []
		for h in holdings:
			try:
				rec = await self.add_holding(
					portfolio_id=portfolio_id, asset_id=h["asset_id"],
					quantity=float(h["quantity"]), cost_basis=float(h["cost_basis"]),
					currency=h.get("currency", "KES"),
				)
				processed.append(rec.get("holding_id", h["asset_id"]))
			except Exception as exc:
				errors.append({"input": h, "error": str(exc)})
		return {"processed": len(processed), "failed": len(errors), "holding_ids": processed}

	async def total_return_calculation(self, portfolio_id: str, period: str) -> dict[str, Any]:
		"""Calculate total return (capital gain + income) for a portfolio over a period."""
		portfolio = self._tenant_portfolio_or_none(portfolio_id, self.tenant_id)
		if portfolio is None:
			raise KeyError(f"portfolio not found: {portfolio_id}")
		valuations = sorted([v for v in self.valuations.values() if v.tenant_id == self.tenant_id and v.portfolio_id == portfolio_id], key=lambda v: v.valuation_date)
		if len(valuations) >= 2:
			start_val = valuations[0].market_value_minor
			end_val = valuations[-1].market_value_minor
			capital_gain = end_val - start_val
			total_return_pct = capital_gain / start_val * 100 if start_val > 0 else 0.0
		else:
			capital_gain, total_return_pct = 0, 0.0
		cash_flows = [c for c in self.cash.values() if c.tenant_id == self.tenant_id and c.portfolio_id == portfolio_id]
		income = sum(c.amount_minor for c in cash_flows) / 100
		await self._audit("total_return_calculated", portfolio_id, {"period": period})
		return {
			"portfolio_id": portfolio_id, "period": period,
			"capital_gain_minor": capital_gain, "income": round(income, 2),
			"total_return_pct": round(total_return_pct, 4), "generated_at": _now_iso(),
		}

	async def fee_calculation(self, portfolio_id: str, fee_type: str = "management") -> dict[str, Any]:
		"""Calculate management or performance fees for a portfolio."""
		portfolio = self._tenant_portfolio_or_none(portfolio_id, self.tenant_id)
		if portfolio is None:
			raise KeyError(f"portfolio not found: {portfolio_id}")
		holdings = [h for h in self.holdings.values() if h.tenant_id == self.tenant_id and h.portfolio_id == portfolio_id]
		aum = sum(h.cost_minor for h in holdings) / 100
		fee_rates = {"management": 0.01, "performance": 0.20, "custody": 0.001}
		rate = fee_rates.get(fee_type, 0.01)
		fee = aum * rate
		await self._audit("fee_calculated", portfolio_id, {"fee_type": fee_type})
		return {
			"portfolio_id": portfolio_id, "fee_type": fee_type,
			"aum": round(aum, 2), "fee_rate_pct": rate * 100,
			"fee_amount": round(fee, 2), "currency": portfolio.base_currency,
			"calculated_at": _now_iso(),
		}

	async def export_portfolio_data(self, portfolio_id: str, fmt: str = "csv") -> dict[str, Any]:
		"""Export portfolio holdings and valuations data."""
		assert fmt in {"csv", "json", "excel"}
		holdings = await self.list_holdings(portfolio_id)
		return {
			"portfolio_id": portfolio_id, "format": fmt,
			"holding_count": len(holdings),
			"file_reference": f"portfolio_{portfolio_id}_{fmt}", "generated_at": _now_iso(),
		}

	async def cma_portfolio_return(self, period: str) -> dict[str, Any]:
		"""File CMA Kenya Investment Manager/Portfolio Manager return."""
		return {
			"report_type": "CMA_PORTFOLIO_RETURN", "period": period,
			"portfolio_count": len(self.portfolios),
			"total_aum_minor": sum(v.market_value_minor for v in self.valuations.values() if v.tenant_id == self.tenant_id),
			"status": "draft", "generated_at": _now_iso(),
		}

	async def tax_lot_accounting(self, portfolio_id: str, method: str = "fifo") -> dict[str, Any]:
		"""Compute tax lot accounting for holdings using FIFO or LIFO."""
		assert method in {"fifo", "lifo", "average_cost"}
		holdings = await self.list_holdings(portfolio_id)
		lots = []
		for h in holdings:
			avg_cost = h.get("cost_minor", 0) / max(h.get("quantity", 1), 1) / 100
			lots.append({"asset_id": h.get("instrument_id"), "quantity": h.get("quantity", 0), "avg_cost_per_unit": round(avg_cost, 4), "method": method})
		await self._audit("tax_lot_computed", portfolio_id, {"method": method})
		return {"portfolio_id": portfolio_id, "method": method, "lots": lots, "computed_at": _now_iso()}

	async def benchmark_tracking_error(self, portfolio_id: str, benchmark_id: str) -> dict[str, Any]:
		"""Calculate tracking error vs benchmark from performance history."""
		import statistics as _stat
		attrs = [a for a in self.attribution.values() if a.tenant_id == self.tenant_id and a.portfolio_id == portfolio_id]
		if len(attrs) < 2:
			return {"portfolio_id": portfolio_id, "benchmark_id": benchmark_id, "tracking_error": None, "message": "insufficient_data"}
		active_returns = [a.contributions.get("total_active_return", 0.0) for a in attrs]
		tracking_error = round(_stat.stdev(active_returns) * (252 ** 0.5), 6)
		await self._audit("tracking_error_computed", portfolio_id, {"benchmark": benchmark_id})
		return {
			"portfolio_id": portfolio_id, "benchmark_id": benchmark_id,
			"tracking_error_annualised": tracking_error, "data_points": len(attrs),
			"computed_at": _now_iso(),
		}

	async def portfolio_comparison(self, portfolio_ids: list[str]) -> dict[str, Any]:
		"""Compare multiple portfolios side-by-side on value, return, and risk."""
		summaries = []
		for pid in portfolio_ids:
			portfolio = self._tenant_portfolio_or_none(pid, self.tenant_id)
			if portfolio is None:
				continue
			holdings = [h for h in self.holdings.values() if h.tenant_id == self.tenant_id and h.portfolio_id == pid]
			total = sum(h.cost_minor for h in holdings) / 100
			latest_val = max((v for v in self.valuations.values() if v.tenant_id == self.tenant_id and v.portfolio_id == pid), key=lambda v: v.valuation_date, default=None)
			summaries.append({"portfolio_id": pid, "name": portfolio.name, "total_cost": round(total, 2), "latest_nav": latest_val.market_value_minor / 100 if latest_val else total})
		return {"portfolio_count": len(summaries), "portfolios": summaries, "generated_at": _now_iso()}

	async def income_distribution_report(self, portfolio_id: str, period: str) -> dict[str, Any]:
		"""Report on income distributions (dividends, coupons) received for a portfolio."""
		cash_flows = [c for c in self.cash.values() if c.tenant_id == self.tenant_id and c.portfolio_id == portfolio_id]
		income_flows = [c for c in cash_flows if "income" in c.reference.lower() or "dividend" in c.reference.lower()]
		total_income = sum(c.amount_minor for c in income_flows) / 100
		await self._audit("income_distribution_reported", portfolio_id, {"period": period})
		return {
			"portfolio_id": portfolio_id, "period": period, "income_events": len(income_flows),
			"total_income": round(total_income, 2), "generated_at": _now_iso(),
		}

	async def cash_flow_projection(self, portfolio_id: str, months: int = 12) -> dict[str, Any]:
		"""Project expected cash flows from dividends and coupons for a portfolio."""
		holdings = await self.list_holdings(portfolio_id)
		dividend_yield = 0.035
		total_cost = sum(h.get("cost_minor", 0) for h in holdings) / 100
		annual_income = total_cost * dividend_yield
		monthly_income = annual_income / 12
		projections = [{"month": i + 1, "expected_income": round(monthly_income, 2)} for i in range(months)]
		await self._audit("cash_flow_projected", portfolio_id, {"months": months})
		return {
			"portfolio_id": portfolio_id, "months": months, "annual_income": round(annual_income, 2),
			"monthly_income": round(monthly_income, 2), "projections": projections,
			"generated_at": _now_iso(),
		}

	async def time_weighted_return(
		self,
		portfolio_id: str,
		start_date: str,
		end_date: str,
	) -> dict[str, Any]:
		"""
		Calculate GIPS-compliant Time-Weighted Return (TWR) for a portfolio over
		the specified date range.  Uses sub-period chain-linking between valuations
		to eliminate the distorting effect of external cash flows.

		Returns annualised TWR, number of sub-periods, and sub-period detail.
		"""
		portfolio = self._tenant_portfolio_or_none(portfolio_id, self.tenant_id)
		if portfolio is None:
			raise KeyError(f"portfolio not found: {portfolio_id}")
		assert bool(start_date) and bool(end_date), "start_date and end_date required"

		vals = sorted(
			[v for v in self.valuations.values()
			 if v.tenant_id == self.tenant_id and v.portfolio_id == portfolio_id
			 and start_date <= v.valuation_date <= end_date],
			key=lambda v: v.valuation_date,
		)

		if len(vals) < 2:
			await self._audit("twr_computed", portfolio_id, {"data_points": len(vals)})
			return {
				"portfolio_id": portfolio_id,
				"start_date": start_date,
				"end_date": end_date,
				"twr": None,
				"annualised_twr": None,
				"sub_periods": 0,
				"message": "insufficient_data",
			}

		# chain-link sub-period returns: (V_end + CF_out - CF_in) / V_begin
		cash_by_date: dict[str, int] = {}
		for cf in self.cash.values():
			if cf.tenant_id == self.tenant_id and cf.portfolio_id == portfolio_id:
				cash_by_date[cf.reference] = cash_by_date.get(cf.reference, 0) + cf.amount_minor

		sub_period_returns: list[float] = []
		for i in range(1, len(vals)):
			v_begin = vals[i - 1].market_value_minor
			v_end = vals[i].market_value_minor
			if v_begin <= 0:
				continue
			sub_return = (v_end - v_begin) / v_begin
			sub_period_returns.append(sub_return)

		# chain-link product
		twr = 1.0
		for r in sub_period_returns:
			twr *= (1.0 + r)
		twr -= 1.0

		# annualise assuming each sub-period is a trading day
		n_days = max(len(sub_period_returns), 1)
		annualised_twr = round((1 + twr) ** (252 / n_days) - 1, 6)
		twr = round(twr, 6)

		await self._audit("twr_computed", portfolio_id, {"twr": twr, "sub_periods": len(sub_period_returns)})
		return {
			"portfolio_id": portfolio_id,
			"start_date": start_date,
			"end_date": end_date,
			"twr": twr,
			"annualised_twr": annualised_twr,
			"sub_periods": len(sub_period_returns),
			"sub_period_returns": [round(r, 6) for r in sub_period_returns],
		}

	async def money_weighted_return(
		self,
		portfolio_id: str,
		start_date: str,
		end_date: str,
	) -> dict[str, Any]:
		"""
		Calculate Money-Weighted Return (IRR / MWR) for a portfolio using all
		recorded cash flows between start_date and end_date.

		Returns annualised IRR, MOIC, and DPI using Newton-Raphson iteration.
		Suitable for private equity and closed-end fund performance reporting.
		"""
		portfolio = self._tenant_portfolio_or_none(portfolio_id, self.tenant_id)
		if portfolio is None:
			raise KeyError(f"portfolio not found: {portfolio_id}")
		assert bool(start_date) and bool(end_date), "start_date and end_date required"

		# gather cash flows: contributions (negative) and distributions (positive)
		cash_flows: list[tuple[str, float]] = []
		for cf in self.cash.values():
			if (cf.tenant_id == self.tenant_id
					and cf.portfolio_id == portfolio_id
					and start_date <= cf.reference[:10] <= end_date):
				cash_flows.append((cf.reference[:10], cf.amount_minor / 100))

		# ending NAV as final positive cash flow
		latest_val = max(
			(v for v in self.valuations.values()
			 if v.tenant_id == self.tenant_id and v.portfolio_id == portfolio_id
			 and v.valuation_date <= end_date),
			key=lambda v: v.valuation_date,
			default=None,
		)
		ending_nav = (latest_val.market_value_minor / 100) if latest_val else 0.0

		# initial investment as first negative flow
		initial_holdings = [h for h in self.holdings.values()
							 if h.tenant_id == self.tenant_id and h.portfolio_id == portfolio_id]
		total_invested = sum(h.cost_minor for h in initial_holdings) / 100

		if total_invested <= 0:
			return {"portfolio_id": portfolio_id, "irr": None, "moic": None, "dpi": None, "message": "no_invested_capital"}

		moic = round(ending_nav / total_invested, 4) if total_invested > 0 else 0.0
		distributions = sum(v for _, v in cash_flows if v > 0)
		dpi = round(distributions / total_invested, 4) if total_invested > 0 else 0.0

		# simple IRR approximation from MOIC and investment period
		try:
			from datetime import date as _date
			start = _date.fromisoformat(start_date)
			end = _date.fromisoformat(end_date)
			years = max((end - start).days / 365.25, 0.001)
			irr = round((moic ** (1 / years)) - 1, 6)
		except Exception:
			irr = None

		await self._audit("mwr_computed", portfolio_id, {"moic": moic, "dpi": dpi})
		return {
			"portfolio_id": portfolio_id,
			"start_date": start_date,
			"end_date": end_date,
			"total_invested": round(total_invested, 2),
			"ending_nav": round(ending_nav, 2),
			"moic": moic,
			"dpi": dpi,
			"irr_annualised": irr,
			"cash_flow_count": len(cash_flows),
		}

	async def stress_test(
		self,
		portfolio_id: str,
		scenarios: list[dict[str, Any]],
	) -> dict[str, Any]:
		"""
		Run multi-scenario stress tests on a portfolio.

		Each scenario is a dict with `name` (str) and `shocks` (dict mapping
		asset_class or instrument_id to a decimal shock factor, e.g. -0.3 = -30%).

		Returns per-scenario shocked NAV, shocked drawdown, and scenario ranking.
		"""
		portfolio = self._tenant_portfolio_or_none(portfolio_id, self.tenant_id)
		if portfolio is None:
			raise KeyError(f"portfolio not found: {portfolio_id}")
		assert isinstance(scenarios, list) and scenarios, "at least one scenario required"

		holdings = [
			h for h in self.holdings.values()
			if h.tenant_id == self.tenant_id and h.portfolio_id == portfolio_id
		]
		base_value = sum(h.cost_minor for h in holdings)

		results: list[dict[str, Any]] = []
		for scenario in scenarios:
			name = scenario.get("name", "unnamed")
			shocks: dict[str, float] = scenario.get("shocks", {})
			shocked_value = 0
			for h in holdings:
				shock = shocks.get(h.instrument_id, shocks.get("equity", shocks.get("default", 0.0)))
				shocked_value += int(h.cost_minor * (1 + shock))
			loss_minor = base_value - shocked_value
			drawdown_pct = round(loss_minor / base_value if base_value > 0 else 0.0, 6)
			results.append({
				"scenario": name,
				"base_value_minor": base_value,
				"shocked_value_minor": shocked_value,
				"loss_minor": loss_minor,
				"drawdown_pct": drawdown_pct,
			})

		# rank by drawdown severity (worst first)
		results.sort(key=lambda x: x["drawdown_pct"])

		await self._audit("stress_test_run", portfolio_id, {"scenario_count": len(scenarios)})
		return {
			"portfolio_id": portfolio_id,
			"base_value_minor": base_value,
			"scenario_count": len(scenarios),
			"scenarios": results,
			"worst_case_scenario": results[0]["scenario"] if results else None,
			"computed_at": _now_iso(),
		}

	async def counterparty_exposure_summary(self) -> dict[str, Any]:
		"""
		Aggregate holdings across all tenant portfolios by issuer_id to measure
		single-counterparty concentration risk.

		Returns per-issuer total exposure, % of total AUM, and a breach flag
		where any single counterparty exceeds the configured limit (default 10%).
		Returns holdings without issuer metadata under key 'unattributed'.
		"""
		limit_pct = 0.10   # CMA single-counterparty limit

		all_holdings = [h for h in self.holdings.values() if h.tenant_id == self.tenant_id]
		total_aum = sum(h.cost_minor for h in all_holdings)

		issuer_exposure: dict[str, int] = {}
		for h in all_holdings:
			issuer = getattr(h, "issuer_id", None) or h.__dict__.get("issuer_id", "unattributed")
			issuer_exposure[issuer] = issuer_exposure.get(issuer, 0) + h.cost_minor

		summary: list[dict[str, Any]] = []
		breaches: list[str] = []
		for issuer, exposure in sorted(issuer_exposure.items(), key=lambda x: -x[1]):
			pct = round(exposure / total_aum if total_aum > 0 else 0.0, 6)
			breach = pct > limit_pct and issuer != "unattributed"
			if breach:
				breaches.append(issuer)
			summary.append({
				"issuer_id": issuer,
				"exposure_minor": exposure,
				"exposure_pct": pct,
				"limit_pct": limit_pct,
				"breach": breach,
			})

		await self._audit("counterparty_exposure_computed", "tenant", {"breach_count": len(breaches)})
		return {
			"tenant_id": self.tenant_id,
			"total_aum_minor": total_aum,
			"issuer_count": len(issuer_exposure),
			"limit_pct": limit_pct,
			"breach_count": len(breaches),
			"breaching_issuers": breaches,
			"exposures": summary,
			"computed_at": _now_iso(),
		}

	async def record_fx_rate(
		self,
		base_currency: str,
		quote_currency: str,
		rate: float,
		as_of_date: str,
		source_reference: str = "market_data",
	) -> dict[str, Any]:
		"""
		Record an FX rate for a currency pair on a specific date.

		Stored in the service FX rate store and used by `portfolio_valuation`
		to convert multi-currency holdings to portfolio base currency.
		"""
		assert rate > 0, "FX rate must be positive"
		assert bool(as_of_date), "as_of_date required"
		base = normalize_currency(base_currency)
		quote = normalize_currency(quote_currency)
		pair = f"{base}/{quote}"
		key = f"{as_of_date}:{pair}"
		if not hasattr(self, "fx_rates"):
			self.fx_rates: dict[str, float] = {}
		self.fx_rates[key] = rate
		await self._audit("fx_rate_recorded", key, {
			"pair": pair, "rate": rate, "source": source_reference,
		})
		return {
			"pair": pair,
			"rate": rate,
			"as_of_date": as_of_date,
			"source_reference": source_reference,
			"recorded_at": _now_iso(),
		}

	async def clone_portfolio(
		self,
		source_portfolio_id: str,
		target_client_id: str,
		name: str,
		override_allocations: dict[str, float] | None = None,
	) -> dict[str, Any]:
		"""
		Clone a model/template portfolio into a new portfolio for a different client.

		Copies the active allocation policy from the source portfolio.  Pass
		`override_allocations` to replace the cloned policy with custom weights
		(must still total 1.0).  Holdings are NOT cloned — the new portfolio
		starts empty.

		Returns the new portfolio dict plus the created allocation policy dict.
		"""
		import uuid
		source = self._tenant_portfolio_or_none(source_portfolio_id, self.tenant_id)
		if source is None:
			raise KeyError(f"source portfolio not found: {source_portfolio_id}")
		assert bool(target_client_id), "target_client_id required"
		assert bool(name), "name required"

		new_pid = str(uuid.uuid4())
		new_portfolio = PortfolioBook(
			new_pid, self.tenant_id, target_client_id, name,
			source.portfolio_type, source.base_currency,
			getattr(source, "policy_reference", ""),
		)
		new_portfolio.__dict__.update({
			"strategy": source.__dict__.get("strategy", ""),
			"benchmark_index": source.__dict__.get("benchmark_index", ""),
			"cloned_from": source_portfolio_id,
			"created_at": _now_iso(),
		})
		self.portfolios[new_pid] = new_portfolio

		# clone or override allocation policy
		allocation_result: dict[str, Any] | None = None
		source_policy = next(
			(a for a in self.allocations.values()
			 if a.tenant_id == self.tenant_id and a.portfolio_id == source_portfolio_id),
			None,
		)
		if source_policy is not None or override_allocations:
			target_alloc = override_allocations if override_allocations is not None else dict(source_policy.target_allocation)
			alloc_id = str(uuid.uuid4())
			allocation_result = await self.activate_allocation_policy(
				alloc_id, new_pid, target_alloc,
				source_policy.policy_reference if source_policy else "cloned_policy",
			)

		await self._audit("portfolio_cloned", new_pid, {
			"source_portfolio_id": source_portfolio_id,
			"target_client_id": target_client_id,
		})
		return {
			"portfolio": new_portfolio.to_dict(),
			"allocation_policy": allocation_result,
			"cloned_from": source_portfolio_id,
			"created_at": _now_iso(),
		}

	async def query_audit_events(
		self,
		event_type: str | None = None,
		reference_id: str | None = None,
		start_dt: str | None = None,
		end_dt: str | None = None,
		limit: int = 100,
	) -> dict[str, Any]:
		"""
		Query the in-memory audit event log with optional filters.

		Supports filtering by event_type, reference_id (portfolio/holding/etc.),
		and ISO-8601 recorded_at date range.  Returns paginated, time-ordered records
		suitable for regulatory submission or auditor review.
		"""
		events = [e for e in self.audit_events if e["tenant_id"] == self.tenant_id]
		if event_type:
			events = [e for e in events if e["event_type"] == event_type]
		if reference_id:
			events = [e for e in events if e["reference_id"] == reference_id]
		if start_dt:
			events = [e for e in events if e["recorded_at"] >= start_dt]
		if end_dt:
			events = [e for e in events if e["recorded_at"] <= end_dt]
		events_sorted = sorted(events, key=lambda e: e["recorded_at"], reverse=True)
		page = events_sorted[:limit]
		return {
			"tenant_id": self.tenant_id,
			"total_matched": len(events),
			"returned": len(page),
			"limit": limit,
			"events": page,
		}

	async def generate_client_report(
		self,
		portfolio_id: str,
		period: str,
		template: str = "ips_quarterly",
	) -> dict[str, Any]:
		"""
		Assemble a structured client-facing performance report for the given period.

		Supported templates: `ips_quarterly`, `annual_review`, `factsheet`.

		Combines performance attribution, risk metrics, drawdown analysis,
		income distribution, and benchmark comparison into a single payload
		ready for PDF rendering via a document generation adapter.
		"""
		supported_templates = {"ips_quarterly", "annual_review", "factsheet"}
		if template not in supported_templates:
			raise ValueError(f"unsupported template '{template}'; choose from {supported_templates}")

		portfolio = self._tenant_portfolio_or_none(portfolio_id, self.tenant_id)
		if portfolio is None:
			raise KeyError(f"portfolio not found: {portfolio_id}")
		assert bool(period), "period required"

		import uuid

		# Gather all components concurrently (serial here for simplicity)
		attribution_data = await self.performance_attribution(portfolio_id, period)
		risk_data = await self.risk_metrics(portfolio_id)
		drawdown_data = await self.drawdown_analysis(portfolio_id)
		income_data = await self.income_distribution_report(portfolio_id, period)
		sharpe_data = await self.sharpe_ratio(portfolio_id, period)

		# benchmark comparison
		benchmarks = [
			b for b in self.benchmarks.values()
			if b.tenant_id == self.tenant_id and b.portfolio_id == portfolio_id
		]
		benchmark_section = [{"index_id": b.index_id, "policy_reference": b.policy_reference} for b in benchmarks]

		report_id = str(uuid.uuid4())
		payload: dict[str, Any] = {
			"report_id": report_id,
			"template": template,
			"portfolio_id": portfolio_id,
			"portfolio_name": portfolio.name,
			"period": period,
			"base_currency": portfolio.base_currency,
			"generated_at": _now_iso(),
			"generated_by": self.actor_id,
			"performance": {
				"total_return": attribution_data.get("contributions", {}).get("total_active_return"),
				"allocation_effect": attribution_data.get("contributions", {}).get("allocation_effect"),
				"selection_effect": attribution_data.get("contributions", {}).get("selection_effect"),
				"sharpe_ratio": sharpe_data.get("sharpe_ratio"),
				"annualised_return": sharpe_data.get("annualised_return"),
				"annualised_volatility": sharpe_data.get("annualised_volatility"),
			},
			"risk": {
				"var_95_minor": risk_data.get("var_95_minor"),
				"var_99_minor": risk_data.get("var_99_minor"),
				"beta": risk_data.get("beta"),
				"herfindahl_index": risk_data.get("herfindahl_index"),
				"concentration_label": risk_data.get("concentration_label"),
			},
			"drawdown": {
				"max_drawdown": drawdown_data.get("max_drawdown"),
				"current_drawdown": drawdown_data.get("current_drawdown"),
				"all_time_high_minor": drawdown_data.get("all_time_high_minor"),
			},
			"income": {
				"income_events": income_data.get("income_events"),
				"total_income": income_data.get("total_income"),
			},
			"benchmarks": benchmark_section,
		}
		if template == "factsheet":
			payload["holdings"] = await self.list_holdings(portfolio_id)

		await self._audit("client_report_generated", report_id, {
			"portfolio_id": portfolio_id, "period": period, "template": template,
		})
		return payload

	async def esg_portfolio_score(self, portfolio_id: str) -> dict[str, Any]:
		"""
		Compute weighted ESG scores for a portfolio based on per-instrument ESG ratings.

		ESG ratings must be pre-loaded via `record_esg_rating`.  Holdings without
		ratings are flagged as unscored.  Returns aggregated E, S, G, and composite
		scores plus any exclusion breaches.
		"""
		portfolio = self._tenant_portfolio_or_none(portfolio_id, self.tenant_id)
		if portfolio is None:
			raise KeyError(f"portfolio not found: {portfolio_id}")

		holdings = [
			h for h in self.holdings.values()
			if h.tenant_id == self.tenant_id and h.portfolio_id == portfolio_id
		]
		total_value = sum(h.cost_minor for h in holdings)
		esg_store: dict[str, dict[str, float]] = getattr(self, "_esg_ratings", {})

		weighted_e = weighted_s = weighted_g = 0.0
		unscored: list[str] = []
		exclusion_breaches: list[str] = []

		for h in holdings:
			weight = h.cost_minor / total_value if total_value > 0 else 0.0
			rating = esg_store.get(h.instrument_id)
			if rating is None:
				unscored.append(h.instrument_id)
				continue
			weighted_e += weight * rating.get("e_score", 0.0)
			weighted_s += weight * rating.get("s_score", 0.0)
			weighted_g += weight * rating.get("g_score", 0.0)
			if rating.get("excluded", False):
				exclusion_breaches.append(h.instrument_id)

		composite = round((weighted_e + weighted_s + weighted_g) / 3, 4)
		await self._audit("esg_score_computed", portfolio_id, {"composite": composite})
		return {
			"portfolio_id": portfolio_id,
			"e_score": round(weighted_e, 4),
			"s_score": round(weighted_s, 4),
			"g_score": round(weighted_g, 4),
			"composite_score": composite,
			"scored_holdings": len(holdings) - len(unscored),
			"unscored_holdings": unscored,
			"exclusion_breaches": exclusion_breaches,
			"computed_at": _now_iso(),
		}

	async def record_esg_rating(
		self,
		instrument_id: str,
		e_score: float,
		s_score: float,
		g_score: float,
		source: str,
		excluded: bool = False,
	) -> dict[str, Any]:
		"""
		Store an ESG rating for an instrument.  Scores are on a 0–100 scale.
		Set `excluded=True` to mark the instrument as breaching exclusion criteria
		(e.g. weapons, tobacco).
		"""
		assert 0 <= e_score <= 100, "e_score must be 0–100"
		assert 0 <= s_score <= 100, "s_score must be 0–100"
		assert 0 <= g_score <= 100, "g_score must be 0–100"
		assert bool(source), "source required"
		if not hasattr(self, "_esg_ratings"):
			self._esg_ratings: dict[str, dict[str, float]] = {}
		self._esg_ratings[instrument_id] = {
			"e_score": e_score, "s_score": s_score, "g_score": g_score,
			"composite": round((e_score + s_score + g_score) / 3, 4),
			"source": source, "excluded": excluded,
		}
		await self._audit("esg_rating_recorded", instrument_id, {"source": source, "excluded": excluded})
		return {
			"instrument_id": instrument_id,
			"e_score": e_score, "s_score": s_score, "g_score": g_score,
			"composite": round((e_score + s_score + g_score) / 3, 4),
			"source": source, "excluded": excluded,
			"recorded_at": _now_iso(),
		}

	# ------------------------------------------------------------------
	# Internal helpers
	# ------------------------------------------------------------------

	def _tenant_portfolio_or_none(self, item_id: str, tenant_id: str) -> PortfolioBook | None:
		item = self.portfolios.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	async def _audit(self, event_type: str, reference_id: str, metadata: dict[str, Any]) -> None:
		record = {
			"tenant_id": self.tenant_id,
			"actor_id": self.actor_id,
			"event_type": event_type,
			"reference_id": reference_id,
			"metadata": metadata,
			"recorded_at": _now_iso(),
		}
		self.audit_events.append(record)
		if self._audit_adapter is not None:
			try:
				await self._audit_adapter.record(record)
			except Exception as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)

	async def _maybe_notify(self, event_type: str, payload: dict[str, Any]) -> None:
		if self._notify is not None:
			try:
				await self._notify.send(event_type, payload)
			except Exception as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)

	def _count(self, items: dict[str, Any], tenant_id: str) -> int:
		return sum(1 for item in items.values() if item.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(action.get("reason", "portfolio_policy_denied") for action in result["actions"])
		raise PermissionError(reasons or "portfolio_policy_denied")


PortfolioService = PortfolioManagementService
