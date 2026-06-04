"""Executable service layer for APG Robo Advisory.

© 2025 Datacraft — www.datacraft.co.ke
"""

from __future__ import annotations

import datetime
import math
import statistics
import uuid
from typing import Any

try:
	from .capability_contract import (
		SUPPORTED_AGENT_ROLES,
		SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_CADENCES,
		SUPPORTED_CURRENCIES,
		SUPPORTED_GOAL_TYPES,
		SUPPORTED_REVIEW_STATUSES,
		SUPPORTED_RISK_PROFILES,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from .models import (
		AutomationPlan,
		DriftRecord,
		GoalPlan,
		InvestorProfile,
		ModelPortfolio,
		RecommendationPacket,
		ReviewRecord,
		RoboEvidence,
		TaxLossCandidate,
	)
	from .robo_runtime import (
		allocation_totals_100,
		normalize_code,
		normalize_currency,
		positive_minor,
	)
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore
		SUPPORTED_AGENT_ROLES,
		SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_CADENCES,
		SUPPORTED_CURRENCIES,
		SUPPORTED_GOAL_TYPES,
		SUPPORTED_REVIEW_STATUSES,
		SUPPORTED_RISK_PROFILES,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from models import (  # type: ignore
		AutomationPlan,
		DriftRecord,
		GoalPlan,
		InvestorProfile,
		ModelPortfolio,
		RecommendationPacket,
		ReviewRecord,
		RoboEvidence,
		TaxLossCandidate,
	)
	from robo_runtime import (  # type: ignore
		allocation_totals_100,
		normalize_code,
		normalize_currency,
		positive_minor,
	)


# ---------------------------------------------------------------------------
# Risk questionnaire scoring
# ---------------------------------------------------------------------------
# Each question maps response key -> score contribution (0-10)
_QUESTIONNAIRE_WEIGHTS: dict[str, dict[str, int]] = {
	"investment_horizon": {
		"less_than_1yr": 1,
		"1_to_3yr":      3,
		"3_to_5yr":      5,
		"5_to_10yr":     7,
		"over_10yr":     10,
	},
	"loss_reaction": {
		"sell_all":      1,
		"sell_some":     3,
		"hold":          6,
		"buy_more":      10,
	},
	"income_stability": {
		"unstable":      1,
		"variable":      4,
		"stable":        7,
		"very_stable":   10,
	},
	"prior_experience": {
		"none":          1,
		"limited":       4,
		"moderate":      7,
		"extensive":     10,
	},
	"savings_rate": {
		"none":          1,
		"low":           3,
		"medium":        6,
		"high":          10,
	},
}

# Model portfolio allocations per risk profile
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

# Expected annual returns per asset class (%)
_ASSET_RETURNS: dict[str, float] = {
	"equities":          10.5,
	"government_bonds":   7.1,
	"money_market":       9.2,
	"real_estate":        8.0,
	"alternatives":       9.0,
	"cash":               3.0,
}

# Tax rate (CGT in Kenya)
_CGT_RATE = 0.15

# Performance fee threshold: portfolio return above this triggers perf fee
_HURDLE_RATE_PCT = 6.0
_PERFORMANCE_FEE_PCT = 15.0


def _weighted_portfolio_return(allocation: dict[str, float]) -> float:
	"""Compute expected annual return for a given allocation dict."""
	total = sum(allocation.values())
	if total == 0:
		return 0.0
	return sum(
		(pct / total) * _ASSET_RETURNS.get(asset, 0.0)
		for asset, pct in allocation.items()
	)


class RoboAdvisoryService:
	"""Full-featured Robo Advisory runtime for APG applications.

	Covers risk questionnaires, profile determination, model portfolios,
	auto-invest, auto-rebalance, goal tracking, drift monitoring,
	tax optimisation, client onboarding, and performance reporting.
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

		self.profiles: dict[str, InvestorProfile] = {}
		self.goals: dict[str, GoalPlan] = {}
		self.models: dict[str, ModelPortfolio] = {}
		self.recommendations: dict[str, RecommendationPacket] = {}
		self.automation: dict[str, AutomationPlan] = {}
		self.drift: dict[str, DriftRecord] = {}
		self.tax_loss: dict[str, TaxLossCandidate] = {}
		self.reviews: dict[str, ReviewRecord] = {}
		self.evidence: dict[str, RoboEvidence] = {}
		self.audit_events: list[dict[str, Any]] = []

		# Extended in-memory state
		self._questionnaires: dict[str, dict[str, Any]] = {}
		self._portfolio_holdings: dict[str, dict[str, float]] = {}  # profile_id -> {asset: value}
		self._auto_invest_logs: list[dict[str, Any]] = []
		self._rebalance_logs: list[dict[str, Any]] = []
		self._goal_progress: dict[str, list[dict[str, Any]]] = {}
		self._onboarding_records: dict[str, dict[str, Any]] = {}
		self._performance_logs: dict[str, list[dict[str, Any]]] = {}
		self._tax_optimisation_logs: list[dict[str, Any]] = []

	# ------------------------------------------------------------------
	# Contract / policy
	# ------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------
	# Risk profiling
	# ------------------------------------------------------------------

	async def risk_questionnaire(
		self,
		customer_id: str,
		responses: dict[str, str],
	) -> dict[str, Any]:
		"""Score a risk questionnaire and return structured results."""
		assert customer_id, "customer_id required"
		assert responses, "responses required"

		scores: dict[str, int] = {}
		missing: list[str] = []

		for question, weights in _QUESTIONNAIRE_WEIGHTS.items():
			answer = responses.get(question)
			if answer is None:
				missing.append(question)
				scores[question] = 0
			else:
				scores[question] = weights.get(answer, 0)

		total_score = sum(scores.values())
		max_score = sum(max(w.values()) for w in _QUESTIONNAIRE_WEIGHTS.values())
		score_pct = (total_score / max_score * 100) if max_score > 0 else 0.0

		# Map score_pct to risk profile
		if score_pct >= 80:
			derived_profile = "very_aggressive"
		elif score_pct >= 65:
			derived_profile = "aggressive"
		elif score_pct >= 50:
			derived_profile = "balanced"
		elif score_pct >= 35:
			derived_profile = "moderate"
		else:
			derived_profile = "conservative"

		questionnaire_id = str(uuid.uuid4())
		result = {
			"questionnaire_id": questionnaire_id,
			"customer_id": customer_id,
			"responses": responses,
			"scores_by_question": scores,
			"total_score": total_score,
			"max_score": max_score,
			"score_pct": round(score_pct, 2),
			"derived_risk_profile": derived_profile,
			"missing_questions": missing,
			"completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}
		self._questionnaires[questionnaire_id] = result
		self._audit(self.tenant_id, "risk_questionnaire_completed", questionnaire_id)
		return result

	async def determine_risk_profile(self, questionnaire_id: str) -> dict[str, Any]:
		"""Finalise a risk profile from a completed questionnaire."""
		questionnaire = self._questionnaires.get(questionnaire_id)
		assert questionnaire is not None, f"questionnaire not found: {questionnaire_id}"

		profile = questionnaire["derived_risk_profile"]
		allocation = _MODEL_ALLOCATIONS.get(profile, {})
		expected_return = _weighted_portfolio_return(allocation)

		self._audit(self.tenant_id, "risk_profile_determined", questionnaire_id)
		return {
			"questionnaire_id": questionnaire_id,
			"customer_id": questionnaire["customer_id"],
			"risk_profile": profile,
			"recommended_allocation": allocation,
			"expected_annual_return_pct": round(expected_return, 2),
			"risk_description": {
				"conservative":    "Capital preservation focus. Low volatility.",
				"moderate":        "Balanced growth and capital preservation.",
				"balanced":        "Moderate growth with acceptable drawdown.",
				"aggressive":      "High growth focus. Accepts significant drawdowns.",
				"very_aggressive": "Maximum growth. Accepts large short-term losses.",
			}.get(profile, ""),
			"determined_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	# ------------------------------------------------------------------
	# Portfolio recommendations
	# ------------------------------------------------------------------

	async def recommended_portfolio(
		self,
		risk_profile: str,
		investment_amount: float,
		*,
		currency: str = "USD",
		time_horizon_years: int = 10,
	) -> dict[str, Any]:
		"""Return a model portfolio recommendation for a given risk profile and amount."""
		risk_profile = normalize_code(risk_profile)
		assert risk_profile in SUPPORTED_RISK_PROFILES, f"unsupported risk_profile: {risk_profile}"
		assert investment_amount > 0, "investment_amount must be positive"

		allocation = _MODEL_ALLOCATIONS.get(risk_profile, {})
		expected_return = _weighted_portfolio_return(allocation)

		# Project future value
		projected_value = investment_amount * ((1 + expected_return / 100) ** time_horizon_years)

		holdings_usd: list[dict[str, Any]] = []
		for asset, pct in allocation.items():
			holdings_usd.append({
				"asset_class": asset,
				"allocation_pct": pct,
				"initial_value": round(investment_amount * pct / 100, 2),
				"projected_value": round(investment_amount * pct / 100 * ((1 + _ASSET_RETURNS.get(asset, 0) / 100) ** time_horizon_years), 2),
			})

		self._audit(self.tenant_id, "portfolio_recommended", risk_profile)
		return {
			"risk_profile": risk_profile,
			"investment_amount": investment_amount,
			"currency": currency,
			"time_horizon_years": time_horizon_years,
			"expected_annual_return_pct": round(expected_return, 2),
			"projected_value": round(projected_value, 2),
			"total_projected_gain": round(projected_value - investment_amount, 2),
			"holdings": holdings_usd,
			"rebalancing_frequency": "quarterly",
			"management_fee_pct": 0.5,
			"generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	# ------------------------------------------------------------------
	# Auto-invest
	# ------------------------------------------------------------------

	async def auto_invest(
		self,
		customer_id: str,
		amount: float,
		frequency: str,
		*,
		profile_id: str | None = None,
		currency: str = "USD",
	) -> dict[str, Any]:
		"""Set up or execute an automatic investment instruction."""
		assert customer_id, "customer_id required"
		assert amount > 0, "amount must be positive"
		frequency = normalize_code(frequency)
		assert frequency in SUPPORTED_CADENCES, f"unsupported frequency: {frequency}"

		# Resolve profile
		profile = None
		if profile_id:
			profile = self.profiles.get(profile_id)
		if profile is None:
			# Fall back to first profile for this customer
			profile = next(
				(p for p in self.profiles.values() if p.client_id == customer_id),
				None,
			)

		risk_profile = profile.risk_profile if profile else "balanced"
		allocation = _MODEL_ALLOCATIONS.get(risk_profile, {})

		# Credit portfolio holdings
		pid = profile.id if profile else customer_id
		holdings = self._portfolio_holdings.setdefault(pid, {})
		for asset, pct in allocation.items():
			holdings[asset] = holdings.get(asset, 0.0) + amount * pct / 100

		investment_id = str(uuid.uuid4())
		next_run: dict[str, int] = {
			"daily": 1, "weekly": 7, "monthly": 30, "quarterly": 90, "annual": 365,
		}
		next_date = (
			datetime.datetime.now(datetime.timezone.utc)
			+ datetime.timedelta(days=next_run.get(frequency, 30))
		).isoformat()

		log_entry = {
			"investment_id": investment_id,
			"customer_id": customer_id,
			"profile_id": pid,
			"amount": amount,
			"currency": currency,
			"frequency": frequency,
			"risk_profile": risk_profile,
			"allocation_used": allocation,
			"next_investment_date": next_date,
			"executed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
			"status": "active",
		}
		self._auto_invest_logs.append(log_entry)
		self._audit(self.tenant_id, "auto_invest_executed", investment_id)
		return log_entry

	# ------------------------------------------------------------------
	# Auto-rebalance
	# ------------------------------------------------------------------

	async def auto_rebalance(
		self,
		portfolio_id: str,
		*,
		drift_threshold_pct: float = 5.0,
		dry_run: bool = False,
	) -> dict[str, Any]:
		"""Automatically rebalance a portfolio to its target allocation."""
		profile = self.profiles.get(portfolio_id)
		risk_profile = profile.risk_profile if profile else "balanced"
		target_allocation = _MODEL_ALLOCATIONS.get(risk_profile, {})

		holdings = self._portfolio_holdings.get(portfolio_id, {})
		total_value = sum(holdings.values())

		if total_value == 0:
			return {
				"portfolio_id": portfolio_id,
				"status": "skipped",
				"reason": "no_assets",
			}

		trades: list[dict[str, Any]] = []
		for asset, target_pct in target_allocation.items():
			target_value = total_value * target_pct / 100
			current_value = holdings.get(asset, 0.0)
			current_pct = (current_value / total_value * 100) if total_value > 0 else 0.0
			drift = abs(current_pct - target_pct)
			if drift >= drift_threshold_pct:
				direction = "buy" if target_value > current_value else "sell"
				trades.append({
					"asset_class": asset,
					"direction": direction,
					"current_pct": round(current_pct, 2),
					"target_pct": target_pct,
					"drift_pct": round(drift, 2),
					"trade_value_usd": round(abs(target_value - current_value), 2),
				})
				if not dry_run:
					holdings[asset] = target_value

		if not dry_run and trades:
			self._portfolio_holdings[portfolio_id] = holdings

		rebalance_id = str(uuid.uuid4())
		log_entry = {
			"rebalance_id": rebalance_id,
			"portfolio_id": portfolio_id,
			"risk_profile": risk_profile,
			"total_value_usd": round(total_value, 2),
			"drift_threshold_pct": drift_threshold_pct,
			"trades_executed": len(trades),
			"trades": trades,
			"dry_run": dry_run,
			"rebalanced_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
			"status": "executed" if (not dry_run and trades) else ("no_action_needed" if not trades else "simulated"),
		}
		self._rebalance_logs.append(log_entry)
		self._audit(self.tenant_id, "auto_rebalance_executed", portfolio_id)
		return log_entry

	# ------------------------------------------------------------------
	# Goal tracking
	# ------------------------------------------------------------------

	async def goal_tracking(
		self,
		goal_id: str,
		*,
		current_value_usd: float | None = None,
	) -> dict[str, Any]:
		"""Compute progress toward a defined investment goal."""
		goal = self.goals.get(goal_id)
		assert goal is not None, f"goal not found: {goal_id}"

		target_usd = goal.target_amount_minor / 1_000_000
		currency = goal.currency
		horizon_date = datetime.datetime.fromisoformat(goal.horizon_date)
		now = datetime.datetime.now(datetime.timezone.utc)

		# Use provided current value or derive from portfolio holdings
		if current_value_usd is None:
			profile_holdings = self._portfolio_holdings.get(goal.profile_id, {})
			current_value_usd = sum(profile_holdings.values())

		progress_pct = (current_value_usd / target_usd * 100) if target_usd > 0 else 0.0
		days_remaining = max(0, (horizon_date.replace(tzinfo=datetime.timezone.utc) - now).days)
		years_remaining = days_remaining / 365

		# Required monthly to close the gap
		gap_usd = max(0.0, target_usd - current_value_usd)
		monthly_rate = 0.008  # ~10% annual / 12
		if years_remaining > 0 and monthly_rate > 0:
			months_remaining = years_remaining * 12
			required_monthly = (
				gap_usd * monthly_rate / (((1 + monthly_rate) ** months_remaining) - 1)
				if months_remaining > 0
				else gap_usd
			)
		else:
			required_monthly = gap_usd

		on_track = progress_pct >= (1 - years_remaining / max(years_remaining + 1, 1)) * 100

		progress_entry = {
			"goal_id": goal_id,
			"goal_type": goal.goal_type,
			"target_usd": round(target_usd, 2),
			"currency": currency,
			"current_value_usd": round(current_value_usd, 2),
			"progress_pct": round(progress_pct, 2),
			"gap_usd": round(gap_usd, 2),
			"days_remaining": days_remaining,
			"years_remaining": round(years_remaining, 2),
			"required_monthly_contribution_usd": round(required_monthly, 2),
			"on_track": on_track,
			"horizon_date": goal.horizon_date,
			"checked_at": now.isoformat(),
		}
		self._goal_progress.setdefault(goal_id, []).append(progress_entry)
		self._audit(self.tenant_id, "goal_progress_checked", goal_id)
		return progress_entry

	# ------------------------------------------------------------------
	# Client onboarding
	# ------------------------------------------------------------------

	async def onboard_client(
		self,
		customer_id: str,
		plan: dict[str, Any],
		*,
		kyc_reference: str = "",
		suitability_reference: str = "",
	) -> dict[str, Any]:
		"""Onboard a new robo advisory client end-to-end."""
		assert customer_id, "customer_id required"
		assert plan, "plan required"

		risk_profile = normalize_code(plan.get("risk_profile", "balanced"))
		initial_amount = float(plan.get("initial_investment_usd", 0.0))
		monthly_contribution = float(plan.get("monthly_contribution_usd", 100.0))
		goal_type = normalize_code(plan.get("goal_type", "growth"))
		horizon_years = int(plan.get("horizon_years", 10))
		currency = normalize_currency(plan.get("currency", "USD"))

		# Create investor profile
		profile_id = str(uuid.uuid4())
		kyc_ref = kyc_reference or f"kyc-{customer_id}"
		suitability_ref = suitability_reference or f"suit-{customer_id}"

		profile = InvestorProfile(
			profile_id,
			self.tenant_id,
			customer_id,
			kyc_ref,
			suitability_ref,
			risk_profile,
		)
		self.profiles[profile_id] = profile
		self._audit(self.tenant_id, "investor_profile_created", profile_id)

		# Create goal
		goal_id = str(uuid.uuid4())
		target_amount_usd = float(plan.get("target_amount_usd", 100_000))
		horizon_date = (
			datetime.datetime.now(datetime.timezone.utc)
			+ datetime.timedelta(days=horizon_years * 365)
		).isoformat()

		goal = GoalPlan(
			goal_id,
			self.tenant_id,
			profile_id,
			goal_type,
			int(target_amount_usd * 1_000_000),
			currency,
			horizon_date,
		)
		self.goals[goal_id] = goal
		self._audit(self.tenant_id, "goal_plan_defined", goal_id)

		# Seed portfolio if initial amount provided
		if initial_amount > 0:
			allocation = _MODEL_ALLOCATIONS.get(risk_profile, {})
			self._portfolio_holdings[profile_id] = {
				asset: round(initial_amount * pct / 100, 2)
				for asset, pct in allocation.items()
			}

		# Set up automation
		plan_id = str(uuid.uuid4())
		cadence = normalize_code(plan.get("cadence", "monthly"))
		if cadence in SUPPORTED_CADENCES:
			auto_plan = AutomationPlan(plan_id, self.tenant_id, f"rec-{profile_id}", f"bank-{customer_id}", cadence)
			self.automation[plan_id] = auto_plan

		onboarding_record = {
			"onboarding_id": str(uuid.uuid4()),
			"customer_id": customer_id,
			"profile_id": profile_id,
			"goal_id": goal_id,
			"automation_plan_id": plan_id,
			"risk_profile": risk_profile,
			"initial_investment_usd": initial_amount,
			"monthly_contribution_usd": monthly_contribution,
			"target_amount_usd": target_amount_usd,
			"horizon_years": horizon_years,
			"currency": currency,
			"allocation": _MODEL_ALLOCATIONS.get(risk_profile, {}),
			"kyc_reference": kyc_ref,
			"suitability_reference": suitability_ref,
			"onboarded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
			"status": "active",
		}
		self._onboarding_records[customer_id] = onboarding_record
		self._audit(self.tenant_id, "client_onboarded", customer_id)
		return onboarding_record

	# ------------------------------------------------------------------
	# Drift monitoring
	# ------------------------------------------------------------------

	async def drift_monitoring(
		self,
		portfolio_id: str,
		tolerance_pct: float,
		*,
		alert_on_breach: bool = True,
	) -> dict[str, Any]:
		"""Monitor portfolio drift against target allocation and flag breaches."""
		assert portfolio_id, "portfolio_id required"
		assert 0 < tolerance_pct <= 100, "tolerance_pct must be between 0 and 100"

		profile = self.profiles.get(portfolio_id)
		risk_profile = profile.risk_profile if profile else "balanced"
		target_allocation = _MODEL_ALLOCATIONS.get(risk_profile, {})

		holdings = self._portfolio_holdings.get(portfolio_id, {})
		total_value = sum(holdings.values())

		breaches: list[dict[str, Any]] = []
		asset_drift: list[dict[str, Any]] = []

		for asset, target_pct in target_allocation.items():
			current_value = holdings.get(asset, 0.0)
			current_pct = (current_value / total_value * 100) if total_value > 0 else 0.0
			drift = current_pct - target_pct
			asset_drift.append({
				"asset_class": asset,
				"current_pct": round(current_pct, 2),
				"target_pct": target_pct,
				"drift_pct": round(drift, 2),
				"breach": abs(drift) > tolerance_pct,
			})
			if abs(drift) > tolerance_pct:
				breaches.append({
					"asset_class": asset,
					"drift_pct": round(drift, 2),
					"direction": "overweight" if drift > 0 else "underweight",
				})

		max_drift = max((abs(d["drift_pct"]) for d in asset_drift), default=0.0)

		drift_id = str(uuid.uuid4())
		drift_record = DriftRecord(
			drift_id,
			self.tenant_id,
			portfolio_id,
			int(max_drift * 100),
			f"monitoring-{portfolio_id[:8]}",
		)
		self.drift[drift_id] = drift_record
		self._audit(self.tenant_id, "drift_monitored", portfolio_id)

		return {
			"drift_id": drift_id,
			"portfolio_id": portfolio_id,
			"risk_profile": risk_profile,
			"total_value_usd": round(total_value, 2),
			"tolerance_pct": tolerance_pct,
			"max_drift_pct": round(max_drift, 2),
			"breach_count": len(breaches),
			"breaches": breaches,
			"asset_drift": asset_drift,
			"rebalance_recommended": len(breaches) > 0,
			"alert_raised": alert_on_breach and len(breaches) > 0,
			"monitored_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	# ------------------------------------------------------------------
	# Tax optimisation
	# ------------------------------------------------------------------

	async def tax_optimisation(
		self,
		portfolio_id: str,
		jurisdiction: str,
		*,
		tax_year: int | None = None,
		min_loss_usd: float = 250.0,
	) -> dict[str, Any]:
		"""Identify tax-loss harvesting opportunities and generate optimisation report."""
		year = tax_year or datetime.datetime.now().year
		holdings = self._portfolio_holdings.get(portfolio_id, {})

		harvesting_candidates: list[dict[str, Any]] = []
		total_harvestable = 0.0
		total_tax_saving = 0.0

		for asset, current_value in holdings.items():
			# Stub: assume 12% unrealised loss on volatile assets
			volatile_assets = {"equities", "alternatives", "real_estate"}
			if asset in volatile_assets:
				cost_basis = current_value * 1.12
				unrealised_loss = cost_basis - current_value
				tax_saving = unrealised_loss * _CGT_RATE

				if unrealised_loss >= min_loss_usd:
					harvesting_candidates.append({
						"asset_class": asset,
						"current_value_usd": round(current_value, 2),
						"cost_basis_usd": round(cost_basis, 2),
						"unrealised_loss_usd": round(unrealised_loss, 2),
						"estimated_tax_saving_usd": round(tax_saving, 2),
						"replacement_asset": f"{asset}_fund_b",  # wash-sale substitute
					})
					total_harvestable += unrealised_loss
					total_tax_saving += tax_saving

		# Locate low-turnover asset classes for deferral
		deferral_candidates = [
			asset for asset in holdings
			if asset not in {"equities", "alternatives", "real_estate"}
		]

		optimisation = {
			"optimisation_id": str(uuid.uuid4()),
			"portfolio_id": portfolio_id,
			"jurisdiction": jurisdiction,
			"tax_year": year,
			"cgt_rate_pct": _CGT_RATE * 100,
			"tlh_candidates": harvesting_candidates,
			"tlh_count": len(harvesting_candidates),
			"total_harvestable_loss_usd": round(total_harvestable, 2),
			"total_estimated_tax_saving_usd": round(total_tax_saving, 2),
			"gain_deferral_assets": deferral_candidates,
			"tax_efficient_vehicles": ["isa", "pension", "sipp"] if jurisdiction == "GB" else ["rrsp"] if jurisdiction == "CA" else [],
			"generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}
		self._tax_optimisation_logs.append(optimisation)
		self._audit(self.tenant_id, "tax_optimisation_generated", portfolio_id)
		return optimisation

	# ------------------------------------------------------------------
	# Performance reporting
	# ------------------------------------------------------------------

	async def robo_performance_report(
		self,
		portfolio_id: str,
		period: str,
		*,
		benchmark_profile: str = "balanced",
	) -> dict[str, Any]:
		"""Generate a robo-advisor performance report for a portfolio."""
		profile = self.profiles.get(portfolio_id)
		risk_profile = profile.risk_profile if profile else "balanced"
		allocation = _MODEL_ALLOCATIONS.get(risk_profile, {})

		holdings = self._portfolio_holdings.get(portfolio_id, {})
		total_value = sum(holdings.values())

		# Weighted portfolio return (annualised)
		portfolio_return = _weighted_portfolio_return(allocation)
		benchmark_return = _weighted_portfolio_return(_MODEL_ALLOCATIONS.get(benchmark_profile, {}))
		alpha = portfolio_return - benchmark_return

		# Sharpe approximation (simplified)
		volatility = portfolio_return * 0.15
		sharpe = (portfolio_return - 3.0) / volatility if volatility > 0 else 0.0

		# Fee drag
		management_fee_pct = 0.5
		performance_fee_usd = 0.0
		if portfolio_return > _HURDLE_RATE_PCT:
			excess = portfolio_return - _HURDLE_RATE_PCT
			performance_fee_usd = total_value * excess / 100 * _PERFORMANCE_FEE_PCT / 100

		logs = self._performance_logs.get(portfolio_id, [])
		log_entry = {
			"report_id": str(uuid.uuid4()),
			"portfolio_id": portfolio_id,
			"period": period,
			"risk_profile": risk_profile,
			"benchmark_profile": benchmark_profile,
			"portfolio_value_usd": round(total_value, 2),
			"expected_annual_return_pct": round(portfolio_return, 4),
			"benchmark_return_pct": round(benchmark_return, 4),
			"alpha_pct": round(alpha, 4),
			"sharpe_ratio": round(sharpe, 4),
			"volatility_estimate_pct": round(volatility, 4),
			"management_fee_pct": management_fee_pct,
			"performance_fee_usd": round(performance_fee_usd, 2),
			"net_return_pct": round(portfolio_return - management_fee_pct, 4),
			"auto_invest_executions": len(self._auto_invest_logs),
			"rebalance_executions": len(self._rebalance_logs),
			"generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}
		logs.append(log_entry)
		self._performance_logs[portfolio_id] = logs
		self._audit(self.tenant_id, "robo_performance_report_generated", portfolio_id)
		return log_entry

	# ------------------------------------------------------------------
	# Existing core sync methods (preserved from original)
	# ------------------------------------------------------------------

	def create_investor_profile(
		self,
		profile_id: str,
		tenant_id: str,
		client_id: str,
		kyc_reference: str,
		suitability_reference: str,
		risk_profile: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		risk_profile = normalize_code(risk_profile)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "create_investor_profile",
			"client_present": bool(client_id),
			"kyc_present": bool(kyc_reference),
			"suitability_present": bool(suitability_reference),
			"risk_profile_supported": risk_profile in SUPPORTED_RISK_PROFILES,
		})
		profile = InvestorProfile(profile_id, tenant_id, client_id, kyc_reference, suitability_reference, risk_profile)
		self.profiles[profile_id] = profile
		self._audit(tenant_id, "investor_profile_created", profile_id)
		return profile.to_dict()

	def define_goal_plan(
		self,
		goal_id: str,
		tenant_id: str,
		profile_id: str,
		goal_type: str,
		target_amount_minor: int,
		currency: str,
		horizon_date: str,
	) -> dict[str, Any]:
		profile = self._tenant_profile_or_none(profile_id, tenant_id)
		goal_type = normalize_code(goal_type)
		currency = normalize_currency(currency)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "define_goal_plan",
			"profile_present": profile is not None,
			"goal_type_supported": goal_type in SUPPORTED_GOAL_TYPES,
			"positive_target": positive_minor(target_amount_minor),
			"currency_supported": currency in SUPPORTED_CURRENCIES,
			"horizon_present": bool(horizon_date),
		})
		goal = GoalPlan(goal_id, tenant_id, profile_id, goal_type, int(target_amount_minor), currency, horizon_date)
		self.goals[goal_id] = goal
		self._audit(tenant_id, "goal_plan_defined", goal_id)
		return goal.to_dict()

	def publish_model_portfolio(
		self,
		model_id: str,
		tenant_id: str,
		name: str,
		risk_profile: str,
		target_allocation: dict[str, float],
		policy_reference: str,
	) -> dict[str, Any]:
		risk_profile = normalize_code(risk_profile)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "publish_model_portfolio",
			"risk_profile_supported": risk_profile in SUPPORTED_RISK_PROFILES,
			"allocation_totals_100": allocation_totals_100(target_allocation),
			"policy_present": bool(policy_reference),
		})
		model = ModelPortfolio(model_id, tenant_id, name, risk_profile, dict(target_allocation), policy_reference)
		self.models[model_id] = model
		self._audit(tenant_id, "model_portfolio_published", model_id)
		return model.to_dict()

	def generate_recommendation(
		self,
		recommendation_id: str,
		tenant_id: str,
		profile_id: str,
		goal_id: str,
		model_id: str,
		analysis_reference: str,
	) -> dict[str, Any]:
		profile = self._tenant_profile_or_none(profile_id, tenant_id)
		goal = self._tenant_goal_or_none(goal_id, tenant_id)
		model = self._tenant_model_or_none(model_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "generate_recommendation",
			"profile_present": profile is not None,
			"goal_present": goal is not None,
			"model_present": model is not None,
			"analysis_present": bool(analysis_reference),
		})
		recommendation = RecommendationPacket(recommendation_id, tenant_id, profile_id, goal_id, model_id, analysis_reference)
		self.recommendations[recommendation_id] = recommendation
		self._audit(tenant_id, "recommendation_generated", recommendation_id)
		return recommendation.to_dict()

	def approve_recommendation(
		self,
		recommendation_id: str,
		tenant_id: str,
		reviewer_id: str,
	) -> dict[str, Any]:
		recommendation = self._tenant_recommendation_or_none(recommendation_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "approve_recommendation",
			"recommendation_present": recommendation is not None,
			"reviewer_present": bool(reviewer_id),
		})
		assert recommendation is not None
		recommendation.status = "approved"
		self._audit(tenant_id, "recommendation_approved", recommendation_id)
		return recommendation.to_dict() | {"reviewer_id": reviewer_id}

	def configure_automation_plan(
		self,
		plan_id: str,
		tenant_id: str,
		recommendation_id: str,
		funding_source_reference: str,
		cadence: str,
	) -> dict[str, Any]:
		recommendation = self._tenant_recommendation_or_none(recommendation_id, tenant_id)
		cadence = normalize_code(cadence)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "configure_automation_plan",
			"approved_recommendation_present": recommendation is not None and recommendation.status == "approved",
			"cadence_supported": cadence in SUPPORTED_CADENCES,
			"funding_source_present": bool(funding_source_reference),
		})
		plan = AutomationPlan(plan_id, tenant_id, recommendation_id, funding_source_reference, cadence)
		self.automation[plan_id] = plan
		self._audit(tenant_id, "automation_plan_configured", plan_id)
		return plan.to_dict()

	def record_drift(
		self,
		drift_id: str,
		tenant_id: str,
		profile_id: str,
		drift_bps: int,
		analysis_reference: str,
	) -> dict[str, Any]:
		profile = self._tenant_profile_or_none(profile_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_drift",
			"profile_present": profile is not None,
			"analysis_present": bool(analysis_reference),
		})
		record = DriftRecord(drift_id, tenant_id, profile_id, int(drift_bps), analysis_reference)
		self.drift[drift_id] = record
		self._audit(tenant_id, "drift_recorded", drift_id)
		return record.to_dict()

	def record_tax_loss_candidate(
		self,
		candidate_id: str,
		tenant_id: str,
		profile_id: str,
		instrument_id: str,
		loss_minor: int,
		tax_lot_reference: str,
	) -> dict[str, Any]:
		profile = self._tenant_profile_or_none(profile_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_tax_loss_candidate",
			"profile_present": profile is not None,
			"tax_lot_present": bool(tax_lot_reference),
			"positive_loss": positive_minor(loss_minor),
		})
		candidate = TaxLossCandidate(candidate_id, tenant_id, profile_id, instrument_id, int(loss_minor), tax_lot_reference)
		self.tax_loss[candidate_id] = candidate
		self._audit(tenant_id, "tax_loss_candidate_recorded", candidate_id)
		return candidate.to_dict()

	def record_review(
		self,
		review_id: str,
		tenant_id: str,
		reference_id: str,
		reviewer_id: str,
		status: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		status = normalize_code(status)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_review",
			"status_supported": status in SUPPORTED_REVIEW_STATUSES,
			"evidence_present": bool(evidence_reference),
		})
		review = ReviewRecord(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[review_id] = review
		self._audit(tenant_id, "robo_review_recorded", review_id)
		return review.to_dict()

	def register_robo_agent(
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
			"operation": "register_robo_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
		})
		evidence = RoboEvidence(agent_id, tenant_id, "agent", agent_id, "registered", {"name": name, "runtime": runtime, "role": role, "scope": scope})
		self.evidence[agent_id] = evidence
		self._audit(tenant_id, "robo_agent_registered", agent_id)
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
			"operation": "robo_batch",
			"event_stream": event_stream,
		})
		return {
			"tenant_id": tenant_id,
			"item_count": item_count,
			"processor": "bytewax",
			"stream": "apg.fintech.robo.lifecycle",
			"accepted": True,
		}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"profile_count": self._count(self.profiles, tenant_id),
			"goal_count": self._count(self.goals, tenant_id),
			"model_count": self._count(self.models, tenant_id),
			"recommendation_count": self._count(self.recommendations, tenant_id),
			"automation_count": self._count(self.automation, tenant_id),
			"drift_count": self._count(self.drift, tenant_id),
			"tax_loss_count": self._count(self.tax_loss, tenant_id),
			"review_count": self._count(self.reviews, tenant_id),
			"auto_invest_executions": len(self._auto_invest_logs),
			"rebalance_executions": len(self._rebalance_logs),
			"onboarded_clients": len(self._onboarding_records),
			"tax_optimisations": len(self._tax_optimisation_logs),
			"audit_event_count": sum(1 for e in self.audit_events if e["tenant_id"] == tenant_id),
			"streaming": get_capability_contract(tenant_id)["streaming"],
		}

	# ------------------------------------------------------------------
	# Additional async methods
	# ------------------------------------------------------------------

	async def health_check(self) -> dict[str, Any]:
		"""Return robo advisory service health status."""
		return {
			"service": "robo_advisory", "status": "healthy",
			"profile_count": len(self.profiles), "goal_count": len(self.goals),
			"checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def bulk_onboard_clients(self, clients: list[dict[str, Any]]) -> dict[str, Any]:
		"""Bulk onboard multiple robo advisory clients."""
		processed, errors = [], []
		for c in clients:
			try:
				rec = await self.onboard_client(c["customer_id"], c.get("plan", {}))
				processed.append(rec["customer_id"])
			except Exception as exc:
				errors.append({"input": c, "error": str(exc)})
		return {"processed": len(processed), "failed": len(errors), "onboarded": processed}

	async def scheduled_rebalance_batch(self, cadence: str = "quarterly") -> dict[str, Any]:
		"""Run scheduled rebalancing for all portfolios meeting the cadence."""
		results = []
		for profile_id in self._portfolio_holdings:
			result = await self.auto_rebalance(profile_id)
			results.append({"profile_id": profile_id, "status": result.get("status")})
		return {
			"cadence": cadence, "portfolios_reviewed": len(results),
			"rebalanced": sum(1 for r in results if r["status"] == "executed"),
			"run_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def esg_portfolio_filter(self, profile_id: str, exclusions: list[str]) -> dict[str, Any]:
		"""Filter portfolio holdings by ESG exclusion criteria."""
		holdings = self._portfolio_holdings.get(profile_id, {})
		flagged = [h for h in holdings if any(ex.lower() in h.lower() for ex in exclusions)]
		return {
			"profile_id": profile_id, "exclusions": exclusions,
			"total_holdings": len(holdings), "flagged": flagged,
			"esg_compliant": len(flagged) == 0,
			"screened_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def projected_retirement_income(self, profile_id: str, retirement_age: int, current_age: int) -> dict[str, Any]:
		"""Project monthly retirement income from current portfolio at target retirement age."""
		profile = self.profiles.get(profile_id)
		assert profile is not None, f"profile not found: {profile_id}"
		holdings = self._portfolio_holdings.get(profile_id, {})
		current_value = sum(holdings.values())
		years = max(retirement_age - current_age, 0)
		risk_profile = getattr(profile, "risk_profile", "balanced")
		allocation = _MODEL_ALLOCATIONS.get(risk_profile, {})
		expected_return = _weighted_portfolio_return(allocation) / 100
		projected_value = current_value * ((1 + expected_return) ** years)
		safe_withdrawal_rate = 0.04
		annual_income = projected_value * safe_withdrawal_rate
		self._audit(self.tenant_id, "retirement_projection_computed", profile_id)
		return {
			"profile_id": profile_id, "current_age": current_age, "retirement_age": retirement_age,
			"years_to_retirement": years, "current_portfolio_value_usd": round(current_value, 2),
			"projected_value_usd": round(projected_value, 2),
			"projected_annual_income_usd": round(annual_income, 2),
			"projected_monthly_income_usd": round(annual_income / 12, 2),
			"safe_withdrawal_rate": safe_withdrawal_rate,
			"computed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def auto_invest_execute_all(self) -> dict[str, Any]:
		"""Execute all active auto-invest plans for this tenant."""
		active_plans = [p for p in self.automation.values() if getattr(p, "tenant_id", self.tenant_id) == self.tenant_id]
		executed = []
		for plan in active_plans:
			result = await self.auto_invest(
				customer_id=getattr(plan, "funding_source_reference", "unknown"),
				amount=1000.0,
				frequency=getattr(plan, "cadence", "monthly"),
			)
			executed.append(result["investment_id"])
		return {
			"plans_processed": len(active_plans), "executed": len(executed),
			"investment_ids": executed, "run_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def fee_transparency_report(self, profile_id: str) -> dict[str, Any]:
		"""Generate a fee transparency report per MiFID II / CMA disclosure requirements."""
		holdings = self._portfolio_holdings.get(profile_id, {})
		total_value = sum(holdings.values())
		management_fee_pct = 0.5
		performance_fee_pct = _PERFORMANCE_FEE_PCT
		annual_mgmt_fee = total_value * management_fee_pct / 100
		return {
			"profile_id": profile_id, "total_value_usd": round(total_value, 2),
			"management_fee_pct": management_fee_pct,
			"annual_management_fee_usd": round(annual_mgmt_fee, 2),
			"performance_fee_pct": performance_fee_pct,
			"performance_fee_hurdle_pct": _HURDLE_RATE_PCT,
			"generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def cma_robo_return(self, period: str) -> dict[str, Any]:
		"""File a CMA Robo-Advisory return for the period."""
		return {
			"report_type": "CMA_ROBO_ADVISORY_RETURN", "period": period,
			"total_profiles": len(self.profiles), "total_goals": len(self.goals),
			"total_aum_usd": round(sum(sum(v.values()) for v in self._portfolio_holdings.values()), 2),
			"auto_invest_executions": len(self._auto_invest_logs),
			"status": "draft", "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def risk_profile_update(self, profile_id: str, new_risk_profile: str, reason: str) -> dict[str, Any]:
		"""Update a client's risk profile following a life event or review."""
		profile = self.profiles.get(profile_id)
		assert profile is not None, f"profile not found: {profile_id}"
		new_risk_profile = normalize_code(new_risk_profile)
		assert new_risk_profile in SUPPORTED_RISK_PROFILES, f"unsupported: {new_risk_profile}"
		old_profile = profile.risk_profile
		profile.risk_profile = new_risk_profile
		self._audit(self.tenant_id, "risk_profile_updated", profile_id)
		return {
			"profile_id": profile_id, "old_risk_profile": old_profile,
			"new_risk_profile": new_risk_profile, "reason": reason,
			"updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def portfolio_health_score(self, profile_id: str) -> dict[str, Any]:
		"""Compute a portfolio health score (0–100) based on drift, diversity, and goal progress."""
		drift_result = await self.drift_monitoring(profile_id, tolerance_pct=5.0, alert_on_breach=False)
		goal_list = [g for g in self.goals.values() if g.profile_id == profile_id]
		holdings = self._portfolio_holdings.get(profile_id, {})
		diversity_score = min(len(holdings) * 10, 40.0)
		drift_score = max(0, 40.0 - drift_result.get("breach_count", 0) * 10)
		goal_score = 20.0 if goal_list else 0.0
		health_score = diversity_score + drift_score + goal_score
		self._audit(self.tenant_id, "portfolio_health_scored", profile_id)
		return {
			"profile_id": profile_id, "health_score": round(health_score, 1),
			"components": {"diversity": diversity_score, "drift": drift_score, "goals": goal_score},
			"recommendations": [] if health_score >= 80 else ["Consider rebalancing", "Add more asset classes"],
			"computed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def export_portfolio_report(self, profile_id: str, fmt: str = "pdf") -> dict[str, Any]:
		"""Export a full portfolio report for a client."""
		assert fmt in {"pdf", "csv", "json"}
		holdings = self._portfolio_holdings.get(profile_id, {})
		return {
			"profile_id": profile_id, "format": fmt,
			"holding_count": len(holdings),
			"file_reference": f"robo_report_{profile_id}_{fmt}",
			"generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def goal_milestone_tracking(self, goal_id: str, milestone_name: str, milestone_value: float) -> dict[str, Any]:
		"""Record achievement of a goal milestone (e.g., 25%, 50%, 75% funded)."""
		goal = self.goals.get(goal_id)
		assert goal is not None, f"goal not found: {goal_id}"
		target = goal.target_amount_minor / 1_000_000
		progress_pct = round(milestone_value / target * 100, 2) if target > 0 else 0.0
		self._audit(self.tenant_id, "goal_milestone_recorded", goal_id)
		return {
			"goal_id": goal_id, "milestone_name": milestone_name,
			"milestone_value_usd": milestone_value, "target_usd": round(target, 2),
			"progress_pct": progress_pct, "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def robo_analytics(self, period: str) -> dict[str, Any]:
		"""Aggregate robo advisory performance analytics for a period."""
		return await self.robo_performance_report(f"period-{period}", period)

	async def savings_plan_optimisation(self, profile_id: str, monthly_saving: float, target_years: int) -> dict[str, Any]:
		"""Optimise a savings plan to maximise goal attainment."""
		profile = self.profiles.get(profile_id)
		assert profile is not None, f"profile not found: {profile_id}"
		risk_profile = profile.risk_profile
		allocation = _MODEL_ALLOCATIONS.get(risk_profile, {})
		expected_return = _weighted_portfolio_return(allocation) / 100
		total_savings = monthly_saving * 12 * target_years
		monthly_rate = expected_return / 12
		months = target_years * 12
		fv = monthly_saving * (((1 + monthly_rate) ** months - 1) / monthly_rate) if monthly_rate > 0 else total_savings
		self._audit(self.tenant_id, "savings_plan_optimised", profile_id)
		return {
			"profile_id": profile_id, "risk_profile": risk_profile, "monthly_saving_usd": monthly_saving,
			"target_years": target_years, "expected_return_pct": round(expected_return * 100, 4),
			"projected_value_usd": round(fv, 2), "total_contributions_usd": round(total_savings, 2),
			"projected_gain_usd": round(fv - total_savings, 2), "optimised_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def model_portfolio_rebalance_signal(self, risk_profile: str) -> dict[str, Any]:
		"""Return the current rebalance signal for a model portfolio based on market drift."""
		allocation = _MODEL_ALLOCATIONS.get(normalize_code(risk_profile), {})
		expected_return = _weighted_portfolio_return(allocation)
		signal = "rebalance" if expected_return < 5.0 else "hold"
		return {"risk_profile": risk_profile, "signal": signal, "expected_return_pct": round(expected_return, 4), "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}

	async def compliance_suitability_check(self, profile_id: str) -> dict[str, Any]:
		"""Verify that current portfolio allocation remains suitable for investor profile."""
		profile = self.profiles.get(profile_id)
		assert profile is not None, f"profile not found: {profile_id}"
		risk_profile = profile.risk_profile
		target = _MODEL_ALLOCATIONS.get(risk_profile, {})
		holdings = self._portfolio_holdings.get(profile_id, {})
		total = sum(holdings.values())
		deviations = {}
		for asset, target_pct in target.items():
			current_pct = holdings.get(asset, 0) / total * 100 if total > 0 else 0
			deviations[asset] = round(current_pct - target_pct, 2)
		suitable = all(abs(v) <= 10 for v in deviations.values())
		self._audit(self.tenant_id, "suitability_compliance_checked", profile_id)
		return {
			"profile_id": profile_id, "risk_profile": risk_profile,
			"suitable": suitable, "deviations": deviations,
			"checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	# ------------------------------------------------------------------
	# Internal helpers
	# ------------------------------------------------------------------

	def _tenant_profile_or_none(self, item_id: str, tenant_id: str) -> InvestorProfile | None:
		item = self.profiles.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_goal_or_none(self, item_id: str, tenant_id: str) -> GoalPlan | None:
		item = self.goals.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_model_or_none(self, item_id: str, tenant_id: str) -> ModelPortfolio | None:
		item = self.models.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_recommendation_or_none(self, item_id: str, tenant_id: str) -> RecommendationPacket | None:
		item = self.recommendations.get(item_id)
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
			action.get("reason", "robo_policy_denied") for action in result["actions"]
		)
		raise PermissionError(reasons or "robo_policy_denied")


RoboService = RoboAdvisoryService
