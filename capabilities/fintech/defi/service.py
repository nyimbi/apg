"""Executable service layer for APG Decentralized Finance.

© 2025 Datacraft — www.datacraft.co.ke
"""

from __future__ import annotations

import datetime
import math
import statistics
import uuid
from typing import Any
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

try:
	from .capability_contract import (
		SUPPORTED_ACTION_STATUSES,
		SUPPORTED_ACTION_TYPES,
		SUPPORTED_AGENT_ROLES,
		SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_GOVERNANCE_VOTES,
		SUPPORTED_POSITION_TYPES,
		SUPPORTED_PROTOCOL_TYPES,
		SUPPORTED_REVIEW_STATUSES,
		SUPPORTED_REWARD_TYPES,
		SUPPORTED_RISK_TIERS,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from .defi_runtime import non_negative_int, normalize_code, positive_int, present
	from .models import (
		DeFiAction,
		DeFiAgent,
		DeFiProtocol,
		DeFiReview,
		GovernanceProposal,
		LiquidityPosition,
		RewardAccrual,
		RiskAssessment,
		YieldStrategy,
	)
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore
		SUPPORTED_ACTION_STATUSES,
		SUPPORTED_ACTION_TYPES,
		SUPPORTED_AGENT_ROLES,
		SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_GOVERNANCE_VOTES,
		SUPPORTED_POSITION_TYPES,
		SUPPORTED_PROTOCOL_TYPES,
		SUPPORTED_REVIEW_STATUSES,
		SUPPORTED_REWARD_TYPES,
		SUPPORTED_RISK_TIERS,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from defi_runtime import non_negative_int, normalize_code, positive_int, present  # type: ignore
	from models import (  # type: ignore
		DeFiAction,
		DeFiAgent,
		DeFiProtocol,
		DeFiReview,
		GovernanceProposal,
		LiquidityPosition,
		RewardAccrual,
		RiskAssessment,
		YieldStrategy,
	)


# ---------------------------------------------------------------------------
# Protocol registry — indicative APY and liquidity data
# ---------------------------------------------------------------------------
_PROTOCOL_REGISTRY: dict[str, dict[str, Any]] = {
	"uniswap_v3":    {"type": "amm",     "chain": "ethereum", "tvl_usd": 4_200_000_000, "base_apy_pct": 8.5},
	"aave_v3":       {"type": "lending", "chain": "ethereum", "tvl_usd": 11_000_000_000, "base_apy_pct": 4.2},
	"compound_v3":   {"type": "lending", "chain": "ethereum", "tvl_usd": 2_800_000_000, "base_apy_pct": 3.8},
	"curve_3pool":   {"type": "amm",     "chain": "ethereum", "tvl_usd": 3_500_000_000, "base_apy_pct": 6.1},
	"pancakeswap_v3":{"type": "amm",     "chain": "bsc",      "tvl_usd": 1_200_000_000, "base_apy_pct": 12.4},
	"yearn_v3":      {"type": "yield",   "chain": "ethereum", "tvl_usd": 800_000_000,   "base_apy_pct": 9.7},
	"lido":          {"type": "liquid_staking", "chain": "ethereum", "tvl_usd": 22_000_000_000, "base_apy_pct": 4.1},
	"makerdao":      {"type": "cdp",     "chain": "ethereum", "tvl_usd": 6_500_000_000, "base_apy_pct": 0.0},
}

# Token price table (USD)
_TOKEN_PRICES: dict[str, float] = {
	"ETH":   3480.0,
	"WBTC":  67500.0,
	"USDC":  1.0,
	"USDT":  1.0,
	"DAI":   1.0,
	"LINK":  15.2,
	"UNI":   9.1,
	"AAVE":  95.0,
	"COMP":  58.0,
	"CRV":   0.48,
	"CVX":   3.2,
	"stETH": 3460.0,
	"MKR":   2_100.0,
	"CAKE":  2.45,
}

# Liquidation threshold: health_factor_bps below which triggers liquidation
_LIQUIDATION_HF_BPS = 11_000  # 110%
_SAFE_HF_BPS = 15_000         # 150%
_CRITICAL_HF_BPS = 12_000     # 120%

# AMM fee tiers (bps)
_AMM_FEE_TIERS: dict[str, int] = {
	"uniswap_v3": 30,    # 0.3% standard tier
	"curve_3pool": 4,    # 0.04% stable pools
	"pancakeswap_v3": 25,
}


def _usd_value(token: str, amount_minor: int) -> float:
	"""Convert token amount_minor (10^6 units) to USD."""
	price = _TOKEN_PRICES.get(token.upper(), 1.0)
	return (amount_minor / 1_000_000) * price


class DecentralizedFinanceService:
	"""Full-featured DeFi service for APG applications.

	Covers liquidity pools, yield farming, lending/borrowing, AMM swaps,
	health factor monitoring, liquidation alerts, governance, and analytics.
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

		self.protocols: dict[str, DeFiProtocol] = {}
		self.positions: dict[str, LiquidityPosition] = {}
		self.actions: dict[str, DeFiAction] = {}
		self.strategies: dict[str, YieldStrategy] = {}
		self.rewards: dict[str, RewardAccrual] = {}
		self.governance: dict[str, GovernanceProposal] = {}
		self.risk_assessments: dict[str, RiskAssessment] = {}
		self.reviews: dict[str, DeFiReview] = {}
		self.agents: dict[str, DeFiAgent] = {}
		self.audit_events: list[dict[str, Any]] = []

		# Extended in-memory state
		self._loans: dict[str, dict[str, Any]] = {}
		self._pool_deposits: dict[str, list[dict[str, Any]]] = {}
		self._farm_positions: dict[str, list[dict[str, Any]]] = {}
		self._swaps: list[dict[str, Any]] = []
		self._liquidation_alerts: list[dict[str, Any]] = []

	# ------------------------------------------------------------------
	# Contract / policy
	# ------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------
	# Liquidity pool operations
	# ------------------------------------------------------------------

	async def liquidity_pool_deposit(
		self,
		customer_id: str,
		pool_id: str,
		token_a_amount: float,
		token_b_amount: float,
		*,
		token_a: str = "USDC",
		token_b: str = "ETH",
		slippage_tolerance_pct: float = 0.5,
	) -> dict[str, Any]:
		"""Deposit token pair into an AMM liquidity pool."""
		assert customer_id, "customer_id required"
		assert pool_id, "pool_id required"
		assert token_a_amount > 0 and token_b_amount > 0, "both amounts must be positive"

		protocol_data = _PROTOCOL_REGISTRY.get(pool_id, {})
		fee_bps = _AMM_FEE_TIERS.get(pool_id, 30)

		# Calculate LP tokens issued (simplified constant-product)
		token_a_usd = token_a_amount * _TOKEN_PRICES.get(token_a.upper(), 1.0)
		token_b_usd = token_b_amount * _TOKEN_PRICES.get(token_b.upper(), 1.0)
		total_usd = token_a_usd + token_b_usd
		lp_tokens = math.sqrt(token_a_amount * token_b_amount)

		deposit_id = str(uuid.uuid4())
		deposit = {
			"deposit_id": deposit_id,
			"customer_id": customer_id,
			"pool_id": pool_id,
			"token_a": token_a.upper(),
			"token_b": token_b.upper(),
			"token_a_amount": token_a_amount,
			"token_b_amount": token_b_amount,
			"total_deposit_usd": round(total_usd, 4),
			"lp_tokens_issued": round(lp_tokens, 8),
			"fee_tier_bps": fee_bps,
			"estimated_apy_pct": protocol_data.get("base_apy_pct", 0.0),
			"slippage_tolerance_pct": slippage_tolerance_pct,
			"deposited_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
			"status": "active",
		}
		self._pool_deposits.setdefault(customer_id, []).append(deposit)
		self._audit(self.tenant_id, "liquidity_pool_deposit", deposit_id)
		return deposit

	async def liquidity_pool_withdraw(
		self,
		customer_id: str,
		pool_id: str,
		lp_tokens: float,
		*,
		min_token_a: float = 0.0,
		min_token_b: float = 0.0,
	) -> dict[str, Any]:
		"""Withdraw liquidity from an AMM pool by burning LP tokens."""
		assert customer_id, "customer_id required"
		assert lp_tokens > 0, "lp_tokens must be positive"

		deposits = self._pool_deposits.get(customer_id, [])
		pool_deposits = [d for d in deposits if d["pool_id"] == pool_id and d["status"] == "active"]
		assert pool_deposits, f"no active deposits found for pool: {pool_id}"

		total_lp = sum(d["lp_tokens_issued"] for d in pool_deposits)
		assert total_lp >= lp_tokens, f"insufficient LP tokens: have {total_lp}, need {lp_tokens}"

		ratio = lp_tokens / total_lp
		total_a = sum(d["token_a_amount"] for d in pool_deposits)
		total_b = sum(d["token_b_amount"] for d in pool_deposits)

		# Impermanent loss approximation (0.5% for illustration)
		il_factor = 0.995
		withdraw_a = total_a * ratio * il_factor
		withdraw_b = total_b * ratio * il_factor

		# Fee income accrued
		fee_income_usd = (
			sum(d["total_deposit_usd"] for d in pool_deposits)
			* ratio
			* pool_deposits[0].get("estimated_apy_pct", 8.0)
			/ 100
			* (30 / 365)  # assume 30-day average hold
		)

		withdrawal_id = str(uuid.uuid4())
		self._audit(self.tenant_id, "liquidity_pool_withdrawal", withdrawal_id)
		return {
			"withdrawal_id": withdrawal_id,
			"customer_id": customer_id,
			"pool_id": pool_id,
			"lp_tokens_burned": lp_tokens,
			"token_a_received": round(withdraw_a, 8),
			"token_b_received": round(withdraw_b, 8),
			"fee_income_usd": round(fee_income_usd, 4),
			"impermanent_loss_factor": il_factor,
			"withdrawn_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
			"status": "settled",
		}

	# ------------------------------------------------------------------
	# Yield farming
	# ------------------------------------------------------------------

	async def yield_farming_enrol(
		self,
		customer_id: str,
		farm_id: str,
		amount: float,
		*,
		token: str = "USDC",
		lock_weeks: int = 0,
	) -> dict[str, Any]:
		"""Enrol a token amount into a yield farm."""
		assert customer_id, "customer_id required"
		assert farm_id, "farm_id required"
		assert amount > 0, "amount must be positive"

		protocol_data = _PROTOCOL_REGISTRY.get(farm_id, {})
		apy_pct = protocol_data.get("base_apy_pct", 5.0)
		bonus_apy = lock_weeks * 0.1  # 0.1% per week locked
		total_apy = min(apy_pct + bonus_apy, 200.0)

		unlock_at = None
		if lock_weeks > 0:
			unlock_at = (
				datetime.datetime.now(datetime.timezone.utc)
				+ datetime.timedelta(weeks=lock_weeks)
			).isoformat()

		position_id = str(uuid.uuid4())
		token_usd = amount * _TOKEN_PRICES.get(token.upper(), 1.0)
		position = {
			"position_id": position_id,
			"customer_id": customer_id,
			"farm_id": farm_id,
			"token": token.upper(),
			"amount": amount,
			"amount_usd": round(token_usd, 4),
			"apy_pct": total_apy,
			"lock_weeks": lock_weeks,
			"unlock_at": unlock_at,
			"estimated_annual_reward_usd": round(token_usd * total_apy / 100, 4),
			"enrolled_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
			"status": "active",
		}
		self._farm_positions.setdefault(customer_id, []).append(position)
		self._audit(self.tenant_id, "yield_farming_enrolled", position_id)
		return position

	async def claim_farming_rewards(
		self,
		customer_id: str,
		farm_id: str,
	) -> dict[str, Any]:
		"""Claim accrued yield farming rewards."""
		positions = [
			p for p in self._farm_positions.get(customer_id, [])
			if p["farm_id"] == farm_id and p["status"] == "active"
		]
		assert positions, f"no active farm positions for: {farm_id}"

		total_reward_usd = sum(
			p["amount_usd"] * p["apy_pct"] / 100 * (7 / 365)  # 7-day accrual stub
			for p in positions
		)
		claim_id = str(uuid.uuid4())
		self._audit(self.tenant_id, "farming_rewards_claimed", claim_id)
		return {
			"claim_id": claim_id,
			"customer_id": customer_id,
			"farm_id": farm_id,
			"positions_count": len(positions),
			"total_reward_usd": round(total_reward_usd, 4),
			"reward_token": "USDC",
			"claimed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
			"status": "settled",
		}

	# ------------------------------------------------------------------
	# Lending & borrowing
	# ------------------------------------------------------------------

	async def lending_deposit(
		self,
		customer_id: str,
		token: str,
		amount: float,
		*,
		protocol_id: str = "aave_v3",
	) -> dict[str, Any]:
		"""Deposit tokens into a lending protocol to earn supply interest."""
		assert customer_id, "customer_id required"
		assert amount > 0, "amount must be positive"
		token_sym = token.upper()
		protocol_data = _PROTOCOL_REGISTRY.get(protocol_id, {})
		supply_apy = protocol_data.get("base_apy_pct", 3.5)
		token_usd = amount * _TOKEN_PRICES.get(token_sym, 1.0)

		# Receive aToken equivalent
		a_token = f"a{token_sym}"
		deposit_id = str(uuid.uuid4())
		deposit = {
			"deposit_id": deposit_id,
			"customer_id": customer_id,
			"protocol_id": protocol_id,
			"token": token_sym,
			"amount": amount,
			"amount_usd": round(token_usd, 4),
			"a_token": a_token,
			"a_token_amount": amount,  # 1:1 at deposit
			"supply_apy_pct": supply_apy,
			"estimated_annual_interest_usd": round(token_usd * supply_apy / 100, 4),
			"deposited_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
			"status": "active",
		}
		self._audit(self.tenant_id, "lending_deposit_made", deposit_id)
		return deposit

	async def borrow_against_collateral(
		self,
		customer_id: str,
		collateral_token: str,
		borrow_token: str,
		amount: float,
		*,
		protocol_id: str = "aave_v3",
		collateral_amount: float | None = None,
	) -> dict[str, Any]:
		"""Borrow a token using another as collateral. Returns loan record."""
		assert customer_id, "customer_id required"
		assert amount > 0, "amount must be positive"
		c_sym = collateral_token.upper()
		b_sym = borrow_token.upper()

		borrow_usd = amount * _TOKEN_PRICES.get(b_sym, 1.0)
		# LTV ratio: 75% standard
		ltv = 0.75
		required_collateral_usd = borrow_usd / ltv
		collateral_price = _TOKEN_PRICES.get(c_sym, 1.0)
		required_collateral = required_collateral_usd / collateral_price

		actual_collateral = collateral_amount or required_collateral * 1.2  # 20% buffer
		actual_collateral_usd = actual_collateral * collateral_price
		health_factor = actual_collateral_usd * ltv / borrow_usd

		# Borrow APY
		borrow_apy_pct = 6.5  # stub

		loan_id = str(uuid.uuid4())
		loan = {
			"loan_id": loan_id,
			"customer_id": customer_id,
			"protocol_id": protocol_id,
			"collateral_token": c_sym,
			"collateral_amount": actual_collateral,
			"collateral_usd": round(actual_collateral_usd, 4),
			"borrow_token": b_sym,
			"borrow_amount": amount,
			"borrow_usd": round(borrow_usd, 4),
			"ltv_ratio": ltv,
			"health_factor": round(health_factor, 4),
			"health_factor_bps": int(health_factor * 10_000),
			"borrow_apy_pct": borrow_apy_pct,
			"estimated_annual_interest_usd": round(borrow_usd * borrow_apy_pct / 100, 4),
			"liquidation_threshold": 0.825,
			"borrowed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
			"status": "active",
		}
		self._loans[loan_id] = loan
		self._audit(self.tenant_id, "collateral_loan_opened", loan_id)
		return loan

	async def repay_loan(
		self,
		loan_id: str,
		amount: float,
		*,
		full_repay: bool = False,
	) -> dict[str, Any]:
		"""Repay part or all of a collateral loan."""
		loan = self._loans.get(loan_id)
		assert loan is not None, f"loan not found: {loan_id}"
		assert loan["status"] == "active", f"loan not active: {loan['status']}"

		outstanding = loan["borrow_amount"]
		repay_amount = outstanding if full_repay else min(amount, outstanding)
		remaining = outstanding - repay_amount
		loan["borrow_amount"] = remaining

		if remaining <= 0:
			loan["status"] = "repaid"
			# Release collateral
			collateral_released = loan["collateral_amount"]
		else:
			collateral_released = 0.0
			# Recalculate health factor
			borrow_usd = remaining * _TOKEN_PRICES.get(loan["borrow_token"], 1.0)
			collateral_usd = loan["collateral_usd"]
			loan["health_factor"] = round(collateral_usd * loan["ltv_ratio"] / borrow_usd, 4) if borrow_usd > 0 else 999.0
			loan["health_factor_bps"] = int(loan["health_factor"] * 10_000)

		self._audit(self.tenant_id, "loan_repaid", loan_id)
		return {
			"loan_id": loan_id,
			"repaid_amount": repay_amount,
			"remaining_balance": remaining,
			"collateral_released": collateral_released,
			"status": loan["status"],
			"health_factor": loan["health_factor"],
			"repaid_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def collateral_health_factor(self, loan_id: str) -> dict[str, Any]:
		"""Return current health factor and risk status for a loan."""
		loan = self._loans.get(loan_id)
		assert loan is not None, f"loan not found: {loan_id}"

		hf = loan["health_factor"]
		hf_bps = loan["health_factor_bps"]

		if hf_bps < _LIQUIDATION_HF_BPS:
			risk_level = "liquidatable"
		elif hf_bps < _CRITICAL_HF_BPS:
			risk_level = "critical"
		elif hf_bps < _SAFE_HF_BPS:
			risk_level = "warning"
		else:
			risk_level = "safe"

		self._audit(self.tenant_id, "health_factor_checked", loan_id)
		return {
			"loan_id": loan_id,
			"health_factor": hf,
			"health_factor_bps": hf_bps,
			"risk_level": risk_level,
			"liquidation_threshold_bps": _LIQUIDATION_HF_BPS,
			"safe_threshold_bps": _SAFE_HF_BPS,
			"collateral_token": loan["collateral_token"],
			"borrow_token": loan["borrow_token"],
			"outstanding_borrow": loan["borrow_amount"],
			"checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def liquidation_risk_alert(self, loan_id: str) -> dict[str, Any]:
		"""Evaluate and register a liquidation risk alert for a loan."""
		hf_data = await self.collateral_health_factor(loan_id)
		loan = self._loans[loan_id]

		alert_required = hf_data["health_factor_bps"] < _SAFE_HF_BPS
		alert = {
			"alert_id": str(uuid.uuid4()),
			"loan_id": loan_id,
			"customer_id": loan["customer_id"],
			"risk_level": hf_data["risk_level"],
			"health_factor": hf_data["health_factor"],
			"health_factor_bps": hf_data["health_factor_bps"],
			"alert_required": alert_required,
			"recommended_action": (
				"add_collateral" if hf_data["risk_level"] in {"critical", "warning"}
				else ("repay_debt" if hf_data["risk_level"] == "liquidatable" else "none")
			),
			"created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

		if alert_required:
			self._liquidation_alerts.append(alert)
			self._audit(self.tenant_id, "liquidation_alert_raised", loan_id)

		return alert

	# ------------------------------------------------------------------
	# AMM swap
	# ------------------------------------------------------------------

	async def amm_swap(
		self,
		customer_id: str,
		token_in: str,
		token_out: str,
		amount_in: float,
		*,
		protocol_id: str = "uniswap_v3",
		slippage_tolerance_pct: float = 0.5,
	) -> dict[str, Any]:
		"""Execute a token swap through an AMM protocol."""
		assert customer_id, "customer_id required"
		assert amount_in > 0, "amount_in must be positive"
		t_in = token_in.upper()
		t_out = token_out.upper()
		assert t_in != t_out, "cannot swap token for itself"

		price_in = _TOKEN_PRICES.get(t_in, 1.0)
		price_out = _TOKEN_PRICES.get(t_out, 1.0)
		usd_value = amount_in * price_in
		fee_bps = _AMM_FEE_TIERS.get(protocol_id, 30)
		fee_pct = fee_bps / 10_000
		fee_usd = usd_value * fee_pct
		net_usd = usd_value - fee_usd
		# Price impact: simplified 0.02% per 1000 USD traded
		price_impact_pct = usd_value / 1_000_000 * 2.0
		amount_out = (net_usd / price_out) * (1 - price_impact_pct / 100)

		min_amount_out = amount_out * (1 - slippage_tolerance_pct / 100)

		swap_id = str(uuid.uuid4())
		swap = {
			"swap_id": swap_id,
			"customer_id": customer_id,
			"protocol_id": protocol_id,
			"token_in": t_in,
			"token_out": t_out,
			"amount_in": amount_in,
			"amount_out": round(amount_out, 8),
			"min_amount_out": round(min_amount_out, 8),
			"price_in_usd": price_in,
			"price_out_usd": price_out,
			"implied_rate": round(price_in / price_out, 8),
			"fee_bps": fee_bps,
			"fee_usd": round(fee_usd, 4),
			"price_impact_pct": round(price_impact_pct, 4),
			"slippage_tolerance_pct": slippage_tolerance_pct,
			"usd_value": round(usd_value, 4),
			"executed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
			"status": "settled",
		}
		self._swaps.append(swap)
		self._audit(self.tenant_id, "amm_swap_executed", swap_id)
		return swap

	# ------------------------------------------------------------------
	# Portfolio summary
	# ------------------------------------------------------------------

	async def portfolio_defi_summary(self, customer_id: str) -> dict[str, Any]:
		"""Aggregate DeFi positions: pools, farms, loans, and rewards."""
		pool_deposits = self._pool_deposits.get(customer_id, [])
		farm_positions = self._farm_positions.get(customer_id, [])
		loans = [l for l in self._loans.values() if l["customer_id"] == customer_id]
		customer_swaps = [s for s in self._swaps if s["customer_id"] == customer_id]

		active_pool_usd = sum(
			d["total_deposit_usd"] for d in pool_deposits if d["status"] == "active"
		)
		active_farm_usd = sum(
			p["amount_usd"] for p in farm_positions if p["status"] == "active"
		)
		total_borrowed_usd = sum(
			l["borrow_usd"] for l in loans if l["status"] == "active"
		)
		total_collateral_usd = sum(
			l["collateral_usd"] for l in loans if l["status"] == "active"
		)
		net_usd = active_pool_usd + active_farm_usd + total_collateral_usd - total_borrowed_usd

		at_risk_loans = [
			l["loan_id"] for l in loans
			if l["status"] == "active" and l.get("health_factor_bps", 99999) < _SAFE_HF_BPS
		]

		self._audit(self.tenant_id, "defi_portfolio_summary_generated", customer_id)
		return {
			"customer_id": customer_id,
			"pool_deposit_count": len([d for d in pool_deposits if d["status"] == "active"]),
			"active_pool_usd": round(active_pool_usd, 4),
			"farm_position_count": len([p for p in farm_positions if p["status"] == "active"]),
			"active_farm_usd": round(active_farm_usd, 4),
			"active_loan_count": len([l for l in loans if l["status"] == "active"]),
			"total_borrowed_usd": round(total_borrowed_usd, 4),
			"total_collateral_usd": round(total_collateral_usd, 4),
			"net_position_usd": round(net_usd, 4),
			"total_swaps": len(customer_swaps),
			"at_risk_loan_ids": at_risk_loans,
			"liquidation_alerts_pending": len(self._liquidation_alerts),
			"generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	# ------------------------------------------------------------------
	# Risk dashboard
	# ------------------------------------------------------------------

	async def defi_risk_dashboard(
		self,
		*,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""System-wide DeFi risk dashboard for the tenant."""
		tid = tenant_id or self.tenant_id
		all_loans = list(self._loans.values())
		critical_loans = [
			l for l in all_loans
			if l["status"] == "active" and l.get("health_factor_bps", 99999) < _CRITICAL_HF_BPS
		]
		liquidatable_loans = [
			l for l in all_loans
			if l["status"] == "active" and l.get("health_factor_bps", 99999) < _LIQUIDATION_HF_BPS
		]

		protocols_by_risk = {
			"low":    [k for k, v in _PROTOCOL_REGISTRY.items() if v["tvl_usd"] > 5_000_000_000],
			"medium": [k for k, v in _PROTOCOL_REGISTRY.items() if 1_000_000_000 <= v["tvl_usd"] <= 5_000_000_000],
			"high":   [k for k, v in _PROTOCOL_REGISTRY.items() if v["tvl_usd"] < 1_000_000_000],
		}

		total_pool_usd = sum(
			d["total_deposit_usd"]
			for deposits in self._pool_deposits.values()
			for d in deposits
			if d["status"] == "active"
		)
		total_farm_usd = sum(
			p["amount_usd"]
			for positions in self._farm_positions.values()
			for p in positions
			if p["status"] == "active"
		)

		risk_assessments = [
			r for r in self.risk_assessments.values() if r.tenant_id == tid
		]
		high_risk_count = sum(
			1 for r in risk_assessments if r.risk_tier in {"high", "critical"}
		)

		self._audit(tid, "defi_risk_dashboard_generated", tid)
		return {
			"tenant_id": tid,
			"total_active_loans": len([l for l in all_loans if l["status"] == "active"]),
			"critical_loan_count": len(critical_loans),
			"liquidatable_loan_count": len(liquidatable_loans),
			"liquidation_alerts_total": len(self._liquidation_alerts),
			"total_pool_deposits_usd": round(total_pool_usd, 4),
			"total_farm_deposits_usd": round(total_farm_usd, 4),
			"protocol_risk_breakdown": protocols_by_risk,
			"risk_assessment_count": len(risk_assessments),
			"high_risk_assessment_count": high_risk_count,
			"swap_count": len(self._swaps),
			"protocols_monitored": len(_PROTOCOL_REGISTRY),
			"generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def protocol_analytics(
		self,
		protocol_id: str,
		period: str,
	) -> dict[str, Any]:
		"""Compute usage analytics for a specific DeFi protocol."""
		protocol_data = _PROTOCOL_REGISTRY.get(protocol_id, {})
		swaps_on_protocol = [s for s in self._swaps if s["protocol_id"] == protocol_id]
		swap_volumes = [s["usd_value"] for s in swaps_on_protocol]

		self._audit(self.tenant_id, "protocol_analytics_computed", protocol_id)
		return {
			"protocol_id": protocol_id,
			"period": period,
			"chain": protocol_data.get("chain", "unknown"),
			"protocol_type": protocol_data.get("type", "unknown"),
			"tvl_usd": protocol_data.get("tvl_usd", 0),
			"base_apy_pct": protocol_data.get("base_apy_pct", 0.0),
			"swap_count": len(swaps_on_protocol),
			"swap_volume_usd": round(sum(swap_volumes), 4),
			"avg_swap_usd": round(statistics.mean(swap_volumes), 4) if swap_volumes else 0.0,
			"computed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def liquidation_history(
		self,
		customer_id: str | None = None,
	) -> list[dict[str, Any]]:
		"""Return liquidation alert history, optionally filtered by customer."""
		if customer_id:
			return [a for a in self._liquidation_alerts if a.get("customer_id") == customer_id]
		return list(self._liquidation_alerts)

	# ------------------------------------------------------------------
	# Existing core methods (preserved from original)
	# ------------------------------------------------------------------

	def register_protocol(
		self,
		protocol_id: str,
		tenant_id: str,
		protocol_type: str,
		network_reference: str,
		protocol_reference: str,
		owner_id: str,
		evidence_reference: str,
		risk_tier: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		protocol_type = normalize_code(protocol_type)
		risk_tier = normalize_code(risk_tier)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "register_protocol",
			"protocol_type_supported": protocol_type in SUPPORTED_PROTOCOL_TYPES,
			"network_present": present(network_reference),
			"protocol_reference_present": present(protocol_reference),
			"owner_present": present(owner_id),
			"evidence_present": present(evidence_reference),
			"risk_tier_supported": risk_tier in SUPPORTED_RISK_TIERS,
		})
		item = DeFiProtocol(
			protocol_id, tenant_id, protocol_type, network_reference,
			protocol_reference, owner_id, evidence_reference, risk_tier,
		)
		self.protocols[protocol_id] = item
		self._audit(tenant_id, "defi_protocol_registered", protocol_id)
		return item.to_dict()

	def open_position(
		self,
		position_id: str,
		tenant_id: str,
		protocol_id: str,
		account_reference: str,
		asset_pair_reference: str,
		position_type: str,
		amount_minor: int,
		collateral_minor: int,
		health_factor_bps: int,
		evidence_reference: str,
	) -> dict[str, Any]:
		protocol = self._tenant_protocol_or_none(protocol_id, tenant_id)
		position_type = normalize_code(position_type)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "open_position",
			"protocol_present": protocol is not None,
			"account_present": present(account_reference),
			"asset_pair_present": present(asset_pair_reference),
			"position_type_supported": position_type in SUPPORTED_POSITION_TYPES,
			"amount_valid": positive_int(amount_minor),
			"collateral_valid": non_negative_int(collateral_minor),
			"health_factor_valid": positive_int(health_factor_bps),
			"evidence_present": present(evidence_reference),
		})
		item = LiquidityPosition(
			position_id, tenant_id, protocol_id, account_reference,
			asset_pair_reference, position_type, int(amount_minor),
			int(collateral_minor), int(health_factor_bps), evidence_reference,
		)
		self.positions[position_id] = item
		self._audit(tenant_id, "defi_position_opened", position_id)
		return item.to_dict()

	def record_action(
		self,
		action_id: str,
		tenant_id: str,
		protocol_id: str,
		position_id: str,
		action_type: str,
		amount_minor: int,
		requester_id: str,
		approval_reference: str,
		evidence_reference: str,
		status: str = "requested",
	) -> dict[str, Any]:
		protocol = self._tenant_protocol_or_none(protocol_id, tenant_id)
		position = self._tenant_position_or_none(position_id, tenant_id)
		action_type = normalize_code(action_type)
		status = normalize_code(status)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_action",
			"protocol_present": protocol is not None,
			"position_present": position is not None,
			"position_protocol_match": position is not None and position.protocol_id == protocol_id,
			"action_type_supported": action_type in SUPPORTED_ACTION_TYPES,
			"amount_valid": positive_int(amount_minor),
			"requester_present": present(requester_id),
			"approval_present": present(approval_reference),
			"evidence_present": present(evidence_reference),
			"status_supported": status in SUPPORTED_ACTION_STATUSES,
		})
		item = DeFiAction(
			action_id, tenant_id, protocol_id, position_id, action_type,
			int(amount_minor), requester_id, approval_reference, evidence_reference, status,
		)
		self.actions[action_id] = item
		self._audit(tenant_id, "defi_action_recorded", action_id)
		return item.to_dict()

	def register_yield_strategy(
		self,
		strategy_id: str,
		tenant_id: str,
		protocol_id: str,
		strategy_reference: str,
		target_apy_bps: int,
		max_risk_tier: str,
		owner_id: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		protocol = self._tenant_protocol_or_none(protocol_id, tenant_id)
		max_risk_tier = normalize_code(max_risk_tier)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_yield_strategy",
			"protocol_present": protocol is not None,
			"strategy_reference_present": present(strategy_reference),
			"target_apy_valid": non_negative_int(target_apy_bps),
			"max_risk_supported": max_risk_tier in SUPPORTED_RISK_TIERS,
			"owner_present": present(owner_id),
			"evidence_present": present(evidence_reference),
		})
		item = YieldStrategy(
			strategy_id, tenant_id, protocol_id, strategy_reference,
			int(target_apy_bps), max_risk_tier, owner_id, evidence_reference,
		)
		self.strategies[strategy_id] = item
		self._audit(tenant_id, "defi_yield_strategy_registered", strategy_id)
		return item.to_dict()

	def record_reward(
		self,
		reward_id: str,
		tenant_id: str,
		position_id: str,
		reward_type: str,
		asset_reference: str,
		amount_minor: int,
		evidence_reference: str,
	) -> dict[str, Any]:
		position = self._tenant_position_or_none(position_id, tenant_id)
		reward_type = normalize_code(reward_type)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_reward",
			"position_present": position is not None,
			"reward_type_supported": reward_type in SUPPORTED_REWARD_TYPES,
			"asset_present": present(asset_reference),
			"amount_valid": positive_int(amount_minor),
			"evidence_present": present(evidence_reference),
		})
		item = RewardAccrual(
			reward_id, tenant_id, position_id, reward_type,
			asset_reference, int(amount_minor), evidence_reference,
		)
		self.rewards[reward_id] = item
		self._audit(tenant_id, "defi_reward_recorded", reward_id)
		return item.to_dict()

	def record_governance_vote(
		self,
		proposal_id: str,
		tenant_id: str,
		protocol_id: str,
		proposal_reference: str,
		vote_choice: str,
		voter_id: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		protocol = self._tenant_protocol_or_none(protocol_id, tenant_id)
		vote_choice = normalize_code(vote_choice)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_governance_vote",
			"protocol_present": protocol is not None,
			"proposal_present": present(proposal_reference),
			"vote_supported": vote_choice in SUPPORTED_GOVERNANCE_VOTES,
			"voter_present": present(voter_id),
			"evidence_present": present(evidence_reference),
		})
		item = GovernanceProposal(
			proposal_id, tenant_id, protocol_id, proposal_reference,
			vote_choice, voter_id, evidence_reference,
		)
		self.governance[proposal_id] = item
		self._audit(tenant_id, "defi_governance_vote_recorded", proposal_id)
		return item.to_dict()

	def record_risk_assessment(
		self,
		assessment_id: str,
		tenant_id: str,
		reference_id: str,
		risk_tier: str,
		reviewer_id: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		risk_tier = normalize_code(risk_tier)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_risk_assessment",
			"reference_present": present(reference_id),
			"risk_tier_supported": risk_tier in SUPPORTED_RISK_TIERS,
			"reviewer_present": present(reviewer_id),
			"evidence_present": present(evidence_reference),
		})
		item = RiskAssessment(assessment_id, tenant_id, reference_id, risk_tier, reviewer_id, evidence_reference)
		self.risk_assessments[assessment_id] = item
		self._audit(tenant_id, "defi_risk_assessment_recorded", assessment_id)
		return item.to_dict()

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
			"reviewer_present": present(reviewer_id),
			"evidence_present": present(evidence_reference),
		})
		item = DeFiReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[review_id] = item
		self._audit(tenant_id, "defi_review_recorded", review_id)
		return item.to_dict()

	def register_defi_agent(
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
			"operation": "register_defi_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
		})
		item = DeFiAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[agent_id] = item
		self._audit(tenant_id, "defi_agent_registered", agent_id)
		return item.to_dict()

	def validate_agent_action(
		self,
		tenant_id: str,
		privileged_scope: bool,
		human_approval_recorded: bool,
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation": "defi_agent_action",
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
		})
		return {"tenant_id": tenant_id, "accepted": True, "privileged_scope": privileged_scope}

	def validate_batch(
		self,
		tenant_id: str,
		item_count: int,
		event_stream: str = "bytewax",
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation": "defi_batch",
			"event_stream": event_stream,
		})
		return {
			"tenant_id": tenant_id,
			"item_count": item_count,
			"processor": "bytewax",
			"stream": "apg.fintech.defi.lifecycle",
			"accepted": True,
		}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"protocol_count": self._count(self.protocols, tenant_id),
			"position_count": self._count(self.positions, tenant_id),
			"critical_protocol_count": sum(
				1 for item in self.protocols.values()
				if item.tenant_id == tenant_id and item.risk_tier == "critical"
			),
			"action_count": self._count(self.actions, tenant_id),
			"open_action_count": sum(
				1 for item in self.actions.values()
				if item.tenant_id == tenant_id and item.status in {"requested", "approved", "submitted"}
			),
			"yield_strategy_count": self._count(self.strategies, tenant_id),
			"reward_count": self._count(self.rewards, tenant_id),
			"governance_vote_count": self._count(self.governance, tenant_id),
			"risk_assessment_count": self._count(self.risk_assessments, tenant_id),
			"review_count": self._count(self.reviews, tenant_id),
			"agent_count": self._count(self.agents, tenant_id),
			"active_loan_count": len([l for l in self._loans.values() if l["status"] == "active"]),
			"liquidation_alert_count": len(self._liquidation_alerts),
			"swap_count": len(self._swaps),
			"audit_event_count": sum(1 for ev in self.audit_events if ev["tenant_id"] == tenant_id),
			"streaming": get_capability_contract(tenant_id)["streaming"],
		}

	# ------------------------------------------------------------------
	# Additional async methods
	# ------------------------------------------------------------------

	async def health_check(self) -> dict[str, Any]:
		"""Return DeFi service health status."""
		return {
			"service": "defi", "status": "healthy",
			"active_loans": len([l for l in self._loans.values() if l["status"] == "active"]),
			"liquidation_alerts": len(self._liquidation_alerts),
			"checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def yield_optimizer(self, customer_id: str, amount: float, token: str, risk_tolerance: str = "medium") -> dict[str, Any]:
		"""Find the optimal yield farming strategy for a given amount and risk tolerance."""
		protocols_by_risk = {
			"low": ["lido", "aave_v3"],
			"medium": ["compound_v3", "yearn_v3", "curve_3pool"],
			"high": ["uniswap_v3", "pancakeswap_v3"],
		}
		candidates = protocols_by_risk.get(risk_tolerance, protocols_by_risk["medium"])
		best_protocol = max(candidates, key=lambda p: _PROTOCOL_REGISTRY.get(p, {}).get("base_apy_pct", 0))
		best_apy = _PROTOCOL_REGISTRY.get(best_protocol, {}).get("base_apy_pct", 5.0)
		self._audit(self.tenant_id, "yield_optimizer_run", customer_id)
		return {
			"customer_id": customer_id, "amount": amount, "token": token.upper(),
			"risk_tolerance": risk_tolerance, "recommended_protocol": best_protocol,
			"expected_apy_pct": best_apy,
			"estimated_annual_yield_usd": round(amount * _TOKEN_PRICES.get(token.upper(), 1.0) * best_apy / 100, 4),
			"optimized_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def flash_loan_simulation(self, amount: float, token: str, strategy: str) -> dict[str, Any]:
		"""Simulate a flash loan arbitrage strategy (no real capital required)."""
		token_sym = token.upper()
		flash_fee_pct = 0.09
		fee = amount * flash_fee_pct / 100
		simulated_profit = amount * 0.02 - fee
		return {
			"simulation_id": str(uuid.uuid4()), "token": token_sym, "amount": amount,
			"flash_fee_pct": flash_fee_pct, "fee": fee,
			"strategy": strategy, "simulated_profit": round(simulated_profit, 4),
			"profitable": simulated_profit > 0, "status": "simulated",
			"simulated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def impermanent_loss_calculator(self, token_a: str, token_b: str, initial_price_ratio: float, current_price_ratio: float, initial_amount_usd: float) -> dict[str, Any]:
		"""Calculate impermanent loss for an AMM position given price ratio change."""
		import math
		ratio = current_price_ratio / initial_price_ratio
		il = 2 * math.sqrt(ratio) / (1 + ratio) - 1
		il_amount = initial_amount_usd * abs(il)
		return {
			"token_a": token_a.upper(), "token_b": token_b.upper(),
			"initial_price_ratio": initial_price_ratio, "current_price_ratio": current_price_ratio,
			"price_change_pct": round((ratio - 1) * 100, 2),
			"impermanent_loss_pct": round(il * 100, 4),
			"impermanent_loss_usd": round(il_amount, 4),
			"hold_vs_provide_diff_usd": round(-il_amount, 4),
			"calculated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def defi_portfolio_rebalance(self, customer_id: str, target_allocation: dict[str, float]) -> dict[str, Any]:
		"""Rebalance DeFi portfolio towards a target protocol allocation."""
		pool_deposits = self._pool_deposits.get(customer_id, [])
		farm_positions = self._farm_positions.get(customer_id, [])
		total_usd = sum(d["total_deposit_usd"] for d in pool_deposits if d["status"] == "active")
		total_usd += sum(p["amount_usd"] for p in farm_positions if p["status"] == "active")
		trades = []
		for protocol, target_pct in target_allocation.items():
			target_usd = total_usd * target_pct / 100
			current_usd = sum(d["total_deposit_usd"] for d in pool_deposits if d["pool_id"] == protocol and d["status"] == "active")
			delta = target_usd - current_usd
			if abs(delta) > 10:
				trades.append({"protocol": protocol, "action": "increase" if delta > 0 else "decrease", "amount_usd": abs(round(delta, 2))})
		self._audit(self.tenant_id, "defi_rebalance_computed", customer_id)
		return {
			"customer_id": customer_id, "total_value_usd": round(total_usd, 4),
			"target_allocation": target_allocation, "trades_required": trades,
			"computed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def staking_rewards_harvest(self, customer_id: str) -> dict[str, Any]:
		"""Harvest all pending staking rewards for a customer across farms."""
		total_harvested_usd = 0.0
		claims = []
		for farm_id in {p["farm_id"] for p in self._farm_positions.get(customer_id, []) if p["status"] == "active"}:
			try:
				claim = await self.claim_farming_rewards(customer_id, farm_id)
				total_harvested_usd += claim.get("total_reward_usd", 0)
				claims.append(claim)
			except Exception as _exc:
				_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
		self._audit(self.tenant_id, "staking_rewards_harvested", customer_id)
		return {
			"customer_id": customer_id, "farms_harvested": len(claims),
			"total_harvested_usd": round(total_harvested_usd, 4),
			"claims": claims, "harvested_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def gas_fee_estimator(self, chain: str, operation: str) -> dict[str, Any]:
		"""Estimate gas fees for a DeFi operation on a specified chain."""
		chain_lower = chain.lower()
		base_gas = {"ethereum": 21_000, "polygon": 21_000, "bsc": 21_000, "arbitrum": 21_000}
		op_multiplier = {"swap": 3.5, "deposit": 2.5, "withdraw": 2.8, "stake": 2.0, "borrow": 3.0, "repay": 2.5}
		gas_limit = int(base_gas.get(chain_lower, 21_000) * op_multiplier.get(operation.lower(), 2.0))
		gas_price_gwei = {"ethereum": 28.5, "polygon": 3.2, "bsc": 5.0, "arbitrum": 0.1}.get(chain_lower, 20.0)
		fee_eth = gas_limit * gas_price_gwei / 1e9
		eth_price = _TOKEN_PRICES.get("ETH", 3480.0)
		fee_usd = fee_eth * eth_price
		return {
			"chain": chain, "operation": operation, "gas_limit": gas_limit,
			"gas_price_gwei": gas_price_gwei, "fee_eth": round(fee_eth, 8),
			"fee_usd": round(fee_usd, 4), "estimated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def defi_risk_score(self, customer_id: str) -> dict[str, Any]:
		"""Compute a DeFi-specific risk score for a customer based on positions and leverage."""
		loans = [l for l in self._loans.values() if l["customer_id"] == customer_id and l["status"] == "active"]
		avg_hf = sum(l.get("health_factor", 2.0) for l in loans) / max(len(loans), 1)
		leverage = sum(l.get("borrow_usd", 0) for l in loans)
		risk_score = max(0, 100 - int(avg_hf * 20)) + min(50, int(leverage / 10_000))
		risk_score = min(100, risk_score)
		return {
			"customer_id": customer_id, "active_loans": len(loans),
			"avg_health_factor": round(avg_hf, 4), "total_leverage_usd": round(leverage, 2),
			"defi_risk_score": risk_score, "risk_level": "high" if risk_score >= 70 else "medium" if risk_score >= 40 else "low",
			"computed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def compound_interest_calculator(self, principal: float, apy_pct: float, years: float) -> dict[str, Any]:
		"""Calculate compound interest for a DeFi deposit."""
		final_value = principal * ((1 + apy_pct / 100) ** years)
		return {
			"principal": principal, "apy_pct": apy_pct, "years": years,
			"final_value": round(final_value, 4), "interest_earned": round(final_value - principal, 4),
			"calculated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def tvl_dashboard(self) -> dict[str, Any]:
		"""Return Total Value Locked (TVL) dashboard across all protocols."""
		protocol_tvl = {pid: meta["tvl_usd"] for pid, meta in _PROTOCOL_REGISTRY.items()}
		total_tvl = sum(protocol_tvl.values())
		self._audit(self.tenant_id, "tvl_dashboard_generated", self.tenant_id)
		return {
			"total_tvl_usd": total_tvl, "by_protocol": protocol_tvl,
			"protocol_count": len(protocol_tvl), "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def lending_rate_feed(self) -> dict[str, Any]:
		"""Return current supply and borrow rates across lending protocols."""
		rates = {
			"aave_v3": {"eth_supply_apy": 4.2, "eth_borrow_apy": 6.1, "usdc_supply_apy": 5.8, "usdc_borrow_apy": 7.2},
			"compound_v3": {"eth_supply_apy": 3.8, "eth_borrow_apy": 5.5, "usdc_supply_apy": 5.2, "usdc_borrow_apy": 6.8},
		}
		return {"rates": rates, "source": "indicative", "fetched_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}

	async def export_defi_data(self, customer_id: str, fmt: str = "json") -> dict[str, Any]:
		"""Export DeFi position data for a customer."""
		assert fmt in {"json", "csv", "excel"}
		summary = await self.portfolio_defi_summary(customer_id)
		return {
			"customer_id": customer_id, "format": fmt,
			"pool_count": summary["pool_deposit_count"],
			"farm_count": summary["farm_position_count"],
			"file_reference": f"defi_{customer_id}_{fmt}",
			"generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	# ------------------------------------------------------------------
	# Internal helpers
	# ------------------------------------------------------------------

	def _tenant_protocol_or_none(self, item_id: str, tenant_id: str) -> DeFiProtocol | None:
		item = self.protocols.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_position_or_none(self, item_id: str, tenant_id: str) -> LiquidityPosition | None:
		item = self.positions.get(item_id)
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
			action.get("reason", "defi_policy_denied")
			for action in result["actions"]
		)
		raise PermissionError(reasons or "defi_policy_denied")


FintechDeFiService = DecentralizedFinanceService
DecentralisedFinanceService = DecentralizedFinanceService
