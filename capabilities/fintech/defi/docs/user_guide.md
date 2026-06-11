# Decentralized Finance — User Guide

**Capability ID**: `fintech_defi` | **Domain**: `fintech` | **Version**: `1.2.0`
© 2025 Datacraft — www.datacraft.co.ke

---

## Overview

`fintech_defi` is the APG capability for governed DeFi operations. It covers the full DeFi lifecycle: AMM liquidity provision, yield farming, lending and borrowing, token swaps, governance participation, risk assessment, and analytics. All operations are tenant-scoped, audited, and streamed via Bytewax+NATS.

---

## Installation

```bash
pip install apg-fintech-defi
```

---

## Quick Start

```python
from apg_fintech_defi.service import DecentralizedFinanceService

svc = DecentralizedFinanceService(tenant_id="acme", actor_id="trader-01")

# Deposit into a liquidity pool
deposit = await svc.liquidity_pool_deposit(
    "customer-123", "uniswap_v3",
    token_a_amount=1000.0, token_b_amount=0.287,
    token_a="USDC", token_b="ETH",
)

# Borrow against collateral
loan = await svc.borrow_against_collateral(
    "customer-123", "ETH", "USDC", 5000.0,
    protocol_id="aave_v3",
)

# Check health factor
hf = await svc.collateral_health_factor(loan["loan_id"])
print(hf["risk_level"])  # "safe" | "warning" | "critical" | "liquidatable"
```

---

## Core Features

### Liquidity Pool Management

#### Deposit

```python
result = await svc.liquidity_pool_deposit(
    customer_id="cust-001",
    pool_id="curve_3pool",
    token_a_amount=10_000.0,
    token_b_amount=10_000.0,
    token_a="USDC",
    token_b="DAI",
    slippage_tolerance_pct=0.1,
)
# Returns: deposit_id, lp_tokens_issued, total_deposit_usd, estimated_apy_pct
```

#### Withdraw

```python
result = await svc.liquidity_pool_withdraw(
    customer_id="cust-001",
    pool_id="curve_3pool",
    lp_tokens=99.5,
    min_token_a=9900.0,
    min_token_b=9900.0,
)
# Returns: token_a_received, token_b_received, fee_income_usd, impermanent_loss_factor
```

---

### Yield Farming

#### Enrol

```python
position = await svc.yield_farming_enrol(
    "cust-001", "yearn_v3", 50_000.0,
    token="USDC",
    lock_weeks=12,  # +0.1% APY bonus per lock week
)
# Returns: position_id, apy_pct, lock_weeks, unlock_at, estimated_annual_reward_usd
```

#### Claim Rewards

```python
claim = await svc.claim_farming_rewards("cust-001", "yearn_v3")
# Returns: claim_id, total_reward_usd, positions_count
```

#### Bulk Harvest

```python
harvest = await svc.staking_rewards_harvest("cust-001")
# Harvests all active farms in one call
# Returns: farms_harvested, total_harvested_usd, claims[]
```

---

### Lending and Borrowing

#### Supply to Lending Protocol

```python
deposit = await svc.lending_deposit(
    "cust-001", "USDC", 100_000.0,
    protocol_id="aave_v3",
)
# Returns: a_token, a_token_amount, supply_apy_pct, estimated_annual_interest_usd
```

#### Borrow Against Collateral

```python
loan = await svc.borrow_against_collateral(
    "cust-001",
    collateral_token="ETH",
    borrow_token="USDC",
    amount=10_000.0,
    protocol_id="aave_v3",
    collateral_amount=5.0,  # optional: specify exact collateral
)
# Returns: loan_id, health_factor, health_factor_bps, borrow_apy_pct, liquidation_threshold
```

#### Repay

```python
result = await svc.repay_loan(loan["loan_id"], 5_000.0)
result = await svc.repay_loan(loan["loan_id"], 0, full_repay=True)
```

#### Health Factor Monitoring

```python
hf = await svc.collateral_health_factor(loan["loan_id"])
# risk_level: "safe" (≥150%), "warning" (120-150%), "critical" (110-120%), "liquidatable" (<110%)

alert = await svc.liquidation_risk_alert(loan["loan_id"])
# Registers alert if health_factor_bps < 15000 (safe threshold)
```

---

### AMM Swaps

#### Single-Hop Swap

```python
swap = await svc.amm_swap(
    "cust-001", "USDC", "ETH", 3480.0,
    protocol_id="uniswap_v3",
    slippage_tolerance_pct=0.5,
)
# Returns: amount_out, fee_usd, price_impact_pct, implied_rate
```

#### MEV-Resistant Multi-Route Swap

```python
routed = await svc.smart_route_swap(
    "cust-001", "USDC", "ETH", 50_000.0,
    max_splits=3,
    slippage_tolerance_pct=0.3,
)
# Splits across uniswap_v3, pancakeswap_v3, curve_3pool by TVL weight
# Returns: total_amount_out, routes[], weighted_price_impact_pct, mev_resistant=True
```

---

### Atomic Collateral Substitution

Swap collateral while maintaining health factor in a single operation:

```python
result = await svc.atomic_collateral_swap(
    loan_id=loan["loan_id"],
    new_collateral_token="WBTC",
    new_collateral_amount=0.5,
)
# Asserts new health_factor > liquidation threshold before committing
# Returns: old/new collateral details, flash_fee_usd, new_health_factor
```

---

### Liquid Staking Optimiser

```python
recommendation = await svc.liquid_staking_optimiser(
    customer_id="cust-001",
    stake_amount=10.0,
    token="ETH",
)
# Compares Lido, Rocket Pool, cbETH etc.
# Returns: best_recommendation (combined_defi_apy_pct, integration_path), all_options[]
```

---

### Portfolio and Analytics

#### Portfolio Summary

```python
summary = await svc.portfolio_defi_summary("cust-001")
# Returns: active_pool_usd, active_farm_usd, total_borrowed_usd, net_position_usd, at_risk_loan_ids
```

#### Yield Optimiser

```python
opt = await svc.yield_optimizer(
    "cust-001", 100_000.0, "USDC",
    risk_tolerance="medium",  # "low" | "medium" | "high"
)
# Returns: recommended_protocol, expected_apy_pct, estimated_annual_yield_usd
```

#### Portfolio Rebalance

```python
plan = await svc.defi_portfolio_rebalance(
    "cust-001",
    target_allocation={"uniswap_v3": 40.0, "aave_v3": 35.0, "yearn_v3": 25.0},
)
# Returns: trades_required[], total_value_usd
```

#### Real-Yield Dashboard

Decompose APY into fee-revenue yield vs. token-emission subsidy:

```python
dashboard = await svc.real_yield_dashboard()
# Per protocol: real_yield_pct, emission_yield_pct, sustainability ("sustainable" | "subsidy_dependent")
```

#### Cross-Chain Position Sync

```python
sync = await svc.cross_chain_position_sync(
    "cust-001",
    chains=["ethereum", "arbitrum", "base"],
)
# Returns: chain_positions{chain: estimated_position_usd}, total_cross_chain_usd
# Publishes to NATS: apg.fintech.defi.crosschain
```

#### Protocol Health Oracle

```python
health = await svc.protocol_health_oracle("aave_v3")
# Returns: health_score (0-100), status ("healthy"/"degraded"/"critical"),
#          factors{tvl_score, apy_plausibility_score, activity_score, liquidation_penalty}
```

#### Strategy Backtesting

```python
backtest = await svc.backtest_yield_strategy(
    strategy_config={"initial_usd": 100_000.0, "protocol": "yearn_v3"},
    weeks=52,
)
# Returns: final_usd, annualised_return_pct, sharpe_ratio, max_drawdown_pct, alpha_usd
```

---

### Risk and Compliance

#### DeFi Risk Dashboard (System-Wide)

```python
risk = await svc.defi_risk_dashboard(tenant_id="acme")
# Returns: critical_loan_count, liquidatable_loan_count, protocol_risk_breakdown{low/medium/high}
```

#### DeFi Risk Score (Per Customer)

```python
score = await svc.defi_risk_score("cust-001")
# Returns: defi_risk_score (0-100), risk_level, avg_health_factor, total_leverage_usd
```

#### Tax Event Ledger

```python
ledger = await svc.tax_event_ledger(
    "cust-001",
    method="fifo",       # "fifo" | "lifo" | "hifo"
    tax_year=2025,
)
# Returns: total_gain_usd, total_loss_usd, net_taxable_gain_usd, disposals[]
```

---

### Governance

#### Record Vote

```python
svc.record_governance_vote(
    "proposal-001", "acme", "aave_v3",
    proposal_reference="AAVE-IP-301",
    vote_choice="for",
    voter_id="dao-member-7",
    evidence_reference="ipfs://Qm...",
)
```

#### Simulate Proposal Outcome

```python
simulation = await svc.governance_outcome_simulation(
    "proposal-001",
    parameter_deltas={"ltv_ratio": 0.80, "supply_apy_pct": 1.5},
)
# Returns: impacts[] with old/new health_factor per loan and delta_apy_pct per deposit
```

---

### Utility Methods

```python
# Gas fee estimation
gas = await svc.gas_fee_estimator(chain="ethereum", operation="swap")
# Returns: gas_limit, gas_price_gwei, fee_eth, fee_usd

# Impermanent loss calculation
il = await svc.impermanent_loss_calculator(
    "USDC", "ETH",
    initial_price_ratio=3480.0,
    current_price_ratio=4200.0,
    initial_amount_usd=10_000.0,
)
# Returns: impermanent_loss_pct, impermanent_loss_usd, hold_vs_provide_diff_usd

# Compound interest
ci = await svc.compound_interest_calculator(
    principal=10_000.0, apy_pct=9.7, years=3.0
)
# Returns: final_value, interest_earned

# Flash loan simulation
sim = await svc.flash_loan_simulation(100_000.0, "USDC", "arb_usdc_usdt")
# Returns: simulated_profit, profitable

# TVL dashboard
tvl = await svc.tvl_dashboard()
# Returns: total_tvl_usd, by_protocol{}

# Lending rate feed
rates = await svc.lending_rate_feed()
# Returns: rates{aave_v3, compound_v3} with supply/borrow APYs per token

# Export
export = await svc.export_defi_data("cust-001", fmt="csv")
```

---

## Protocol Registry

Pre-registered protocols with indicative APY and TVL data:

| Protocol | Type | Chain | TVL | Base APY |
|----------|------|-------|-----|----------|
| uniswap_v3 | AMM | Ethereum | $4.2B | 8.5% |
| aave_v3 | Lending | Ethereum | $11B | 4.2% |
| compound_v3 | Lending | Ethereum | $2.8B | 3.8% |
| curve_3pool | AMM | Ethereum | $3.5B | 6.1% |
| pancakeswap_v3 | AMM | BNB Chain | $1.2B | 12.4% |
| yearn_v3 | Yield Vault | Ethereum | $800M | 9.7% |
| lido | Liquid Staking | Ethereum | $22B | 4.1% |
| makerdao | CDP | Ethereum | $6.5B | 0.0% |

---

## Health Factor Reference

| Range | Risk Level | Recommended Action |
|-------|-----------|-------------------|
| ≥ 150% | safe | No action required |
| 120–150% | warning | Consider adding collateral |
| 110–120% | critical | Add collateral immediately |
| < 110% | liquidatable | Repay debt or face liquidation |

---

## Streaming Architecture

All lifecycle events stream via Bytewax+NATS:

| NATS Subject | Events |
|-------------|--------|
| `apg.fintech.defi.lifecycle` | All write operations |
| `apg.fintech.defi.prices` | Price oracle updates (consumed by service) |
| `apg.fintech.defi.crosschain` | Cross-chain position sync results |
| `apg.fintech.defi.approvals` | Multi-sig approval workflow events |
| `apg.fintech.defi.analytics` | Real-yield and backtest results |

---

## Composability

Reference this capability in `.apg` source files:

```apg
use fintech_defi;
```

Composition triggers:

- `"liquidity pool"` — routes to `liquidity_pool_deposit` / `liquidity_pool_withdraw`
- `"yield farm"` / `"stake"` — routes to `yield_farming_enrol`
- `"borrow"` / `"collateral"` — routes to `borrow_against_collateral`
- `"swap"` / `"exchange"` — routes to `smart_route_swap`
- `"governance vote"` — routes to `record_governance_vote`
- `"health factor"` / `"liquidation risk"` — routes to `collateral_health_factor`
- `"real yield"` / `"APY breakdown"` — routes to `real_yield_dashboard`
- `"tax"` / `"cost basis"` — routes to `tax_event_ledger`
- `"cross-chain"` — routes to `cross_chain_position_sync`
- `"backtest"` — routes to `backtest_yield_strategy`

---

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `FINTECH_DEFI_`.

```bash
FINTECH_DEFI_DB_URL=postgresql://...
FINTECH_DEFI_NATS_URL=nats://localhost:4222
FINTECH_DEFI_LIQUIDATION_HF_BPS=11000
FINTECH_DEFI_SAFE_HF_BPS=15000
```

---

## Further Reading

- `service.py` — Business logic implementation (1300+ lines)
- `models.py` — Dataclass models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `capability_contract.py` — Policy rules and contract
- `README.md` — Quick reference
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 prioritised improvement proposals
- `SPECIFICATION.md` — Original capability specification
