# Decentralized Finance

## Overview

Decentralized Finance provides governed operations over DeFi protocols: protocol registry, position management (supply, borrow, liquidity, stake, vault share), action execution workflow (deposit, withdraw, borrow, repay, swap, stake, unstake, claim, rebalance), yield strategy management, reward accruals, governance voting, risk tier assessments, and reviews. Every action against a DeFi protocol requires an approval reference before it is recorded, enforcing human oversight over autonomous on-chain interactions.

Position-protocol consistency is enforced — actions on a position must reference the same protocol the position belongs to. Health factors must be positive on position opening. All DeFi lifecycle events stream to `apg.fintech.defi.lifecycle` via Bytewax+NATS.

## Capability ID

`fintech_defi` | Version: 1.2.0

## Provides

| Service | Description |
|---------|-------------|
| defi_protocol_workflow | Register DeFi protocols with type, network reference, risk tier, owner, and evidence |
| defi_position_workflow | Open positions with protocol, account, asset pair, type, amount, collateral, and health factor |
| defi_action_workflow | Record protocol actions with type, amount, requester, approval, and status |
| defi_yield_strategy_workflow | Register yield strategies with target APY, max risk tier, owner, and evidence |
| defi_reward_workflow | Record protocol rewards with type, asset, amount, and evidence |
| defi_governance_workflow | Record and simulate governance votes and their impact on active positions |
| defi_risk_workflow | Record risk assessments with tier, reviewer, and evidence |
| defi_review_workflow | Governance reviews for protocols, positions, and strategy decisions |
| defi_agent_workflow | Register AI agents for protocol monitoring, position reconciliation, and liquidity risk |
| defi_analytics_workflow | Real-yield decomposition, backtesting, and cross-chain position aggregation |
| defi_trading_workflow | MEV-resistant multi-route swaps, atomic collateral substitution, liquid staking optimisation |
| defi_tax_workflow | Cost-basis tax ledger (FIFO/LIFO/HIFO) for DeFi disposals |

## Requires

| Capability | Purpose |
|------------|---------|
| auth | Authentication |
| audl | Audit trail |
| ntfy | Operations notifications |
| nlpc | NLP processing |
| keym | Key management |
| fintech_blockchain | Network and protocol infrastructure |
| fintech_crypto | Asset and custody backing |
| fintech_wallets | Wallet references for positions |
| fintech_risk | Risk tier and appetite context |
| fintech_compliance | Compliance obligation evidence |
| fintech_regtech | Regulatory framework context |
| fintech_aml | AML screening for DeFi activity |
| fintech_kyc | Identity verification for account holders |

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| protocols.supported_protocol_types | list | lending_pool, liquidity_pool, staking, yield_vault, dex, bridge, derivatives, insurance_pool | Protocol categories |
| protocols.supported_risk_tiers | list | low, medium, high, critical | Risk tier classification |
| positions.supported_position_types | list | supply, borrow, liquidity, stake, vault_share, short, long, cover | Position types |
| actions.supported_action_types | list | deposit, withdraw, borrow, repay, swap, stake, unstake, claim, rebalance | Action types |
| governance.supported_vote_choices | list | for, against, abstain | Valid vote choices |
| streaming.nats_subjects | list | apg.fintech.defi.lifecycle, apg.fintech.defi.prices, apg.fintech.defi.crosschain, apg.fintech.defi.approvals, apg.fintech.defi.analytics | NATS event subjects |

## API Routes

| Name | Path | Method | Permission | Group |
|------|------|--------|------------|-------|
| dashboard | /fintech-defi/dashboard | GET | fintech_defi:view | Overview |
| protocols | /fintech-defi/protocols | GET/POST | fintech_defi:protocols | Protocols |
| positions | /fintech-defi/positions | GET/POST | fintech_defi:positions | Portfolio |
| actions | /fintech-defi/actions | GET/POST | fintech_defi:actions | Operations |
| yield_strategies | /fintech-defi/yield-strategies | GET/POST | fintech_defi:yield | Strategies |
| rewards | /fintech-defi/rewards | GET/POST | fintech_defi:rewards | Portfolio |
| governance | /fintech-defi/governance | GET/POST | fintech_defi:governance | Governance |
| risk | /fintech-defi/risk | GET/POST | fintech_defi:risk | Risk |
| reviews | /fintech-defi/reviews | GET/POST | fintech_defi:reviews | Governance |
| agents | /fintech-defi/agents | GET/POST | fintech_defi:admin | Automation |
| analytics | /fintech-defi/analytics | GET | fintech_defi:view | Analytics |
| real_yield | /fintech-defi/analytics/real-yield | GET | fintech_defi:view | Analytics |
| backtest | /fintech-defi/analytics/backtest | POST | fintech_defi:yield | Analytics |
| cross_chain | /fintech-defi/positions/cross-chain | GET | fintech_defi:positions | Portfolio |
| tax_ledger | /fintech-defi/tax/ledger | GET | fintech_defi:view | Compliance |
| settings | /fintech-defi/settings | GET/POST | fintech_defi:admin | Administration |

## New Service Methods (v1.2.0)

| Method | Description |
|--------|-------------|
| `real_yield_dashboard()` | Decompose protocol APY into fee-revenue yield vs. token-emission yield |
| `smart_route_swap()` | MEV-resistant multi-route swap splitting order across AMM protocols by TVL weight |
| `protocol_health_oracle()` | Score protocol health 0-100 using TVL, activity, APY plausibility, and liquidation exposure |
| `atomic_collateral_swap()` | Flash-loan-funded atomic collateral substitution with health-factor validation |
| `tax_event_ledger()` | FIFO/LIFO/HIFO cost-basis gain/loss ledger for all DeFi disposals |
| `liquid_staking_optimiser()` | Recommend optimal LST and downstream DeFi compounding path |
| `backtest_yield_strategy()` | Synthetic historical backtest with Sharpe ratio and max drawdown |
| `governance_outcome_simulation()` | Model impact of a governance proposal on active positions and APYs |
| `cross_chain_position_sync()` | Aggregate positions across Ethereum, Arbitrum, Base, Polygon, BNB Chain |

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| protocol_risk_supported | Unsupported risk tier on protocol | deny |
| position_health_factor_valid | Negative or zero health factor | deny |
| position_collateral_valid | Negative collateral amount | deny |
| action_position_protocol_match | Action position belongs to different protocol | deny |
| action_approval_required | Action without approval reference | deny |
| strategy_target_apy_valid | Negative target APY | deny |
| governance_vote_supported | Unsupported vote choice | deny |
| defi_batch_requires_bytewax | Batch without Bytewax | deny |
| privileged_defi_agent_action_requires_human_approval | AI agent privileged scope without approval | deny |
| atomic_collateral_swap_health_factor | Post-swap health factor below liquidation threshold | assert |
| smart_route_swap_min_splits | max_splits must be ≥ 1 | assert |

## Data Models

| Model | Key Fields |
|-------|-----------|
| DeFiProtocol | id, protocol_type, network_reference, protocol_reference, risk_tier, owner_id, evidence_reference |
| LiquidityPosition | id, protocol_id, account_reference, asset_pair_reference, position_type, amount_minor, collateral_minor, health_factor_bps, evidence_reference |
| DeFiAction | id, protocol_id, position_id, action_type, amount_minor, requester_id, approval_reference, status, evidence_reference |
| YieldStrategy | id, protocol_id, strategy_reference, target_apy_bps, max_risk_tier, owner_id, evidence_reference |
| RewardAccrual | id, position_id, reward_type, asset_reference, amount_minor, evidence_reference |
| GovernanceProposal | id, protocol_id, proposal_reference, vote_choice, voter_id, evidence_reference |
| RiskAssessment | id, reference_id, risk_tier, reviewer_id, evidence_reference |
| DeFiReview | id, reference_id, reviewer_id, status, evidence_reference |
| DeFiAgent | id, name, runtime, role, scope |

## Streaming Events

Events emitted via Bytewax+NATS to the fintech event stream.

| Event | NATS Subject | Trigger |
|-------|-------------|---------|
| defi_protocol_registered | apg.fintech.defi.lifecycle | Protocol registered |
| defi_position_opened | apg.fintech.defi.lifecycle | Position opened |
| defi_action_recorded | apg.fintech.defi.lifecycle | Protocol action recorded |
| defi_yield_strategy_registered | apg.fintech.defi.lifecycle | Yield strategy registered |
| defi_reward_recorded | apg.fintech.defi.lifecycle | Reward accrual recorded |
| defi_governance_vote_recorded | apg.fintech.defi.lifecycle | Governance vote cast |
| defi_risk_assessment_recorded | apg.fintech.defi.lifecycle | Risk assessment recorded |
| defi_review_recorded | apg.fintech.defi.lifecycle | Governance review completed |
| defi_agent_registered | apg.fintech.defi.lifecycle | AI agent registered |
| real_yield_dashboard_generated | apg.fintech.defi.analytics | Real-yield report computed |
| smart_route_swap_executed | apg.fintech.defi.lifecycle | Multi-route swap settled |
| protocol_health_oracle_checked | apg.fintech.defi.lifecycle | Protocol health scored |
| atomic_collateral_swap_executed | apg.fintech.defi.lifecycle | Collateral substituted |
| tax_event_ledger_generated | apg.fintech.defi.lifecycle | Tax ledger produced |
| liquid_staking_optimised | apg.fintech.defi.lifecycle | LST recommendation generated |
| strategy_backtest_run | apg.fintech.defi.analytics | Backtest completed |
| governance_outcome_simulated | apg.fintech.defi.lifecycle | Proposal impact modelled |
| cross_chain_positions_synced | apg.fintech.defi.crosschain | Cross-chain sync completed |

## Edge Cases Handled

- Protocol-position consistency: an action on a position is rejected if `position_protocol_match` is false
- Health factor must be strictly positive at position opening
- Collateral is non-negative (zero valid for unsecured positions)
- Target APY on yield strategies must be non-negative
- Every DeFi action requires an approval reference — autonomous execution without oversight is blocked
- `atomic_collateral_swap` asserts post-swap health factor exceeds liquidation threshold before committing
- `smart_route_swap` falls back to single-hop if no AMM protocols are registered
- `backtest_yield_strategy` requires `initial_usd > 0`
- `governance_outcome_simulation` requires the referenced proposal to exist in the governance registry

## Composability

- **Upstream**: `fintech_blockchain` provides network infrastructure; `fintech_crypto` provides custody accounts and asset references backing DeFi positions
- **Downstream**: `fintech_portfolio` consumes position and balance data for portfolio-level tracking; `fintech_risk` receives DeFi risk assessments as exposure records; `fintech_tax` consumes the tax event ledger
- **Peer**: Deployed alongside `fintech_crypto` (asset operations) and `fintech_blockchain` (network layer) in a full digital asset stack
- **Streaming**: Price updates consumed from `apg.fintech.defi.prices` (Bytewax pipeline); governance approval events published to `apg.fintech.defi.approvals`; cross-chain sync results on `apg.fintech.defi.crosschain`

## Development Notes

- `SUPPORTED_PROTOCOL_TYPES` maps to DeFi primitive categories; `insurance_pool` covers on-chain coverage products
- The `action_position_protocol_match` rule fires when `position_protocol_match: False` — callers must compute this before invoking the rule engine
- Governance votes use `for/against/abstain` mapping to standard on-chain governance; `abstain` counts toward quorum
- Health factor convention follows Aave/Compound: values > 1.0 are solvent, < 1.0 subject to liquidation
- `real_yield_dashboard` uses observed swap volumes to estimate fee revenue; inject real on-chain data for production accuracy
- `backtest_yield_strategy` uses `random.gauss` for APY noise — seed with a fixed value in tests for determinism
- Streaming platform: Bytewax+NATS. Do not substitute Kafka or Spark Streaming.
