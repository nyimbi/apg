# Algorithmic Trading

## Overview

Governed strategy-driven trading operations for APG fintech applications. Covers the full lifecycle: strategy registration with policy controls, multi-factor signal attachment with freshness SLAs, backtesting with evidence validation, risk limit activation, order intent staging with pre-trade risk checks, execution recording, position snapshots, TWAP/VWAP execution scheduling, settlement reporting, surveillance alerts, and compliance reviews.

Every order intent requires both a risk limit reference and an explicit approval — two independent checks. Backtests with zero trades are rejected. All lifecycle events stream to `apg.fintech.trading.lifecycle` via Bytewax.

## Capability ID

`fintech_trading`  Version: 2.0.0

## Provides

| Service | Description |
|---------|-------------|
| trading_strategy_workflow | Register strategies with type, asset class, owner, and policy reference |
| trading_signal_workflow | Attach signal sources with freshness SLA, lineage, and source reference |
| trading_backtest_workflow | Record backtests with period, positive trade count, data source, and metrics |
| trading_risk_limit_workflow | Set risk limits with metric, positive limit, and approval evidence |
| trading_order_intent_workflow | Stage order intents with pre-trade risk check, instrument, type, quantity, and approval |
| trading_execution_workflow | Record executions with venue, filled quantity, and source evidence |
| trading_position_workflow | Record position snapshots with strategy, as-of date, and source |
| trading_surveillance_workflow | Record surveillance alerts with severity and evidence |
| trading_review_workflow | Governance reviews for strategies, risk limits, and surveillance |
| trading_agent_workflow | Register AI agents for strategy review, signal quality, backtest, and risk limit review |

## Requires

| Capability | Purpose |
|------------|---------|
| auth | Authentication |
| audl | Audit trail |
| ntfy | Operations notifications |
| nlpc | NLP processing |
| keym | Key management |
| fintech_portfolio | Portfolio position context |
| fintech_wealth | Client mandate context |
| fintech_robo | Robo model portfolio signals |
| fintech_payments | Settlement execution |
| fintech_wallets | Wallet-based settlement |
| fintech_kyc | Trader identity |
| fintech_aml | AML compliance |
| fintech_fraud | Fraud risk context |
| bia | Trading analytics |
| fin_rpt | Trading reports |

## Quick Start

```python
from capabilities.fintech.trading.service import AlgorithmicTradingService

svc = AlgorithmicTradingService(tenant_id="t1", actor_id="trader_01")

# Register a strategy
await svc.register_strategy(
    "strat_01", "Momentum KES Equities", "momentum", "equity", "policy/algo-eq-001"
)

# Set a risk limit (requires approval reference)
await svc.set_risk_limit("rl_01", "strat_01", "max_notional", 5_000_000, "approval/rm-2026-01")

# Place an order (runs pre-trade risk check automatically)
result = await svc.place_order(
    account_id="acc_01", symbol="SCOM", order_type="limit",
    side="buy", quantity=1000, price=15.50, time_in_force="day",
    strategy_id="strat_01", risk_limit_id="rl_01", approval_reference="approval/rm-2026-01"
)
# result["risk_check"]["passed"] → True/False

# Run a backtest
await svc.backtest_strategy("strat_01", "2025-Q1", initial_capital=1_000_000)

# Dashboard
await svc.dashboard_summary()
```

## New Methods

### `place_order` — pre-trade risk check baked in

```python
result = await svc.place_order(
    account_id="acc_01", symbol="EQTY", order_type="limit",
    side="buy", quantity=500, price=42.00, time_in_force="gtc",
    strategy_id="strat_01", risk_limit_id="rl_01",
    approval_reference="approval/rm-2026-03"
)
# {"order_id": "...", "status": "pending", "risk_check": {"passed": True, "messages": []}}
```

### `twap_execution` — TWAP schedule planning

```python
schedule = await svc.twap_execution(
    strategy_id="strat_01", symbol="SCOM",
    total_quantity=10_000, duration_minutes=60, slices=6
)
# {"slice_count": 6, "slice_quantity": 1666.6667, "schedule": [{"slice": 1, "execute_at_minute": 10.0, ...}, ...]}
```

### `vwap_calculation` — VWAP from live execution history

```python
vwap = await svc.vwap_calculation("strat_01", "2026-05")
# {"vwap": 14.3250, "total_volume": 8500.0, "computed_at": "..."}
```

### `mark_to_market` — real-time P&L estimate

```python
mtm = await svc.mark_to_market("acc_01")
# {"total_unrealised_pnl_minor": 12400.0, "positions": [...]}
```

### `margin_call_check` — margin deficiency detection

```python
margin = await svc.margin_call_check("acc_01")
# {"margin_status": "ok" | "deficient", "recommendations": [...]}
```

### `post_trade_analytics` — slippage and market impact

```python
analytics = await svc.post_trade_analytics("strat_01", "2026-05")
# {"slippage_bps": 5.0, "market_impact_bps": 2.5, "implementation_shortfall_bps": 7.5, ...}
```

## World-Class Enhancements (v2.0)

| # | Enhancement | Summary |
|---|-------------|---------|
| 1 | Smart Order Routing (SOR) | Multi-venue routing across NSE, ATS, dark pools, OTC with best-execution scoring and per-order audit trail |
| 2 | Intraday P&L Attribution | Streaming ledger: every fill updates a live position register with unrealised P&L and delta-adjusted exposure |
| 3 | Microstructure-Aware Scheduling | TWAP/VWAP slicers consume live order-book depth and intraday volume curves; cuts market impact 30–60% vs. naïve slicing |
| 4 | Multi-Factor Signal Aggregation | Weighted momentum/mean-reversion/sentiment/ML factor pipeline with explained-variance breakdown and lineage per factor |
| 5 | Regime Detection & Adaptive Strategy | HMM/change-point classifier auto-adjusts strategy weights and risk limits for trending, mean-reverting, and vol regimes |
| 6 | LOB Simulation for Backtests | Event-driven limit-order-book simulator replays tick data with fill probabilities and adverse-selection cost models |
| 7 | Portfolio-Level Risk Aggregation | Cross-strategy covariance matrix, portfolio VaR/CVaR, auto-rebalancing on limit breach |
| 8 | Pre-Trade TCA | Explicit (commission, fees, stamp duty) + implicit (spread, square-root market impact) + timing cost in every order response |
| 9 | Circuit Breakers & Kill Switch | Tiered: strategy drawdown → account daily loss → market-wide vol spike; single idempotent call cancels all orders and pages risk desk |
| 10 | Explainable AI (XAI) for Signals | SHAP-value breakdown per ML trade signal satisfies model governance (EU AI Act, CMA guidelines) |
| 11 | Settlement & Fail Management | T+2/T+3 deadline tracking, buy-in/sell-out instruction generation, fail penalty calculation, DVP integration with `fintech_payments` |
| 12 | Latency & Execution Quality Monitoring | Nanosecond timestamps at order creation/submission/ack/fill; p50/p95/p99 per venue; Prometheus-compatible metrics endpoint |
| 13 | Crypto & Cross-Border FX | WebSocket connectors to CEX (Binance, Kraken), DEX (Uniswap/Infura), FX ECN (EBS, Currenex); cross-venue arb detection; USD/KES hedging |
| 14 | Regulatory Reporting Automation | CMA algo notifications, NSE weekly transaction reports, FRC large-holding disclosures, FATCA/CRS filings via template engine with digital signature |
| 15 | Event-Sourced Audit Log | SHA-256 content-addressed, chained, tenant-key-signed immutable event store; full state reconstruction at any point in time; MAR/EMIR compliant |

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| strategies.supported_types | list | mean_reversion, momentum, market_making, pairs, arbitrage, hedging, execution_algo | Strategy categories |
| strategies.supported_asset_classes | list | equity, fixed_income, fx, fund, commodity, crypto | Asset classes |
| orders.supported_order_types | list | market, limit, stop, twap, vwap, iceberg | Order types |
| executions.supported_venues | list | exchange, ats, otc, dark_pool, internal_cross | Execution venues |
| surveillance.supported_severities | list | low, medium, high, critical | Alert severity levels |

## API Routes

| Name | Path | Method | Permission | Group |
|------|------|--------|------------|-------|
| dashboard | /fintech-trading/dashboard | GET | fintech_trading:view | Overview |
| strategies | /fintech-trading/strategies | GET/POST | fintech_trading:strategies | Strategies |
| signals | /fintech-trading/signals | GET/POST | fintech_trading:signals | Strategies |
| backtests | /fintech-trading/backtests | GET/POST | fintech_trading:backtests | Validation |
| risk | /fintech-trading/risk | GET/POST | fintech_trading:risk | Risk |
| orders | /fintech-trading/orders | GET/POST | fintech_trading:orders | Trading |
| executions | /fintech-trading/executions | GET/POST | fintech_trading:executions | Trading |
| positions | /fintech-trading/positions | GET/POST | fintech_trading:positions | Risk |
| surveillance | /fintech-trading/surveillance | GET/POST | fintech_trading:surveillance | Governance |
| reviews | /fintech-trading/reviews | GET/POST | fintech_trading:reviews | Governance |
| agents | /fintech-trading/agents | GET/POST | fintech_trading:admin | Automation |
| settings | /fintech-trading/settings | GET/POST | fintech_trading:admin | Administration |

## Business Rules

| Rule | Condition | Effect |
|------|-----------|--------|
| strategy_type_supported | Unsupported strategy type | deny |
| strategy_asset_class_supported | Unsupported asset class | deny |
| strategy_policy_reference_required | Strategy without policy reference | deny |
| signal_freshness_required | Signal without freshness SLA | deny |
| backtest_positive_trade_count | Backtest with zero or negative trade count | deny |
| backtest_data_source_required | Backtest without data source reference | deny |
| risk_positive_limit | Risk limit with zero or negative value | deny |
| risk_approval_required | Risk limit without approval | deny |
| order_risk_limit_required | Order without risk limit reference | deny |
| order_approval_required | Order intent without approval | deny |
| order_positive_quantity | Order with zero or negative quantity | deny |
| execution_venue_supported | Unsupported execution venue | deny |
| execution_positive_filled_quantity | Execution with zero or negative filled quantity | deny |
| surveillance_evidence_required | Surveillance alert without evidence | deny |
| trading_batch_requires_bytewax | Batch without Bytewax | deny |
| privileged_trading_agent_action_requires_human_approval | AI agent privileged scope without approval | deny |

## Data Models

| Model | Key Fields |
|-------|-----------|
| TradingStrategy | id, name, owner_id, strategy_type, asset_class, policy_reference, status |
| SignalSource | id, strategy_id, source_reference, freshness_sla, lineage_reference |
| BacktestRun | id, strategy_id, period, trade_count, data_source_reference, metrics |
| RiskLimit | id, strategy_id, metric, limit_value, approval_reference, status |
| OrderIntent | id, strategy_id, risk_limit_id, instrument_reference, order_type, quantity, approval_reference, status |
| ExecutionRecord | id, order_id, venue, filled_quantity, execution_price, source_reference, status |
| PositionSnapshot | id, strategy_id, as_of_date, gross_exposure_minor, net_exposure_minor, source_reference |
| SurveillanceAlert | id, strategy_id, severity, evidence_reference, status |

## Streaming Events

Events emitted to `apg.fintech.trading.lifecycle` via Bytewax.

| Event | Trigger |
|-------|---------|
| trading_strategy_registered | Strategy registered |
| signal_source_attached | Signal source attached |
| backtest_recorded | Backtest recorded |
| risk_limit_set | Risk limit activated |
| order_placed | Order staged with pre-trade risk check result |
| order_cancelled | Order cancelled |
| algo_strategy_executed | Algo strategy triggered |
| execution_recorded | Execution recorded |
| position_snapshot_recorded | Position snapshot recorded |
| mark_to_market | Positions marked to market |
| vwap_calculated | VWAP computed for strategy |
| twap_planned | TWAP execution schedule created |
| surveillance_alert_recorded | Surveillance alert raised |
| trading_review_recorded | Review completed |
| trading_agent_registered | AI agent registered |
| settlement_report_generated | Settlement report produced |
| algo_performance_report_generated | Performance report produced |

## Edge Cases

- Both risk limit reference AND explicit approval are required for every order intent — a risk limit without approval (or vice versa) is insufficient
- Backtests with zero trades are rejected — minimum trade count of 1 required
- Signal freshness SLA required at registration — stale signals without a documented SLA are rejected
- Pre-trade risk check runs synchronously in `place_order`; orders exceeding hard notional limits or naked short-sells are rejected before staging
- Position snapshots are point-in-time; `as_of_date` marks the timestamp but the engine does not validate against current live positions
- `internal_cross` venue requires the same evidence as external venue executions

## Composability

- **Upstream**: `fintech_portfolio` provides the investment book; `fintech_robo` provides model-driven signals; market data adapters provide price feeds
- **Downstream**: `fintech_payments` and `fintech_wallets` execute settlement; `fintech_portfolio` receives execution records as holding updates
- **Peer**: Deployed alongside `fintech_portfolio` and `fintech_robo` in a full investment management stack

---

*Datacraft © 2025 | www.datacraft.co.ke*
