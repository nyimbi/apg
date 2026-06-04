# Algorithmic Trading

## Overview
Algorithmic Trading provides governed strategy-driven trading operations: strategy registration with asset class and policy controls, signal source attachment with freshness SLAs and lineage, backtesting with trade count and data source evidence, risk limit activation with approval, order intent staging with instrument and approval gates, execution recording, position snapshots, trading surveillance, and governance reviews. Every order intent requires both a risk limit reference and an explicit approval before it can be staged — preventing unsanctioned automated order flow.

Backtests require a positive trade count; a backtest with zero trades is rejected. All trading lifecycle events stream to `apg.fintech.trading.lifecycle` via Bytewax.

## Capability ID
`fintech_trading`  Version: 1.1.0

## Provides
| Service | Description |
|---------|-------------|
| trading_strategy_workflow | Register strategies with type, asset class, owner, and policy reference |
| trading_signal_workflow | Attach signal sources with strategy, freshness SLA, lineage, and source reference |
| trading_backtest_workflow | Record backtests with period, positive trade count, data source, and metrics |
| trading_risk_limit_workflow | Set risk limits with metric, positive limit, and approval evidence |
| trading_order_intent_workflow | Stage order intents with instrument, type, quantity, risk limit, and approval |
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
| Backtest | id, strategy_id, period_start, period_end, trade_count, data_source_reference, metrics |
| RiskLimit | id, strategy_id, metric, limit_value, approval_reference, status |
| OrderIntent | id, strategy_id, risk_limit_id, instrument_reference, order_type, quantity, approval_reference, status |
| Execution | id, order_id, venue, filled_quantity, execution_price, source_reference, status |
| PositionSnapshot | id, strategy_id, as_of_date, positions, source_reference |
| SurveillanceAlert | id, strategy_id, severity, evidence_reference, status |

## Streaming Events
Events emitted to the fintech event stream via Bytewax.
| Event | Trigger |
|-------|---------|
| trading_strategy_registered | Strategy registered |
| signal_source_attached | Signal source attached |
| backtest_recorded | Backtest recorded |
| risk_limit_set | Risk limit activated |
| order_intent_staged | Order intent staged |
| execution_recorded | Execution recorded |
| position_snapshot_recorded | Position snapshot recorded |
| surveillance_alert_recorded | Surveillance alert raised |
| trading_review_recorded | Review completed |
| trading_agent_registered | AI agent registered |

## Edge Cases Handled
- Both risk limit reference AND explicit approval are required for every order intent — two independent checks; a risk limit without approval (or approval without a risk limit) is insufficient
- Backtests with zero trades are rejected — a strategy that has never traded cannot demonstrate live behavior; a minimum trade count of 1 is required
- Signal freshness SLA is required at signal source registration — this enforces data freshness contracts; stale signals without a documented SLA are rejected
- Position snapshots capture the state at a specific `as_of_date`; the rule engine only checks that the date is present — it does not validate the snapshot represents current positions
- `internal_cross` venue is supported for off-exchange crossing within a single institution; it requires the same evidence as external venue executions

## Composability
- **Upstream**: `fintech_portfolio` provides the investment book context that trading strategies operate against; `fintech_robo` provides model-driven signals for automated order generation; market data adapters provide price feeds
- **Downstream**: `fintech_payments` and `fintech_wallets` execute settlement for completed trades; `fintech_portfolio` receives execution records as holding updates
- **Peer**: Deployed alongside `fintech_portfolio` (investment books) and `fintech_robo` (model signals) in a full investment management stack

## Development Notes
- `execution_algo` strategy type (TWAP, VWAP, iceberg) maps to algorithmic execution strategies, not alpha-generating strategies; these are included alongside directional strategies in the same governance framework
- `dark_pool` and `ats` venue types support off-exchange and alternative trading system executions; these require the same evidence as exchange executions
- Position snapshots are point-in-time records; intraday position tracking is not supported by the rule engine; the `as_of_date` field marks the snapshot timestamp
- Risk limits are per-strategy, not per-instrument or per-order; a single strategy can have multiple limits covering different risk metrics (gross exposure, net exposure, VaR)
