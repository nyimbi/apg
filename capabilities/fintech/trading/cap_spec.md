# APG Capability Spec: Algorithmic Trading

- Capability id: `fintech_trading`
- Display name: `Algorithmic Trading`
- Version: `1.1.0`
- Target: `python`
- Runtime profile: package-backed capability
- Stream processor: `bytewax`
- Stream: `apg.fintech.trading.lifecycle`

## Provides

`trading_strategy_workflow`, `trading_signal_workflow`,
`trading_backtest_workflow`, `trading_risk_limit_workflow`,
`trading_order_intent_workflow`, `trading_execution_workflow`,
`trading_position_workflow`, `trading_surveillance_workflow`,
`trading_review_workflow`, and `trading_agent_workflow`.

## Composition

The package composes with APG identity, audit, notification, language,
key-management, Portfolio Management, Wealth Management, Robo Advisory,
payments, wallets, KYC, AML, fraud, analytics, reporting, market-data, and
Bytewax capabilities.
