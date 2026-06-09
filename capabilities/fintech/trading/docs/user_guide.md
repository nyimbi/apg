# Algorithmic Trading

**Capability ID**: `fintech_trading` | **Domain**: `fintech` | **Version**: `1.1.0`

## Description

Algorithmic Trading provides governed strategy-driven trading operations: strategy registration with asset class and policy controls, signal source attachment with freshness SLAs and lineage, backtesting with trade count and data source evidence, risk limit activation with approval, order intent staging with instrument and approval gates, execution recording, position snapshots, trading surveillance, and governance reviews. Every order intent requires both a risk limit reference and an explicit approval before it can be staged — preventing unsanctioned automated order flow.

## Installation

```bash
pip install apg-fintech-trading
```

## Provides

- `trading_strategy_workflow`
- `trading_signal_workflow`
- `trading_backtest_workflow`
- `trading_risk_limit_workflow`
- `trading_order_intent_workflow`

## Requires

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/fintech-trading/dashboard` | `fintech_trading:view` | Overview |
| `/fintech-trading/strategies` | `fintech_trading:strategies` | Strategies |
| `/fintech-trading/signals` | `fintech_trading:signals` | Strategies |
| `/fintech-trading/backtests` | `fintech_trading:backtests` | Validation |
| `/fintech-trading/risk` | `fintech_trading:risk` | Risk |
| `/fintech-trading/orders` | `fintech_trading:orders` | Trading |
| `/fintech-trading/executions` | `fintech_trading:executions` | Trading |
| `/fintech-trading/positions` | `fintech_trading:positions` | Risk |

## Key Service Methods

- `describe()`
- `evaluate()`
- `register_strategy()`
- `get_strategy()`
- `list_strategies()`
- `deactivate_strategy()`
- `attach_signal_source()`
- `place_order()`
- `cancel_order()`
- `order_status()`

_(See `service.py` for complete API.)_

## Interoperability

`fintech_trading` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use fintech_trading;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `FINTECH_TRADING_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
