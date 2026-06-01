# Algorithmic Trading

Algorithmic Trading is an executable APG capability for strategy registration,
signal lineage, backtesting, risk limits, order-intent staging, execution
evidence, position snapshots, surveillance alerts, review workflows, and
AI-assisted trading governance.

The package is dependency-light and can run inside generated Python
applications. Production deployments bind the adapter keys in the capability
contract to APG identity, audit, notifications, language/NLP, key management,
Portfolio Management, Wealth Management, Robo Advisory, payments, wallets, KYC,
AML, fraud, analytics, reporting, market data, and Bytewax services.

## Use

```python
from capabilities.fintech.trading import AlgorithmicTradingService

service = AlgorithmicTradingService()
strategy = service.register_strategy(
    "strategy-1", "tenant-1", "owner-1", "Momentum Core",
    "momentum", "equity", "policy-1"
)
service.attach_signal_source(
    "signal-1", "tenant-1", strategy["id"], "market-feed-1",
    "PT5S", "lineage-1"
)
limit = service.set_risk_limit(
    "limit-1", "tenant-1", strategy["id"], "gross_exposure",
    2500000, "risk-approval-1"
)
service.stage_order_intent(
    "order-1", "tenant-1", strategy["id"], limit["id"],
    "ETF-1", "limit", 100.0, "order-approval-1"
)
```

## Capability Surfaces

- Strategy registration with owner, type, asset class, and policy controls.
- Signal-source attachment with source, freshness SLA, and lineage evidence.
- Backtest recording with period, trade count, source data, and metrics.
- Risk-limit activation with metric, positive limit, and approval evidence.
- Order-intent staging with instrument, order type, risk limit, quantity, and
  approval controls.
- Execution recording with supported venue, filled quantity, and source
  evidence.
- Position snapshots for gross and net exposure views.
- Surveillance alerts and review workflows for governance escalation.
- Provider-neutral AI agent registration across Codex, Claude Code, OpenCode,
  and Pi runtimes.
- Dashboard, strategy, signal, backtest, risk, order, execution, position,
  surveillance, review, settings, and agent view models.
- Deterministic rule engine and Bytewax lifecycle stream metadata.

## Integration Boundaries

Live market-data ingestion, venue connectivity, smart order routing, official
performance attribution, custody settlement, transaction-cost analysis,
regulator filing, and durable Bytewax workers stay behind adapter boundaries.
