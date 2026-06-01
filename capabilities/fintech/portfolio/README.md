# Portfolio Management

Portfolio Management is an executable APG capability for regulated investment
books, holdings, allocation policy, valuations, benchmarks, risk exposure,
performance attribution, cash movements, corporate actions, compliance
breaches, reviews, and AI-assisted portfolio operations.

The package is dependency-light and can run inside generated Python
applications. Production deployments bind the adapter keys in the capability
contract to APG identity, audit, notifications, language/NLP, key management,
wealth, robo advisory, payments, wallets, KYC, AML, fraud, analytics,
reporting, market data, and Bytewax services.

## Use

```python
from capabilities.fintech.portfolio import PortfolioManagementService

service = PortfolioManagementService()
portfolio = service.create_portfolio_book(
    "portfolio-1", "tenant-1", "owner-1", "Core Portfolio",
    "discretionary", "USD", "ips-1"
)
service.record_holding(
    "holding-1", "tenant-1", portfolio["id"], "ETF-1", 12.5, 1000000, "USD"
)
service.activate_allocation_policy(
    "allocation-1", "tenant-1", portfolio["id"],
    {"equity": 60, "fixed_income": 35, "cash": 5}, "policy-1"
)
service.record_valuation(
    "valuation-1", "tenant-1", portfolio["id"], 1500000,
    "USD", "2026-06-01", "pricing-source-1"
)
```

## Capability Surfaces

- Portfolio book creation with owner, policy, type, and base currency controls.
- Holding ledger records with instrument, quantity, cost, and currency evidence.
- Allocation policy activation with target allocations that must total 100%.
- Portfolio valuation capture with source and valuation-date evidence.
- Benchmark assignment for performance and policy comparison.
- Risk exposure recording with source, as-of date, metric, and limit evidence.
- Performance attribution recording by period, benchmark, and contribution set.
- Cash movement recording with amount, currency, and source reference.
- Corporate-action recording with supported action types and evidence.
- Compliance breach and review workflows for governance escalation.
- Provider-neutral AI agent registration across Codex, Claude Code, OpenCode,
  and Pi runtimes.
- Dashboard, book, holding, allocation, valuation, benchmark, risk,
  attribution, cash, corporate-action, compliance, review, settings, and agent
  view models.
- Deterministic rule engine and Bytewax lifecycle stream metadata.

## Integration Boundaries

Live custody, broker routing, market data feeds, tax-lot accounting, statement
rendering, billing collection, regulator filing, and durable Bytewax workers
stay behind adapter boundaries.
