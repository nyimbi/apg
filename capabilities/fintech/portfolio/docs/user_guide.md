# Portfolio Management

**Capability ID**: `fintech_portfolio` | **Domain**: `fintech` | **Version**: `1.1.0`

## Description

Portfolio Management provides regulated investment book operations: portfolio book creation, holding ledger recording, allocation policy activation (totals must equal exactly 100%), valuation capture, benchmark assignment, risk exposure tracking, performance attribution, cash movement recording, corporate action processing, compliance breach recording, and governance reviews. It is the investment operations layer for discretionary, advisory, model, and execution-only portfolios.

## Installation

```bash
pip install apg-fintech-portfolio
```

## Provides

- `portfolio_book_workflow`
- `portfolio_holding_workflow`
- `portfolio_allocation_policy_workflow`
- `portfolio_valuation_workflow`
- `portfolio_benchmark_workflow`

## Requires

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/fintech-portfolio/dashboard` | `fintech_portfolio:view` | Overview |
| `/fintech-portfolio/portfolios` | `fintech_portfolio:portfolios` | Books |
| `/fintech-portfolio/holdings` | `fintech_portfolio:holdings` | Books |
| `/fintech-portfolio/allocations` | `fintech_portfolio:allocations` | Policy |
| `/fintech-portfolio/valuations` | `fintech_portfolio:valuations` | Operations |
| `/fintech-portfolio/benchmarks` | `fintech_portfolio:benchmarks` | Policy |
| `/fintech-portfolio/risk` | `fintech_portfolio:risk` | Risk |
| `/fintech-portfolio/attribution` | `fintech_portfolio:attribution` | Performance |

## Key Service Methods

- `describe()`
- `evaluate()`
- `create_portfolio()`
- `get_portfolio()`
- `list_portfolios()`
- `close_portfolio()`
- `add_holding()`
- `remove_holding()`
- `get_holding()`
- `list_holdings()`

_(See `service.py` for complete API.)_

## Interoperability

`fintech_portfolio` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use fintech_portfolio;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `FINTECH_PORTFOLIO_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
