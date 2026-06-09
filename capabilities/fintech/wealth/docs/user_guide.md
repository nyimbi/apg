# Wealth Management

**Capability ID**: `fintech_wealth` | **Domain**: `fintech` | **Version**: `1.1.0`

## Description

Wealth Management provides regulated advisory and portfolio services: client profile onboarding with KYC, tax, and risk evidence; suitability assessment across risk tolerance, investment horizon, and goals; portfolio creation with advisor assignment and investment policy statement; advisory mandate setup (advisory, discretionary, model, execution-only); portfolio rebalance proposals with exact 100% allocation totals and analysis evidence; trade order staging with approval gates for large orders; performance recording; and fee schedule management. It is the client-facing wealth services layer that backs Robo Advisory and Portfolio Management.

## Installation

```bash
pip install apg-fintech-wealth
```

## Provides

- `wealth_client_profile_workflow`
- `suitability_profile_workflow`
- `portfolio_management_workflow`
- `advisory_mandate_workflow`
- `portfolio_rebalance_workflow`

## Requires

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/fintech-wealth/dashboard` | `fintech_wealth:view` | Overview |
| `/fintech-wealth/clients` | `fintech_wealth:clients` | Clients |
| `/fintech-wealth/suitability` | `fintech_wealth:suitability` | Clients |
| `/fintech-wealth/portfolios` | `fintech_wealth:portfolios` | Portfolios |
| `/fintech-wealth/mandates` | `fintech_wealth:mandates` | Portfolios |
| `/fintech-wealth/rebalances` | `fintech_wealth:rebalances` | Portfolios |
| `/fintech-wealth/orders` | `fintech_wealth:orders` | Trading |
| `/fintech-wealth/performance` | `fintech_wealth:performance` | Reporting |

## Key Service Methods

- `describe()`
- `evaluate()`
- `client_suitability_assessment()`
- `create_portfolio()`
- `portfolio_rebalance()`
- `asset_allocation_review()`
- `performance_report()`
- `tax_loss_harvesting()`
- `dividend_reinvestment()`
- `financial_plan()`

_(See `service.py` for complete API.)_

## Interoperability

`fintech_wealth` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use fintech_wealth;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `FINTECH_WEALTH_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
