# Treasury Management System

**Capability ID**: `fintech_treasury` | **Domain**: `fintech` | **Version**: `1.1.0`

## Description

Treasury Management System provides a world-class, standalone-deployable implementation of treasury management system capabilities for the APG platform. It can be installed independently and composed with other APG capabilities via the standard contract interface.

## Installation

```bash
pip install apg-fintech-treasury
```

## Provides

- `cash_position_management`
- `treasury_dealing_workflow`
- `counterparty_limit_governance`
- `settlement_instruction_workflow`
- `fx_rate_management`

## Requires

- `auth`
- `audl`
- `ntfy`
- `keym`
- `fintech_payments`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/fintech-treasury/dashboard` | `fintech_treasury:view` | Overview |
| `/fintech-treasury/cash` | `fintech_treasury:manage_cash` | Cash |
| `/fintech-treasury/dealing` | `fintech_treasury:deal` | Dealing |
| `/fintech-treasury/limits` | `fintech_treasury:manage_limits` | Risk |
| `/fintech-treasury/settlement` | `fintech_treasury:settle` | Settlement |
| `/fintech-treasury/fx` | `fintech_treasury:manage_fx` | FX |
| `/fintech-treasury/liquidity` | `fintech_treasury:manage_liquidity` | Liquidity |
| `/fintech-treasury/nostro` | `fintech_treasury:reconcile` | Reconciliation |

## Key Service Methods

- `_audit_event()`
- `cash_position()`
- `liquidity_forecast()`
- `fx_exposure_report()`
- `hedge_instrument_create()`
- `hedge_effectiveness_test()`
- `bank_relationship_management()`
- `intercompany_loan()`
- `money_market_placement()`
- `fx_forward_booking()`

_(See `service.py` for complete API.)_

## Interoperability

`fintech_treasury` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use fintech_treasury;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `FINTECH_TREASURY_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
