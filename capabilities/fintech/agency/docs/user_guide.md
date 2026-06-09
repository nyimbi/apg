# Agency Banking

**Capability ID**: `fintech_agency` | **Domain**: `fintech` | **Version**: `1.1.0`

## Description

Agency Banking extends financial services reach through a network of accredited third-party outlets — retail shops, pharmacies, petrol stations, mobile agents, cooperatives, and community banks — operating under a governed program structure. Each outlet holds a float account, serves KYC/AML-verified customers, and processes transactions across services including cash-in/out, bill payment, airtime, loan disbursement, card services, and government payments.

## Installation

```bash
pip install apg-fintech-agency
```

## Provides

- `agency_program_governance`
- `agency_outlet_lifecycle`
- `agency_agent_accreditation`
- `agency_float_management`
- `agency_customer_workflow`

## Requires

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/fintech-agency/dashboard` | `fintech_agency:view` | Overview |
| `/fintech-agency/programs` | `fintech_agency:manage_programs` | Programs |
| `/fintech-agency/outlets` | `fintech_agency:manage_outlets` | Network |
| `/fintech-agency/agents` | `fintech_agency:manage_agents` | Network |
| `/fintech-agency/float-accounts` | `fintech_agency:float` | Liquidity |
| `/fintech-agency/customers` | `fintech_agency:customers` | Customers |
| `/fintech-agency/transactions` | `fintech_agency:transactions` | Transactions |
| `/fintech-agency/cash-movements` | `fintech_agency:liquidity` | Liquidity |

## Key Service Methods

- `describe()`
- `evaluate()`
- `register_program()`
- `onboard_outlet()`
- `accredit_agent()`
- `open_float_account()`
- `onboard_customer()`
- `record_transaction()`
- `record_cash_movement()`
- `settle_commission()`

_(See `service.py` for complete API.)_

## Interoperability

`fintech_agency` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use fintech_agency;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `FINTECH_AGENCY_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
