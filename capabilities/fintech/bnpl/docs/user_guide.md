# Buy Now Pay Later

**Capability ID**: `fintech_bnpl` | **Domain**: `fintech` | **Version**: `1.1.0`

## Description

Buy Now Pay Later manages the lifecycle of deferred payment products for consumers and merchants: BNPL program governance, consumer and merchant onboarding, checkout session capture, affordability decisioning, repayment plan creation, installment scheduling, merchant settlement, and dispute handling. It enforces consumer protection through mandatory KYC, AML, fraud evidence, and explicit fee disclosure at every stage where a consumer commits to debt.

## Installation

```bash
pip install apg-fintech-bnpl
```

## Provides

- `bnpl_merchant_program_governance`
- `consumer_bnpl_lifecycle`
- `merchant_checkout_workflow`
- `affordability_decisioning`
- `bnpl_plan_workflow`

## Requires

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/fintech-bnpl/dashboard` | `fintech_bnpl:view` | Overview |
| `/fintech-bnpl/programs` | `fintech_bnpl:manage_programs` | Programs |
| `/fintech-bnpl/consumers` | `fintech_bnpl:manage_consumers` | Consumers |
| `/fintech-bnpl/merchants` | `fintech_bnpl:manage_merchants` | Merchants |
| `/fintech-bnpl/checkouts` | `fintech_bnpl:manage_checkouts` | Checkout |
| `/fintech-bnpl/affordability` | `fintech_bnpl:decisioning` | Risk |
| `/fintech-bnpl/plans` | `fintech_bnpl:plans` | Plans |
| `/fintech-bnpl/installments` | `fintech_bnpl:installments` | Plans |

## Key Service Methods

- `describe()`
- `evaluate()`
- `register_merchant_program()`
- `onboard_consumer()`
- `register_merchant()`
- `create_checkout_session()`
- `record_affordability_decision()`
- `create_bnpl_plan()`
- `schedule_installment()`
- `record_merchant_settlement()`

_(See `service.py` for complete API.)_

## Interoperability

`fintech_bnpl` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use fintech_bnpl;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `FINTECH_BNPL_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
