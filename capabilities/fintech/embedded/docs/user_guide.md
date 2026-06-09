# Embedded Finance

**Capability ID**: `fintech_embedded` | **Domain**: `fintech` | **Version**: `1.1.0`

## Description

Embedded Finance enables non-financial businesses to offer financial products inside their own applications without owning banking infrastructure. It manages partner program onboarding, host application registration, product placement publishing, customer consent capture, and the end-to-end lifecycle of embedded accounts, payments, card offers, lending offers, settlement batches, and revenue share — all within a consent-scoped access model.

## Installation

```bash
pip install apg-fintech-embedded
```

## Provides

- `partner_program_workflow`
- `host_application_workflow`
- `embedded_product_placement_workflow`
- `embedded_customer_consent_workflow`
- `embedded_account_workflow`

## Requires

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/fintech-embedded/dashboard` | `fintech_embedded:view` | Overview |
| `/fintech-embedded/programs` | `fintech_embedded:programs` | Partners |
| `/fintech-embedded/applications` | `fintech_embedded:applications` | Partners |
| `/fintech-embedded/placements` | `fintech_embedded:placements` | Products |
| `/fintech-embedded/consents` | `fintech_embedded:consents` | Consent |
| `/fintech-embedded/accounts` | `fintech_embedded:accounts` | Journeys |
| `/fintech-embedded/payments` | `fintech_embedded:payments` | Journeys |
| `/fintech-embedded/cards` | `fintech_embedded:cards` | Journeys |

## Key Service Methods

- `describe()`
- `evaluate()`
- `register_partner_program()`
- `register_host_application()`
- `publish_product_placement()`
- `capture_customer_consent()`
- `open_embedded_account()`
- `initiate_embedded_payment()`
- `offer_embedded_card()`
- `create_lending_offer()`

_(See `service.py` for complete API.)_

## Interoperability

`fintech_embedded` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use fintech_embedded;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `FINTECH_EMBEDDED_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
