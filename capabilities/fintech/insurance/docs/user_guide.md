# InsurTech

**Capability ID**: `fintech_insurance` | **Domain**: `fintech` | **Version**: `1.1.0`

## Description

InsurTech manages the end-to-end lifecycle of insurance operations: policyholder onboarding, product publishing across life, health, property, motor, travel, crop, and microinsurance lines, quote generation with underwriting evidence, policy binding, premium recording, claim intake, document management, risk assessment, reinsurance attachment, compliance alerts, and governance reviews. It is designed for regulated insurance operations where every quote must have an underwriting reference and every claim must have supporting evidence.

## Installation

```bash
pip install apg-fintech-insurance
```

## Provides

- `insurance_policyholder_workflow`
- `insurance_product_workflow`
- `insurance_quote_workflow`
- `insurance_policy_workflow`
- `insurance_premium_workflow`

## Requires

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/fintech-insurance/dashboard` | `fintech_insurance:view` | Overview |
| `/fintech-insurance/policyholders` | `fintech_insurance:policyholders` | Customers |
| `/fintech-insurance/products` | `fintech_insurance:products` | Products |
| `/fintech-insurance/quotes` | `fintech_insurance:quotes` | Underwriting |
| `/fintech-insurance/policies` | `fintech_insurance:policies` | Policies |
| `/fintech-insurance/premiums` | `fintech_insurance:premiums` | Policies |
| `/fintech-insurance/claims` | `fintech_insurance:claims` | Claims |
| `/fintech-insurance/documents` | `fintech_insurance:documents` | Claims |

## Key Service Methods

- `describe()`
- `evaluate()`
- `onboard_policyholder()`
- `get_policyholder()`
- `list_policyholders()`
- `publish_product()`
- `generate_quote()`
- `create_policy()`
- `bind_policy()`
- `underwrite_policy()`

_(See `service.py` for complete API.)_

## Interoperability

`fintech_insurance` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use fintech_insurance;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `FINTECH_INSURANCE_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
