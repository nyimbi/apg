# Customer Management

**Capability ID**: `telecom_cus` | **Domain**: `telecom` | **Version**: `1.0.0`

## Description

End-to-end customer lifecycle management covering onboarding, KYC verification, plan activation, SIM and device management, and customer service case tracking. Enforces KYC requirements, credit checks for postpaid plans, IMEI blacklist checks, and tenant-scoped PII access controls.

## Installation

```bash
pip install apg-telecom-cus
```

## Provides

- `customer_lifecycle_workflow`
- `kyc_workflow`
- `plan_management_workflow`
- `sim_management_workflow`
- `device_management_workflow`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/telecom-cus/dashboard` | `telecom_cus:view` | Overview |
| `/telecom-cus/customers` | `telecom_cus:customers` | Customers |
| `/telecom-cus/customers/<id>` | `telecom_cus:customers` | Customers |
| `/telecom-cus/kyc` | `telecom_cus:kyc` | Compliance |
| `/telecom-cus/plans` | `telecom_cus:plans` | Products |
| `/telecom-cus/sims` | `telecom_cus:sims` | Assets |
| `/telecom-cus/devices` | `telecom_cus:devices` | Assets |
| `/telecom-cus/cases` | `telecom_cus:cases` | Support |

## Key Service Methods

- `describe()`
- `evaluate()`
- `create_customer()`
- `update_customer_status()`
- `submit_kyc_document()`
- `verify_kyc()`
- `reject_kyc()`
- `activate_plan()`
- `provision_sim()`
- `update_sim_status()`

_(See `service.py` for complete API.)_

## Interoperability

`telecom_cus` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use telecom_cus;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `TELECOM_CUS_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
