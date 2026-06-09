# Rental Operations

**Capability ID**: `realestate_ren` | **Domain**: `realestate` | **Version**: `1.0.0`

## Description

End-to-end tenancy lifecycle: application, referencing, right-to-rent checks, deposit registration and accounting, rent collection with shortfall detection, arrears management and legal escalation, notice serving, and renewal pipeline management. Produces a live rent roll for any property.

## Installation

```bash
pip install apg-realestate-ren
```

## Provides

- `tenancy_lifecycle_management`
- `rent_collection_engine`
- `arrears_management_workflow`
- `deposit_accounting`
- `tenancy_renewal_pipeline`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/realestate/ren/dashboard` | `realestate_ren:view` | Overview |
| `/realestate/ren/tenancies` | `realestate_ren:tenancies` | Tenancies |
| `/realestate/ren/tenancies/<id>` | `realestate_ren:tenancies` | Tenancies |
| `/realestate/ren/referencing` | `realestate_ren:referencing` | Onboarding |
| `/realestate/ren/rent-collection` | `realestate_ren:rent_collection` | Collections |
| `/realestate/ren/arrears` | `realestate_ren:arrears` | Collections |
| `/realestate/ren/deposits` | `realestate_ren:deposits` | Financial |
| `/realestate/ren/renewals` | `realestate_ren:renewals` | Planning |

## Key Service Methods

- `create_tenancy()`
- `get_tenancy()`
- `list_tenancies()`
- `activate_tenancy()`
- `update_tenancy()`
- `record_rent_payment()`
- `list_payments()`
- `_update_arrears()`
- `_clear_arrears()`
- `record_arrears()`

_(See `service.py` for complete API.)_

## Interoperability

`realestate_ren` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use realestate_ren;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `REALESTATE_REN_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
