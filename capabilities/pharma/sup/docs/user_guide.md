# Pharmaceutical Supply Chain

**Capability ID**: `pharma_sup` | **Domain**: `pharma` | **Version**: `1.0.0`

## Description

Manages the pharmaceutical supply chain from active ingredient sourcing through CMO management, demand planning, import licensing, supply security monitoring, purchase order management, and supply contract lifecycle. Enforces approved supplier list requirements, quality agreement obligations, import license verification, and dual sourcing requirements for high-risk products.

## Installation

```bash
pip install apg-pharma-sup
```

## Provides

- `active_ingredient_sourcing_workflow`
- `cmo_management_workflow`
- `demand_planning_workflow`
- `import_licensing_workflow`
- `supply_security_monitoring_workflow`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/pharma-sup/dashboard` | `pharma_sup:view` | Overview |
| `/pharma-sup/suppliers` | `pharma_sup:suppliers` | Suppliers |
| `/pharma-sup/suppliers/<id>` | `pharma_sup:suppliers` | Suppliers |
| `/pharma-sup/asl` | `pharma_sup:asl` | Suppliers |
| `/pharma-sup/cmo` | `pharma_sup:cmo` | CMO |
| `/pharma-sup/cmo/<id>` | `pharma_sup:cmo` | CMO |
| `/pharma-sup/demand` | `pharma_sup:demand` | Planning |
| `/pharma-sup/sop` | `pharma_sup:sop` | Planning |

## Key Service Methods

- `describe()`
- `evaluate()`
- `create_supplier()`
- `qualify_supplier()`
- `suspend_supplier()`
- `get_supplier()`
- `list_suppliers()`
- `activate_cmo()`
- `list_cmos()`
- `create_forecast()`

_(See `service.py` for complete API.)_

## Interoperability

`pharma_sup` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use pharma_sup;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `PHARMA_SUP_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
