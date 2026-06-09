# Pharmaceutical Manufacturing

**Capability ID**: `pharma_mfg` | **Domain**: `pharma` | **Version**: `1.0.0`

## Description

Manages pharmaceutical manufacturing operations from batch record creation through equipment qualification, yield management, deviation handling, line clearance, raw material management, and QP batch release. Enforces GMP compliance, electronic batch records, QP release signatures, and equipment qualification requirements at every production step.

## Installation

```bash
pip install apg-pharma-mfg
```

## Provides

- `batch_record_management_workflow`
- `manufacturing_execution_workflow`
- `equipment_qualification_workflow`
- `yield_management_workflow`
- `deviation_management_workflow`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/pharma-mfg/dashboard` | `pharma_mfg:view` | Overview |
| `/pharma-mfg/batches` | `pharma_mfg:batches` | Production |
| `/pharma-mfg/batches/<id>` | `pharma_mfg:batches` | Production |
| `/pharma-mfg/batches/<id>/ebr` | `pharma_mfg:ebr` | Production |
| `/pharma-mfg/lines` | `pharma_mfg:lines` | Production |
| `/pharma-mfg/equipment` | `pharma_mfg:equipment` | Equipment |
| `/pharma-mfg/equipment/qualification` | `pharma_mfg:qualification` | Equipment |
| `/pharma-mfg/materials` | `pharma_mfg:materials` | Materials |

## Key Service Methods

- `describe()`
- `evaluate()`
- `create_batch()`
- `start_batch()`
- `release_batch()`
- `reject_batch()`
- `get_batch()`
- `list_batches()`
- `register_equipment()`
- `qualify_equipment()`

_(See `service.py` for complete API.)_

## Interoperability

`pharma_mfg` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use pharma_mfg;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `PHARMA_MFG_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
