# Product Registration

**Capability ID**: `pharma_reg` | **Domain**: `pharma` | **Version**: `1.0.0`

## Description

Manages pharmaceutical product registration across global regulatory regions including dossier compilation, eCTD validation, authority interactions, approval tracking, variation management, renewal lifecycle, certificate storage, and multi-regional procedure coordination. Enforces QP sign-off, eCTD validation, and 180-day renewal alert requirements.

## Installation

```bash
pip install apg-pharma-reg
```

## Provides

- `registration_application_workflow`
- `dossier_compilation_workflow`
- `authority_interaction_workflow`
- `approval_tracking_workflow`
- `lifecycle_maintenance_workflow`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/pharma-reg/dashboard` | `pharma_reg:view` | Overview |
| `/pharma-reg/registrations` | `pharma_reg:registrations` | Registrations |
| `/pharma-reg/registrations/<id>` | `pharma_reg:registrations` | Registrations |
| `/pharma-reg/dossiers` | `pharma_reg:dossiers` | Dossiers |
| `/pharma-reg/dossiers/<id>` | `pharma_reg:dossiers` | Dossiers |
| `/pharma-reg/approvals` | `pharma_reg:approvals` | Approvals |
| `/pharma-reg/interactions` | `pharma_reg:interactions` | Authority |
| `/pharma-reg/procedures` | `pharma_reg:procedures` | Procedures |

## Key Service Methods

- `describe()`
- `evaluate()`
- `prepare_dossier()`
- `dossier_completeness_check()`
- `submit_registration()`
- `track_review_status()`
- `respond_to_query()`
- `registration_approval()`
- `variation_application()`
- `annual_renewal()`

_(See `service.py` for complete API.)_

## Interoperability

`pharma_reg` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use pharma_reg;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `PHARMA_REG_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
