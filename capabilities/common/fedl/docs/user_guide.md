# Federated Learning

**Capability ID**: `fedl` | **Domain**: `common` | **Version**: `1.0.0`

## Description

FEDL is the APG capability for privacy-preserving collaborative model training. It lets generated applications create governed federations, attest participants, run approved training rounds, collect participant updates, apply poisoning

## Installation

```bash
pip install apg-common-fedl
```

## Provides

- `federated_learning`
- `privacy_preserving_training`
- `federation_agent_composition`

## Requires

- `aicr`
- `mlcm`
- `encr`
- `mten`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/fedl/dashboard` | `fedl:view` | Overview |
| `/fedl/federations` | `fedl:manage_federations` | Federations |
| `/fedl/participants` | `fedl:view_participants` | Federations |
| `/fedl/attestation` | `fedl:manage_federations` | Federations |
| `/fedl/rounds` | `fedl:run_rounds` | Training |
| `/fedl/updates` | `fedl:run_rounds` | Training |
| `/fedl/aggregation` | `fedl:run_rounds` | Training |
| `/fedl/privacy` | `fedl:manage_privacy` | Governance |

## Key Service Methods

- `describe()`
- `evaluate()`
- `create_federation()`
- `register_participant()`
- `start_round()`
- `submit_update()`
- `aggregate_updates()`
- `release_model()`
- `retire_federation()`
- `register_federation_agent()`

_(See `service.py` for complete API.)_

## Interoperability

`fedl` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use fedl;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `FEDL_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
