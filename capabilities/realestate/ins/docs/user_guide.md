# Property Insurance

**Capability ID**: `realestate_ins` | **Domain**: `realestate` | **Version**: `1.0.0`

## Description

End-to-end property insurance portfolio management: policy creation and binding with asset schedules, claims lodgement through settlement with large-claim senior-approval gates, endorsement issuance, premium allocation across properties, automated coverage gap detection, insurer/broker registry, and renewal pipeline tracking.

## Installation

```bash
pip install apg-realestate-ins
```

## Provides

- `policy_lifecycle_management`
- `asset_schedule_management`
- `claims_processing_workflow`
- `premium_allocation_engine`
- `coverage_gap_analysis`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/realestate/ins/dashboard` | `realestate_ins:view` | Overview |
| `/realestate/ins/policies` | `realestate_ins:policies` | Policies |
| `/realestate/ins/assets` | `realestate_ins:assets` | Assets |
| `/realestate/ins/claims` | `realestate_ins:claims` | Claims |
| `/realestate/ins/premiums` | `realestate_ins:premiums` | Financial |
| `/realestate/ins/gaps` | `realestate_ins:gaps` | Analysis |
| `/realestate/ins/endorsements` | `realestate_ins:endorsements` | Policies |
| `/realestate/ins/insurers` | `realestate_ins:insurers` | Registry |

## Key Service Methods

- `register_insurer()`
- `get_insurer()`
- `list_insurers()`
- `create_policy()`
- `get_policy()`
- `list_policies()`
- `bind_policy()`
- `update_policy()`
- `get_renewal_pipeline()`
- `add_asset_to_schedule()`

_(See `service.py` for complete API.)_

## Interoperability

`realestate_ins` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use realestate_ins;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `REALESTATE_INS_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
