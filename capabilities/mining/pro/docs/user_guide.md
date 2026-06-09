# Mine Production Operations

**Capability ID**: `mining_pro` | **Domain**: `mining` | **Version**: `1.0.0`

## Description

Manages daily mine production operations including shift reporting, ore and waste movement tracking, blast design and firing authorisation, grade control boundary management, stockpile inventory, and production scheduling. Enforces a strict blast status state machine, requires fire authority before detonation, and gates grade boundary changes behind approval workflows to prevent unauthorised ore/waste misclassification.

## Installation

```bash
pip install apg-mining-pro
```

## Provides

- `shift_report_workflow`
- `production_ledger_management`
- `blast_design_workflow`
- `blast_firing_authorization`
- `ore_tracking_management`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/mining-pro/dashboard` | `mining_pro:view` | Overview |
| `/mining-pro/shifts` | `mining_pro:view` | Shift Operations |
| `/mining-pro/shifts/create` | `mining_pro:write` | Shift Operations |
| `/mining-pro/shifts/:id` | `mining_pro:view` | Shift Operations |
| `/mining-pro/production` | `mining_pro:view` | Production |
| `/mining-pro/ore-tracking` | `mining_pro:write` | Production |
| `/mining-pro/blasts` | `mining_pro:view` | Blasting |
| `/mining-pro/blasts/create` | `mining_pro:blast_design` | Blasting |

## Key Service Methods

- `create_shift_report()`
- `get_shift_report()`
- `update_shift_report()`
- `submit_shift_report()`
- `approve_shift_report()`
- `list_shift_reports()`
- `create_blast()`
- `get_blast()`
- `update_blast()`
- `approve_blast_design()`

_(See `service.py` for complete API.)_

## Interoperability

`mining_pro` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use mining_pro;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `MINING_PRO_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
