# Real Estate Accounting

**Capability ID**: `realestate_acc` | **Domain**: `realestate` | **Version**: `1.0.0`

## Description

Provides the full property accounting stack: chart-of-accounts management, journal entry posting with period controls, service charge raising and approval, CAM (Common Area Maintenance) reconciliation, IFRS 16 lease liability and right-of-use asset schedules, revenue recognition under multiple methods, dual-control period close, and tenant account statements.

## Installation

```bash
pip install apg-realestate-acc
```

## Provides

- `property_ledger_management`
- `service_charge_accounting`
- `cam_reconciliation_workflow`
- `ifrs16_lease_accounting`
- `revenue_recognition_engine`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/realestate/acc/dashboard` | `realestate_acc:view` | Overview |
| `/realestate/acc/ledger` | `realestate_acc:ledger` | Ledger |
| `/realestate/acc/journals` | `realestate_acc:journals` | Ledger |
| `/realestate/acc/service-charges` | `realestate_acc:service_charges` | Charges |
| `/realestate/acc/cam` | `realestate_acc:cam` | Charges |
| `/realestate/acc/ifrs16` | `realestate_acc:ifrs16` | Compliance |
| `/realestate/acc/revenue` | `realestate_acc:revenue` | Revenue |
| `/realestate/acc/period-close` | `realestate_acc:period_close` | Periods |

## Key Service Methods

- `create_account()`
- `get_account()`
- `list_accounts()`
- `update_account()`
- `create_journal_entry()`
- `approve_journal_entry()`
- `post_journal_entry()`
- `reverse_journal_entry()`
- `list_journals()`
- `raise_service_charge()`

_(See `service.py` for complete API.)_

## Interoperability

`realestate_acc` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use realestate_acc;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `REALESTATE_ACC_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
