# Backup and Restore

**Capability ID**: `bkup` | **Domain**: `common` | **Version**: `1.0.0`

## Description

BKUP provides governed backup, restore, retention, and continuity operations for APG applications. It covers tenant backup plans, encrypted snapshots, restore approval, stale restore-test review, retention disposition, legal-hold

## Installation

```bash
pip install apg-common-bkup
```

## Provides

- `backup_plan_governance`
- `snapshot_vault`
- `restore_governance`
- `retention_governance`
- `continuity_reporting`

## Requires

- `encr`
- `conf`
- `audl`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/bkup/dashboard` | `bkup:view` | Overview |
| `/bkup/plans` | `bkup:manage_plans` | Plans |
| `/bkup/snapshots` | `bkup:view` | Backups |
| `/bkup/backup` | `bkup:run_backup` | Backups |
| `/bkup/restore` | `bkup:restore` | Recovery |
| `/bkup/restore/approvals` | `bkup:approve_restore` | Recovery |
| `/bkup/retention` | `bkup:admin` | Governance |
| `/bkup/retention/dispositions` | `bkup:approve_retention` | Governance |

## Key Service Methods

- `uuid7str()`
- `_audit()`
- `create_backup_plan()`
- `backup_schedule()`
- `backup_run()`
- `incremental_backup()`
- `differential_backup()`
- `restore_from()`
- `verify_backup()`
- `test_restore()`

_(See `service.py` for complete API.)_

## Interoperability

`bkup` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use bkup;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `BKUP_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
