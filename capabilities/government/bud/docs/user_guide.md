# Budget Management

**Capability ID**: `government_bud` | **Domain**: `government` | **Version**: `1.0.0`

## Description

Programme budgeting, vote accounting, commitment control, budget revisions, fiscal reporting, and Treasury submission for government entities. Enforces appropriation limits, prevents over-commitment, and ensures every budget revision carries a treasury notification reference.

## Installation

```bash
pip install apg-government-bud
```

## Provides

- `budget_programme_workflow`
- `vote_accounting_workflow`
- `budget_revision_workflow`
- `commitment_control_workflow`
- `expenditure_recording_workflow`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/government-bud/dashboard` | `government_bud:view` | Overview |
| `/government-bud/budgets` | `government_bud:budgets` | Planning |
| `/government-bud/votes` | `government_bud:votes` | Planning |
| `/government-bud/revisions` | `government_bud:revisions` | Revisions |
| `/government-bud/commitments` | `government_bud:commitments` | Execution |
| `/government-bud/expenditures` | `government_bud:expenditures` | Execution |
| `/government-bud/reports` | `government_bud:reports` | Reporting |
| `/government-bud/approvals` | `government_bud:approvals` | Governance |

## Key Service Methods

- `describe()`
- `evaluate()`
- `record_budget()`
- `create_budget_ceiling()`
- `requisition()`
- `commitment_check()`
- `payment_approval()`
- `budget_revision()`
- `expenditure_report()`
- `budget_vs_actual()`

_(See `service.py` for complete API.)_

## Interoperability

`government_bud` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use government_bud;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `GOVERNMENT_BUD_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
