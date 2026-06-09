# Project Accounting

**Capability ID**: `ppm_pac` | **Domain**: `ppm` | **Version**: `1.0.0`

## Description

Project Accounting (pac) provides complete financial tracking for projects: cost capture, revenue recognition under multiple WIP methods, milestone billing, budget control, and profitability reporting. Every transaction is tenant-scoped, approval-gated, and streamed via Bytewax for real-time financial visibility.

## Installation

```bash
pip install apg-ppm-pac
```

## Provides

- `project_cost_tracking`
- `revenue_recognition_workflow`
- `wip_accounting_workflow`
- `milestone_billing_workflow`
- `project_profitability_reporting`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/ppm-pac/dashboard` | `ppm_pac:view` | Overview |
| `/ppm-pac/accounts` | `ppm_pac:accounts` | Accounts |
| `/ppm-pac/accounts/<id>` | `ppm_pac:accounts` | Accounts |
| `/ppm-pac/costs` | `ppm_pac:costs` | Costs |
| `/ppm-pac/revenue` | `ppm_pac:revenue` | Revenue |
| `/ppm-pac/wip` | `ppm_pac:wip` | WIP |
| `/ppm-pac/billing` | `ppm_pac:billing` | Billing |
| `/ppm-pac/budgets` | `ppm_pac:budgets` | Budgets |

## Key Service Methods

- `describe()`
- `evaluate()`
- `create_account()`
- `get_account()`
- `list_accounts()`
- `project_budget_setup()`
- `cost_code_create()`
- `record_timesheet_cost()`
- `record_expense()`
- `purchase_order_project()`

_(See `service.py` for complete API.)_

## Interoperability

`ppm_pac` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use ppm_pac;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `PPM_PAC_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
