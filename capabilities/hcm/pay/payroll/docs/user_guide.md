# Payroll Management

**Capability ID**: `pay_payroll` | **Domain**: `hcm` | **Version**: `2.2.0`

## Description

`pay_payroll` is the APG capability packet for governed payroll periods, pay groups, employee pay profiles, pay components, time imports, payroll runs, line items, tax calculations, adjustments, payment batches, payslips, tax filings, and payroll-agent review. It keeps the package boundary dependency-light so generated APG applications can compose it immediately while production deployments attach durable employee data, time, benefits, general ledger, banking, tax authority, workflow, audit, notification, and Bytewax topology through adapters.

## Installation

```bash
pip install apg-hcm-payroll
```

## Provides

- `payroll_period_lifecycle`
- `pay_group_lifecycle`
- `employee_pay_profile_lifecycle`
- `pay_component_lifecycle`
- `payroll_tax_rule_lifecycle`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/hcm/payroll/dashboard` | `pay_payroll:view` | Overview |
| `/hcm/payroll/periods` | `pay_payroll:manage_periods` | Setup |
| `/hcm/payroll/pay-groups` | `pay_payroll:manage_setup` | Setup |
| `/hcm/payroll/profiles` | `pay_payroll:manage_profiles` | Employees |
| `/hcm/payroll/components` | `pay_payroll:manage_setup` | Setup |
| `/hcm/payroll/tax-rules` | `pay_payroll:manage_tax_rules` | Compliance |
| `/hcm/payroll/time-imports` | `pay_payroll:manage_runs` | Processing |
| `/hcm/payroll/runs` | `pay_payroll:manage_runs` | Processing |

## Key Service Methods

- `describe()`
- `evaluate()`
- `create_payroll_period()`
- `create_pay_group()`
- `create_employee_pay_profile()`
- `create_pay_component()`
- `record_time_import()`
- `start_payroll_run()`
- `add_line_item()`
- `record_tax()`

_(See `service.py` for complete API.)_

## Interoperability

`pay_payroll` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use pay_payroll;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `PAY_PAYROLL_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
