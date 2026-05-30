# Payroll Runtime Summary

`pay_payroll` is a package-backed APG capability for payroll periods, pay groups, employee pay profiles, pay components, time imports, payroll runs, payroll lines, taxes, adjustments, payment batches, payslips, tax filings, and payroll-agent review.

## Runtime Contract

- Capability id: `pay_payroll`
- Display name: `Payroll`
- Version: `2.1.0`
- Target: `python`
- Entrypoint: `app.py`
- Service: `service.py`
- API helpers: `api.py`
- View models: `views.py`
- Stream processor: `bytewax`
- Event stream: `apg.hcm.pay.payroll.lifecycle`
- Theme: `payroll_control`

## Provides

- `payroll_period_lifecycle`
- `pay_group_lifecycle`
- `employee_pay_profile_lifecycle`
- `pay_component_lifecycle`
- `time_import_lifecycle`
- `payroll_run_lifecycle`
- `payroll_line_item_lifecycle`
- `payroll_tax_lifecycle`
- `payroll_adjustment_lifecycle`
- `payroll_payment_workflow`
- `payslip_lifecycle`
- `payroll_tax_filing_lifecycle`
- `payroll_dashboard_service`
- `payroll_agents`

## Requires

- `auth`
- `audl`
- `ntfy`
- `composition_events`
- `composition_config`
- `workflow`
- `employee_profile_lifecycle`
- `time_attendance`
- `benefits`
- `general_ledger`
- `banking`
- `tax_authority`

## Screens

The package exposes dashboard, periods, pay groups, profiles, components, time imports, runs, line items, taxes, adjustments, payments, payslips, tax filings, agents, and settings routes under `/hcm/payroll`.

## Guardrail Coverage

The deterministic rule engine covers tenant context, write policy attachment, period completeness, pay group completeness, employee pay profile completeness, bank profile review, component completeness, time import validation, overtime approval, run creation, line item validation, tax records, adjustments, approvals, posting, payment batches, payslips, filings, Bytewax routing, payroll-agent runtime and role, and privileged payroll-agent approval.

## Adapter Boundary

The package intentionally avoids live payroll engine, HRIS, time, benefits, general ledger, banking, tax authority, workflow, audit, notification, and AI-runtime imports at the top level. Production applications attach those systems through APG adapters while using this packet for the executable lifecycle contract and composition surface.
