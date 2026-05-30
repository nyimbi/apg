# Payroll Specification

## Purpose

`pay_payroll` gives APG applications a governed payroll capability for payroll periods, pay groups, employee pay profiles, pay components, time imports, payroll runs, line items, taxes, adjustments, payments, payslips, tax filings, and payroll-agent review. The package must run without external services and make every production integration boundary explicit.

## Capability Identity

- Capability id: `pay_payroll`
- Display name: `Payroll`
- Version: `2.1.0`
- Target: `python`
- Profile: `capability`
- Event stream: `apg.hcm.pay.payroll.lifecycle`
- Stream processor: `bytewax`
- Theme: `payroll_control`

## Domain Records

Payroll records include periods, pay groups, employee pay profiles, pay components, time imports, payroll runs, line items, tax records, adjustments, payment batches, payslips, tax filings, and payroll agents. Each record is tenant-scoped, has a status, and emits audit-style lifecycle evidence.

Supported pay frequencies are weekly, biweekly, semimonthly, and monthly. Supported component types are earning, deduction, tax, benefit, reimbursement, and garnishment. Supported payment methods are bank transfer, check, cash, mobile money, and pay card. Supported tax scopes are employee, employer, statutory, local, and social security.

## Lifecycle Workflows

### Payroll Setup

1. Create payroll period with dates, pay date, frequency, and currency.
2. Create pay group with country, owner, currency, and frequency.
3. Create employee pay profile with payment method, tax id, currency, and base pay.
4. Create reusable pay components.

### Payroll Processing

1. Record time imports and overtime evidence.
2. Start payroll run for a period and pay group.
3. Add earnings, deductions, benefits, reimbursements, and garnishment line items.
4. Record employee and employer tax amounts.
5. Record approved adjustments.
6. Approve and post the payroll run.

### Payments And Compliance

1. Create payment batch from an approved run with positive net pay.
2. Publish payslips only from posted runs and with privacy basis.
3. Create approved tax filings for the run and authority.
4. Emit lifecycle evidence through Bytewax metadata.

### AI-Agent Composition

1. Register payroll agents with supported runtime and role.
2. Limit agent scope to inspection, preparation, validation, and recommendation.
3. Require human approval for privileged payroll actions.

## Rule Engine

The deterministic rule engine returns `decision`, `matched_rules`, and `effects`. Rules cover tenant context, write policy attachment, period completeness, pay group completeness, employee pay profile validation, component validation, time import validation, payroll run creation, line item validation, tax records, adjustments, approval, posting, payments, payslips, tax filings, Bytewax routing, agent runtime and role support, and privileged-agent approval.

## UI Contract

The capability exposes APG screen metadata for dashboard, periods, pay groups, profiles, components, time imports, runs, line items, taxes, adjustments, payments, payslips, filings, agents, and settings. `views.py` returns framework-neutral screen models so generated Python applications can render the capability without importing Flask-AppBuilder.

## Event Contract

Lifecycle events use stream `apg.hcm.pay.payroll.lifecycle`, key `tenant_id`, and processor `bytewax`. Events cover payroll setup, runs, line items, tax records, adjustments, approvals, posting, payments, payslips, tax filings, and payroll-agent registration.

## Acceptance Criteria

- Top-level imports must not require Flask, SQLAlchemy, databases, Redis, payroll engines, banking systems, tax systems, or AI runtimes.
- `get_capability_contract()` must expose configuration, rules, UI, theme, streaming, provides, and requires.
- The service must enforce guardrails before state changes.
- API helpers, view models, semantic model, manifest, and self-test must be executable.
- Bytewax must be the only lifecycle stream processor named by the contract.
- Tests must cover contract shape, guardrails, lifecycle execution, API helpers, view models, app self-test, Bytewax metadata, and payroll-agent metadata.
