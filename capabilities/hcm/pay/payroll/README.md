# Payroll Capability

`pay_payroll` is the APG capability packet for governed payroll periods, pay groups, employee pay profiles, pay components, time imports, payroll runs, line items, tax calculations, adjustments, payment batches, payslips, tax filings, and payroll-agent review. It keeps the package boundary dependency-light so generated APG applications can compose it immediately while production deployments attach durable employee data, time, benefits, general ledger, banking, tax authority, workflow, audit, notification, and Bytewax topology through adapters.

## What It Provides

- Tenant-scoped payroll periods with frequency, dates, pay date, and currency.
- Pay-group lifecycle for country, owner, currency, and payroll cadence.
- Employee pay profiles with payment method, tax id, currency, base pay, and bank-profile review.
- Pay components for earnings, deductions, tax, benefits, reimbursements, and garnishments.
- Time imports with overtime approval gates.
- Payroll runs with line items, taxes, adjustments, approval, posting, payment batches, payslips, and tax filings.
- First-class payroll agents for Codex, Claude Code, OpenCode, and Pi review teams.
- APG UI route metadata, framework-neutral screen models, compact theme tokens, semantic metadata, package manifest, and release evidence.

## Package Layout

- `SPECIFICATION.md` defines records, workflows, rules, UI, events, adapter boundaries, and acceptance criteria.
- `PLAN.md` records the implementation and review plan for this lifecycle packet.
- `cap_spec.md` summarizes the current executable runtime contract.
- `capability_contract.py` exposes the executable APG contract and deterministic rule engine.
- `service.py` implements the dependency-light lifecycle service.
- `api.py` exposes composition helpers and legacy endpoint shims.
- `views.py` exposes framework-neutral screen models and legacy view shims.
- `app.py` exposes semantic model, component manifest, and self-test.
- `tests/test_package_contract.py` verifies the package contract, lifecycle, guardrails, API, views, and app surface.

## Runtime Lifecycle

1. Create payroll periods and pay groups.
2. Create employee pay profiles and pay components.
3. Import time records and overtime evidence.
4. Start payroll runs and add line items, taxes, and adjustments.
5. Approve and post payroll runs.
6. Create payment batches, publish payslips, and create tax filings.
7. Register payroll agents that inspect, prepare, and recommend within explicit human-approval boundaries.

## Usage

```python
from capabilities.hcm.pay.payroll import PayrollManagementService

service = PayrollManagementService()
period = service.create_payroll_period(
	"period-jan",
	"tenant-a",
	"January Payroll",
	"monthly",
	"2026-01-01",
	"2026-01-31",
	"2026-02-01",
	"USD",
)
pay_group = service.create_pay_group(
	"group-us",
	"tenant-a",
	"US-MONTHLY",
	"US Monthly",
	"monthly",
	"USD",
	"US",
	"payroll-owner",
)
profile = service.create_employee_pay_profile(
	"profile-1",
	"tenant-a",
	"employee-1",
	pay_group["id"],
	"bank_transfer",
	"TAX-123",
	"USD",
	5000,
	"reviewer",
)
print(service.dashboard_summary("tenant-a"))
```

Generated APG applications can use `api.py`:

```python
from capabilities.hcm.pay.payroll import api

status = api.capability_status("tenant-a")
records = api.list_records("runs", "tenant-a")
```

## Guardrails

- Tenant context is required.
- Write operations require policy context.
- Payroll periods require name, supported frequency, start date, end date, pay date, and supported currency.
- Pay groups require code, name, supported frequency, supported currency, country, and owner.
- Employee pay profiles require employee, pay group, supported payment method, tax id, supported currency, and review for bank transfer profiles.
- Pay components require code, name, supported type, currency, and taxable flag.
- Time imports require period, employee profile, nonnegative hours, source, and overtime approval when overtime is present.
- Payroll runs require period, pay group, and initiator.
- Line items require run, profile, component, amount, and review for negative amounts.
- Tax records require run, profile, supported scope, authority, and amount.
- Adjustments require run, profile, reason, and approval.
- Posting requires payroll approval.
- Payment batches require approved runs, payment date, and positive net pay.
- Payslips require posted runs and privacy basis.
- Tax filings require run, authority, period reference, and approval.
- Payroll batches and events require Bytewax metadata.
- Payroll agents must use supported runtimes and roles.
- Privileged payroll-agent actions require recorded human approval.

## Integration Boundary

This package does not start live payroll engines, banking transfers, tax submissions, or general-ledger postings by default. Production deployments should bind these concerns through adapters:

- identity, authorization, and tenant policy;
- audit vault and event replication;
- employee data, time attendance, benefits, and compensation sources;
- general ledger and cost accounting;
- banking, mobile money, and payment processors;
- tax authority reporting and statutory filing systems;
- notification and workflow routing;
- durable Bytewax topology and event sinks;
- AI-agent runtime orchestration.

## Focused Verification

```bash
./.venv/bin/python -m py_compile capabilities/hcm/pay/payroll/__init__.py capabilities/hcm/pay/payroll/capability_contract.py capabilities/hcm/pay/payroll/service.py capabilities/hcm/pay/payroll/api.py capabilities/hcm/pay/payroll/views.py capabilities/hcm/pay/payroll/app.py capabilities/hcm/pay/payroll/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/hcm/pay/payroll/tests/test_package_contract.py
./.venv/bin/python capabilities/hcm/pay/payroll/app.py
./.venv/bin/apg capabilities inspect pay_payroll --json
./.venv/bin/apg capabilities publish-plan capabilities/hcm/pay/payroll --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/hcm/pay/payroll --json
```
