# Payroll Capability

`pay_payroll` is the APG capability packet for governed payroll periods, pay groups, employee pay profiles, pay components, time imports, payroll runs, line items, tax calculations, adjustments, payment batches, payslips, tax filings, and payroll-agent review. It keeps the package boundary dependency-light so generated APG applications can compose it immediately while production deployments attach durable employee data, time, benefits, general ledger, banking, tax authority, workflow, audit, notification, and Bytewax+NATS topology through adapters.

## What It Provides

- Tenant-scoped payroll periods with frequency, dates, pay date, and currency.
- Pay-group lifecycle for country, owner, currency, and payroll cadence.
- Employee pay profiles with payment method, tax id, currency, base pay, and bank-profile review.
- Pay components for earnings, deductions, tax, benefits, reimbursements, and garnishments.
- Time imports with overtime approval gates.
- Payroll runs with line items, taxes, adjustments, approval, posting, payment batches, payslips, and tax filings.
- First-class payroll agents for Codex, Claude Code, OpenCode, and Pi review teams.
- APG UI route metadata, framework-neutral screen models, compact theme tokens, semantic metadata, package manifest, and release evidence.

### African Payroll Engine

Full multi-country support for KE, UG, TZ, GH, NG, RW, ZM:

| Feature | Method |
|---------|--------|
| PAYE calculation | `calculate_paye` |
| Statutory deductions (NSSF, NHIF, NAPSA, RSSB, SSNIT, PenCom) | `calculate_statutory_deductions` |
| Overtime (time-and-half, double-time, public holiday) | `calculate_overtime` |
| Mid-month pro-ration | `mid_month_hire_calculation` |
| Full gross-to-net payroll run | `run_payroll` |
| Bonus payroll | `process_bonus_payroll` |
| Terminal benefits (gratuity, severance, leave encashment) | `calculate_terminal_benefits` |
| Leave encashment | `calculate_leave_encashment` |
| Payslip generation | `generate_payslip` |
| P9 annual tax certificate | `generate_p9_form` |
| Statutory returns (NSSF/NHIF schedules) | `generate_statutory_returns` |
| NSSF contribution schedule | `nssf_schedules_report` |
| Bank transfer file | `bank_transfer_file` |
| GL journal posting | `gl_posting` |
| Payroll variance report | `payroll_variance_report` |
| Salary advance deduction | `apply_salary_advance_deduction` |
| Court garnishment processing | `process_garnishment` |
| Expatriate tax equalisation | `expatriate_tax_calculation` |
| Salary sacrifice pension | `salary_sacrifice_pension` |

### New World-Class Features (v2.3+)

| Feature | Method(s) |
|---------|-----------|
| Pay simulation / what-if engine | `simulate_pay_change` |
| Multi-currency FX-aware payroll | `set_fx_rate`, `convert_pay_currency` |
| Bulk payroll reversal & GL negation | `reverse_payroll_run` |
| IAS 19 leave liability accrual | `accrue_leave_liability`, `get_leave_liability_summary` |
| Versioned tax table management | `upsert_tax_table` |
| AI-assisted anomaly detection | `detect_payroll_anomalies` |
| Benefits-in-kind (BIK) registration & reporting | `register_benefit_in_kind`, `generate_bik_report` |
| Configurable multi-level approval policy | `configure_approval_policy`, `submit_for_approval` |

## Package Layout

- `SPECIFICATION.md` defines records, workflows, rules, UI, events, adapter boundaries, and acceptance criteria.
- `PLAN.md` records the implementation and review plan for this lifecycle packet.
- `cap_spec.md` summarises the current executable runtime contract.
- `WORLD_CLASS_IMPROVEMENTS.md` catalogues 15 strategic improvements with competitor benchmarks.
- `capability_contract.py` exposes the executable APG contract and deterministic rule engine.
- `service.py` implements the dependency-light lifecycle service.
- `api.py` exposes composition helpers and legacy endpoint shims.
- `views.py` exposes framework-neutral screen models and legacy view shims.
- `app.py` exposes semantic model, component manifest, and self-test.
- `tests/test_package_contract.py` verifies the package contract, lifecycle, guardrails, API, views, and app surface.
- `docs/user_guide.md` — comprehensive user guide.
- `domain/` — business rules, event definitions, calculation primitives, adapters.

## Runtime Lifecycle

1. Create payroll periods and pay groups.
2. Create employee pay profiles and pay components.
3. Optionally configure approval policies and BIK registrations.
4. Import time records and overtime evidence.
5. Start payroll runs — or call `run_payroll` for a fully automated gross-to-net run.
6. Run `detect_payroll_anomalies` before submission.
7. Submit for approval via `submit_for_approval`; approvers clear levels via `approve_payroll_run`.
8. Post and create payment batches; publish payslips.
9. Generate P9/P9A, NSSF schedules, bank transfer files, and GL journals.
10. Register payroll agents that inspect, prepare, and recommend within human-approval boundaries.

## Quick Start

```python
import asyncio
from capabilities.hcm.pay.payroll import PayrollManagementService

svc = PayrollManagementService(tenant_id="acme")

# Create period and pay group
period = svc.create_payroll_period(
    "period-jan-26", "acme", "January 2026", "monthly",
    "2026-01-01", "2026-01-31", "2026-02-01", "KES",
)
pay_group = svc.create_pay_group(
    "pg-ke", "acme", "KE-MONTHLY", "Kenya Monthly",
    "monthly", "KES", "KE", "payroll-owner",
)

# Profile
profile = svc.create_employee_pay_profile(
    "prof-1", "acme", "emp-1", pay_group["id"],
    "bank_transfer", "A123456789K", "KES", 120_000, "reviewer",
)

# PAYE calculation
paye = asyncio.run(svc.calculate_paye(120_000, "KE"))
print(paye)
# {'country': 'KE', 'paye_payable': 26100.0, 'effective_rate': 0.2175, ...}

# Pay simulation
sim = asyncio.run(svc.simulate_pay_change(
    "emp-1", {"base_pay": 150_000, "pension_ee_pct": 5}, "acme",
))
print(sim["net_change"])

# Anomaly detection
run = asyncio.run(svc.run_payroll("period-jan-26", "acme", pay_group["id"], "payroll-admin"))
anomalies = asyncio.run(svc.detect_payroll_anomalies(run["id"], "acme"))
print(anomalies["cleared"])
```

## Generated APG Applications

```python
from capabilities.hcm.pay.payroll import api

status = api.capability_status("acme")
records = api.list_records("runs", "acme")
```

## Guardrails

- Tenant context required for all operations.
- Write operations require policy context attached.
- Payroll reversal is restricted to posted/approved/paid runs; already-reversed runs raise `PayrollError`.
- Anomaly detection flags high-severity items that must be signed off before `approve_payroll_run`.
- BIK types must be one of: `company_car`, `housing`, `medical`, `meals`, `fuel`, `other`.
- FX rate lookups raise `PayrollError` when no rate is found for the requested date.
- Approval policy requires at least one level and a positive SLA.
- `simulate_pay_change` never writes; it is pure computation safe to call in read-only contexts.

## Integration Boundary

This package does not start live payroll engines, banking transfers, tax submissions, or general-ledger postings by default. Production deployments should bind these through adapters:

- Identity, authorisation, and tenant policy.
- Audit vault and event replication (NATS JetStream subjects: `payroll.*`).
- Employee data, time attendance, benefits, and compensation sources.
- General ledger and cost accounting.
- Banking, mobile money, and payment processors.
- Tax authority reporting and statutory filing systems.
- Notification and workflow routing.
- Durable Bytewax+NATS topology and event sinks.
- AI-agent runtime orchestration.

## Focused Verification

```bash
./.venv/bin/python -m py_compile capabilities/hcm/pay/payroll/service.py
./.venv/bin/pytest -q capabilities/hcm/pay/payroll/tests/
./.venv/bin/python capabilities/hcm/pay/payroll/app.py
./.venv/bin/apg capabilities inspect pay_payroll --json
```
