# Payroll Management — User Guide

**Capability ID**: `pay_payroll` | **Domain**: `hcm` | **Version**: `2.3.0`

---

## Overview

`pay_payroll` is the APG capability packet for governed payroll processing across seven African jurisdictions (Kenya, Uganda, Tanzania, Ghana, Nigeria, Rwanda, Zambia). It provides a complete gross-to-net payroll engine — PAYE calculation, statutory deductions, payslips, P9/P10 forms, GL journals, bank transfer files, and a suite of world-class features including pay simulation, payroll reversal, IAS 19 leave liability, BIK reporting, and AI-assisted anomaly detection.

The service is dependency-light: it works with no external infrastructure and integrates with databases, NATS, banking APIs, and tax authority portals through adapters in `domain/adapters.py`.

---

## Installation

```bash
pip install apg-hcm-payroll
```

Or within the APG monorepo:

```python
from capabilities.hcm.pay.payroll import PayrollManagementService
```

---

## Core Concepts

| Concept | Description |
|---------|-------------|
| **Payroll Period** | A date range with frequency (monthly, bi-weekly, weekly) and pay date |
| **Pay Group** | Employees sharing a country, currency, and pay cadence |
| **Employee Pay Profile** | Per-employee record holding base pay, tax ID, bank account, and pay group assignment |
| **Pay Component** | Named earning, deduction, benefit, reimbursement, or garnishment with taxability flag |
| **Payroll Run** | The execution of gross-to-net calculation for a period/pay-group combination |
| **Line Item** | One component amount for one employee in one run |
| **Payslip** | Published pay advice for a single employee-run |
| **Tax Filing** | Statutory return submitted to a revenue authority |
| **Approval Policy** | Multi-level sign-off configuration with SLA enforcement |

---

## Setup Workflow

### 1. Create a Payroll Period

```python
from capabilities.hcm.pay.payroll import PayrollManagementService

svc = PayrollManagementService(tenant_id="acme")

period = svc.create_payroll_period(
    "period-2026-01",  # period_id
    "acme",            # tenant_id
    "January 2026",    # name
    "monthly",         # frequency: monthly | bi_weekly | weekly | semi_monthly
    "2026-01-01",      # start_date
    "2026-01-31",      # end_date
    "2026-02-05",      # pay_date
    "KES",             # currency
)
```

### 2. Create a Pay Group

```python
pay_group = svc.create_pay_group(
    "pg-ke-001", "acme",
    "KE-MONTHLY",         # code — unique within tenant
    "Kenya Monthly Staff",
    "monthly",
    "KES",
    "KE",                 # country ISO code
    "hr-manager-001",     # owner_id
)
```

### 3. Create Employee Pay Profiles

```python
profile = svc.create_employee_pay_profile(
    "prof-emp001", "acme",
    "emp-001",             # employee_id
    pay_group["id"],
    "bank_transfer",       # payment_method: bank_transfer | mobile_money | cash | cheque
    "A123456789K",         # KRA PIN / tax ID
    "KES",
    120_000,               # base_pay (monthly gross)
    "reviewer-01",         # reviewed_by — required for bank_transfer
    basic_pay=72_000,      # basic salary (60% of gross by default if omitted)
    hire_date="2022-03-01",
    bank_account="0123456789",
)
```

### 4. Create Pay Components

```python
housing_comp = svc.create_pay_component(
    "comp-housing", "acme",
    "HOUSE",               # code
    "Housing Allowance",
    "earning",             # type: earning | deduction | benefit | reimbursement | garnishment
    "KES",
    False,                 # taxable — False for housing in KE (exempt up to limit)
)
```

---

## Running Payroll

### Automated Gross-to-Net Run

`run_payroll` handles the entire waterfall: loads all active profiles in the pay group, computes statutory deductions, PAYE, and net pay, stores line items and tax records, and returns a run record with full payslip lines.

```python
import asyncio

run = asyncio.run(svc.run_payroll(
    period["id"],
    "acme",
    pay_group["id"],
    "payroll-admin-01",
    employee_filter=None,  # None = all employees in group
))
print(run["totals"])
# {'gross': 120000.0, 'deductions': 9360.0, 'taxes': 26100.0, 'net': 84540.0, ...}
```

### Manual Run (Step-by-Step)

```python
run = svc.start_payroll_run("run-jan-26", "acme", period["id"], pay_group["id"], "admin")

# Add earnings
basic_comp = svc.create_pay_component("comp-basic", "acme", "BASIC", "Basic Salary", "earning", "KES", True)
svc.add_line_item("li-001", "acme", run["id"], profile["id"], basic_comp["id"], 72_000, "reviewer")

# Record taxes
svc.record_tax("tax-paye-001", "acme", run["id"], profile["id"], "income_tax", "KRA", 18_300)

# Record statutory deductions
svc.record_adjustment("adj-nssf", "acme", run["id"], profile["id"], -2_160, "NSSF EE", "system")
```

---

## Tax Calculations

### PAYE

```python
paye = asyncio.run(svc.calculate_paye(
    120_000,   # gross_monthly
    "KE",      # country: KE | UG | TZ | GH | NG | RW | ZM
    allowances={"insurance_premium": 3_000, "mortgage_interest": 20_000},
    deductions={"nssf_ee": 2_160},
))
# Returns: taxable_income, tax_before_relief, reliefs_applied, paye_payable, effective_rate
```

### Statutory Deductions

```python
stat = asyncio.run(svc.calculate_statutory_deductions(
    {"basic_pay": 72_000},  # employee dict
    120_000,                 # gross
    "KE",
))
# Returns: ee_total, er_total, breakdown (NSSF, NHIF, NITA per country)
```

### Versioned Tax Tables

When a budget cycle changes PAYE rates, load the new table without a code deploy:

```python
asyncio.run(svc.upsert_tax_table(
    "KE",
    "2026-07-01",  # new rates effective July 2026
    [
        (24_000,       0.10),
        (8_333,        0.25),
        (float("inf"), 0.30),
    ],
    {
        "personal_relief": 2_400,
        "insurance_relief_rate": 0.15,
        "insurance_relief_max": 5_000,
        "mortgage_relief_rate": 0.25,
        "mortgage_relief_max": 25_000,
        "pension_relief_max": 20_000,
    },
    "acme",
))
```

`calculate_paye` automatically selects the most-recent table whose `valid_from <= pay_period_start_date`.

---

## Pay Simulation

Simulate the impact of a salary change before committing:

```python
sim = asyncio.run(svc.simulate_pay_change(
    "emp-001",
    {
        "base_pay": 150_000,
        "pension_ee_pct": 5,        # voluntary additional pension as % of gross
        "housing_allowance": 10_000,
    },
    "acme",
))
print(sim["before"]["net"], "→", sim["after"]["net"])
print("Net change:", sim["net_change"])
print("PAYE change:", sim["paye_change"])
```

`simulate_pay_change` is read-only — no records are written or modified.

---

## Multi-Currency Payroll

For expatriates paid in USD while statutory obligations are in KES:

```python
# Store today's rate
asyncio.run(svc.set_fx_rate("USD", "KES", 128.50, "2026-01-01", "acme"))

# Convert net pay for bank file
result = asyncio.run(svc.convert_pay_currency(
    5_000,       # USD net pay
    "USD", "KES",
    "2026-01-31",
    "acme",
))
print(result["converted_amount"])  # 642_500.0 KES
```

---

## Approval Workflows

### Configure a Multi-Level Policy

```python
asyncio.run(svc.configure_approval_policy(
    pay_group["id"],
    levels=[
        {"level": 1, "role": "finance_manager"},
        {"level": 2, "role": "cfo"},
    ],
    amount_threshold=5_000_000,  # dual sign-off required for runs above KES 5M
    sla_hours=24,
    tenant_id="acme",
))
```

### Submit and Approve

```python
# Submit for approval
req = asyncio.run(svc.submit_for_approval(run["id"], "payroll-admin", "acme"))

# Level-1 approval
svc.approve_payroll_run(run["id"], "acme", "finance-mgr-01")

# Post and pay
svc.post_payroll_run(run["id"], "acme", "finance-mgr-01")
svc.create_payment_batch("batch-jan-26", "acme", run["id"], "2026-02-05")
```

---

## Anomaly Detection

Run before seeking approval to gate on anomalies:

```python
report = asyncio.run(svc.detect_payroll_anomalies(run["id"], "acme", sensitivity="medium"))
if not report["cleared"]:
    for anomaly in report["anomalies"]:
        if anomaly["severity"] == "high":
            print(f"HIGH: {anomaly['employee_id']} — {anomaly['check']}: {anomaly['detail']}")
```

Checks performed:
1. **Gross statistical outlier** — employee gross > mean + Nσ (N=2 at medium sensitivity).
2. **New bank account** — bank account not seen in any prior approved run.
3. **Same-day hire and payroll** — employee hired on the same date the run was created.

---

## Payroll Reversal

```python
reversal = asyncio.run(svc.reverse_payroll_run(
    run["id"],
    "Incorrect bonus included — reprocess required",
    "cfo-001",
    "acme",
))
# Original run → status: 'reversed'
# Negating GL entries created automatically
# Bank files marked 'voided'
```

After reversal, call `run_payroll` with corrected data to create a fresh run.

---

## Leave Liability (IAS 19)

```python
# Accrue for one employee
accrual = asyncio.run(svc.accrue_leave_liability(
    "emp-001",
    21.5,          # unconsumed leave days
    "2026-01-31",
    "acme",
))
print(accrual["liability_amount"])  # KES amount posted to GL account 2130

# Month-end summary
summary = asyncio.run(svc.get_leave_liability_summary("acme", "2026-01"))
print(summary["total_liability"])
```

---

## Benefits in Kind (BIK)

```python
# Register a company car BIK
asyncio.run(svc.register_benefit_in_kind(
    "emp-001",
    "company_car",
    10_000,         # KES/month imputed value
    "2026-01-01",
    "acme",
))

# Annual BIK report for KRA submission
report = asyncio.run(svc.generate_bik_report("acme", "2026-12-31"))
print(report["grand_total_bik"])
```

BIK is added to `gross_monthly` inside `calculate_paye` for imputed tax purposes; it does not produce a cash payment line.

---

## Overtime & Special Pays

```python
# Overtime
ot = asyncio.run(svc.calculate_overtime(
    "emp-001", 173.33, 20.0, "time_and_half", "acme",
))

# Bonus payroll
bonus_run = asyncio.run(svc.process_bonus_payroll(
    period["id"], "acme", pay_group["id"], "hr-admin",
    bonuses={"emp-001": 30_000, "emp-002": 25_000},
))

# Terminal benefits (gratuity on exit)
terminal = asyncio.run(svc.calculate_terminal_benefits(
    "emp-001", "2026-01-31", "retrenchment", "acme",
))
```

---

## Statutory Reports

```python
# P9 annual tax certificate
p9 = asyncio.run(svc.generate_p9_form("emp-001", "2025", "acme"))

# NSSF schedule
nssf = asyncio.run(svc.nssf_schedules_report(run["id"], "acme"))

# Statutory returns (all schemes for period)
returns = asyncio.run(svc.generate_statutory_returns(run["id"], "acme"))

# Bank transfer file (EFT format)
btf = asyncio.run(svc.bank_transfer_file(run["id"], "acme", "EQUITY_BANK_KE"))

# GL journal entries
gl = asyncio.run(svc.gl_posting(run["id"], "acme", "CFO-001"))
```

---

## Expatriate Tax Equalisation

```python
teq = asyncio.run(svc.expatriate_tax_calculation(
    "emp-expat-001",
    "2026-01",
    "acme",
    home_country="GB",
    host_country="KE",
    company_bearing_tax=True,
))
print(teq["company_tax_cost"])   # amount employer bears above hypo-tax
print(teq["net_to_employee"])    # guaranteed net (hypo-tax basis)
```

---

## Salary Sacrifice Pension

```python
sacrifice = asyncio.run(svc.salary_sacrifice_pension(
    "emp-001",
    5.0,          # 5% of gross — or pass fixed amount with is_percentage=False
    "acme",
    is_percentage=True,
))
print(sacrifice["paye_saving"])          # monthly PAYE saving
print(sacrifice["net_pay_after_sacrifice"])
```

---

## Variance Report

```python
variance = asyncio.run(svc.payroll_variance_report(
    "run-jan-26", "run-feb-26", "acme",
))
# Returns: gross_delta, net_delta, tax_delta, employee_count_delta, detail per employee
```

---

## Payroll Dashboard

```python
summary = svc.dashboard_summary("acme")
# Returns: period_count, run_count, profile_count, net_pay_total, streaming config, ...
```

---

## Streaming Architecture

All mutations emit events via `_emit`. In production, bind a NATS client:

```python
import nats

async def setup():
    nc = await nats.connect("nats://localhost:4222")
    svc = PayrollManagementService(tenant_id="acme")
    svc.nats_client = nc   # adapter pattern — see domain/adapters.py
```

NATS subjects emitted:
- `payroll.period.created`, `payroll.run.started`, `payroll.run.approved`
- `payroll.run.posted`, `payroll.run.reversed`, `payroll.payslip.published`
- `payroll.tax_filing.created`, `payroll.bik.registered`
- `payroll.leave_liability.accrued`, `payroll.approval_policy.configured`

Bytewax pipelines subscribe to these subjects for downstream processing (GL sync, notifications, analytics).

---

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
| `/hcm/payroll/simulate` | `pay_payroll:view` | Self-Service |
| `/hcm/payroll/anomalies` | `pay_payroll:manage_runs` | Compliance |
| `/hcm/payroll/bik` | `pay_payroll:manage_setup` | Compliance |

---

## Interoperability

Reference in `.apg` source files:

```apg
use pay_payroll;
```

Compose with:
- `hcm_employees` — employee master data
- `hcm_leave` — leave balance feeds for leave encashment and IAS 19 accruals
- `fin_gl` — chart of accounts for GL posting
- `fin_banking` — bank file delivery
- `ntfy` — payslip delivery notifications
- `conf` — tenant-scoped configuration

---

## Configuration

All keys are tenant-scoped via the `conf` capability or environment variables prefixed `PAY_PAYROLL_`:

| Key | Default | Description |
|-----|---------|-------------|
| `PAY_PAYROLL_DEFAULT_COUNTRY` | `KE` | Country for new pay groups |
| `PAY_PAYROLL_WORKING_DAYS` | `21` | Days/month for leave liability |
| `PAY_PAYROLL_GL_GROSS_EXPENSE` | `5001` | GL account for gross pay expense |
| `PAY_PAYROLL_GL_PAYE_PAYABLE` | `2101` | GL account for PAYE payable |
| `PAY_PAYROLL_GL_NET_PAY` | `2110` | GL account for net pay payable |
| `PAY_PAYROLL_GL_LEAVE_LIABILITY` | `2130` | GL account for leave liability |
| `PAY_PAYROLL_NATS_SUBJECT_PREFIX` | `payroll` | NATS subject prefix |

---

## Further Reading

- `service.py` — Business logic (3 800+ lines)
- `models.py` — SQLAlchemy and Pydantic data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `capability_contract.py` — Deterministic rule engine
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 strategic enhancements with competitor benchmarks
- `domain/calculations.py` — Pure calculation primitives
- `domain/rules.py` — Business rule definitions
- `domain/events.py` — Event type catalogue
