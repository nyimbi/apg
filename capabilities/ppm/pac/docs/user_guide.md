# Project Accounting — User Guide

**Capability ID**: `ppm_pac` | **Domain**: `ppm` | **Version**: `1.1.0`

## Description

Project Accounting (pac) provides complete financial tracking for projects: cost capture, revenue recognition under multiple WIP methods, milestone billing, earned value management, budget control, cash flow forecasting, variance root-cause analysis, and external ledger reconciliation. Every transaction is tenant-scoped, approval-gated, and streamed via Bytewax for real-time financial visibility.

---

## Installation

```bash
pip install apg-ppm-pac
```

---

## Quick Start

```python
import asyncio
from apg_ppm_pac.service import ProjectAccountingService

svc = ProjectAccountingService(tenant_id="acme", actor_id="controller-01")

# 1. Open a project account
account = svc.create_account(
    account_id="pa-001",
    tenant_id="acme",
    project_id="proj-web-relaunch",
    name="Website Relaunch",
    status="active",
    currency="USD",
    budget_amount=120_000.0,
    owner_id="pm-alice",
    evidence_reference="SOW-2026-001",
)

# 2. Define cost codes
async def setup():
    await svc.cost_code_create("proj-web-relaunch", "LAB", "Labour", 80_000.0)
    await svc.cost_code_create("proj-web-relaunch", "SUB", "Subcontractors", 30_000.0)
    await svc.cost_code_create("proj-web-relaunch", "EXP", "Expenses", 10_000.0)

    # 3. Post timesheet costs
    await svc.record_timesheet_cost(
        project_id="proj-web-relaunch",
        cost_code="LAB",
        hours=80.0,
        rate=150.0,
        period="2026-06",
    )

    # 4. Run earned value analysis
    ev = await svc.earned_value_analysis("proj-web-relaunch", "2026-06")
    print(f"CPI: {ev['cpi']}, SPI: {ev['spi']}, EAC: {ev['eac']}")

asyncio.run(setup())
```

---

## Core Concepts

### Project Account
A `ProjectAccount` is the top-level financial container for a project. It carries:
- Budget at Completion (BAC) in a specified currency
- Owner and approval references for governance
- Status: `active`, `closed`, `on_hold`

Every cost, revenue, and adjustment transaction is keyed to an account.

### Cost Codes
Cost codes subdivide the project budget into trackable line items (e.g. LAB, SUB, EXP). Every cost entry references a cost code, enabling granular variance reporting.

### WIP Methods
Revenue recognition uses one of:
- `percentage_of_completion` — revenue proportional to cost progress
- `milestone` — revenue from approved milestone invoices
- `time_and_materials` — revenue equals actual labour + expenses

### Earned Value Management (EVM)
The service computes the full EVM suite per period:
- **PV** (Planned Value), **EV** (Earned Value), **AC** (Actual Cost)
- **SPI** (Schedule Performance Index = EV/PV)
- **CPI** (Cost Performance Index = EV/AC)
- **EAC** (Estimate at Completion = BAC/CPI)
- **ETC** (Estimate to Complete = EAC − AC)
- **CV** (Cost Variance = EV − AC), **SV** (Schedule Variance = EV − PV)

---

## Service Methods

### Account Management

#### `create_account(account_id, tenant_id, project_id, name, status, currency, budget_amount, owner_id, evidence_reference)`
Opens a new project accounting record. All fields are required.

```python
svc.create_account(
    account_id="pa-002", tenant_id="acme", project_id="proj-crm",
    name="CRM Implementation", status="active", currency="USD",
    budget_amount=250_000.0, owner_id="pm-bob", evidence_reference="PO-55123",
)
```

#### `get_account(account_id, tenant_id)` / `list_accounts(tenant_id)`
Retrieve a single account or all accounts for the tenant.

---

### Budget Setup

#### `await project_budget_setup(project_id, budget_lines)`
Define itemised budget lines in a single call.

```python
await svc.project_budget_setup("proj-crm", [
    {"cost_code": "LAB", "description": "Labour", "budget_amount": 150_000, "cost_type": "direct"},
    {"cost_code": "LIC", "description": "Software Licences", "budget_amount": 50_000, "cost_type": "direct"},
    {"cost_code": "TRV", "description": "Travel", "budget_amount": 10_000, "cost_type": "indirect"},
])
```

#### `await cost_code_create(project_id, code, description, budget)`
Register a cost code independently.

---

### Cost Recording

#### `await record_timesheet_cost(project_id, cost_code, hours, rate, period)`
Post labour cost from timesheet hours. Updates the cost code actual and variance.

```python
await svc.record_timesheet_cost("proj-crm", "LAB", 120.0, 175.0, "2026-07")
```

#### `await record_expense(project_id, cost_code, amount, category, approved_by)`
Record an approved non-labour expense.

```python
await svc.record_expense("proj-crm", "TRV", 2_400.0, "accommodation", "controller-01")
```

#### `await purchase_order_project(project_id, supplier, items, total)`
Raise a supplier PO against the project.

#### `await invoice_project_cost(project_id, invoice_id, allocation)`
Allocate an incoming supplier invoice across cost codes.

```python
await svc.invoice_project_cost("proj-crm", "INV-9901", {"LIC": 25_000.0, "LAB": 5_000.0})
```

#### `record_cost(cost_id, tenant_id, account_id, cost_type, transaction_type, amount, ...)`
Low-level cost transaction entry. Use `record_timesheet_cost` or `record_expense` for most cases.

---

### Revenue and WIP

#### `await revenue_recognition_project(project_id, method, period)`
Auto-recognise revenue using the specified WIP method.

```python
rev = await svc.revenue_recognition_project("proj-crm", "percentage_of_completion", "2026-07")
print(f"Recognised: {rev['revenue_recognised']}")
```

#### `recognise_revenue(recognition_id, tenant_id, account_id, revenue_type, wip_method, amount, ...)`
Manual revenue recognition entry with full approval and evidence references.

#### `post_wip_adjustment(wip_id, tenant_id, account_id, adjustment_amount, description, auditor_id, evidence_reference)`
Post a WIP accounting adjustment requiring auditor sign-off.

---

### Billing

#### `raise_invoice(invoice_id, tenant_id, account_id, billing_type, amount, milestone_reference, approval_reference, evidence_reference)`
Raise a milestone billing invoice. Supported billing types: `fixed_price`, `time_and_materials`, `milestone`, `retainer`, `cost_plus`, `progress`.

#### `override_budget(override_id, tenant_id, account_id, original_budget, revised_budget, reason, controller_approval_reference, evidence_reference)`
Record a controller-approved budget revision.

---

### Earned Value Management

#### `await earned_value_analysis(project_id, period)`
Compute PV, EV, AC, SPI, CPI, EAC, ETC, CV, SV for the specified period.

```python
ev = await svc.earned_value_analysis("proj-crm", "2026-07")
print(ev)
# {'cpi': 0.92, 'spi': 1.05, 'eac': 271739.13, 'etc': 121739.13, ...}
```

#### `await ev_trend_analysis(project_id, periods=None)`
Time-series of EVM metrics across stored snapshots with CPI/SPI trend direction and EAC confidence band.

```python
trend = await svc.ev_trend_analysis("proj-crm", ["2026-05", "2026-06", "2026-07"])
print(f"CPI trend: {trend['cpi_trend']}, EAC P80: {trend['eac_p80']}")
```

#### `await three_point_eac(project_id, optimistic_cpi, pessimistic_cpi)`
PERT-weighted EAC with P50, P80, P90 confidence percentiles.

```python
eac = await svc.three_point_eac("proj-crm", optimistic_cpi=1.05, pessimistic_cpi=0.80)
print(f"Expected EAC: {eac['eac_expected']}, P80: {eac['eac_p80']}")
```

---

### Profitability and Reporting

#### `await project_profitability(project_id, period)`
Full P&L: revenue, direct labour, expenses, procurement, gross margin, overhead, net margin.

#### `await project_cost_report(project_id, period)`
Detailed budget vs actual by cost code with variance percentages and status flags.

#### `await budget_vs_actual(project_id, tenant_id=None)`
Quick budget variance: budget, actual, variance amount and percentage, under/over status.

#### `await period_cost_summary(project_id, periods)`
Per-period actual costs, burn rate, and estimated periods to completion.

```python
summary = await svc.period_cost_summary("proj-crm", ["2026-05", "2026-06", "2026-07"])
print(f"Burn rate: {summary['burn_rate_per_period']} / period")
print(f"Periods to completion: {summary['estimated_periods_to_completion']}")
```

#### `profitability_report(tenant_id, account_id, method='gross_margin')`
Synchronous profitability summary for a specific account.

---

### Cash Flow Forecasting

#### `await cash_flow_forecast(project_id, periods)`
Project period-by-period inflow (milestone invoices) and outflow (committed POs) with cumulative running balance.

```python
cf = await svc.cash_flow_forecast("proj-crm", ["2026-07", "2026-08", "2026-09"])
for row in cf["periods"]:
    print(f"{row['period']}: in={row['inflow']} out={row['outflow']} net={row['net']} cum={row['cumulative']}")
```

---

### Budget Alerts

#### `await check_budget_thresholds(project_id, warn_pct=80.0, critical_pct=95.0)`
Evaluate every cost code and return structured alerts with severity `ok` | `warn` | `critical`.

```python
alerts = await svc.check_budget_thresholds("proj-crm", warn_pct=75.0, critical_pct=90.0)
for a in alerts["alerts"]:
    if a["severity"] != "ok":
        print(f"{a['cost_code']}: {a['utilisation_pct']}% used — {a['severity']}")
```

---

### Accruals

#### `await post_accrual(project_id, cost_code, amount, accrual_type, period, reversal_period, description="")`
Post a period-end accrual that auto-reverses in `reversal_period`.  
Supported `accrual_type` values: `labour`, `grni`, `overhead`, `other`.

```python
await svc.post_accrual(
    project_id="proj-crm",
    cost_code="LAB",
    amount=12_000.0,
    accrual_type="labour",
    period="2026-07",
    reversal_period="2026-08",
    description="August labour accrual — contractor invoices pending",
)
```

---

### Variance Classification

#### `await classify_variance(project_id, cost_code)`
Classify budget variance as: `scope_creep` | `rate_variance` | `volume_variance` | `timing_variance` | `true_overrun` | `on_track`.

```python
vc = await svc.classify_variance("proj-crm", "LAB")
print(f"Classification: {vc['classification']} (confidence: {vc['confidence']})")
print(vc["evidence"])
```

---

### Intercompany Recharging

#### `await create_intercompany_recharge(from_project_id, to_project_id, cost_code, amount, markup_pct, evidence_reference)`
Recharge costs from one project to another with optional transfer pricing markup.  
Creates paired debit/credit entries linked by a shared `recharge_id`.

```python
recharge = await svc.create_intercompany_recharge(
    from_project_id="proj-shared-services",
    to_project_id="proj-crm",
    cost_code="LAB",
    amount=5_000.0,
    markup_pct=10.0,
    evidence_reference="IC-AGREE-2026-04",
)
print(f"Recharge amount (with markup): {recharge['recharge_amount']}")
```

---

### External Ledger Reconciliation

#### `await reconcile_with_external_ledger(project_id, external_entries, tolerance_pct=0.5)`
Match external ERP entries (SAP, Xero, Sage) against internal transactions. Returns matched, unmatched, and total variance.

```python
erp_lines = [
    {"cost_code": "LAB", "period": "2026-07", "amount": 21_000.0, "external_ref": "SAP-8812"},
    {"cost_code": "TRV", "period": "2026-07", "amount": 2_400.0, "external_ref": "SAP-8813"},
]
rec = await svc.reconcile_with_external_ledger("proj-crm", erp_lines, tolerance_pct=1.0)
print(f"Matched: {rec['matched_count']}, Unmatched external: {rec['unmatched_external_count']}")
print(f"Total variance: {rec['total_variance']}")
```

---

### Agents and Automation

#### `register_agent(agent_id, tenant_id, name, runtime, role, scope)`
Register an accounting automation agent (e.g. cost accrual bot, EV snapshot scheduler).

#### `validate_agent_action(tenant_id, privileged_scope, human_approval_recorded)`
Enforce human-in-the-loop approval before privileged agent actions.

---

### Compliance and Audit

#### `await accounting_compliance_check(tenant_id=None)`
Check accounts for missing owners and transactions lacking approved status.

#### `await get_audit_events(tenant_id=None)`
Return all audit events for the tenant.

#### `await export_accounting_data(tenant_id=None, format='json')`
Export all accounts and cost transactions as JSON or CSV.

---

## Business Rules Reference

| Rule | Trigger | Effect |
|------|---------|--------|
| tenant_context_required | No tenant_id | deny |
| cost_amount_positive | amount <= 0 | deny |
| revenue_approval_required | No approval_reference | deny |
| negative_revenue_denied | amount <= 0 on revenue | deny |
| wip_auditor_required | No auditor_id on WIP | deny |
| backdated_cost_requires_justification | backdated=True, no justification | deny |
| budget_override_requires_controller | No controller_approval_reference | deny |
| cross_tenant_cost_access_denied | cross-tenant access attempt | deny |
| cost_batch_requires_bytewax | event_stream != "bytewax" | deny |
| privileged_agent_action_requires_human_approval | privileged_scope=True, no approval | deny |
| accrual_reversal_period_different | period == reversal_period | deny |
| intercompany_self_recharge_denied | from_project == to_project | deny |

---

## Composability

```apg
use ppm_pac;
```

| Integration | How |
|-------------|-----|
| **ppm_tex** | Time entries feed `record_timesheet_cost` automatically |
| **ppm_pbl** | Cost baseline provides PV curve; `percent_complete` for EV |
| **ppm_res** | Resource cost rates used in timesheet cost calculations |
| **ERP adapters** | `reconcile_with_external_ledger` accepts any ERP line format |
| **Bytewax** | All audit events publish to `apg.ppm.pac.lifecycle` stream |

---

## Configuration

All keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `PPM_PAC_`.

| Variable | Default | Description |
|----------|---------|-------------|
| `PPM_PAC_TENANT_ID` | `default` | Default tenant context |
| `PPM_PAC_DB_URL` | None | PostgreSQL URL for persistent store |
| `PPM_PAC_WARN_THRESHOLD_PCT` | `80` | Budget utilisation warn level |
| `PPM_PAC_CRITICAL_THRESHOLD_PCT` | `95` | Budget utilisation critical level |
| `PPM_PAC_RECONCILE_TOLERANCE_PCT` | `0.5` | Ledger match tolerance |

---

## Further Reading

- `service.py` — Full business logic implementation
- `models.py` — Data model dataclasses
- `capability_contract.py` — Supported enumerations and business rules
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views
- `WORLD_CLASS_IMPROVEMENTS.md` — Enhancement backlog and rationale
- `README.md` — Quick reference
