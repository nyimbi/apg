# Real Estate Accounting — User Guide

**Capability ID**: `realestate_acc` | **Domain**: `realestate` | **Version**: `1.1.0`

## Description

Full property accounting stack covering chart-of-accounts management, journal entry posting with period controls, service charge raising and approval, CAM reconciliation and waterfall allocation, IFRS 16 lease liability and right-of-use asset schedules (including lease modifications), lease incentive amortisation, percentage-rent recognition, budget vs. actual variance reporting, service charge dispute and credit-note workflow, dual-control period close with pre-close checklist, and tenant account statements.

---

## Installation

```bash
pip install apg-realestate-acc
```

---

## Quick Start

```python
import asyncio
from decimal import Decimal
from datetime import date
from capabilities.realestate.acc.service import AccService
from capabilities.realestate.acc.models import (
    AccountCreate, AccountType, LedgerType,
    JournalEntryCreate, JournalLine, JournalType,
    ServiceChargeCreate, ChargeType,
)

svc = AccService(tenant_id="t1", actor_id="user@datacraft.co.ke")

async def demo():
    # Create an account
    acct = await svc.create_account(AccountCreate(
        tenant_id="t1", code="4000", name="Rental Income",
        account_type=AccountType.revenue, ledger_type=LedgerType.rental_income,
        created_by="user@datacraft.co.ke",
    ))
    print(acct.id, acct.code)

asyncio.run(demo())
```

---

## Core Concepts

### Chart of Accounts

Every financial transaction is linked to an account in the chart of accounts. Accounts are typed
(`asset`, `liability`, `equity`, `revenue`, `expense`, `contra`) and classified by ledger type
(`property_ledger`, `service_charge`, `rental_income`, etc.).

```python
acct = await svc.create_account(AccountCreate(
    tenant_id="t1",
    property_id="prop-001",
    code="2100",
    name="Service Charge Payable",
    account_type=AccountType.liability,
    ledger_type=LedgerType.service_charge,
    currency="KES",
    created_by="admin",
))
```

### Journal Entries

Journals must balance (total debit == total credit, within 0.01).  The lifecycle is:
`draft` → `pending_approval` → `approved` → `posted`.  Reversals can only be created for
`posted` journals and automatically swap debit/credit on every line.

```python
entry = await svc.create_journal_entry(JournalEntryCreate(
    tenant_id="t1",
    journal_type=JournalType.manual,
    reference="JNL-2026-001",
    period="2026-06",
    journal_date=date.today(),
    description="Service charge accrual",
    currency="KES",
    created_by="admin",
    lines=[
        JournalLine(account_id=acct.id, account_code="2100",
                    description="SC accrual", debit=Decimal("50000")),
        JournalLine(account_id=acct.id, account_code="4000",
                    description="SC income", credit=Decimal("50000")),
    ],
))
await svc.approve_journal_entry(entry.id, "t1", "manager@datacraft.co.ke")
await svc.post_journal_entry(entry.id, "t1")
```

### Service Charges

Raise and approve service charges per property, per period.

```python
charge = await svc.raise_service_charge(ServiceChargeCreate(
    tenant_id="t1",
    property_id="prop-001",
    charge_type=ChargeType.service_charge,
    lease_id="lease-001",
    description="Q2 service charge",
    amount=Decimal("120000"),
    period="2026-06",
    due_date=date(2026, 6, 30),
    vat_rate=Decimal("0.16"),
    created_by="admin",
))
await svc.approve_service_charge(charge.id, "t1", "manager@datacraft.co.ke")
```

### Service Charge Dispute and Credit Note

When a tenant contests a posted charge, use the dispute workflow.

```python
dispute = await svc.raise_service_charge_dispute(
    charge_id=charge.id,
    tenant_id="t1",
    raised_by="tenant-rep@acme.co.ke",
    dispute_reason="Incorrect utilities allocation",
    disputed_amount=Decimal("20000"),
)

credit_note = await svc.issue_credit_note(
    dispute_id=dispute["id"],
    tenant_id="t1",
    issued_by="manager@datacraft.co.ke",
    credit_amount=Decimal("20000"),
    reason="dispute_resolved",
)
```

### CAM Reconciliation

```python
cam = await svc.start_cam_reconciliation(CamReconciliationCreate(
    tenant_id="t1",
    property_id="prop-001",
    period_year=2026,
    estimated_costs=Decimal("2400000"),
    actual_costs=Decimal("2520000"),
    lease_ids=["lease-001", "lease-002"],
    created_by="admin",
))
await svc.approve_cam_reconciliation(cam.id, "t1", "director@datacraft.co.ke")
await svc.settle_cam_reconciliation(cam.id, "t1")
```

#### CAM Waterfall Allocation

Distribute the settled variance across leases proportionally by net lettable area.

```python
allocation = await svc.allocate_cam_to_leases(
    cam_id=cam.id,
    tenant_id="t1",
    lease_area_map={
        "lease-001": Decimal("1200"),  # 1200 sqm
        "lease-002": Decimal("800"),   # 800 sqm
    },
    allocation_basis="nla",
)
# allocation["lease_allocations"]["lease-001"]["adjustment"] == pro-rata share
```

### IFRS 16 Lease Schedules

```python
from capabilities.realestate.acc.models import Ifrs16ScheduleCreate, Ifrs16Category

schedule = await svc.generate_ifrs16_schedule(Ifrs16ScheduleCreate(
    tenant_id="t1",
    lease_id="lease-001",
    category=Ifrs16Category.finance_lease,
    commencement_date=date(2024, 1, 1),
    expiry_date=date(2029, 12, 31),
    annual_payment=Decimal("600000"),
    discount_rate=Decimal("0.12"),
    created_by="admin",
))
print(schedule.rou_asset, schedule.lease_liability)
```

#### IFRS 16 Lease Modification (Remeasurement)

Use when rent is reviewed, term extended, or partial surrender occurs.

```python
mod = await svc.remeasure_ifrs16_lease(
    schedule_id=schedule.id,
    tenant_id="t1",
    modification_date=date(2026, 6, 1),
    revised_annual_payment=Decimal("660000"),
    revised_discount_rate=Decimal("0.13"),
    revised_expiry_date=date(2030, 12, 31),
    modified_by="admin",
)
print(mod["remeasurement_delta"], mod["delta_direction"])
```

### Lease Incentive Amortisation

Amortise rent-free periods and fit-out contributions straight-line over the lease term.

```python
amort = await svc.amortise_lease_incentive(
    lease_id="lease-001",
    incentive_amount=Decimal("360000"),   # 3 months rent-free
    lease_start=date(2024, 1, 1),
    lease_end=date(2029, 12, 31),
    tenant_id="t1",
    period="2026-06",
    incentive_type="rent_free",
)
print(amort["monthly_charge"], amort["deferred_balance"])
```

### Percentage Rent (Turnover Rent)

```python
pct_rent = await svc.recognise_percentage_rent(
    lease_id="lease-retail-001",
    period="2026-06",
    tenant_id="t1",
    turnover_amount=Decimal("8000000"),
    base_rent=Decimal("300000"),
    breakpoint=Decimal("0"),              # ignored for natural breakpoint
    percentage_rate=Decimal("0.06"),
    breakpoint_type="natural",            # BP = base_rent / rate = 5,000,000
)
print(pct_rent["variable_rent"], pct_rent["total_rent"])
```

### Revenue Recognition

```python
from capabilities.realestate.acc.models import RevenueScheduleCreate, RevenueMethod

rev = await svc.create_revenue_schedule(RevenueScheduleCreate(
    tenant_id="t1",
    lease_id="lease-001",
    property_id="prop-001",
    method=RevenueMethod.straight_line,
    start_date=date(2024, 1, 1),
    end_date=date(2029, 12, 31),
    total_contract_value=Decimal("36000000"),
    created_by="admin",
))
recognised = await svc.recognise_revenue_for_period(rev.id, "t1", "2026-06")
print(recognised["amount"])
```

### Accounting Periods

```python
from capabilities.realestate.acc.models import AccountingPeriodCreate

period = await svc.open_period(AccountingPeriodCreate(
    tenant_id="t1",
    period="2026-06",
    opened_by="admin",
))

# Review checklist before closing
checklist = await svc.get_period_close_checklist("2026-06", "t1")
if checklist["checklist_complete"]:
    await svc.close_period(period.id, "t1", "director@datacraft.co.ke", "cfo@datacraft.co.ke")
else:
    print("Blocking items:", [i for i in checklist["items"] if not i["complete"]])
```

### Budget vs. Actual Variance Report

```python
# First create a budget
budget = await svc.service_charge_budget(
    property_id="prop-001",
    year=2026,
    budget_items=[
        {"category": "insurance", "amount": 480000},
        {"category": "utilities", "amount": 960000},
        {"category": "management_fee", "amount": 240000},
    ],
    tenant_id="t1",
)

# Then produce the variance report after the period
report = await svc.budget_variance_report(
    property_id="prop-001",
    year=2026,
    tenant_id="t1",
    tolerance_pct=Decimal("0.10"),
)
for item in report["line_items"]:
    if item["exceeds_tolerance"]:
        print(f"OVER TOLERANCE: {item['category']} {item['variance_pct']:.1f}%")
```

### Property Acquisition and Depreciation

```python
acq = await svc.property_acquisition_cost(
    property_id="prop-001",
    purchase_price=Decimal("50000000"),
    transaction_costs={"stamp_duty": 250000, "legal_fees": 150000, "valuation_fee": 50000},
    tenant_id="t1",
    acquisition_date=date(2024, 1, 15),
    funded_by="mortgage",
)

dep = await svc.depreciation_charge(
    property_id="prop-001",
    method="straight_line",
    period="2026-06",
    tenant_id="t1",
    useful_life_years=50,
    residual_value=Decimal("5000000"),
)
print(dep["monthly_depreciation"])
```

### Investment Property Revaluation (IAS 40)

```python
rev = await svc.revaluation_gain_loss(
    property_id="prop-001",
    old_value=Decimal("50000000"),
    new_value=Decimal("55000000"),
    tenant_id="t1",
    effective_date=date(2026, 6, 30),
    valuation_reference="CBRE-2026-Q2",
    measurement_model="fair_value",
)
print(rev["is_gain"], rev["change"])

disclosure = await svc.ifrs_investment_property(
    property_id="prop-001",
    period="2026-06",
    measurement_model="fair_value",
    tenant_id="t1",
    fair_value=Decimal("55000000"),
)
```

### Analytics

```python
summary = await svc.real_estate_analytics(period="2026-06", tenant_id="t1")
print(summary["service_charge_income"])
print(summary["net_revaluation"])
print(summary["total_lease_liability_ifrs16"])
```

---

## Configuration

All keys are tenant-scoped. Set via the `conf` capability or `REALESTATE_ACC_*` env vars.

| Key | Default | Description |
|-----|---------|-------------|
| `journals.approval_required_above_amount` | 50000 | KES threshold for mandatory approval |
| `service_charges.cam_methods` | `pro_rata` | Supported CAM allocation bases |
| `ifrs16.discount_rate_required` | `true` | Mandate discount rate on all schedules |
| `governance.dual_control_for_period_close` | `true` | Two distinct approvers for period close |
| `budgets.variance_tolerance_pct` | `0.10` | Default tolerance for variance reporting |

---

## Interoperability

```apg
use realestate_acc;
```

| Consumed capability | Purpose |
|--------------------|---------|
| `realestate_lea` | Lease data for IFRS 16 schedules and revenue recognition |
| `realestate_ren` | Turnover figures for percentage-rent recognition |
| `realestate_prm` | Property-level cost data for CAM reconciliation |
| `auth` | Actor identity for approvals and audit events |
| `audl` | Immutable audit event stream |
| `ntfy` | Notifications on period close, CAM settlement, dispute raised |
| `wflo` | Approval workflows for journals, service charges, and CAM |
| `schd` | Auto-generation of recurring and reversing journals |

---

## Further Reading

- `service.py` — Business logic implementation (all async methods)
- `models.py` — Pydantic v2 data models
- `api.py` — REST API endpoints (Flask-AppBuilder blueprint)
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
- `WORLD_CLASS_IMPROVEMENTS.md` — Roadmap of 15 planned enhancements
- `cap_spec.md` — Full capability specification
- `SPECIFICATION.md` — Technical specification
