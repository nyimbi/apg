# Rental Operations — User Guide

**Capability ID**: `realestate_ren` | **Domain**: `realestate` | **Version**: `1.1.0`

## Description

End-to-end tenancy lifecycle: application, referencing, right-to-rent checks, deposit
registration and accounting, rent collection with shortfall detection, arrears management
and legal escalation, notice serving, renewal pipeline management, void tracking, inspection
workflow, and rent review. Produces a live rent roll with versioned snapshots for any property.

## Installation

```bash
pip install apg-realestate-ren
```

## Quick Start

```python
from apg_realestate_ren.service import RenService
from datetime import date
from decimal import Decimal

svc = RenService(tenant_id="acme", actor_id="admin")

# 1. Advertise a vacant unit
listing = await svc.advertise_unit(
    unit_id="unit-101",
    rent=Decimal("45000"),
    available_from=date.today(),
    listing_description="2-bed apartment, Westlands",
    tenant_id="acme",
    listing_channels=["portal", "social"],
)

# 2. Receive a tenant application
application = await svc.tenant_application(
    unit_id="unit-101",
    applicant_id="applicant-001",
    employment_details={"employer": "Safaricom", "annual_income": 1_800_000},
    guarantor=None,
    tenant_id="acme",
    move_in_date=date(2026, 7, 1),
)

# 3. Run referencing + credit check
ref = await svc.reference_check(
    application_id=application["id"],
    reference_type="employer",
    tenant_id="acme",
    outcome="satisfactory",
)
credit = await svc.credit_check_tenant(
    application_id=application["id"],
    tenant_id="acme",
    credit_score=720,
    provider="experian",
)

# 4. Sign tenancy (creates + activates + registers deposit in one call)
tenancy = await svc.sign_tenancy(
    unit_id="unit-101",
    tenant_entity_id="applicant-001",
    rent=Decimal("45000"),
    deposit=Decimal("90000"),
    start_date=date(2026, 7, 1),
    tenant_id="acme",
)

# 5. Record move-in inspection
inspection = await svc.record_inspection(
    tenancy_id=tenancy["id"],
    tenant_id="acme",
    inspection_type="move_in",
    condition_items=[
        {"room": "living_room", "item": "carpet", "condition": "new", "grade": 5},
        {"room": "kitchen", "item": "oven", "condition": "good", "grade": 4},
    ],
    inspector_id="inspector-001",
    photos=["doc-photo-001", "doc-photo-002"],
)

# 6. Collect monthly rent
payment = await svc.collect_rent(
    unit_id="unit-101",
    period="2026-07",
    amount=Decimal("45000"),
    payment_method="bank_transfer",
    tenant_id="acme",
    tenancy_id=tenancy["id"],
)

# 7. Generate formal rent receipt
receipt = await svc.generate_rent_receipt(
    payment_id=payment["id"],
    tenant_id="acme",
)
# receipt["receipt_number"] => "REC-2026-0001"
```

## Tenancy Lifecycle

### States

```
application -> referencing -> approved -> notice_signed -> active
    -> notice_served -> holding_over -> vacating -> vacated
    -> dispute
```

### Activation Pre-conditions

All three must be `True` before `activate_tenancy()` succeeds:
1. `deposit_registered`
2. `referencing_complete`
3. `right_to_rent_checked` (residential tenancies)

Use `sign_tenancy()` to bypass the multi-step flow in test or simple scenarios — it
creates and activates the tenancy and registers the deposit atomically.

## Rent Collection

### Standard Flow

```python
# Short payment auto-creates arrears
payment = await svc.collect_rent(
    unit_id="unit-101", period="2026-08",
    amount=Decimal("30000"),           # 15,000 short
    payment_method="mpesa", tenant_id="acme",
)
# payment["is_short_payment"] == True
# payment["shortfall"] == "15000"
```

### Arrears Chase Automation

```python
# Schedule automated multi-step chase
chase = await svc.schedule_arrears_chase(
    arrears_id=arrears["id"],
    tenant_id="acme",
    chase_sequence=[
        {"days_after": 3, "method": "email", "template": "arrears_reminder_1"},
        {"days_after": 7, "method": "sms", "template": "arrears_reminder_2"},
        {"days_after": 14, "method": "letter", "template": "formal_demand"},
    ],
)
```

### Arrears Escalation to Legal

Legal action is gate-checked: `days_overdue >= 90` and `amount_overdue > 0` required.

```python
await svc.escalate_arrears_to_legal(arrears_id=arrears["id"], tenant_id="acme")
```

## Deposit Accounting

```python
# Register deposit
deposit = await svc.register_deposit(DepositCreate(
    tenant_id="acme", tenancy_id=tenancy.id,
    deposit_type=DepositType.cash_deposit,
    amount=Decimal("90000"), created_by="admin",
))

# Deduct at end of tenancy (evidence required)
deduction = await svc.deduct_from_deposit(DepositDeductionCreate(
    tenant_id="acme", deposit_id=deposit.id,
    reason="Carpet replacement — move-out inspection ref INS-001",
    amount=Decimal("15000"),
    evidence_document_ids=["doc-001", "doc-002"],
    created_by="admin",
))

# Release remainder
await svc.release_deposit(deposit.id, tenant_id="acme", released_by="admin")
```

## Rent Review

Statutory notice periods are enforced automatically based on `rent_frequency`:

| Frequency | Minimum notice |
|-----------|---------------|
| weekly | 7 days |
| monthly | 28 days |
| quarterly | 84 days |
| annual | 365 days |

```python
proposal = await svc.propose_rent_increase(
    tenancy_id=tenancy["id"],
    tenant_id="acme",
    new_rent=Decimal("48000"),
    effective_date=date(2027, 1, 1),
    proposed_by="admin",
    reason="Annual CPI review",
)

# After effective_date passes:
await svc.apply_rent_increase(
    proposal_id=proposal["id"],
    tenant_id="acme",
    applied_by="admin",
)
```

## Inspections

Condition grades: `1` (damaged) to `5` (new/perfect). Grade < 3 flags for remediation
and marks `deposit_deduction_eligible = True` on move-out inspections.

```python
inspection = await svc.record_inspection(
    tenancy_id=tenancy["id"], tenant_id="acme",
    inspection_type="move_out",
    condition_items=[
        {"room": "bedroom", "item": "wall_paint", "condition": "marked", "grade": 2,
         "notes": "Scuffs and marks, repainting required"},
        {"room": "kitchen", "item": "oven", "condition": "clean", "grade": 4},
    ],
    inspector_id="inspector-001",
    photos=["photo-checkout-001"],
)
# inspection["deposit_deduction_eligible"] == True
# inspection["items_requiring_remediation"] == 1
```

## Void Tracking

```python
void = await svc.record_void_period(
    unit_id="unit-101", tenant_id="acme",
    start_date=date(2026, 9, 1), end_date=date(2026, 9, 20),
    reason="between_tenancies",
)

report = await svc.get_void_report(
    tenant_id="acme",
    period_start=date(2026, 1, 1), period_end=date(2026, 12, 31),
)
# report["void_rate_pct"], report["by_reason"]
```

## Rent Roll Snapshots

```python
# Snapshot at month-end
snap_june = await svc.snapshot_rent_roll(
    tenant_id="acme", label="2026-06-30-month-end"
)
snap_july = await svc.snapshot_rent_roll(
    tenant_id="acme", label="2026-07-31-month-end"
)

# Compare for auditor
diff = await svc.compare_rent_rolls(
    snapshot_id_a=snap_june["id"],
    snapshot_id_b=snap_july["id"],
    tenant_id="acme",
)
# diff["added"], diff["removed"], diff["changed"], diff["gross_rent_delta"]
```

## Tenancy Statement

```python
statement = await svc.get_tenancy_statement(
    tenancy_id=tenancy["id"], tenant_id="acme",
    from_date=date(2026, 7, 1), to_date=date(2026, 9, 30),
)
# statement["closing_balance"], statement["days_in_arrears"], statement["ledger"]
```

## Compliance Check

```python
result = await svc.run_compliance_check(
    tenancy_id=tenancy["id"], tenant_id="acme", jurisdiction="KE"
)
# result["overall_status"]  => "pass" | "fail"
# result["items"]           => per-item verdicts with remediation guidance
```

Supported jurisdictions: `KE` (Kenya), `GB` (United Kingdom), `ZA` (South Africa),
`NG` (Nigeria), `GH` (Ghana).  GB adds EPC, gas safety cert, and EICR checks.

## Notices

```python
# Serve formal Section 21 (GB) with computed expiry
notice = await svc.serve_notice_formal(
    unit_id="unit-101", notice_type="section_21",
    notice_date=date.today(), reason="Landlord requires possession",
    tenant_id="acme",
)
# notice["expiry_date"] auto-computed as notice_date + 56 days
```

Default notice periods:

| Type | Days |
|------|------|
| section_21 | 56 |
| section_8 | 14 |
| notice_to_quit | 28 |
| break_notice | 90 |
| section_25/26 | 180 |

## Renewal Pipeline

```python
# Tenancies expiring within 3 months
pipeline = await svc.get_renewal_pipeline(tenant_id="acme", months_ahead=3)
# [{"tenancy_id": ..., "days_remaining": 45, ...}, ...]

# Initiate renewal offer
renewal = await svc.initiate_renewal(TenancyRenewalCreate(
    tenant_id="acme", tenancy_id=tenancy.id,
    renewal_type="fixed_term",
    new_start_date=date(2027, 7, 1), new_end_date=date(2028, 6, 30),
    new_rent=Decimal("48000"), created_by="admin",
))

await svc.accept_renewal(renewal.id, tenant_id="acme")
```

## Analytics

```python
kpis = await svc.rental_analytics(period="2026-07", tenant_id="acme")
# kpis["active_tenancies"], kpis["rent_collected"],
# kpis["total_arrears"], kpis["renewals_due_3_months"]
```

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/realestate/ren/dashboard` | `realestate_ren:view` | Overview |
| `/realestate/ren/tenancies` | `realestate_ren:tenancies` | Tenancies |
| `/realestate/ren/tenancies/<id>` | `realestate_ren:tenancies` | Tenancies |
| `/realestate/ren/tenancies/<id>/statement` | `realestate_ren:tenancies` | Tenancies |
| `/realestate/ren/referencing` | `realestate_ren:referencing` | Onboarding |
| `/realestate/ren/inspections` | `realestate_ren:inspections` | Onboarding |
| `/realestate/ren/rent-collection` | `realestate_ren:rent_collection` | Collections |
| `/realestate/ren/arrears` | `realestate_ren:arrears` | Collections |
| `/realestate/ren/deposits` | `realestate_ren:deposits` | Financial |
| `/realestate/ren/renewals` | `realestate_ren:renewals` | Planning |
| `/realestate/ren/voids` | `realestate_ren:voids` | Planning |
| `/realestate/ren/rent-roll` | `realestate_ren:rent_roll` | Reports |
| `/realestate/ren/compliance/<id>` | `realestate_ren:compliance` | Compliance |

## Configuration

All keys are tenant-scoped. Set via `conf` capability or `REALESTATE_REN_*` env vars.

| Key | Default | Description |
|-----|---------|-------------|
| `arrears.legal_threshold_days` | 90 | Days before legal escalation allowed |
| `deposits.registration_required` | true | Mandate deposit scheme registration |
| `renewals.early_warning_days` | 90 | Days before expiry to trigger renewal reminder |
| `rent_review.min_notice_days_override` | null | Override statutory minimum notice |
| `compliance.jurisdiction` | KE | Default jurisdiction for compliance checks |

## Interoperability

```apg
use realestate_ren;
```

| Capability | Integration |
|-----------|-------------|
| `realestate_lea` | Consumes lease terms for expected rent amounts |
| `realestate_acc` | Posts rent receipts as journal entries |
| `realestate_prm` | Unit availability feedback on vacate/void |
| `realestate_ten` | Tenant portal communication |
| `ntfy` | Overdue rent, renewal due, notice served alerts |
| `schd` | Arrears chase and renewal reminder scheduling |
| `mqeb` | CloudEvents bus for all rental state transitions |
| `wflo` | Legal escalation approval workflow |

## Further Reading

- `service.py` — Full business logic with docstrings
- `models.py` — Pydantic v2 data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and schemas
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 improvement proposals
- `SPECIFICATION.md` — Full capability specification
- `README.md` — Quick-reference capability contract
