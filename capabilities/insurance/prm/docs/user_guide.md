# Premium & Billing (ins_prm) — User Guide

**Version**: 2.0 | **Domain**: Insurance | **Copyright**: © 2025 Datacraft

---

## Overview

`ins_prm` manages all premium financial flows within the APG insurance platform:
schedule creation, instalment tracking, partial and full payment collection,
payment bounce handling, automated dunning, IFRS 17 accruals, statutory levy
calculation, lapse state management, and tamper-evident audit export.

---

## Supported Payment Frequencies

| Frequency | Instalments | Notes |
|---|---|---|
| `annual` | 1 | Full premium at inception |
| `semi_annual` | 2 | 6-month intervals |
| `quarterly` | 4 | ~91-day intervals |
| `monthly` | 12 | ~30-day intervals |

---

## Quick Start

```python
from capabilities.insurance.prm.service import PremiumBillingService
from decimal import Decimal

svc = PremiumBillingService(tenant_id="acme_insurance")

# 1. Create a quarterly schedule for KES 45 000
schedule = await svc.create_schedule(
    tenant_id="acme_insurance",
    policy_id="pol-001",
    policy_number="POL-2026-001",
    total_premium=Decimal("45000"),
    frequency="quarterly",
    inception_date="2026-01-01",
    expiry_date="2026-12-31",
)

# 2. List the generated instalments
instalments = await svc.list_instalments("acme_insurance", schedule["id"])
# → 4 records of KES 11 250 each, due 2026-01-01, 2026-04-02, 2026-07-02, 2026-10-01

# 3. Collect the first instalment via M-Pesa
collection = await svc.collect_payment(
    tenant_id="acme_insurance",
    instalment_id=instalments[0]["id"],
    payment_method="mpesa",
    payment_reference="QW123456",
    amount=Decimal("11250"),
    collected_by="cashier_01",
)
```

---

## Workflow Reference

### A. Premium Schedule Lifecycle

```
create_schedule()
       │
       ▼
  [status: active]
       │
   instalments generated
       │
   collect_payment() / record_partial_payment()
       │
       ▼
  [status: fully_paid]   ← all instalments settled

  OR

  evaluate_lapse_status()
       │
  pending → overdue → in_grace → lapsed
```

---

### B. Creating a Premium Schedule

```python
schedule = await svc.create_schedule(
    tenant_id="acme_insurance",
    policy_id="pol-001",
    policy_number="POL-2026-001",
    total_premium=Decimal("120000"),
    frequency="monthly",
    inception_date="2026-01-01",
    expiry_date="2026-12-31",
    currency="KES",
)
```

The service automatically:
- Divides total_premium into equal instalments
- Absorbs rounding in the final instalment
- Sets `status="active"` on the schedule
- Emits `premium_schedule_created` audit event

---

### C. Collecting Full Premium

```python
col = await svc.collect_payment(
    tenant_id="acme_insurance",
    instalment_id="inst-abc",
    payment_method="mpesa",           # or bank_transfer, card, cash, cheque, direct_debit, bancassurance
    payment_reference="ABC123XYZ",
    amount=Decimal("10000"),
    collected_by="cashier_01",
)
```

On success:
- Instalment `status → paid`
- Schedule `collected_amount` incremented, `outstanding_amount` decremented
- Schedule `status → fully_paid` if no outstanding amount remains
- KPI accumulator updated (O(1) dashboard reads)

---

### D. Partial Payments (M-Pesa Common Pattern)

```python
# Pay KES 6 000 against a KES 10 000 instalment
partial = await svc.record_partial_payment(
    tenant_id="acme_insurance",
    instalment_id="inst-abc",
    payment_method="mpesa",
    payment_reference="MPESA001",
    amount=Decimal("6000"),
    collected_by="agent_007",
)
# instalment status → "partial", paid_so_far = 6000

# Pay the remaining KES 5 000 (KES 1 000 overpayment)
partial2 = await svc.record_partial_payment(
    tenant_id="acme_insurance",
    instalment_id="inst-abc",
    payment_method="mpesa",
    payment_reference="MPESA002",
    amount=Decimal("5000"),
    collected_by="agent_007",
)
# instalment status → "paid"
# KES 1 000 overpayment auto-credited to next pending instalment
```

---

### E. Payment Bounce Handling

```python
bounce = await svc.record_payment_bounce(
    tenant_id="acme_insurance",
    collection_id="col-xyz",
    bounce_reason="MPESA_REVERSAL",
    bounce_fee=Decimal("500"),      # configurable per insurer policy
)
# collection status → "bounced"
# instalment status → "pending" (re-opens for dunning)
# BounceCharge record created for KES 500
# payment_bounced audit event emitted
```

**Note**: The bounce fee is levied as a separate `prm_bounce_charge` record. It does not
automatically deduct from the next payment; it must be collected separately or waived
by an authorised user.

---

### F. Lapse State Machine

IRA Kenya requires explicit grace-period tracking. Use `evaluate_lapse_status()` as part
of a nightly batch:

```python
lapse_result = await svc.evaluate_lapse_status(
    tenant_id="acme_insurance",
    schedule_id="sch-001",
    grace_period_days=30,       # IRA standard; configurable per product
)
# Returns: {"transitions": [{"instalment_id": ..., "from": "pending", "to": "in_grace"}]}
# Emits: lapse_warning or policy_lapsed audit events
```

State transitions:
- `pending` → `overdue` (1–7 days overdue)
- `overdue` → `in_grace` (8–30 days overdue)
- `in_grace` → `lapsed` (>30 days overdue)

When any instalment lapses, the parent schedule `status → lapsed`.

---

### G. Statutory Levy Calculation (IRA Kenya)

```python
levies = await svc.compute_statutory_levies(
    tenant_id="acme_insurance",
    gross_premium=Decimal("100000"),
    effective_date="2026-01-01",
)
# Returns:
# {
#   "gross_premium": "100000",
#   "levies": [
#     {"code": "IRA_TRAINING_LEVY", "rate": "0.002", "amount": "200.00", ...},
#     {"code": "PHCF",              "rate": "0.0025","amount": "250.00", ...},
#     {"code": "STAMP_DUTY",        "rate": "0.001", "amount": "100.00", ...},
#   ],
#   "total_levies": "550.00",
#   "net_premium": "99450.00"
# }
```

Custom levy tables can be passed via `levy_overrides` to handle product-specific or
future gazette changes without code modification.

---

### H. IFRS 17 Earned Premium (PAA Method)

Required for every IFRS-reporting insurer at each month-end close.

```python
accrual = await svc.compute_earned_premium(
    tenant_id="acme_insurance",
    schedule_id="sch-001",
    reporting_date="2026-06-30",
)
# {
#   "written_premium":          "45000.00",
#   "earned_premium":           "22397.26",   # 181/365 days
#   "unearned_premium_reserve": "22602.74",
# }
```

Post `earned_premium` as **DR Premium Receivable / CR Premium Income** and
`unearned_premium_reserve` as **DR Premium Income / CR UPR** in the finance ledger.

---

### I. Dunning Workflow

Run as a nightly batch to advance overdue instalments through the escalation ladder:

```python
result = await svc.run_dunning_cycle(
    tenant_id="acme_insurance",
    grace_period_days=7,
)
print(result["advanced_count"])   # number of instalments that changed dunning level

for action in result["dispatches"]:
    # Feed into notification dispatcher (ntf_sms / ntf_email / agent_task)
    print(action["dunning_level"], action["action_required"], action["policy_id"])
```

Dunning thresholds:

| Level | Days Overdue | Action |
|---|---|---|
| `REMINDER_1` | 7 | Courtesy SMS reminder |
| `REMINDER_2` | 14 | Email reminder with payment link |
| `FORMAL_NOTICE` | 21 | Formal written notice (registered post) |
| `LAPSE_WARNING` | 30 | Agent dispatch + lapse evaluation |

Each dunning action is idempotent per level — an instalment will not advance the same
level twice until it regresses (e.g., partial payment resets `dunning_level`).

---

### J. Lapse Risk Scoring (AI/ML)

Score a schedule 30 days before a due date to prioritise retention outreach:

```python
score = await svc.score_lapse_risk(
    tenant_id="acme_insurance",
    schedule_id="sch-001",
)
# {
#   "score": 0.72,
#   "band": "high",
#   "features": {
#     "avg_days_overdue": 12.5,
#     "overdue_ratio": 0.5,
#     "partial_pay_frequency": 0.25,
#     "payment_method_volatility": 0.667,
#   }
# }
```

Route `band == "high"` schedules to proactive retention agents or SMS payment-plan offers.

---

### K. Real-Time Collection KPIs

```python
kpis = await svc.get_collection_kpis(tenant_id="acme_insurance")
# {
#   "collection_ratio": 0.683,
#   "total_billed":    "4500000.00",
#   "total_collected": "3075000.00",
#   "overdue_aging_buckets": {"0_30": 45, "31_60": 12, "61_90": 5, "90_plus": 2},
#   "channel_mix": {"mpesa": 0.72, "bank_transfer": 0.18, "cash": 0.10},
# }
```

KPIs are maintained incrementally — O(1) reads suitable for real-time dashboards.

---

### L. Audit Chain Export

```python
chain = await svc.export_audit_chain(tenant_id="acme_insurance")
for event in chain:
    print(event["chain_hash"], event["event_type"])
# Each event includes prev_hash + chain_hash for tamper-evidence verification
# Verify integrity: sha256(prev_hash + json(event fields)) == chain_hash
```

Submit the chain export directly to IRA Kenya or FSCA for regulatory inspection.
Each event is enriched with `chain_hash` and `prev_hash` — verifiable without a
blockchain node.

---

## Premium Calculation Reference

```python
result = await svc.calculate_premium(
    tenant_id="acme_insurance",
    product_code="MOTOR_COMP",
    sum_insured=Decimal("2500000"),
    base_rate=Decimal("0.035"),
    loadings={
        "fleet_loading": Decimal("0.005"),
        "young_driver":  Decimal("0.003"),
    },
    discounts={
        "no_claims_bonus": Decimal("0.004"),
    },
)
# net_rate = 0.035 + 0.008 - 0.004 = 0.039
# gross_premium = 2 500 000 × 0.039 = 97 500.00
```

---

## Refund Processing

```python
refund = await svc.process_refund(
    tenant_id="acme_insurance",
    policy_id="pol-001",
    refund_amount=Decimal("5000"),
    reason="policy_cancelled_mid_term",
    payee_account="1234567890",
    authorised_by="finance_mgr_01",
)
# Guards: refund_amount must not exceed (collected - previously_refunded)
```

---

## Period Reconciliation

```python
recon = await svc.reconcile_period(
    tenant_id="acme_insurance",
    period_start="2026-05-01",
    period_end="2026-05-31",
    reconciled_by="finance_team",
)
# Returns: total_collected, total_refunded, net_premium, collection_count, refund_count
```

---

## Debit Order Management

```python
# Setup
do = await svc.setup_debit_order(
    tenant_id="acme_insurance",
    policy_id="pol-001",
    schedule_id="sch-001",
    bank_account="0011223344",
    bank_code="KCB",
    collection_day=5,           # must be 1–28
    authorised_by="policyholder_signature",
)

# Cancel
await svc.cancel_debit_order(
    tenant_id="acme_insurance",
    debit_order_id=do["id"],
    reason="customer_request",
)
```

---

## Integration with Other APG Capabilities

| Capability | How it connects |
|---|---|
| `ins_pol` (Policy Admin) | Subscribe to `policy_lapsed` events; suspend coverage on lapse |
| `ins_clm` (Claims) | Check `outstanding_amount == 0` before authorising claim payment |
| `ins_uwt` (Underwriting) | Use `score_lapse_risk()` output in renewal underwriting decisions |
| `fin_led` (Finance Ledger) | Post `compute_earned_premium()` output as IFRS 17 journal entries |
| `ntf_sms` / `ntf_email` | Dispatch dunning actions returned by `run_dunning_cycle()` |
| `rpt_reg` (Regulatory Reporting) | Submit `export_audit_chain()` for IRA / FSCA filings |

---

## Error Reference

| Exception | Cause |
|---|---|
| `PermissionError: tenant_context_required` | `tenant_id` empty or missing |
| `KeyError: schedule_not_found:{id}` | Wrong tenant or non-existent schedule |
| `KeyError: instalment_not_found:{id}` | Wrong tenant or non-existent instalment |
| `PermissionError: instalment_already_paid` | Attempting to collect against a paid instalment |
| `PermissionError: collection_already_bounced` | Bounce already recorded for this collection |
| `ValueError: unsupported_frequency:{f}` | Use one of: annual, semi_annual, quarterly, monthly |
| `ValueError: unsupported_payment_method:{m}` | Use one of: mpesa, bank_transfer, card, cash, cheque, direct_debit, bancassurance |
| `ValueError: amount_must_be_positive` | Payment amount must be > 0 |
| `ValueError: refund_exceeds_collected_premium` | Refund amount > (collected - previously refunded) |
| `PermissionError: cannot_delete_schedule_with_collections` | Cannot delete a schedule that has received payments |

---

## Coding Standards Compliance

- `from __future__ import annotations` — all deferred type evaluation
- All monetary values are `Decimal` — never `float`
- All methods are `async def`
- Tabs, not spaces
- `guard_tenant_id()` / `guard_non_empty_string()` at every public entry point
- `_log = logging.getLogger(__name__)` — structured logging throughout
- No bare `except:` — all exceptions are typed
- Audit events emitted for every state-changing operation
