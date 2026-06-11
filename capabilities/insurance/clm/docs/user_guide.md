# Claims Management (ins_clm) — User Guide

© 2025 Datacraft | Author: Nyimbi Odero

## Overview

The Claims Management capability delivers a complete claims lifecycle from First Notification of Loss
through payment, fraud control, litigation, subrogation, and regulatory reporting. All service methods
are async and accept `tenant_id` as the first argument for full multi-tenancy isolation.

---

## Core Concepts

| Concept | Description |
|---------|-------------|
| **FNOL** | First Notification of Loss — the entry point for every claim |
| **Reserve** | Money set aside to cover expected claim payout (OCR, IBNR, ALAE, ULAE) |
| **STP** | Straight-Through Processing — auto-approve simple claims without human touch |
| **Complexity Tier** | simple / standard / complex / catastrophic — drives adjuster routing |
| **Fraud Score** | 0.0–1.0 continuous risk score; ≥ 0.75 auto-flags the claim |
| **Velocity Risk** | Rolling-window anomaly detection per policy / claimant |
| **Subrogation** | Recovery of claim costs from liable third parties |
| **Litigation Matter** | Court-managed recovery or defence, tracked with full event log |
| **SLA** | Configurable acknowledge / assess / settle day limits with breach detection |

---

## Claim Status Lifecycle

```
fnol → under_assessment → reserved → approved → partially_paid → fully_paid
                                              ↘ repudiated
                                              ↘ withdrawn (from fnol only)
                                              ↘ subrogation (post-paid)
```

---

## Use Cases

### 1. FNOL Registration

```python
claim = await svc.register_fnol(
    tenant_id="acme_insurance",
    policy_id="pol-001",
    policy_number="POL-2026-001",
    claimant_name="Jane Smith",
    claimant_id="ID-99999",
    incident_date="2026-05-20",
    incident_description="Vehicle rear-ended at intersection",
    estimated_loss=Decimal("350000"),
    reported_by="agent_002",
    currency="KES",
)
```

### 2. Claim Velocity Check (pre-FNOL fraud gate)

```python
velocity = await svc.check_claim_velocity(
    tenant_id="acme_insurance",
    policy_id="pol-001",
    claimant_id="ID-99999",
    window_days=30,
)
# Returns: velocity_risk_level: low | medium | high
```

### 3. Complexity Triage

```python
complexity = await svc.score_claim_complexity(
    tenant_id="acme_insurance",
    claim_id=claim["id"],
    injury_involved=True,
    commercial_vehicle=False,
    catastrophe_code=None,
)
# Returns: complexity_tier: simple | standard | complex | catastrophic
```

### 4. Straight-Through Processing

```python
stp = await svc.evaluate_stp_eligibility(
    tenant_id="acme_insurance",
    claim_id=claim["id"],
    stp_loss_ceiling=Decimal("50000"),
    lookback_days=90,
)
if stp["eligible"]:
    # Claim is auto-approved — no further manual steps needed
    print("Settled:", stp["auto_approved_amount"])
else:
    print("Manual review required:", stp["reasons"])
```

### 5. Assessor Assignment & Report

```python
assessment = await svc.assign_assessor(
    tenant_id="acme_insurance",
    claim_id=claim["id"],
    assessor_id="assessor_007",
    assigned_by="claims_manager",
)

report = await svc.submit_assessment_report(
    tenant_id="acme_insurance",
    claim_id=claim["id"],
    assessed_loss=Decimal("320000"),
    recommendation="approve",
    findings="Third-party liability confirmed. Repair estimate validated.",
    assessor_id="assessor_007",
)
```

### 6. Reserve Management

```python
reserve = await svc.set_reserve(
    tenant_id="acme_insurance",
    claim_id=claim["id"],
    reserve_amount=Decimal("320000"),
    reserve_type="outstanding",
    set_by="claims_manager",
    justification="Post-assessment outstanding reserve",
)

adequacy = await svc.check_reserve_adequacy(
    tenant_id="acme_insurance",
    claim_id=claim["id"],
)
# Returns: adequacy_status: adequate | warning | critical
# Auto-emits reserve_adequacy_warning when utilisation >= 0.85
```

### 7. Excess / Deductible Computation

```python
excess_result = await svc.compute_applicable_excess(
    tenant_id="acme_insurance",
    claim_id=claim["id"],
    excess_schedule=[
        {"type": "basic", "amount": "15000", "applies_when": "always"},
        {"type": "voluntary", "amount": "10000", "applies_when": "voluntary"},
        {"type": "young_driver", "amount": "20000", "applies_when": "young_driver"},
    ],
)
# Returns: net_payable after stacking all applicable excesses
```

### 8. Payment Processing

```python
payment = await svc.process_payment(
    tenant_id="acme_insurance",
    claim_id=claim["id"],
    payment_amount=Decimal("295000"),
    payment_type="full",
    payee_name="Jane Smith",
    payee_account="KE123456789",
    payment_reference="PAY-2026-001",
    authorised_by="finance_director",
)
```

### 9. Fraud Detection & Flagging

```python
# Score-based assessment
fraud = await svc.assess_fraud_risk(
    tenant_id="acme_insurance",
    claim_id=claim["id"],
    fraud_score=0.82,
    indicators=["multiple_claims_same_quarter", "payee_account_repeated"],
    assessed_by="fraud_analyst",
    recommendation="refer_to_siu",
)

# Manual flag
await svc.flag_claim_fraud(
    tenant_id="acme_insurance",
    claim_id=claim["id"],
    reason="SIU confirmed staged accident",
    flagged_by="siu_investigator",
)
```

### 10. Claim Approval & Repudiation

```python
# Approve
await svc.approve_claim(
    tenant_id="acme_insurance",
    claim_id=claim["id"],
    approved_amount=Decimal("295000"),
    approved_by="underwriting_manager",
)

# Repudiate
await svc.repudiate_claim(
    tenant_id="acme_insurance",
    claim_id=claim["id"],
    reason="Policy condition breach — driver unlicensed",
    authorised_by="claims_director",
)
```

### 11. Subrogation

```python
sub = await svc.initiate_subrogation(
    tenant_id="acme_insurance",
    claim_id=claim["id"],
    third_party_name="ABC Trucking Ltd",
    third_party_id="KRA-BUS-123456",
    recovery_amount=Decimal("295000"),
    legal_reference="NAIROBI/CIVIL/2026/001",
)

# Record partial recovery
await svc.record_subrogation_recovery(
    tenant_id="acme_insurance",
    subrogation_id=sub["id"],
    recovered_amount=Decimal("150000"),
)
```

### 12. Litigation Management

```python
matter = await svc.open_litigation_matter(
    tenant_id="acme_insurance",
    claim_id=claim["id"],
    law_firm_id="firm_kariuki_and_co",
    case_reference="NAIROBI/HCCC/2026/123",
    court="Nairobi High Court (Commercial Division)",
    first_hearing_date="2026-08-15",
    litigation_reserve_uplift=Decimal("50000"),
    opened_by="legal_manager",
)

await svc.log_litigation_event(
    tenant_id="acme_insurance",
    litigation_id=matter["id"],
    event_type="hearing",
    description="Preliminary hearing — directions issued",
    legal_cost=Decimal("45000"),
    new_phase="discovery",
)
```

### 13. Multi-Currency FX Conversion

```python
fx = await svc.convert_claim_currency(
    tenant_id="acme_insurance",
    claim_id=claim["id"],
    target_currency="USD",
    fx_rate=Decimal("130.25"),
    fx_source="CBK_MIDRATE_2026-05-20",
)
# FX rate and all converted values are immutable once recorded
```

### 14. Regulatory Large-Loss Notifications

```python
notifications = await svc.generate_large_loss_notifications(
    tenant_id="acme_insurance",
    threshold=Decimal("1000000"),
    lookback_hours=24,
)
# Returns IRA Kenya C-4 compatible notification records
```

### 15. SLA Compliance Dashboard

```python
sla = await svc.sla_compliance_dashboard(
    tenant_id="acme_insurance",
    acknowledge_hours=72,
    assess_days=14,
    settle_days=90,
)
# Returns: { compliant, warning, breached, by_claim: [...], total_open }
# Emits sla_breach_detected for each breached claim
```

### 16. Portfolio Analytics

```python
# Claims summary
summary = await svc.claims_summary(tenant_id="acme_insurance")

# Loss ratio
loss_ratio = await svc.loss_ratio_report(
    tenant_id="acme_insurance",
    earned_premium=Decimal("50000000"),
)
```

---

## Fraud Thresholds

| Score Range | Outcome |
|-------------|---------|
| 0.00 – 0.74 | No auto-flag; assessor may review manually |
| 0.75 – 1.00 | `fraud_flag=True` automatically set on claim |

Manual flagging via `flag_claim_fraud()` bypasses the score threshold.

---

## Velocity Risk Levels

| Policy claims in window | Claimant claims in window | Risk Level |
|------------------------|--------------------------|------------|
| 0–2 | 0–1 | low |
| 3–4 | 2–3 | medium |
| 5+ | 4+ | high — emits `velocity_alert` |

---

## Reserve Types

| Type | Description |
|------|-------------|
| `outstanding` | Case reserve for known, open claims |
| `ibnr` | Incurred But Not Reported reserves |
| `allocated_loss_adjustment` | Defence and cost containment per claim |
| `unallocated_loss_adjustment` | Overhead loss adjustment expenses |

---

## Payment Types

| Type | Description |
|------|-------------|
| `partial` | Interim payment on open claim |
| `full` | Final settlement payment |
| `advance` | Pre-settlement advance |
| `ex_gratia` | Goodwill payment without admission |
| `recoverable_advance` | Advance subject to repayment |

---

## Supported Currencies

Any ISO 4217 currency code is accepted. Default is `KES`. FX conversions recorded via
`convert_claim_currency()` are immutable for audit compliance.

---

## Error Reference

| Error | Cause |
|-------|-------|
| `tenant_context_required` | `tenant_id` empty or None |
| `claim_not_found:{id}` | Claim does not exist or belongs to different tenant |
| `only_fnol_claims_can_be_withdrawn` | Withdrawal attempted on non-FNOL claim |
| `claim_must_be_reserved_or_approved_for_payment` | Payment before reserve set |
| `payment_exceeds_outstanding_reserve` | Payment amount > reserve - paid |
| `subrogation_requires_paid_claim` | Subrogation on unpaid claim |
| `excess_computation_requires_reserved_or_approved_claim` | Excess on wrong status |
| `fx_rate_must_be_positive` | FX rate <= 0 |
| `source_and_target_currency_identical` | FX conversion to same currency |
