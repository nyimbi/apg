# Land Registry — User Guide

## Overview

The Land Registry capability (`gov_lnd`) provides a complete digital cadastre and title
management system. It covers the full land administration lifecycle from parcel registration
through title issuance, transfers, adjudication of claims, encumbrance management, valuation
rolls, stamp duty computation, lease management, caution workflows, spousal consent
enforcement, rates ledger, survey plan registry, dispute escalation, and title certificate
generation.

## Core Concepts

| Concept | Description |
|---------|-------------|
| **Parcel** | A surveyed unit of land identified by a unique parcel number |
| **Title** | Legal document of ownership linked to a parcel |
| **Transfer** | Conveyance of title ownership from one party to another |
| **Adjudication** | Formal process to determine rightful ownership of disputed land |
| **Encumbrance** | Charge, mortgage, or restriction registered against a title |
| **Valuation Roll** | Annual assessment of land values for rating purposes |
| **Caution** | Provisional blocking instrument lodged against a title (LRA 2012 s.71) |
| **Lease** | Leasehold agreement against a titled parcel with term and rent |
| **Stamp Duty** | Transfer tax computed per Kenya Stamp Duty Act Cap 480 |
| **Escalation** | Appeal of an adjudication decision to the tribunal or court |
| **Survey Plan** | Deposited plan from a licensed surveyor (Survey Act Cap 299) |

---

## Workflows

### 1. Register a New Parcel

```python
p = await svc.register_parcel(
    parcel_number="NAIROBI/WESTLANDS/001",
    county="Nairobi",
    sub_county="Westlands",
    location="Parklands",
    area_hectares=0.125,
    land_use="residential",
    tenant_id="lands_kenya",
)
```

### 2. Issue a Title Deed

```python
t = await svc.issue_title(
    parcel_id=p["id"],
    title_number="IR 12345",
    owner_id="owner-001",
    owner_name="John Doe",
    issue_date="2025-01-15",
    tenure_type="freehold",
    issued_by="Registrar of Titles",
    tenant_id="lands_kenya",
)
```

### 3. Register a Mortgage / Encumbrance

```python
enc = await svc.register_encumbrance(
    title_id=t["id"],
    encumbrance_type="mortgage",
    holder_id="bank-001",
    holder_name="Kenya Commercial Bank",
    amount_kes=5_000_000,
    start_date="2025-02-01",
    instrument_reference="MORT-2025-001",
    registered_by="Registrar of Titles",
    tenant_id="lands_kenya",
)
```

### 4. Initiate and Complete a Land Transfer with Stamp Duty

```python
# Step 1: Initiate transfer
tr = await svc.initiate_transfer(
    title_id=t["id"],
    transferor_id="owner-001", transferor_name="John Doe",
    transferee_id="owner-002", transferee_name="Jane Smith",
    consideration_kes=8_500_000,
    transfer_date="2025-03-01",
    instrument_number="TRANS-2025-001",
    approved_by="Registrar",
    tenant_id="lands_kenya",
)

# Step 2: Compute stamp duty
duty = await svc.compute_stamp_duty(tr["id"], 8_500_000, "residential", tenant_id="lands_kenya")
# duty["total_payable_kes"] → "448500.00"
# Breakdown: stamp_duty_kes=340000, cgt_kes=425000, registration_fee_kes=...

# Step 3: Record payment from KRA
await svc.record_duty_payment(
    tr["id"], "KRA-REF-2025-001", 448500, "RCT-KRA-001", "John Doe",
    tenant_id="lands_kenya",
)

# Step 4: Complete (ownership transferred)
await svc.complete_transfer(tr["id"], tenant_id="lands_kenya")
```

### 5. Parcel Subdivision

```python
result = await svc.subdivide_parcel(
    parent_parcel_id=p["id"],
    child_parcels=[
        {"parcel_number": "NAIROBI/WESTLANDS/001A", "area_hectares": 0.06, "land_use": "residential"},
        {"parcel_number": "NAIROBI/WESTLANDS/001B", "area_hectares": 0.06, "land_use": "residential"},
    ],
    survey_reference="SURVEY-2025-001",
    authorized_by="Chief Registrar",
    tenant_id="lands_kenya",
)
# result["child_count"] == 2; parent is now status="subdivided"
```

### 6. Title Chain of Ownership

```python
chain = await svc.get_title_chain(parcel_id=p["id"], tenant_id="lands_kenya")
# chain["chain"] → [{event, from_owner, to_owner, consideration_kes, date, instrument}, ...]
# chain["integrity_hash"] → SHA-256 of all instrument references
```

### 7. Register a Lease

```python
lease = await svc.register_lease(
    title_id=t["id"],
    lessee_id="lessee-001",
    lessee_name="Acme Ltd",
    start_date="2025-01-01",
    term_years=10,
    annual_rent_kes=240_000,
    registered_by="Registrar",
    tenant_id="lands_kenya",
)
# lease["expiry_date"] == "2035-01-01"
# lease["total_rent_kes"] == "2400000.00"

# Later: renew
await svc.renew_lease(lease["id"], extension_years=5, new_annual_rent_kes=300_000,
                       renewed_by="Registrar", tenant_id="lands_kenya")
```

### 8. Caution Workflow (LRA 2012 s.71–73)

```python
# Lodge a caution (auto-expires in 60 days)
c = await svc.lodge_caution(
    title_id=t["id"],
    cautioner_id="creditor-001",
    cautioner_name="First Bank",
    grounds="Unpaid mortgage debt",
    tenant_id="lands_kenya",
    expiry_days=60,
)

# Confirm via court order → becomes permanent restriction
await svc.confirm_caution(c["id"], court_order_ref="COURT-ORD-001",
                           confirmed_by="Judge", tenant_id="lands_kenya")

# OR withdraw voluntarily
await svc.withdraw_caution(c["id"], withdrawal_reason="Debt settled",
                            withdrawn_by="creditor-001", tenant_id="lands_kenya")

# Bulk expire stale cautions
result = await svc.expire_stale_cautions(tenant_id="lands_kenya")
# result["expired_count"]
```

### 9. Spousal Consent & Matrimonial Property (LRA 2012 s.93)

```python
# Flag title as matrimonial property
await svc.flag_matrimonial_property(
    title_id=t["id"], reason="Registered marital home",
    flagged_by="Registrar", tenant_id="lands_kenya",
)

# Register consent before any transfer
await svc.register_spousal_consent(
    title_id=t["id"],
    spouse_id="spouse-001",
    spouse_name="Mary Doe",
    consent_date="2025-02-15",
    witness_id="witness-001",
    tenant_id="lands_kenya",
)
# Without this, initiate_transfer raises PermissionError("spousal_consent_required")
```

### 10. Land Rates Ledger & Arrears

```python
# Assess rates for the year
assessment = await svc.assess_land_rates(
    parcel_id=p["id"], rate_year=2025,
    rate_per_hectare_kes=15_000, tenant_id="lands_kenya",
)

# Record a partial payment
await svc.record_rates_payment(
    assessment_id=assessment["id"],
    amount_paid_kes=1_000,
    payment_date="2025-07-01",
    receipt_number="CNT-RCT-001",
    paid_by="owner-001",
    tenant_id="lands_kenya",
)

# Compute arrears with 2% monthly penalty
arrears = await svc.compute_rates_arrears(
    parcel_id=p["id"], as_of_date="2025-12-31",
    tenant_id="lands_kenya",
)
# arrears["total_arrears_kes"] includes principal + penalty interest
```

### 11. Survey Plan Registry

```python
# Register a licensed surveyor
sv = await svc.register_surveyor(
    surveyor_id="sv-001", name="John Survey Ltd",
    licence_number="SRV-LIC-001", licence_expiry_date="2030-12-31",
    tenant_id="lands_kenya",
)

# Deposit a survey plan (validates licence is not expired)
plan = await svc.deposit_survey_plan(
    parcel_id=p["id"],
    surveyor_id="sv-001",
    plan_number="PLAN-2025-001",
    plan_date="2025-03-15",
    plan_document_ref="doc://survey-plans/PLAN-2025-001.pdf",
    tenant_id="lands_kenya",
)

plans = await svc.list_survey_plans(p["id"], tenant_id="lands_kenya")
```

### 12. Adjudication Escalation

```python
adj = await svc.submit_adjudication(
    parcel_id=p["id"], claimant_id="clm-001", claimant_name="Dave Mwangi",
    claim_basis="adverse_possession", evidence_reference="EV-001",
    adjudicator_id="adj-officer-001", tenant_id="lands_kenya",
)
await svc.decide_adjudication(adj["id"], "rejected", "Insufficient evidence",
                               tenant_id="lands_kenya")

# Escalate to tribunal
esc = await svc.escalate_adjudication(
    adjudication_id=adj["id"],
    escalation_type="tribunal",   # tribunal | elc | high_court
    tribunal_ref="TRIB-2025-001",
    grounds="Procedural error in primary adjudication",
    escalated_by="clm-001",
    tenant_id="lands_kenya",
)

# Record judgement
await svc.record_tribunal_decision(
    escalation_id=esc["id"],
    decision="overturned",
    decision_date="2025-09-01",
    judgement_ref="ELC-JUDG-2025-001",
    recorded_by="Court Clerk",
    tenant_id="lands_kenya",
)
```

### 13. Title Certificate Generation

```python
cert = await svc.generate_title_certificate(
    title_id=t["id"],
    generated_by="Chief Registrar",
    tenant_id="lands_kenya",
)
# cert["certificate_payload"] — full structured payload
# cert["qr_code_seed"]         — SHA-256 seed for QR code
# cert["digital_signature_placeholder"] — downstream signing hook
```

---

## Supported Reference Values

### Land Uses
`residential`, `commercial`, `agricultural`, `industrial`, `mixed_use`, `public`,
`conservation`, `institutional`

### Tenure Types
`freehold`, `leasehold`, `community`, `government`

### Encumbrance Types
`mortgage`, `caveat`, `charge`, `easement`, `restriction`, `lien`, `covenant`, `caution`

### Valuation Methods
`market_comparison`, `income_capitalisation`, `depreciated_replacement_cost`,
`residual_method`

### Adjudication Outcomes
`approved`, `rejected`, `referred`, `appealed`, `withdrawn`

### Escalation Types
`tribunal`, `elc`, `high_court`

### Stamp Duty Exemption Types
`first_time_buyer`, `government`, `ngo`, `inheritance`, `court_order`

---

## Stamp Duty Rate Schedule (Stamp Duty Act Cap 480)

| Land Use | Stamp Duty Rate |
|----------|----------------|
| residential | 4% |
| commercial | 4% |
| industrial | 4% |
| mixed_use | 4% |
| institutional | 4% |
| agricultural | 2% |
| public | 0% |
| conservation | 0% |

Registration fees are tiered: KES 2,500 base (≤1M), +0.1% on 1M–5M, +0.05% above 5M.
CGT is computed at 5% of consideration (simplified; no cost-basis input required).

---

## Error Codes

| Code | Meaning |
|------|---------|
| `PermissionError("active_title_exists")` | Parcel already has an active title |
| `PermissionError("can_only_transfer_active_titles")` | Transfer blocked — title not active |
| `PermissionError("title_encumbered_cannot_transfer")` | Caveat or restriction blocks transfer |
| `PermissionError("spousal_consent_required")` | Matrimonial property — consent missing |
| `PermissionError("surveyor_licence_expired")` | Surveyor's licence past expiry date |
| `PermissionError("zero_payment_requires_exemption")` | Zero-value payment without exemption record |
| `KeyError` | Resource not found or belongs to a different tenant |
| `ValueError` | Business rule violated (bad land use, negative area, etc.) |

---

## Audit Events

Every mutating operation emits a structured audit event retrievable via
`get_audit_events(tenant_id)`. Key event types:

`parcel_registered`, `parcel_subdivided`, `parcel_updated`, `parcel_deregistered`,
`title_issued`, `title_updated`, `title_cancelled`, `title_rectified`,
`transfer_initiated`, `transfer_completed`, `stamp_duty_computed`, `duty_payment_recorded`,
`encumbrance_registered`, `encumbrance_discharged`,
`adjudication_submitted`, `adjudication_decided`, `adjudication_escalated`,
`tribunal_decision_recorded`,
`valuation_recorded`, `valuation_approved`,
`land_search_conducted`, `land_rates_assessed`, `rates_payment_recorded`,
`lease_registered`, `lease_renewed`,
`caution_lodged`, `caution_confirmed`, `caution_withdrawn`,
`spousal_consent_registered`, `matrimonial_property_flagged`,
`surveyor_registered`, `survey_plan_deposited`,
`certificate_generated`
