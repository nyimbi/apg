# Property Insurance — User Guide

**Capability ID**: `realestate_ins` | **Domain**: `realestate` | **Version**: `2.0.0`

---

## Description

End-to-end property insurance portfolio management covering policy lifecycle, asset schedules, claims processing, endorsement management, premium allocation, coverage gap analysis, parametric catastrophe triggers, fraud scoring, subrogation recovery, portfolio stress testing, tenant premium apportionment, broker performance scoring, and structured certificate issuance.

---

## Installation

```bash
pip install apg-realestate-ins
```

---

## Quick Start

```python
import asyncio
from decimal import Decimal
from datetime import date
from capabilities.realestate.ins.service import InsService
from capabilities.realestate.ins.models import (
    InsurerCreate, InsurerGrade,
    PolicyCreate, PolicyType, ValuationBasis,
    ClaimCreate, ClaimType,
)

svc = InsService(tenant_id="tenant-001", actor_id="admin")

async def main():
    # Register insurer
    insurer = await svc.register_insurer(InsurerCreate(
        tenant_id="tenant-001",
        name="Jubilee Insurance",
        grade=InsurerGrade.preferred,
        email="jubilee@example.com",
        created_by="admin",
    ))

    # Create and bind a policy
    policy = await svc.create_policy(PolicyCreate(
        tenant_id="tenant-001",
        policy_number="POL-2025-001",
        policy_type=PolicyType.property_all_risk,
        insurer_id=insurer.id,
        commencement_date=date(2025, 1, 1),
        expiry_date=date(2026, 1, 1),
        sum_insured=Decimal("50000000"),
        annual_premium=Decimal("250000"),
        valuation_basis=ValuationBasis.reinstatement_cost,
        created_by="admin",
    ))
    bound = await svc.bind_policy(policy.id, "tenant-001")

asyncio.run(main())
```

---

## Core Workflows

### Policy Lifecycle

```python
# Create → Bind → Endorse → Renew
policy = await svc.create_policy(payload)
bound  = await svc.bind_policy(policy.id, tenant_id)
end    = await svc.issue_endorsement(EndorsementCreate(...))
review = await svc.policy_renewal_review(policy.id, tenant_id, renewal_decision="renew")
```

### Structured Renewal Pipeline

Advance through stages with full event history:

```python
# 90 days out — send RFQ to broker
await svc.advance_renewal_stage(policy_id, tenant_id, "rfq_sent", broker_id="BRK-001")

# 60 days out — record market quotes
await svc.advance_renewal_stage(policy_id, tenant_id, "quotes_received",
    broker_id="BRK-001",
    market_quotes=[
        {"insurer": "Jubilee", "premium": 245000, "sum_insured": 50000000},
        {"insurer": "CIC", "premium": 260000, "sum_insured": 50000000},
    ])

# 30 days out — approve
await svc.advance_renewal_stage(policy_id, tenant_id, "approved", notes="Board approved")

# Bind
await svc.advance_renewal_stage(policy_id, tenant_id, "bound")
```

### Claims Processing

```python
# Lodge
claim = await svc.lodge_claim(ClaimCreate(
    tenant_id="tenant-001",
    policy_id=policy.id,
    claim_type=ClaimType.partial_loss,
    peril="fire",
    incident_date=date(2025, 6, 15),
    description="Kitchen fire, ground floor",
    estimated_loss=Decimal("3500000"),
    property_id="PROP-001",
    created_by="property_manager",
))

# Score for fraud before processing
fraud = await svc.score_claim_fraud_risk(claim.id, "tenant-001")
# fraud["fraud_score"]  → 0–100
# fraud["route_to_senior_adjuster"] → bool

# Appoint loss adjuster
await svc.loss_adjuster_appointment(claim.id, "ADJ-001", "tenant-001",
    adjuster_firm="Cunningham Lindsey")

# Reinstatement costing
await svc.reinstatement_costing(claim.id, [
    {"description": "Kitchen rebuild", "cost": "2800000"},
    {"description": "Contents replacement", "cost": "700000"},
], "tenant-001", surveyor_id="QS-001")

# Approve and settle
await svc.approve_claim(claim.id, "tenant-001", Decimal("3200000"), senior_approved=True)
await svc.claim_settlement(claim.id, Decimal("3200000"), date.today(), "tenant-001",
    settlement_basis="agreed_settlement", payment_reference="PAY-2025-0789")
```

### Attaching Evidence

```python
import hashlib

file_bytes = open("incident_photo.jpg", "rb").read()
sha256 = hashlib.sha256(file_bytes).hexdigest()

evidence = await svc.attach_claim_evidence(
    claim_id=claim.id,
    tenant_id="tenant-001",
    evidence_type="photo",
    file_reference="s3://bucket/claims/photo.jpg",
    file_hash_sha256=sha256,
    description="Kitchen fire damage",
    uploaded_by="property_manager",
)
```

### Subrogation Recovery

When a settled claim was caused by a third party:

```python
# Open recovery file
sub = await svc.initiate_subrogation(
    claim_id=claim.id,
    tenant_id="tenant-001",
    liable_party_id="CONTRACTOR-042",
    liable_party_name="Acme Plumbing Ltd",
    recovery_basis="contractor_negligence",
    estimated_recovery=Decimal("2000000"),
    assigned_to="legal_team",
)

# Record partial recovery
await svc.record_subrogation_recovery(sub["id"], "tenant-001", Decimal("1200000"),
    payment_reference="LEGAL-REC-001")
```

### Parametric Insurance (Catastrophe Perils)

Trigger auto-claim when an oracle reading exceeds the agreed threshold:

```python
result = await svc.parametric_trigger_evaluate(
    property_id="PROP-WESTLANDS",
    peril="flood",
    measurement_value=Decimal("185"),  # mm rainfall
    measurement_unit="mm",
    threshold_value=Decimal("150"),
    tenant_id="tenant-001",
    data_source="KMD-Oracle",
    measurement_date=date.today(),
)
# result["triggered"] == True  → claim auto-lodged and approved
# result["auto_claim"] contains the ClaimResponse dict
```

### Premium Allocation and Tenant Apportionment

```python
# Allocate across units by floor area
result = await svc.apportion_insurance_to_tenants(
    policy_id=policy.id,
    tenant_id="tenant-001",
    tenant_unit_map=[
        {"tenant_id": "TENANT-A", "unit_id": "U101", "floor_area_sqm": 120},
        {"tenant_id": "TENANT-B", "unit_id": "U102", "floor_area_sqm": 80},
        {"tenant_id": "TENANT-C", "unit_id": "U201", "floor_area_sqm": 200},
    ],
    apportionment_basis="floor_area",
    period="2025-06",
)
# result["apportioned_charges"] → per-tenant insurance charge schedule
```

### Certificate Issuance

```python
cert = await svc.issue_certificate(
    policy_id=policy.id,
    tenant_id="tenant-001",
    certificate_type="mortgage_endorsement",
    beneficiary_name="KCB Bank",
    beneficiary_reference="LOAN-2025-KCB-001",
    issued_by="admin",
)
# cert["cert_number"]  → "CERT-XXXXXXXX"
# Pass cert dict to pdf/docx skill for rendering
```

### Under-Insurance Check

```python
check = await svc.under_insurance_check(
    property_id="PROP-001",
    tenant_id="tenant-001",
    current_rebuild_cost=Decimal("65000000"),
)
# check["under_insured"] → True/False
# check["adequacy_ratio_pct"] → e.g. 76.9
# check["insurance_gap"] → KES shortfall
```

### Portfolio Stress Testing

```python
result = await svc.run_portfolio_stress_test(
    tenant_id="tenant-001",
    scenario_name="Westlands_1in100_Flood",
    affected_perils=["flood"],
    pml_factor=Decimal("0.35"),
    affected_location="Westlands",
)
# result["gross_pml"]         → total exposure before reinsurance
# result["net_retained_loss"] → after XL reinsurance recovery
# result["capital_adequacy_flag"] → bool
```

### Loss Run Report

```python
loss_run = await svc.generate_loss_run(
    tenant_id="tenant-001",
    years=5,
    policy_id=policy.id,
)
# loss_run["annual_breakdown"] → year-by-year freq/severity
# loss_run["severity_trend"]   → "increasing" | "stable" | "decreasing"
```

### Broker Scorecard

```python
scorecard = await svc.get_broker_scorecard(
    broker_id="BRK-001",
    tenant_id="tenant-001",
    period_years=3,
)
# scorecard["scorecard_score"] → 0–100
# scorecard["scorecard_band"]  → "preferred" | "approved" | "conditional"
# scorecard["retention_rate_pct"]
# scorecard["avg_quote_turnaround_days"]
```

### Insurance Analytics

```python
analytics = await svc.insurance_analytics("2025", "tenant-001")
# analytics["loss_ratio_pct"]
# analytics["total_sum_insured"]
# analytics["expiring_90_days"]
```

---

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/realestate/ins/dashboard` | `realestate_ins:view` | Overview |
| `/realestate/ins/policies` | `realestate_ins:policies` | Policies |
| `/realestate/ins/assets` | `realestate_ins:assets` | Assets |
| `/realestate/ins/claims` | `realestate_ins:claims` | Claims |
| `/realestate/ins/premiums` | `realestate_ins:premiums` | Financial |
| `/realestate/ins/gaps` | `realestate_ins:gaps` | Analysis |
| `/realestate/ins/endorsements` | `realestate_ins:endorsements` | Policies |
| `/realestate/ins/insurers` | `realestate_ins:insurers` | Registry |
| `/realestate/ins/certificates` | `realestate_ins:certificates` | Compliance |
| `/realestate/ins/loss-run` | `realestate_ins:reporting` | Reporting |
| `/realestate/ins/stress-test` | `realestate_ins:reporting` | Reporting |
| `/realestate/ins/brokers/<id>/scorecard` | `realestate_ins:reporting` | Reporting |

---

## Service Method Reference

### Insurer Registry
| Method | Description |
|--------|-------------|
| `register_insurer(payload)` | Register insurer with grade |
| `get_insurer(insurer_id, tenant_id)` | Fetch by ID |
| `list_insurers(tenant_id, grade?)` | Filter by grade |

### Policy Management
| Method | Description |
|--------|-------------|
| `create_policy(payload)` | Create policy with perils and deductibles |
| `get_policy(policy_id, tenant_id)` | Fetch by ID |
| `list_policies(tenant_id, property_id?, status?)` | Filter policies |
| `bind_policy(policy_id, tenant_id)` | Activate after asset schedule validation |
| `update_policy(policy_id, tenant_id, updates)` | Update status/premium/dates |
| `get_renewal_pipeline(tenant_id, days_ahead?)` | Expiring policies window |
| `advance_renewal_stage(policy_id, tenant_id, stage, ...)` | Structured renewal transitions |
| `policy_renewal_review(policy_id, tenant_id, ...)` | Record broker recommendation and decision |

### Asset Schedule
| Method | Description |
|--------|-------------|
| `add_asset_to_schedule(payload)` | Add typed asset with valuation basis |
| `schedule_insurance(policy_id, asset_id, insured_value, ...)` | Schedule with coverage type |
| `list_policy_assets(tenant_id, policy_id)` | Assets on a policy |
| `remove_asset_from_schedule(asset_id, tenant_id)` | Remove asset |

### Claims
| Method | Description |
|--------|-------------|
| `lodge_claim(payload)` | Lodge claim with peril and policy validation |
| `get_claim(claim_id, tenant_id)` | Fetch by ID |
| `list_claims(tenant_id, policy_id?, status?)` | Filter claims |
| `approve_claim(claim_id, tenant_id, approved_value, senior_approved)` | Approve |
| `settle_claim(claim_id, tenant_id, settlement_amount)` | Settle |
| `claim_settlement(claim_id, amount, date, tenant_id, ...)` | Full settlement with metadata |
| `claim_notification(property_id, incident_type, ...)` | Initial insurer notification |
| `score_claim_fraud_risk(claim_id, tenant_id)` | Fraud score 0–100 |
| `attach_claim_evidence(claim_id, tenant_id, type, file_ref, sha256, ...)` | Evidence vault |
| `initiate_subrogation(claim_id, tenant_id, liable_party_id, ...)` | Open recovery file |
| `record_subrogation_recovery(subrogation_id, tenant_id, amount, ...)` | Record recovery |

### Endorsements
| Method | Description |
|--------|-------------|
| `issue_endorsement(payload)` | Issue typed endorsement, adjusts sum insured |
| `list_endorsements(tenant_id, policy_id?)` | List endorsements |

### Premium
| Method | Description |
|--------|-------------|
| `allocate_premium(payload)` | Run allocation across properties |
| `premium_allocation(policy_id, units, tenant_id, method?, ...)` | Simplified allocation |
| `apportion_insurance_to_tenants(policy_id, tenant_id, tenant_unit_map, ...)` | Billing charge schedule |

### Coverage Analysis
| Method | Description |
|--------|-------------|
| `detect_coverage_gaps(tenant_id, property_id)` | Auto-detect gaps |
| `record_coverage_gap(payload)` | Record a gap with severity |
| `list_coverage_gaps(tenant_id, property_id?, resolved?)` | List gaps |
| `under_insurance_check(property_id, tenant_id, current_rebuild_cost?)` | Adequacy ratio |

### Advanced
| Method | Description |
|--------|-------------|
| `parametric_trigger_evaluate(property_id, peril, measurement_value, ...)` | Auto-claim on oracle trigger |
| `issue_certificate(policy_id, tenant_id, certificate_type, ...)` | Issue formal certificate |
| `generate_loss_run(tenant_id, years?, policy_id?, property_id?)` | 5-year loss history |
| `run_portfolio_stress_test(tenant_id, scenario_name, affected_perils, pml_factor, ...)` | PML scenario |
| `loss_adjuster_appointment(claim_id, adjuster_id, tenant_id, ...)` | Appoint adjuster |
| `reinstatement_costing(claim_id, items, tenant_id, ...)` | QS cost estimate |
| `captive_insurance_management(captive_id, period, tenant_id, ...)` | Captive P&L and solvency |
| `get_broker_scorecard(broker_id, tenant_id, period_years?)` | Broker KPI score |
| `insurance_analytics(period, tenant_id)` | Portfolio analytics |
| `get_insurance_summary(tenant_id)` | Dashboard summary |

---

## Interoperability

```apg
use realestate_ins;
```

`realestate_ins` publishes events on `mqeb` consumed by:
- `realestate_acc` — premium and settlement postings, tenant insurance charges
- `ntfy` — renewal due, critical gap, large claim, parametric trigger alerts
- `audl` — immutable claim and endorsement audit trail
- `realestate_val` — reinstatement cost valuation requests
- `realestate_lea` — tenant damage subrogation recovery linkage
- `docx`/`pdf` — certificate and loss run document rendering

---

## Configuration

All keys are tenant-scoped. Set via `conf` capability or `REALESTATE_INS_*` env vars.

| Key | Default | Description |
|-----|---------|-------------|
| `claims.large_claim_threshold` | 1,000,000 | KES requiring senior approval |
| `renewals.early_warning_days` | 90 | Days before expiry for first alert |
| `gaps.auto_alert_on_critical` | true | Mandatory alert on critical severity gap |
| `fraud.high_risk_threshold` | 70 | Score above which claim routed to senior adjuster |
| `parametric.auto_approve` | true | Auto-approve parametric trigger claims |
| `stress_test.xl_retention_limit` | 10,000,000 | XL reinsurance retention in KES |

---

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Pydantic v2 data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views
- `README.md` — Quick reference
- `WORLD_CLASS_IMPROVEMENTS.md` — Roadmap for v3 enhancements
- `SPECIFICATION.md` — Full capability specification
