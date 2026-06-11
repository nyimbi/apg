# Construction Management (realestate_con) — User Guide

**Capability ID**: `realestate_con` | **Domain**: `realestate` | **Version**: `1.1.0`

## What This Capability Does

`realestate_con` manages the full construction project and contract lifecycle: from contract drafting and execution, through on-site progress tracking (milestones, variations, payment certificates), quality management (snagging and defect tracking), risk and drawing registers, extension of time claims, and final handover with a Practical Completion certificate.

---

## Installation

```bash
pip install apg-realestate-con
```

Or within the APG monorepo:

```bash
uv pip install -e capabilities/realestate/con
```

---

## Quick Start

```python
import asyncio
from capabilities.realestate.con.service import ConService
from capabilities.realestate.con.models import (
    ContractCreate, ContractType, ContractParty, PartyRole,
    ContractorCreate, ContractorGrade, MilestoneCreate, MilestoneType,
)
from datetime import date
from decimal import Decimal

svc = ConService(tenant_id="t_001", actor_id="user_pm_1")

async def main():
    # 1. Draft a construction contract
    contract = await svc.draft_contract(
        contract_type="construction_contract",
        parties=[
            {"party_id": "dev_001", "party_name": "Datacraft Developments", "role": "developer"},
            {"party_id": "con_001", "party_name": "Acme Builders Ltd", "role": "contractor"},
        ],
        property_id="prop_riverside_001",
        terms={"contract_value": 50_000_000, "currency": "KES", "programme_weeks": 78},
        tenant_id="t_001",
        governing_law="Laws of Kenya",
    )
    print(contract["contract_ref"])  # e.g. CONTR-0190AF3B

    # 2. Register the contractor
    from capabilities.realestate.con.models import ContractorCreate, ContractorGrade
    contractor = await svc.register_contractor(ContractorCreate(
        tenant_id="t_001",
        name="Acme Builders Ltd",
        contractor_type="main_contractor",
        email="info@acmebuilders.co.ke",
        phone="+254700000001",
        grade=ContractorGrade.preferred,
        specialisms=["civils", "superstructure", "fit-out"],
        created_by="user_pm_1",
    ))

    # 3. Create a milestone
    milestone = await svc.create_milestone(MilestoneCreate(
        tenant_id="t_001",
        contract_id=contract["id"],
        milestone_type=MilestoneType.completion,
        title="Substructure Complete",
        due_date=date(2026, 10, 15),
        amount=Decimal("5000000"),
        created_by="user_pm_1",
    ))

asyncio.run(main())
```

---

## Core Workflows

### Contract Lifecycle

```
draft_contract()
    → sign_contract_party()   [repeat for each party]
    → contract_review()       [legal review, approved=True]
    → execute_contract()      [status: draft → active]
    → contract_close() / issue_practical_completion_certificate()
    → release_retention()
```

### Snagging Workflow

```python
# Inspector creates snag during walkdown
snag = await svc.create_snag_item(
    contract_id="contr_001",
    tenant_id="t_001",
    title="Hairline crack in column C3 finish",
    location="Level 2, Grid C3",
    trade="finishes",
    severity="major",
    reported_by="inspector_jane",
    evidence_ids=["photo_001", "photo_002"],
)
# SLA due_date is auto-set to today + 7 days for major severity

# Contractor resolves and provides evidence
await svc.resolve_snag_item(
    snag_id=snag["id"],
    tenant_id="t_001",
    resolution_notes="Crack cleaned, filled with matching render, cured.",
    resolved_by="site_manager_john",
    evidence_ids=["photo_003_after"],
)

# QS/PM checks overall status
summary = await svc.get_snag_summary(tenant_id="t_001", contract_id="contr_001")
# {"total_snags": 47, "open_snags": 3, "resolved_snags": 44, "by_trade": {...}}
```

### Payment Certificate

```python
cert = await svc.issue_payment_certificate(
    contract_id="contr_001",
    tenant_id="t_001",
    period_end=date(2026, 8, 31),
    gross_value=Decimal("8_500_000"),
    variations_included=["vo_001", "vo_002"],
    certified_by="qs_james",
    retention_percentage=Decimal("5"),
    advance_payment_deduction=Decimal("500_000"),
)
# net_certified = 8,500,000 × 0.95 - 500,000 = 7,575,000
```

### Variation Order

```python
# Raise a variation
vo = await svc.raise_variation(VariationOrderCreate(
    tenant_id="t_001",
    contract_id="contr_001",
    variation_type=VariationType.scope_change,
    description="Additional basement car park level",
    amount_change=Decimal("3_200_000"),
    timeline_change_days=21,
    created_by="user_pm_1",
))

# Approve
await svc.approve_variation(
    vo_id=vo.id,
    tenant_id="t_001",
    approved_by="director_001",
    board_approval=True,   # required because > 500k
)
```

### Extension of Time (EOT) Claim

```python
# Contractor submits EOT for employer risk delay
eot = await svc.submit_extension_of_time(
    contract_id="contr_001",
    tenant_id="t_001",
    days_claimed=14,
    cause="Late release of design drawings for Level 5",
    cause_category="employer_risk",    # eligible for EOT
    submitted_by="contractor_pm",
    affected_milestone_ids=["ms_001", "ms_002"],
)

# PM/CA assesses the claim
await svc.assess_extension_of_time(
    eot_id=eot["id"],
    tenant_id="t_001",
    days_awarded=10,
    assessed_by="contract_admin",
    assessment_notes="10 of 14 days substantiated by drawing register evidence.",
)
# affected milestones are automatically extended by 10 days
```

### Risk Register

```python
risk = await svc.register_risk(
    contract_id="contr_001",
    tenant_id="t_001",
    title="Adverse ground conditions in Grid A-B",
    category="ground_conditions",
    probability=0.35,
    impact_cost=Decimal("4_000_000"),
    impact_days=30,
    owner="geotech_lead",
    mitigation_action="Commission additional boreholes before piling commences",
)
# risk_score = probability_band (2) × impact_band (5) = 10

# Get top risks (score >= 10)
top_risks = await svc.get_risk_register(
    tenant_id="t_001",
    contract_id="contr_001",
    min_risk_score=10,
)
```

### Drawing Register

```python
# Register first issue
dwg = await svc.register_drawing(
    contract_id="contr_001",
    tenant_id="t_001",
    drawing_number="A-001",
    revision="P1",
    title="Ground Floor Plan",
    discipline="architectural",
    document_id="doc_a001_p1",
    drawn_by="arch_studio",
)

# Issue revision — previous revision auto-superseded
dwg_rev2 = await svc.register_drawing(
    contract_id="contr_001",
    tenant_id="t_001",
    drawing_number="A-001",
    revision="P2",
    title="Ground Floor Plan (Revised)",
    discipline="architectural",
    document_id="doc_a001_p2",
    drawn_by="arch_studio",
)

# Site gets current set (no superseded drawings)
current = await svc.get_current_drawing_set(
    tenant_id="t_001",
    contract_id="contr_001",
    discipline="architectural",
)
```

### Practical Completion Certificate

```python
# Must have ≤ 0 open snags (configurable) and commissioning complete
try:
    pc = await svc.issue_practical_completion_certificate(
        contract_id="contr_001",
        tenant_id="t_001",
        issued_by="contract_admin",
        dlp_months=12,
        outstanding_snags_allowed=0,
        commissioning_complete=True,
        o_and_m_manuals_received=True,
    )
    # pc["dlp_end"] = issued_date + 12 months
    # contract status → completed
    # triggers retention release workflow via NATS event
except ValueError as e:
    print(e)  # "practical_completion_blocked: 3 open snags, threshold is 0"
```

### Dispute Management

```python
dispute = await svc.dispute_management(
    contract_id="contr_001",
    tenant_id="t_001",
    dispute_type="delay_dispute",
    claimed_amount=Decimal("2_100_000"),
    claimant_id="contractor_001",
    respondent_id="developer_001",
    dispute_description="14-day delay in site possession caused programme loss",
    resolution_method="adjudication",
)

# Resolve after adjudicator's decision
await svc.resolve_dispute(
    dispute_id=dispute.id,
    tenant_id="t_001",
    resolution_summary="Adjudicator awarded contractor 10 days at KES 150k/day = KES 1.5M",
)
```

---

## Snag Severity SLA Reference

| Severity | Auto SLA (days) | Typical examples |
|----------|----------------|-----------------|
| critical | 2 | Structural crack, flooding, electrical hazard |
| major | 7 | Significant finishing defect, door not closing |
| minor | 14 | Paint touch-up, minor scratch, grout gap |
| observation | 28 | Aesthetic item, non-urgent note |

---

## EOT Cause Category Eligibility

| Cause Category | Eligible for EOT | Eligible for Loss & Expense |
|----------------|-----------------|----------------------------|
| employer_risk | Yes | Yes |
| neutral_risk | Yes | No |
| force_majeure | Yes | No |
| contractor_risk | No | No |

---

## NATS Event Reference

All events are published to NATS JetStream. Subjects follow `con.<entity>.<action>`.

| Subject | Trigger |
|---------|---------|
| `con.contract.executed` | `execute_contract()` |
| `con.contract.terminated` | `terminate_contract()` |
| `con.contract.pc_cert_issued` | `issue_practical_completion_certificate()` |
| `con.snag.created` | `create_snag_item()` |
| `con.snag.resolved` | `resolve_snag_item()` |
| `con.drawing.superseded` | `register_drawing()` on existing drawing number |
| `con.eot.granted` | `assess_extension_of_time()` with days_awarded > 0 |
| `con.eot.rejected` | `assess_extension_of_time()` with days_awarded = 0 |
| `con.payment_cert.issued` | `issue_payment_certificate()` |
| `con.milestone.overdue` | `get_overdue_milestones()` scan |
| `con.notice.default_served` | `default_notice()` |
| `con.retention.released` | `release_retention()` |

---

## Interoperability

```apg
use realestate_con;
```

Key integration points:
- **`realestate_acc`**: Payment certificate net certified amounts posted as AP invoices; retention balances tracked as liabilities; PC certificate triggers retention release.
- **`realestate_mai`**: Contractor registry shared; post-DLP defects handled as maintenance work orders.
- **`realestate_prm`**: Management contract terms linked to property management configuration.
- **`ntfy`**: NATS events consumed to generate SMS/email alerts to site teams, developers, and lenders.
- **`audl`**: All mutating operations emit immutable audit records.

---

## Configuration Reference

All keys are tenant-scoped via the `conf` capability or environment variables prefixed `REALESTATE_CON_`.

| Key | Default | Description |
|-----|---------|-------------|
| `REALESTATE_CON_BOARD_APPROVAL_THRESHOLD` | 500000 | KES VO amount requiring board approval |
| `REALESTATE_CON_RETENTION_PCT` | 5.0 | Default retention percentage |
| `REALESTATE_CON_GRADING_REVIEW_MONTHS` | 12 | Contractor grade review frequency |
| `REALESTATE_CON_DLP_MONTHS` | 12 | Default DLP duration in months |
| `REALESTATE_CON_EOT_NOTICE_DAYS` | 28 | Compensation event notice window |

---

## Further Reading

- `service.py` — Business logic implementation (all async methods)
- `models.py` — Pydantic v2 data models
- `api.py` — REST API endpoint definitions
- `views.py` — Flask-AppBuilder views
- `README.md` — Quick reference and method index
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 planned enhancements
- `tests/test_service.py` — Service unit tests
- `domain/events.py` — NATS event schema definitions
