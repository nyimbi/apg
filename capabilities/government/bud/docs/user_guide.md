# Budget & Financial Planning — User Guide

**Capability ID**: `government_bud` | **Domain**: `government` | **Version**: `2.0.0`

---

## 1. Introduction

The `government_bud` capability provides the full government budget lifecycle from budget preparation through execution, reporting, and year-end closure. It enforces Public Finance Management Act (PFMA) rules, supports Medium-Term Expenditure Framework (MTEF) planning, Programme-Based Budgeting (PBB), inter-government fiscal transfers (IGFT), and generates IPSAS-aligned financial statements.

All operations are tenant-scoped. Every state-changing action emits an audit event and, where indicated, a CloudEvent to the configured NATS subject via bytewax.

---

## 2. Installation

```bash
pip install apg-government-bud
```

Or in development:

```bash
cd capabilities/government/bud
pip install -e .
```

---

## 3. Quick Start

```python
from apg_government_bud.service import BudgetManagementService

svc = BudgetManagementService(tenant_id="ke_treasury", actor_id="budget_officer_001")

# Create a vote ceiling
ceiling = svc.create_budget_ceiling(
    programme="Health Services",
    vote="V2101",
    amount=500_000_000.0,
    fiscal_year="2025/2026",
)
vote_id = ceiling["vote_id"]

# Raise a requisition
req = svc.requisition(
    department_id="MOH-HQ",
    amount=10_000_000.0,
    purpose="Medical supplies Q1",
    programme_code="V2101",
)

# Commitment check and record
check = svc.commitment_check(department_id="MOH-HQ", requisition_id=req["id"])
if check["can_commit"]:
    import uuid
    commitment = svc.record_commitment(
        commitment_id=str(uuid.uuid4()).replace("-", ""),
        tenant_id="ke_treasury",
        vote_id=vote_id,
        commitment_type="purchase_order",
        amount=10_000_000.0,
        approval_reference="APR-2025-001",
        supplier_reference="SUP-MEDPRO-001",
        evidence_reference="LPO-2025-001",
    )

# Generate Budget vs Actual
bva = svc.budget_vs_actual(vote_id=vote_id, period="Q1-FY2025/26")
print(f"Absorption: {bva['absorption_rate_pct']}%")
```

---

## 4. Budget Preparation

### 4.1 MTEF Rolling Three-Year Envelopes

```python
import asyncio

envelopes = asyncio.run(svc.mtef_rolling_envelope(
    baseline_year="2025/2026",
    gdp_growth_pct=5.5,
    inflation_pct=7.0,
    deficit_target_pct_gdp=3.0,
    sector_shares={
        "health": 15.0,
        "education": 20.0,
        "infrastructure": 25.0,
        "security": 10.0,
        "general_public_services": 30.0,
    },
))
# envelopes["envelopes"] contains Year 1/2/3 sector ceilings
```

### 4.2 Budget Circular Workflow

Currently handled via `create_budget_ceiling()` and `record_budget()`. Each circular issuance should be logged with an evidence reference corresponding to the gazette notice or official circular number.

### 4.3 Multi-Year Budget Plans

```python
plan = svc.multi_year_budget_plan(
    programme="Road Infrastructure",
    years=["2025/2026", "2026/2027", "2027/2028"],
    allocations={"2025/2026": 200_000_000, "2026/2027": 250_000_000, "2027/2028": 300_000_000},
)
```

### 4.4 Revenue Projection

```python
rev = svc.revenue_projection(
    fiscal_year="2025/2026",
    revenue_streams=[
        {"name": "Income Tax", "projected_amount": 800_000_000},
        {"name": "VAT", "projected_amount": 600_000_000},
        {"name": "Excise Duty", "projected_amount": 150_000_000},
    ],
)
```

---

## 5. Budget Execution

### 5.1 Commitment Control

The commitment control cycle enforces vote balance sufficiency at each stage:

1. `requisition()` — Department raises request against a programme code
2. `commitment_check()` — Verify vote has sufficient balance
3. `record_commitment()` — Reserve funds; debits available balance
4. `payment_approval()` — Finance officer approves payment
5. `record_expenditure()` — Actual payment recorded
6. `commitment_liquidation()` — Close commitment on payment

```python
# Step 4-5-6
payment = svc.payment_approval(
    commitment_id=commitment["id"],
    payment_amount=9_500_000.0,
    approved_by="CFO-001",
)

expenditure = svc.record_expenditure(
    expenditure_id=str(uuid.uuid4()).replace("-", ""),
    tenant_id="ke_treasury",
    commitment_id=commitment["id"],
    expenditure_type="goods",
    amount=9_500_000.0,
    approval_reference=payment["reference"],
    payee_reference="MEDPRO-INV-2025-101",
    evidence_reference="CERT-DELIVERY-001",
)

svc.commitment_liquidation(
    commitment_id=commitment["id"],
    liquidation_amount=9_500_000.0,
)
```

### 5.2 Treasury Single Account Movements

```python
tsa = svc.treasury_single_account(
    movement_type="debit",
    amount=9_500_000.0,
    reference=payment["reference"],
)
```

### 5.3 TSA Reconciliation

```python
recon = asyncio.run(svc.reconcile_tsa_with_expenditures(tolerance=0.01))
print(f"Reconciled: {recon['reconciliation_rate_pct']}%")
if recon["unmatched_count"] > 0:
    print("Unmatched TSA movements:", recon["unmatched"])
```

### 5.4 Budget Revisions

```python
# Supplementary budget
supp = svc.supplementary_budget(
    vote_id=vote_id,
    additional_amount=50_000_000.0,
    reason="Emergency medical supplies surge",
    authority="GAZETTE-SUPP-2025-012",
)

# Virement (reallocation between votes)
transfer = svc.inter_agency_transfer(
    source_vote_id=vote_id,
    target_vote_id=other_vote_id,
    amount=5_000_000.0,
    authority="VIREMENT-AUTH-2025-003",
)
```

---

## 6. Programme-Based Budgeting (PBB)

Link votes to performance indicators and compute composite achievement scores:

```python
scorecard = asyncio.run(svc.pbb_scorecard(
    vote_id=vote_id,
    indicators=[
        {"name": "Hospitals constructed", "category": "output", "target": 10, "actual": 7, "unit": "count"},
        {"name": "Under-5 mortality rate", "category": "outcome", "target": 45, "actual": 38, "unit": "per_1000"},
        {"name": "Health budget released", "category": "input", "target": 500_000_000, "actual": 490_000_000, "unit": "KES"},
    ],
))
print(f"Performance band: {scorecard['performance_band']}")
print(f"Reallocation recommended: {scorecard['reallocation_recommended']}")
```

---

## 7. Fiscal Risk & Contingent Liabilities

### 7.1 Register a Fiscal Risk

```python
risk = asyncio.run(svc.register_fiscal_risk(
    risk_category="public_debt_guarantee",
    description="Government guarantee on parastatals bonds",
    probability=0.25,
    max_exposure=2_000_000_000.0,
    trigger_condition="parastatal_default",
    mitigation_action="credit_enhancement_fund",
))
```

### 7.2 Compute Total Exposure

```python
exposure = asyncio.run(svc.compute_contingent_liability_exposure())
print(f"Expected exposure: KES {exposure['total_expected_exposure']:,.2f}")
```

### 7.3 Stress Testing

```python
stress = asyncio.run(svc.stress_test_budget(scenarios=[
    {"name": "base", "revenue_change_pct": 0, "expenditure_pressure_pct": 0},
    {"name": "revenue_shock_15pct", "revenue_change_pct": -15, "expenditure_pressure_pct": 5},
    {"name": "commodity_crash", "revenue_change_pct": -25, "expenditure_pressure_pct": 10},
]))
breaches = [s for s in stress["scenarios"] if s["breach_flag"]]
```

---

## 8. Inter-Government Fiscal Transfers (IGFT)

```python
allocation = asyncio.run(svc.compute_igft_allocation(
    total_shareable_revenue=600_000_000_000.0,
    units=[
        {"id": "nairobi", "name": "Nairobi", "population": 4_400_000, "poverty_index": 0.28, "land_area_km2": 695},
        {"id": "turkana", "name": "Turkana", "population": 926_000, "poverty_index": 0.72, "land_area_km2": 77_000},
        # ... remaining 45 counties
    ],
    formula_weights={"equal_share": 25.0, "population": 45.0, "poverty_index": 20.0, "land_area": 10.0},
    constitutional_floor_pct=15.0,
))
print(f"Floor met: {allocation['floor_met']}")
```

---

## 9. Arrears Management

### 9.1 Register Arrear

```python
arrear = asyncio.run(svc.register_payment_arrear(
    creditor_id="CONTRACTOR-BLDG-001",
    creditor_class="contractor",
    original_due_date="2025-03-31",
    amount=15_000_000.0,
    penalty_rate_pct=1.5,
    legal_exposure=500_000.0,
))
```

### 9.2 Generate Payment Plan

```python
plan = asyncio.run(svc.generate_arrears_payment_plan(
    available_cash=30_000_000.0,
))
print(f"Items cleared: {plan['items_fully_cleared']}")
print(f"Residual cash: KES {plan['residual_cash']:,.2f}")
```

---

## 10. Expenditure Anomaly Detection

```python
anomalies = asyncio.run(svc.detect_expenditure_anomalies(
    sensitivity=0.85,
    flag_round_numbers=True,
    flag_year_end_spikes=True,
))
high_risk = [a for a in anomalies["anomalies"] if a["suspicion_score"] > 0.6]
```

When `OLLAMA_BASE_URL` is set, a local ML model augments heuristic scores. No expenditure data leaves the Ministry network.

---

## 11. IPSAS Financial Reporting

```python
ipsas = asyncio.run(svc.generate_ipsas_accrual_report(fiscal_period="FY2025/2026"))
print(ipsas["statement_of_financial_performance"])
print(ipsas["statement_of_financial_position"])
```

---

## 12. Parliamentary Estimates

```python
estimates = asyncio.run(svc.generate_parliamentary_estimates(
    fiscal_year="2026/2027",
    include_prior_year_actuals=True,
    include_pbb_scores=True,
))
# Pass estimates to document generation capability for PDF/DOCX output
```

---

## 13. Donor Fund Management

```python
donor_proj = svc.donor_funded_budget(
    project_code="USAID-HEALTH-001",
    donor_id="USAID",
    grant_amount=25_000_000.0,
    conditions="Quarterly financial reports; procurement via donor rules; no co-mingling",
)
```

---

## 14. Fiscal Year Close

```python
close = svc.fiscal_year_close(fiscal_year="2024/2025")
print(f"Votes closed: {close['votes_closed']}")
print(f"Uncommitted balance lapsed: KES {close['uncommitted_balance_lapsed']:,.2f}")
```

---

## 15. Reporting

### Dashboard

```python
dashboard = svc.dashboard_summary(tenant_id="ke_treasury")
```

### Public Finance Report

```python
pfm = svc.public_finance_report(period="Q3-FY2025/26")
print(f"Absorption rate: {pfm['fiscal_summary']['absorption_rate_pct']}%")
```

### Audit Trail

```python
trail = svc.audit_trail_report(tenant_id="ke_treasury", limit=50)
```

### Variance Alerts

```python
alerts = svc.variance_alert(threshold_pct=20.0)
for a in alerts["alerts"]:
    print(f"Vote {a['vote_code']}: {a['variance_pct']}% variance")
```

---

## 16. Streaming Architecture

All write operations emit events. The streaming pipeline uses **bytewax** as the processing engine with **NATS** as the message broker:

```
Service method → audit_event → bytewax pipeline → NATS subject → downstream subscribers
```

Key subjects:
- `apg.government.bud.lifecycle` — core budget lifecycle events
- `apg.government.bud.commitment.{tenant_id}` — real-time commitment events
- `apg.government.bud.mtef` — MTEF envelope changes
- `apg.government.bud.pbb` — PBB scorecard updates
- `apg.government.bud.risk` — fiscal risk registrations
- `apg.government.bud.igft` — IGFT allocations
- `apg.government.bud.anomaly` — expenditure anomaly alerts
- `apg.government.bud.arrears` — arrears registry events
- `apg.government.bud.parliament.submission` — parliamentary estimates packages
- `apg.government.bud.tsa.reconciliation` — TSA reconciliation results

---

## 17. Business Rules Reference

| Rule | Policy Key | Enforced In |
|---|---|---|
| Tenant context required | `tenant_context_present` | All write methods |
| Commitment requires balance | `sufficient_balance` | `record_commitment()` |
| No negative vote balance | `negative_balance` | `record_commitment()` |
| Revision requires treasury notification | `treasury_notification_present` | `record_revision()` |
| Cross-vote transfer requires approval | `approval_present` | `inter_agency_transfer()` |
| IGFT must meet constitutional floor | `floor_met` | `compute_igft_allocation()` |
| Agent action requires human approval | `human_approval_recorded` | `validate_agent_action()` |
| Batch must route to bytewax | `event_stream=bytewax` | `validate_batch()` |

---

## 18. Composability

Reference this capability in `.apg` source files:

```apg
use government_bud;
```

Downstream capabilities that consume `government_bud` events:
- `intel_alerts` — variance and anomaly alerts trigger operational alerts
- `intel_dashboard` — fiscal KPIs feed executive dashboards
- `government_con` — contract award triggers commitment recording
- `government_tax` — revenue receipts update AIA vote balances
- `government_csr` — citizen service receipts update vote accounts

---

## 19. Further Reading

- `service.py` — Complete business logic implementation
- `models.py` — Dataclass models
- `capability_contract.py` — Policy rules and supported values
- `api.py` — REST API endpoint definitions
- `views.py` — Flask-AppBuilder views and Pydantic request/response schemas
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 planned capability enhancements
- `SPECIFICATION.md` — Detailed capability specification
