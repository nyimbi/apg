# Supplier Relationship Management User Guide

## Overview

`scm_srm` manages the full supplier lifecycle: onboarding and approval, periodic scorecard measurement, risk assessment, collaboration messaging, formal performance reviews, preferred supplier designation, certification tracking, ESG scoring, contract lifecycle management, structured development plans, portfolio segmentation, peer benchmarking, escalation management, and portfolio-level risk heatmaps.

## Supplier Status Flow

```
pending_approval → [onboarding in_progress] → active → probation
                                                      → suspended
                                                      → blacklisted
                                                      → inactive
```

## Key Use Cases

- **Supplier onboarding**: Register → start structured checklist onboarding → approve once checklist is 100% complete.
- **Scorecard**: Quarterly multi-dimension scoring (quality, delivery, responsiveness, cost, sustainability) with trend analysis.
- **Risk management**: Record financial, geopolitical, operational, compliance, ESG, and concentration risks; portfolio heatmap for board reporting.
- **Collaboration portal**: Send forecasts, PO updates, complaints, and escalations directly to supplier contacts.
- **Performance reviews**: Formal periodic reviews with action items and next review scheduling.
- **Preferred suppliers**: Grant/revoke preferred status with rationale; filter sourcing to preferred list.
- **Certifications**: Track ISO, halal, organic, fair-trade, and custom certs with expiry dates.
- **ESG scoring**: E/S/G sub-scores with weighted composite; evidence URL trail; period-by-period history.
- **Contract lifecycle**: Register contracts, track values, auto-renew flags, notice periods; query expiring within N days.
- **Development plans**: Structured remediation for underperforming suppliers with milestones, budget, and target score.
- **Segmentation**: Kraljic-style 2×2 matrix (risk_score), spend category, or geography strategies.
- **Benchmarking**: Compare a supplier's latest scorecard against named peers; surfaces delta per dimension.
- **Escalation management**: Raise and resolve formal escalations with severity and due dates.
- **Risk heatmap**: Category × severity matrix with hotspot ranking for portfolio risk overview.

---

## API Reference

### Create Supplier

```
POST /api/scm/srm/suppliers
{
  "tenant_id": "acme",
  "name": "Acme Packaging Ltd",
  "supplier_code": "PKG-001",
  "country": "KE",
  "category": "packaging",
  "payment_terms": "NET45"
}
```

### Start Onboarding

```
POST /api/scm/srm/onboarding
{
  "tenant_id": "acme",
  "supplier_id": "supp-xyz",
  "assigned_to": "procurement.manager"
}
```

Response includes a `checklist` array. Advance items with:

```
PUT /api/scm/srm/onboarding/{onboarding_id}/items/0
{ "tenant_id": "acme" }
```

### Approve Supplier (requires completed onboarding)

```
POST /api/scm/srm/suppliers/{id}/approve
{ "tenant_id": "acme", "approved_by": "cpo@acme.com" }
```

### Create Scorecard

```
POST /api/scm/srm/scorecards
{
  "tenant_id": "acme",
  "supplier_id": "supp-xyz",
  "period": "2026-Q2",
  "quality_score": 8.5,
  "delivery_score": 9.0,
  "responsiveness_score": 7.5,
  "cost_score": 8.0,
  "sustainability_score": 7.0,
  "reviewed_by": "procurement.manager"
}
```

### Scorecard Trend

```
GET /api/scm/srm/scorecards/{supplier_id}/trend?dimension=quality_score&tenant_id=acme
```

Returns `series` (period → value) and `trend` (improving | declining | stable | insufficient_data).

### Record ESG Score

```
POST /api/scm/srm/esg-scores
{
  "tenant_id": "acme",
  "supplier_id": "supp-xyz",
  "period": "2026-Q2",
  "environmental_score": 7.5,
  "social_score": 8.0,
  "governance_score": 8.5,
  "assessed_by": "esg.team",
  "evidence_urls": ["https://docs.acme.com/esg/supp-xyz-2026-Q2.pdf"]
}
```

Composite = `E×0.4 + S×0.3 + G×0.3 = 7.85`.

### Register Contract

```
POST /api/scm/srm/contracts
{
  "tenant_id": "acme",
  "supplier_id": "supp-xyz",
  "contract_reference": "MSA-2024-001",
  "contract_type": "master_supply_agreement",
  "start_date": "2024-01-01",
  "end_date": "2027-01-01",
  "value": 1200000,
  "currency": "USD",
  "auto_renew": false,
  "notice_period_days": 90
}
```

Query contracts expiring within 90 days:

```
GET /api/scm/srm/contracts?expiring_within_days=90&tenant_id=acme
```

### Create Development Plan

```
POST /api/scm/srm/development-plans
{
  "tenant_id": "acme",
  "supplier_id": "supp-xyz",
  "plan_title": "Delivery Reliability Improvement",
  "objectives": [
    "Implement EDI PO acknowledgement within 24h",
    "Reduce late deliveries from 18% to <5%"
  ],
  "target_score": 9.0,
  "target_date": "2026-12-31",
  "assigned_to": "supplier.dev.manager",
  "budget": 25000,
  "currency": "USD"
}
```

Update progress:

```
PUT /api/scm/srm/development-plans/{plan_id}/progress
{
  "tenant_id": "acme",
  "progress_pct": 50,
  "milestone_note": "EDI integration completed and tested"
}
```

### Segment Suppliers

```
GET /api/scm/srm/suppliers/segment?strategy=risk_score&tenant_id=acme
```

Returns four segments: `strategic`, `leverage`, `bottleneck`, `non_critical`.

Alternative strategies: `spend_category`, `geography`.

### Benchmark Supplier

```
POST /api/scm/srm/suppliers/{id}/benchmark
{
  "tenant_id": "acme",
  "peer_supplier_ids": ["supp-aaa", "supp-bbb", "supp-ccc"]
}
```

Returns delta vs peer mean per dimension. Negative delta = underperforming vs peers.

### Concentration Risk Report

```
GET /api/scm/srm/suppliers/concentration-risk?threshold_pct=40&tenant_id=acme
```

Identifies single-source categories and geographic concentration.

### Risk Heatmap

```
GET /api/scm/srm/risk-heatmap?tenant_id=acme
```

Returns category × risk_level count matrix plus sorted hotspots list.

### Raise Escalation

```
POST /api/scm/srm/escalations
{
  "tenant_id": "acme",
  "supplier_id": "supp-xyz",
  "title": "Repeated late shipments Q2 2026",
  "description": "Three consecutive late deliveries exceeding 5-day SLA",
  "severity": "high",
  "raised_by": "ops.manager",
  "due_date": "2026-07-15"
}
```

Resolve:

```
POST /api/scm/srm/escalations/{escalation_id}/resolve
{
  "tenant_id": "acme",
  "resolution": "Supplier committed to dedicated production line; KPI review in 30 days",
  "resolved_by": "cpo@acme.com"
}
```

---

## Data Model Summary

| Record Type | Key Fields |
|-------------|------------|
| scm_srm_supplier | id, name, supplier_code, country, category, status, preferred, risk_level, overall_score, esg_composite |
| scm_srm_scorecard | id, supplier_id, period, quality/delivery/responsiveness/cost/sustainability/overall scores |
| scm_srm_risk_assessment | id, supplier_id, risk_category, risk_level, mitigation_plan, status |
| scm_srm_esg_score | id, supplier_id, period, E/S/G sub-scores, composite_score, evidence_urls |
| scm_srm_contract | id, supplier_id, contract_reference, start/end dates, value, auto_renew, notice_period_days |
| scm_srm_development_plan | id, supplier_id, objectives, target_score, progress_pct, milestones |
| scm_srm_escalation | id, supplier_id, title, severity, status, resolution |
| scm_srm_onboarding | id, supplier_id, checklist, completion_pct, status |
| scm_srm_certification | id, supplier_id, cert_type, valid_from, valid_until |
| scm_srm_collaboration_message | id, supplier_id, subject, message_type, status |
| scm_srm_performance_review | id, supplier_id, review_period, action_items, next_review_date |
