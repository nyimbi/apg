# County / Devolved Services (gov_cty)

County revenue collection, permit issuance, social welfare, devolved health, public works ticketing.

## Overview

End-to-end county government service management: collect rates and fees, issue business/building permits, manage social welfare programmes, run devolved health facilities, and track public works maintenance tickets. Multi-tenant, designed for Kenya's 47 counties.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/government/cty/health | Service health check |
| GET | /api/government/cty/dashboard | Dashboard metrics |
| GET | /api/government/cty/revenues | List revenues |
| GET | /api/government/cty/revenues/{id} | Get revenue record |
| POST | /api/government/cty/revenues | Collect revenue |
| POST | /api/government/cty/revenues/{id}/confirm | Confirm payment |
| GET | /api/government/cty/revenues/summary | Revenue summary |
| GET | /api/government/cty/permits | List permits |
| GET | /api/government/cty/permits/{id} | Get permit |
| POST | /api/government/cty/permits | Apply for permit |
| PUT | /api/government/cty/permits/{id} | Update permit |
| DELETE | /api/government/cty/permits/{id} | Delete permit |
| GET | /api/government/cty/welfare | List welfare applications |
| GET | /api/government/cty/welfare/{id} | Get application |
| POST | /api/government/cty/welfare | Apply for welfare |
| PUT | /api/government/cty/welfare/{id} | Update application |
| GET | /api/government/cty/health-facilities | List facilities |
| POST | /api/government/cty/health-facilities | Register facility |
| POST | /api/government/cty/patients | Register patient |
| GET | /api/government/cty/tickets | List tickets |
| GET | /api/government/cty/tickets/{id} | Get ticket |
| POST | /api/government/cty/tickets | Create ticket |
| PUT | /api/government/cty/tickets/{id} | Update ticket |
| DELETE | /api/government/cty/tickets/{id} | Delete ticket |
| GET | /api/government/cty/audit-events | List audit events |

## World-Class Enhancements (v2.0)

**I1. Permit Renewal Workflow** — `renew_permit()` clones issued permits with linked predecessor and reset dates, eliminating re-issue from scratch [Lifecycle Management]

**I2. Welfare Payment Disbursement Ledger** — `disburse_welfare_payment()` records actual cash-out via mobile_money/bank/cash with cumulative `total_disbursed_kes` per application [Financial Accuracy]

**I3. Revenue Reconciliation & Variance Report** — `reconcile_revenue()` compares confirmed collections against budget allocations, returning variance, coverage ratio, and per-type shortfall [Fiscal Accountability]

**I4. Permit Inspection Scheduling** — `schedule_permit_inspection()` / `record_inspection_outcome()` enforce mandatory site visits before issuance with inspector assignment, checklist, and photo evidence [Compliance Enforcement]

**I5. Health Facility Bed Census** — `update_facility_bed_census()` / `get_bed_census_report()` provide real-time available/occupied/reserved bed tracking across facilities [Resource Allocation]

**I6. Citizen Feedback & Satisfaction Scoring** — `submit_citizen_feedback()` on closed tickets (rating 1–5) and `calculate_service_satisfaction_score()` aggregated per sub-county [Service Quality]

**I7. Market Stall Fee Collection Cycle** — `collect_stall_fee()` / `get_stall_arrears()` / `list_stalls_in_arrears()` close the gap between `monthly_fee_kes` and actual payment tracking [Revenue Completeness]

**I8. County Budget Expenditure Tracking** — `record_expenditure()` with LPO/invoice reference and `get_budget_utilization()` returning burn rate and projected year-end status [Fiscal Control]

**I9. Contractor Performance Scoring** — `rate_contractor_performance()` per completed ticket and `get_contractor_scorecard()` aggregating average score, jobs completed, and complaints [Procurement Integrity]

**I10. Ward-Level Service Delivery Analytics** — `ward_equity_report()` returns per-ward revenue, permits, welfare approvals, open tickets, and composite equity index [Equity Monitoring]

**I11. Automated Permit Expiry Notifications Queue** — `get_expiring_permits(days)` with contact details and `mark_expiry_notification_sent()` for outreach tracking [Proactive Compliance]

**I12. Bulk Revenue Import from M-Pesa Paybill** — `bulk_import_revenues()` ingests batch Safaricom paybill files with deduplication by receipt number and import summary [Interoperability]

**I13. Public Expenditure Transparency Feed** — `get_transparency_feed()` returns sanitised, PII-free expenditure and budget data suitable for county open data portal publication [Open Government]

**I14. Health Referral Chain Management** — `create_health_referral()` / `update_referral_status()` track patient transfers between facilities with urgency, clinical reason, and acceptance state [Care Continuity]

**I15. Devolved Environmental Inspection Registry** — `register_environmental_inspection()` / `list_environmental_violations()` record EIA compliance checks on permitted sites for NEMA audit readiness [Regulatory Completeness]

## New Methods

Three high-impact v2.0 additions covering fiscal control, equity analytics, and revenue interoperability:

### `reconcile_revenue()` — Fiscal Accountability (I3)

Compare collected revenue against budget targets for a period and detect shortfalls early.

```python
from capabilities.government.cty.service import CountyService

svc = CountyService(tenant_id="nairobi")

result = await svc.reconcile_revenue(
    tenant_id="nairobi",
    period="2025-Q1",
    budget_allocation_kes=50_000_000.0,
)
# {
#   "period": "2025-Q1",
#   "collected_kes": 43_200_000.0,
#   "budget_kes": 50_000_000.0,
#   "variance_kes": -6_800_000.0,
#   "coverage_ratio": 0.864,
#   "shortfall_by_type": {"land_rates": -3_200_000.0, "market_fees": -1_100_000.0},
# }
```

### `ward_equity_report()` — Equity Monitoring (I10)

Per-ward breakdown of all service delivery metrics plus a composite equity index, satisfying Ward Development Fund reporting requirements.

```python
report = await svc.ward_equity_report(tenant_id="nairobi", budget_year=2025)
# [
#   {
#     "ward": "Westlands",
#     "revenue_collected_kes": 12_400_000.0,
#     "permits_issued": 87,
#     "welfare_approved": 143,
#     "open_tickets": 22,
#     "equity_index": 0.73,
#   },
#   ...
# ]
```

### `bulk_import_revenues()` — Interoperability (I12)

Ingest a Safaricom Daraja paybill settlement batch; deduplicates by receipt number and returns a structured import summary.

```python
paybill_records = [
    {"receipt_number": "QGH7X2K3L1", "amount_kes": 4500.0, "payer_phone": "0712345678",
     "revenue_type": "single_business_permit", "payment_date": "2025-03-31"},
    {"receipt_number": "QGH7X2K3L1", "amount_kes": 4500.0, ...},  # duplicate — skipped
    {"receipt_number": "RJK9M4N8P2", "amount_kes": 12000.0, "payer_phone": "0723456789",
     "revenue_type": "land_rates", "payment_date": "2025-03-31"},
]

summary = await svc.bulk_import_revenues(records=paybill_records, tenant_id="nairobi")
# {
#   "total_submitted": 3,
#   "imported": 2,
#   "duplicates_skipped": 1,
#   "errors": [],
#   "total_amount_kes": 16500.0,
# }
```
