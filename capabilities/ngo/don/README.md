# Donor Relationship Management (ngo_don)

Donor registry, communication history, pledge tracking, receipt generation, stewardship plans.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/ngo/don/health` | Service health check |
| GET | `/api/ngo/don/` | List donors |
| POST | `/api/ngo/don/` | Create donor |
| GET | `/api/ngo/don/<id>` | Get donor |
| PUT | `/api/ngo/don/<id>` | Update donor |
| DELETE | `/api/ngo/don/<id>` | Deactivate donor |
| GET | `/api/ngo/don/search?q=` | Search donors |
| GET | `/api/ngo/don/<id>/communications` | Communication history |
| POST | `/api/ngo/don/<id>/communications` | Log communication |
| GET | `/api/ngo/don/<id>/pledges` | List pledges |
| POST | `/api/ngo/don/<id>/pledges` | Create pledge |
| GET | `/api/ngo/don/<id>/receipts` | List receipts |
| POST | `/api/ngo/don/<id>/receipts` | Generate receipt |
| GET | `/api/ngo/don/<id>/stewardship` | Stewardship plans |
| POST | `/api/ngo/don/<id>/stewardship` | Create stewardship plan |
| GET | `/api/ngo/don/<id>/history` | Giving history |
| GET | `/api/ngo/don/portfolio/summary` | Portfolio summary |
| GET | `/api/ngo/don/portfolio/retention` | Retention analysis |
| GET | `/api/ngo/don/pledges/overdue` | Overdue pledges |
| GET | `/api/ngo/don/audit-events` | Audit log |

## World-Class Enhancements (v2.0)

Fifteen improvements that elevate ngo_don from functional CRUD to an AI-augmented donor intelligence platform competitive with Salesforce NPSP, Blackbaud Raiser's Edge, and Bloomerang.

**I1. AI-Powered Donor Propensity Scoring** — RFM + engagement-velocity score (0–100) per donor via `score_donor()` / `bulk_rescore()`; triages large portfolios to the highest-revenue relationships [AI/ML]

**I2. Lapsed-Donor Win-Back Campaign Engine** — `compute_lapse_risk()` segments donors at 90/180/365-day silence thresholds; `get_winback_candidates()` returns cadence-aware re-engagement lists [Feature]

**I3. Recurring-Gift Scheduling and Auto-Fulfillment Tracking** — `schedule_recurring_pledge()` generates forward-dated instalments; `advance_instalment()` auto-receipts and projects the next due date [Feature]

**I4. Soft-Credit and Household Attribution** — `link_household()` groups records; `soft_credit()` distributes receipt value to advisors/household members; `donor_giving_history()` includes hard + soft totals [Feature]

**I5. Tax-Deductibility Compliance Engine** — `generate_annual_tax_certificate()` applies KE/US jurisdiction rules; `bulk_issue_tax_certificates()` runs concurrently via `asyncio.gather` [Compliance]

**I6. Donor Portal Self-Service Token Generation** — `generate_portal_token()` issues 24 h signed tokens scoped to one donor; `validate_portal_token()` returns a read-only view with usage audit [UX/Security]

**I7. Duplicate-Donor Detection and Merge** — `find_duplicate_candidates()` scores pairs (name + email + phone fuzzy); `merge_donors()` re-parents all child records and soft-deletes the duplicate [Data Quality]

**I8. Multi-Currency Pledge Reporting with FX Normalization** — `set_fx_rate()` stores date-stamped rates; `portfolio_summary()` and `donor_giving_history()` accept a `reporting_currency` parameter [Compliance/Finance]

**I9. Stewardship Touchpoint Compliance Dashboard** — `stewardship_compliance_report()` ranks at-risk plans by completion rate; `touchpoints_due_this_month()` surfaces upcoming required contacts [Feature/UX]

**I10. Donation Impact Reporting Linkage** — `link_donation_to_impact()` associates a receipt with `programme_id` + `impact_metric`; `donor_impact_statement()` aggregates outcomes for personalised reports [Feature/Integration]

**I11. GDPR / Data-Privacy Consent Management** — `record_consent()` stores immutable channel-level consent events; `get_current_consent()` replays the log; `log_communication()` enforces channel consent [Compliance/Security]

**I12. Automated Receipt Delivery via Notification Hub** — `queue_receipt_delivery()` writes a delivery job (email/WhatsApp); integrates with APG `ngo_msg` event bus; tracks delivery status on the receipt record [Integration/UX]

**I13. Pledge Reminder Escalation Workflow** — `get_pledge_reminder_schedule()` computes next reminder and escalation owner by days-overdue tier; `record_pledge_reminder_sent()` logs each reminder [Feature/Automation]

**I14. Donor Lifecycle Stage Classification** — `classify_lifecycle_stage()` derives stage from cumulative giving + streak + years-active; `get_upgrade_candidates()` returns donors near next-tier threshold with ask amount [AI/ML]

**I15. Board-Ready Giving Trend Export** — `generate_trend_report()` returns monthly receipt totals with YoY/MoM deltas by donor type; serialisable to CSV/JSON or passed to `fin_rpt` capability [Feature/Compliance]

## New Methods

### `score_donor` / `bulk_rescore` — Propensity scoring

```python
from capabilities.ngo.don.service import DonorRelationshipService

svc = DonorRelationshipService(tenant_id="ke_wildlife_trust")

# Score a single donor (0–100, higher = higher likelihood to give)
score = await svc.score_donor("don_01j...")
# {"donor_id": "don_01j...", "score": 82, "recency": 95, "frequency": 78, "monetary": 71, "engagement": 84}

# Nightly batch rescore of the full portfolio
result = await svc.bulk_rescore()
# {"rescored": 3142, "duration_ms": 1840}
```

### `compute_lapse_risk` + `get_winback_candidates` — Win-back segmentation

```python
# Tag every donor with a lapse risk level based on last-receipt date
await svc.compute_lapse_risk()

# Retrieve segmented candidates for re-engagement outreach
candidates = await svc.get_winback_candidates(risk_levels=["high", "critical"])
# [
#   {"donor_id": "...", "lapse_risk": "critical", "days_silent": 412,
#    "last_gift_amount": "5000.00", "recommended_cadence": "personal_call"},
#   ...
# ]
```

### `generate_annual_tax_certificate` — Compliance receipt generation

```python
from decimal import Decimal

# Issue a single tax certificate for fiscal year 2025 (KE jurisdiction by default)
cert = await svc.generate_annual_tax_certificate(
    donor_id="don_01j...",
    fiscal_year=2025,
    jurisdiction="KE",
)
# {"certificate_id": "cert_...", "donor_id": "...", "total_deductible": "125000.00",
#  "currency": "KES", "issued_at": "2026-01-28T09:00:00Z", "receipts_included": 7}

# Batch-issue for all active donors in parallel
summary = await svc.bulk_issue_tax_certificates(fiscal_year=2025, jurisdiction="KE")
# {"issued": 284, "skipped": 12, "errors": 0}
```

## Core Service Methods (v1)

| Method | Signature | Description |
|--------|-----------|-------------|
| `health_check` | `() -> dict` | Liveness + store stats |
| `create_donor` | `(name, email?, ...) -> dict` | Create donor record |
| `list_donors` | `(status?, donor_type?) -> list` | Filtered donor list |
| `search_donors` | `(query) -> list` | Full-text search across name/email/phone |
| `create_pledge` | `(donor_id, amount, currency, ...) -> dict` | Create pledge |
| `generate_receipt` | `(donor_id, pledge_id?, amount, ...) -> dict` | Generate and number receipt |
| `overdue_pledges` | `() -> list` | All pledges past due date |
| `portfolio_summary` | `() -> dict` | Aggregate totals, donor counts, YTD |
| `retention_analysis` | `() -> dict` | Cohort retention rates |
| `create_stewardship_plan` | `(donor_id, tier, frequency, ...) -> dict` | Stewardship plan |
| `record_stewardship_touchpoint` | `(plan_id, notes?) -> dict` | Log a completed touchpoint |
| `donor_giving_history` | `(donor_id) -> dict` | Full gift history with totals |
| `bulk_import_donors` | `(donors: list[dict]) -> dict` | Batch import with dedup |

## Data Models

Core tables (prefix `ngo_`): `ngo_donor`, `ngo_communication`, `ngo_pledge`, `ngo_receipt`, `ngo_stewardship_plan`, `ngo_stewardship_touchpoint`, `ngo_audit_event`.

v2.0 tables added by enhancements: `ngo_soft_credit`, `ngo_portal_token`, `ngo_fx_rate`, `ngo_impact_link`, `ngo_consent_event`, `ngo_receipt_delivery_job`.

## Composability

| Downstream capability | Integration point |
|-----------------------|-------------------|
| `ngo_msg` | Receipt delivery jobs, win-back campaign triggers |
| `fin_rpt` | `generate_trend_report()` output, FX-normalised portfolio data |
| `ngo_prog` | Impact linkage via `programme_id` in `link_donation_to_impact()` |
| `fin_tax` | Annual tax certificate payloads |

## Dependencies

- Python 3.12+, `pydantic>=2`, `uuid6`
- PostgreSQL 15+ (primary store)
- APG event bus (optional — gracefully degraded if absent)
