# Grant Management (ngo_grn)

Grant pipeline, proposal management, budget tracking, disbursement, compliance reporting, audits.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/ngo/grn/health` | Service health check |
| GET | `/api/ngo/grn/` | List grants |
| POST | `/api/ngo/grn/` | Create grant |
| GET | `/api/ngo/grn/<id>` | Get grant |
| PUT | `/api/ngo/grn/<id>` | Update grant |
| DELETE | `/api/ngo/grn/<id>` | Delete grant |
| POST | `/api/ngo/grn/<id>/activate` | Activate grant |
| POST | `/api/ngo/grn/<id>/close` | Close grant |
| GET | `/api/ngo/grn/<id>/proposals` | List proposals |
| POST | `/api/ngo/grn/<id>/proposals` | Submit proposal |
| GET | `/api/ngo/grn/<id>/budget-lines` | List budget lines |
| POST | `/api/ngo/grn/<id>/budget-lines` | Create budget line |
| GET | `/api/ngo/grn/<id>/disbursements` | List disbursements |
| POST | `/api/ngo/grn/<id>/disbursements` | Record disbursement |
| GET | `/api/ngo/grn/<id>/compliance-reports` | List compliance reports |
| POST | `/api/ngo/grn/<id>/compliance-reports` | Submit compliance report |
| GET | `/api/ngo/grn/<id>/audit-findings` | List audit findings |
| POST | `/api/ngo/grn/<id>/audit-findings` | Record audit finding |
| GET | `/api/ngo/grn/<id>/summary` | Donor-facing summary |
| GET | `/api/ngo/grn/portfolio/summary` | Portfolio summary |
| GET | `/api/ngo/grn/audit-events` | Audit event log |

---

## World-Class Enhancements (v2.0)

15 targeted improvements elevating ngo_grn to competitive grant intelligence on par with Fluxx, Submittable, and Salesforce NPSP.

**I1. AI-Powered Grant Eligibility Scoring** — Weighted 0–100 fit score per grant using sector match, country alignment, donor range, and relationship history to eliminate low-probability proposal effort. [AI/ML]

**I2. Deadline Calendar & Reporting Obligation Tracker** — Structured obligation store with configurable advance-warning periods; `get_upcoming_obligations(days_ahead)` returns items sorted by urgency with `days_remaining`. [Feature]

**I3. Multi-Currency Disbursement with Real-Time FX Tracking** — Attaches `exchange_rate`, `base_currency`, `base_amount` to disbursements; `get_fx_variance_report(grant_id)` surfaces phantom variances between contract and actual FX. [Feature]

**I4. Budget Revision Workflow with Version History** — Immutable audit trail for budget modifications; requires `justification` for >10% line increases per EU PRAG/USAID ADS 303; `get_budget_revision_history(line_id)`. [Compliance]

**I5. Burn Rate Analysis & Underspend Forecasting** — Projects `forecast_spend_at_close` from daily burn rate; raises `underspend_risk` flag when forecast < 90% of budget at grant close. [AI/ML]

**I6. Donor CRM Integration Points** — `get_donor_grant_history(donor_id)` returns aggregated win rate, total awarded, average duration, and open grant count; joins with ngo_crm via shared `donor_id`. [Integration]

**I7. Compliance Risk Score per Grant** — Weighted risk score (critical findings ×40, overdue reports ×15, anomalies ×10); `get_compliance_risk_score(grant_id)` returns score, tier, and contributing_factors. [Compliance]

**I8. Programmatic Sub-Grant Tracking** — `create_sub_grant` validates amount <= undisbursed parent balance; `get_sub_grant_tree` returns nested prime + sub-grant hierarchy with aggregated financials per 2 CFR 200.332. [Feature]

**I9. Automated No-Cost Extension Request Workflow** — `request_no_cost_extension` validates active grant within 90 days of end; `approve_nce` updates grant end_date with full status trail. [Feature]

**I10. Disbursement Anomaly Detection** — On `create_disbursement`, blocks duplicate references and post-end-date payments; warns on amounts >3x grant average or payment method changes; returns `anomaly_flags`. [Security]

**I11. Grant Reporting Template Library** — Pre-seeded USAID, EU, GIZ, and DFID formats; `get_report_template(donor_type, report_type)` returns structured field list; `create_compliance_report` validates required fields against template. [UX]

**I12. Portfolio Sector & Country Heat Map Data** — `get_portfolio_heat_map()` returns concentration stats by sector and country; flags any dimension exceeding 40% of portfolio as `concentration_risk: true`. [Feature]

**I13. Milestone & Deliverable Tracking** — `_milestones` store with disbursement gating; `complete_milestone(milestone_id, completed_by, evidence_url)` gates linked disbursement release per EU/World Bank instruments. [Feature]

**I14. Donor-Specific Compliance Rule Engine** — Configurable rule store per donor pattern (USAID, EU, DFID, Gates); `check_compliance_rules(grant_id)` evaluates overhead caps, procurement rules, and reporting frequencies; returns `violations` and `compliant`. [Compliance]

**I15. Automated Narrative Report Generation** — `generate_narrative_draft(grant_id, report_type, period_start, period_end)` assembles disbursement totals, budget utilisation, and milestones into LLM-ready context; returns structured `draft_sections` for human review. [AI/ML]

---

## New Methods

Three high-impact async methods added in v2.0:

### `get_burn_rate_analysis(grant_id)`

Real-time underspend forecasting. Call 60–90 days before grant close to catch reallocation windows before donor clawback.

```python
svc = GrantManagementService(tenant_id="ke-ngo")
analysis = await svc.get_burn_rate_analysis("grn_abc123")
# {
#   "grant_id": "grn_abc123",
#   "daily_burn_rate": 1250.00,
#   "forecast_spend_at_close": 87500.00,
#   "budget_total": 100000.00,
#   "underspend_risk": true,
#   "forecast_utilisation_pct": 87.5,
#   "days_remaining": 14
# }
```

### `get_compliance_risk_score(grant_id)`

Exception-triage scoring for compliance officers managing 20+ active grants. Weighted formula: `critical_findings×40 + high_findings×20 + medium_findings×5 + overdue_reports×15 + disbursement_anomalies×10`, clamped 0–100.

```python
risk = await svc.get_compliance_risk_score("grn_abc123")
# {
#   "grant_id": "grn_abc123",
#   "score": 65,
#   "tier": "high",
#   "contributing_factors": [
#     {"factor": "overdue_reports", "count": 2, "weight": 30},
#     {"factor": "high_findings", "count": 1, "weight": 20},
#     {"factor": "disbursement_anomalies", "count": 1, "weight": 10}
#   ]
# }
```

### `get_sub_grant_tree(grant_id)`

Returns full nested hierarchy of prime grant + sub-grants with aggregated financials. Required for USAID 2 CFR 200.332 pass-through reporting.

```python
tree = await svc.get_sub_grant_tree("grn_prime001")
# {
#   "grant_id": "grn_prime001",
#   "title": "USAID Resilience Program",
#   "amount": 500000.00,
#   "sub_grants": [
#     {
#       "grant_id": "grn_sub001",
#       "title": "County Implementation - Kisumu",
#       "amount": 120000.00,
#       "disbursed": 45000.00,
#       "sub_grants": []
#     }
#   ],
#   "total_sub_grant_committed": 120000.00,
#   "undisbursed_balance": 380000.00
# }
```

---

## Capability Composition

| Keyword | Composes With | Purpose |
|---------|---------------|---------|
| `donor_id` | `ngo_crm` | Unified donor + grant relationship view |
| `sub_grant` | `ngo_grn` (self) | Pass-through hierarchy |
| `milestone_id` | `ngo_prg` (programmes) | Disbursement gating by deliverable |
| `compliance_rule` | `ngo_aud` (audit) | Shared rule evaluation |

---

© 2025 Datacraft — www.datacraft.co.ke
