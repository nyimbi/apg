# Lease Management

## Overview
Full lease lifecycle from heads of terms through abstraction, activation, rent escalation, option tracking, IFRS 16/ASC 842 schedule generation, rent reviews, assignments, and expiry pipeline management. AI-assisted abstraction with mandatory human verification before activation.

Includes advanced analytics: full amortisation schedules, multi-year rent escalation projections, ERV benchmarking, holding-over detection, break option cost modelling, discount rate sensitivity analysis, lease KPI dashboards, and data integrity validation.

## Capability ID
`realestate_lea`

## Provides
- `lease_abstraction_engine`: AI-assisted extraction of key lease terms with human verification
- `rent_escalation_scheduler`: Fixed %, CPI-linked, ratchet, open market, and stepped escalations
- `lease_option_tracker`: Break, renewal, purchase, expansion, and contraction options with notice alerts
- `ifrs16_asc842_compliance`: Present-value ROU asset and lease liability amortisation schedules
- `full_amortisation_schedule`: Complete period-by-period or annual/quarterly amortisation table
- `rent_escalation_projection`: Multi-year rent projection with CPI forecast and stepped schedules
- `lease_expiry_pipeline`: Rolling expiry dashboard with holding-over detection and transition
- `rent_review_workflow`: Upward-only, indexed, and open-market reviews with backdating controls
- `erv_benchmarking`: Over/under-rented analysis, reversion potential, and time-to-reversion
- `break_option_modelling`: NPV break vs. stay analysis with penalty, dilaps, and relocation costs
- `discount_rate_sensitivity`: Lease liability sensitivity matrix across a rate range
- `lease_kpi_dashboard`: Occupancy rate, WAULT, rent collection efficiency, and balance sheet KPIs
- `data_integrity_validation`: Cross-record consistency checks across reviews, options, and subleases
- `lease_assignment_management`: Assignment and subletting with landlord consent enforcement
- `lease_renewal_workflow`: Investment committee escalation for major renewals

## Requires
| Capability | Reason |
|-----------|--------|
| `auth` | Approval authority for activation and reviews |
| `audl` | Immutable audit for rent changes |
| `mten` | Multi-tenant isolation |
| `conf` | Lease policy configuration |
| `ntfy` | Option expiry and lease expiry alerts |
| `wflo` | Review and assignment approval workflows |
| `nlpc` | AI-assisted lease abstraction |
| `comp` | IFRS 16 / ASC 842 compliance guardrails |
| `mqeb` | Publish lease events |
| `schd` | Schedule escalation trigger dates |

## Configuration
| Key | Default | Description |
|-----|---------|-------------|
| `options.early_warning_days` | 180 | Days before exercise window to alert |
| `abstractions.ai_assisted` | true | Use NLP for abstraction |
| `ifrs16.asc842_categories` | 4 categories | Supported lease categories |
| `rent_reviews.notice_required_days` | 30 | Minimum notice before review |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|-----------|
| `/realestate/lea/leases` | GET/POST | List/create leases | `leases` |
| `/realestate/lea/leases/<id>/activate` | POST | Activate lease | `leases` |
| `/realestate/lea/leases/<id>/surrender` | POST | Surrender lease | `leases` |
| `/realestate/lea/leases/<id>/amortisation` | GET | Full amortisation schedule | `leases` |
| `/realestate/lea/leases/<id>/escalation-projection` | GET | Multi-year rent projection | `leases` |
| `/realestate/lea/leases/<id>/erv-benchmark` | POST | ERV benchmarking | `leases` |
| `/realestate/lea/leases/<id>/break-cost` | POST | Break option cost model | `leases` |
| `/realestate/lea/leases/<id>/rate-sensitivity` | GET | Discount rate sensitivity | `leases` |
| `/realestate/lea/leases/<id>/integrity` | GET | Data integrity check | `leases` |
| `/realestate/lea/abstraction` | POST | Create abstraction | `abstraction` |
| `/realestate/lea/abstraction/<id>/verify` | POST | Verify abstraction | `abstraction` |
| `/realestate/lea/escalations` | GET/POST | Escalations | `escalations` |
| `/realestate/lea/escalations/<id>/apply` | POST | Apply escalation | `escalations` |
| `/realestate/lea/options` | POST | Create option | `options` |
| `/realestate/lea/options/<id>/exercise` | POST | Exercise option | `options` |
| `/realestate/lea/options/expiring` | GET | Expiring options | `options` |
| `/realestate/lea/ifrs16` | POST | Generate schedule | `ifrs16` |
| `/realestate/lea/ifrs16/<id>/reclassify` | POST | Reclassify (auditor) | `ifrs16` |
| `/realestate/lea/expiry` | GET | Expiry pipeline | `view` |
| `/realestate/lea/holding-over` | GET | Detect holding-over leases | `leases` |
| `/realestate/lea/analytics/kpi` | GET | Lease KPI dashboard | `analytics` |
| `/realestate/lea/analytics/portfolio` | GET | Portfolio analytics | `analytics` |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| `activation_requires_verified_abstraction` | not verified | deny |
| `escalation_type_supported` | unsupported type | deny |
| `option_exercise_requires_notice` | notice not served | deny |
| `option_exercise_window_required` | outside window | deny |
| `ifrs16_requires_discount_rate` | no rate set | deny |
| `ifrs16_reclassification_requires_auditor` | no auditor approval | deny |
| `assignment_requires_landlord_consent` | no consent ref | deny |
| `forfeiture_requires_legal_process` | process incomplete | deny |
| `renewal_requires_investment_committee` | high value, no IC | deny |

## Data Models
- `LeaseCreate/Response` — full lease header with rent, dates, area, and abstraction status
- `LeaseAbstractionCreate/Response` — AI-extracted fields with exception tracking
- `RentEscalationCreate/Response` — escalation type with old/new rent tracking
- `LeaseOptionCreate/Response` — option window, notice days, exercise status
- `RentReviewCreate/Response` — review type, proposed/agreed rent, backdating auth
- `Ifrs16ScheduleCreate/Response` — ROU asset, liability, 12-month amortisation schedule
- `LeaseAssignmentCreate/Response` — assignment type with landlord consent reference

## Streaming Events
- `lease_created`, `lease_signed`, `lease_activated`, `lease_expired`, `lease_surrendered`
- `rent_escalation_applied`, `rent_review_commenced`, `rent_review_agreed`
- `option_exercised`, `option_lapsed`, `option_expiring_soon`
- `ifrs16_schedule_generated`, `lease_expiry_alert_sent`
- `assignment_completed`, `subletting_approved`

## Edge Cases Handled
- Activation blocked until abstraction is verified (not just complete)
- Escalation double-apply prevented (applied flag checked)
- IFRS 16 discount rate validated 0 < rate < 1 at model level
- Option exercise outside window returns hard denial even with notice
- Reclassification requires auditor approval to prevent silent balance sheet changes
- Expiry pipeline sorts by days_remaining ascending for urgency

## World-Class Enhancements (v2.0)

1. **Persistent DB Adapter** — `PostgresLeaseStore` via SQLAlchemy async ORM; `LeaseStoreProtocol` interface decouples persistence
2. **Event Sourcing** — Domain events (`LeaseCreated`, `LeaseAmended`, etc.) with full audit replay; IFRS 16 disclosure history
3. **Rent Escalation Projections** — `project_rent_escalation_schedule` multi-year table with CPI forecasts and stepped schedules
4. **Dilapidation Provisions** — `calculate_dilapidation_provision` covering pre-lease, interim, and terminal schedules (IFRS 37)
5. **ERV Benchmarking** — `benchmark_against_erv` over/under-rented status, reversion potential, and time-to-reversion
6. **Full Amortisation Schedule** — `full_amortisation_schedule` every period to expiry; quarterly/annual summarisation; CSV/JSON export
7. **Covenant Monitoring** — `record_covenant` / `test_covenant_compliance` for rent cover, DSCR, net worth thresholds
8. **LLM Lease Abstraction** — `extract_lease_abstract_llm` via local Ollama (Llama 3/Mistral); maps JSON output to `LeaseAbstractionCreate`
9. **Multi-Currency FX Translation** — `translate_portfolio_to_reporting_currency` with functional and presentation currency columns
10. **Abstraction Quality Scoring** — `score_abstraction_quality` 0–100 score with per-field gap analysis
11. **Break Option Modelling** — `model_break_option_cost` NPV break vs. stay matrix with penalty, dilaps, fit-out write-off, relocation
12. **KPI Dashboard** — `lease_kpi_dashboard` vacancy rate, OCR, WAULT, reversion yield, void liability with time-series trends
13. **Holding-Over Detection** — `detect_holding_over` transitions overdue leases, records uplift rent, emits notification events
14. **Data Integrity Validation** — `validate_lease_data_integrity` cross-record checks: review dates, IFRS 16 reconciliation, sublease term bounds
15. **Discount Rate Sensitivity** — `discount_rate_sensitivity` liability matrix per 100bps; breakeven rate for operating vs. finance flip

## New Methods

### `full_amortisation_schedule`
```python
schedule = await svc.full_amortisation_schedule(
    lease_id="lease-uuid",
    summarise_by="quarterly",  # "monthly" | "quarterly" | "annual"
    tenant_id="tenant-uuid",
)
# Returns: {"lease_id": ..., "periods": [{"period": 1, "opening_balance": ...,
#   "interest": ..., "principal": ..., "payment": ..., "closing_balance": ...}, ...],
#   "summary": {"total_payments": ..., "total_interest": ..., "total_principal": ...}}
```

### `detect_holding_over`
```python
results = await svc.detect_holding_over(
    as_of_date=date(2026, 6, 1),
    uplift_pct=Decimal("0.10"),  # 10% above passing rent
    tenant_id="tenant-uuid",
)
# Returns: list of leases transitioned to holding_over status with new_rent,
# holding_over_since, and notification event IDs
```

### `discount_rate_sensitivity`
```python
matrix = await svc.discount_rate_sensitivity(
    lease_id="lease-uuid",
    rate_min=Decimal("0.03"),
    rate_max=Decimal("0.09"),
    step=Decimal("0.005"),
    tenant_id="tenant-uuid",
)
# Returns: {"rates": [...], "liabilities": [...], "rou_assets": [...],
#   "delta_per_100bps": ..., "breakeven_rate": ...}
```

## New Methods (v2 additions)

| Method | Description |
|--------|-------------|
| `full_amortisation_schedule(lease_id, summarise_by)` | Complete schedule for entire term; monthly/quarterly/annual grouping |
| `project_rent_escalation_schedule(lease_id, years, cpi_forecast)` | Multi-year compounded rent projection |
| `detect_holding_over(as_of_date, uplift_pct)` | Find and transition overdue leases to holding-over status |
| `validate_lease_data_integrity(lease_id)` | Cross-record consistency checks with issue/warning detail |
| `discount_rate_sensitivity(lease_id, rate_min, rate_max, step)` | Liability sensitivity matrix across rate range |
| `lease_kpi_dashboard(tenant_id, period)` | Occupancy, WAULT, collection efficiency, balance sheet KPIs |
| `model_break_option_cost(lease_id, break_date)` | NPV break vs. stay with penalty, dilaps, relocation components |
| `benchmark_against_erv(lease_id, erv_per_sqm_annual)` | Over/under-rented analysis and reversion potential |

## Composability Notes
- Provides IFRS 16 schedules consumed by `realestate_acc`
- Activating a lease triggers unit status update in `realestate_prm`
- Rent escalations feed into `realestate_ren` rent collection expected amounts
- Option tracking integrates with `realestate_ren` renewal pipeline
- ERV benchmark feeds from `realestate_val` valuation data
- KPI dashboard data surfaces in `realestate_ren` reporting module
- Holding-over detection triggers alerts via `ntfy` capability
