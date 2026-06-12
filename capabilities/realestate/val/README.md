# Property Valuation

## Overview
Full-cycle property valuation: comparable sales database, DCF model builder with range-validated discount rates, mass appraisal engine (regression, spatial, hedonic, AI AVM), valuation roll with automatic supersession, revaluation cycle management, Red Book sign-off enforcement with independent valuer validation, and structured challenge workflow requiring counter-evidence.

## Capability ID
`realestate_val`

## Provides
- `comparable_sales_analysis`: Verified comparable database with adjustment factors
- `dcf_valuation_engine`: Multi-year DCF with exit yield, rental growth, and capex allowance
- `mass_appraisal_engine`: 5 model types including AI AVM with calibration requirement
- `valuation_roll_management`: Current valuation roll with automatic supersession
- `revaluation_cycle_management`: 9 trigger types including IFRS reporting date
- `valuation_report_generation`: Desktop, restricted, Red Book, and mass appraisal reports
- `yield_analysis`: NIY, equivalent yield, reversionary yield, and 3 other types
- `valuer_panel_management`: RICS, API, and internal valuer grades
- `valuation_challenge_workflow`: Evidence-gated challenge against published valuations
- `valuation_benchmarking`: Portfolio value trends and market comparisons

## Requires
| Capability | Reason |
|-----------|--------|
| `auth` | Sign-off and challenge review authority |
| `audl` | Immutable valuation audit trail |
| `mten` | Multi-tenant isolation |
| `conf` | Discount rate range and model configuration |
| `ntfy` | Revaluation due and challenge alerts |
| `wflo` | Sign-off and challenge approval workflows |
| `nlpc` | Comparable data text extraction |
| `comp` | RICS Red Book compliance |
| `mqeb` | Publish valuation lifecycle events |
| `schd` | Revaluation cycle scheduling |

## Configuration
| Key | Default | Description |
|-----|---------|-------------|
| `dcf.min_discount_rate` | 0.03 | Minimum discount rate (3%) |
| `dcf.max_discount_rate` | 0.30 | Maximum discount rate (30%) |
| `mass_appraisal.calibration_required` | true | Mandate model calibration |
| `valuers.independence_required_for_red_book` | true | Independent valuer for Red Book |

## API Routes
| Path | Method | Description | Permission |
|------|--------|-------------|-----------|
| `/realestate/val/valuations` | GET/POST | List/instruct valuations | `valuations` |
| `/realestate/val/valuations/<id>/sign-off` | POST | Sign off | `valuations` |
| `/realestate/val/valuations/<id>/publish` | POST | Publish (immutable) | `valuations` |
| `/realestate/val/comparables` | GET/POST | Comparable database | `comparables` |
| `/realestate/val/comparables/<id>/verify` | POST | Verify comparable | `comparables` |
| `/realestate/val/dcf` | POST | Run DCF model | `dcf` |
| `/realestate/val/mass-appraisal` | POST | Run mass appraisal | `mass_appraisal` |
| `/realestate/val/roll` | GET/POST | Valuation roll | `roll` |
| `/realestate/val/yields/<property_id>` | GET | Yield calculation | `yields` |
| `/realestate/val/challenges` | GET/POST | Challenges | `challenges` |
| `/realestate/val/challenges/<id>/resolve` | POST | Resolve challenge | `challenges` |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| `valuation_requires_qualified_valuer` | no qualified valuer | deny |
| `red_book_requires_independent_valuer` | non-independent | deny |
| `sign_off_requires_approved_valuer_grade` | internal_valuer grade | deny |
| `dcf_discount_rate_in_range` | < 3% or > 30% | deny (Pydantic) |
| `mass_appraisal_requires_calibrated_model` | not calibrated | deny |
| `challenge_requires_counter_evidence` | no evidence docs | deny (Pydantic) |
| `published_valuation_immutable` | status = published | deny |
| `challenge_requires_active_valuation` | non-challengeable status | deny |

## Data Models
- `ValuerCreate/Response` — graded valuer with independence flag and firm details
- `ComparableCreate/Response` — transaction with price, area, adjustments, verification
- `ValuationCreate/Response` — instruction with method, purpose, report type, and valuer
- `DcfModelCreate/Response` — full DCF parameters with NPV, capital value, cash flow schedule
- `ValuationRollEntryCreate/Response` — roll entry with supersession tracking
- `MassAppraisalRunCreate/Response` — model run with results per property
- `ValuationChallengeCreate/Response` — evidence-backed challenge with counter valuation

## Streaming Events
- `valuation_instructed`, `valuation_completed`, `valuation_approved`, `valuation_published`
- `comparable_added`, `comparable_verified`
- `dcf_model_run`, `mass_appraisal_run_completed`
- `revaluation_cycle_triggered`, `valuation_roll_updated`
- `valuation_challenged`, `challenge_resolved`

## Edge Cases Handled
- Published valuations are truly immutable: any write attempt denied at rule layer
- Valuation roll automatically supersedes the previous entry for the same property
- DCF discount rate range validated at Pydantic model layer (0.03–0.30)
- Challenge requires at least one counter-evidence document at Pydantic level
- Red Book requires independent valuer; internal valuers cannot publish Red Book reports
- Mass appraisal runs return a results list even in simulation mode
- Yield calculation handles zero purchase price via explicit ValueError

## World-Class Enhancements (v2.0)

1. **Hedonic Regression** — OLS model on property attributes (area, age, location, bedrooms, condition) vs comparable transactions; returns R², coefficients, prediction interval (IAAO Std 6).
2. **AVM with Confidence Bands** — IDW spatial weighting + time-decay adjustments; returns `value_low/central/high` and `confidence` tier for downstream gating (Basel III).
3. **IRR via Newton-Raphson Bisection** — `_compute_irr` bisection solver 0–100%, 1 bp convergence; populates `irr` on `DcfModelResponse` (RICS VPS 4).
4. **DCF Sensitivity Grid** — sweeps `discount_rate` and `exit_yield` ±150 bps in 50 bp steps; 2-D capital value grid with ±1σ recommended range.
5. **Residual Land Value** — GDV − build costs − finance − developer profit − transaction costs = residual land value (RICS VPS 12).
6. **Reinstatement Cost Assessment** — BCIS elemental rates per property type/region; per-element breakdown with professional fees and debris removal for insurance sum insured.
7. **Comparable Adjustment Matrix** — paired sales grid: time, size, condition (RICS 1–3), location adjustments; returns adjusted price, narrative, and `reliability_score` 0–100.
8. **Portfolio Variance Report** — like-for-like capital growth with acquisition/disposal split; IFRS 13 / IAS 40 disclosure format.
9. **HABU Analysis** — ranks 3–5 alternative uses by NPV; attaches `valuation_uncertainty` flag when uses are within ±20% (RICS VPS 3).
10. **Market Rent Review Modelling** — upwards-only, upwards/downwards, CPI-linked, and fixed-step clauses; returns revised rent, uplift %, next review date, and IFRS 16 trigger.
11. **Structured Report Generation** — JSON/PDF report with comparable evidence table, sensitivity analysis, RICS Red Book disclaimer, valuer signature, and immutable hash.
12. **Bulk Comparable Import** — batch ingest with fuzzy dedup on (address, date, price ±2%, ±30 days); returns `inserted/skipped_duplicate/validation_errors`.
13. **Equivalent Yield Calculator** — term/reversion dual-rate annuity IRR; returns NIY, equivalent yield, reversionary yield, and running yield in one call.
14. **Revaluation Trigger Detector** — scans roll for age, active planning permissions, >10% rent change, and IFRS reporting date proximity; returns prioritised list with urgency scores.
15. **Market Data Integration Adapter** — `MarketDataAdapter` ABC with pluggable connectors (HMLR, MSCI/IPD, CoStar); normalises to `ComparableCreate` and calls `bulk_import_comparables()`.

## New Methods

### `run_avm` — Confidence-graded automated valuation

```python
result = await svc.run_avm(
    tenant_id="t1",
    property_id="prop_123",
    floor_area_sqm=Decimal("120"),
    location_lat=Decimal("-1.286"),
    location_lng=Decimal("36.817"),
    bedrooms=3,
    property_type="residential",
)
# result.value_central, result.value_low, result.value_high
# result.confidence  -> "very_high" | "high" | "medium" | "low"
# result.comparables_used -> int
```

### `dcf_sensitivity_analysis` — Investment committee stress-test grid

```python
grid = await svc.dcf_sensitivity_analysis(
    tenant_id="t1",
    valuation_id="val_456",
    base_discount_rate=Decimal("0.08"),
    base_exit_yield=Decimal("0.055"),
    annual_rent=Decimal("120000"),
    holding_period_years=10,
)
# grid.capital_value_grid  -> dict keyed by (discount_rate, exit_yield) -> Decimal
# grid.recommended_range_low, grid.recommended_range_high -> ±1σ bounds
```

### `detect_revaluation_triggers` — Automated revaluation queue

```python
triggers = await svc.detect_revaluation_triggers(
    tenant_id="t1",
    max_age_months=24,
    ifrs_reporting_days_ahead=30,
)
# triggers -> list[RevaluationTrigger]
# each: .property_id, .trigger_reason, .urgency_score (0-100), .last_valuation_date
```

## New in This Release

| Method | Description |
|--------|-------------|
| `dcf_sensitivity_analysis()` | 2-D capital value grid sweeping discount rate and exit yield ±150 bps |
| `residual_land_valuation()` | GDV minus development costs giving residual land value (RICS VPS 12) |
| `calculate_equivalent_yield()` | NIY, equivalent yield, reversionary yield via IRR bisection |
| `run_avm()` | IDW automated valuation model with confidence bands (low/medium/high/very_high) |
| `model_rent_review()` | Rent review clause engine with IFRS 16 remeasurement trigger |
| `bulk_import_comparables()` | Batch comparable ingest with fuzzy deduplication |
| `apply_comparable_adjustments()` | Structured adjustment matrix with reliability score |
| `portfolio_variance_report()` | IAS 40/IFRS 13 like-for-like movement with acquisition/disposal split |
| `detect_revaluation_triggers()` | Scan roll for age, IFRS proximity, and rent movement triggers |

## Composability Notes
- Valuation figures feed `realestate_prm` property current_valuation field
- IFRS 16 commencement valuations triggered by `realestate_lea` lease activation; `model_rent_review()` returns `ifrs16_trigger` flag
- Insurance reinstatement values cross-reference `realestate_ins` asset schedule
- DCF rental income inputs sourced from `realestate_ren` rent roll
- `detect_revaluation_triggers()` output can be consumed by `schd` to schedule automatic revaluation instructions
- `run_avm()` confidence bands gate auto-approval in mortgage origination workflows (`realestate_mrt`)
