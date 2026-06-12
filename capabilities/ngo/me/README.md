# M&E — Monitoring & Evaluation (ngo_me)

Indicator framework, data collection, progress reporting, impact assessment, learning cycles.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/ngo/me/health` | Health check |
| GET | `/api/ngo/me/indicators` | List indicators |
| POST | `/api/ngo/me/indicators` | Create indicator |
| GET | `/api/ngo/me/indicators/<id>` | Get indicator |
| PUT | `/api/ngo/me/indicators/<id>` | Update indicator |
| DELETE | `/api/ngo/me/indicators/<id>` | Delete indicator |
| GET | `/api/ngo/me/indicators/<id>/trend` | Trend analysis |
| POST | `/api/ngo/me/data-collections` | Collect data point |
| POST | `/api/ngo/me/data-collections/bulk` | Bulk data collection |
| GET | `/api/ngo/me/data-collections` | List data collections |
| GET | `/api/ngo/me/progress-reports` | List progress reports |
| POST | `/api/ngo/me/progress-reports` | Create progress report |
| POST | `/api/ngo/me/progress-reports/<id>/submit` | Submit report |
| GET | `/api/ngo/me/evaluations` | List evaluations |
| POST | `/api/ngo/me/evaluations` | Create evaluation |
| GET | `/api/ngo/me/learning-cycles` | List learning cycles |
| POST | `/api/ngo/me/learning-cycles` | Create learning cycle |
| POST | `/api/ngo/me/learning-cycles/<id>/findings` | Add findings |
| GET | `/api/ngo/me/dashboard/<programme_id>` | Indicator dashboard |
| GET | `/api/ngo/me/impact/<programme_id>` | Impact summary |
| GET | `/api/ngo/me/audit-events` | Audit log |

## World-Class Enhancements (v2.0)

Fifteen improvements closing the gap against DevResults, Apricot, DHIS2, and Vera Solutions Amp Impact:

**I1. Theory of Change Linkage** — Trace every indicator to a ToC node via `toc_node_id`/`causal_pathway`; `validate_indicator_chain()` returns a completeness score. [Feature]

**I2. SMART Indicator Validator** — `validate_indicator_smart()` scores Specific/Measurable/Achievable/Relevant/Time-bound dimensions with remediation hints. [AI/ML]

**I3. Disaggregation-Aware Aggregation Engine** — `aggregate_disaggregated_data()` pivots raw collection records by sex, age-band, or location into totals and sub-totals. [Feature]

**I4. Milestone Tracking and Deadline Alerts** — `create_milestone()` / `check_overdue_milestones()` expose quarterly checkpoint gaps before report time. [Feature]

**I5. Donor-Ready Report Export (OECD DAC Markers)** — `tag_report_dac_markers()` + `export_report_oecd()` serialise reports to OECD CRS++ schema with policy-marker tagging. [Compliance]

**I6. Cost-Efficiency Analysis (Cost per Beneficiary)** — `record_expenditure()` + `compute_cost_efficiency()` compute cost-per-unit and breakdown by cost category using `Decimal`. [Feature]

**I7. Beneficiary Registry Integration** — `get_unique_beneficiary_count()` cross-checks collected beneficiary totals against the `ngo_beneficiaries` adapter deduplicated count. [Integration]

**I8. Adaptive Management Triggers** — `set_adaptive_trigger()` + `evaluate_adaptive_triggers()` surface decision points when achievement_pct drops below threshold. [AI/ML]

**I9. Survey / Instrument Builder** — `create_survey_instrument()` + `ingest_survey_responses()` map ODK/KoboToolbox fields directly to indicator data-collection records. [Feature]

**I10. Inter-Rater Reliability Scoring** — `compute_inter_rater_reliability()` calculates Cohen's kappa (categorical) and CV% (continuous) and flags outliers >2σ. [Feature]

**I11. Log Frame / Results Framework Builder** — `create_logframe()` / `export_logframe()` build goal→purpose→output→activity hierarchies with means-of-verification and assumptions. [Feature]

**I12. Real-Time GIS Location Tagging** — `tag_data_collection_location()` + `get_geographic_coverage()` attach GPS coordinates and return admin-unit bbox summaries per indicator. [Feature]

**I13. Automated Variance Explanation Engine** — `generate_variance_explanation()` categorises deviation (on_track/minor/major/critical) and produces a structured narrative template. [AI/ML]

**I14. Programme Portfolio Benchmarking** — `portfolio_benchmark()` ranks programmes by avg_achievement_pct, on/off-track counts, and evaluation ratings with percentile ranks. [Analytics]

**I15. Evidence Repository with Version Control** — `store_evidence()` / `search_evidence()` maintain semver-versioned, tag-filtered institutional evidence records. [Feature]

## New Methods

Three high-impact async methods from `MEService` worth integrating immediately.

### `trend_analysis` — time-series direction per indicator

```python
svc = MEService(tenant_id="prog-001")
trend = await svc.trend_analysis(indicator_id="ind-abc123")
# {
#   "indicator_id": "ind-abc123",
#   "code": "HH-01",
#   "name": "Households reached",
#   "target_value": 5000,
#   "current_value": 3200,
#   "data_points": [{"date": "2026-01-31", "value": 800, "period": "Q1"}, ...],
#   "trend": "increasing",
#   "generated_at": "2026-06-12T09:00:00Z"
# }
```

### `bulk_collect_data` — atomically load a reporting period

```python
results = await svc.bulk_collect_data([
    {"indicator_id": "ind-abc123", "value": 900, "period": "Q2-2026",
     "collection_date": "2026-06-01", "collected_by": "field-officer-7"},
    {"indicator_id": "ind-def456", "value": 42, "period": "Q2-2026",
     "collection_date": "2026-06-01", "collected_by": "field-officer-7"},
])
# {"created": 2, "errors": [], "collections": [...]}
```

### `indicator_performance_dashboard` — programme-level achievement overview

```python
dashboard = await svc.indicator_performance_dashboard(programme_id="prog-001")
# {
#   "programme_id": "prog-001",
#   "indicators": [
#     {"indicator_id": "...", "name": "...", "achievement_pct": 64.0,
#      "status": "on_track", "trend": "increasing"},
#     ...
#   ],
#   "summary": {"total": 12, "on_track": 8, "off_track": 4},
#   "generated_at": "2026-06-12T09:00:00Z"
# }
```
