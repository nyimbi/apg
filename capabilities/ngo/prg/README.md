# Programme & Project Monitoring (ngo_prg)

Logframe management, activity tracking, output/outcome recording, field data collection.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/ngo/prg/health` | Health check |
| GET | `/api/ngo/prg/` | List programmes |
| POST | `/api/ngo/prg/` | Create programme |
| GET | `/api/ngo/prg/<id>` | Get programme |
| PUT | `/api/ngo/prg/<id>` | Update programme |
| DELETE | `/api/ngo/prg/<id>` | Delete programme |
| POST | `/api/ngo/prg/<id>/activate` | Activate programme |
| GET | `/api/ngo/prg/<id>/logframes` | List logframes |
| POST | `/api/ngo/prg/<id>/logframes` | Create logframe |
| GET | `/api/ngo/prg/<id>/activities` | List activities |
| POST | `/api/ngo/prg/<id>/activities` | Create activity |
| GET | `/api/ngo/prg/<id>/outputs` | List outputs |
| POST | `/api/ngo/prg/<id>/field-data` | Submit field data |
| GET | `/api/ngo/prg/<id>/field-data` | List field data |
| GET | `/api/ngo/prg/<id>/progress` | Progress report |
| GET | `/api/ngo/prg/<id>/gantt` | Gantt chart data |
| GET | `/api/ngo/prg/portfolio/overview` | Portfolio overview |
| GET | `/api/ngo/prg/audit-events` | Audit log |

## World-Class Enhancements (v2.0)

**I1. Theory of Change Pathway Management** — Store causal nodes/edges; trace Impact→Outcome→Output→Activity chains and validate logframe coverage. [Feature]

**I2. Indicator Baseline & Target Tracking with SMART Validation** — Record baseline/target as `Decimal` with collection frequency; heuristic SMART checks return structured findings with fix hints. [Compliance]

**I3. Budget Variance & Burn-Rate Analytics** — Planned-vs-actual spend, monthly burn rate, months-to-zero, and green/amber/red risk colour per activity and programme rollup. [Performance]

**I4. Milestone Critical Path Analysis** — Topological sort of activity dependencies; identifies float, flags zero-float critical activities, returns earliest/latest start dates. [Feature]

**I5. Adaptive Management Trigger Rules** — Register condition expressions (field, operator, threshold, window_days); evaluate all rules against live outputs and emit `adaptive_management_alert` events. [AI/ML]

**I6. Beneficiary Disaggregation & Reach Tracking** — Record counts keyed by sex/age_band/disability/location; pivot to cross-tab donor template report via `beneficiary_summary()`. [Compliance]

**I7. Evidence Document Attachment Registry** — Link document URI+hash to any output or field data record; `generate_evidence_pack()` returns a per-output evidence manifest for audit packs. [Feature]

**I8. Work Plan Auto-Generation from Logframe** — Read approved logframe outputs, create placeholder activities with proportional budget splits in draft state, return structured work plan for PM review. [Feature]

**I9. Donor Report Template Engine** — Pre-mapped templates for USAID ADS 201, FCDO logframe, UN OCHA sitrep; renders ~80% of report content from live data and flags data gaps. [Feature]

**I10. IATI Standard XML Export** — Map programme/logframe/budget/output data to IATI 2.03 schema, validate mandatory fields, return well-formed XML bytes for registry submission. [Integration]

**I11. Geospatial Activity Heatmap Data** — Store GeoJSON point/polygon with ISO 3166-2/GADM codes; `get_geographic_coverage()` returns a FeatureCollection for direct Leaflet/MapLibre consumption. [Feature]

**I12. Risk Register with Escalation Scoring** — Log probability×impact scores (1-5 each) per risk; `risk_heatmap()` returns matrix data with threshold-triggered escalation flags. [Compliance]

**I13. Offline-First Field Data Queue** — Accept batches with device_id and pre-timestamps; deduplicate on (device_id, collection_date, location) composite key; bulk-insert and return sync manifest. [UX]

**I14. Learning Review & After-Action Record** — Structured records of what worked/failed with root causes and recommended changes; filterable by sector for cross-programme knowledge extraction. [Feature]

**I15. Programme Health Score (Composite KPI)** — Weighted 0-100 index: budget_adherence 25%, activity_completion 25%, output_achievement 25%, risk_level 15%, data_quality 10%; returns colour band + breakdown. [AI/ML]

## New Methods

The three highest-impact additions from v2.0 — covering portfolio visibility, donor compliance, and adaptive learning.

### `compute_programme_health_score(programme_id)`

Surfaces lagging programmes at a glance for portfolio managers overseeing 20+ programmes.

```python
svc = ProgrammeService(tenant_id="ke-rift")
score = await svc.compute_programme_health_score("prg-001")
# {
#   "programme_id": "prg-001",
#   "score": 74,
#   "band": "amber",
#   "breakdown": {
#     "budget_adherence": 82,
#     "activity_completion": 68,
#     "output_achievement": 71,
#     "risk_level": 60,
#     "data_quality": 90
#   }
# }
```

### `generate_donor_report(programme_id, template_id)`

Reduces reporting burden by generating donor-formatted report sections from live programme data.

```python
report = await svc.generate_donor_report(
    programme_id="prg-001",
    template_id="fcdo_annual",   # "usaid_fy" | "fcdo_annual" | "un_ocha_sitrep"
)
# Returns structured dict aligned to FCDO Annual Review format;
# "data_gaps" key lists sections requiring manual input before submission.
```

### `evaluate_adaptive_triggers(programme_id)`

Closes the USAID CLA learn-adapt loop by running all registered rules against current output data.

```python
# First register a trigger
await svc.register_adaptive_trigger(
    programme_id="prg-001",
    field="achievement_pct",
    operator="lt",
    threshold=60,
    window_days=180,
    recommended_action="Convene adaptive management review; consider activity redesign.",
)

# Then evaluate — typically called on a schedule or after each field data sync
alerts = await svc.evaluate_adaptive_triggers("prg-001")
# [
#   {
#     "trigger_id": "trg-abc",
#     "fired": True,
#     "current_value": 54.3,
#     "recommended_action": "Convene adaptive management review...",
#     "event": "adaptive_management_alert"
#   }
# ]
```
