# Exploration Data Management — User Guide

**Capability ID**: `mining_exp` | **Domain**: `mining` | **Version**: `1.1.0`

## Description

Manages the full lifecycle of mineral exploration data: drill-hole collar logging, downhole deviation surveys with 3D desurveying, geological interval logging, geochemical assay management with gap detection and compositing, QAQC monitoring, bulk density tracking, spatial licence boundary enforcement, competent person credential management, resource domain definition, resource estimation workflows, JORC/NI 43-101/SAMREC compliance reporting, and exploration expenditure analytics.

---

## Installation

```bash
pip install apg-mining-exp
```

---

## Quick Start

```python
import asyncio
from datetime import datetime
from apg_mining_exp.service import ExpService

svc = ExpService(tenant_id="acme_mining")

async def main():
    # 1. Register a licence
    lic = await svc.register_licence(
        licence_number="EL12345",
        area_coords=[
            {"lon": 32.1, "lat": -1.2},
            {"lon": 32.5, "lat": -1.2},
            {"lon": 32.5, "lat": -1.6},
            {"lon": 32.1, "lat": -1.6},
        ],
        holder_id="acme",
        expiry=datetime(2028, 12, 31),
    )

    # 2. Create a drill hole
    hole = await svc.drill_hole_create(
        hole_id="ABDD001",
        location={"easting": 32.3, "northing": -1.4, "elevation_m": 1220.0},
        total_depth=350.0,
        drill_type="DD",
        created_by="geo_jane",
        licence_id=lic["id"],
        azimuth_deg=90.0,
        dip_deg=-60.0,
    )

    # 3. Record downhole deviation surveys
    for depth in [0, 50, 100, 200, 300, 350]:
        await svc.record_downhole_survey(
            hole_id="ABDD001", depth_m=depth,
            azimuth_deg=91.0, dip_deg=-60.5,
            survey_tool="Maxibor", surveyed_by="geo_jane",
        )

    # 4. Desurvey for 3D coordinates
    path = await svc.desurvey_hole("ABDD001")

    # 5. Log core intervals
    await svc.log_drill_core(
        hole_id="ABDD001", from_depth=120.0, to_depth=122.0,
        lithology="GRD", structure="foliated", mineralisation="disseminated_sulphide",
        logged_by="geo_jane", recovery_pct=98.5, rqd_pct=85.0,
    )

    # 6. Record assay results
    await svc.assay_result(
        hole_id="ABDD001", from_depth=120.0, to_depth=121.0,
        element="AU", grade=3.45, unit="g/t",
        batch_id="BATCH001", lab_id="SGS_NAI", created_by="geo_jane",
    )

    # 7. Detect sampling gaps
    gaps = await svc.detect_sampling_gaps("ABDD001", gap_threshold_m=0.5)

    # 8. Composite to 2m benches
    comps = await svc.composite_assay_intervals(
        hole_id="ABDD001", element="AU", composite_length_m=2.0
    )

asyncio.run(main())
```

---

## Service Methods Reference

### Drillhole Collar Management

| Method | Signature | Description |
|---|---|---|
| `create_drillhole_collar` | `(payload, created_by)` | Register a new collar with coordinate validation |
| `get_drillhole_collar` | `(id)` | Retrieve collar by record UUID |
| `get_drillhole_collar_by_hole_id` | `(hole_id)` | Lookup collar by field hole identifier |
| `list_drillhole_collars` | `(prospect, hole_type, limit, offset)` | Filtered collar listing |
| `update_drillhole_actual_depth` | `(id, actual_depth_m)` | Record as-drilled depth on completion |

### Drill Hole Management (Extended)

| Method | Signature | Description |
|---|---|---|
| `drill_hole_create` | `(hole_id, location, total_depth, drill_type, created_by, ...)` | Create hole with type validation |
| `get_drill_hole` | `(hole_id)` | Lookup by field identifier |
| `list_drill_holes` | `(licence_id, status)` | Filtered listing |

### Downhole Deviation Surveys

| Method | Signature | Description |
|---|---|---|
| `record_downhole_survey` | `(hole_id, depth_m, azimuth_deg, dip_deg, survey_tool, surveyed_by)` | Add a survey station. Validates azimuth [0,360) and dip [-90,0] |
| `desurvey_hole` | `(hole_id)` | Apply SPWLA minimum-curvature algorithm. Returns list of `{depth_m, easting, northing, elevation_m}` at each station |

The minimum-curvature desurveying algorithm is the industry standard (SPWLA) for converting azimuth/dip survey stations to 3D XYZ coordinates. Accuracy degrades with sparse survey spacing; survey every 30m or less is recommended.

### Core Logging

| Method | Signature | Description |
|---|---|---|
| `log_drill_core` | `(hole_id, from_depth, to_depth, lithology, structure, mineralisation, logged_by, recovery_pct, rqd_pct)` | Log a core interval. Validates hole existence and depth bounds |
| `get_core_log_for_hole` | `(hole_id)` | All core intervals sorted by depth |

### Assay Results

| Method | Signature | Description |
|---|---|---|
| `import_assay_results` | `(payloads, created_by)` | Bulk import with collar existence and interval overlap validation |
| `assay_result` | `(hole_id, from_depth, to_depth, element, grade, unit, ...)` | Single assay record with overlap detection |
| `get_assay_results_for_hole` | `(hole_id)` | Sorted by from_m |
| `list_assay_results_for_hole` | `(hole_id, element)` | Optionally filter by element |
| `list_assays` | `(commodity, min_grade, limit, offset)` | Tenant-wide filtered listing |
| `flag_qaqc_result` | `(assay_id, flag)` | Attach QAQC flag (use standard codes: BLANK_FAIL, CRM_FAIL, DUP_FAIL) |

### Sampling Gap Detection

| Method | Signature | Description |
|---|---|---|
| `detect_sampling_gaps` | `(hole_id, gap_threshold_m, expected_to_depth_m)` | Identify unsampled depth intervals. Returns `{total_assayed_m, total_gap_m, gap_pct, gaps[]}` |

JORC Table 1 Section 9 requires disclosure of unsampled intervals. Run this method before submitting any resource estimate to confirm full sampling coverage.

### Grade Compositing

| Method | Signature | Description |
|---|---|---|
| `composite_assay_intervals` | `(hole_id, element, composite_length_m, from_depth_m, to_depth_m)` | Fixed-length weighted-average compositing. Marks partial intervals at the toe |

Bench compositing aligns assay data to regularised intervals for block model inputs. A `composite_length_m` equal to the mining bench height (typically 2–5 m) is recommended.

### Geology Logging

| Method | Signature | Description |
|---|---|---|
| `log_geology_interval` | `(payload, created_by)` | Log geological interval with lithology, oxidation, RQD, TCR |
| `get_geology_for_hole` | `(hole_id)` | All intervals sorted by depth |
| `list_geology_by_lithology` | `(lithology_code)` | Cross-hole lithology query |

### Resource Domains

| Method | Signature | Description |
|---|---|---|
| `define_resource_domain` | `(name, lithology_codes, commodity, cut_off_grade, grade_unit)` | Define an estimation domain by lithology membership |
| `assign_intervals_to_domains` | `(hole_ids)` | Batch-assign geology intervals. Returns `{assigned_by_domain, unassigned_intervals, multi_domain_intervals}` |

Domains are the primary mechanism for preventing grade smearing across geological boundaries. Always define domains before running resource estimates.

### Resource Estimates

| Method | Signature | Description |
|---|---|---|
| `create_resource_estimate` | `(payload, created_by)` | Create JORC/NI43-101 estimate. CP assignment mandatory |
| `resource_estimate` | `(deposit_id, method, classification, ...)` | Extended estimate with full JORC fields |
| `update_resource_estimate` | `(id, update)` | Partial update — blocked for approved estimates |
| `approve_resource_estimate` | `(id, reviewer_id, notes)` | CP approval gate |
| `publish_resource_estimate` | `(id)` | Public disclosure — requires prior approval |
| `list_resource_estimates` | `(classification, commodity, published_only)` | Filtered listing |
| `jorc_compliance_check` | `(estimate_id)` | Run JORC Table 1 checklist — 6 mandatory items |

### Competent Person Registry

| Method | Signature | Description |
|---|---|---|
| `register_competent_person` | `(cp_id, full_name, professional_body, membership_number, commodity_specialisations, credential_expiry)` | Register or update a CP. Idempotent on cp_id |
| `validate_competent_person` | `(cp_id, commodity)` | Returns `{valid, days_to_expiry, issues[]}`. Checks active status, expiry, and commodity scope |

Expired or unregistered CPs will cause resource estimate and compliance report workflows to fail. Register all CPs before beginning exploration activities and configure renewal alerts 90 days before expiry.

### Bulk Density

| Method | Signature | Description |
|---|---|---|
| `record_bulk_density` | `(hole_id, from_m, to_m, method, value_t_m3, lithology_code, measured_by)` | Record BD measurement. Physical bounds [1.0, 5.5] t/m³ enforced |
| `summarise_bulk_density` | `(hole_ids, lithology_code)` | Returns `{mean, std_dev, min, max, by_lithology{}}` |

JORC requires traceability of bulk density data to resource tonnes. Target a minimum of one BD measurement per significant lithological unit per 100m vertical depth.

### Spatial Licence Validation

| Method | Signature | Description |
|---|---|---|
| `check_collar_within_licence` | `(hole_id, licence_id)` | Point-in-polygon test for a single collar |
| `validate_programme_against_licence` | `(licence_id)` | Batch validation of all holes on a licence. Returns `{all_within_boundary, outside_details[]}` |

Run `validate_programme_against_licence` before submitting any quarterly report or NI 43-101 technical report. Drilling outside the licence boundary is a regulatory breach.

### Geophysics Surveys

| Method | Signature | Description |
|---|---|---|
| `geophysics_survey` | `(survey_type, area, data, conducted_by, licence_id, survey_date)` | Record survey. Accepts IP, MT, gravity, aeromagnetic, seismic, TEM, CSAMT, ground_mag, radiometric |
| `list_geophysics_surveys` | `(survey_type, licence_id)` | Filtered listing |

### Exploration Targets (JORC cl.17)

| Method | Signature | Description |
|---|---|---|
| `report_exploration_target` | `(deposit_id, tonnage_low, tonnage_high, grade_low, grade_high, commodity, grade_unit, reported_by, basis, caution_statement)` | JORC 2012 cl.17 compliant target reporting. Automatically prepends standard caution statement if none supplied |

### Expenditure Tracking

| Method | Signature | Description |
|---|---|---|
| `record_expenditure` | `(licence_id, period, category, amount_usd, currency, exchange_rate, notes, recorded_by)` | Log an expenditure line item. Categories: drilling, geophysics, assaying, geochemistry, overhead, permitting, geological_mapping, remote_sensing, admin |
| `compute_cost_per_resource_unit` | `(licence_id, period, commodity)` | Cost-per-ounce-equivalent or cost-per-tonne KPI for investor reporting |

### Compliance Reports

| Method | Signature | Description |
|---|---|---|
| `create_compliance_report` | `(payload, created_by)` | Create JORC/NI43-101/SAMREC report shell |
| `sign_off_compliance_report` | `(id, competent_person_id)` | CP sign-off — only the assigned CP may sign |
| `publish_compliance_report` | `(id)` | Public disclosure after CP sign-off |
| `list_compliance_reports` | `(published_only)` | Listing with optional filter |

### Analytics and Reporting

| Method | Signature | Description |
|---|---|---|
| `get_exploration_summary` | `()` | Tenant-level KPI snapshot |
| `exploration_analytics` | `(licence_id, period)` | Detailed analytics: metres drilled, samples, grade by element, recovery |
| `quarterly_report` | `(licence_id, period)` | Full quarterly report bundle: analytics + resource position + significant intercepts |

---

## Business Rules

| Rule | Effect |
|---|---|
| Duplicate `hole_id` within tenant | DENY at creation |
| Assay interval overlap (same hole + element) | DENY import |
| Collar not found for assay import | DENY entire batch atomically |
| Resource estimate update when APPROVED | DENY — supersede instead |
| CP sign-off by wrong person | DENY |
| Publish without CP sign-off | DENY |
| Bulk density outside [1.0, 5.5] t/m³ | DENY with physical bounds error |
| Downhole survey depth exceeds hole total | DENY |
| Desurvey with no survey stations | DENY with descriptive error |
| Cross-tenant data access | DENY |

---

## JORC Compliance Checklist (`jorc_compliance_check`)

The method runs 6 mandatory JORC Table 1 checks:

| Check | JORC Reference |
|---|---|
| competent_person_assigned | Clause 9 — CP sign-off required |
| effective_date_present | Table 1 Section 1 |
| valid_classification | 2012 classification hierarchy |
| methodology_documented | Section 5 — estimation method |
| grade_unit_specified | Reporting transparency |
| tonnes_positive | Non-zero resource required |

All 6 must pass before `jorc_compliant=True` is set on the estimate.

---

## Workflow: Drill, Sample, Estimate

```
register_licence
  └─ drill_hole_create
       ├─ record_downhole_survey (multiple stations)
       │    └─ desurvey_hole → 3D coordinates
       ├─ log_drill_core (per interval)
       ├─ record_bulk_density (per lithology unit)
       ├─ assay_result (per sample)
       │    └─ detect_sampling_gaps → verify coverage
       │    └─ composite_assay_intervals → bench composites
       └─ log_geology_interval
            └─ assign_intervals_to_domains

register_competent_person
  └─ validate_competent_person (before estimate)

resource_estimate
  └─ jorc_compliance_check
       └─ approve_resource_estimate (reviewer_id)
            └─ publish_resource_estimate

create_compliance_report
  └─ sign_off_compliance_report (cp_id)
       └─ publish_compliance_report

record_expenditure (by period/category)
  └─ compute_cost_per_resource_unit → investor KPI
```

---

## Configuration Keys

All keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `MINING_EXP_`.

| Key | Default | Description |
|---|---|---|
| `drill_holes.collar_survey_required` | `true` | Requires surveyed collar coordinates |
| `drill_holes.down_hole_survey_required` | `true` | Requires downhole deviation surveys |
| `sampling.qaqc_insertion_required` | `true` | Enforces QAQC sample insertion per batch |
| `assays.lab_cert_required` | `true` | Lab certificate ref mandatory on all assays |
| `resources.competent_person_required` | `true` | CP assignment mandatory for resource estimates |
| `reporting.public_disclosure_review_required` | `true` | Approval required before external publication |
| `compositing.default_length_m` | `2.0` | Default bench composite length |
| `bulk_density.min_coverage_per_1000t` | `1` | Minimum BD measurements per 1000t block |

---

## Interoperability

```apg
use mining_exp;
```

Downstream consumers:

| Capability | Data Consumed |
|---|---|
| `mining_pro` | Resource estimates for grade control cutoffs |
| `mining_ore` | Geology intervals for feed characterisation |
| `mining_env` | JORC reports for ESG/closure documentation |
| `mining_3d` | Desurveyed 3D collar paths for block model alignment |
| `mining_fin` | Expenditure records for exploration budget reporting |
| `geos` | Collar coordinates for spatial indexing and map tile serving |
| `ragn` | Assay data for geological RAG queries |
| `wflo` | CP credential registry for approval gating |

---

## Further Reading

- `service.py` — Full business logic implementation
- `models.py` — Pydantic v2 data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views
- `README.md` — Quick capability reference
- `WORLD_CLASS_IMPROVEMENTS.md` — Roadmap of 15 high-impact enhancements
- `SPECIFICATION.md` — Full capability specification
- `tests/` — Unit and integration tests
