# Clinical Trials Management — User Guide

**Capability ID**: `pharma_ctr` | **Domain**: `pharma` | **Version**: `1.1.0`
**© 2025 Datacraft** | www.datacraft.co.ke

---

## Description

`pharma_ctr` manages the complete clinical trial lifecycle: protocol management, site activation, randomisation, EDC data capture, adverse event reporting, safety signal detection, regulatory submissions, and GCP inspection readiness. All operations are tenant-scoped and GCP-enforced per ICH E6(R3).

---

## Installation

```bash
pip install apg-pharma-ctr
```

---

## Quick Start

```python
from apg_pharma_ctr.service import ClinicalTrialsService
from apg_pharma_ctr.models import ClinicalTrialCreate, TrialSiteCreate, TrialPatientCreate, AdverseEventCreate

svc = ClinicalTrialsService()
TENANT = "sponsor-acme"

# 1. Register a trial
trial = svc.register_trial(
    tenant_id=TENANT,
    protocol="ACME-2026-001",
    phase="phase_2",
    sponsor="ACME Pharma",
    indication="NSCLC",
    target_enrollment=120,
    created_by="dr.jones@acme.com",
)

# 2. Activate after IRB approval
activated = svc.activate_trial(trial["id"], TENANT, "IRB-2026-0441")

# 3. Add a site
site_payload = TrialSiteCreate(
    tenant_id=TENANT,
    trial_id=trial["id"],
    site_number="SITE-001",
    country="KE",
    principal_investigator_id="pi-001",
    target_enrollment=40,
)
site = svc.select_site(site_payload)

# 4. Initiate site and record SIV
from datetime import datetime
siv = svc.site_initiation_visit(
    tenant_id=TENANT,
    site_id=site.id,
    monitor_id="mon-007",
    visit_date=datetime.utcnow(),
    checklist={
        "protocol_review": True,
        "GCP_training_verified": True,
        "ICF_versions_confirmed": True,
        "EDC_access_granted": True,
        "IRB_approval_on_file": True,
    },
)

# 5. Enrol patient
patient_payload = TrialPatientCreate(
    tenant_id=TENANT,
    trial_id=trial["id"],
    site_id=site.id,
    patient_code="PAT-001",
)
patient = svc.enrol_patient(patient_payload, informed_consent_date=datetime.utcnow())

# 6. Randomise patient (standard block randomisation)
rand = svc.randomise_patient(
    patient_id=patient.id,
    tenant_id=TENANT,
    trial_id=trial["id"],
    randomisation_method="block",
    treatment_arm="A",
    randomisation_code="RC-001",
    randomised_by="dr.jones@acme.com",
)

# 7. Report an adverse event
ae = svc.report_adverse_event(
    tenant_id=TENANT,
    trial_id=trial["id"],
    subject_id=patient.id,
    event_type="neutropenia",
    severity="severe",
    seriousness="serious",
    outcome="recovering",
    narrative="Subject developed grade 3 neutropenia on Day 14 post-dose...",
    reported_by="site-nurse-001",
)

# 8. Dashboard
summary = svc.dashboard_summary(TENANT)
```

---

## Core Workflows

### Protocol Management

```python
# Create protocol version
proto = svc.create_protocol(TENANT, trial["id"], version="v1.0", created_by="dr.jones@acme.com")

# Approve after IRB review
approved_proto = svc.approve_protocol(
    protocol_id=proto.id,
    tenant_id=TENANT,
    irb_approval_reference="IRB-2026-0441",
    approved_by="irb-chair@hospital.ke",
)
```

### CRF Data Collection and Query Management

```python
# Collect CRF data
crf = svc.collect_crf_data(
    tenant_id=TENANT,
    visit_id="VISIT-001",
    subject_id=patient.id,
    form_data={"weight_kg": 72.5, "blood_pressure_systolic": 128, "haemoglobin_g_dl": 11.2},
    collected_by="data-entry-001",
)

# Validate CRF
report = svc.validate_crf(TENANT, crf["id"])

# Raise a query
query = svc.query_management(
    tenant_id=TENANT,
    crf_id=crf["id"],
    query_type="out_of_range",
    query_text="Weight 72.5 kg is inconsistent with baseline 68 kg; confirm.",
    raised_by="mon-007",
)
```

### Adverse Event and Safety Reporting

```python
# Classify causality
causality = svc.classify_ae_causality(
    tenant_id=TENANT,
    ae_id=ae["id"],
    causality="probable",
    assessment_by="dr.jones@acme.com",
)

# File SAR to agencies
sar = svc.report_sar(tenant_id=TENANT, ae_id=ae["id"], agencies=["FDA", "EMA"])

# SMC periodic report
smc = svc.safety_monitoring_committee_report(
    tenant_id=TENANT,
    trial_id=trial["id"],
    period="Q2-2026",
    prepared_by="safety-officer@acme.com",
)
```

### Regulatory Submissions

```python
# File a standard submission
sub = svc.regulatory_submission(
    tenant_id=TENANT,
    trial_id=trial["id"],
    agency="FDA",
    submission_type="IND",
    package_items=["protocol_v1.0", "investigator_brochure", "informed_consent_form"],
    submitted_by="ra-team@acme.com",
)
```

### Database Lock and CSR

```python
# Lock database
lock = svc.database_lock(
    tenant_id=TENANT,
    trial_id=trial["id"],
    lock_reason="End of data collection — all queries resolved",
    locked_by="dm-lead@acme.com",
)

# Generate Clinical Study Report
csr = svc.generate_clinical_study_report(
    tenant_id=TENANT,
    trial_id=trial["id"],
    prepared_by="medical-writer@acme.com",
)
```

---

## Advanced Async Methods

All async methods require `await` inside an async context or `asyncio.run()`.

### Response-Adaptive Randomisation (RAR)

Uses Thompson sampling (Bayesian bandit) to dynamically allocate subjects to arms based on accumulating outcome data. Pre-specify in the SAP before use.

```python
import asyncio

rand_result = asyncio.run(svc.adaptive_randomisation(
    tenant_id=TENANT,
    trial_id=trial["id"],
    subject_id=patient.id,
    prior_arm_outcomes={
        "A": {"successes": 12, "failures": 3},
        "B": {"successes": 7, "failures": 8},
    },
    stratification_factors={"site": "SITE-001", "gender": "M"},
))
# Returns: selected_arm, allocation_probabilities, posterior_means
```

### Safety Signal Detection

PRR (Proportional Reporting Ratio) disproportionality analysis across AE data. Signals when PRR >= 2 and chi-square >= 4 with >= 3 events.

```python
signals = asyncio.run(svc.detect_safety_signals(
    tenant_id=TENANT,
    trial_id=trial["id"],
    min_event_count=3,
))
# Returns: signals list with event_type, PRR, chi_square, recommended_action
```

### GCP Inspection Readiness Score

Composite 0–100 score across 5 components: TMF completeness, query closure, deviation closure, AE timeliness, protocol compliance. Aligned with TransCelerate RBM metrics.

```python
readiness = asyncio.run(svc.compute_inspection_readiness_score(
    tenant_id=TENANT,
    trial_id=trial["id"],
))
# Returns: inspection_readiness_score, risk_level (LOW/MEDIUM/HIGH_RISK), component breakdown
```

### Protocol Amendment Impact Analysis

Identifies subjects requiring re-consent, additional assessments, or eligibility re-review when a protocol amendment is issued.

```python
impact = asyncio.run(svc.protocol_amendment_impact(
    tenant_id=TENANT,
    trial_id=trial["id"],
    new_protocol_id=amended_proto.id,
    old_protocol_id=approved_proto.id,
    changed_sections=["eligibility_criteria", "dosing_schedule"],
))
# Returns: subjects_requiring_reconsent, subjects_needing_additional_assessments, affected_subject_ids
```

### SUSAR Narrative Generation (ICH E2B(R3))

Generates a structured narrative from AE record, causality assessment, and CRF data. Enhanced by local LLM (Ollama) when `OLLAMA_BASE_URL` is set.

```python
narrative = asyncio.run(svc.generate_susar_narrative(
    tenant_id=TENANT,
    ae_id=ae["id"],
    include_lab_values=True,
))
# Returns: patient_section, event_section, drug_section, causality_section, narrative_text
```

### Blinded Sample Size Re-estimation (Cui-Hung-Wang)

Adjusts target enrollment at a pre-specified interim using only the pooled variance estimate — no unblinding required.

```python
ssr = asyncio.run(svc.blinded_sample_size_reestimation(
    tenant_id=TENANT,
    trial_id=trial["id"],
    information_fraction=0.5,
    pooled_variance=14.2,
    original_target_enrollment=120,
    target_power=0.90,
))
# Returns: adjusted_target_enrollment, enrollment_increase, regulatory_justification
```

### IMP Supply Forecast

Projects per-site IMP demand for the next N weeks from enrolment velocity. Flags sites within 2 weeks of stock-out.

```python
forecast = asyncio.run(svc.imp_supply_forecast(
    tenant_id=TENANT,
    trial_id=trial["id"],
    horizon_weeks=12,
))
# Returns: site_forecasts with reorder_trigger, recommended_resupply_units
```

### eCTD Submission Package Assembly

Organises TMF documents into eCTD m1–m5 module structure and validates completeness against agency requirements.

```python
ectd_pkg = asyncio.run(svc.ectd_submission_package(
    tenant_id=TENANT,
    trial_id=trial["id"],
    agency="FDA",
    submission_type="IND",
    submitted_by="ra-team@acme.com",
))
# Returns: ectd_tree, missing_modules, package_complete
```

---

## Dashboard

```python
summary = svc.dashboard_summary(TENANT)
# Includes: trial_count, patient_count, ae_count, open_crf_queries,
# unresolved_protocol_deviations, susar_candidates, tmf_document_count,
# smc_reports, interim_analyses
```

---

## Configuration Reference

All keys are tenant-scoped and set via the `conf` capability or environment variables prefixed `PHARMA_CTR_`.

| Key | Description | Default |
|-----|-------------|---------|
| `adverse_events.reporting_timeline_hours.sadie` | SADIE deadline (hours) | 24 |
| `adverse_events.reporting_timeline_hours.susar` | SUSAR deadline (days) | 15 |
| `patients.ic_required` | Informed consent mandatory | true |
| `randomisation.ivrs_integration` | Use IVRS for randomisation | true |
| `inspection_readiness.tmf_minimum_docs` | Minimum TMF docs for 100% completeness | 10 |
| `OLLAMA_BASE_URL` | Ollama server URL for LLM-enhanced narratives | unset |

---

## Regulatory Compliance

| Standard | Coverage |
|----------|----------|
| ICH E6(R3) | GCP enforcement at every service boundary |
| ICH E2A/E2B(R3) | Expedited SAR/SUSAR reporting and narrative format |
| ICH E3 | CSR section structure |
| ICH E9(R1) | Adaptive design (RAR, SSR) statistical principles |
| 21 CFR Part 11 | Electronic records audit trail via `audl` capability |
| EU CT Regulation 536/2014 | CTA and EudraCT registration placeholders |
| FDA eCTD guidance | m1–m5 module assembly in `ectd_submission_package()` |

---

## Composability

```apg
use pharma_ctr;
```

- Safety data → `pharma_pvi` (post-market surveillance)
- Protocol amendments → `pharma_qms` (change control)
- Regulatory submissions → `pharma_reg` (approval tracking)
- IMP forecasts → `pharma_mfg` (supply chain scheduling)

---

## Further Reading

- `service.py` — Full business logic implementation
- `models.py` — SQLAlchemy and Pydantic data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views
- `WORLD_CLASS_IMPROVEMENTS.md` — Roadmap of 15 advanced enhancements
- `cap_spec.md` — Detailed capability specification
