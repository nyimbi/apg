# Pharmacovigilance Intelligence — User Guide

**Capability ID**: `pharma_pvi` | **Domain**: `pharma` | **Version**: `1.1.0`
**© 2025 Datacraft** | Author: Nyimbi Odero

---

## Description

Manages the complete pharmacovigilance lifecycle: adverse event intake, MedDRA coding, ICSR submission (EudraVigilance, FDA FAERS, WHO VigiBase), safety signal management, PSUR/PBRER/DSUR generation, SUSAR line listings, RMP safety concern linkage, and regulatory timeline compliance enforcement.

Enforces ICH E2B(R3), ICH E2A, ICH E2C(R2), ICH E2F, EMA GVP Module VI, and FDA 21 CFR Part 314/600 reporting obligations.

---

## Installation

```bash
pip install apg-pharma-pvi
```

---

## Quick Start

```python
from capabilities.pharma.pvi.service import PharmacovigilanceService
from capabilities.pharma.pvi.models import AdvEventCaseCreate
from datetime import datetime

svc = PharmacovigilanceService(tenant_id="acme", actor_id="pv_officer_1")

# 1. Create a case
case = svc.create_case(AdvEventCaseCreate(
    tenant_id="acme",
    case_number="AE-2025-001",
    source="spontaneous",
    case_type="serious_adverse_reaction",
    product_id="DRUG-X",
    suspect_drug="droxinifib 100mg",
    report_date=datetime.utcnow(),
    serious=True,
    created_by="pv_officer_1",
))

# 2. MedDRA code + narrative
case = svc.process_case(
    case.id, "acme",
    narrative="Patient reported severe rash...",
    causality="probable",
    meddra_pt="rash",
    meddra_soc="Skin and subcutaneous tissue disorders",
    processed_by="pv_officer_1",
)

# 3. AI narrative draft (async)
import asyncio
result = asyncio.run(svc.generate_case_narrative(case.id, "acme"))
print(result["narrative"])  # ICH E2B(R3) G.k.9 compliant draft

# 4. Submit ICSR
submission = svc.submit_icsr(
    tenant_id="acme",
    case_id=case.id,
    regulatory_database="eudravigilance",
    submission_type="expedited",
    due_date=datetime.utcnow(),
    e2b_r3_formatted=True,
    created_by="pv_officer_1",
)
```

---

## Core Workflows

### Adverse Event Intake

```python
# High-level intake (returns dict, no model validation required)
ae = svc.report_adverse_event(
    drug_id="DRUG-X",
    patient_demographics={"age": 45, "sex": "F"},
    event_description="Severe hepatotoxicity",
    causality="probable",
    seriousness="serious",
    outcome="recovering",
    tenant_id="acme",
    source="spontaneous",
    meddra_pt="hepatocellular_injury",
)
# ae["icsr_required"] == True, ae["reporting_deadline"] computed
```

### Case Triage

```python
triage = svc.case_triage(case.id, "susar", "acme")
# triage["icsr_deadline_days"] == 7 for SUSAR
# triage["recommended_workflow"] == "expedited"
```

### Duplicate Detection (async)

```python
dups = asyncio.run(svc.auto_detect_duplicates(
    case_id=case.id,
    tenant_id="acme",
    similarity_threshold=0.75,
))
for candidate in dups["potential_duplicates"]:
    print(candidate["candidate_case_id"], candidate["similarity_score"])
```

Similarity components:
| Field | Weight |
|-------|--------|
| suspect_drug (exact) | 30% |
| meddra_pt (exact) | 30% |
| onset_date (±7 days) | 15% |
| patient_age (±5y) | 15% |
| patient_sex | 10% |

### Timeline Compliance Check (async)

```python
compliance = asyncio.run(svc.check_timeline_compliance(tenant_id="acme"))
print(compliance["compliance_score"])   # 0–100
print(compliance["breach_count"])        # submissions past due_date
for b in compliance["breaches"]:
    print(b["case_id"], b["days_delta"], b["severity"])
```

Breach severity: `critical` (>3 days late), `warning` (0–3 days late).

### Batch ICSR Submission (async)

```python
result = asyncio.run(svc.batch_submit_icsrs(
    tenant_id="acme",
    case_ids=["case-1", "case-2", "case-3"],
    regulatory_database="eudravigilance",
    created_by="pv_officer_1",
))
print(result["submitted_count"], result["failed_count"])
# failed cases returned with error detail for targeted retry
```

---

## Signal Management

### Create and Evaluate

```python
signal = svc.create_signal(
    tenant_id="acme", signal_number="SIG-001", product_id="DRUG-X",
    signal_type="disproportionality", meddra_pt="hepatocellular_injury",
    description="ROR 4.2 (95% CI 2.1–8.4), n=12",
    detected_by="pv_officer_1", detection_method="ror_analysis",
    created_by="pv_officer_1",
)
signal = svc.evaluate_signal(
    signal.id, "acme",
    strength_of_evidence="strong",
    clinical_review_reference="CLR-2025-042",
)
```

### Statistical Disproportionality

```python
detection = svc.signal_detection(
    drug_id="DRUG-X",
    event_terms=["hepatocellular_injury", "rash", "nausea"],
    analysis_period="2025-H1",
    tenant_id="acme",
    method="disproportionality",
    threshold_ror=2.0,
)
for r in detection["results"]:
    if r["signal_detected"]:
        print(r["event_term"], "ROR:", r["ror"])
```

### RMP Safety Concern Linkage (async)

```python
update = asyncio.run(svc.update_rmp_safety_concern(
    rmp_id="RMP-DRUG-X-v3",
    concern_id="SC-HEPA-01",
    signal_id=signal.id,
    tenant_id="acme",
    concern_type="identified_risk",
    rationale="ROR 4.2 meets EMA GVP Module IX threshold for signal confirmation",
    updated_by="qppv_1",
))
# emits rmp_update_required event to pharma_reg
```

---

## PSUR / PBRER Generation

```python
from datetime import timedelta

psur = svc.create_psur(
    tenant_id="acme", report_number="PSUR-2025-Q2", product_id="DRUG-X",
    report_type="pbrer",
    data_lock_point=datetime.utcnow(),
    international_birth_date=datetime(2020, 1, 15),
    period_start=datetime(2025, 1, 1),
    period_end=datetime(2025, 6, 30),
    ibrd_reference="IBD-2020-001",
    created_by="pv_officer_1",
)

# Check EMA EURD deadline before submission
deadline = asyncio.run(svc.psur_eurd_deadline_check(
    tenant_id="acme",
    drug_id="DRUG-X",
    active_substance="droxinifib",
    ibrd="2020-01-15",
    warn_days=90,
))
print(deadline["days_until_deadline"], deadline["urgency"])
# urgency: ok | warning | critical | overdue

# Submit after B/R assessment
psur = svc.submit_psur(psur.id, "acme", benefit_risk_assessed=True)
```

---

## DSUR and SUSAR Line Listings (Clinical Trial PV)

### DSUR (ICH E2F)

```python
dsur = asyncio.run(svc.generate_dsur(
    drug_id="DRUG-X",
    trial_id="CT-2025-PHASE3",
    period="2025-annual",
    tenant_id="acme",
    ibrd="2020-01-15",
    executive_summary="No new safety signals identified in the reporting period.",
))
# dsur["sections"] — 13-section ICH E2F structure
# dsur["susar_line_listing_required"] — True if serious CT cases present
```

### SUSAR Line Listing

```python
listing = asyncio.run(svc.generate_susar_line_listing(
    trial_id="CT-2025-PHASE3",
    tenant_id="acme",
    product_id="DRUG-X",
    format="eudraCT",
))
for entry in listing["listing"]:
    print(entry["case_number"], entry["meddra_pt"], entry["causality"])
```

Supported formats: `eudraCT`, `ctis`, `csv`.

---

## AI Narrative Generation

Requires `OLLAMA_BASE_URL` environment variable. Falls back to structured template if Ollama is unavailable.

```python
import os
os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"

result = asyncio.run(svc.generate_case_narrative(
    case_id=case.id,
    tenant_id="acme",
    model="llama3.1:8b",
))
print(result["narrative"])          # ICH E2B(R3) G.k.9 compliant
print(result["ai_generated"])       # True if Ollama responded
print(result["requires_medical_review"])  # Always True
```

Narratives are stored as drafts on the case (`AdvEventCase.narrative`) with medical reviewer sign-off required before ICSR submission.

---

## Literature Monitoring

```python
# Record a new article
record = svc.record_literature(
    tenant_id="acme",
    database_source="pubmed",
    article_reference="PMID:38901234",
    title="Hepatotoxicity associated with droxinifib: a case series",
    created_by="lit_screener_1",
    authors="Smith J, Jones A",
    publication_date=datetime(2025, 5, 10),
)

# Assess as relevant and link to product
record = svc.mark_literature_relevant(record.id, "acme", "med_reviewer_1", "DRUG-X")

# List relevant records only
relevant = svc.list_literature("acme", relevant_only=True)
```

---

## Regulatory Submissions

### EudraVigilance

```python
ev_result = svc.submit_to_eudravigilance(case.id, "acme")
# ev_result["acknowledgement_pending"] == True
```

### FDA FAERS

```python
fda_result = svc.submit_to_fda_aers(case.id, "acme")
# fda_result["form"] == "medwatch_3500a"
```

---

## PV System Audit

```python
audit = svc.pv_audit("2025-H1", "acme", auditor_id="auditor_1")
print(audit["compliance_score"])    # 0–100 composite score
print(audit["late_icsr_submissions"])
print(audit["open_signals"])
print(audit["pending_follow_ups"])
```

---

## Dashboard

```python
summary = svc.dashboard_summary("acme")
# Keys: case_count, open_cases, serious_cases, icsr_submission_count,
#       signal_count, open_signals, psur_count, pending_follow_ups,
#       literature_count, label_proposals, pbrer_reports, audit_event_count
```

---

## Streaming Events

All operations emit structured audit events to `apg.pharma.pvi.lifecycle`:

| Event | Trigger |
|-------|---------|
| `ae_received` | AE intake |
| `case_created` | Case creation |
| `case_processed` | MedDRA coding complete |
| `case_closed` | Case closure |
| `duplicate_detected` | Duplicate linkage |
| `duplicate_detection_run` | Auto-duplicate scan |
| `timeline_breach_detected` | ICSR past due date |
| `icsr_submitted` | ICSR submission |
| `batch_icsr_submission_completed` | Batch ICSR done |
| `signal_detected` | New signal identified |
| `signal_evaluated` | Signal clinical review |
| `signal_closed` | Signal closure |
| `narrative_drafted` | AI narrative created |
| `dsur_generated` | DSUR report created |
| `susar_line_listing_generated` | SUSAR listing created |
| `rmp_safety_concern_updated` | RMP signal linkage |
| `rmp_update_required` | RMP version bump needed |
| `psur_deadline_approaching` | EURD deadline warning |
| `psur_submitted` | PSUR/PBRER submission |
| `literature_screened` | Literature record recorded |
| `literature_match_found` | Relevant article identified |
| `medical_review_completed` | Medical review done |
| `label_update_proposed` | Label change proposal |
| `pv_audit_completed` | System audit run |

---

## Configuration

All keys are tenant-scoped. Set via `conf` capability or env vars prefixed `PHARMA_PVI_`.

| Key | Default | Description |
|-----|---------|-------------|
| `case_processing.reporting_timelines.7day_expedited` | `7` | SUSAR deadline (days) |
| `case_processing.reporting_timelines.15day_expedited` | `15` | Serious AE deadline (days) |
| `literature.screening_frequency_days` | `7` | Literature DB scan interval |
| `psur.submission_timeline_days` | `70` | Days after DLP for PSUR submission |
| `signal_detection.ror_threshold` | `2.0` | Default ROR signal threshold |
| `signal_detection.min_cases` | `3` | Minimum cases for signal |
| `narrative.ollama_model` | `llama3.1:8b` | Model for narrative generation |
| `duplicate.similarity_threshold` | `0.75` | Auto-duplicate proposal threshold |

---

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/pharma-pvi/dashboard` | `pharma_pvi:view` | Overview |
| `/pharma-pvi/cases/intake` | `pharma_pvi:cases` | Cases |
| `/pharma-pvi/cases` | `pharma_pvi:cases` | Cases |
| `/pharma-pvi/cases/<id>` | `pharma_pvi:cases` | Cases |
| `/pharma-pvi/cases/follow-up` | `pharma_pvi:follow_up` | Cases |
| `/pharma-pvi/signals` | `pharma_pvi:signals` | Signal Detection |
| `/pharma-pvi/signals/<id>` | `pharma_pvi:signals` | Signal Detection |
| `/pharma-pvi/literature` | `pharma_pvi:literature` | Literature |
| `/pharma-pvi/psur` | `pharma_pvi:psur` | Periodic Reports |
| `/pharma-pvi/dsur` | `pharma_pvi:psur` | Periodic Reports |
| `/pharma-pvi/rmp` | `pharma_pvi:signals` | Risk Management |
| `/pharma-pvi/audit` | `pharma_pvi:audit` | Compliance |

---

## Further Reading

- `service.py` — Business logic and all async methods
- `models.py` — Pydantic v2 data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 detailed enhancement specifications
- `README.md` — Quick reference and streaming events
- `cap_spec.md` — Capability specification
