# Commercial Operations & Pharmacovigilance — User Guide

**Capability ID**: `pharma_com` | **Domain**: `pharma` | **Version**: `1.1.0`

## Description

Manages pharmaceutical field force activities including territory management, sales rep assignments, physician call recording, PDMA-compliant sample dispensing, HCP interaction tracking, aggregate spend management, and commercial planning. Enforces Sunshine Act / CMS Open Payments reporting and PDMA compliance rules at every transactional boundary.

Also provides a pharmacovigilance (PV) layer: ICSR lifecycle management, MedDRA term coding, adverse event signal detection (Reporting Odds Ratio), regulatory submission packaging for EMA/FDA/PMDA, duplicate ICSR detection, signal triage scoring, Open Payments report generation, and CAPA management.

## Installation

```bash
pip install apg-pharma-com
```

## Provides

- `territory_management_workflow`
- `sales_rep_management_workflow`
- `call_activity_workflow`
- `sample_management_workflow`
- `hcp_interaction_workflow`
- `icsr_lifecycle_workflow`
- `signal_detection_workflow`
- `meddra_coding_workflow`
- `regulatory_submission_workflow`
- `open_payments_workflow`
- `signal_triage_workflow`
- `duplicate_detection_workflow`
- `capa_workflow`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`
- `qms` (CAPA routing)
- `pvi` (signal handoff)

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/pharma-com/dashboard` | `pharma_com:view` | Overview |
| `/pharma-com/territories` | `pharma_com:territories` | Territory |
| `/pharma-com/territories/<id>` | `pharma_com:territories` | Territory |
| `/pharma-com/reps` | `pharma_com:reps` | Field Force |
| `/pharma-com/calls` | `pharma_com:calls` | Field Force |
| `/pharma-com/samples` | `pharma_com:samples` | Samples |
| `/pharma-com/samples/reconcile` | `pharma_com:samples_admin` | Samples |
| `/pharma-com/interactions` | `pharma_com:interactions` | HCP Engagement |
| `/pharma-com/pv/icsrs` | `pharma_com:pv` | Pharmacovigilance |
| `/pharma-com/pv/signals` | `pharma_com:pv` | Pharmacovigilance |
| `/pharma-com/pv/submissions` | `pharma_com:submissions` | Pharmacovigilance |
| `/pharma-com/reporting/open-payments` | `pharma_com:reporting` | Reporting |
| `/pharma-com/capa` | `pharma_com:capa` | Quality |

## Key Service Methods

### Commercial Operations (synchronous)

- `describe()` — Return capability contract
- `evaluate()` — Evaluate rules against a context
- `create_territory()` / `get_territory()` / `list_territories()` / `update_territory()`
- `assign_rep()` / `get_rep()` / `list_reps()` / `list_reps_by_territory()`
- `record_call()` / `list_calls()` / `list_calls_by_physician()`
- `dispense_sample()` / `list_samples()` / `reconcile_samples()`
- `record_interaction()` / `list_interactions()`
- `record_spend()` / `get_aggregate_spend_summary()`
- `create_plan()` / `approve_plan()` / `list_plans()`
- `set_target()` / `list_targets()`
- `territory_assignment()` — Reassign rep to a different territory
- `call_plan()` — Generate a rep's call plan for a period
- `hcp_visit_record()` — Record a complete HCP visit with samples and detailing
- `pdma_compliance_check()` — Validate a visit for PDMA/EFPIA compliance
- `sample_management()` — Manage sample inventory transactions
- `spend_tracking()` — Track HCP spend with cap enforcement
- `prescriber_analytics()` — Territory coverage and sample rate analytics
- `market_access_tracking()` / `get_market_access_by_product()`
- `promotional_material_approval()` — MLR workflow for promotional materials
- `commercial_analytics()` — Comprehensive commercial performance report
- `dashboard_summary()` — Field force KPI dashboard

### Pharmacovigilance (async)

- `create_icsr()` — Create an Individual Case Safety Report (ICH E2B R3 aligned)
- `encode_meddra_term()` — Map verbatim adverse event term to MedDRA PT/LLT code
- `detect_adverse_event_signals()` — ROR disproportionality signal detection
- `initiate_regulatory_submission()` — Package ICSRs for EMA/FDA/PMDA submission
- `generate_open_payments_report()` — CMS Sunshine Act annual report
- `compute_signal_triage_score()` — Composite priority score (0–100) with tier
- `detect_duplicate_icsrs()` — Probabilistic ICSR deduplication
- `create_capa()` — Open a CAPA record from a compliance violation

### Utility (async)

- `export_records()` — Export tenant records as JSON or CSV
- `health_check()` — Service liveness probe
- `compliance_report()` — GxP compliance status summary
- `bulk_create_records()` — Batch record creation
- `analytics_summary()` — High-level analytics by period

## Quick-Start Examples

### Create an ICSR and detect signals

```python
import asyncio
from apg_pharma_com import CommercialOperationsService

svc = CommercialOperationsService(tenant_id="acme")

async def main():
    # Create an ICSR
    icsr = await svc.create_icsr(
        tenant_id="acme",
        reporter_type="spontaneous",
        reporter_id="hcp-001",
        patient_age=45,
        patient_sex="F",
        suspect_products=["prod-atorvastatin-20mg"],
        adverse_reactions=["myalgia", "elevated CK"],
        reaction_onset_date="2025-03-10",
        seriousness_criteria=["hospitalisation"],
        causality_assessment="probable",
        created_by="pv-officer-001",
    )
    print(icsr["id"], icsr["status"])  # -> <uuid7> draft

    # Encode a reaction term to MedDRA
    coded = await svc.encode_meddra_term("headache", "acme")
    print(coded["meddra_code"], coded["matched_term"])  # -> 10019211 Headache

    # Run ROR signal detection
    signal = await svc.detect_adverse_event_signals(
        tenant_id="acme",
        product_id="prod-atorvastatin-20mg",
        reaction_term="myalgia",
        ror_threshold=2.0,
        min_case_count=3,
    )
    print(signal["signal_detected"], signal["signal_strength"])

asyncio.run(main())
```

### Submit an ICSR to the FDA

```python
async def submit():
    submission = await svc.initiate_regulatory_submission(
        tenant_id="acme",
        icsr_ids=[icsr["id"]],
        authority="FDA",
        submission_type="expedited_15day",
        submission_deadline="2025-03-25T00:00:00Z",
        prepared_by="pv-officer-001",
    )
    print(submission["tracking_number"])  # -> FDA-<uuid7-prefix>
```

### Generate Open Payments report

```python
async def report():
    rpt = await svc.generate_open_payments_report("acme", 2025)
    print(rpt["total_amount"], rpt["report_ready"])
```

### Open a CAPA from a PDMA violation

```python
async def capa():
    record = await svc.create_capa(
        tenant_id="acme",
        violation_type="pdma_breach",
        violation_reference="visit-abc123",
        root_cause="Rep dispensed samples without obtaining HCP signature.",
        corrective_action="Retrieve retroactive signature from HCP within 48 hours.",
        preventive_action="Add mandatory signature capture step to mobile CRM before sample screen closes.",
        responsible_person_id="rep-mgr-007",
        due_date="2025-04-01",
        created_by="compliance-officer-001",
        priority="high",
    )
    print(record["id"], record["status"])  # -> <uuid7> open
```

## Interoperability

`pharma_com` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use pharma_com;
```

Downstream integrations:
- `pharma_rec` — Sunshine Act reporting data feed
- `grc` — Compliance data aggregation
- `pharma_sup` — Territory forecast feeds demand planning
- `pvi` — PV signal handoff for case management
- `qms` — CAPA lifecycle management
- `intel` — Signal triage dashboard widgets

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `PHARMA_COM_`.

| Environment Variable | Config Key | Default |
|---------------------|-----------|---------|
| `PHARMA_COM_SPEND_CAP` | `compliance.aggregate_spend_cap` | `500.0` |
| `PHARMA_COM_RECEIPT_THRESHOLD` | `spend.receipt_required_above` | `25.0` |
| `PHARMA_COM_APPROVAL_THRESHOLD` | `spend.pre_approval_required_above` | `100.0` |
| `PHARMA_COM_PV_ROR_THRESHOLD` | `pv.ror_threshold` | `2.0` |
| `PHARMA_COM_PV_MIN_CASES` | `pv.min_case_count` | `3` |
| `PHARMA_COM_MEDDRA_RELEASE` | `pv.meddra_release` | `26.1` |

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Pydantic v2 data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views
- `README.md` — Quick reference
- `WORLD_CLASS_IMPROVEMENTS.md` — Prioritised improvement roadmap
- `SPECIFICATION.md` — Detailed capability specification
- `cap_spec.md` — Capability contract specification
