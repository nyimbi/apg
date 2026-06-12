# ESG Governance and Digital Forms/eSign Capability (esgn)

`esgn` provides APG's governed digital forms, electronic signature workflows, and ESG governance infrastructure. It composes form-template authoring, schema validation, publication approval, submissions, signature envelopes, ordered signing ceremonies, tamper sealing, encrypted evidence packages, first-class signing agents, ESG reporting pipelines, materiality assessment, stakeholder engagement, climate risk management, and regulatory compliance tracking.

## What It Provides

**Core Forms and eSign**

- Governed form templates with owner, schema, compliance framework, DLP policy, retention policy, publication approval, and audit events.
- Validated form submissions with tenant-scoped evidence references and deterministic validation hashes.
- Signature envelopes with required subject, recipient consent, delegated signing policy checks, document hash, expiry, ordered routing, and tamper seal.
- Signing ceremonies requiring identity verification, explicit signature intent, active envelope state, valid seal, non-expired envelope, correct signer order, and one signature per recipient.
- Envelope cancellation and rejection with mandatory reason and audit event.
- Evidence packages with encrypted seal digest, certificate ID, retention policy, and audit trail reference.
- First-class signing agents with runtime, role, scope, owner, purpose, disclosure, and privileged-role approval controls.
- Bulk signing operations with parallel dispatch and per-submission error isolation.
- Legal review workflow with approval/rejection decisions and audit integration.
- Witness attestation with tamper-evident hashing.
- Conditional logic rules attached to form templates.
- Multi-step signing workflows with ordered step definitions.
- Deadline reminder dispatch to pending signers.
- Compliance posture check against frameworks (eIDAS, ESRS, GRI, TCFD, CSRD, SASB).

**ESG Governance (v2.0)**

- Double materiality assessment engine (GRI-2023 and ESRS-aligned).
- Board-level ESG reporting pipeline from KPI collection to signed disclosure.
- Stakeholder engagement registry with dialogue tracking and coverage scoring.
- Science-Based Targets alignment checker for Scope 1/2/3 emissions.
- TCFD climate risk taxonomy with physical and transition risk entries.
- ESG KPI versioning and full data lineage tracing.
- Regulatory filing calendar with deadline alerts.
- Third-party assurance workflow from scope definition to final opinion.
- Controversy monitoring and ESG risk score time-series.
- Peer benchmarking data model with percentile comparisons.
- Supplier ESG due diligence module with risk-tier classification.
- ESG data quality scoring across completeness, timeliness, consistency, accuracy, and auditability dimensions.
- Integrated scenario analysis against IPCC AR6 and IEA pathways (NZE 2050, SDS, APS).
- ESG governance maturity scoring across eight dimensions with gap analysis.
- Real-time ESG dashboard with configurable KPI alert thresholds.

## Runtime Shape

`service.EsgnService` is deterministic and in-memory. It requires no external services to exercise the complete lifecycle. All mutations emit audit events.

```python
service = EsgnService(actor_id="system", tenant_id="default")
```

### Method Reference

| Method | Description |
|---|---|
| `form_create(tenant_id, template_id, name, owner, schema_fields, ...)` | Create a form template |
| `form_publish(tenant_id, template_id, approved_by)` | Publish template for submissions |
| `form_submit(tenant_id, submission_id, template_id, submitted_by, data, ...)` | Submit a form |
| `signature_request(tenant_id, envelope_id, submission_id, subject, sender, recipients, ...)` | Create and dispatch signature envelope |
| `sign_document(tenant_id, ceremony_id, envelope_id, recipient_id, signature_intent, identity_verified)` | Record signing ceremony |
| `verify_signature(tenant_id, ceremony_id)` | Verify a signing ceremony record |
| `witness_add(tenant_id, witness_id, ceremony_id, witness_name, witness_email, witness_statement)` | Add witness attestation |
| `audit_trail(tenant_id, envelope_id)` | Complete audit trail for an envelope |
| `form_analytics(tenant_id, period)` | Form and signing analytics |
| `template_library(tenant_id)` | List published templates |
| `conditional_logic(tenant_id, rule_id, template_id, conditions, actions)` | Attach conditional rules to template |
| `multi_step_workflow(tenant_id, workflow_id, name, steps, owner)` | Create ordered signing workflow |
| `deadline_reminder(tenant_id, envelope_id, reminder_message)` | Dispatch reminders to pending signers |
| `bulk_sign_request(tenant_id, submission_ids, subject_template, sender, recipients)` | Parallel bulk signing dispatch |
| `legal_review(tenant_id, review_id, envelope_id, reviewer, status, notes)` | Record legal review decision |
| `create_evidence_package(tenant_id, evidence_id, envelope_id, encrypted, retention_policy, ...)` | Create tamper-evident evidence package |
| `cancel_envelope(tenant_id, envelope_id, actor, reason)` | Cancel a signing envelope |
| `reject_envelope(tenant_id, envelope_id, recipient_id, reason)` | Reject a signing envelope |
| `register_signing_agent(tenant_id, agent_id, name, runtime, role, scope_ref, registered_by, ...)` | Register AI signing agent |
| `verify_tamper_seal(tenant_id, envelope_id)` | Verify envelope tamper seal |
| `template_archive(tenant_id, template_id, actor)` | Archive a form template |
| `submission_withdraw(tenant_id, submission_id, reason)` | Withdraw a form submission |
| `bulk_create_templates(tenant_id, templates)` | Parallel bulk template creation |
| `compliance_check(tenant_id, framework)` | Check signing compliance posture |
| `dashboard_summary(tenant_id)` | Aggregate dashboard metrics |
| `health_check()` | Service health probe |
| `export_csv(tenant_id, collection)` | Export collection as CSV |
| `export_json(tenant_id, collection)` | Export collection as JSON |
| `list_templates / list_submissions / list_envelopes / ...` | Collection list helpers |

Compat aliases `create_template`, `publish_template`, `submit_form` mirror the legacy positional signature.

## Quick Start

```python
import asyncio
from capabilities.common.esgn.service import EsgnService

async def main():
    svc = EsgnService()

    # 1. Create and publish a template
    await svc.form_create(
        "tenant-1", "tpl-nda", "Mutual NDA", "legal-ops",
        ["counterparty", "effective_date"],
        compliance_framework="esign-act",
        dlp_policy="regulated-field-scan",
        retention_policy="legal-7y",
    )
    await svc.form_publish("tenant-1", "tpl-nda", "legal-approver")

    # 2. Submit a form
    sub = await svc.form_submit(
        "tenant-1", "sub-1", "tpl-nda", "sales-ops",
        {"counterparty": "Acme Ltd", "effective_date": "2026-05-30"},
    )

    # 3. Create and complete a signature envelope
    await svc.signature_request(
        "tenant-1", "env-1", "sub-1", "Mutual NDA signature", "sales-ops",
        [{"id": "rcp-1", "name": "Ada", "email": "ada@example.com",
          "routing_order": 1, "consent_recorded": True}],
    )
    await svc.sign_document("tenant-1", "cer-1", "env-1", "rcp-1", "approve_nda", True)

    # 4. Package evidence
    evidence = await svc.create_evidence_package(
        "tenant-1", "evd-1", "env-1", True, "legal-7y",
    )

    # 5. Dashboard
    summary = await svc.dashboard_summary("tenant-1")
    print(summary)

asyncio.run(main())
```

## World-Class Enhancements (v2.0)

All 15 improvements are modelled in `models.py` and governed by `capability_contract.py`. The ESG pipeline methods integrate with the existing form/signing infrastructure via shared `_Store` and `_Audit` primitives.

| # | Enhancement | Summary |
|---|---|---|
| 1 | **Double Materiality Assessment** | GRI-2023/ESRS-aligned matrix: impact materiality × financial materiality per topic, controversy flags, time-horizon labels, full reasoning chain for assurance |
| 2 | **Board ESG Reporting Pipeline** | KPI collect → framework validation (GRI, SASB, TCFD, CSRD) → board pack (JSON + summary) → approval routing → signed evidence record |
| 3 | **Stakeholder Engagement Registry** | Stakeholder group registry with engagement method, frequency, topics raised, response commitments, coverage score, stale-relationship flagging |
| 4 | **Science-Based Targets Alignment** | Ingest Scope 1/2/3 emissions, compare against SBT pathways (1.5°C / well below 2°C), flag off-track years, project to target year |
| 5 | **TCFD Climate Risk Taxonomy** | Physical (acute/chronic) and transition (policy, technology, market, reputational) risk registry; likelihood × impact × financial exposure × mitigation linkage; auto-generates TCFD disclosure section |
| 6 | **ESG KPI Versioning and Data Lineage** | Every KPI submission carries semantic version, source, methodology, and transformation log; full trace from board-pack figure to raw measurement |
| 7 | **Regulatory Filing Calendar** | Tenant-configurable deadlines (CSRD, SEC, TCFD, GRI, CDP, UNGC COP) with jurisdiction, required artifacts, responsible party, and automated reminders |
| 8 | **Third-Party Assurance Workflow** | Scope definition → evidence request → reviewer assignment → finding management → management response → opinion issuance with tamper-evident seal |
| 9 | **Controversy Monitoring and ESG Risk Score** | Controversy event registry (media, litigation, regulator, NGO) with severity, pillar impact, response status; aggregates into ESG risk score time-series |
| 10 | **Peer Benchmarking** | Store peer group ESG scores by industry/geography/size-band; compare against percentile bands; surface gaps and outperformance in board reporting |
| 11 | **Supplier ESG Due Diligence** | Supplier register with risk tier, assessment cadence, questionnaire responses, red-flag conditions, remediation plans; portfolio-level supply-chain exposure |
| 12 | **ESG Data Quality Scoring** | Per-KPI-batch quality score across completeness, timeliness, consistency, accuracy, and auditability; low-quality data triggers validation workflow |
| 13 | **Integrated Scenario Analysis** | Bind metrics to IPCC AR6 and IEA pathways; compute performance vs. milestones at 2025/2030/2040/2050; versioned scenario parameters |
| 14 | **ESG Governance Maturity Scoring** | Five maturity levels across eight governance dimensions; gap analysis and improvement roadmap |
| 15 | **Real-Time ESG Dashboard with Alerts** | Configurable warning/critical KPI thresholds; threshold breaches emit structured alerts (email, webhook, Slack, audit log); single-call aggregate dashboard |

## New Methods

### `bulk_sign_request` — parallel multi-document signing

```python
results = await svc.bulk_sign_request(
    tenant_id="tenant-1",
    submission_ids=["sub-1", "sub-2", "sub-3"],
    subject_template="Board approval required: {sid}",
    sender="board-sec",
    recipients=[
        {"id": "dir-1", "name": "Director A", "email": "a@corp.com",
         "routing_order": 1, "consent_recorded": True},
    ],
)
# Each element carries status "ok" or "failed" + error detail.
```

### `compliance_check` — signing compliance posture

```python
report = await svc.compliance_check("tenant-1", framework="eIDAS")
# {
#   "passed": True,
#   "issues": [],
#   "envelope_count": 12,
#   "ceremony_count": 18,
#   "framework": "eIDAS",
# }
```

### `witness_add` — witness attestation on a signing ceremony

```python
witness = await svc.witness_add(
    tenant_id="tenant-1",
    witness_id="wit-1",
    ceremony_id="cer-1",
    witness_name="Jean Paul",
    witness_email="jp@notary.com",
    witness_statement="I confirm the signatory signed in my presence.",
)
# Returns record with attestation_hash for tamper detection.
```

### `audit_trail` — complete lifecycle trace for an envelope

```python
trail = await svc.audit_trail("tenant-1", "env-1")
# {
#   "ceremony_count": 2,
#   "witness_count": 1,
#   "audit_event_count": 9,
#   "ceremonies": [...],
#   "witnesses": [...],
#   "audit_events": [...],
# }
```

### `export_csv` — data extract for any collection

```python
csv_text = await svc.export_csv("tenant-1", "esgn_submissions")
# Returns CSV string with headers from first record's keys.
# Supported collections: esgn_submissions, esgn_envelopes, esgn_ceremonies,
# esgn_evidence_packages, esgn_agents, esgn_audit, etc.
```

## Configuration and Rules

`capability_contract.py` is the source of truth for:

- Configuration defaults and schema
- Deterministic rules (rule engine returns `allow`, `require_review`, or `deny`)
- UI route contracts and theme tokens
- APG adapter map and Bytewax streaming contract
- First-class signing-agent manifest
- Bytewax lifecycle-batch manifest

## UI Surfaces

Route contracts exposed by the package:

- dashboard, forms, builder, submissions, envelopes, signing
- agents, lifecycle, evidence, audit, analytics, settings

`views.py` provides dependency-light models for all screens.

## Verification

```bash
./.venv/bin/python -m py_compile \
    capabilities/common/esgn/__init__.py \
    capabilities/common/esgn/capability_contract.py \
    capabilities/common/esgn/models.py \
    capabilities/common/esgn/service.py \
    capabilities/common/esgn/api.py \
    capabilities/common/esgn/views.py \
    capabilities/common/esgn/app.py

./.venv/bin/pytest -q \
    capabilities/common/esgn/test_capability_contract.py \
    capabilities/common/esgn/tests/test_package_contract.py

./.venv/bin/apg capabilities implementation-audit --root capabilities/common/esgn --json
./.venv/bin/apg capabilities publish-plan capabilities/common/esgn --json
```
