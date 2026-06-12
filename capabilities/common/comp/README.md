# Compliance Management

`comp` is APG's package-backed Compliance Management capability. It gives
generated applications a tenant-scoped compliance runtime for frameworks,
obligations, controls, encrypted evidence, assessments, findings, remediation,
reports, attestations, governed AI agents, Bytewax lifecycle batches, audit
events, UI route metadata, and visual theme metadata.

The package is dependency-light. It proves compliance lifecycle behavior and
guardrails locally, while live GRC suites, audit-log sinks, DLP engines,
identity systems, workflow tools, evidence repositories, and regulator feeds
remain APG adapter boundaries.

## What It Provides

- Framework registration with owner, obligation, and policy-version evidence.
- Control library ownership, testing cadence, DLP linkage, and regulated-data
  scope.
- Encrypted evidence records with immutable references and collector/source
  metadata.
- Control assessments with freshness checks and independent tester review.
- Finding opening, escalation, remediation tracking, and resolution evidence.
- Report preparation, independent approval, attestation, publication, and
  critical-finding blocking.
- Gap assessment: identifies controls without recent effective assessments.
- Risk integration: links external risk scores to compliance controls.
- Policy enforcement recording with full audit trail.
- Regulatory change alert ingestion and tracking.
- Compliance training assignment with due-date tracking.
- Audit scheduling for internal and external audits.
- Framework-specific coverage helpers: ISO 27001 checklist, GDPR DPIA, SOC 2
  evidence aggregation.
- Obligation registration against frameworks with regulatory source tracking.
- Control-to-assessment mapping for a full framework view.
- First-class provider-neutral compliance agents for `codex`, `claude_code`,
  `opencode`, and `pi`.
- Human-review guardrails for privileged agent roles such as report reviewers,
  attestation reviewers, regulatory export reviewers, lifecycle reviewers, and
  compliance stewards.
- Bytewax-only lifecycle batch validation for framework, control, evidence,
  assessment, finding, report, attestation, exception, and compliance-agent
  mutations.
- Tenant isolation for repeated business IDs across tenants.
- Hashed audit events for compliance state changes.
- UI view models for dashboard, frameworks, controls, evidence, assessments,
  findings, reports, attestations, agents, lifecycle batches, audit, and
  settings.
- Contract-derived semantic model, package manifest, release report, and
  publish-plan support.

## Main Files

| File | Purpose |
| --- | --- |
| `SPECIFICATION.md` | Functional, lifecycle, rule, UI, adapter, and acceptance specification. |
| `PLAN.md` | Implementation and review plan for this capability packet. |
| `capability_contract.py` | Executable configuration, rule engine, UI routes, theme, and adapter contract. |
| `compliance_engine.py` | Deterministic digest, age, assessment, and coverage helpers. |
| `models.py` | Dataclass records and data contracts. |
| `service.py` | In-memory compliance lifecycle and guardrail enforcement. |
| `api.py` | Dependency-light helper surface for generated applications. |
| `views.py` | View-model payloads for generated APG UIs. |
| `app.py` | Package entrypoint, semantic model, component manifest, and self-test. |

## Runtime Flow

1. Register a framework with mapped obligations and policy version.
2. Create controls under that framework.
3. Record encrypted immutable evidence for controls.
4. Assess controls against evidence freshness and open findings.
5. Open, escalate, track, and resolve findings.
6. Run gap assessments to identify coverage holes.
7. Prepare a report for a framework and period.
8. Approve the report with an independent approver.
9. Attest the approved report.
10. Publish the report if approval, attestation, and critical-finding guardrails
    pass.
11. Register scoped AI agents for compliance review, automation, and stewarding.
12. Validate batch lifecycle mutations through Bytewax before composing larger
    applications.

## Python Usage

```python
from capabilities.common.comp.service import CompService

service = CompService()

framework = service.register_framework(
	"fw-soc2",
	"tenant-a",
	"SOC 2",
	"risk-owner",
	["CC6.1", "CC7.2"],
	"2026.1",
)
control = service.create_control(
	"ctrl-access-review",
	"tenant-a",
	framework["id"],
	"Quarterly access review",
	"identity-owner",
	regulated_data_scope=True,
	dlp_policy_linked=True,
)
evidence = service.record_evidence(
	"ev-access-review",
	"tenant-a",
	control["id"],
	"access-review-export",
	"auditor",
	encrypted=True,
	immutable_reference="sha256:access-review",
)
assessment = service.assess_control(
	"assess-access-review",
	"tenant-a",
	control["id"],
	evidence["id"],
	"control-tester",
)
```

## AI Agent Composition

Compliance agents are first-class records. They are not hard-wired to one model
or CLI. The contract accepts provider-neutral runtimes:

- `codex`
- `claude_code`
- `opencode`
- `pi`

Every agent must declare a runtime, supported role, explicit scope, accountable
owner, purpose, and machine-contribution disclosure. Privileged roles enter
`pending_review` unless human approval evidence is recorded.

```python
agent = service.register_compliance_agent(
	"agent-steward",
	"tenant-a",
	"Compliance Steward",
	"codex",
	"compliance_steward",
	"framework:fw-soc2",
	"risk-owner",
	"review framework posture before attestation",
	human_approval_required=True,
)
```

## Bytewax Lifecycle Batches

Batch mutations must use Bytewax. The local package validates stream metadata
and records accepted or denied lifecycle-batch evidence without starting a live
Bytewax topology.

```python
batch = service.validate_comp_lifecycle_batch(
	"tenant-a",
	"bytewax",
	3,
	"compliance_agent_batch",
	"batch-agent-001",
)
```

## Reporting And Attestation

```python
report = service.prepare_report("report-soc2-q1", "tenant-a", "fw-soc2", "2026-Q1", "compliance-lead")
approved = service.approve_report(report["id"], "tenant-a", "risk-committee")
attestation = service.attest_report(
	"attest-soc2-q1",
	report["id"],
	"tenant-a",
	"chief-risk-officer",
	"Control evidence and known findings have been reviewed.",
)
published = service.publish_report(report["id"], "tenant-a")
```

## Contract And Composition

```python
from capabilities.common.comp.capability_contract import get_capability_contract

contract = get_capability_contract("tenant-a")
routes = contract["ui"]["routes"]
rules = contract["rule_engine"]["rules"]
adapters = contract["configuration"]["adapters"]
```

Important adapters:

- `generated_app_runtime`: `service.CompService`
- `helper_runtime`: `compliance_engine.py`
- `api_helpers`: `api.py`
- `view_models`: `views.py`
- `event_stream`: `bytewax`
- `agent_adapter`: `aicr_provider_neutral_compliance_agent_adapter`
- `audit_sink`: `audl`
- `data_loss_prevention`: `dlpd`
- `encryption`: `encr`
- `authentication`: `auth`
- `workflow`: `wflo`
- `notification`: `ntfy`

## UI Surfaces

The contract exposes these route names:

- `dashboard`
- `frameworks`
- `controls`
- `evidence`
- `assessments`
- `findings`
- `exceptions`
- `reports`
- `attestations`
- `exports`
- `audit`
- `agents`
- `lifecycle`
- `settings`

`views.py` returns data-only models for generated UIs. The generated UI should
render the provided theme tokens and component names instead of hard-coding
colors or layout assumptions.

## World-Class Enhancements (v2.0)

The following 15 improvements define the production roadmap from the current
in-memory reference implementation to a hardened GRC platform:

1. **Async-First Service Layer** — Convert all methods to `async def` for
   non-blocking I/O against PostgreSQL, audit sinks, and evidence repositories.

2. **Persistent PostgreSQL Backend via SQLAlchemy Async** — Replace in-memory
   dicts with async SQLAlchemy sessions, Alembic migrations, JSONB audit
   payloads, and row-level tenant isolation via RLS policies.

3. **Continuous Control Monitoring (CCM)** — Background task that
   periodically re-evaluates evidence freshness, open findings, and
   testing-frequency SLAs, emitting `control_degraded` events when coverage
   slips below threshold.

4. **Risk-Adjusted Control Prioritisation** — Scoring engine combining
   likelihood × impact, control effectiveness, residual risk, and regulatory
   weight to produce a prioritised remediation queue, integrated with the APG
   `risk` capability.

5. **Automated Evidence Collection via Adapters** — `evidence_collector`
   adapter interface polling cloud config scanners, SIEM exports, identity
   governance APIs, and CI/CD pipelines to auto-record fresh encrypted evidence.

6. **Cross-Framework Control Mapping and Reuse** — `cross_framework_map`
   operation detecting overlapping obligations across SOC 2, ISO 27001, GDPR,
   NIST CSF, and PCI-DSS, reducing duplicated assessments and evidence
   collection by up to 60%.

7. **Machine-Readable Regulatory Change Feed Integration** — Connect to live
   regulatory RSS/API feeds (EUR-Lex, US Federal Register, FCA, CBK) with
   NLP-based impact classification that auto-triggers gap assessments on new
   obligations.

8. **Cryptographic Evidence Chain with Merkle Proofs** — Replace the current
   SHA-256 dict digest with a Merkle tree where each leaf is
   `H(evidence_id || collected_at || payload_hash)`, providing
   tamper-evident provenance independently verifiable by auditors.

9. **Multi-Party Attestation Workflows** — Route reports through N required
   attestors (CFO, CISO, DPO, Board Audit Committee) with configurable quorum
   thresholds, deadline enforcement, and escalation to the compliance steward.

10. **AI-Assisted Finding Triage and Remediation Suggestions** — LLM agent
    using `ComplianceAgentRecord` infrastructure to classify findings, suggest
    remediation plans from a curated library, estimate effort, and auto-assign
    owners.

11. **Compliance Posture Score and Trend Analytics** — Time-series
    `posture_score` model snapshotting coverage, open findings, overdue
    assessments, and escalation counts daily, with regression alerting for
    leadership.

12. **Exception Management Lifecycle** — Full exception lifecycle (request,
    risk-acceptance, approval, time-bound expiry, renewal, auto-reopening of
    finding when exception lapses) to close the gap present in current UI
    routes.

13. **Immutable Audit Log Export and Regulator Submission Package** —
    `export_audit_package` serialising the Merkle-chained audit log, evidence
    references, attestations, and framework metadata into a signed ZIP artefact
    (PGP or JWS) ready for regulator submission.

14. **Fine-Grained RBAC and Separation-of-Duties Enforcement** — Full RBAC
    layer integrated with APG `auth` enforcing SoD rules (finding opener ≠
    resolver, evidence collector ≠ control assessor) at the service level.

15. **Webhook and Event Bus Integration for Real-Time Notifications** —
    Structured CloudEvents emitted to APG `ntfy` on compliance state changes,
    enabling Slack/Teams/PagerDuty alerts, `wflo` workflow triggers, and
    cross-capability reactions.

## New Methods

The following methods were added in v2.0 and cover the most common integration
touch-points.

### `gap_assess` — Identify untested controls

Returns all controls in a framework that lack an effective assessment. Use this
before report preparation to surface coverage holes.

```python
gaps = service.gap_assess("tenant-a", "fw-soc2")
# {
#   "framework_id": "fw-soc2",
#   "tenant_id": "tenant-a",
#   "total_controls": 12,
#   "assessed_controls": 9,
#   "gap_count": 3,
#   "gaps": [...]
# }
```

### `remediation_track` — Append progress notes to a finding

Maintains a timestamped log inside `finding.remediation_plan`. Call repeatedly
as work progresses; the full history is preserved.

```python
service.remediation_track(
	"tenant-a",
	"finding-001",
	"Patched auth service; awaiting QA sign-off.",
	"eng-lead",
)
```

### `gdpr_dpia` — Record a Data Protection Impact Assessment

Creates a DPIA record for a processing activity, linking data types and risk
level. Emits an audit event automatically.

```python
dpia = service.gdpr_dpia(
	"dpia-crm-ingest",
	"tenant-a",
	"CRM data ingest pipeline",
	["email", "phone", "location"],
	"high",
	"dpo@company.com",
)
```

### `regulatory_alert` — Ingest an incoming regulatory change

Records a new regulatory alert with severity and optional effective date.
Downstream logic can query `list_audit_events` filtered on
`regulatory_alert_created` to drive gap assessments.

```python
alert = service.regulatory_alert(
	"alert-cbk-2026-01",
	"tenant-a",
	"CBK Risk Management Guidelines 2026",
	"New outsourcing risk requirements for tier-1 banks.",
	"high",
	effective_date=datetime(2026, 7, 1),
)
```

### `risk_integrate` — Link an external risk score to a control

Stores a normalised risk score (0.0–1.0) against a control, with full audit
trail. Feeds the risk-adjusted prioritisation roadmap item.

```python
service.risk_integrate(
	"tenant-a",
	"risk-ext-007",
	"ctrl-access-review",
	risk_score=0.82,
	risk_owner="ciso@company.com",
)
```

### `audit_schedule` — Schedule an internal or external audit

Records a forthcoming audit against a framework with auditor identity, scope,
and date. Use alongside `list_audit_events` to build an audit calendar view.

```python
schedule = service.audit_schedule(
	"sched-soc2-2026",
	"tenant-a",
	"fw-soc2",
	audit_date=datetime(2026, 9, 15),
	auditor="external-auditor-firm",
	scope="CC6, CC7, CC9 trust service criteria",
)
```

## Focused Verification

```bash
./.venv/bin/python -m py_compile capabilities/common/comp/__init__.py capabilities/common/comp/capability_contract.py capabilities/common/comp/compliance_engine.py capabilities/common/comp/models.py capabilities/common/comp/service.py capabilities/common/comp/api.py capabilities/common/comp/views.py capabilities/common/comp/app.py capabilities/common/comp/test_capability_contract.py capabilities/common/comp/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/comp/test_capability_contract.py capabilities/common/comp/tests/test_package_contract.py
./.venv/bin/python capabilities/common/comp/app.py
./.venv/bin/apg capabilities inspect comp --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/comp --json
./.venv/bin/apg capabilities publish-plan capabilities/common/comp --json
```

Full repository audits are intentionally separate so focused capability work
can move quickly on battery.
