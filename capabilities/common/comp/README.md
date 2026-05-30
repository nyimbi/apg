# Compliance Management

`comp` is APG's package-backed Compliance Management capability. It gives
generated applications a tenant-scoped compliance runtime for frameworks,
obligations, controls, encrypted evidence, assessments, findings, remediation,
reports, attestations, audit events, UI route metadata, and visual theme
metadata.

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
- Finding opening, escalation, remediation, and resolution evidence.
- Report preparation, independent approval, attestation, publication, and
  critical-finding blocking.
- Tenant isolation for repeated business IDs across tenants.
- Hashed audit events for compliance state changes.
- UI view models for dashboard, frameworks, controls, evidence, assessments,
  findings, reports, attestations, audit, and settings.
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
5. Open, escalate, and resolve findings.
6. Prepare a report for a framework and period.
7. Approve the report with an independent approver.
8. Attest the approved report.
9. Publish the report if approval, attestation, and critical-finding guardrails
   pass.

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
- `settings`

`views.py` returns data-only models for generated UIs. The generated UI should
render the provided theme tokens and component names instead of hard-coding
colors or layout assumptions.

## Focused Verification

```bash
./.venv/bin/python -m py_compile capabilities/common/comp/__init__.py capabilities/common/comp/capability_contract.py capabilities/common/comp/compliance_engine.py capabilities/common/comp/models.py capabilities/common/comp/service.py capabilities/common/comp/api.py capabilities/common/comp/views.py capabilities/common/comp/app.py capabilities/common/comp/test_capability_contract.py capabilities/common/comp/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/comp/test_capability_contract.py capabilities/common/comp/tests/test_package_contract.py
./.venv/bin/python -c "from capabilities.common.comp import app; r=app.self_test(); print(r); assert r['passed']"
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/comp --json
./.venv/bin/apg capabilities publish-plan capabilities/common/comp --json
```

Full repository audits are intentionally separate so focused capability work
can move quickly on battery.
