# APG Security Framework Capability

The Security Framework capability (`secu`) is the executable security control
plane for generated APG applications. It provides tenant-scoped security
policy, device posture, threat indicator, access assessment, compliance
control, policy exception, incident response, governed AI security-agent,
Bytewax lifecycle, quarantine, and audit timeline surfaces without requiring
live SIEM, EDR, SOAR, IAM, GRC, DLP, or AI-provider integrations.

SECU is intentionally dependency-light at package runtime. Generated
applications can compose and test the capability locally, while production
systems can later bind live security providers through adapters that preserve
the same rule and lifecycle contracts.

## What SECU Provides

- Tenant-scoped security policies with owners, security levels, required
  controls, target surfaces, enabled state, and tags.
- Device posture records with trust state, managed state, risk score,
  indicators, and automatic quarantine for compromised or high-risk devices.
- Threat indicators with type, value, severity, source, TTL, and active state.
- Access assessment through the deterministic SECU rule engine, producing
  `allow`, `challenge`, `quarantine`, or `deny` decisions with matched rules
  and required actions.
- Compliance control posture with implemented, evidence-required,
  non-compliant, and waived statuses.
- Policy exception workflow with request, independent review, approval,
  rejection, notes, and expiry guardrails.
- Incident response workflow with open, contain, and resolve lifecycle states,
  critical incident containment requirements, and resolution evidence checks.
- Governed security-agent registration for `codex`, `claude_code`,
  `opencode`, and `pi` runtimes, including owner, purpose, scope, role,
  disclosure, and privileged-role human approval evidence.
- Durable review evidence for policy exceptions, compliance evidence gaps,
  privileged security-agent review, denied security lifecycle batch routing,
  and audit events.
- Bytewax lifecycle stream metadata and batch-routing guardrail for security
  lifecycle mutations.
- Audit events for governed security state transitions.
- API helpers and UI view models for generated applications.
- Capability contract, visual theme, semantic model, and release evidence for
  APG composition tooling.

## Runtime Surfaces

The primary dependency-light service is `SecuService` in `service.py`.

```python
from capabilities.common.secu.service import SecuService

service = SecuService()
policy = service.create_policy(
	tenant_id="tenant-a",
	name="Privileged access",
	owner="secops",
	security_level="restricted",
	required_controls=["mfa", "device_trust"],
	applies_to=["admin_console"],
)

exception = service.request_policy_exception(
	tenant_id="tenant-a",
	exception_id="break-glass-1",
	policy_id=policy["id"],
	requested_by="app-owner",
	reason="Emergency production repair",
	expires_at="2099-01-01T00:00:00Z",
)

approved = service.decide_policy_exception(
	tenant_id="tenant-a",
	exception_id=exception["id"],
	reviewer="security-reviewer",
	decision="approved",
	notes="Time-bound exception with compensating monitoring.",
)
```

Critical incidents fail closed unless a containment plan is supplied, and
resolution fails closed until containment evidence exists.

```python
incident = service.open_incident(
	tenant_id="tenant-a",
	incident_id="inc-1",
	title="Privileged credential exposure",
	severity="critical",
	opened_by="soc-analyst",
	containment_plan="Disable token, rotate secret, and isolate affected host.",
)
contained = service.contain_incident(
	tenant_id="tenant-a",
	incident_id=incident["id"],
	actor="incident-commander",
	containment_action="Token disabled and host isolated.",
	containment_evidence="audit://incident/inc-1/containment",
)
resolved = service.resolve_incident(
	tenant_id="tenant-a",
	incident_id=contained["id"],
	resolved_by="incident-commander",
	resolution="Credentials rotated and monitoring confirmed clean.",
	notes="Post-incident review attached.",
)
```

Security agents are explicit participants in review and response workflows.
Privileged roles such as `incident_responder`, `compliance_reviewer`, and
`exception_reviewer` are retained as `pending_review` evidence when human
approval is not required.

```python
agent = service.register_security_agent(
	tenant_id="tenant-a",
	agent_id="incident-agent",
	name="Incident Response Agent",
	runtime="claude-code",
	role="incident-responder",
	scope="Summarize containment evidence for human responders.",
	owner="secops",
	purpose="Incident evidence review.",
	human_approval_required=True,
	policy_ref="secu-agent-policy",
)

assert agent["runtime"] == "claude_code"
assert agent["role"] == "incident_responder"
```

## Durable Review Evidence

SECU preserves review and remediation state for generated security consoles.
Policy exceptions, compliance controls that require evidence, privileged
security-agent registrations, security lifecycle batch validations, and audit
events carry the same policy evidence fields:

- `policy_decision`;
- `matched_rules`;
- `review_reasons`;
- `review_evidence`.

Generated applications can compose the active review queue:

```python
pending = service.list_pending_reviews("tenant-a")
```

Denied non-Bytewax lifecycle batches are also stored through
`list_security_lifecycle_batches()` before `PermissionError` is raised, so
operators can see the routing violation that must be remediated.

Batch security lifecycle mutation intent must route through Bytewax:

```python
service.validate_security_lifecycle_batch(
	tenant_id="tenant-a",
	event_stream="bytewax",
	mutation_count=3,
)
```

## API Helpers

`api.py` exposes package-level helpers backed by a shared `SERVICE` instance:

- `capability_status`
- `create_policy`
- `record_device_posture`
- `register_threat_indicator`
- `assess_access`
- `record_compliance_control`
- `request_policy_exception`
- `decide_policy_exception`
- `open_incident`
- `contain_incident`
- `resolve_incident`
- `register_security_agent`
- `validate_security_lifecycle_batch`
- `list_security_lifecycle_batches`
- `list_pending_reviews`
- `list_security_posture`

These helpers are designed for generated APG applications and package smoke
tests. Long-running services should inject or wrap `SecuService` explicitly.

## UI View Models

`views.py` provides dependency-light data models for APG UI composition:

- dashboard
- risk console
- threat console
- policy workbench
- policy exception queue
- incident response console
- device quarantine console
- compliance console
- security-agent roster
- security audit timeline
- rule workbench
- settings

The UI contract also declares theme components for risk meters, threat
indicators, policy cards, compliance badges, exception queues, incident panels,
quarantine lists, security-agent rosters, Bytewax stream indicators, and audit
timelines.

## Guardrails

The deterministic rule engine currently enforces:

- known malicious network origins are denied;
- compromised devices are quarantined;
- critical risk scores are denied;
- high-risk access requires step-up challenge;
- compliance violations require audit evidence;
- policy exceptions require an independent reviewer;
- expired policy exceptions cannot be approved;
- critical incidents require a containment plan;
- incident resolution requires containment evidence.
- security agents must use supported runtimes, supported roles, and explicit
  scope;
- privileged security-agent roles require human approval evidence or review;
- security lifecycle batch operations must use Bytewax and denied validations
  preserve review evidence.

These rules are exposed through `capability_contract.py`, `app.py`,
`semantic_model.json`, and `release_report.json` so APG composition tooling can
reason about SECU without importing live provider SDKs.

## Adapter Boundaries

The local runtime does not bind directly to SIEM, SOAR, threat-intelligence,
EDR, MDM, IAM, MFA, GRC, DLP, notification, ticketing, AI threat detection, or
production persistence providers. Those integrations should be introduced as
adapters that call the existing service methods and preserve the fail-closed
guardrails.

## World-Class Enhancements (v2.0)

1. **MITRE ATT&CK Correlation** — maps threat indicators to kill-chain tactics/techniques (T-codes) for responder context.
2. **Bayesian Risk Fusion** — replaces static weighted-average with a self-calibrating Bayesian network per risk dimension.
3. **Impossible-Travel Engine** — haversine distance + timestamp delta flags credential stuffing when implied speed exceeds ~900 km/h.
4. **Streaming STIX/TAXII + OpenCTI Ingestion** — async IoC ingestion with TTL expiry and deduplication into `ThreatIndicatorRecord`.
5. **Zero-Trust Continuous Verification** — sliding-window `SecurityContext` tokens re-evaluated on every significant event, not just at login.
6. **Federated Compliance Evidence Graph** — SHA-256-addressed DAG lets controls share evidence across frameworks (SOC 2 → GDPR) without duplication.
7. **ML False-Positive Feedback Loop** — analyst verdicts retrain a per-tenant online model to reduce alert fatigue without a full MLOps pipeline.
8. **HSM/TPM Attestation** — TPM 2.0 quote verification upgrades device `trust_level` to `TRUSTED`; failed attestation immediately sets `COMPROMISED`.
9. **SIEM Push Adapter (CEF/LEEF/JSON-LD)** — non-blocking `asyncio.Queue` translates audit events to three SIEM formats without blocking assessment.
10. **Automated Playbook Engine** — `SecurityPlaybook` triggers evaluate rule DSL conditions and execute ordered steps with per-step timeout and rollback.
11. **DLP Classification Tagging** — regex + Ollama embedding model attaches `DataSensitivityTag` to resource access; high-sensitivity triggers step-up.
12. **Segregation-of-Duties Conflict Detection** — `check_sod_conflicts` cross-references grants against a tenant SoD matrix in O(n) via `frozenset` pairs.
13. **Cryptographic Audit Chain** — per-tenant Merkle tree chains `SecurityAuditEventRecord` entries; `verify_audit_chain` detects any retroactive tampering.
14. **Adaptive MFA Orchestration** — risk-graduated selector: 0-49 no-op, 50-69 TOTP, 70-84 FIDO2, 85-100 voice + session freeze.
15. **SECU-QL Threat-Hunting Language** — declarative mini-language (`HUNT events WHERE ... WITHIN 24h GROUPBY user_id`) with zero external parser dependencies.

## New Methods

The three highest-impact async methods added in v2.0 are
`assess_security_context`, `predict_threats`, and `assess_compliance`.

### `assess_security_context` — full risk + policy evaluation in one call

```python
import asyncio
from capabilities.common.secu.service import APGSecurityFrameworkService
from capabilities.common.secu.models import SecurityContext, UserContext, NetworkContext

async def main():
    svc = APGSecurityFrameworkService()
    await svc.initialize()

    ctx = SecurityContext(
        tenant_id="tenant-a",
        user=UserContext(user_id="u1", email="ops@example.com", roles=["admin"]),
        network=NetworkContext(ip_address="203.0.113.5", country_code="RU"),
    )
    enriched = await svc.assess_security_context(ctx)
    print(enriched.risk_score.level, enriched.recommended_actions)

asyncio.run(main())
```

### `predict_threats` — anomaly + signature + behavioural threat detection

```python
async def scan_for_threats(ctx: SecurityContext):
    svc = APGSecurityFrameworkService()
    await svc.initialize()

    threats = await svc.threat_detector.predict_threats(ctx)
    for t in threats:
        print(t.threat_type, t.confidence, t.severity)
        # e.g. ThreatType.CREDENTIAL_STUFFING  0.91  Severity.HIGH
```

### `assess_compliance` — full framework gap analysis with recommendations

```python
from capabilities.common.secu.models import ComplianceFramework

async def compliance_check(tenant_id: str):
    svc = APGSecurityFrameworkService()
    await svc.initialize()

    status = await svc.assess_tenant_compliance(tenant_id, ComplianceFramework.SOC2)
    report = await svc.compliance_engine.generate_compliance_report(
        ComplianceFramework.SOC2, tenant_id
    )
    print(status.overall_status, report["recommendations"])
```

## Focused Verification

Use the focused package proof while iterating on SECU:

```bash
./.venv/bin/python -m py_compile capabilities/common/secu/__init__.py capabilities/common/secu/models.py capabilities/common/secu/security_runtime.py capabilities/common/secu/service.py capabilities/common/secu/api.py capabilities/common/secu/views.py capabilities/common/secu/capability_contract.py capabilities/common/secu/app.py capabilities/common/secu/tests/test_capability_contract.py capabilities/common/secu/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/secu/tests/test_capability_contract.py capabilities/common/secu/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/secu --json
./.venv/bin/apg capabilities publish-plan capabilities/common/secu --json
```
