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
`exception_reviewer` fail closed unless human approval is required.

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
- privileged security-agent roles require human approval;
- security lifecycle batch operations must use Bytewax.

These rules are exposed through `capability_contract.py`, `app.py`,
`semantic_model.json`, and `release_report.json` so APG composition tooling can
reason about SECU without importing live provider SDKs.

## Adapter Boundaries

The local runtime does not bind directly to SIEM, SOAR, threat-intelligence,
EDR, MDM, IAM, MFA, GRC, DLP, notification, ticketing, AI threat detection, or
production persistence providers. Those integrations should be introduced as
adapters that call the existing service methods and preserve the fail-closed
guardrails.

## Focused Verification

Use the focused package proof while iterating on SECU:

```bash
./.venv/bin/python -m py_compile capabilities/common/secu/__init__.py capabilities/common/secu/models.py capabilities/common/secu/security_runtime.py capabilities/common/secu/service.py capabilities/common/secu/api.py capabilities/common/secu/views.py capabilities/common/secu/capability_contract.py capabilities/common/secu/app.py capabilities/common/secu/tests/test_capability_contract.py capabilities/common/secu/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/secu/tests/test_capability_contract.py capabilities/common/secu/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/secu --json
./.venv/bin/apg capabilities publish-plan capabilities/common/secu --json
```
