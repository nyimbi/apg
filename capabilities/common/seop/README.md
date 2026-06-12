# SEOP - Security Operations

SEOP is the APG security-operations capability. It gives generated applications a composable runtime for detections, incident response, response playbooks, posture controls, audit evidence, governed AI agents, UI view models, visual theming, and Bytewax lifecycle events.

Use SEOP when an application needs to accept security signals, triage anomalies, open incidents, execute approved response playbooks, track posture coverage, close incidents with evidence, or compose AI agents into security review lanes.

## What It Provides

- Detection intake with alert source, anomaly confidence, severity, signal references, owner, and required actions.
- Incident response with owner, severity, linked detections, evidence, escalation, response state, and closure.
- Approved response playbooks and governed response execution.
- Security posture controls with `gap`, `partial`, and `covered` coverage bands.
- Audit trail for lifecycle events.
- First-class SEOP agents for Codex, Claude Code, OpenCode, and Pi based review lanes.
- APG Python UI view models for dashboard, queues, workbench, audit, and settings.
- Visual theme tokens for security operations screens.
- Bytewax stream metadata for lifecycle events.

## Core Runtime

```python
from capabilities.common.seop import SeopService

service = SeopService()

detection = service.create_detection(
	tenant_id="tenant-a",
	title="Privileged anomaly",
	alert_source="siem",
	anomaly_confidence=0.95,
	severity="high",
	signal_refs=["alert-1"],
)

incident = service.open_incident(
	tenant_id="tenant-a",
	title="Privileged compromise",
	owner="secops-lead",
	severity="critical",
	detection_ids=[detection["id"]],
	escalation_recorded=True,
	evidence_refs=["case://evidence/1"],
)

playbook = service.approve_playbook(
	tenant_id="tenant-a",
	name="Isolate privileged account",
	owner="secops-lead",
	steps=["disable token", "isolate endpoint", "notify owner"],
	approved_by="ciso",
)

service.execute_response(
	tenant_id="tenant-a",
	incident_id=incident["id"],
	playbook_id=playbook["id"],
	action="isolate endpoint",
	actor="analyst-1",
	containment_reviewed=True,
)

service.close_incident(
	tenant_id="tenant-a",
	incident_id=incident["id"],
	closure_evidence="case://closure/1",
	actor="analyst-1",
	post_incident_review="case://review/1",
	compliance_mapping="control://iso-27001/A.5",
)
```

## AI Agent Composition

SEOP treats AI agents as governed composition elements rather than comments or external notes.

```python
agent = service.register_seop_agent(
	tenant_id="tenant-a",
	name="Detection reviewer",
	runtime="codex",
	role="detection_reviewer",
	scope="review high-confidence detections and prepare analyst summaries",
	owner="secops-lead",
)

decision = service.validate_agent_response_action(
	tenant_id="tenant-a",
	agent_id=agent["id"],
	incident_severity="critical",
	human_approval_recorded=False,
)

assert decision["decision"] == "deny"
```

Supported runtimes:

- `codex`
- `claude_code`
- `opencode`
- `pi`

Supported roles:

- `detection_reviewer`
- `incident_commander`
- `response_reviewer`
- `playbook_author`
- `posture_reviewer`
- `compliance_reviewer`

Critical agent-driven response actions require human approval.

## Rule Engine

The deterministic rule engine protects the lifecycle:

- tenant context is mandatory;
- detections require trusted alert sources and Bytewax event routing;
- incidents require owner, evidence, and critical escalation;
- responses require approved playbooks, actor identity, and containment review;
- high-confidence anomalies require review;
- closure requires evidence, post-incident review, and compliance mapping;
- agents require supported runtime and role;
- critical agent-driven response requires human approval.

Rules are exposed through `evaluate_capability_rules()` and `SeopService.evaluate()`.

## UI Surfaces

`views.py` exposes route-backed models for:

- dashboard: `/seop/dashboard`
- detection console: `/seop/detections`
- incident queue: `/seop/incidents`
- triage queue: `/seop/triage`
- playbook manager: `/seop/playbooks`
- response actions: `/seop/responses`
- posture: `/seop/posture`
- agent workbench: `/seop/agents`
- audit trail: `/seop/audit`
- settings: `/seop/settings`

These view models are intentionally framework-neutral so APG generated Python applications can compose them into their chosen UI shell.

## Event Stream

SEOP publishes lifecycle metadata for Bytewax:

- processor: `bytewax`
- stream: `apg.seop.lifecycle`
- key: `tenant_id`

Events:

- `detection_created`
- `incident_opened`
- `playbook_approved`
- `response_executed`
- `incident_closed`
- `seop_agent_registered`

## Adapter Boundaries

The package does not directly call live SIEM, SOAR, EDR, MDM, ZTNA, DLP, ticketing, compliance, threat-intelligence, or telemetry systems. Add those integrations as adapters around the stable service methods and stream metadata.

## Verification

Battery-conscious package verification:

```bash
./.venv/bin/python -m py_compile capabilities/common/seop/__init__.py capabilities/common/seop/capability_contract.py capabilities/common/seop/models.py capabilities/common/seop/ops_runtime.py capabilities/common/seop/service.py capabilities/common/seop/api.py capabilities/common/seop/views.py capabilities/common/seop/app.py capabilities/common/seop/test_capability_contract.py capabilities/common/seop/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/seop/test_capability_contract.py capabilities/common/seop/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/seop --json
./.venv/bin/apg capabilities publish-plan capabilities/common/seop --json
```

Run broader repository checks only when battery and time allow.

---

## World-Class Enhancements (v2.0)

- **I1.** SEOP — World-Class Improvement Proposals
- **I2.** Full async surface
- **I3.** Persistent backing store via async SQLAlchemy
- **I4.** Structured event publishing via CloudEvents
- **I5.** MITRE ATT&CK enrichment pipeline
- **I6.** Correlation engine for multi-signal detections
- **I7.** Risk-scored incident prioritisation
- **I8.** Automated playbook selection
- **I9.** SLA tracking and breach alerting
- **I10.** Threat-intel feed deduplication and expiry
- **I11.** Compliance control mapping service
- **I12.** Analyst workload balancing
- **I13.** Detection quality metrics and false-positive feedback loop
- **I14.** Automated evidence collection harness
- **I15.** Zero-trust posture continuous monitoring

See `WORLD_CLASS_IMPROVEMENTS.md` for full justification and implementation details.
