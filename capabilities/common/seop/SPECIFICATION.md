# SEOP Capability Specification

## 1. Identity

- Capability ID: `seop`
- Name: Security Operations
- Category: common security capability
- Runtime target: APG Python capability package
- Primary users: security analysts, incident commanders, response engineers, compliance reviewers, generated ERP application owners

SEOP gives generated applications a governed security-operations layer. It turns alerts, anomaly scores, incidents, response playbooks, posture controls, audit trails, and AI-assisted review lanes into a composable APG capability.

## 2. Scope

SEOP owns the executable lifecycle for:

- alert detection intake and review;
- incident opening, ownership, severity, evidence, escalation, response, and closure;
- approved response playbooks and response execution;
- posture-control coverage tracking;
- security-operations audit evidence;
- governed AI agent composition for detection review, incident command, response review, playbook authoring, posture review, and compliance review;
- Bytewax-backed lifecycle event publication.

External SIEM, SOAR, EDR, ZTNA, DLP, ticketing, threat-intelligence, and compliance systems stay behind adapters. The package defines deterministic contracts and state transitions that remain stable when those adapters are added.

## 3. Provided Services

- `detection_pipeline`
- `incident_response`
- `threat_triage`
- `response_playbooks`
- `security_posture`
- `seop_agents`

## 4. Required Services

- `secu` for security policy and access-control integration
- `anom` for anomaly context
- `moni` for telemetry and alert feeds
- `logt` for log evidence integration
- `audl` for durable audit publication

Optional integrations include ZTNA, DLP, compliance, SIEM, SOAR, EDR, ticketing, and threat-intelligence adapters.

## 5. Domain Model

### Detection

A detection captures the first actionable security signal. It contains tenant, title, alert source, severity, anomaly confidence, signal references, owner, status, matched rules, required actions, and creation timestamp.

Statuses:

- `new`
- `review_required`
- `triaged`
- `linked`

### Incident

An incident groups one or more detections, evidence references, owner, severity, escalation status, response state, and closure details.

Statuses:

- `open`
- `escalated`
- `responding`
- `contained`
- `closed`

### Playbook

A playbook defines approved response steps with owner and approver identity. SEOP does not execute response actions against unapproved playbooks.

### Response Action

A response action links an incident, an approved playbook, an action, an actor, and required follow-up actions.

### Posture Control

A posture control records coverage for a security operations domain and classifies it as `gap`, `partial`, or `covered`.

### SEOP Agent

A SEOP agent is a first-class composition element with tenant, name, runtime, role, scope, owner, status, and human approval policy.

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

Agents can recommend and prepare work, but critical response actions require human approval.

### Audit Event

An audit event records tenant, event type, subject, message, actor, severity, metadata, and timestamp.

## 6. Rule Engine

The deterministic rule engine enforces:

- tenant context on every executable operation;
- trusted alert source on detections;
- Bytewax lifecycle stream use for detection events;
- accountable owner on incidents;
- evidence before incident queue entry;
- escalation on critical incidents;
- approved playbooks before response execution;
- actor identity and containment review for response execution;
- triage review for high-confidence anomalies;
- post-incident review and compliance mapping before closure;
- supported agent runtime and role;
- human approval for critical agent-driven response actions.

Rules return `allow`, `require_review`, or `deny` with required actions. Generated applications should surface these decisions directly rather than hiding policy outcomes.

## 7. Workflows

### Detection Workflow

1. Create detection from tenant, title, alert source, confidence, severity, and optional signal references.
2. Evaluate source, stream, confidence, and review rules.
3. Store detection as `new` or `review_required`.
4. Emit `detection_created` to the Bytewax lifecycle stream.

### Incident Workflow

1. Open incident with owner, severity, linked detections or evidence references.
2. Require escalation for critical severity.
3. Link detections and move them to `linked`.
4. Emit `incident_opened`.

### Response Workflow

1. Approve playbook with owner, ordered steps, and approver.
2. Execute response only against tenant-owned incident and approved playbook.
3. Require actor and containment review.
4. Move incident to `responding`.
5. Emit `response_executed`.

### Closure Workflow

1. Attach closure evidence.
2. Attach post-incident review.
3. Attach compliance mapping.
4. Move incident to `closed`.
5. Emit `incident_closed`.

### Agent Workflow

1. Register agent with tenant, name, runtime, role, scope, and owner.
2. Enforce supported runtime and role.
3. Require human approval for critical agent-driven response actions.
4. Emit `seop_agent_registered`.

## 8. UI Contract

SEOP exposes APG Python view models for:

- `/seop/dashboard`
- `/seop/detections`
- `/seop/incidents`
- `/seop/triage`
- `/seop/playbooks`
- `/seop/responses`
- `/seop/posture`
- `/seop/agents`
- `/seop/audit`
- `/seop/settings`

The UI should be dense, operator-focused, and optimized for scanning queues, severity, ownership, required actions, response state, and evidence gaps.

## 9. Theming

The default theme is `seop_security_ops`. It defines compact density, severity indicators, priority lists, approval chips, coverage chips, review lanes, and event-stream timelines. Tenant themes may override tokens while preserving route and component semantics.

## 10. Event Stream

SEOP lifecycle events use Bytewax:

- stream: `apg.seop.lifecycle`
- key: `tenant_id`
- events: `detection_created`, `incident_opened`, `playbook_approved`, `response_executed`, `incident_closed`, `seop_agent_registered`

Any live stream adapter must preserve event names, tenant keys, and audit metadata.

## 11. Acceptance Criteria

- Contract exposes configuration, schema, deterministic rules, UI, theme, provided services, required services, and streaming metadata.
- Service executes detection, incident, playbook, response, posture, closure, agent, and audit lifecycles without external dependencies.
- Rules deny invalid tenant, source, stream, owner, evidence, escalation, playbook, actor, containment, closure, agent, and critical approval states.
- API helpers expose the executable lifecycle.
- View models expose dashboard, consoles, queues, workbench, audit, and settings surfaces.
- Generated package artifacts reflect the current contract.
- Focused package tests pass.
