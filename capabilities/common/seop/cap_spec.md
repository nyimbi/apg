# Security Operations Capability Specification

- Capability Name: Security Operations
- Capability ID: `seop`
- Category: common
- Version: 1.0.0

## Purpose

SEOP provides the executable APG security-operations runtime for generated applications. It coordinates detections, anomaly triage, incidents, response playbooks, posture controls, AI-assisted review lanes, lifecycle events, and audit evidence through deterministic package behavior.

## Provided Services

- `detection_pipeline`
- `incident_response`
- `threat_triage`
- `response_playbooks`
- `security_posture`
- `seop_agents`

## Required Services

- `secu`
- `anom`
- `moni`
- `logt`
- `audl`

## Current Runtime

The package exposes `SeopService`, API helpers, UI view models, a deterministic rule engine, a visual theme contract, and package publication evidence.

The service can:

- create detections from trusted alert sources;
- require Bytewax lifecycle routing for detection events;
- open incidents with owners, evidence, linked detections, severity, and escalation;
- approve response playbooks;
- execute response actions with actor and containment review;
- record posture controls;
- close incidents with closure evidence, post-incident review, and compliance mapping;
- register governed SEOP agents;
- validate critical agent-driven response actions;
- expose audit events and dashboard summaries.

## Rules

- `tenant_context_required`
- `detection_requires_alert_source`
- `detection_requires_bytewax_stream`
- `incident_requires_owner`
- `incident_requires_evidence`
- `critical_incident_requires_escalation`
- `response_requires_playbook_approval`
- `response_requires_actor`
- `response_requires_containment_review`
- `high_confidence_anomaly_requires_review`
- `closure_requires_post_incident_review`
- `closure_requires_compliance_mapping`
- `seop_agent_runtime_supported`
- `seop_agent_role_supported`
- `critical_agent_action_requires_human_approval`

## UI

SEOP exposes route-backed APG Python view models for dashboard, detection console, incident queue, triage, playbooks, response actions, posture, agent workbench, audit trail, and settings.

## Theme

SEOP uses the `seop_security_ops` theme with compact density, severity pills, priority queues, approval chips, coverage chips, agent review lanes, and audit timelines.

## Event Stream

SEOP lifecycle events are described by the Bytewax stream manifest:

- processor: `bytewax`
- stream: `apg.seop.lifecycle`
- key: `tenant_id`

## Detailed Packet

See `SPECIFICATION.md`, `PLAN.md`, and `README.md` for the complete lifecycle packet, implementation plan, usage examples, and focused verification commands.
