# Alert Management Specification

## Purpose

Alert Management lets APG applications convert governed signals into
actionable, auditable alerts. It is designed for watch centers, threat teams,
incident teams, fraud teams, public-safety teams, executives, and partners who
need reliable alert triage, escalation, notification, assignment, resolution,
and review.

## Users

- Rule stewards who configure alert triggers.
- Analysts who triage signals and alerts.
- Supervisors who approve escalations and notifications.
- Operations teams that receive assignments and close alerts.
- AI-agent supervisors who delegate bounded rule, signal, alert, escalation,
  notification, and resolution review work.

## Functional Scope

- Authorities: lawful alerting mandates with scope, classification, approver,
  expiry, and evidence.
- Workspaces: watch-center, threat, incident, fraud, public-safety, executive,
  and partner alerting containers.
- Rules: threshold, anomaly, watchlist, correlation, prediction, geofence,
  case-trigger, and manual rules.
- Signals: metrics, indicators, events, forecasts, threats, case updates,
  geospatial signals, and partner notices.
- Alerts: early warnings, critical alerts, watchlist hits, incident alerts,
  fraud alerts, threat alerts, and system alerts.
- Escalations: approved escalation to supervisors, incident teams, case teams,
  executives, partners, field teams, or watch centers.
- Notifications: approved in-app, email, SMS, secure-message, webhook,
  case-note, or briefing-queue notifications.
- Assignments: analyst, supervisor, incident commander, case owner, field team,
  and partner-owner assignments.
- Resolutions: confirmed, false positive, duplicate, mitigated, escalated,
  closed, or monitoring outcomes.
- Reviews: human review outcomes for lifecycle artifacts.
- AI agents: provider-neutral runtimes with bounded roles and explicit scope.

## Out Of Scope

This package does not send live notifications, page external systems, write
case-management records, persist queues, run durable stream topologies, execute
live correlation engines, or mutate graph/RAG stores. Those remain adapter
responsibilities until their contracts are explicit.

## Lifecycle

1. Record authority.
2. Create alert workspace.
3. Record alert rule.
4. Record signal.
5. Record alert.
6. Record approved escalation.
7. Record approved notification.
8. Record assignment.
9. Record approved resolution.
10. Record human review.
11. Register bounded AI agents.
12. Route lifecycle batches through Bytewax.

## Rule Engine

The deterministic rule engine denies missing tenant context, unsupported
taxonomy values, missing evidence, missing authority, invalid signal
confidence, unsupported severities, missing owners, missing approvals,
non-Bytewax batches, unsupported agent runtimes or roles, missing agent scope,
privileged agent actions without approval, unapproved escalation, unapproved
notification, alert suppression, evidence fabrication, privacy bypass,
autonomous closure, and severity downgrade.

## UI And Theme

The capability exposes APG Python UI route metadata for dashboard,
authorities, workspaces, rules, signals, alerts, escalations, notifications,
assignments, resolutions, reviews, agents, and settings. The theme uses
compact, work-focused tokens under `intel_alerts_control`.

## Adapter Boundaries

Generated applications compose this capability with auth, audit, notification,
NLP, graph, RAG, and geospatial capabilities. Production integrations should
bind notification delivery, ticketing, case-management writes, incident
systems, graph/RAG writes, durable queues, and Bytewax workers through adapters
without bypassing this package's deterministic rules.

