# Alert Management Capability Specification

`intel_alerts` is the APG Alert Management capability. It turns governed rules
and signals into alerts, escalations, notifications, assignments, resolutions,
reviews, UI models, Bytewax lifecycle events, and provider-neutral AI-agent
composition surfaces.

## Capability Summary

- Capability ID: `intel_alerts`
- Display name: Alert Management
- Target: Python executable capability package
- Event processor: Bytewax
- Event stream: `apg.intel.alerts.lifecycle`
- Theme: `intel_alerts_control`
- Agent runtimes: `codex`, `claude_code`, `opencode`, `pi`

## Composition Interfaces

The package provides authority, workspace, rule, signal, alert, escalation,
notification, assignment, resolution, review, and AI-agent workflows. It
requires APG auth, audit, notification, NLP, graph, RAG, and geospatial
capabilities so generated applications can compose alerts with identity,
evidence, routing, enrichment, graph context, map context, and dissemination.

## Runtime Shape

The service keeps tenant-scoped in-memory records for the executable baseline
while leaving live notification delivery, durable stream workers, case-system
writes, ticketing, paging, graph/RAG writes, and external incident systems
behind adapter boundaries.

## Governance

Every write path evaluates deterministic rules before mutation. The rules
require tenant context, policy attachment, lawful authority, evidence,
classification, rule ownership, signal confidence, escalation approval,
notification approval, resolution approval, Bytewax routing, and human approval
for privileged AI-agent scopes. Unapproved escalation, unapproved
notification, alert suppression, evidence fabrication, privacy bypass,
autonomous closure, and severity downgrade are denied.

