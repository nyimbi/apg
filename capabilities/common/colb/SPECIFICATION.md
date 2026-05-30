# Collaboration Tools Capability Specification

## Purpose

`colb` is the APG common capability for governed collaboration. It lets generated applications compose tenant-scoped workspaces, realtime sessions, shared artifacts, annotations, decision records, presence, protocol health, AI collaborators, audit, analytics, and visual theming.

## Scope

The capability must support:

- Tenant-local workspaces with owners, participants, external participants, retention policy, and membership review.
- Realtime sessions with secure transport, protocol health, event bus evidence, recording-retention controls, and participant membership checks.
- Shared artifacts with artifact policy, version history, DLP checks, duplicate-ID protection, annotations, and decision records.
- Presence and cursor state for active sessions.
- AI collaborators with registration, workspace scope, and contribution disclosure.
- UI route contracts and dependency-light view models.
- Bytewax-backed event-stream configuration for batch collaboration mutations.

## Dependencies

Required:

- `chat` for communication composition.
- `ntfy` for notification composition.
- `auth` for actor identity and permission composition.

Optional:

- `mqeb`, `mten`, `audl`, `wflo`, `vidc`, `nlpc`, `secu`, and `cach`.

## Configuration

The authoritative configuration lives in `capability_contract.py` and includes:

- `workspaces`
- `sessions`
- `artifacts`
- `annotations`
- `presence`
- `protocols`
- `ai_agents`
- `security`
- `governance`
- `retention`
- `observability`
- `adapters`
- `ui`
- `theme`

## Rules

The deterministic rule engine covers:

- tenant context
- workspace owner, name, participant, retention, external policy, external expiry, and large membership review
- session workspace, owner, active workspace, secure transport, protocol health, event-bus evidence, recording retention, and participant membership
- presence session and participant checks
- artifact policy, version history, DLP, and duplicate IDs
- annotation artifact, author, and body checks
- decision annotation, owner, and evidence checks
- workspace export approval
- AI collaborator registration, scope, and disclosure checks
- audit evidence
- cross-tenant access denial
- Bytewax batch mutation enforcement

## Runtime

`collaboration_runtime.CollaborationRuntime` is the generated-application runtime. It stores deterministic in-memory state for:

- workspaces
- sessions
- artifacts
- annotations
- decisions
- presence
- audit events

The runtime uses tenant-qualified keys so public IDs can repeat across tenants.

## Production Boundary

Existing production-oriented files remain in the package, including `production_app.py`, `service.py`, `api.py`, `views.py`, WebRTC, WebSocket, and protocol managers. The generated runtime does not start servers or require these integrations.

## UI

The UI contract exposes:

- dashboard
- workspaces
- sessions
- presence
- artifacts
- annotations
- decisions
- agents
- protocols
- analytics
- audit
- settings

## Out Of Scope

This packet does not run live WebSocket servers, WebRTC sessions, MQTT/gRPC transports, databases, browser UIs, DLP services, external AI-agent CLIs, or live Bytewax workers. Those belong behind adapters or later production integration slices.
