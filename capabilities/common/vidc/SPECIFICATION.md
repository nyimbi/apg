# Video Conferencing Capability Specification

## Purpose

`vidc` is the APG common capability for governed video meetings. It lets generated applications compose tenant-scoped rooms, realtime meeting sessions, participants, external guest controls, recordings, captions, AI meeting agents, audit trails, analytics, UI screens, visual theming, and event-stream policy.

## Scope

The capability must support:

- Tenant-local meeting rooms with owners, moderation policies, guest policies, waiting-room controls, and status.
- Meeting lifecycle with accountable host, secure transport, screen-share policy, participant capacity review, recording controls, and closure.
- Participant records for hosts, cohosts, participants, guests, and observers.
- Recording artifacts with consent, encryption, retention policy, and access-audit controls.
- Caption artifacts with transcript references and configured language support.
- AI meeting agents with registration, runtime, role, explicit meeting scope, and visible contribution disclosure.
- Computer-vision assistance as an adapter-governed capability that requires policy evidence before use.
- Bytewax-backed event-stream configuration for batch video-meeting mutations.
- UI route contracts and dependency-light view models for generated applications.

## Dependencies

Required:

- `colb` for collaboration composition.
- `mqeb` for event/message composition.
- `cvsn` for computer-vision assist composition.

Optional:

- `ntfy`, `nlpc`, `audl`, `auth`, and `them`.

## Configuration

The authoritative configuration lives in `capability_contract.py` and includes:

- `meetings`
- `media`
- `recordings`
- `meeting_agents`
- `governance`
- `observability`
- `adapters`
- `ui`
- `theme`

## Rules

The deterministic rule engine covers:

- tenant context
- room name, owner, and moderation policy
- room selection, host accountability, secure transport, and screen-share policy
- external guest policy and waiting-room review
- recording consent, encryption, retention, and access audit
- large-meeting capacity review
- participant meeting and user references
- caption transcript and supported-language checks
- computer-vision assist policy
- AI meeting agent registration, scope, and disclosure
- meeting state-change audit evidence
- cross-tenant access denial
- Bytewax batch mutation enforcement

## Runtime

`service.VidcService` is the generated-application runtime. It stores deterministic in-memory state for:

- rooms
- meetings
- participants
- recordings
- captions
- meeting agents
- audit events

The runtime enforces the same guardrails exposed by the contract rule engine and keeps live media infrastructure behind adapter boundaries.

## UI

The UI contract exposes:

- dashboard
- meetings
- rooms
- participants
- recordings
- captions
- agents
- analytics
- audit
- settings

## Production Boundary

This packet does not start WebRTC/SFU servers, manage TURN/STUN credentials, store video blobs, perform live transcription, run live computer-vision models, operate external AI-agent CLIs, or start Bytewax workers. Those are production adapters behind the APG composition layer.

## Acceptance Gates

- `README.md`, `SPECIFICATION.md`, and `PLAN.md` describe the package clearly.
- `capability_contract.py` exposes configuration, deterministic rules, UI, theme, and adapter metadata.
- Runtime/API/view tests prove positive lifecycle behavior and negative guardrail behavior.
- `semantic_model.json`, `package_manifest.json`, and `release_report.json` match the current contract.
- Focused compile, pytest, implementation audit, publish-plan, stale-marker scan, and diff check pass.
