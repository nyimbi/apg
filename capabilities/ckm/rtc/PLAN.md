# CKM Real-Time Collaboration Packet Plan

## Scope

Build `ckm_rtc` as a coherent lifecycle and guardrail packet for APG
applications that need collaboration sessions, participant policy, presence,
messages, media controls, decision capture, AI-agent review, UI metadata,
theme metadata, Bytewax stream governance, and publishable package evidence.

## Implementation Packets

1. Specification and contract
   - Replace stale narrative in `cap_spec.md` with a pointer to the active
     specification.
   - Add `SPECIFICATION.md` for the normative behavior.
   - Expand `capability_contract.py` with configuration, rules, UI routes,
     theme metadata, provides/requires, and Bytewax streaming.

2. Dependency-light lifecycle
   - Add sessions, participants, messages, decisions, and RTC-agent data
     contracts.
   - Implement `RtcLifecycleService` for session creation, joining, presence,
     messages, screen sharing, recording, decision capture, agent registration,
     batch mutation validation, audit events, and dashboard summary.
   - Keep live signaling, media transport, databases, and stream workers behind
     adapters.

3. Package entrypoint and legacy runtime preservation
   - Preserve the legacy FastAPI/WebSocket runtime as `runtime_app.py`.
   - Make `app.py` the dependency-light package entrypoint used by publish
     checks, semantic-model checks, and generated application composition.

4. Documentation and generated evidence
   - Add root package `README.md` with practical usage and composition notes.
   - Refresh semantic model, package manifest, and release evidence from the
     live contract.
   - Update the progress log with proof commands and review notes.

5. Focused proof and review
   - Add a root focused contract/lifecycle test that avoids legacy media and
     database integration fixtures.
   - Run compile checks, focused tests, semantic probes, implementation audit,
     publish plan, stale-marker scan, and diff checks.
   - Review tenant isolation, participant policy, consent, AI-agent boundaries,
     Bytewax guardrails, import behavior, and generated evidence consistency.

## Out Of Scope

- Live WebSocket/WebRTC/SIP/RTMP/Socket.IO/gRPC deployment.
- Durable database migrations and legacy integration test suite.
- Browser-rendered UI.
- Production media recording and storage.
- Live Bytewax topology deployment.
- Full repository test suite.

## Review Checklist

- Contract is registry-valid and APG Python route metadata uses practical
  targets.
- Dependency-light package import does not start FastAPI, database, WebSocket,
  or media services.
- Participant policy blocks unauthorized joins.
- Sensitive messages require review.
- Screen sharing and recording enforce permission/consent.
- Decision capture requires trace evidence.
- AI-agent guardrails include runtime, role, scope, registration, and
  contribution disclosure.
- Batch mutation is rejected unless the event stream is Bytewax.
- Generated semantic evidence matches the executable contract.
