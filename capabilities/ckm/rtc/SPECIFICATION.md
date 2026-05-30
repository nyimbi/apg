# CKM Real-Time Collaboration Specification

## Purpose

The CKM Real-Time Collaboration capability (`ckm_rtc`) lets generated APG
applications compose tenant-scoped collaboration sessions, presence awareness,
real-time messaging, media controls, page-level collaboration, decision capture,
audit evidence, and AI-agent assistance into larger ERP, CRM, CKM, GRC, and
workflow applications.

This package owns the executable contract, deterministic guardrails,
dependency-light lifecycle service, UI route metadata, theme metadata, Bytewax
stream declaration, generated semantic evidence, and focused proof commands.
Live WebSocket, WebRTC, SIP, RTMP, database, scheduler, media gateway, and
stream-worker deployments remain adapter concerns.

## Users And Jobs

- Operators create collaboration rooms tied to a business record, case,
  document, incident, close process, or workflow.
- Participants join sessions, update presence, post messages, share screens,
  start recordings, and capture decisions.
- Compliance reviewers inspect participant policy, consent, sensitive content,
  decisions, and audit evidence.
- Platform engineers bind live signaling, media, persistent storage,
  observability, notification, and Bytewax workers.
- AI agents assist with facilitation, decision review, transcript review, risk
  moderation, and workflow handoff under explicit registration and disclosure.

## Capability Boundary

`ckm_rtc` provides:

- collaboration session lifecycle;
- participant policy and join governance;
- presence heartbeat and context-awareness state;
- real-time message lifecycle and sensitive-content review;
- screen-share and recording consent guardrails;
- decision capture with trace evidence;
- page collaboration and form delegation contract metadata;
- AI RTC-agent registration and policy enforcement;
- Bytewax stream metadata for batch collaboration mutation.

`ckm_rtc` requires:

- `auth` for identity and permission context;
- `conf` for tenant configuration;
- `audl` for durable audit evidence;
- `ckm_not` for notification routing and participant alerts.

## Lifecycle

Session lifecycle:

1. A session is created with tenant, owner, context reference, and participant
   policy.
2. Participants join only when allowed by the session policy.
3. Presence is updated with heartbeat evidence.
4. Messages, media events, and decisions are recorded against the active
   session.
5. Session state changes require audit evidence.

Messaging lifecycle:

1. A message references an active session and author.
2. Sensitive content can require review before broad sharing.
3. The package records posted, review-required, or blocked status.
4. Provider adapters deliver the real-time fanout.

Media lifecycle:

1. Screen sharing requires explicit permission.
2. Recording requires participant consent evidence.
3. Live media signaling and transport use adapters.
4. Media events are audit-visible.

Decision lifecycle:

1. A participant captures a decision with decision text, owner, and trace
   reference.
2. Missing trace evidence denies capture.
3. Downstream workflow or task creation remains an adapter responsibility.

AI-agent lifecycle:

1. Agent is registered with runtime, role, scope, tenant, and disclosure.
2. Runtime must be one of `codex`, `claude_code`, `opencode`, or `pi`.
3. Role must be one of the configured RTC review roles.
4. Agent contributions are audit-visible and cannot bypass policy decisions.

## Rule Engine

Rules must deny or require review for:

- missing tenant context;
- missing session owner;
- missing participant policy;
- joining when not allowed by participant policy;
- presence updates without heartbeat evidence;
- messaging outside an active session;
- sensitive messages without review;
- screen sharing without permission;
- recording without consent;
- decisions without trace evidence;
- unregistered, unsupported, unscoped, or undisclosed AI agents;
- lifecycle state changes without audit evidence;
- batch RTC mutations that do not use Bytewax.

## UI And Theme

The APG Python UI contract exposes dashboard, rooms, presence, messages, media,
decisions, agents, rules, analytics, audit, and settings routes. The theme uses
compact operational density with distinct treatments for sessions, heartbeat,
retention, consent, decision trace, agent scope, stream health, and audit
decisions.

## Streaming

Batch RTC mutation must use Bytewax. The stream topic is
`apg.ckm_rtc.lifecycle`, and state covers sessions, participants, presence,
messages, media events, decisions, RTC agents, and audit events. Live Bytewax
topology deployment is an adapter concern, but the package declares and
enforces the guardrail.

## Adapter Boundaries

Adapters must handle:

- WebSocket/WebRTC/SIP/RTMP/Socket.IO/gRPC transport;
- persistent database storage;
- media recording storage;
- notification routing through `ckm_not`;
- authentication and permission checks through `auth`;
- audit durability through `audl`;
- scheduler and workflow handoff;
- Bytewax lifecycle topology and operational monitoring.

## Acceptance Gates

- Contract validates through the APG capability registry.
- Configuration schema includes sessions, presence, messaging, media,
  collaboration, RTC agents, governance, observability, adapters, UI, and theme.
- Rules cover participant policy, presence heartbeat, messages, sensitive
  content, screen share, recording, decisions, agents, audit, and Bytewax.
- Lifecycle service can create sessions, admit participants, update presence,
  post messages, enforce media guardrails, capture decisions, register agents,
  summarize state, and validate batch mutation streams.
- Generated semantic evidence exposes provides/requires, routes, rules, theme,
  and streaming.
- README, specification, plan, progress log, focused tests, implementation
  audit, publish plan, and stale-marker scan are current.
