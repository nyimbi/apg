# Chat and Messaging Capability Specification

## Purpose

`chat` is the APG common capability for tenant-scoped communication. It enables generated applications to compose rooms, direct messages, presence, moderation, retention, AI-agent participants, audit, analytics, and visual theming without requiring live realtime infrastructure during generation.

## Scope

The capability must support:

- Tenant-local rooms and direct-message spaces.
- Room owners, members, external guests, and retention policies.
- Message sending with attachment metadata, delivery receipts, content moderation, duplicate-message protection, and audit.
- Presence and typing indicators.
- Large-room access review.
- Moderation review for restricted content and access-review items.
- First-class AI-agent participants and governance agents with registration, supported runtimes, supported roles, explicit scope, accountable owner, declared purpose, contribution disclosure, and human approval evidence for privileged roles.
- Bytewax lifecycle batch validation for room, message, thread, reaction, presence, moderation, retention, guest-access, and chat-agent mutation streams.
- UI route contracts and view models for operational use.
- Bytewax-backed event-stream configuration for batch chat mutations.

## Dependencies

Required:

- `ntfy` for downstream notification composition.
- `mqeb` for message/event bus composition.
- `auth` for authenticated sender and user context.

Optional:

- `mten`, `audl`, `nlpc`, `colb`, `secu`, and `cach`.

## Configuration

The authoritative configuration lives in `capability_contract.py` and includes:

- `rooms`
- `messaging`
- `presence`
- `moderation`
- `ai_agents`
- `security`
- `governance`
- `retention`
- `observability`
- `agents`
- `streaming`
- `adapters`
- `ui`
- `theme`

## Rules

The rule engine must be deterministic and side-effect free. The minimum guardrail set covers:

- tenant context
- room ownership, names, members, retention, external guest policy, guest expiry, and large-room review
- active room, sender identity, sender membership, message payload, message length, duplicate message IDs, and delivery event-bus evidence
- restricted content moderation
- attachment scan evidence
- DLP review for external sharing
- audit evidence for state changes
- thread, reaction, edit, and delete authorization checks
- presence identity and typing membership checks
- moderation reviewer and decision checks
- retention export approval
- AI-agent registration, scope, and disclosure checks
- first-class chat-agent runtime, role, scope, owner, purpose, contribution-disclosure, and privileged-role approval checks
- Bytewax lifecycle batch operation and mutation-count checks
- cross-tenant access denial
- Bytewax batch mutation enforcement

## Runtime

`service.ChatService` is the generated-application runtime. It stores deterministic in-memory state for:

- rooms
- messages
- presence
- moderation queue items
- chat agents
- lifecycle batch records
- audit events

The service uses tenant-qualified storage keys so public business IDs can repeat safely across tenants.

## UI

The UI contract must expose these screens:

- dashboard
- rooms
- direct
- messages
- presence
- agents
- lifecycle
- moderation
- retention
- audit
- analytics
- settings

Each screen must have route metadata, permissions, and theme-backed view model support.

## AI-Agent Composition

AI agents are first-class chat participants and governable application components when enabled by configuration. A generated application may add agents backed by Codex, Claude Code, OpenCode, Pi, or another adapter, but every agent message or lifecycle contribution must satisfy:

- registered agent identity
- explicit room scope
- supported runtime
- supported role
- accountable owner
- declared purpose
- machine contribution disclosure
- human approval evidence for privileged moderation, retention, guest-access, attachment, lifecycle, and chat-steward roles
- visible response disclosure
- tenant-local access
- audit evidence

The capability does not invoke external agent CLIs directly.

## Lifecycle Composition

CHAT lifecycle batches must be Bytewax-backed. The top-level `streaming` manifest declares the `chat.lifecycle` stream, `event_time` watermarking, the required `bytewax` processor, supported mutation operations, and the topic names generated applications should use when connecting durable infrastructure. The in-memory runtime validates these batches before they can be treated as accepted lifecycle state.

## Out Of Scope

This packet intentionally excludes:

- live WebSocket servers
- durable message brokers
- live Bytewax workers
- file scanning services
- external identity providers
- external AI-agent runtimes
- browser-rendered UI verification
- database persistence

Those belong in adapters or later production integration slices.
