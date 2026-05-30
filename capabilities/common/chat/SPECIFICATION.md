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
- AI-agent participants with registration, scope, and response disclosure.
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
- cross-tenant access denial
- Bytewax batch mutation enforcement

## Runtime

`service.ChatService` is the generated-application runtime. It stores deterministic in-memory state for:

- rooms
- messages
- presence
- moderation queue items
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
- moderation
- retention
- audit
- analytics
- settings

Each screen must have route metadata, permissions, and theme-backed view model support.

## AI-Agent Composition

AI agents are first-class chat participants when enabled by configuration. A generated application may add agents backed by Codex, Claude Code, OpenCode, Pi, or another adapter, but every agent message must satisfy:

- registered agent identity
- explicit room scope
- visible response disclosure
- tenant-local access
- audit evidence

The capability does not invoke external agent CLIs directly.

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
