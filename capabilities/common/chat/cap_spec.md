# Chat and Messaging Capability Specification

- **Capability Name**: Chat and Messaging
- **Capability ID**: `chat`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package implements the executable APG contract for `chat` as a
dependency-light team messaging runtime. It provides tenant rooms, membership
and guest policy state, retained message streams, delivery receipts, presence
state, moderation queue items, audit events, UI route metadata, semantic-model
publication, and publish-plan evidence without requiring an external realtime
broker.

## Provided Services

- `direct_messages`
- `team_rooms`
- `message_moderation`
- `presence`
- `message_retention`
- `capability_rules`

## Required Services

- `tenant_context`
- `message_event_bus`
- `notification_delivery`
- `authentication`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations. Rooms require owners, membership, retention policy, and guest
policy when external participants are present. Messages enforce maximum length,
membership, delivery metadata, and moderation rules. Large rooms can be created
as pending-review state instead of silently passing.

## Rules

- `tenant_context_required`
- `room_requires_owner`
- `retention_policy_required`
- `external_guest_requires_policy`
- `restricted_content_requires_moderation`
- `large_room_requires_review`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model. The dashboard view model surfaces
rooms, messages, presence, moderation queues, audit events, and conversation
summary metrics from `ChatService`.

## Theme

The package uses the `chat_team_messaging` APG theme contract.

## Runtime Behavior

`ChatService` maintains deterministic in-memory registries for rooms, messages,
presence state, moderation queue items, and audit events. `chat_engine.py`
generates canonical message fingerprints and thread keys from message payloads.
Room creation enforces tenant, owner, retention, guest-policy, and large-room
review rules. Message sending enforces active room membership, message length,
restricted-content moderation, attachment metadata, and delivery receipts.

## Known Integration Boundary

This package intentionally avoids live WebSocket, push notification, or message
broker calls. Realtime fanout, durable event streaming, external notification
delivery, identity enforcement, and collaborative document hooks should be
composed through APG capabilities such as `mqeb`, `ntfy`, `auth`, `audl`,
`nlpc`, `colb`, and `mten`.
