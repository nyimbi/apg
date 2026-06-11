# WSBL World-Class Improvements

**Capability**: Website Builder + WebSocket Broker (wsbl)
**Date**: 2026-06-11
**Author**: Nyimbi Odero

---

## Overview

The wsbl service layer is functionally complete for governed website composition
and publication. The following 15 improvements target real-time collaboration,
presence, room management, and operational excellence — turning wsbl into a
live, multi-user editing platform with WebSocket broker semantics layered on top
of the existing site/page/component governance model.

---

## 1. Async-first Method Signatures

All new service methods should be `async def`. Synchronous callers pay an
unnecessary abstraction tax when the service eventually connects to real async
transports (Redis pub/sub, asyncpg, Bytewax workers). Convert the entire public
surface to `async def` and provide thin sync wrappers only for CLI/test shims.

**Impact**: unblocks I/O-bound operations; aligns with FastAPI/Starlette
integration patterns.

---

## 2. WebSocket Connection Registry

Add `async_connect` / `async_disconnect` to maintain a live connection registry
keyed by `(tenant_id, connection_id)`. Include transport metadata (IP, protocol
version, compression) and last-seen heartbeat. This registry is the foundation
for presence, room membership, and targeted messaging.

**Impact**: enables per-connection routing, eviction of stale sockets, and
audit-grade connection lifecycle logging.

---

## 3. Room Management Subsystem

Add `async_room_create`, `async_room_join`, `async_room_leave`, and
`async_room_close` methods. Rooms are logical channels scoped to a
`(tenant_id, site_id)` context, enabling page-level or component-level editing
isolation. Room membership is append-logged for audit reconstruction.

**Impact**: replaces ad-hoc broadcast patterns with governed, traceable channels.

---

## 4. Presence Protocol

Add `async_presence_update` and `async_presence_snapshot`. Each connection
carries cursor position, active page, active component, and intent
(`viewing | editing | reviewing`). Snapshots are delivered on room join and
on 5-second heartbeat intervals. Presence data expires after
`presence_ttl_seconds` (default: 30).

**Impact**: enables Google Docs-style live cursors and editor conflict prevention.

---

## 5. Typed Message Bus with Schema Validation

Replace raw `dict` message payloads with a discriminated-union message type
hierarchy (`EditMessage`, `PresenceMessage`, `SystemMessage`, `AuditMessage`).
Validate on ingress with Pydantic v2 `model_validate`. Reject malformed frames
before they reach business logic.

**Impact**: eliminates entire class of runtime KeyError / silent data loss bugs.

---

## 6. Optimistic Lock + Conflict Resolution on Page Sections

Add `expected_version: int` to `add_page_section`. Raise `ConflictError` if
`page.version != expected_version`. Callers retry with a merge or a forced
override. This removes the silent last-write-wins behaviour when two editors add
sections concurrently.

**Impact**: correct concurrent editing semantics without full CRDT overhead.

---

## 7. Targeted Room Broadcast

Add `async_broadcast` method: deliver a message to all connections in a room
with fan-out tracking. Accept an `exclude_connection_ids` set to suppress
echo to the sender. Return a delivery receipt (`delivered: int`,
`failed: list[str]`).

**Impact**: powers real-time section updates, component-approval notifications,
and live publish-status banners.

---

## 8. Rate-Limiting Middleware Hook

Add `async_check_rate_limit(tenant_id, connection_id, operation)` that
reads from a `BoundedCache`-backed token-bucket. Configurable per-tenant
burst and sustained rates. Method raises `RateLimitError` before any
business logic executes.

**Impact**: protects the broker from runaway agents and malformed clients
without requiring an external API gateway.

---

## 9. Collaborative Editing Session Lifecycle

Add `async_session_start` and `async_session_end`. A session wraps a
`(tenant_id, site_id, page_id, actor_id)` tuple with a heartbeat timeout.
Sessions emit `session_started` / `session_ended` audit events and drive
presence expiry. On abnormal disconnect, sessions are auto-closed via
`async_reap_stale_sessions`.

**Impact**: clean resource accounting; ensures presence data is never left
orphaned.

---

## 10. Pub/Sub Adapter Interface

Define an `AbstractBrokerBackend` protocol with `publish`, `subscribe`, and
`unsubscribe` async methods. Ship two concrete adapters:
`InMemoryBrokerBackend` (default, used in tests) and a `RedisBrokerBackend`
stub. The `WsblService` accepts a backend via constructor injection.

**Impact**: production deployments swap to Redis without touching business logic;
test suite stays dependency-free.

---

## 11. Component Lock / Unlock for Live Editing

Add `async_lock_component` and `async_unlock_component`. While a connection
holds a lock, other connections in the room receive a `component_locked`
presence event and their edit attempts are rejected with `ComponentLockedError`.
Locks auto-expire after `lock_ttl_seconds` (default: 60).

**Impact**: prevents simultaneous destructive edits to shared components during
live review sessions.

---

## 12. WebSocket Heartbeat and Liveness Probe

Add `async_heartbeat(connection_id, tenant_id)` that refreshes the last-seen
timestamp. Add `async_prune_dead_connections(tenant_id, max_idle_seconds)` that
scans the registry and evicts connections that missed two consecutive heartbeat
windows. Emit `connection_reaped` audit events.

**Impact**: keeps the connection registry accurate; prevents ghost presence
entries from accumulating.

---

## 13. Channel-Scoped Access Control

Add `async_authorize_channel(tenant_id, connection_id, channel, required_perm)`
that evaluates capability rules before admitting a connection to a room or
broadcast channel. Integrates with the existing `evaluate()` engine and WSBL
agent role model.

**Impact**: enforces tenant-scoped RBAC at the transport layer, not just the
API layer.

---

## 14. Structured Telemetry Events

Add `async_emit_telemetry(event_type, payload)` that serialises a structured
`TelemetryEvent` to the Bytewax stream `apg.wsbl.realtime`. Events include
latency buckets, room occupancy, message rates, and error codes.

**Impact**: feeds observability dashboards without requiring instrumentation
inside every method.

---

## 15. Collaborative Annotation Layer

Add `async_annotate_section` and `async_list_annotations`. Annotations are
`(section_id, actor_id, text, resolved)` records attached to a page section.
All connected room members receive `annotation_added` / `annotation_resolved`
events in real time. Annotations persist in the audit log.

**Impact**: replaces out-of-band Slack/email review threads with in-context,
traceable feedback loops.
