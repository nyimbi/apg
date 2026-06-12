# WSBL - Website Builder + WebSocket Broker

WSBL is the APG capability for governed website and page composition, extended
with a real-time WebSocket broker layer for live collaborative editing.  It
gives generated applications a composable runtime for tenant sites, domains,
pages, components, public-site controls, publishing, rollback, accessibility,
privacy-consent policy, AI-assisted review, Bytewax lifecycle events, and
real-time presence, rooms, and collaborative sessions.

Use WSBL when an application needs a website builder that can move from draft
composition to controlled publication without skipping governance evidence, and
where multiple editors can collaborate in real time on the same site or page.

## What WSBL Provides

**Website Builder (synchronous)**

- Tenant-scoped site registry.
- Domain registration and validation state.
- Versioned pages built from structured sections.
- Governed standard and custom component library.
- Custom component review and policy attribution.
- Publication requests with approval, domain, section, preview, accessibility,
  and privacy-consent gates.
- Durable review evidence for custom components, publish requests, denied
  publish attempts, agent publish checks, batch publish checks, and audit
  events.
- Rollback of published site versions.
- First-class WSBL agents for Codex, Claude Code, OpenCode, and Pi based review
  lanes.
- Bytewax lifecycle stream metadata.
- Dashboard, site, page, editor, component, publishing, analytics, agent,
  policy, and settings view models.

**WebSocket Broker (async, v2.0)**

- Tenant-scoped WebSocket connection registry with transport metadata.
- Heartbeat tracking and idle-connection pruning.
- Room creation, join, leave, and close with capacity enforcement.
- Presence protocol: cursor position, active page/component, intent TTL.
- Presence snapshots delivered on room join.
- Fan-out broadcast with delivery receipts and sender exclusion.
- Collaborative editing sessions with heartbeat-based lifecycle.
- Stale session reaping.
- Exclusive component locks with TTL and connection-scoped ownership.
- In-context section annotations visible to all room members.
- Channel-level access control integrated with the WSBL policy engine.
- Pluggable pub/sub backend (`InMemoryBrokerBackend` / `RedisBrokerBackend`).
- Structured telemetry events emitted to `apg.wsbl.realtime`.
- Typed discriminated-union message bus with Pydantic v2 ingress validation.
- Optimistic locking on page sections (`expected_version`) with `ConflictError`.
- Per-tenant rate limiting via token-bucket middleware hook.

## Quick Start

```python
from capabilities.common.wsbl import WsblService

service = WsblService()

site = service.create_site(
    site_key="marketing",
    tenant_id="tenant-a",
    name="Marketing Site",
    owner_id="owner-1",
    primary_domain="www.example.test",
    domain_validated=True,
)

component = service.create_component(
    component_key="hero",
    tenant_id="tenant-a",
    name="Hero Banner",
    custom=True,
    reviewed=True,
    reviewed_by="reviewer-1",
    policy_id="component-policy",
)

page = service.create_page(
    site_id=site["id"],
    slug="home",
    title="Home",
    tenant_id="tenant-a",
)

page = service.add_page_section(
    page_id=page["id"],
    component_id=component["id"],
    content={"headline": "Welcome"},
    actor_id="editor-1",
)
```

## WebSocket Broker: Quick Start

```python
import asyncio
from capabilities.common.wsbl import WsblService

service = WsblService()

async def demo():
    # Connect two editors
    await service.async_connect("tenant-a", "conn-1", "editor-1")
    await service.async_connect("tenant-a", "conn-2", "editor-2")

    # Open a collaboration room for the site
    site = service.create_site("blog", "tenant-a", "Blog", "editor-1")
    room = await service.async_room_create(
        "tenant-a", "room-blog", site["id"], "editor-1", room_type="collaboration"
    )
    await service.async_room_join("tenant-a", "room-blog", "conn-1", "editor-1")
    await service.async_room_join("tenant-a", "room-blog", "conn-2", "editor-2")

    # Publish presence
    await service.async_presence_update(
        "tenant-a", "conn-1", "editor-1", intent="editing"
    )

    # Broadcast a content-change event
    receipt = await service.async_broadcast(
        "tenant-a", "room-blog",
        message={"type": "section_update", "section_id": "s1"},
        actor_id="editor-1",
        exclude_connection_ids=["conn-1"],
    )
    assert receipt["delivered"] == 1

    # Start a collaborative session
    page = service.create_page(site["id"], "home", "Home", "tenant-a")
    session = await service.async_session_start(
        "tenant-a", "conn-1", site["id"], page["id"], "editor-1"
    )

    # Lock a component while editing
    comp = service.create_component("hero", "tenant-a", "Hero", reviewed=True, reviewed_by="r1")
    await service.async_lock_component("tenant-a", comp["id"], "conn-1", "editor-1")

    # Annotate a section
    annot = await service.async_annotate_section(
        "tenant-a", page["id"], "section-1", "reviewer-1", "Needs stronger CTA"
    )

    # Clean up
    await service.async_unlock_component("tenant-a", comp["id"], "conn-1", "editor-1")
    await service.async_session_end("tenant-a", session["id"], "editor-1")
    await service.async_room_close("tenant-a", "room-blog", "editor-1")

asyncio.run(demo())
```

## Publishing

Publishing requires approval, validated domains, structured sections, preview
evidence, accessibility evidence, consent-policy handling, and Bytewax stream
metadata.

```python
request = service.create_publish_request(
    site_id=site["id"],
    requested_by="publisher-1",
    approval_recorded=True,
    accessibility_passed=True,
    consent_policy_attached=True,
    preview_evidence_present=True,
    event_stream="bytewax",
)

published = service.publish_site(
    publish_request_id=request["id"],
    actor_id="publisher-1",
)
```

## WSBL Agents

WSBL treats website-builder agents as governed composition elements.

```python
agent = service.register_wsbl_agent(
    tenant_id="tenant-a",
    name="Publish reviewer",
    runtime="codex",
    role="publish_reviewer",
    scope="review publish evidence and accessibility gates",
)

decision = service.validate_agent_publish_action(
    tenant_id="tenant-a",
    agent_id=agent["id"],
    action="publish_site",
    privileged_scope=True,
)

assert decision["decision"] == "deny"
```

Privileged agent publish actions require human approval:

```python
decision = service.validate_agent_publish_action(
    tenant_id="tenant-a",
    agent_id=agent["id"],
    action="publish_site",
    privileged_scope=True,
    human_approval_ref="approval://agent/publish",
)

assert decision["decision"] == "allow"
```

## Review Evidence

WSBL keeps review and denial evidence durable. Custom components created
without review are stored as `review_required` with `decision`,
`matched_rules`, `review_reasons`, and `audit_evidence`. Publish requests that
need consent-policy review are stored the same way, while hard-denied publish
attempts are persisted as `denied` before `PermissionError` is raised.

Generated applications can call `list_pending_reviews()` or use the dashboard,
component library, publish queue, and policy-center view models to compose
approval queues without replaying policy rules.

## Batch Publish Guardrail

Batch publishing must use Bytewax stream coordination.

```python
decision = service.validate_batch_publish(
    tenant_id="tenant-a",
    site_count=3,
    event_stream="bytewax",
)

assert decision["decision"] == "allow"
```

## World-Class Enhancements (v2.0)

The following 15 improvements were applied to the broker layer to support
production-grade multi-user editing:

1. **Async-first method signatures** — entire public surface converted to
   `async def`; sync wrappers provided only for CLI/test shims.
2. **WebSocket connection registry** — `async_connect` / `async_disconnect`
   maintain a live `(tenant_id, connection_id)` registry with transport
   metadata and heartbeat timestamps.
3. **Room management subsystem** — `async_room_create/join/leave/close` scope
   channels to `(tenant_id, site_id)`; membership is append-logged for audit.
4. **Presence protocol** — `async_presence_update` / `async_presence_snapshot`
   carry cursor position, active page/component, and intent; TTL-expiring,
   delivered on room join and on heartbeat.
5. **Typed message bus** — discriminated-union hierarchy
   (`EditMessage`, `PresenceMessage`, `SystemMessage`, `AuditMessage`)
   validated with Pydantic v2 on ingress; malformed frames rejected before
   business logic.
6. **Optimistic locking on page sections** — `expected_version` parameter on
   `add_page_section`; raises `ConflictError` on version mismatch, eliminating
   silent last-write-wins.
7. **Targeted room broadcast** — `async_broadcast` fans out with
   `exclude_connection_ids` sender-echo suppression; returns
   `{delivered, failed, sent_at}` delivery receipt.
8. **Rate-limiting middleware hook** — `async_check_rate_limit` uses a
   `BoundedCache`-backed token bucket; configurable per-tenant burst and
   sustained rates; raises `RateLimitError` before business logic.
9. **Collaborative session lifecycle** — `async_session_start` /
   `async_session_end` / `async_reap_stale_sessions`; emit audit events;
   presence data never left orphaned on abnormal disconnect.
10. **Pub/sub adapter interface** — `AbstractBrokerBackend` protocol;
    `InMemoryBrokerBackend` (default) and `RedisBrokerBackend` stub; injected
    via constructor for zero-touch production swap.
11. **Component lock / unlock** — `async_lock_component` / `async_unlock_component`
    give connections exclusive edit rights with TTL auto-expiry; contention
    raises `ComponentLockedError`.
12. **WebSocket heartbeat and liveness probe** — `async_heartbeat` refreshes
    last-seen; `async_prune_dead_connections` evicts idle connections and emits
    `connection_reaped` audit events.
13. **Channel-scoped access control** — `async_authorize_channel` evaluates
    capability rules via the existing `evaluate()` engine before admitting a
    connection; tenant-scoped RBAC at the transport layer.
14. **Structured telemetry events** — `async_emit_telemetry` serialises a
    `TelemetryEvent` (latency buckets, room occupancy, message rates, error
    codes) to the `apg.wsbl.realtime` Bytewax stream.
15. **Collaborative annotation layer** — `async_annotate_section` /
    `async_list_annotations` attach `(section_id, actor_id, text, resolved)`
    records; all room members receive `annotation_added` / `annotation_resolved`
    events in real time; persisted in the audit log.

## New Methods

### async_broadcast — fan-out with delivery receipt

```python
receipt = await service.async_broadcast(
    tenant_id="tenant-a",
    room_id="room-blog",
    message={"type": "section_update", "section_id": "s1", "content": {...}},
    actor_id="editor-1",
    exclude_connection_ids=["conn-1"],   # suppress echo to sender
)
# {"delivered": 2, "failed": [], "sent_at": "2026-06-12T..."}
```

### async_lock_component / async_unlock_component — exclusive edit locks

```python
lock = await service.async_lock_component(
    tenant_id="tenant-a",
    component_id=comp["id"],
    connection_id="conn-1",
    actor_id="editor-1",
    lock_ttl_seconds=60,
)
# Other connections receive component_locked presence event; edit attempts
# raise ComponentLockedError until unlock or TTL expiry.

await service.async_unlock_component("tenant-a", comp["id"], "conn-1", "editor-1")
```

### async_annotate_section — in-context review feedback

```python
annotation = await service.async_annotate_section(
    tenant_id="tenant-a",
    page_id=page["id"],
    section_id="hero-section",
    actor_id="reviewer-1",
    text="CTA copy needs stronger verb — 'Get started' not 'Learn more'",
)

open_annotations = await service.async_list_annotations(
    tenant_id="tenant-a",
    page_id=page["id"],
    include_resolved=False,
)
```

### async_prune_dead_connections — connection registry hygiene

```python
# Called by a background worker every 30 seconds.
reaped = await service.async_prune_dead_connections(
    tenant_id="tenant-a",
    max_idle_seconds=30,
)
# Returns list of evicted connection IDs; emits connection_reaped audit events.
```

### async_authorize_channel — transport-layer RBAC

```python
try:
    result = await service.async_authorize_channel(
        tenant_id="tenant-a",
        connection_id="conn-1",
        channel="room-blog",
        required_perm="edit",
        actor_id="editor-1",
    )
    # result["decision"] == "allow"
except PermissionError as exc:
    # channel_access_denied — reject the WebSocket frame before business logic
    ...
```

## Deterministic Rules

WSBL enforces:

- tenant context on all executable operations;
- site ownership;
- domain validation before publishing;
- structured sections before publishing;
- preview evidence before publishing;
- approval before publishing;
- Bytewax stream metadata for publish and rollback;
- custom component review and policy attribution;
- accessibility pass for public sites;
- consent policy review for privacy banners;
- Bytewax coordination for batch publishing;
- supported WSBL-agent runtime and role;
- human approval for privileged agent actions.

## API Helpers

`api.py` provides payload-oriented helpers:

- `capability_status()`
- `create_site()`
- `validate_domain()`
- `create_component()`
- `review_component()`
- `create_page()`
- `add_page_section()`
- `create_publish_request()`
- `publish_site()`
- `rollback_site()`
- `register_wsbl_agent()`
- `validate_agent_publish_action()`
- `validate_batch_publish()`
- `list_pending_reviews()`
- `create_record()`
- `list_records()`
- `list_website_builder()`

**WebSocket Broker service methods (all `async`):**

- `async_connect(tenant_id, connection_id, actor_id, ...)`
- `async_disconnect(tenant_id, connection_id, actor_id, ...)`
- `async_heartbeat(tenant_id, connection_id)`
- `async_prune_dead_connections(tenant_id, max_idle_seconds)`
- `async_room_create(tenant_id, room_id, site_id, actor_id, ...)`
- `async_room_join(tenant_id, room_id, connection_id, actor_id)`
- `async_room_leave(tenant_id, room_id, connection_id, actor_id)`
- `async_room_close(tenant_id, room_id, actor_id)`
- `async_presence_update(tenant_id, connection_id, actor_id, ...)`
- `async_presence_snapshot(tenant_id, room_id)`
- `async_broadcast(tenant_id, room_id, message, actor_id, ...)`
- `async_session_start(tenant_id, connection_id, site_id, page_id, actor_id, ...)`
- `async_session_end(tenant_id, session_id, actor_id)`
- `async_reap_stale_sessions(tenant_id, max_idle_seconds)`
- `async_lock_component(tenant_id, component_id, connection_id, actor_id, ...)`
- `async_unlock_component(tenant_id, component_id, connection_id, actor_id)`
- `async_annotate_section(tenant_id, page_id, section_id, actor_id, text, ...)`
- `async_list_annotations(tenant_id, page_id, include_resolved)`
- `async_authorize_channel(tenant_id, connection_id, channel, required_perm, ...)`
- `list_connections(tenant_id)`
- `list_rooms(tenant_id)`
- `list_component_locks(tenant_id)`

## UI Routes

- dashboard: `/wsbl/dashboard`
- sites: `/wsbl/sites`
- pages: `/wsbl/pages`
- editor: `/wsbl/editor`
- components: `/wsbl/components`
- publishing: `/wsbl/publishing`
- analytics: `/wsbl/analytics`
- agents: `/wsbl/agents`
- policy: `/wsbl/policy`
- settings: `/wsbl/settings`

## Bytewax Stream

WSBL publishes lifecycle metadata for Bytewax:

- processor: `bytewax`
- stream: `apg.wsbl.lifecycle`
- key: `tenant_id`

Events (website builder):

- `site_created`
- `domain_registered`
- `domain_validated`
- `component_created`
- `component_reviewed`
- `page_created`
- `page_section_added`
- `publish_request_created`
- `site_published`
- `site_rolled_back`
- `wsbl_agent_registered`

Events (WebSocket broker — stream `apg.wsbl.realtime`):

- `ws_connected`
- `ws_disconnected`
- `connection_reaped`
- `ws_room_created`
- `ws_room_joined`
- `ws_room_left`
- `ws_room_closed`
- `ws_broadcast`
- `ws_session_started`
- `ws_session_ended`
- `ws_component_locked`
- `ws_component_unlocked`
- `ws_annotation_added`
- `ws_channel_authorized`
- `ws_channel_denied`

## Adapter Boundaries

The in-package service stores records in memory so generated applications,
tests, and publish-plan probes can execute without external infrastructure.
Production systems should attach visual editors, asset stores, preview
renderers, accessibility scanners, consent platforms, analytics collectors, CDN
or static-host deployment, search/sitemap systems, audit sinks, Bytewax
workers, and a real WebSocket transport (e.g. Redis pub/sub via
`RedisBrokerBackend`) through APG adapters.

## Verification

```bash
./.venv/bin/python -m py_compile capabilities/common/wsbl/__init__.py capabilities/common/wsbl/capability_contract.py capabilities/common/wsbl/models.py capabilities/common/wsbl/website_runtime.py capabilities/common/wsbl/service.py capabilities/common/wsbl/api.py capabilities/common/wsbl/views.py capabilities/common/wsbl/app.py capabilities/common/wsbl/test_capability_contract.py capabilities/common/wsbl/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/wsbl/test_capability_contract.py capabilities/common/wsbl/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/wsbl --json
./.venv/bin/apg capabilities publish-plan capabilities/common/wsbl --json
```
