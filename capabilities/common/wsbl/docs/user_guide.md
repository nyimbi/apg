# Website Builder + WebSocket Broker

**Capability ID**: `wsbl` | **Domain**: `common` | **Version**: `1.1.0`

## Description

WSBL is the APG capability for governed website and page composition, extended
with an in-process WebSocket broker for real-time collaborative editing.  It
provides a composable runtime for tenant sites, domains, pages, components,
publishing workflows, live connections, rooms, presence, and collaborative
sessions — all governed through the WSBL policy engine.

## Installation

```bash
pip install apg-common-wsbl
```

## Provides

- `site_management`
- `page_composition`
- `component_library`
- `publishing_workflows`
- `site_theming`
- `realtime_connections`
- `room_management`
- `presence_protocol`
- `collaborative_sessions`
- `component_locking`
- `section_annotations`

## Requires

- `them` (theming)
- `auth` (authentication)
- `ncod` (no-code runtime)
- `accs` (accessibility)
- `cons` (consent)

---

## Part 1 — Website Builder

### Creating a Site

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
```

### Registering and Validating a Domain

```python
domain = service.register_domain(
    site_id=site["id"],
    tenant_id="tenant-a",
    domain="blog.example.test",
)

validated = service.validate_domain(domain["id"], actor_id="owner-1")
```

### Creating a Component

Custom components must be reviewed before they can be added to a page.

```python
# Pre-reviewed custom component
component = service.create_component(
    component_key="hero",
    tenant_id="tenant-a",
    name="Hero Banner",
    custom=True,
    reviewed=True,
    reviewed_by="reviewer-1",
    policy_id="component-policy",
)
```

### Building a Page

```python
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

### Publishing

Publishing enforces approval, domain validation, structured sections, preview
evidence, accessibility, and consent-policy gates.

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

### Rollback

```python
rolled_back = service.rollback_site(
    site_id=site["id"],
    version=1,
    actor_id="publisher-1",
    event_stream="bytewax",
)
```

### WSBL Agents

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
    human_approval_ref="approval://agent/publish",
)

assert decision["decision"] == "allow"
```

### Batch Publish Guardrail

```python
decision = service.validate_batch_publish(
    tenant_id="tenant-a",
    site_count=3,
    event_stream="bytewax",
)
assert decision["decision"] == "allow"
```

### SEO, Forms, Analytics, and Media

```python
service.seo_optimise(page["id"], "My Title", "My description", ["keyword"], "editor-1")
service.form_embed(page["id"], "form-1", "contact", "editor-1")
service.analytics_embed(site["id"], "plausible", "tracking-id", "editor-1")
service.media_upload(site["id"], "img-1", "s3://bucket/hero.jpg", "image/jpeg", "editor-1")
```

### Collaborative Authoring Helpers

```python
service.css_customise(site["id"], "body { font-family: sans-serif; }", "editor-1")
service.mobile_preview(site["id"], "editor-1")
service.ab_test_page(site["id"], page["id"], page2["id"], "test-1", 50, "editor-1")
service.sitemap_generate(site["id"], "editor-1")
service.site_export(site["id"], "editor-1")
```

---

## Part 2 — WebSocket Broker

All broker methods are `async`.  Use them inside an `asyncio` event loop or
from `async def` coroutines.

### Connection Lifecycle

```python
import asyncio

async def main():
    # Connect
    conn = await service.async_connect(
        "tenant-a", "conn-1", "editor-1",
        protocol_version="1.0",
        transport_meta={"ip": "10.0.0.1", "compression": "deflate"},
    )

    # Keep alive
    await service.async_heartbeat("tenant-a", "conn-1")

    # Evict idle connections after 30 s without a heartbeat
    reaped = await service.async_prune_dead_connections("tenant-a", max_idle_seconds=30)

    # Disconnect cleanly
    await service.async_disconnect("tenant-a", "conn-1", "editor-1", reason="normal")

asyncio.run(main())
```

### Room Management

```python
async def rooms_demo():
    site = service.create_site("blog", "tenant-a", "Blog", "editor-1")

    # Create a room
    room = await service.async_room_create(
        "tenant-a", "room-blog", site["id"], "editor-1",
        room_type="collaboration",   # or "review" / "observer"
        max_members=20,
    )

    # Join / leave
    await service.async_connect("tenant-a", "conn-a", "editor-1")
    await service.async_connect("tenant-a", "conn-b", "editor-2")
    await service.async_room_join("tenant-a", "room-blog", "conn-a", "editor-1")
    await service.async_room_join("tenant-a", "room-blog", "conn-b", "editor-2")
    await service.async_room_leave("tenant-a", "room-blog", "conn-b", "editor-2")

    # Close room (evicts all remaining members)
    await service.async_room_close("tenant-a", "room-blog", "editor-1")
```

### Presence Protocol

Presence records carry the current page, component, cursor position, and
intent (`viewing`, `editing`, `reviewing`).  They expire after `ttl_seconds`.

```python
async def presence_demo():
    await service.async_presence_update(
        "tenant-a", "conn-a", "editor-1",
        page_id="page-123",
        component_id="comp-456",
        cursor_position={"line": 12, "col": 5},
        intent="editing",
        ttl_seconds=30,
    )

    # Get all live presence records for a room (expired entries pruned)
    snapshot = await service.async_presence_snapshot("tenant-a", "room-blog")
    for record in snapshot:
        print(record["actor_id"], record["intent"], record["cursor_position"])
```

### Broadcast

```python
async def broadcast_demo():
    receipt = await service.async_broadcast(
        "tenant-a", "room-blog",
        message={"type": "section_update", "section_id": "s1", "data": {...}},
        actor_id="editor-1",
        exclude_connection_ids=["conn-a"],  # Don't echo to the sender
    )
    print(f"Delivered to {receipt['delivered']} connections")
```

### Collaborative Editing Sessions

A session tracks the active editing context for a connection.  Stale sessions
are reaped when no heartbeat is received within `max_idle_seconds`.

```python
async def session_demo():
    page = service.create_page(site["id"], "about", "About", "tenant-a")
    session = await service.async_session_start(
        "tenant-a", "conn-a", site["id"], page["id"], "editor-1"
    )

    # ... editor works ...

    # Explicitly end
    await service.async_session_end("tenant-a", session["id"], "editor-1")

    # Or let the reaper clean up orphaned sessions
    reaped = await service.async_reap_stale_sessions("tenant-a", max_idle_seconds=60)
```

### Component Locking

Prevent two editors from simultaneously modifying the same component.

```python
async def lock_demo():
    comp = service.create_component(
        "nav", "tenant-a", "Navigation", reviewed=True, reviewed_by="r1"
    )

    # Acquire lock
    lock = await service.async_lock_component(
        "tenant-a", comp["id"], "conn-a", "editor-1", lock_ttl_seconds=120
    )

    # Another connection attempting to lock will raise PermissionError
    try:
        await service.async_lock_component(
            "tenant-a", comp["id"], "conn-b", "editor-2"
        )
    except PermissionError as exc:
        print(exc)  # component_locked_by_another:<id>

    # Release
    await service.async_unlock_component("tenant-a", comp["id"], "conn-a", "editor-1")

    # System can always force-unlock
    await service.async_unlock_component("tenant-a", comp["id"], "any-conn", "system")
```

### Section Annotations

In-context review feedback attached to a page section, visible in real time to
all room members.

```python
async def annotation_demo():
    page = service.create_page(site["id"], "home", "Home", "tenant-a")
    section_id = "section-1"

    annot = await service.async_annotate_section(
        "tenant-a", page["id"], section_id, "reviewer-1",
        text="Needs stronger CTA copy",
    )

    # Fetch open annotations
    open_annots = await service.async_list_annotations("tenant-a", page["id"])

    # Fetch all including resolved
    all_annots = await service.async_list_annotations(
        "tenant-a", page["id"], include_resolved=True
    )
```

### Channel Authorization

Authorise a connection against the WSBL policy engine before allowing it to
join a sensitive channel.

```python
async def authz_demo():
    try:
        result = await service.async_authorize_channel(
            "tenant-a", "conn-a", "channel://publish-review", "wsbl:publish"
        )
        print("Allowed:", result["decision"])
    except PermissionError as exc:
        print("Denied:", exc)
```

### Query Helpers

```python
# All live connections for a tenant
connections = service.list_connections("tenant-a")

# All rooms (open and closed)
rooms = service.list_rooms("tenant-a")

# Active component locks
locks = service.list_component_locks("tenant-a")
```

---

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/wsbl/dashboard` | `wsbl:view` | Overview |
| `/wsbl/sites` | `wsbl:manage_sites` | Sites |
| `/wsbl/pages` | `wsbl:build` | Pages |
| `/wsbl/editor` | `wsbl:build` | Build |
| `/wsbl/components` | `wsbl:build` | Build |
| `/wsbl/publishing` | `wsbl:publish` | Release |
| `/wsbl/analytics` | `wsbl:view` | Operations |
| `/wsbl/agents` | `wsbl:admin` | Automation |

---

## Bytewax Streams

| Stream | Key | Description |
|--------|-----|-------------|
| `apg.wsbl.lifecycle` | `tenant_id` | Site, page, and publish lifecycle events |
| `apg.wsbl.realtime` | `tenant_id` | WebSocket broker events |

---

## Interoperability

Reference this capability in `.apg` source files:

```apg
use wsbl;
```

---

## Configuration

All configuration keys are tenant-scoped.  Set via the `conf` capability or
environment variables prefixed with `WSBL_`.

Key broker tuning knobs:

| Variable | Default | Description |
|----------|---------|-------------|
| `WSBL_HEARTBEAT_INTERVAL_S` | `10` | Client heartbeat interval (seconds) |
| `WSBL_MAX_IDLE_S` | `30` | Max idle before connection is reaped |
| `WSBL_COMPONENT_LOCK_TTL_S` | `60` | Default component lock TTL |
| `WSBL_PRESENCE_TTL_S` | `30` | Default presence record TTL |
| `WSBL_ROOM_MAX_MEMBERS` | `50` | Default room capacity |
| `WSBL_SESSION_MAX_IDLE_S` | `60` | Max session idle before reap |

---

## Further Reading

- `service.py` — Business logic and WebSocket broker implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
- `WORLD_CLASS_IMPROVEMENTS.md` — Planned enhancements roadmap
