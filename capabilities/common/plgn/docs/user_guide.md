# Plugin/Extension Framework — User Guide

**Capability ID**: `plgn` | **Domain**: `common` | **Version**: `1.1.0`

---

## Overview

PLGN gives APG applications a tenant-scoped extension system covering the full
plugin lifecycle: manifest registration, permission review, sandbox policy,
marketplace listing, release gating, installation, activation, event hooks,
sandboxed execution, analytics, dependency resolution, and AI governance agents.

The service is fully async-capable. Every write path has a native `async`
counterpart for use in async frameworks (FastAPI, Starlette, asyncio services).
Sync methods remain for backward compatibility and scripting contexts.

---

## Installation

```bash
pip install apg-common-plgn
```

---

## Provides

- `plugin_registry`
- `extension_marketplace`
- `permission_review`
- `sandbox_policy`
- `plugin_release_lifecycle`
- `async_event_dispatch`

## Requires

- `auth`
- `secu`
- `conf`
- `audl`

---

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/plgn/dashboard` | `plgn:view` | Overview |
| `/plgn/marketplace` | `plgn:install` | Marketplace |
| `/plgn/plugins` | `plgn:view` | Plugins |
| `/plgn/manifests` | `plgn:publish` | Plugins |
| `/plgn/permissions` | `plgn:review` | Security |
| `/plgn/sandbox` | `plgn:review` | Security |
| `/plgn/releases` | `plgn:publish` | Release |
| `/plgn/agents` | `plgn:admin` | Operations |

---

## Quick Start

```python
from capabilities.common.plgn import PlgnService

svc = PlgnService()

# 1. Register
plugin = svc.register_plugin(
    name="Risk Scorer", version="1.0.0", author="alice",
    entry_point="risk_scorer.main",
    permissions=["identity", "network:external"],
    tenant_id="acme",
    plugin_id="risk-scorer",
    external_plugin=True,
    external_review_recorded=True,
    permission_review_recorded=True,
)

# 2. Review permissions
svc.review_permissions(
    review_id="rev-001", tenant_id="acme", plugin_id="risk-scorer",
    reviewer="security-lead",
    approved_scopes=["identity", "network:external"],
    secret_access_allowed=True,
)

# 3. Attach sandbox policy
svc.attach_sandbox_policy(
    policy_id="sbx-001", tenant_id="acme", plugin_id="risk-scorer",
    policy_name="restricted",
    network_access="allow", filesystem_access="read_only",
    tool_allowlist=["score_customer"],
)

# 4. Publish marketplace listing
svc.publish_listing(
    listing_id="lst-001", tenant_id="acme", plugin_id="risk-scorer",
    title="Risk Scorer Extension",
    publisher_verified=True, curated=True,
)

# 5. Create release
svc.create_release(
    release_id="rel-001", tenant_id="acme", plugin_id="risk-scorer",
    version="1.0.0", channel="stable",
    signature_ref="sha256:abc123",
)

# 6. Install and enable
inst = svc.install_plugin("risk-scorer", "acme", installed_by="admin")
svc.enable_plugin(inst["id"], "acme", actor="admin")
```

---

## Service Methods Reference

### Synchronous

| Method | Description |
|--------|-------------|
| `describe(tenant_id)` | Return capability contract for a tenant |
| `evaluate(context)` | Evaluate governance rules against a context dict |
| `register_plugin(...)` | Register a plugin manifest with security posture |
| `install_plugin(plugin_id, tenant_id, ...)` | Install a registered plugin |
| `uninstall_plugin(plugin_id, tenant_id, ...)` | Remove an active installation |
| `plugin_update(tenant_id, plugin_id, new_version, ...)` | Bump version, audit trail |
| `enable_plugin(installation_id, tenant_id, actor)` | Activate an installation |
| `plugin_disable(tenant_id, installation_id, ...)` | Deactivate without uninstall |
| `plugin_health_check(plugin_id, tenant_id, ...)` | Signature/sandbox/dep/perm checks |
| `plugin_event_hook(event_name, plugin_id, handler, ...)` | Register event hook |
| `hook_fire(tenant_id, event_name, payload, ...)` | Serial event dispatch |
| `plugin_sandboxed_execution(plugin_id, method, params, ...)` | Policy-gated sandbox call |
| `plugin_permission_check(plugin_id, permission, ...)` | Per-scope permission query |
| `plugin_marketplace_listing(tenant_id, ...)` | Channel/curation-filtered listing |
| `plugin_analytics(tenant_id, period)` | Aggregated plugin statistics |
| `dashboard_summary(tenant_id)` | Overview counts for a tenant |
| `plugin_dependency_resolution(plugin_ids, tenant_id)` | Topological resolve + conflict detection |
| `review_permissions(...)` | Record a permission review |
| `attach_sandbox_policy(...)` | Attach a sandbox policy to a plugin |
| `publish_listing(...)` | Publish a marketplace listing |
| `create_release(...)` | Gate and record a plugin release |
| `register_plgn_agent(...)` | Register an AI governance agent |
| `audit_plugin_action(...)` | Explicit audit record |
| `list_plugins / list_installations / list_releases / ...` | Tenant-scoped list helpers |

### Async (native coroutines)

Every async method yields to the event loop at I/O boundaries. In a production
deployment the yield points are where real I/O (remote registries, signing services,
sandbox workers, time-series stores) would be awaited.

All async methods also run `_guard_tenant_id` and `_guard_non_empty_string`
checks on critical arguments before yielding, so callers receive `ValueError`
immediately rather than after the first event-loop yield.

| Method | Description |
|--------|-------------|
| `async_register_plugin(...)` | Non-blocking registration; hook for remote signing/scanning |
| `async_install_plugin(...)` | Non-blocking install; hook for package-registry downloads |
| `async_uninstall_plugin(plugin_id, tenant_id, ...)` | Non-blocking uninstall; hook for sandbox teardown and publisher webhooks |
| `async_plugin_update(tenant_id, plugin_id, new_version, ...)` | Non-blocking version bump; hook for artifact integrity re-check |
| `async_plugin_disable(tenant_id, installation_id, ...)` | Non-destructive deactivation; hook for notification adapters |
| `async_review_permissions(review_id, tenant_id, plugin_id, ...)` | Non-blocking permission review; hook for remote IAM/policy service |
| `async_attach_sandbox_policy(policy_id, tenant_id, plugin_id, ...)` | Non-blocking policy attachment; hook for sandbox worker notification |
| `async_create_release(release_id, tenant_id, plugin_id, ...)` | Non-blocking release with readiness gate; hook for sign/scan pipeline |
| `async_publish_listing(listing_id, tenant_id, plugin_id, ...)` | Non-blocking listing; hook for CDN invalidation and search-index update |
| `async_plugin_permission_check(plugin_id, permission, tenant_id, ...)` | Non-blocking per-scope check; hook for OPA/Cedar evaluation |
| `async_dashboard_summary(tenant_id)` | Non-blocking dashboard counts; hook for read-replica/materialised-view |
| `async_register_plgn_agent(tenant_id, name, runtime, role, scope, ...)` | Non-blocking agent registration; hook for OIDC token verification |
| `async_plugin_marketplace_billing(tenant_id, plugin_id, quantity, unit_price, currency, ...)` | Decimal-accurate billing record; hook for Stripe/M-Pesa adapter |
| `async_hook_fire(tenant_id, event_name, payload, handler_timeout_ms)` | Concurrent fan-out dispatch with per-handler timeout isolation |
| `async_health_check_all(tenant_id, checks, concurrency)` | Semaphore-bounded parallel health checks |
| `async_bulk_install(plugin_ids, tenant_id, concurrency, stop_on_first_error)` | Concurrent multi-plugin install |
| `async_dependency_resolve(plugin_ids, tenant_id)` | Non-blocking dependency resolution |
| `async_sandboxed_execution(plugin_id, method, parameters, timeout_ms)` | Wall-clock-timeout sandbox call via `asyncio.wait_for` |
| `async_plugin_analytics(tenant_id, period)` | Non-blocking analytics; hook for time-series adapters |
| `async_search_marketplace(query, tenant_id, channel, curated_only, limit, offset)` | Paginated search; hook for full-text/vector index |
| `async_audit_query(tenant_id, event_type, actor, since, limit)` | Filtered, sorted audit event retrieval |

---

## Async Usage Examples

### Concurrent event dispatch

```python
import asyncio
from capabilities.common.plgn import PlgnService

svc = PlgnService()
# ... (register plugin, attach hooks) ...

async def on_order_created(order_id: str):
    report = await svc.async_hook_fire(
        tenant_id="acme",
        event_name="order.created",
        payload={"order_id": order_id},
        handler_timeout_ms=500,  # per-handler timeout; others continue on failure
    )
    failed = [d for d in report["dispatched"] if d["status"] != "dispatched"]
    if failed:
        print(f"Partial dispatch failure: {failed}")
```

### Parallel health checks

```python
async def nightly_health_sweep(tenant_id: str):
    reports = await svc.async_health_check_all(tenant_id, concurrency=20)
    unhealthy = [r for r in reports if r["overall"] != "healthy"]
    return unhealthy
```

### Bulk plugin install

```python
async def provision_tenant(tenant_id: str, plugin_ids: list[str]):
    result = await svc.async_bulk_install(
        plugin_ids=plugin_ids,
        tenant_id=tenant_id,
        concurrency=5,
        stop_on_first_error=False,  # collect all errors
    )
    print(f"installed={result['installed']} failed={result['failed']}")
    for o in result["outcomes"]:
        if o["status"] == "failed":
            print(f"  {o['plugin_id']}: {o['error']}")
```

### Sandbox execution with timeout

```python
async def call_plugin(plugin_id: str, tenant_id: str):
    try:
        result = await svc.async_sandboxed_execution(
            plugin_id=plugin_id,
            method="compute_score",
            parameters={"customer_id": "c-42"},
            tenant_id=tenant_id,
            timeout_ms=3000,
        )
        return result["result"]
    except asyncio.TimeoutError:
        return {"error": "sandbox_timeout"}
```

### Paginated marketplace search

```python
async def search(query: str, page: int = 0, page_size: int = 10):
    return await svc.async_search_marketplace(
        query=query,
        tenant_id="acme",
        curated_only=True,
        limit=page_size,
        offset=page * page_size,
    )
```

### Filtered audit query

```python
async def recent_installs(tenant_id: str):
    return await svc.async_audit_query(
        tenant_id=tenant_id,
        event_type="plugin_installed",
        since="2026-01-01T00:00:00+00:00",
        limit=50,
    )
```

---

## Event Hook System

Plugins subscribe to named events and receive payloads at dispatch time.

```python
# Subscribe
svc.plugin_event_hook(
    event_name="user.login",
    plugin_id="audit-logger",
    handler="audit_logger.hooks.on_user_login",
    tenant_id="acme",
    priority=10,  # lower = higher priority
)

# Async dispatch (concurrent, fault-isolated)
await svc.async_hook_fire(
    tenant_id="acme",
    event_name="user.login",
    payload={"user_id": "u-99", "ip": "10.0.0.1"},
    handler_timeout_ms=1000,
)
```

The `async_hook_fire` dispatch:
- Runs all handlers concurrently via `asyncio.gather`.
- Time-boxes each handler independently via `asyncio.wait_for`.
- Captures per-handler errors without aborting others.
- Returns a structured dispatch report: `{event_name, hooks_registered, dispatched[{plugin_id, handler, status, error}], fired_at}`.

---

## Sandbox Policy

Sandbox policies gate which capabilities a plugin may exercise at runtime.

```python
svc.attach_sandbox_policy(
    policy_id="sbx-strict",
    tenant_id="acme",
    plugin_id="my-plugin",
    policy_name="strict",
    network_access="deny",         # deny | allow
    filesystem_access="read_only", # read_only | read_write | deny
    secret_access="deny",          # deny | allow (requires permission review)
    tool_allowlist=["tool_a", "tool_b"],
)
```

`plugin_sandboxed_execution` / `async_sandboxed_execution` refuse to run
if no sandbox policy is attached.

---

## Dependency Resolution

```python
resolution = svc.plugin_dependency_resolution(
    plugin_ids=["plugin-a", "plugin-b"],
    tenant_id="acme",
)
# {
#   "resolved_install_order": [...],
#   "missing_plugins": [...],
#   "conflicts": [...],
#   "resolution_successful": True/False,
# }
```

---

## AI Plugin Agents

Register AI agents that assist with plugin governance (manifest review,
permission analysis, release verification):

```python
svc.register_plgn_agent(
    tenant_id="acme",
    name="Manifest Reviewer",
    runtime="claude_code",     # claude_code | codex | opencode | pi
    role="manifest_reviewer",  # marketplace_reviewer | permission_reviewer | ...
    scope="Review plugin manifests for schema, dependency, and permission compliance",
)
```

---

## Composition

PLGN composes with other APG capabilities via the composition engine:

```apg
use plgn;
```

| Capability | Role |
|---|---|
| `auth` | Publisher, reviewer, installer, administrator identity |
| `secu` | Permission review policy, sensitive-scope enforcement, package scanning |
| `conf` | Tenant install policies, extension configuration baselines |
| `audl` | Durable audit evidence |
| `regy` | Service and plugin discovery |
| `sbox` | Sandbox enforcement at the process level |
| `wflo` | Review and release workflow orchestration |

Batch plugin mutation and release lifecycle events use the `bytewax` stream adapter.

---

## Configuration

All keys are tenant-scoped. Set via the `conf` capability or environment variables
prefixed with `PLGN_`.

---

## Marketplace Billing

Use `async_plugin_marketplace_billing` to record paid plugin installs. All
monetary values use `Decimal` arithmetic — no floating-point rounding errors.

```python
import asyncio
from capabilities.common.plgn import PlgnService

svc = PlgnService()

async def bill_install(tenant_id: str, plugin_id: str):
    record = await svc.async_plugin_marketplace_billing(
        tenant_id=tenant_id,
        plugin_id=plugin_id,
        quantity=5,          # 5 seats
        unit_price="4.99",   # Decimal string — exact
        currency="USD",
        billed_by="billing-service",
    )
    print(f"Billed {record['total_amount']} {record['currency']}")
    # "Billed 24.95 USD"
```

---

## Async Permission Review

```python
async def review(tenant_id: str, plugin_id: str):
    return await svc.async_review_permissions(
        review_id="rev-async-001",
        tenant_id=tenant_id,
        plugin_id=plugin_id,
        reviewer="security-lead",
        approved_scopes=["identity", "network:external"],
        secret_access_allowed=True,
        notes="Reviewed 2026-06-11",
    )
```

---

## Async Sandbox Policy

```python
async def configure_sandbox(tenant_id: str, plugin_id: str):
    return await svc.async_attach_sandbox_policy(
        policy_id="sbx-async-001",
        tenant_id=tenant_id,
        plugin_id=plugin_id,
        policy_name="strict",
        network_access="deny",
        filesystem_access="read_only",
        secret_access="deny",
        tool_allowlist=["score_customer"],
    )
```

---

## Async Dashboard

```python
async def dashboard(tenant_id: str):
    return await svc.async_dashboard_summary(tenant_id)
```

---

## Async Plugin Governance Agent

```python
async def register_agent(tenant_id: str):
    return await svc.async_register_plgn_agent(
        tenant_id=tenant_id,
        name="Manifest Reviewer",
        runtime="claude_code",
        role="manifest_reviewer",
        scope="Review plugin manifests for schema, dependency, and permission compliance",
        contribution_disclosed=True,
    )
```

---

## Input Guards

All async methods call `_guard_tenant_id` and `_guard_non_empty_string` before
the first `await`. Blank or None values raise `ValueError` synchronously, so
callers can use a plain `try/except ValueError` without needing to `await`
first:

```python
try:
    result = await svc.async_plugin_update(
        tenant_id="",        # blank — raises immediately
        plugin_id="p1",
        new_version="2.0.0",
    )
except ValueError as exc:
    print(exc)  # "tenant_id_required"
```

---

## Further Reading

- `service.py` — Business logic and all async methods
- `models.py` — Pydantic/dataclass models
- `plugin_runtime.py` — Deterministic IDs, normalization helpers, release readiness
- `capability_contract.py` — Governance rules, routes, adapters, stream metadata
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 architectural enhancement proposals
