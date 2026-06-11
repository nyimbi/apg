# PLGN Plugin/Extension Framework Capability

PLGN gives APG applications a tenant-scoped extension system: plugin manifests,
curated marketplace listings, permission review, sandbox policy, release
gates, installation, activation, plugin-governance agents, UI metadata, theme
tokens, audit evidence, and Bytewax-backed lifecycle events.

The package stays dependency-light. Production plugin registries, package
stores, signing services, security scanners, sandbox runtimes, identity
providers, audit sinks, workflow engines, and Bytewax workers are represented
as APG adapters in the executable contract and are bound by the host
application.

## What It Provides

- Plugin registry with tenant, owner, publisher, version, release channel,
  permissions, dependencies, signature, manifest, dependency, scan, and
  external-review posture.
- Permission review records for approved scopes, denied scopes, sensitive
  permissions, and secret-access decisions.
- Sandbox policy records for network, filesystem, secret, and tool allowlist
  constraints.
- Marketplace listing workflow with curation, publisher verification, and
  tenant install policy.
- Release, installation, and enablement lifecycle guarded by policy evidence.
- First-class AI plugin agents with runtime, role, scope, registration, and
  contribution-disclosure guardrails.
- UI route, API, view-model, theme, semantic-model, package-manifest, and
  release-report evidence.

## Main Files

- `SPECIFICATION.md` defines the normative capability behavior.
- `PLAN.md` records the implementation packet plan.
- `capability_contract.py` is the executable source of configuration, rules,
  routes, theme, adapters, provides/requires, and Bytewax stream metadata.
- `models.py` defines tenant-scoped manifests, reviews, sandbox policies,
  listings, releases, installations, audit events, and agents.
- `plugin_runtime.py` contains deterministic IDs, release-channel and install
  policy normalization, scope helpers, and release-readiness checks.
- `service.py` implements the runtime facade.
- `api.py` exposes package-safe helper functions.
- `views.py` exposes UI view models.
- `test_capability_contract.py` proves lifecycle behavior and generated
  evidence.

## Basic Usage

```python
from capabilities.common.plgn import PlgnService

service = PlgnService()
service.register_plugin(
    plugin_id="risk-scorer",
    tenant_id="tenant-demo",
    name="Risk scorer extension",
    owner="extension-owner",
    version="1.2.0",
    publisher="Datacraft",
    release_channel="stable",
    permissions=["identity", "network:external"],
    dependencies=["auth", "secu"],
    external_plugin=True,
    external_review_recorded=True,
    permission_review_recorded=True,
)
service.review_permissions(
    review_id="review-risk",
    tenant_id="tenant-demo",
    plugin_id="risk-scorer",
    reviewer="security-reviewer",
    approved_scopes=["identity", "network:external"],
    secret_access_allowed=True,
)
service.attach_sandbox_policy(
    policy_id="sandbox-risk",
    tenant_id="tenant-demo",
    plugin_id="risk-scorer",
    policy_name="restricted-tools",
    tool_allowlist=["score_customer"],
)
```

## AI Plugin Agents

Register AI agents before they assist with plugin governance:

```python
agent = service.register_plgn_agent(
    tenant_id="tenant-demo",
    name="Manifest reviewer",
    runtime="codex",
    role="manifest_reviewer",
    scope="Review plugin manifest schema, dependencies, permissions, and release evidence",
)
```

Supported runtimes are `codex`, `claude_code`, `opencode`, and `pi`.
Supported roles cover marketplace, manifest, permission, sandbox, release, and
compatibility review.

## Composition

PLGN composes with:

- `auth` for publisher, reviewer, installer, and administrator identity.
- `secu` for permission review, sensitive-scope policy, and package scanning.
- `conf` for tenant install policies and extension configuration baselines.
- `audl` for durable audit evidence.
- `regy` for service and plugin discovery publication.
- `sbox` for sandbox enforcement.
- `wflo` for review and release workflows.

Batch plugin mutation and plugin release lifecycle events must use the
`bytewax` event-stream adapter.

## Async API

Every public method has a native `async` counterpart for use in async hosts.

```python
import asyncio
from capabilities.common.plgn import PlgnService

svc = PlgnService()

async def main():
    # Non-blocking registration — hook point for remote signing/scanning
    plugin = await svc.async_register_plugin(
        name="analytics-ext", version="2.0.0", author="alice",
        entry_point="analytics.main", permissions=[], tenant_id="t1",
    )

    # Concurrent fan-out event dispatch with per-handler timeout isolation
    await svc.async_hook_fire(
        tenant_id="t1", event_name="data.ingested",
        payload={"rows": 1000}, handler_timeout_ms=500,
    )

    # Parallel health checks across all plugins in the tenant
    reports = await svc.async_health_check_all("t1", concurrency=10)

    # Install a set of plugins concurrently
    result = await svc.async_bulk_install(
        plugin_ids=["analytics-ext", "report-ext"],
        tenant_id="t1", concurrency=5,
    )

    # Paginated marketplace search (hook for full-text/vector adapter)
    hits = await svc.async_search_marketplace("analytics", tenant_id="t1", limit=10)

    # Filtered audit query, newest-first
    events = await svc.async_audit_query("t1", event_type="plugin_registered")

asyncio.run(main())
```

| Async method | Purpose |
|---|---|
| `async_register_plugin` | Non-blocking registration; I/O hook for signing/scanning |
| `async_install_plugin` | Non-blocking install; hook for package-registry downloads |
| `async_hook_fire` | Concurrent fan-out dispatch with per-handler timeout |
| `async_health_check_all` | Semaphore-bounded parallel health checks |
| `async_bulk_install` | Concurrent multi-plugin install with stop-on-first-error |
| `async_dependency_resolve` | Non-blocking dependency resolution |
| `async_sandboxed_execution` | Sandbox call with `asyncio.wait_for` wall-clock timeout |
| `async_plugin_analytics` | Non-blocking analytics; hook for time-series adapters |
| `async_search_marketplace` | Paginated search; hook for full-text/vector index |
| `async_audit_query` | Filtered, sorted audit event retrieval |

## Verification

Focused verification for this packet:

```bash
./.venv/bin/python -m py_compile capabilities/common/plgn/__init__.py capabilities/common/plgn/capability_contract.py capabilities/common/plgn/models.py capabilities/common/plgn/plugin_runtime.py capabilities/common/plgn/service.py capabilities/common/plgn/api.py capabilities/common/plgn/views.py capabilities/common/plgn/app.py capabilities/common/plgn/test_capability_contract.py
./.venv/bin/pytest -q capabilities/common/plgn/test_capability_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/plgn --json
./.venv/bin/apg capabilities publish-plan capabilities/common/plgn --json
```

Live package registries, signing providers, security scanners, remote sandbox
runtimes, rendered UI, and Bytewax workers are integration concerns outside the
package proof.
