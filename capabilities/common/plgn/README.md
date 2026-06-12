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
  constraints; extended with CPU/memory quotas and syscall allowlists.
- Marketplace listing workflow with curation, publisher verification,
  semantic search, and tenant install policy.
- Release, installation, and enablement lifecycle guarded by policy evidence.
- First-class AI plugin agents with runtime, role, scope, registration, and
  contribution-disclosure guardrails.
- Async-first public API: every method has a native `async` counterpart.
- Concurrent fan-out event dispatch with per-handler timeout isolation.
- Semaphore-bounded parallel health checks and bulk installs.
- Dependency resolution with conflict detection (PubGrub solver hook).
- Cryptographic signature verification pipeline with trust-level output.
- Supply-chain CVE risk scoring with configurable threshold policies.
- Capability-token (PASETO/Macaroon) permission model for offline sandbox checks.
- Hot-reload with zero-downtime plugin swap and automatic rollback.
- Policy-as-code governance via OPA/Cedar with per-tenant versioned documents.
- Federated registry with trust tiers for multi-deployment ecosystems.
- W3C trace propagation for distributed plugin execution telemetry.
- SemVer compatibility matrix with install-blocking on hard incompatibilities.
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
- `service.py` implements the runtime facade (`PluginExtensionService` / `PlgnService`).
- `api.py` exposes package-safe helper functions.
- `views.py` exposes UI view models.
- `test_capability_contract.py` proves lifecycle behavior and generated evidence.

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

Supported runtimes: `codex`, `claude_code`, `opencode`, `pi`.  
Supported roles: marketplace, manifest, permission, sandbox, release, and compatibility review.

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
| `async_uninstall_plugin` | Non-blocking uninstall; hook for sandbox teardown and publisher webhooks |
| `async_plugin_update` | Non-blocking version bump; hook for artifact integrity re-check |
| `async_plugin_disable` | Non-blocking deactivation; hook for notification adapters |
| `async_review_permissions` | Non-blocking permission review; hook for remote IAM/policy service |
| `async_attach_sandbox_policy` | Non-blocking policy attachment; hook for sandbox worker notification |
| `async_create_release` | Non-blocking release creation; hook for signing/scan pipeline |
| `async_publish_listing` | Non-blocking listing publication; hook for CDN and search-index update |
| `async_plugin_permission_check` | Non-blocking per-scope check; hook for OPA/Cedar evaluation |
| `async_dashboard_summary` | Non-blocking dashboard aggregation; hook for read-replica query |
| `async_register_plgn_agent` | Non-blocking agent registration; hook for OIDC token verification |
| `async_plugin_marketplace_billing` | Decimal-accurate billing record; hook for Stripe/M-Pesa adapter |
| `async_hook_fire` | Concurrent fan-out dispatch with per-handler timeout |
| `async_health_check_all` | Semaphore-bounded parallel health checks |
| `async_bulk_install` | Concurrent multi-plugin install with stop-on-first-error |
| `async_dependency_resolve` | Non-blocking dependency resolution |
| `async_sandboxed_execution` | Sandbox call with `asyncio.wait_for` wall-clock timeout |
| `async_plugin_analytics` | Non-blocking analytics; hook for time-series adapters |
| `async_search_marketplace` | Paginated search; hook for full-text/vector index |
| `async_audit_query` | Filtered, sorted audit event retrieval |

## New Methods

### Concurrent fan-out event dispatch

```python
# Register hooks first
svc.hook_register(
    tenant_id="t1", event_name="order.created",
    plugin_id="notifier-ext", handler="notifier.on_order_created", priority=10,
)

# Fire — all handlers run concurrently; each is independently time-boxed
report = await svc.async_hook_fire(
    tenant_id="t1", event_name="order.created",
    payload={"order_id": "ord-123"}, handler_timeout_ms=800,
)
# report["dispatched"] lists per-handler status: "dispatched" | "timeout" | "error"
```

### Semaphore-bounded bulk install

```python
result = await svc.async_bulk_install(
    plugin_ids=["ext-a", "ext-b", "ext-c"],
    tenant_id="t1",
    concurrency=3,
    stop_on_first_error=True,   # cancels remaining on first failure
)
# result keys: installed, failed, cancelled, outcomes
```

### Sandboxed execution with wall-clock timeout

```python
try:
    exec_record = await svc.async_sandboxed_execution(
        plugin_id="risk-scorer", method="score",
        parameters={"customer_id": "cust-42"},
        tenant_id="t1", timeout_ms=3000,
    )
except asyncio.TimeoutError:
    # sandbox worker did not respond within 3 s
    ...
```

### Audit query with filtering

```python
# Retrieve the last 50 install events for a tenant, newest-first
events = await svc.async_audit_query(
    tenant_id="t1",
    event_type="plugin_installed",
    since="2026-01-01T00:00:00+00:00",
    limit=50,
)
```

### Marketplace billing (Decimal-accurate)

```python
bill = await svc.async_plugin_marketplace_billing(
    tenant_id="t1", plugin_id="risk-scorer",
    quantity=5, unit_price="4.99", currency="USD",
    billed_by="billing-service",
)
# bill["total_amount"] == "24.95" — no floating-point drift
```

## World-Class Enhancements (v2.0)

All 15 improvements are designed around production requirements for a
multi-tenant extension system. Each has a corresponding hook point in the
async API.

| # | Improvement | Category | Summary |
|---|---|---|---|
| I1 | Async-First Architecture | Architecture | Native `async def` for every public method; `asyncio.TaskGroup` for structured concurrency with automatic cancellation propagation. |
| I2 | Structured Async Event Bus | Eventing | `async_hook_fire` with `asyncio.gather` + per-handler `asyncio.wait_for`; `EventBusMiddleware` protocol; `DeadLetterQueue`; structured `DispatchReport`. |
| I3 | Cryptographic Signature Pipeline | Security | Multi-step async pipeline: fetch artifact hash, verify ECDSA-P256/Ed25519, OCSP/CRL revocation, write immutable audit record; `trust_level` replaces boolean flag. |
| I4 | Supply-Chain CVE Risk Scoring | Security | `async_supply_chain_scan` calling OSV.dev/Grype; composite score 0–100; tenant-configurable threshold policies gate registration (warn/block/quarantine). |
| I5 | Versioned Config Schema & Migration | Developer Experience | `PluginConfigSchema` with append-only `SchemaRevision` log; JSON Schema Draft 2020-12 validation; breaking changes require explicit migration annotations. |
| I6 | Capability-Token Permission Model | Security | PASETO v4 local tokens with resource, actions, expiry, delegation depth; offline sandbox verification; `async_grant/revoke/list_capability_tokens`. |
| I7 | Hot-Reload Zero-Downtime Swap | Operational Excellence | `async_hot_reload_plugin`: shadow slot, semaphore quiesce, atomic entry-point swap, migration fn, automatic rollback on first failure within `grace_window_ms`. |
| I8 | Hierarchical Sandbox Profiles | Security | Extended `SandboxPolicy` with `cpu_millicores`, `memory_mb`, `syscall_allowlist`, `inter_plugin_calls`; time-limited permission escalation with automatic expiry. |
| I9 | Marketplace Semantic Search | Marketplace | `async_recommend_plugins` scoring by BM25 popularity, health ratio, capability compatibility, and cosine similarity via pgvector/Meilisearch adapter. |
| I10 | PubGrub Dependency Solver | Developer Experience | `async_solve_dependencies` via `resolvelib`; returns `SolveResult{install_order, locked_versions, explanation_tree}`; cached by constraint hash. |
| I11 | Multi-Stage Release Pipeline | Release Engineering | `ReleasePipeline` stages: draft→sign→scan→review→approve→publish→notify; each idempotent and resumable; `async_advance_release_stage` with human-gate webhooks. |
| I12 | Plugin Telemetry W3C Trace | Observability | `async_traced_execution` propagates `traceparent`/`tracestate` into sandbox workers; `async_plugin_slo_report` returns `{p50,p95,p99,error_rate}`; OTEL adapter. |
| I13 | Cross-Version Compatibility Matrix | Reliability | `CompatibilityMatrix` per tenant; `async_check_compatibility` evaluates PEP 440/SemVer constraints; blocks install on `compatible=False`; queryable artifact. |
| I14 | Federated Plugin Registry | Ecosystem | `RemoteRegistry` with trust tiers `trusted/verified/community`; `async_sync_remote_registry` pulls/reconciles manifests; `async_publish_to_registry` pushes signed manifests. |
| I15 | Policy-as-Code Governance | Governance | `PolicyDocument` model (OPA/Cedar engine); `async_evaluate_policy` delegates to configured engine; `async_update_policy` appends `PolicyRevision` with diff and full audit trail. |

## Verification

```bash
./.venv/bin/python -m py_compile \
    capabilities/common/plgn/__init__.py \
    capabilities/common/plgn/capability_contract.py \
    capabilities/common/plgn/models.py \
    capabilities/common/plgn/plugin_runtime.py \
    capabilities/common/plgn/service.py \
    capabilities/common/plgn/api.py \
    capabilities/common/plgn/views.py \
    capabilities/common/plgn/app.py \
    capabilities/common/plgn/test_capability_contract.py

./.venv/bin/pytest -q capabilities/common/plgn/test_capability_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/plgn --json
./.venv/bin/apg capabilities publish-plan capabilities/common/plgn --json
```

Live package registries, signing providers, security scanners, remote sandbox
runtimes, rendered UI, and Bytewax workers are integration concerns outside the
package proof.
