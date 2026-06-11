# Plugin/Extension Framework — World-Class Improvements

**Capability**: `plgn` | **Domain**: `common`
**Author**: Nyimbi Odero | **Date**: 2026-06-11

---

## 1. Async-First Architecture

The current service is entirely synchronous. Every I/O-bound operation (DB reads,
external registry fetches, sandbox spawning, event dispatch) blocks the event loop
when called from an async host. Every public method should have a native `async`
counterpart, with the sync variants kept only as thin wrappers for backward
compatibility. This is the single highest-leverage change; it unlocks all other
concurrency improvements.

---

## 2. Structured Async Event Bus

`hook_fire` iterates hooks serially. A proper event bus should fan-out to all
handlers concurrently via `asyncio.gather`, respect per-handler timeouts, capture
individual handler errors without failing the entire dispatch, and emit a
structured dispatch report. Add `async_hook_fire` and an `EventBusMiddleware`
protocol so host apps can inject tracing, deduplication, or rate-limiting without
modifying the core.

---

## 3. Plugin Signature Verification Pipeline

`signature_verified` is currently a boolean flag set at registration. Replace it
with an async multi-step pipeline: `async_verify_signature(plugin_id, artifact_uri,
public_key_ref)` that (a) fetches the artifact hash from the registry adapter,
(b) verifies against the stored public key, (c) checks certificate revocation, and
(d) writes an immutable `SignatureVerification` audit record. Trust level (`trusted`,
`partial`, `untrusted`) replaces the boolean.

---

## 4. Supply-Chain Risk Scoring

`supply_chain_scan_passed` is also a boolean. Replace with an async
`async_supply_chain_scan(plugin_id, dependency_tree)` that calls a configurable
vulnerability-feed adapter (OSV, Snyk, Grype), scores each dependency (CVE count,
max severity, license risk), aggregates a composite risk score 0–100, and surfaces
a `SupplyChainReport` model. Threshold policies gate registration vs. warning vs.
block outcomes.

---

## 5. Versioned Plugin Configuration Schema

Plugins currently carry untyped `metadata`. Introduce a `PluginConfigSchema`
model — a tenant-scoped JSON Schema document that plugins register alongside their
manifest. `async_validate_plugin_config(plugin_id, config_dict)` validates
caller-supplied configuration against the schema before activation, surfacing typed
validation errors. Schema migrations are versioned and stored in an append-only
`SchemaRevision` log.

---

## 6. Capability-Based Permission Model

The flat permission string list (e.g. `"identity"`, `"network:external"`) should
evolve into a capability-token model: hierarchical, composable tokens with
resource-instance scoping (`"data:read:tenant:t1:table:customers"`), expiry,
and delegation depth limits. `async_grant_capability_token` and
`async_revoke_capability_token` replace the permission-review blob. Tokens are
cryptographically signed macaroons or PASETO-backed structures so they can be
verified offline by sandbox workers.

---

## 7. Hot-Reload / Live Plugin Update

No path currently supports hot-swapping a plugin at runtime. Add
`async_hot_reload_plugin(plugin_id, new_artifact_uri, migration_fn)` that
(a) validates the new version, (b) quiesces in-flight sandbox executions,
(c) swaps the entry-point reference, (d) runs schema migrations, and (e) resumes
traffic — all with zero plugin-consumer downtime. Rollback is automatic on the
first execution failure within a configurable grace window.

---

## 8. Hierarchical Sandbox Profiles

`SandboxPolicy` covers network, filesystem, secret, and tool allowlist. Extend with:
- CPU/memory quotas enforced by the sandbox runtime adapter.
- Syscall allowlist (seccomp-style) for native plugins.
- Inter-plugin communication rules (which other plugins a sandboxed plugin may call).
- Time-limited capability escalation with automatic expiry and audit.

`async_validate_sandbox_execution_context` checks all four dimensions before
dispatching a sandboxed call.

---

## 9. Marketplace Search and Recommendation Engine

`plugin_marketplace_listing` returns a flat list with simple channel/curation
filters. Replace with `async_search_marketplace(query, filters, pagination)` backed
by a pluggable index adapter (Meilisearch, Typesense, or pgvector for semantic
search). Add an `async_recommend_plugins(tenant_id, context_tags)` endpoint that
scores plugins by install count, health ratio, compatibility with the tenant's
installed capability set, and semantic similarity to the supplied tags.

---

## 10. Dependency Conflict Resolution with SAT Solver

The current dependency resolver is a depth-first traversal that detects conflicts
but cannot resolve them. Replace with `async_solve_dependencies(constraints)` that
encodes the dependency graph as a SAT or PubGrub-style constraint problem, returns
the minimal compatible install set, and surfaces an explanation tree when the
problem is unsatisfiable. This is the algorithm used by Cargo, Pub, and modern pip.

---

## 11. Declarative Release Pipeline

`create_release` is a single synchronous call. Replace with a multi-stage async
pipeline: `Draft → Sign → Scan → Review → Approve → Publish → Notify`. Each stage
is idempotent, resumable, and emits a `ReleaseStageEvent`. Human-approval gates use
async long-polling or webhook callbacks. Failed stages produce actionable remediation
hints rather than raw error strings.

---

## 12. Plugin Telemetry and Distributed Tracing

`plugin_sandboxed_execution` records execution time and memory locally. Add
`async_traced_execution(plugin_id, method, parameters, trace_context)` that
propagates W3C `traceparent` headers into the sandbox adapter, records span data
(latency percentiles, error rates, resource peaks) to an OTEL-compatible adapter,
and surfaces per-plugin SLO dashboards. Alert rules on p99 latency and error-rate
breaches hook back into the `alerts` capability.

---

## 13. Plugin Compatibility Matrix

Plugins declare compatibility with APG capability versions but there is no
enforcement. Add `async_check_compatibility(plugin_id, capability_versions_dict)`
that evaluates the plugin's declared `requires` ranges against the host's installed
capability versions (SemVer), flags breaking changes, and blocks installation when
hard incompatibilities are detected. The compatibility matrix is persisted and
queryable as a tenant-scoped artifact.

---

## 14. Federated Plugin Registry

All plugins live in one in-process dict. Add an `async_sync_remote_registry(
registry_url, auth_token, sync_policy)` method that pulls manifests from a remote
APG registry, reconciles additions/updates/removals with the local store, and
records a `RegistrySyncReport`. Trust tiers per remote registry control automatic
vs. manual approval of incoming manifests. This enables an ecosystem marketplace
model across APG deployments.

---

## 15. Policy-as-Code for Plugin Governance

`evaluate_capability_rules` is a hard-coded rule engine in `capability_contract.py`.
Replace with `async_evaluate_policy(policy_id, context)` backed by a
policy-as-code adapter (Open Policy Agent, Cedar, or Rego). Policies are stored as
versioned tenant-scoped documents, editable at runtime, with a diff/audit trail.
Policy bundles can be scoped to plugin categories, channels, or individual plugins.
This eliminates the need to redeploy the service to change governance rules.
