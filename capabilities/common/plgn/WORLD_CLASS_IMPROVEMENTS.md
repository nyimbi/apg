# Plugin/Extension Framework — World-Class Improvements

**Capability**: `plgn` | **Domain**: `common`
**Author**: Nyimbi Odero | **Date**: 2026-06-11

---

### I1. Async-First Architecture with Structured Concurrency

**Category**: Architecture | **Justification**: Every I/O-bound operation (DB reads, registry fetches, sandbox spawning, event dispatch) blocks the event loop when called from an async host; making every public method a native coroutine unlocks all other concurrency improvements and matches FastAPI/Starlette expectations. | **Implementation**: Native `async def` counterparts for every public method; sync wrappers delegating to `asyncio.get_event_loop().run_until_complete`; `asyncio.TaskGroup` (Python 3.11+) for structured concurrency with automatic cancellation propagation on first failure. | **Competitor**: Stripe Python SDK v5 — ships sync and async clients generated from the same spec with zero duplication.

---

### I2. Structured Async Event Bus with Middleware Chain

**Category**: Eventing | **Justification**: Serial `hook_fire` cannot support high-throughput streams; fan-out dispatch with per-handler timeout isolation, back-pressure, and pluggable middleware (tracing, dedup, rate-limiting) is required for production workloads. | **Implementation**: `async_hook_fire` using `asyncio.gather` with `asyncio.wait_for` per handler; `EventBusMiddleware` protocol (async callable wrapping dispatch); `DeadLetterQueue` for failed handlers; structured `DispatchReport` dataclass as return value. | **Competitor**: Temporal.io — per-activity timeout isolation and structured failure capture across distributed workflows.

---

### I3. Cryptographic Plugin Signature Verification Pipeline

**Category**: Security | **Justification**: A boolean `signature_verified` flag set at call-site can be trivially spoofed; a multi-step async pipeline anchored to real key material and revocation checks is the only defensible approach for a multi-tenant extension system. | **Implementation**: `async_verify_signature(plugin_id, artifact_uri, public_key_ref)` — (a) fetch artifact hash from registry adapter, (b) verify ECDSA-P256 or Ed25519, (c) OCSP/CRL revocation check, (d) write immutable `SignatureVerification` audit record; replace `signature_verified: bool` with `trust_level: Literal["trusted","partial","untrusted"]`. | **Competitor**: Sigstore/cosign — same pipeline as an OSS primitive with OIDC-based keyless signing.

---

### I4. Supply-Chain CVE Risk Scoring with Threshold Policies

**Category**: Security | **Justification**: `supply_chain_scan_passed: bool` cannot express partial risk; a scored model with configurable threshold policies allows proportionate governance (warn vs. block vs. quarantine) rather than binary pass/fail. | **Implementation**: `async_supply_chain_scan(plugin_id, dependency_tree)` calling OSV.dev / Grype adapter; composite risk score 0–100 from CVE count x severity weights; `SupplyChainReport` model; tenant-configured score thresholds gate registration/install; `supply_chain_score` emitted to OTEL metrics. | **Competitor**: Snyk — scored dependency scanning with per-dependency remediation paths.

---

### I5. Versioned Plugin Configuration Schema with Migration Ledger

**Category**: Developer Experience | **Justification**: Untyped `metadata` dict forces consumers to parse arbitrary JSON; a registered JSON Schema with versioned migrations enables typed validation, IDE completion, and safe schema evolution without service redeployment. | **Implementation**: `PluginConfigSchema` model (tenant-scoped, append-only `SchemaRevision` log); `async_register_config_schema(plugin_id, json_schema_str, version)` validates JSON Schema Draft 2020-12; `async_validate_plugin_config(plugin_id, config_dict)` runs `jsonschema.validate`; breaking changes require explicit migration annotations. | **Competitor**: Shopify app configuration schema — versioned JSON Schema registration with migration tooling.

---

### I6. Capability-Token Permission Model (PASETO / Macaroon)

**Category**: Security | **Justification**: Flat permission strings cannot express resource-instance scoping, expiry, delegation depth, or offline verification; capability tokens eliminate online permission lookups in the sandbox hot path, reducing p99 latency by one network round-trip. | **Implementation**: `CapabilityToken` dataclass (resource, actions, expiry, delegation_depth, tenant_id, issuer_sig); `async_grant_capability_token` issues PASETO v4 local tokens; `async_revoke_capability_token` appends to tenant-scoped revocation set; sandbox workers verify offline; `async_list_capability_tokens` returns active grants per plugin. | **Competitor**: Auth0 Fine-Grained Authorization (FGA) / Google Zanzibar — resource-scoped token model at scale.

---

### I7. Hot-Reload with Zero-Downtime Plugin Swap

**Category**: Operational Excellence | **Justification**: No current path supports swapping a plugin version at runtime; requiring reinstall and re-enable causes service interruption in production tenants with continuous workloads. | **Implementation**: `async_hot_reload_plugin(plugin_id, new_artifact_uri, tenant_id, migration_fn)` — (a) register new version in shadow slot, (b) drain in-flight sandbox executions via semaphore quiesce, (c) atomic entry-point swap, (d) await migration_fn, (e) resume traffic; automatic rollback on first execution failure within configurable `grace_window_ms`; emit `PluginHotReloadEvent` to Bytewax stream. | **Competitor**: Erlang/OTP hot code loading — the 35-year gold standard for zero-downtime updates.

---

### I8. Hierarchical Sandbox Profiles with Resource Quotas

**Category**: Security | **Justification**: `SandboxPolicy` covers only network/filesystem/secret/tools; CPU/memory quotas, syscall allowlists, and inter-plugin call rules are required to prevent resource exhaustion and lateral movement in multi-tenant environments. | **Implementation**: Extend `SandboxPolicy` with `cpu_millicores: int`, `memory_mb: int`, `syscall_allowlist: tuple[str,...]`, `inter_plugin_calls: tuple[str,...]`; `async_validate_sandbox_execution_context(plugin_id, method, params, tenant_id)` checks all dimensions before dispatch; `async_escalate_sandbox_permission(plugin_id, capability, duration_s, approver)` for time-limited escalation with automatic expiry. | **Competitor**: AWS Lambda execution environment — per-invocation CPU/memory/network boundaries.

---

### I9. Marketplace Recommendation Engine with Semantic Search

**Category**: Marketplace | **Justification**: Substring title search is inadequate for discovery in a marketplace with hundreds of plugins; semantic similarity and collaborative-filter recommendations cut time-to-value for tenant onboarding. | **Implementation**: `async_recommend_plugins(tenant_id, context_tags, installed_ids, limit)` scoring by install-count rank (BM25), health ratio, capability compatibility, and cosine similarity between context_tags and plugin description embeddings (pgvector/Meilisearch adapter); `RecommendationResult` with score breakdown; `async_index_plugin(plugin_id)` keeps the index fresh on every registration/update. | **Competitor**: VS Code Marketplace recommendation engine — hybrid popularity + semantic + installed-extension compatibility scoring.

---

### I10. PubGrub Dependency Solver with Explanation Trees

**Category**: Developer Experience | **Justification**: The current DFS conflict detector reports conflicts but cannot resolve them; a constraint solver finds the minimal compatible install set and surfaces human-readable explanation trees for unsatisfiable requests. | **Implementation**: Encode the dependency graph as PubGrub constraints via `resolvelib`; `async_solve_dependencies(requirements: dict[str,str], tenant_id)` returning `SolveResult{install_order, locked_versions, explanation_tree}`; cached by constraint hash; `async_explain_conflict(plugin_id_a, plugin_id_b, tenant_id)` for UI display. | **Competitor**: Cargo's dependency resolver — PubGrub implementation widely regarded as best-in-class.

---

### I11. Declarative Multi-Stage Release Pipeline with Human Gates

**Category**: Release Engineering | **Justification**: A single synchronous `create_release` call cannot model real release governance: sign → scan → review → approve → canary → promote; each stage must be idempotent and resumable to avoid lost work when a human reviewer is slow. | **Implementation**: `ReleasePipeline` model with stages `[draft, sign, scan, review, approve, publish, notify]`; each stage stores state, actor, timestamp, remediation_hints; `async_advance_release_stage(release_id, actor, evidence)` transitions one stage; `async_await_release_approval(release_id, timeout_s)` with webhook callback; failed stages emit `ReleaseStageFailed` with actionable hints. | **Competitor**: GitHub Actions deployment environment protection rules — multi-stage gating with human approval.

---

### I12. Plugin Telemetry with W3C Trace Propagation

**Category**: Observability | **Justification**: Local `execution_time_ms` is not useful for diagnosing cross-service latency; W3C `traceparent` propagation into sandbox workers enables distributed traces across service boundaries with p99 SLOs and error-rate alerts. | **Implementation**: `async_traced_execution(plugin_id, method, parameters, trace_context, tenant_id)` injecting `traceparent`/`tracestate` into sandbox adapter; span data to OTEL adapter (latency, error, resource peaks); `async_plugin_slo_report(plugin_id, tenant_id, window_s)` surfacing `SloReport{p50,p95,p99,error_rate}`; alert rules hooking into the `alerts` capability. | **Competitor**: Datadog APM — automatic trace context propagation into every extension invocation.

---

### I13. Cross-Version Compatibility Matrix with SemVer Enforcement

**Category**: Reliability | **Justification**: Plugins declare APG capability version requirements but enforcement is absent; a silent incompatibility breaks tenants after an APG upgrade — proactive matrix checks with install blocking on hard incompatibilities prevent surprise outages. | **Implementation**: `CompatibilityMatrix` model per tenant: `plugin_id → {requires, tested_against}`; `async_check_compatibility(plugin_id, host_versions, tenant_id)` evaluates PEP 440/SemVer constraints via `packaging.version`; returns `CompatibilityResult{compatible, breaking_changes, warnings}`; blocks install when `compatible=False`; `async_get_compatibility_matrix(tenant_id)` as queryable artifact. | **Competitor**: Gradle dependency constraints — host-version vs. plugin-requirement compatibility enforcement.

---

### I14. Federated Plugin Registry with Trust Tiers

**Category**: Ecosystem | **Justification**: A single in-process dict cannot support multi-deployment APG ecosystems; federated registry with trust tiers allows organisations to share curated plugin catalogs while maintaining per-tenant governance control. | **Implementation**: `RemoteRegistry` model (url, auth_token, trust_tier: `Literal["trusted","verified","community"]`); `async_sync_remote_registry(registry_url, auth_token, sync_policy, tenant_id)` pulls manifests, reconciles with local store, records `RegistrySyncReport`; trust_tier gates automatic vs. manual approval; `async_publish_to_registry(plugin_id, registry_url, tenant_id)` pushes signed manifest to remote APG registry. | **Competitor**: OCI Distribution Spec / ORAS — federated artifact registry with configurable trust policies.

---

### I15. Policy-as-Code Plugin Governance (OPA / Cedar)

**Category**: Governance | **Justification**: Hard-coded rules in `capability_contract.py` require a service redeployment to change governance policy; policy-as-code decouples governance evolution from the deployment cycle and enables per-tenant customisation with a full diff/audit trail for compliance. | **Implementation**: `PolicyDocument` model (tenant_id, policy_id, engine: `Literal["opa","cedar"]`, content, version, created_by); `async_evaluate_policy(policy_id, context, tenant_id)` delegates to configured engine adapter; `async_update_policy(policy_id, new_content, actor, tenant_id)` appends `PolicyRevision` with diff; policies scoped to plugin category, channel, or individual plugin_id; fallback to built-in rules when no document exists. | **Competitor**: AWS Cedar and Open Policy Agent — runtime-evaluable policy-as-code with versioned tenant documents.
