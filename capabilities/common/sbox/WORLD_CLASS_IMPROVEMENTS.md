# SBOX World-Class Improvements

**Capability**: Sandbox/Testing Environment (`sbox`)
**Author**: Nyimbi Odero — Datacraft © 2025

---

## 1. True Async-Native Service Layer

**Problem**: All methods are synchronous. Every I/O-adjacent operation (dataset generation, event delivery, cost recording) blocks the caller.

**Improvement**: Convert the entire service to `async def` methods backed by `asyncio.Lock`-protected in-memory stores. Hot paths like `simulate_event`, `load_test_data`, and `start_run` benefit immediately from cooperative multitasking. Downstream adapters (Bytewax, PostgreSQL, secret vaults) are naturally async; a sync façade wastes that.

**Impact**: 10–100× throughput for concurrent test orchestration. Enables `await service.start_run(...)` composition patterns without thread-pool bridging.

---

## 2. WASM Isolation Backend via Wasmtime

**Problem**: "Isolation" is policy metadata stored in a dict. No code actually runs isolated.

**Improvement**: Integrate `wasmtime-py` as a first-class execution backend. Add `WasmExecutionContext` that compiles user-supplied Python (via Wasm-compiled CPython or Extism) or raw WASM modules into a per-sandbox linear memory region. Provide `execute_wasm(sandbox_id, module_bytes, entrypoint, args)` and `execute_python_isolated(sandbox_id, code, globals_)`.

**Impact**: Actual memory isolation between tenants, CPU accounting per sandbox, and reproducible deterministic execution — not just governance claims.

---

## 3. Snapshot/Restore with Copy-on-Write Semantics

**Problem**: `environment_snapshot` captures a dict; `reset_sandbox` deletes data. There is no true state restoration.

**Improvement**: Implement a COW snapshot tree: every mutating operation creates a delta record. `snapshot_restore(sandbox_id, snap_id)` replays deltas in reverse to return the sandbox, mock catalog, and test data to the exact captured state. Use a `_snapshots: dict[str, SnapshotNode]` tree with parent pointers.

**Impact**: Test replay, bisection debugging, and flaky-test root-cause analysis become deterministic.

---

## 4. Resource Quota Enforcement

**Problem**: A tenant can create unbounded sandboxes, runs, and mock services. There is no admission control.

**Improvement**: Add `TenantQuota` model with configurable limits: `max_sandboxes`, `max_active_runs`, `max_mock_services`, `max_ttl_hours`, `max_concurrent_events`. Enforce at every mutating entry point before persistence. Expose `quota_status(tenant_id)` and `update_quota(tenant_id, ...)` admin methods.

**Impact**: Prevents runaway test infrastructure, enables fair multi-tenant scheduling, and grounds cost tracking in real constraints.

---

## 5. Structured Test Scenario DSL

**Problem**: Scenarios are auto-registered with empty step lists. The "test scenario" concept has no execution semantics.

**Improvement**: Define a `ScenarioStep` dataclass with `action: str`, `target: str`, `params: dict`, `assertion: dict | None`, and `on_failure: str` (continue/abort/retry). Add `define_scenario(scenario_id, steps)`, `validate_scenario(scenario_id)` (static analysis), and `execute_scenario(sandbox_id, scenario_id)` that processes steps sequentially with per-step assertion evaluation and rollback on abort.

**Impact**: Scenarios become executable specifications, not just labels. Test pipelines gain structure comparable to Playwright's test fixtures.

---

## 6. Pluggable Chaos Fault Scheduler

**Problem**: `chaos_inject` records a dict and fires an event but doesn't actually modify sandbox behavior over time.

**Improvement**: Introduce a `ChaosFault` store with `active_faults: dict[str, list[ChaosFault]]`. Faults carry an `expires_at` timestamp. Add `chaos_cancel(fault_id)`, `list_active_faults(sandbox_id)`, and a `_chaos_effective_latency(sandbox_id, service_name)` internal that aggregates active faults. Mock service calls consult this when computing response latency and error rates.

**Impact**: Chaos becomes behavioral, not cosmetic. Load simulation and mock service behavior reflect injected faults, enabling true resilience testing.

---

## 7. Dependency-Aware Sandbox Cloning

**Problem**: Each sandbox is fully independent. Reproducing a production-equivalent test environment requires rebuilding from scratch.

**Improvement**: Add `clone_sandbox(source_sandbox_id, new_name, tenant_id, deep=True)`. Deep clone copies the isolation profile, template reference, all loaded datasets, all registered mocks, and active chaos faults. Shallow clone shares the profile and template by reference. Track `cloned_from` lineage on `SandboxEnvironment`.

**Impact**: Dramatically reduces environment setup time for regression suites. Enables fork-based parallel test variants from a known-good baseline.

---

## 8. Real-Time Event Bus with Subscription API

**Problem**: Simulated events are appended to a list. Nothing consumes them.

**Improvement**: Add an in-process async event bus (`asyncio.Queue` per subscriber). Expose `subscribe_events(sandbox_id, event_types, handler)` returning a subscription token, `unsubscribe(token)`, and `drain_events(sandbox_id)`. `simulate_event` broadcasts to all matching subscribers with backpressure handling.

**Impact**: Test code can `await` event delivery instead of polling. Enables timeout-based integration test assertions: "assert payment.processed arrives within 500ms."

---

## 9. Execution Tracing and Flamegraph Data

**Problem**: `benchmark_run` returns synthetic latency numbers. There is no trace of what actually ran.

**Improvement**: Add `ExecutionTrace` with span records (operation, start_ns, end_ns, parent_span_id). Expose `trace_start(sandbox_id, operation)` returning a context manager that records child spans. `trace_export(sandbox_id, run_id)` emits OpenTelemetry-compatible JSON. `flamegraph_data(sandbox_id, run_id)` returns a folded-stack format suitable for speedscope or d3-flame-graph.

**Impact**: Developers gain real performance visibility into sandbox operations. Identifies where test setup time is actually spent.

---

## 10. Policy-as-Code Rule Engine

**Problem**: `evaluate_capability_rules` is an opaque contract function. Rules cannot be inspected, extended, or tested in isolation.

**Improvement**: Expose `list_policy_rules(tenant_id)`, `add_policy_rule(rule)`, `disable_policy_rule(rule_id)`, and `simulate_policy(context)` (dry-run evaluation). Rules are Pydantic models with `condition: dict`, `action: str` (allow/deny/require_review), and `reason: str`. The engine becomes a first-class service component, testable and auditable.

**Impact**: Operators can add custom governance rules (e.g. "deny sandbox with TTL > 72h unless CFO-approved") without code changes. Policy drift becomes detectable.

---

## 11. Cross-Tenant Sandbox Federation

**Problem**: Sandbox namespacing is purely by `tenant_id`. There is no mechanism for shared infrastructure or cross-tenant test composition.

**Improvement**: Add `federate_sandbox(source_sandbox_id, source_tenant_id, target_tenant_id, shared_permissions)`. Creates a read-only or read-write federation record. `start_run` on a federated sandbox carries the originating tenant's risk score as a floor. Audit events are written to both tenants.

**Impact**: Platform teams can create shared reference sandboxes that product teams use without forking. Reduces infrastructure duplication while maintaining tenant isolation boundaries.

---

## 12. Differential Dataset Comparison

**Problem**: Datasets are loaded as opaque blobs. There is no capability to compare expected vs. actual data state.

**Improvement**: Add `dataset_diff(sandbox_id, dataset_name_a, dataset_name_b, tenant_id)` that produces a structured diff: added_keys, removed_keys, changed_values, schema_drift. Integrate with `assertion_check` via `field: "__dataset_diff__"` assertion type. Support row-count tolerance thresholds.

**Impact**: Data migration tests, ETL validation, and schema evolution checks get a native primitive instead of ad-hoc comparisons in test code.

---

## 13. Automated Security Posture Scoring

**Problem**: `risk_score` is a simple integer. Security posture is opaque beyond that single number.

**Improvement**: Expand to a `SecurityPosture` model with sub-scores: `network_exposure`, `secret_surface`, `data_sensitivity`, `ttl_risk`, `isolation_gap`, `dependency_risk`. Add `security_posture_report(sandbox_id)` that explains each dimension, flags CWE-relevant patterns (e.g. CWE-200 for unmasked production data), and recommends remediation steps.

**Impact**: Security reviews become data-driven. Automated PR checks can gate deployments on posture regression, not just rule pass/fail.

---

## 14. Test Flakiness Detection and Quarantine

**Problem**: There is no record of whether individual tests or scenarios have historically passed or failed inconsistently.

**Improvement**: Add `FlakinesRecord` tracking per-scenario pass/fail history across runs. `flakiness_score(scenario_id, tenant_id)` returns a 0–1 score based on variance in recent N runs. `quarantine_scenario(scenario_id, reason)` marks it as excluded from CI gates with an expiry. `flakiness_report(sandbox_id)` surfaces top offenders.

**Impact**: CI pipelines stop being gated by known-flaky tests. Flakiness becomes visible technical debt with owner attribution.

---

## 15. WASM Module Registry with Signing Verification

**Problem**: There is no inventory of WASM modules or code artifacts that can be executed inside sandboxes.

**Improvement**: Add `WasmModule` model with `id`, `name`, `version`, `hash_sha256`, `signature`, `signer_id`, `trusted: bool`. Expose `register_wasm_module(name, module_bytes, signer_id)`, `verify_wasm_module(module_id)` (checks hash + signature chain), and `list_wasm_modules(tenant_id)`. `execute_wasm` only accepts registered and verified modules.

**Impact**: Supply chain integrity for sandboxed code. Prevents malicious module substitution. Creates an auditable artifact registry aligned with SLSA provenance requirements.

---

*Generated: 2026-06-11 | Datacraft © 2025*
