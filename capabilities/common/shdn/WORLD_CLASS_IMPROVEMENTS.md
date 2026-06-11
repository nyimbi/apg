# SHDN - World-Class Improvements

Fifteen improvements that would raise SHDN from "production-ready" to "best-in-class" lifecycle orchestration.

---

### I1. Progressive Drain with Back-Pressure Signalling
**Category**: Drain Quality | **Justification**: Current `start_drain` records a point-in-time session count but does not model the drain curve. Production systems (Envoy, Istio) emit real-time drain progress so downstream load balancers can stop routing before the drain completes, reducing in-flight errors by up to 90%. | **Implementation**: Add `update_drain_progress(tenant_id, drain_id, active_sessions, queue_depth)` that patches `DrainOperationRecord` fields and emits a NATS `shdn.drain.progress` subject per tick; add a `drain_curve: list[DrainProgressPoint]` field to the record. | **Competitor**: Kubernetes Endpoint Slice controller + preStop hook pattern; Envoy `drain_type: gradual`

---

### I2. SIGTERM / SIGINT Handler Injection
**Category**: OS Signal Handling | **Justification**: Services must bind OS signals to lifecycle state transitions. Without a canonical signal handler, each app re-implements ad-hoc `signal.signal()` calls, creating inconsistent drain ordering and race conditions. Google SRE Workbook §7 mandates signal-driven shutdown as the primary safe-stop mechanism. | **Implementation**: Add `install_signal_handlers(tenant_id, target_id, actor, service_instance)` that registers `SIGTERM`/`SIGINT` via `asyncio.loop.add_signal_handler`, triggering `service_drain` then `graceful_shutdown` in sequence; emit `signal_received` audit event with signal name and timestamp. | **Competitor**: Gunicorn worker signal handling; systemd `KillMode=mixed` + `TimeoutStopSec`

---

### I3. NATS-Backed Lifecycle Event Bus
**Category**: Event Streaming | **Justification**: Bytewax is the batch/stream processor; NATS JetStream is the transport layer. Publishing lifecycle events directly to NATS subjects enables real-time fan-out to monitoring, alerting, and dependent services without polling. Reduces shutdown notification latency from O(seconds) to O(milliseconds). | **Implementation**: Add `publish_lifecycle_event(tenant_id, subject, payload, stream)` using `nats.aio.client`; subjects follow `apg.shdn.<event_type>.<tenant_id>`; persist NATS sequence number on the audit record for exactly-once delivery tracking. | **Competitor**: AWS EventBridge + Lambda; GCP Pub/Sub + Cloud Run; Temporal workflow signals

---

### I4. Dependency-Ordered Shutdown Sequencing
**Category**: Orchestration | **Justification**: Shutting down services without respecting their dependency graph causes cascading failures. Kubernetes drains nodes in reverse dependency order; SHDN currently stores `dependencies` but does not topologically sort the shutdown sequence. | **Implementation**: Add `compute_shutdown_order(tenant_id, plan_id)` using Kahn's algorithm on the `ShutdownTargetRecord.dependencies` graph; return `{order: list[str], cycles: list[list[str]]]}`; reject plans with cyclic dependencies at creation time. | **Competitor**: HashiCorp Terraform destroy order; Helm hook `pre-delete` weight ordering

---

### I5. Graceful HTTP/2 GOAWAY Emitter
**Category**: Connection Draining | **Justification**: HTTP/2 persistent connections are not closed by a TCP FIN during drain; without a GOAWAY frame, clients keep sending requests to a draining server. h2c-aware proxies (nginx, Caddy, Envoy) wait for GOAWAY before removing the upstream. | **Implementation**: Add `send_http2_goaway(tenant_id, target_id, actor, listener_ref, last_stream_id)` that records a GOAWAY advisory with the last accepted stream ID; downstream Bytewax consumer routes this to the actual proxy sidecar via NATS `apg.shdn.goaway.<target_id>`. | **Competitor**: Envoy `GOAWAY` drain; gRPC `gracefulStop()` pattern

---

### I6. Circuit-Breaker Integration for Dependent Services
**Category**: Resilience | **Justification**: When a service drains, its dependents must open their circuit breakers to stop sending traffic. Without explicit circuit-breaker coordination, the drain period sees elevated error rates as in-flight requests hit a half-drained service. Netflix Hystrix / Resilience4j open breakers on planned shutdowns, reducing error rates during drain by 60-70%. | **Implementation**: Add `open_circuit_breakers(tenant_id, target_id, actor, breaker_refs: list[str])` that records `CircuitBreakerOpenRecord` entries and emits NATS `apg.shdn.circuit_open` per breaker; add matching `close_circuit_breakers` for post-restart. | **Competitor**: Resilience4j `CircuitBreaker.transitionToOpenState()`; Polly circuit breaker + IHostApplicationLifetime

---

### I7. Pre-Shutdown Readiness Probe Override
**Category**: Health Gate | **Justification**: During drain, Kubernetes liveness probes must keep passing (process is alive) while readiness probes must fail (stop routing). SHDN currently records a single `health_gate_ref` string but does not distinguish probe types, causing operators to manually manage probe responses. | **Implementation**: Add `set_readiness_probe_state(tenant_id, target_id, actor, ready: bool, probe_url: str)` that stores a `ReadinessProbeStateRecord` and emits NATS `apg.shdn.readiness.<target_id>`; Bytewax consumer can drive actual probe endpoints via the deployment adapter. | **Competitor**: Kubernetes readiness gate `PodReadinessGate`; AWS ALB target deregistration delay

---

### I8. Multi-Phase Shutdown Pipeline with SLA Tracking
**Category**: Orchestration Quality | **Justification**: Enterprise shutdowns have contractual SLAs (e.g., "drain within 5 minutes, backup within 10"). Without phase-level SLA tracking, operators cannot tell which phase caused an SLA breach. Spanner and BigTable publish per-phase shutdown latency in their SLA reports. | **Implementation**: Add `create_shutdown_phase(tenant_id, plan_id, phase_name, phase_type, sla_seconds)` and `complete_shutdown_phase(tenant_id, phase_id, actor, evidence_ref)` that track `started_at`, `completed_at`, and `sla_breach: bool`; expose phase timeline in `shutdown_report`. | **Competitor**: Temporal workflow timers + `getInfo().historyLength`; AWS Step Functions execution timeline

---

### I9. Idempotent Re-entry with Fencing Tokens
**Category**: Safety | **Justification**: Network partitions can cause shutdown operations to be retried, leading to double-shutdown of a service. Distributed systems (DynamoDB, etcd) use fencing tokens (monotonic counters per resource) to reject stale or duplicate mutations. | **Implementation**: Add `acquire_shutdown_fence(tenant_id, target_id, actor)` returning `{fence_token: int, expires_at: str}`; validate `fence_token` on `execute_shutdown` and `emergency_stop`; reject operations with stale tokens with `shutdown_fence_expired` error. | **Competitor**: etcd lease + revision fencing; PostgreSQL `FOR UPDATE SKIP LOCKED` advisory locks

---

### I10. Canary Shutdown Validation
**Category**: Safety | **Justification**: Before shutting down all instances of a service, operators should validate that a single instance can be cleanly shut down and restarted without data loss. PagerDuty and Stripe use canary shutdown patterns in their deployment pipelines to catch state-leakage bugs before fleet-wide shutdowns. | **Implementation**: Add `canary_shutdown_test(tenant_id, target_id, canary_instance_ref, actor, validation_ref)` that marks one instance as canary, executes a simulated drain+shutdown+restart cycle, records the result, and gates the full shutdown plan on `canary_passed: True`. | **Competitor**: Stripe canary deployment + health-check gating; Kubernetes PodDisruptionBudget `maxUnavailable: 1`

---

### I11. Tenant-Isolated Shutdown Budget (PDB Equivalent)
**Category**: Governance | **Justification**: Kubernetes PodDisruptionBudgets prevent too many replicas from being shut down simultaneously. SHDN has no equivalent, so a runaway automation agent could shut down all replicas of a critical service simultaneously. | **Implementation**: Add `set_shutdown_budget(tenant_id, target_id, max_simultaneous_shutdowns, window_seconds)` storing `ShutdownBudgetRecord`; enforce in `execute_shutdown` by counting active executions in the window; reject when budget exceeded with `shutdown_budget_exceeded`. | **Competitor**: Kubernetes `PodDisruptionBudget`; AWS Auto Scaling instance protection

---

### I12. Immutable Audit Trail with Merkle Anchoring
**Category**: Compliance | **Justification**: Regulated industries (finance, health, government) require tamper-evident audit trails. Current `LifecycleAuditEventRecord` is mutable in-memory. Anchoring audit events to a Merkle tree (hash-chain) makes any post-hoc mutation detectable, satisfying SOC 2 CC7.2 and ISO 27001 A.12.4. | **Implementation**: Add `anchor_audit_chain(tenant_id, up_to_event_id)` that computes a rolling SHA-256 hash chain over audit event IDs and stores the chain root; add `verify_audit_chain(tenant_id, from_event_id, to_event_id)` that recomputes and compares; emit NATS `apg.shdn.audit_anchored` with chain root. | **Competitor**: Trillian transparency log (Google); AWS CloudTrail log file integrity validation

---

### I13. Weighted Dependency Criticality Propagation
**Category**: Risk Scoring | **Justification**: Not all dependencies are equally critical. A database dependency is more critical than a cache dependency. Current SHDN stores dependencies as a flat list with no weights, preventing risk-scored shutdown ordering. LinkedIn's deployment system uses criticality-weighted dependency graphs to prioritise drain order. | **Implementation**: Add `set_dependency_weight(tenant_id, target_id, dependency_id, weight: float, dependency_type: str)` storing `DependencyWeightRecord`; incorporate weights into `compute_shutdown_order` to break topological ties; expose weighted risk score in `dashboard_summary`. | **Competitor**: Backstage `dependsOn` with `lifecycle` classification; PagerDuty service dependency graph

---

### I14. Automated Rollback Trigger on Health Degradation
**Category**: Autonomous Safety | **Justification**: If post-restart health checks fail, operators must manually trigger rollback. Automated rollback on health degradation (similar to Argo Rollouts `AnalysisRun` failure handling) reduces MTTR from minutes to seconds on bad deployments. | **Implementation**: Add `watch_post_restart_health(tenant_id, plan_id, target_id, actor, health_probe_ref, threshold_failures: int, auto_rollback: bool)` that stores a `PostRestartWatchRecord`; if `auto_rollback=True` and health failures exceed threshold, automatically invoke `rollback_inflight` and emit `auto_rollback_triggered` audit event. | **Competitor**: Argo Rollouts `AnalysisTemplate` with automatic rollback; Flagger progressive delivery

---

### I15. Cross-Capability Composability Contract (shdn + hlth + moni)
**Category**: Composability | **Justification**: SHDN declares dependencies on `hlth`, `moni`, `bkup`, and `audl` but provides no runtime binding mechanism. APG capabilities that compose SHDN must manually wire these, leading to integration drift. Defining a formal composability contract (adapter interface + event contract) allows APG to auto-generate the wiring. | **Implementation**: Add `bind_capability_adapter(tenant_id, capability_id: str, adapter_ref: str, adapter_config: dict)` storing `CapabilityAdapterBindingRecord`; validate that bound capabilities are in the `requires` list; emit `capability_bound` audit event; expose bound adapters in `describe()` and `dashboard_summary`. | **Competitor**: Open Application Model (OAM) `Trait` bindings; Dapr building block bindings
