# APG Audit Log (audl) — World-Class Improvements

© 2025 Datacraft  www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>

---

### I1. Merkle-Tree Chain Hash with Batch Root Proofs
**Category**: Integrity / Cryptography
**Justification**: SHA-256 linked-list chains are sequential and non-parallelisable. A binary Merkle tree over each epoch batch lets an auditor verify any single event against an O(log n) proof path without replaying the entire chain. AWS CloudTrail Log File Validation and Certificate Transparency both use tree-based proofs for exactly this reason.
**Implementation**: Group events by 1-minute epochs. Compute leaf hashes as `SHA-256(event_checksum + epoch_seq)`. Build a standard binary Merkle tree per epoch; store the root hash in a `AL_epoch_roots` table. Add `merkle_proof_path: list[str]` to `AuditEventResponse`. Service method `async def merkle_proof(event_id)` returns the sibling path and root so any external verifier can reconstruct proof without DB access.
**Competitor**: Google Certificate Transparency RFC 6962; AWS CloudTrail Digest Files.

---

### I2. Immutable PostgreSQL Append-Only Partition with Row-Level Security
**Category**: Storage / Multi-Tenancy
**Justification**: In-memory `_events` dict is not durable. PostgreSQL declarative partitioning by `(tenant_id, YEAR-MONTH)` with `FOR VALUES` keeps hot partitions small. A `RULE ON UPDATE DO INSTEAD NOTHING` plus `RULE ON DELETE DO INSTEAD NOTHING` enforces append-only at the storage layer — independent of application bugs. Row-Level Security policies then guarantee that `SET LOCAL app.tenant_id = 'X'` is the only gate required; all service queries automatically scope.
**Implementation**: Alembic migration creates `apg_audit_events` as a partitioned table. `_persist_audit_event_to_db` already exists; extend it with RLS activation. Add `async def verify_storage_immutability()` that runs `EXPLAIN SELECT` on `UPDATE/DELETE` and asserts zero rows affected.
**Competitor**: PlanetScale immutable tables; Snowflake `FAIL_SAFE + TIME_TRAVEL`.

---

### I3. Structured Log Budget with Backpressure Signalling
**Category**: Reliability / Flow Control
**Justification**: Unbounded `log_event` calls under load cause OOM and log corruption. Netflix Atlas and Datadog Agent both implement per-tenant token-bucket rate limits at the ingestion layer. A budget enforces SLA: high-risk events are never dropped; low-risk events are sampled when budget exhausted.
**Implementation**: Add `LogBudget` dataclass: `max_events_per_minute: int`, `high_risk_always_pass: bool`. `log_event` checks a per-tenant `asyncio.Semaphore`-backed counter. When exhausted it either drops (with a dropped-count counter) or raises `LogBudgetExceededError` (callers can catch and queue). Expose `async def log_budget_status() -> dict` returning consumed/remaining.
**Competitor**: Datadog Agent write-ahead log with backpressure; Splunk HEC acknowledgement protocol.

---

### I4. Monetary Amount Auditing with Decimal Precision
**Category**: Fintech / Correctness
**Justification**: Financial events logged with `float` amounts suffer IEEE-754 drift (e.g., 0.1 + 0.2 ≠ 0.3). PCI-DSS Requirement 10.2 mandates exact capture of transaction values. `Decimal` with quantization to `0.00000001` (8 dp) prevents phantom diffs during forensic audit replay.
**Implementation**: Add `monetary_amount: Decimal | None` and `monetary_currency: str | None` to `AuditEventCreate` / `AuditEventResponse`. Use `Annotated[Decimal, AfterValidator(lambda v: v.quantize(Decimal('0.00000001')))]`. Include amount and currency in the `checksum` pre-image so post-event mutation is detected. New service method `async def log_financial_event(amount, currency, ...)` wraps `log_event` and validates `amount > Decimal(0)`.
**Competitor**: Stripe audit logs use integer cents; Adyen uses exact decimal strings — both avoid float entirely.

---

### I5. Tenant-Scoped Saved Queries with Scheduled Re-Execution
**Category**: Compliance / Automation
**Justification**: Compliance officers run the same GDPR/SOX queries weekly. Storing parameterised queries (already in `AuditQueryResponse`) and re-running them on a cron-like schedule produces durable evidence that continuous monitoring is active — a SOC 2 CC7.2 requirement. Competitors like Sumo Logic call these "Scheduled Searches."
**Implementation**: Add `schedule_cron: str | None` and `next_run_at: datetime | None` to `AuditQueryResponse`. New service method `async def schedule_query(query_id, cron_expr)`. A background `asyncio.Task` (`_query_scheduler`) runs due queries, stores result snapshots in `_query_results: dict[str, list[AuditSearchResult]]`. `async def get_scheduled_query_results(query_id)` returns history.
**Competitor**: Sumo Logic Scheduled Searches; Splunk Saved Searches with schedule.

---

### I6. Cryptographically-Signed Evidence Packages (PKCS#7 / JWS)
**Category**: Legal / Non-Repudiation
**Justification**: A ZIP of JSON events can be tampered after export. PKCS#7 detached signatures (or JSON Web Signatures per RFC 7515) bind the package to the platform's private key. Courts and regulators can verify authenticity without contacting the originating system. This is required for e-discovery under US FRCP Rule 34.
**Implementation**: Add `signature: str | None` and `signing_key_id: str | None` to `EvidencePackageResponse`. `evidence_package_export` signs `pkg_checksum` using the platform RSA/ECDSA private key (loaded from env `APG_SIGNING_KEY_PATH`). Expose `async def verify_evidence_signature(pkg_id)` which re-derives checksum and verifies signature against the public key. Falls back gracefully if key not configured.
**Competitor**: DocuSign eSignature; AWS CloudTrail with KMS-signed digest files.

---

### I7. Behavioral Baseline + Statistical Anomaly Scoring
**Category**: Security / ML
**Justification**: Static risk-score thresholds miss low-and-slow attacks. Building a rolling per-actor baseline (mean + stddev of events-per-hour, avg risk_score) lets `log_event` compute a z-score anomaly flag in O(1) at write time — no ML model required. LinkedIn's SIEM uses exactly this Welford online algorithm for streaming anomaly detection.
**Implementation**: Maintain `_actor_baselines: dict[str, ActorBaseline]` where `ActorBaseline` tracks Welford incremental mean/variance for event rate and risk score. `log_event` updates the baseline and sets `anomaly_score` to `min(1.0, z_score / 3.0)`. New method `async def actor_behavior_baseline(actor_id)` returns the current stats. After N=30 events the baseline is considered stable; before that `anomaly_score = 0.0`.
**Competitor**: Elastic SIEM ML jobs; Panther Labs statistical detection rules.

---

### I8. Cross-Capability Event Correlation via NATS JetStream
**Category**: Integration / Event-Driven
**Justification**: Siloed audit logs across capabilities (auth, payments, intel) mean analysts must manually correlate. Publishing every event to NATS subject `apg.events.audl.>` with the `correlation_id` header lets downstream capabilities (intel, grc) subscribe and auto-correlate without polling. `_publish_audit_to_nats` stub already exists — this completes it.
**Implementation**: Replace the stub with a real `nats.aio` publish using `JetStream.publish`. Include `Nats-Msg-Id: {event_id}` header for exactly-once delivery. Add `async def subscribe_correlated_events(correlation_id) -> AsyncGenerator[dict]` that creates an ephemeral consumer on `apg.events.audl.{correlation_id}` and yields messages. Unit tests use `nats-server` via `pytest-nats-server`.
**Competitor**: Confluent Platform audit log connector; Segment Protocols for cross-system event correlation.

---

### I9. Differential Privacy for Aggregate Compliance Reports
**Category**: Privacy / Analytics
**Justification**: Raw `compliance_report` counts can leak individual-level information via differencing attacks (e.g., run report with/without one user). Adding Laplace-mechanism noise to aggregate counts — calibrated to `sensitivity / epsilon` — makes reports `(epsilon, 0)`-differentially private without breaking compliance utility. Apple and Google both use DP for aggregate telemetry.
**Implementation**: Add `dp_epsilon: float | None = None` to `ComplianceReportCreate`. When set, the service adds `numpy.random.laplace(loc=0, scale=1.0/dp_epsilon)` noise (rounded to int) to each aggregate count in `summary`. Mark the report with `dp_applied: bool = True` so consumers know noise was added. Expose `async def compliance_report_with_dp(req, epsilon)` as the privacy-first entry-point.
**Competitor**: Apple's Private Federated Learning; Google RAPPOR; OpenDP library.

---

### I10. Webhook-Based Real-Time Alert Dispatch
**Category**: Alerting / Integration
**Justification**: The SIEM stream requires a persistent websocket consumer. Many SIEM integrations prefer outbound webhooks (Slack, PagerDuty, Splunk HEC). Alert rules evaluated at write time with sub-100 ms dispatch eliminate polling lag — a PCI-DSS Requirement 10.6 continuous monitoring mandate.
**Implementation**: Add `AlertRule` Pydantic model: `event_types`, `risk_score_min`, `actor_ids`, `webhook_url`, `secret_header`. Store in `_alert_rules: dict[str, AlertRule]`. After `log_event` completes, call `_evaluate_alert_rules(resp)` which fires matching rules via `aiohttp.ClientSession.post` with HMAC-SHA256 `X-APG-Signature` header. Add `async def add_alert_rule(rule)` and `async def list_alert_rules()` to the service.
**Competitor**: PagerDuty Events API v2; Splunk HEC with alert webhook actions.

---

### I11. Immutable Cold Storage Tiering to S3-Compatible Object Store
**Category**: Cost / Compliance
**Justification**: Keeping 7-year audit logs in hot PostgreSQL is 50× more expensive than S3 Glacier. WORM (Write Once Read Many) object lock satisfies SEC Rule 17a-4 immutability requirements. AWS CloudTrail, Splunk SmartStore, and Snowflake all tier cold data to object stores.
**Implementation**: Add `async def archive_to_object_store(policy_id)` that reads events past `archive_after_days`, serialises them as JSONL, computes a SHA-256 bundle checksum, uploads to `s3://{bucket}/audl/{tenant_id}/{year}/{month}/{bundle_id}.jsonl` with `x-amz-object-lock-mode: COMPLIANCE`. Stores a `AL_archive_manifest` DB record with `bundle_checksum` and `s3_key`. `async def restore_from_archive(bundle_id)` downloads and re-imports for query.
**Competitor**: AWS CloudTrail + S3 Glacier; Splunk SmartStore; Elastic ILM cold phase.

---

### I12. OpenTelemetry Trace Context Propagation
**Category**: Observability / Distributed Tracing
**Justification**: Audit events logged outside their originating OTEL trace span lose causality. Propagating `traceparent` / `tracestate` W3C headers into `AuditEventCreate.correlation_id` lets Jaeger / Tempo reconstruct the full request graph alongside its audit trail. Honeycomb and Datadog APM both capture audit events as spans for exactly this reason.
**Implementation**: Add `otel_trace_id: str | None` and `otel_span_id: str | None` to `AuditEventCreate` / `AuditEventResponse`. Include them in the checksum pre-image. `log_event` auto-extracts from `opentelemetry.trace.get_current_span()` if not provided. New method `async def events_by_trace(trace_id)` returns all events for an OTEL trace across the tenant.
**Competitor**: Honeycomb audit events as spans; Datadog APM + Audit Trail correlation.

---

### I13. Read-Model Projection Cache with TTL Invalidation
**Category**: Performance / CQRS
**Justification**: `audit_analytics` and `risk_summary` scan all in-memory events on every call — O(n) per request. At 10M events/tenant this is seconds of latency. A pre-computed read-model updated on every write (CQRS) reduces query latency to O(1). Axon Framework and EventStoreDB both use projection stores for exactly this pattern.
**Implementation**: Add `_read_model: dict[str, Any]` (per-tenant analytics snapshot) and `_read_model_dirty: bool`. `log_event` sets `_read_model_dirty = True`. `audit_analytics` calls `async def _rebuild_read_model()` only when dirty. Use `BoundedCache` (already imported from `capabilities.common.reliability`) with TTL for caching heavy aggregates. `async def invalidate_read_model()` forces rebuild.
**Competitor**: EventStoreDB projections; Axon Framework query model; CQRS.io.

---

### I14. Structured Audit Policy-as-Code with Rego / CEL Evaluation
**Category**: Governance / Policy
**Justification**: Hard-coded compliance checks in `_framework_recommendations` cannot adapt to customer-specific rules without code deploys. Open Policy Agent (OPA) Rego or Google's Common Expression Language (CEL) let compliance officers define `audit_policy.rego` files that the service evaluates at report-generation time. HashiCorp Sentinel and AWS Cedar use exactly this model.
**Implementation**: Add `AuditPolicy` model: `name`, `framework`, `rego_source: str | None`, `cel_expr: str | None`. Store in `_policies_code: dict[str, AuditPolicy]`. `compliance_report` calls `async def _evaluate_policies(events, framework)` which uses `opa_python` or `cel-python` to evaluate each policy against the event batch. Violations are appended to `summary["policy_violations"]`. `async def register_audit_policy(policy)` adds rules at runtime.
**Competitor**: HashiCorp Sentinel; AWS Cedar; OPA Gatekeeper.

---

### I15. Time-Locked Legal Hold with Court-Order Workflow
**Category**: Legal / Compliance
**Justification**: `set_legal_hold(hold=False)` requires no justification and no approval today. US FRCP Rule 37(e) and UK Civil Procedure Rule 31 require a documented, dual-authorised release process. A time-locked hold (cannot be released before `min_hold_until`) plus a two-step approval workflow (request + countersignature) prevents accidental or malicious spoliation.
**Implementation**: Add `min_hold_until: datetime | None`, `release_requested_by: str | None`, `release_approved_by: str | None`, `release_approved_at: datetime | None` to `AuditEventResponse` legal-hold fields. `set_legal_hold(hold=False)` creates a `LegalHoldReleaseRequest` requiring a second actor to call `async def approve_legal_hold_release(event_id, approver_id)`. Before `min_hold_until` the approval is rejected with `LegalHoldMinimumPeriodError`. Emit `legal_hold_release_requested` and `legal_hold_released` domain events.
**Competitor**: Veritas Enterprise Vault legal hold workflow; Exterro Legal Hold Management; Onna eDiscovery holds.
