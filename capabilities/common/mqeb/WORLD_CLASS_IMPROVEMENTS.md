# MQEB World-Class Improvements

**Capability**: Message Queue Event Bus (`mqeb`)
**Author**: Nyimbi Odero
**Copyright**: © 2025 Datacraft

15 targeted improvements that elevate MQEB from a well-structured governance layer to a production-grade, observable, composable event fabric competitive with Apache Kafka, AWS EventBridge, and Google Cloud Pub/Sub.

---

### I1. Backpressure-Aware Flow Control

**Category**: Reliability | **Justification**: Without admission control, `publish_message` allows unbounded `message_queues` growth under burst load, leading to OOM and cascading failure — a 10x reliability gap vs. Kafka's per-topic quota enforcement. | **Implementation**: Add per-topic `max_queue_depth` to `TopicRecord`; implement `async publish_with_backpressure(tenant_id, message_id, topic_id, ...)` that yields a `queue_pressure_ratio` metric, blocks via `asyncio.wait_for`, or redirects overflow to a configurable spill topic. Token-bucket counter in `BoundedCache` with per-topic TTL window. | **Competitor**: Apache Kafka topic-level `max.message.bytes` + producer quotas, AWS SQS `ApproximateNumberOfMessages` alarms.

---

### I2. Cursor-Based Consumer Groups with Committed Offsets

**Category**: Correctness | **Justification**: Current `consume_messages` uses list-slice with no position tracking; a consumer restart causes message loss or re-delivery with no audit trail — breaking exactly-once semantics at the consumer boundary, which Kafka and Pulsar guarantee natively. | **Implementation**: Add `ConsumerGroupRecord(id, tenant_id, subscription_id, partition, committed_offset, updated_at)`; `async commit_offset(tenant_id, group_id, offset)` and `async seek_offset(tenant_id, group_id, offset, actor, reason)` methods. Offset is a monotonic integer over the subscription queue. Persist offset changes as audit events. | **Competitor**: Apache Kafka consumer groups, Apache Pulsar subscription seek API.

---

### I3. Schema Registry Validation with Fail-Close on Regulated Topics

**Category**: Data Quality | **Justification**: Storing `schema_ref` as a free-form string provides zero payload validation. Regulated topics with corrupt or mismatched payloads cause downstream data corruption that is expensive to remediate — Confluent Schema Registry and AWS Glue Schema Registry prevent this class of bug entirely. | **Implementation**: Define `SchemaRegistryAdapter` protocol with `async validate(schema_ref: str, payload: bytes) -> bool`; add `async validate_message_schema(tenant_id, message_id, schema_ref)` to `MqebService` that calls the adapter, caches compiled validators in `BoundedCache`, sets `schema_validated` on `MessageRecord`, and raises `PermissionError("schema_validation_failed")` on `regulated` topics. | **Competitor**: Confluent Schema Registry, AWS Glue Schema Registry, Apicurio Registry.

---

### I4. Distributed Saga with Compensating Transactions

**Category**: Correctness | **Justification**: Multi-step message flows have no native rollback mechanism. A failure at step N of an N-step saga leaves partially applied state with no compensating events — a gap that patterns like AWS Step Functions and Temporal solve with durable saga orchestration. | **Implementation**: Add `SagaRecord(id, tenant_id, steps: list[SagaStep], current_step, status)` and `SagaStep(topic_id, compensation_topic_id, evidence)`; implement `async start_saga`, `async advance_saga`, and `async compensate_saga` methods. Saga state is stored in `MqebService.sagas` dict and each transition emits an audit event. | **Competitor**: AWS Step Functions, Temporal.io, NServiceBus sagas.

---

### I5. Time-Based Message Scheduling with Cancellation

**Category**: Usability | **Justification**: Immediate-only publish forces consumers to implement their own scheduling logic — duplicating infrastructure and losing audit coverage. AWS SQS message timers and RabbitMQ delayed exchanges show that in-fabric scheduling eliminates a class of consumer-side state machines. | **Implementation**: Add `scheduled_at: str` to `MessageRecord`; implement `async schedule_message(tenant_id, message_id, topic_id, producer, scheduled_at_iso, ...)` that inserts into `scheduled_messages: dict[str, MessageRecord]` keyed by timestamp. `_message_processing_loop` drains expired schedule entries into `message_queues`. Add `async cancel_scheduled_message(tenant_id, message_id, actor, reason)` with audit evidence. | **Competitor**: AWS SQS DelaySeconds, RabbitMQ rabbitmq-delayed-message-exchange, Azure Service Bus scheduled enqueue.

---

### I6. Idempotency Store with TTL Eviction

**Category**: Correctness | **Justification**: Two publishes with the same `idempotency_key` both succeed and produce duplicate messages. Exactly-once delivery is logically impossible without deduplication at the producer boundary — AWS SQS FIFO and Kafka's idempotent producer both enforce this. | **Implementation**: Maintain `_idempotency_cache: dict[str, tuple[str, datetime]]` keyed by `(tenant_id, idempotency_key)` → `(message_id, expires_at)`. On `publish_message`, check cache first and return existing `MessageRecord.to_dict()` immediately. Evict entries after topic `retention_days` in `_cleanup_expired_messages`. | **Competitor**: AWS SQS FIFO deduplication, Apache Kafka idempotent producer, Azure Service Bus duplicate detection.

---

### I7. Structured Retry Policies with Exponential Backoff

**Category**: Reliability | **Justification**: `DeliveryAttemptRecord` stores `retry_count` with no policy enforcement. Without configurable backoff, retry storms hit failing endpoints at full rate, worsening cascading failures — a pattern that AWS Lambda event source mappings and Celery retry policies prevent by default. | **Implementation**: Add `RetryPolicyRecord(id, tenant_id, subscription_id, strategy, base_delay_ms, max_attempts, jitter_factor, dead_letter_on_exhaust)`; implement `async apply_retry_policy(tenant_id, attempt_id)` that computes next delivery timestamp via exponential backoff with jitter and auto-routes to DLQ on exhaustion. `async get_retry_policy(tenant_id, subscription_id)` returns current policy. | **Competitor**: AWS Lambda event source mapping retry, Celery retry with exponential backoff, Azure Service Bus retry policies.

---

### I8. Priority Queue with Preemptive Weighted Scheduling

**Category**: Performance | **Justification**: All messages enter a single FIFO queue regardless of `priority`. CRITICAL messages wait behind LOW messages published earlier — a gap that AWS SQS message priority via separate queues and RabbitMQ priority queues solve at the broker level. | **Implementation**: Replace `message_queues: dict[str, list[str]]` with `message_queues: dict[str, dict[str, list[str]]]` keyed by priority tier (critical→high→normal→low). `_process_pending_deliveries` drains tiers in weighted order with configurable `priority_weights: dict[str, int]`. Add `async get_priority_queue_stats(tenant_id, topic_id)` exposing per-tier depths and drain rates. | **Competitor**: RabbitMQ priority queues, AWS SQS separate-queue priority pattern, ActiveMQ JMS priority.

---

### I9. Append-Only Audit Log with HMAC Integrity

**Category**: Security | **Justification**: Audit events stored in a mutable dict can be silently overwritten, undermining forensic integrity for compliance audits. SOC 2, GDPR, and PCI-DSS require tamper-evident logs — a property that AWS CloudTrail log file integrity validation and Google Cloud Audit Logs enforce natively. | **Implementation**: Replace `audit_events: dict` with `_audit_log: list[MqebAuditEventRecord]` and secondary `_audit_index: dict[str, int]` for O(1) lookup. Sign each record with `hmac.new(tenant_secret, record_bytes, sha256).hexdigest()` stored as `integrity_sig`. Add `async stream_audit_events(tenant_id, since_id)` as `AsyncGenerator[dict, None]` for tail consumption. | **Competitor**: AWS CloudTrail log integrity, Google Cloud Audit Logs, Splunk immutable index.

---

### I10. Circuit Breaker for Downstream Delivery Endpoints

**Category**: Reliability | **Justification**: A failing webhook or gRPC endpoint is retried indefinitely, consuming threads and delaying delivery to healthy subscribers. Netflix Hystrix and resilience4j circuit breakers prevent cascade failures by opening the circuit after a configurable failure threshold. | **Implementation**: Add `CircuitBreakerRecord(id, tenant_id, subscription_id, state, failure_count, threshold, reset_timeout_s, tripped_at)`; `_deliver_messages_to_subscription` checks breaker state before delivery and increments `failure_count` on error. Add `async reset_subscription_circuit_breaker(tenant_id, subscription_id, actor, evidence)` for operator override with audit event. | **Competitor**: Netflix Hystrix, resilience4j, AWS SDK adaptive retry with circuit breaker.

---

### I11. Content-Based Message Routing DSL

**Category**: Usability | **Justification**: `_subscription_matches_topic` uses `fnmatch` on topic names only. Header- or payload-based routing requires consumer-side filtering, forcing every consumer to implement duplicate logic — a gap that Apache Camel content-based router and AWS EventBridge rules solve at the broker. | **Implementation**: Add `RoutingRule(id, tenant_id, subscription_id, match_expr: str)` where `match_expr` is a safe DSL (e.g., `priority == "high" AND tenant_id == "tenant-a"`); `evaluate_routing_rule(message_dict, rule)` uses Python `ast.parse` with a restricted visitor (no attribute access, no calls, no builtins). Rules evaluated before queue insertion in `_route_message_to_subscriptions`. | **Competitor**: Apache Camel content-based router, AWS EventBridge rules, RabbitMQ header exchanges.

---

### I12. Tenant Quotas and Token-Bucket Rate Limiting

**Category**: Multi-tenancy | **Justification**: Any tenant can publish unlimited messages at unlimited throughput, allowing a single noisy tenant to starve all others. Kafka per-principal quotas and AWS SQS account-level limits prevent noisy-neighbour effects that destabilize shared infrastructure. | **Implementation**: Add `TenantQuotaRecord(id, tenant_id, max_messages_per_minute, max_bytes_per_minute, max_topics)`; token-bucket counter in `BoundedCache` decremented on `publish_message`. Exceeded quota raises `PermissionError("tenant_quota_exceeded")` with audit event. Add `async get_tenant_quota_status(tenant_id)` returning `{used_messages, used_bytes, utilization_ratio}`. | **Competitor**: Apache Kafka client quotas, AWS SQS account quotas, Google Cloud Pub/Sub resource limits.

---

### I13. Message Batch Publish with Optional Compression

**Category**: Performance | **Justification**: Per-message policy evaluation, audit logging, and routing overhead is multiplied for high-volume producers. Kafka producer batching and gRPC streaming batch RPCs demonstrate 10–50x throughput improvement by amortizing per-message overhead across batches. | **Implementation**: Add `async publish_message_batch(tenant_id, producer, topic_id, messages: list[dict])` that runs policy evaluation once for shared context fields, compresses payload bytes with `zstd` when `compression_type="zstd"`, stores `compressed_size_bytes` and `compression_ratio` on each `MessageRecord`, and emits one audit event with `batch_size`. | **Competitor**: Apache Kafka producer batch.size, AWS SQS SendMessageBatch, Google Cloud Pub/Sub batch settings.

---

### I14. Dead-Letter Queue Lifecycle with Redrive Workflows

**Category**: Reliability | **Justification**: `_process_dead_letter_queues` is a no-op stub. Messages that land in the DLQ are never inspected, redriven, or discarded, making undeliverable message recovery a manual out-of-band process. AWS SQS DLQ redrive and Azure Service Bus dead-letter management provide full in-fabric DLQ lifecycle. | **Implementation**: Implement `async inspect_dead_letter_queue(tenant_id, topic_id)`, `async redrive_dead_letter_messages(tenant_id, dlq_topic_id, reviewer, evidence, max_count)` (re-publishes to originating topic after reviewer evidence check), `async purge_dead_letter_queue(tenant_id, dlq_topic_id, reviewer, reason)`, and `async export_dead_letter_messages(tenant_id, dlq_topic_id)` as `AsyncGenerator`. Configurable `max_redrive_attempts` on `TopicRecord`. | **Competitor**: AWS SQS DLQ redrive policy, Azure Service Bus dead-letter management, RabbitMQ dead-letter exchanges.

---

### I15. OpenTelemetry W3C Trace Context Propagation

**Category**: Observability | **Justification**: Messages crossing service boundaries lose distributed trace context. Correlating a published message with its downstream processing spans requires manual instrumentation per consumer — a gap that Kafka's `opentelemetry-instrumentation-kafka-python` and AWS X-Ray trace header propagation solve automatically at the broker boundary. | **Implementation**: Add `trace_context: dict[str, str]` to `MessageRecord` for W3C Trace Context headers (`traceparent`, `tracestate`). `publish_message` accepts optional `trace_context` parameter and propagates it through `_route_message_to_subscriptions`. `consume_messages` returns messages with trace context intact. When `opentelemetry-api` is importable, auto-inject/extract via standard propagator; fall back to passthrough dict otherwise. | **Competitor**: opentelemetry-instrumentation-kafka-python, AWS X-Ray SQS/SNS trace propagation, Datadog APM distributed tracing.
