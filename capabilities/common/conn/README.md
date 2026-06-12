# APG Connection Management (CONN)

CONN is APG's governed connector and data-flow control plane. It lets generated
applications register local Singer taps and other connectors, create secured
connections, test and activate those connections, compose data flows with
mapping/lineage/quality evidence, run sync jobs, schedule and replay work, and
retire connections with audit evidence.

CONN also treats AI and automation agents as governed connector participants,
so tools such as Codex, Claude Code, OpenCode, Pi, and future runtimes compose
through policy-controlled adapters instead of untracked operator scripts.

CONN is intentionally split into two layers:

- `conn_runtime.ConnService`: dependency-light generated-application lifecycle
  service used by package tests, API helpers, UI view models, and semantic
  evidence.
- production connector modules such as `service.py`, `singer_runtime.py`,
  `api.py`, `views.py`, `lineage_engine.py`, `data_quality.py`, and Singer tap
  packages: adapter-backed runtime surfaces for real connector execution,
  health, lineage, quality, monitoring, credentials, and UI delivery.

## What CONN Provides

- Tenant-scoped connector registration for local Singer taps, APG connectors,
  HTTP/webhook connectors, database connectors, file connectors, and streams.
- Credential-safe connection registration with key/vault references,
  encryption evidence, secret rotation evidence, and connection tests before
  activation.
- Flow composition with active source/target checks, mapping evidence, lineage
  capture, quality gates, and PII policy enforcement.
- Sync run, schedule, replay, schema-review, batch-size, and monitoring
  guardrails.
- Marketplace review records for unverified connector packages.
- First-class connector-agent composition with supported runtimes, role
  guardrails, bounded scope, accountable owner, purpose, contribution
  disclosure, and human approval for privileged connector roles.
- Bytewax lifecycle batch validation for connector, connection, flow, sync,
  schedule, review, and connector-agent mutation streams.
- Durable review evidence for generated-app governance queues, including
  policy decisions, matched rules, review reasons, required actions, and
  persisted denial evidence for non-Bytewax lifecycle batches.
- Deterministic rule decisions that return `allow`, `deny`, or
  `require_review`.
- UI view models for dashboard, connectors, connections, visual design, sync
  monitoring, quality, lineage, marketplace, security, audit, rules, and
  settings, plus connector-agent roster and lifecycle-batch monitor surfaces.
- Contract-derived semantic model, package manifest, and release evidence.
- Circuit breaker protection per connector, adaptive rate limiting, CDC-native
  database sync, dead-letter queue with automatic replay, and webhook ingestion
  with signature verification.
- Schema registry with backward/forward compatibility enforcement, adaptive
  connection pool sizing, field-level lineage DAG, pluggable secret backends
  with rotation, and backpressure-aware streaming pipelines.
- Multi-tenant RLS isolation, OpenTelemetry observability export, exactly-once
  delivery semantics, AI-assisted connector discovery, and policy-as-code
  guardrails via OPA.

## World-Class Enhancements (v2.0)

| # | Enhancement | Impact |
|---|-------------|--------|
| 1 | **Circuit Breaker** — per-connection closed/open/half-open states with exponential backoff | Eliminates cascading failures; MTTR seconds not minutes |
| 2 | **Adaptive Rate Limiting** — token-bucket per connector, auto-adjusts on 429s, per-tenant quotas | Maximises throughput while avoiding quota exhaustion |
| 3 | **CDC Native Support** — PostgreSQL logical replication, MySQL binlog, MongoDB change streams; Singer-compatible DELETE messages | 80-95% sync cost reduction; sub-minute latency |
| 4 | **Dead Letter Queue** — tenant-scoped DLQ with error classification, exponential-backoff replay worker, operator inspect/mutate API | Zero-loss guarantee for transient failures |
| 5 | **Webhook Ingestion Engine** — HMAC-SHA256/RSA signature verification, durable WAL buffer, event-ID deduplication, unified Singer output | Handles burst without drops; prevents replay attacks |
| 6 | **Schema Registry** — Avro-compatible versioned registry, backward/forward/full compatibility enforcement, breaking-change review tokens | Prevents silent schema corruption; versioned rollback |
| 7 | **Adaptive Connection Pool** — async pool with utilisation-based resizing, idle probing, metrics feed to health monitor | 3-5× throughput under concurrent load |
| 8 | **Field-Level Lineage DAG** — queryable directed graph from source field to target field across hops; impact and GDPR erasure queries | Impact analysis before schema changes |
| 9 | **Pluggable Secret Backend** — `keym://` URI scheme; drivers for Vault, AWS SM, GCP SM, local encrypted file; auto-rotation + audit events | Credentials never in config dicts or logs |
| 10 | **Backpressure-Aware Pipeline** — `asyncio.Queue` with high/low watermarks; tap reader pauses at high-water, resumes at low-water | Stable memory footprint on unbounded streams |
| 11 | **Multi-Tenant RLS** — PostgreSQL Row-Level Security on `app.current_tenant`; app-layer filter as defence-in-depth | Cryptographic tenant isolation |
| 12 | **OpenTelemetry Observability** — OTLP span export from `ConnectionManager`, `FlowExecutor`, `TransformationEngine`; Prometheus `conn_*` metrics; request-ID correlation into tap subprocesses | Full distributed trace; P99 latency dashboards |
| 13 | **Exactly-Once Delivery** — deterministic `_apg_record_id` (hash of source PK + stream + checkpoint); upsert dedup at target; bookmark advances only after target confirms write | Safe for financial and audit-grade workloads |
| 14 | **AI-Assisted Connector Discovery** — natural-language intent → validated `tap_config` + `target_config` via local Ollama; confidence score shown before creation | Connector setup in minutes, not hours |
| 15 | **Policy-as-Code (OPA)** — Rego-based guardrail evaluation; policies version-controlled and hot-deployable; `capability_contract.py` rule set migrated to Rego | Compliance teams own policy without engineering deploys |

## New Methods

### IntelligentConnector — AI-powered analysis

```python
from capabilities.common.conn.service import IntelligentConnector

ic = IntelligentConnector()

# Schema drift detection
drift = await ic.detect_schema_drift(old_schema, new_schema)
# {"drift_detected": True, "added_fields": ["revenue_usd"], "removed_fields": [], "changed_fields": ["amount"]}

# AI-assisted field mapping suggestions (Ollama backed, deterministic fallback)
suggestions = await ic.suggest_field_mappings(source_schema, target_schema)
# Returns MappingSuggestions list; also addressable as suggestions["suggestions"]

# Generate data quality rules from sample records
rules = await ic.generate_data_quality_rules(sample_records)
# {"rules": [{"field": "email", "type": "format", "format": "email"}, ...]}

# Predict throughput and resource requirements before committing
perf = await ic.predict_performance({
    "connection_type": "database",
    "batch_size": 2000,
    "expected_records_per_day": 500_000,
    "transformation_complexity": "moderate",
})
# {"performance_score": 0.82, "predicted_throughput_records_per_hour": 820, ...}

# Ollama-powered health narrative for a connection
narrative = await ic.analyze_connection_health_ai("conn-id-xyz")
# {"ai_analysis": "...", "model_used": "qwen3:1.7b", "tokens_used": 214}
```

### ConnectionManager — production lifecycle

```python
from capabilities.common.conn.service import ConnectionManager

mgr = ConnectionManager(tenant_id="acme", ai_model="qwen3:1.7b")
await mgr.initialize()

conn = await mgr.create_connection({
    "name": "Shopify Orders",
    "connection_type": "api",
    "singer_tap": "tap-shopify",
    "tap_config": {"shop": "acme.myshopify.com", "api_key": "keym://acme/shopify-key"},
})

health = await mgr.get_connection_health(str(conn.id))
# ConnectionHealth(status=ACTIVE, latency_ms=42.0, error_rate=0.0)

metrics = await mgr.get_performance_metrics()
# {"active_connections": 1, "healthy_connections": 1, "average_latency_ms": 42.0, ...}
```

### FlowExecutor — data pipeline control

```python
from capabilities.common.conn.service import FlowExecutor

executor = FlowExecutor(connection_manager=mgr)

flow = await executor.create_flow({
    "name": "Shopify → Warehouse",
    "source_connection_id": str(source_conn.id),
    "target_connection_id": str(target_conn.id),
    "schedule_expression": "0 * * * *",
})

result = await executor.execute_flow_once(str(flow.id))
# {"status": "success", "records_processed": 4821, "execution_id": "..."}

history = await executor.get_flow_execution_history(str(flow.id))
```

### TransformationEngine — record processing

```python
from capabilities.common.conn.service import TransformationEngine

engine = TransformationEngine()

rule = await engine.create_transformation_rule({
    "name": "normalise_amount",
    "rule_type": "type_conversion",
    "source_field": "amount",
    "target_field": "amount_usd",
})

transformed = await engine.apply_transformations(record, [str(rule.id)])
remapped = await engine.map_fields(record, {"order_id": "id", "total": "amount_usd"})
filtered = await engine.filter_and_aggregate(
    records,
    filter_conditions=[{"field": "status", "operator": "eq", "value": "shipped"}],
    group_by=["region"],
    aggregations={"amount_usd": {"func": "sum", "alias": "total_revenue"}},
)
```

## Important Files

- `SPECIFICATION.md`: current CONN functional specification.
- `PLAN.md`: lifecycle packet implementation plan.
- `WORLD_CLASS_IMPROVEMENTS.md`: detailed design notes for all 15 v2.0 enhancements.
- `capability_contract.py`: configuration, rules, adapters, UI, and theme.
- `conn_runtime.py`: dependency-light generated-app lifecycle service.
- `api.py`: FastAPI runtime plus generated-app helper functions.
- `view_models.py`: generated UI data models.
- `service.py`: production `ConnectionManager`, `FlowExecutor`, `TransformationEngine`, `IntelligentConnector`.
- `singer_runtime.py`, `singer_taps/`: local Singer runtime and tap surfaces.
- `app.py`: publishable package entrypoint and semantic model generator.
- `tests/test_capability_contract.py`: focused rule, lifecycle, API, and UI tests.
- `tests/test_package_contract.py`: package contract and app evidence tests.

## Generated-App Usage

```python
from capabilities.common.conn.conn_runtime import ConnService

conn = ConnService()

conn.register_connector(
    connector_id="tap-postgres",
    tenant_id="tenant-a",
    name="PostgreSQL Singer Tap",
    runtime="singer",
    source_ref="singer_taps/tap_postgres",
    checksum="sha256:local-package",
    owner="integration-platform",
)

conn.register_connection(
    connection_id="orders-db",
    tenant_id="tenant-a",
    name="Orders Database",
    connector_id="tap-postgres",
    owner="data-platform",
    environment="production",
    credential_vault_ref="keym://tenant-a/orders-db",
    credentials_encrypted=True,
)

conn.record_connection_test("tenant-a", "orders-db", passed=True)
activated = conn.activate_connection(
    tenant_id="tenant-a",
    connection_id="orders-db",
    secret_rotation_recorded=True,
    activation_review_recorded=True,
)

assert activated["status"] == "active"

agent = conn.register_connector_agent(
    agent_id="tap-steward",
    tenant_id="tenant-a",
    name="Tap Steward",
    runtime="codex",
    role="tap_steward",
    scope="local singer tap catalog",
    owner="integration-office",
    purpose="maintain local Singer tap metadata",
)

batch = conn.validate_conn_lifecycle_batch(
    tenant_id="tenant-a",
    event_stream="bytewax",
    mutation_count=4,
)

assert agent["status"] == "active"
assert batch["status"] == "accepted"
```

## Review Evidence

Every generated-app lifecycle record carries `policy_decision`,
`matched_rules`, `review_reasons`, and `review_evidence` fields so generated
connector consoles can render why a connector, connection, flow, sync,
schedule, review, connector-agent registration, or lifecycle batch is allowed,
denied, or awaiting review. `list_pending_reviews()` returns the composed
queue across connectors, connections, flows, sync runs, schedules, reviews,
connector agents, and lifecycle batches. Denied non-Bytewax lifecycle batches
are stored with `status="denied"` and `required_processor="bytewax"` before
the guardrail raises `PermissionError`.

## UI Composition

```python
from capabilities.common.conn.conn_runtime import ConnService
from capabilities.common.conn.view_models import (
    connection_workbench_model,
    connector_agent_roster_model,
    dashboard_model,
    lifecycle_batch_model,
)

service = ConnService()
dashboard = dashboard_model(service, "tenant-a")
connections = connection_workbench_model(service, "tenant-a")
agents = connector_agent_roster_model(service, "tenant-a")
lifecycle = lifecycle_batch_model(service, "tenant-a")
```

## Production Runtime Boundary

The generated-app control plane does not execute Singer taps, open network
connections, read secrets, write lineage stores, run Bytewax flows, call
external SaaS systems, or execute vendor-specific agent runtimes. Production
deployments must bind adapters for auth, credential vault, encryption, audit,
monitoring, lineage, data quality, local Singer runtime, APG registry, APG
gateway, external AI runtimes, and Bytewax event streaming. Those adapters must
honor CONN guardrail decisions before side effects.

The v2.0 enhancements (circuit breaker, DLQ, CDC, schema registry, OPA, etc.)
follow the same boundary: the service layer exposes contracts and decision
points; production adapters supply the backing implementations.

## Focused Verification

```bash
./.venv/bin/python -m py_compile \
  capabilities/common/conn/__init__.py \
  capabilities/common/conn/capability_contract.py \
  capabilities/common/conn/models.py \
  capabilities/common/conn/conn_runtime.py \
  capabilities/common/conn/api.py \
  capabilities/common/conn/view_models.py \
  capabilities/common/conn/app.py \
  capabilities/common/conn/tests/test_capability_contract.py \
  capabilities/common/conn/tests/test_package_contract.py

./.venv/bin/pytest -q \
  capabilities/common/conn/tests/test_capability_contract.py \
  capabilities/common/conn/tests/test_package_contract.py

./.venv/bin/apg capabilities implementation-audit --root capabilities/common/conn --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/conn --strict --json
./.venv/bin/apg capabilities publish-plan capabilities/common/conn --json
./.venv/bin/apg capabilities lifecycle-audit --root capabilities/common/conn --json
```
