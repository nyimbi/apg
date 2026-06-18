# APG Data Virtualization (DVRL)

DVRL is APG's governed data virtualization capability. It gives composed APG
applications a first-class way to register virtual data sources, review schemas,
publish virtual tables, evaluate federated read-query requests, manage cache
decisions, change virtualization policies, retire sources, and expose the
resulting control plane through generated UI models.

DVRL is intentionally split into two layers:

- `DVRLLifecycleService`: a dependency-light lifecycle and guardrail service for
  generated APG applications, package evidence, and fast local tests.
- `DVRLService`: the production-oriented federation runtime for physical
  connectors, query parsing, connector orchestration, cache metadata, NLP
  assistance, Singer integration, APG service integration, and deployment
  adapters.

## What DVRL Provides

- Tenant-scoped virtual source registration with credential-vault and
  encrypted-connection enforcement.
- Source activation approval workflow with durable policy evidence.
- Schema refresh review workflow with drift detection for stale source schemas.
- Virtual table publication with owner, classification, column masking, and
  row-level security filters.
- Federated read-query guardrails: parameterization, write blocking, RBAC,
  lineage capture, sensitive result caching, cost review, cross-source join
  review, and result limits.
- Query push-down optimization for predicate, aggregation, and sort pushdown.
- Named virtual joins between virtual tables (inner/left/right/full/cross).
- Semantic layer definition: metric and dimension mappings over virtual sources.
- Fine-grained access policies scoped to subjects and resource patterns.
- Named caching strategies with TTL, cache level, and invalidation policy.
- Column-level lineage capture (tables, columns, transformation steps).
- Query cache lifecycle decisions with TTL enforcement.
- Virtualization policy review records.
- Source retirement impact review.
- Data product publication backed by virtual sources.
- Compliance reports (GDPR, SOC 2, PCI-DSS coverage metrics).
- First-class virtualization-agent records for AI and automation tools.
- Bytewax lifecycle-batch validation before generated applications apply
  batched DVRL state changes.
- Durable review evidence on reviewable records: `policy_decision`,
  `matched_rules`, `review_reasons`, and `review_evidence`.
- Pending-review queues for sources, schemas, virtual tables, queries, caches,
  policies, virtualization agents, and lifecycle batches.
- Audit events for every lifecycle decision.
- Bulk operations: `bulk_register_sources`, `bulk_publish_virtual_tables`.
- Export in JSON or CSV: sources, queries, audit events.
- Health check endpoint returning per-tenant service status.
- Virtualization analytics KPIs: denial rate, avg query cost, cache count,
  semantic layer count, lineage record count.
- Generated UI view models for dashboard, source manager, schema browser,
  virtual table catalog, query workbench, federation map, cache console,
  policies, metrics, adapter health, agent roster, lifecycle batch monitor,
  audit timeline, and settings.
- Contract-derived semantic model and publishable package metadata.

## Important Files

- `SPECIFICATION.md`: full current functional specification.
- `PLAN.md`: implementation plan and verification strategy for this capability
  packet.
- `capability_contract.py`: executable configuration, rules, UI manifest, and
  theme contract.
- `service.py`: generated-app lifecycle service plus the existing production
  federation runtime (42+ methods).
- `view_models.py`: generated UI data models for APG shells.
- `app.py`: publishable package entrypoint and semantic model generator.
- `test_capability_contract.py`: focused lifecycle, rule, and UI contract
  regression tests.
- `tests/test_package_contract.py`: publishable package contract tests.

## Core API

| Method | Purpose |
|---|---|
| `register_source(...)` | Register a tenant-scoped virtual data source |
| `activate_source(...)` | Approve a source for active use |
| `refresh_schema(...)` | Refresh source schema with review workflow |
| `publish_virtual_table(...)` | Publish a virtual table with classification |
| `execute_query(...)` | Full federated query with all guardrails |
| `query_virtual(...)` | Simplified query entry point with safe defaults |
| `query_federation(...)` | Multi-source federated query with federation defaults |
| `cache_result(...)` | Record a cache decision with TTL |
| `change_policy(...)` | Change a virtualization policy with review |
| `retire_source(...)` | Retire a source with impact review |
| `register_virtualization_agent(...)` | Register an AI/automation agent |
| `validate_dvrl_lifecycle_batch(...)` | Validate a Bytewax lifecycle batch |
| `virtual_table_create(...)` | Create virtual table with full column spec |
| `schema_unify(...)` | Merge multiple source schemas into a unified schema |
| `semantic_layer(...)` | Define metric/dimension semantic layer |
| `semantic_map(...)` | Alias for `semantic_layer` |
| `access_policy(...)` | Define fine-grained access policy |
| `data_lineage(...)` | Capture query lineage (tables, columns, transforms) |
| `caching_strategy(...)` | Define a named caching strategy |
| `push_down_optimise(...)` | Analyze and push predicates to source systems |
| `federation_config(...)` | Configure federation join strategy and parallelism |
| `source_catalog(...)` | Structured catalog of all tenant sources |
| `data_preview(...)` | Synthetic row preview for a virtual table |
| `virtual_join(...)` | Define a named virtual join between two tables |
| `column_masking(...)` | Apply column-level masking to a virtual table |
| `row_filter(...)` | Apply row-level security filter to a virtual table |
| `data_product_publish(...)` | Publish a named data product over virtual sources |
| `virtual_view_refresh(...)` | Refresh metadata and column stats of a virtual table |
| `compliance_report(...)` | Generate data governance compliance report |
| `virtualisation_analytics(...)` | KPIs: denial rate, avg cost, cache/lineage counts |
| `health_check(...)` | Service health for the DVRL capability |
| `bulk_register_sources(...)` | Register multiple sources in one call |
| `bulk_publish_virtual_tables(...)` | Publish multiple virtual tables in one call |
| `export_sources(...)` | Export source records as JSON or CSV |
| `export_queries(...)` | Export query records as JSON or CSV |
| `export_audit_events(...)` | Export audit events as JSON or CSV |
| `dashboard_summary(...)` | Tenant dashboard counts across all record types |
| `list_records(...)` | List records by type for a tenant |
| `list_pending_reviews(...)` | All pending-review items across record types |

## World-Class Enhancements (v2.0)

The following 15 production-grade improvements are specified in
`WORLD_CLASS_IMPROVEMENTS.md` and drive the v2.0 architecture:

1. **Async-First Service Layer** — Full `async` mirror of every core method
   using `asyncio.gather` for concurrent multi-source fetches; 10–50x
   throughput on I/O-bound federated queries.

2. **Distributed Query Plan Optimizer** — Cost-based optimizer that inspects
   per-connector capability flags (`supports_predicate_pushdown`,
   `supports_aggregation`, `max_rows_before_scan`), estimates selectivity from
   column stats, and emits an ordered execution plan with per-step cost bounds.

3. **Columnar Result Streaming with Backpressure** — Async generator yielding
   Apache Arrow `RecordBatch` chunks instead of materializing `list[dict]`; enables
   unlimited result sizes and zero-copy handoff to DuckDB, Polars, or Bytewax.

4. **Schema Change Detection and Drift Alerts** — On every `refresh_schema`,
   diff the new schema against the stored previous version and emit
   `schema.drift_detected` audit events with typed diff payloads
   (`column_added`, `column_dropped`, `type_changed`).

5. **Column-Level Lineage Graph** — Lineage stored as a DAG keyed on
   `(source_column, virtual_column)` edges; `get_lineage_subgraph(tenant_id,
   column_fqn)` returns upstream/downstream columns transitively for
   GDPR Article 30 and SOC 2 CC6 impact analysis.

6. **Tenant-Isolated Credential Vault Integration** — `CredentialVaultAdapter`
   abstract interface; registration validates vault lookup before persisting;
   clear extension point for HashiCorp Vault, AWS Secrets Manager, and GCP
   Secret Manager.

7. **Query Result Diff for Incremental Caching** — Deterministic result-set
   hash on cache miss; stale-while-revalidate semantics; `cache.result_drifted`
   event when hash delta exceeds configurable threshold.

8. **Policy-as-Code with OPA Integration** — `ExternalPolicyEngine` adapter
   proxying rule evaluation to Open Policy Agent via REST; falls back to
   built-in rules when OPA is unavailable; hot-reloadable policy bundles via
   GitOps.

9. **Federated Query Audit with Cryptographic Non-Repudiation** — SHA-256
   chained digest per audit event (`hash(payload + prev_hash)`); chain root
   published to an append-only sink (PostgreSQL, object store, or Bytewax); meets
   PCI-DSS 10.5 and HIPAA §164.312.

10. **Multi-Tenant Namespace Isolation with Rate Limiting** — `DVRLTenantRegistry`
    with per-tenant `DVRLLifecycleService` instances and dedicated resource
    budgets (max concurrent queries, max audit events, max cache entries);
    `TenantQuotaExceededError` on budget breach.

11. **Semantic Layer Query Translation** — `semantic_query(tenant_id, layer_id,
    metrics, dimensions, filters)` translates business-level references to SQL
    using layer definitions and routes through `execute_query` with full
    guardrail coverage.

12. **Connection Pool Health and Circuit Breaker** — Per-connector circuit
    breaker: after N consecutive failures within a rolling window, breaker opens
    with `ConnectorUnavailableError` and a `retry_after` timestamp; half-open
    probe on a background task.

13. **Data Contract Enforcement on Publish** — `DataContract` with required
    columns, type assertions, and nullability constraints; `publish_virtual_table`
    raises `ContractViolationError` with field-level diff when the schema does
    not satisfy the contract.

14. **Async Batch Lineage Import from dbt Artifacts** — `import_dbt_lineage(
    tenant_id, manifest_path, catalog_path, actor)` parses dbt `manifest.json`
    and `catalog.json` and bulk-inserts lineage DAG edges, mapping dbt models
    to registered virtual tables by name.

15. **Real-Time Virtualisation Metrics with OpenTelemetry Export** — Every
    service method instrumented with OpenTelemetry spans and counters; Prometheus
    exposition format endpoint; span attributes include tenant ID, operation name,
    decision, and matched rules.

## New Methods

### Semantic Layer

```python
layer = service.semantic_layer(
    tenant_id="tenant-a",
    layer_id="revenue-layer",
    name="Revenue Metrics",
    source_ids=["orders-wh"],
    metric_definitions={
        "total_revenue": {"aggregation": "sum", "column": "order_value"},
        "order_count": {"aggregation": "count", "column": "order_id"},
    },
    dimension_definitions={
        "region": {"column": "billing_region"},
        "product_line": {"column": "product_category"},
    },
    owner="analytics",
)
assert layer.status == "active"
```

### Column Masking and Row-Level Security

```python
# Mask PII columns
service.column_masking(
    tenant_id="tenant-a",
    table_id="orders",
    columns_to_mask=["customer_email", "phone_number"],
    masking_rule="hash",        # hash | nullify | truncate | redact | tokenise
    actor="data-governance",
)

# Row-level filter for regional isolation
service.row_filter(
    tenant_id="tenant-a",
    table_id="orders",
    filter_id="region-filter",
    filter_expression="billing_region = :user_region",
    applies_to_subjects=["analyst-emea"],
    actor="data-governance",
)
```

### Push-Down Optimization

```python
query = service.query_virtual(
    tenant_id="tenant-a",
    query_id="q-001",
    sql="SELECT region, SUM(revenue) FROM orders WHERE status='closed' GROUP BY region ORDER BY region",
    actor="analyst",
    source_ids=["orders-wh"],
)

result = service.push_down_optimise(
    tenant_id="tenant-a",
    query_id="q-001",
    actor="optimizer",
)
# result["optimisations_applied"] == ["predicate_pushdown", "aggregation_pushdown", "sort_pushdown"]
# result["estimated_cost_saving_pct"] == 45.0
```

### Federation Config and Virtual Joins

```python
service.federation_config(
    tenant_id="tenant-a",
    config_id="crm-wh-federation",
    source_ids=["crm-source", "orders-wh"],
    join_strategy="hash_join",
    pushdown_enabled=True,
    max_parallel_queries=8,
    timeout_seconds=45,
    owner="data-platform",
)

service.virtual_join(
    tenant_id="tenant-a",
    join_id="customer-orders-join",
    left_table_id="customers",
    right_table_id="orders",
    join_type="left",
    join_condition="customers.customer_id = orders.customer_id",
    output_columns=["customer_id", "name", "order_value", "order_date"],
    actor="data-platform",
)
```

### Compliance Report and Analytics

```python
report = service.compliance_report("tenant-a", standard="gdpr")
# {
#   "classification_coverage_pct": 100.0,
#   "encryption_coverage_pct": 100.0,
#   "lineage_records_captured": 12,
#   "compliant": True,
#   ...
# }

kpis = service.virtualisation_analytics("tenant-a")
# {
#   "query_denial_rate": 0.02,
#   "avg_query_cost": 42.5,
#   "cache_count": 18,
#   "semantic_layer_count": 3,
#   ...
# }
```

## Generated-App Usage

```python
from capabilities.common.dvrl.service import DVRLLifecycleService

service = DVRLLifecycleService()

source = service.register_source(
    tenant_id="tenant-a",
    source_id="orders-wh",
    name="Orders Warehouse",
    source_type="warehouse",
    owner="data-platform",
    credentials_vaulted=True,
    connection_encrypted=True,
)

service.activate_source(
    tenant_id="tenant-a",
    source_id="orders-wh",
    approver="risk",
    source_approval_recorded=True,
)

service.publish_virtual_table(
    tenant_id="tenant-a",
    table_id="orders",
    source_id="orders-wh",
    name="Orders",
    owner="analytics",
    classification="internal",
    classification_complete=True,
)

query = service.execute_query(
    tenant_id="tenant-a",
    query_id="q-001",
    sql="SELECT * FROM orders WHERE id = :id",
    actor="analyst",
    source_ids=["orders-wh"],
    data_classification="internal",
    rbac_authorized=True,
    parameterized=True,
    write_query=False,
    lineage_capture_enabled=True,
    estimated_query_cost=50.0,
    cost_review_recorded=False,
    join_source_count=1,
    join_review_recorded=False,
    requested_rows=1000,
    result_contains_sensitive_data=False,
    cache_requested=True,
)

assert query.status == "planned"

agent = service.register_virtualization_agent(
    tenant_id="tenant-a",
    agent_id="query-policy-agent",
    name="Query Policy Agent",
    runtime="codex",
    role="query_policy_reviewer",
    scope="restricted federated query policy recommendations",
    owner="data-governance",
    purpose="review query policy changes before publication",
    human_approval_required=True,
)

batch = service.validate_dvrl_lifecycle_batch(
    tenant_id="tenant-a",
    event_stream="bytewax",
    mutation_count=4,
)

assert agent.status == "active"
assert batch.status == "accepted"
```

Privileged agents and reviewable lifecycle events preserve operator-facing
evidence instead of disappearing into transient exceptions:

```python
pending_agent = service.register_virtualization_agent(
    tenant_id="tenant-a",
    agent_id="pending-query-policy-agent",
    name="Pending Query Policy Agent",
    runtime="claude-code",
    role="query-policy-reviewer",
    scope="restricted federated query policy recommendations",
    owner="data-governance",
    purpose="prepare query policy recommendations for approval",
    human_approval_required=False,
)

assert pending_agent.status == "pending_review"
assert pending_agent.policy_decision == "require_review"
assert pending_agent.review_reasons == ["privileged_agent_human_approval_required"]
assert service.list_pending_reviews("tenant-a")
```

## UI Composition

```python
from capabilities.common.dvrl.service import DVRLLifecycleService
from capabilities.common.dvrl.view_models import dashboard_model, source_manager_model

service = DVRLLifecycleService()
dashboard = dashboard_model(service, "tenant-a")
sources = source_manager_model(service, "tenant-a")
```

The generated UI contract is available from:

```python
from capabilities.common.dvrl.capability_contract import get_capability_contract

routes = get_capability_contract("tenant-a")["ui"]["routes"]
theme = get_capability_contract("tenant-a")["theme"]
```

## Production Runtime Boundary

The lifecycle service does not open physical database, SaaS, object-store,
streaming, or Singer tap connections. Production deployments bind these through
the adapter surfaces in `capability_contract.py`:

- connector registry
- query planner
- execution engine
- metadata catalog
- cache store
- credential vault
- audit sink
- Bytewax event stream runtime
- external AI/automation runtimes such as Codex, Claude Code, OpenCode, and Pi

## Focused Verification

Use focused checks while working on battery:

```bash
./.venv/bin/python -m py_compile \
  capabilities/common/dvrl/__init__.py \
  capabilities/common/dvrl/capability_contract.py \
  capabilities/common/dvrl/service.py \
  capabilities/common/dvrl/api.py \
  capabilities/common/dvrl/view_models.py \
  capabilities/common/dvrl/app.py \
  capabilities/common/dvrl/test_capability_contract.py \
  capabilities/common/dvrl/tests/test_package_contract.py

./.venv/bin/pytest -q \
  capabilities/common/dvrl/test_capability_contract.py \
  capabilities/common/dvrl/tests/test_package_contract.py

./.venv/bin/apg capabilities implementation-audit --root capabilities/common/dvrl --json
./.venv/bin/apg capabilities publish-plan capabilities/common/dvrl --json
```

Full repository tests, live connector tests, rendered browser checks,
performance benchmarks, and production adapter exercises should be run in a
dedicated verification window.
