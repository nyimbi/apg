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

- Tenant-scoped virtual source registration.
- Source owner, supported-type, credential-vault, and encrypted-connection
  checks.
- Source activation approval workflow.
- Schema refresh review workflow for stale source schemas.
- Virtual table publication with owner and classification requirements.
- Federated read-query guardrails for parameterization, write blocking, RBAC,
  lineage, sensitive result caching, cost review, cross-source join review, and
  result limits.
- Query cache lifecycle decisions with TTL enforcement.
- Virtualization policy review records.
- Source retirement impact review.
- First-class virtualization-agent records for AI and automation tools.
- Bytewax lifecycle-batch validation before generated applications apply
  batched DVRL state changes.
- Audit events for every lifecycle decision.
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
  federation runtime.
- `view_models.py`: generated UI data models for APG shells.
- `app.py`: publishable package entrypoint and semantic model generator.
- `test_capability_contract.py`: focused lifecycle, rule, and UI contract
  regression tests.
- `tests/test_package_contract.py`: publishable package contract tests.

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
