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
- Deterministic rule decisions that return `allow`, `deny`, or
  `require_review`.
- UI view models for dashboard, connectors, connections, visual design, sync
  monitoring, quality, lineage, marketplace, security, audit, rules, and
  settings, plus connector-agent roster and lifecycle-batch monitor surfaces.
- Contract-derived semantic model, package manifest, and release evidence.

## Important Files

- `SPECIFICATION.md`: current CONN functional specification.
- `PLAN.md`: lifecycle packet implementation plan.
- `capability_contract.py`: configuration, rules, adapters, UI, and theme.
- `conn_runtime.py`: dependency-light generated-app lifecycle service.
- `api.py`: FastAPI runtime plus generated-app helper functions.
- `view_models.py`: generated UI data models.
- `service.py`: production-oriented connection manager and flow executor.
- `singer_runtime.py`, `singer_taps/`: local Singer runtime and tap surfaces.
- `app.py`: publishable package entrypoint and semantic model generator.
- `tests/test_capability_contract.py`: focused rule, lifecycle, API, and UI
  tests.
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
./.venv/bin/apg capabilities publish-plan capabilities/common/conn --json
```
