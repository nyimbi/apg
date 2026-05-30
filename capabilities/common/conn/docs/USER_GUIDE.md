# CONN User Guide

CONN lets APG applications register connectors, create secured connections,
compose flows, run sync jobs, and keep audit evidence without requiring live
connector execution in generated-app mode.

## Register A Connector

```python
from capabilities.common.conn.conn_runtime import ConnService

service = ConnService()
service.register_connector(
    connector_id="tap-postgres",
    tenant_id="tenant-a",
    name="PostgreSQL Singer Tap",
    runtime="singer",
    source_ref="singer_taps/tap_postgres",
    checksum="sha256:local-package",
    owner="integration-platform",
)
```

Unverified connector sources create a marketplace review record unless review
evidence is already present.

## Register And Activate A Connection

```python
service.register_connection(
    connection_id="orders-db",
    tenant_id="tenant-a",
    name="Orders Database",
    connector_id="tap-postgres",
    owner="data-platform",
    environment="production",
    credential_vault_ref="keym://tenant-a/orders-db",
    credentials_encrypted=True,
)

service.record_connection_test("tenant-a", "orders-db", passed=True)
service.activate_connection(
    "tenant-a",
    "orders-db",
    secret_rotation_recorded=True,
    activation_review_recorded=True,
)
```

Activation is denied until the connection has a passed test and secret rotation
evidence. Production activation requires review evidence.

## Create A Flow

```python
flow = service.create_flow(
    flow_id="orders-to-warehouse",
    tenant_id="tenant-a",
    name="Orders to Warehouse",
    source_connection_id="orders-db",
    target_connection_id="warehouse",
    owner="data-platform",
    mapping_ref="maps/orders-to-warehouse.json",
    quality_gate_ref="quality/orders",
)
```

Flows require active source and target connections, field mapping evidence,
lineage, and a quality gate.

## Start, Schedule, And Replay Sync

```python
run = service.start_sync(
    run_id="sync-001",
    tenant_id="tenant-a",
    flow_id="orders-to-warehouse",
    batch_size=5000,
    monitoring_enabled=True,
)

service.schedule_flow("tenant-a", "orders-nightly", "orders-to-warehouse", "0 1 * * *", "Africa/Nairobi")
service.replay_sync("tenant-a", "sync-001", "sync-001-replay", "idem-001")
```

Large batches require monitoring, schema changes can require review, schedules
require a timezone, and replay requires idempotency evidence.
