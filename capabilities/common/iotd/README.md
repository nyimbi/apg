# IOTD IoT Device Integration Capability

IOTD provides APG applications with a tenant-scoped device-operations runtime:
device identity, certificate ownership, fleet grouping, encrypted telemetry
ingestion, governed command dispatch, command acknowledgement, signed firmware
registration, firmware deployment, stale-device review, health reporting,
device-operation agents, UI metadata, theme tokens, audit evidence, and
Bytewax-backed lifecycle events.

The package stays dependency-light. Production device brokers, edge runtimes,
certificate authorities, audit sinks, digital twins, observability systems, and
Bytewax workers are represented as APG adapters in the executable contract and
are bound by the host application.

## What It Provides

- Device registry with tenant, owner, certificate, fleet, status, last-seen,
  and metadata fields.
- Telemetry ingestion with encryption, schema validation, and Bytewax event
  stream enforcement.
- Command center with dangerous-command approval and command acknowledgement.
- Firmware lifecycle with signed artifacts, target-device validation, and
  deployment records.
- Health reporting for online/offline devices, stale devices, pending commands,
  and unsigned firmware risk.
- First-class AI device-operation agents with runtime, role, scope,
  registration, and contribution-disclosure guardrails.
- Audit events for device, telemetry, command, firmware, health, and agent
  lifecycle changes.
- UI route, API, view-model, theme, semantic-model, package-manifest, and
  release-report evidence.

## Main Files

- `SPECIFICATION.md` defines the normative capability behavior.
- `PLAN.md` records the implementation packet plan.
- `capability_contract.py` is the executable source of configuration, rules,
  routes, theme, adapters, provides/requires, and Bytewax stream metadata.
- `models.py` defines tenant-scoped device, telemetry, command, firmware,
  health, audit, and agent records.
- `device_runtime.py` contains dependency-light schema, freshness, and health
  helpers.
- `service.py` implements the runtime facade.
- `api.py` exposes package-safe function helpers.
- `views.py` exposes UI view models.
- `test_capability_contract.py` proves lifecycle behavior and generated
  evidence.

## Basic Usage

```python
from capabilities.common.iotd import IotdService

service = IotdService()
service.register_device(
    device_id="device-1",
    tenant_id="tenant-demo",
    device_key="device-key-1",
    owner_id="ops-owner",
    certificate_id="cert-1",
    fleet_id="line-a",
)
service.ingest_telemetry(
    event_id="event-1",
    tenant_id="tenant-demo",
    device_id="device-1",
    schema_name="temperature",
    payload={"timestamp": "now", "temperature": 42.5},
)
service.dispatch_command(
    command_id="command-1",
    tenant_id="tenant-demo",
    device_id="device-1",
    command="restart",
    dangerous=True,
    approval_id="approval-1",
)
```

## AI Device-Operation Agents

Register AI agents before they assist with fleet operations:

```python
agent = service.register_iotd_agent(
    tenant_id="tenant-demo",
    name="Fleet reviewer",
    runtime="codex",
    role="fleet_operator",
    scope="Review telemetry anomalies and command risk before shift handoff",
)
```

Supported runtimes are `codex`, `claude_code`, `opencode`, and `pi`. Supported
roles cover fleet operations, telemetry review, command review, firmware
review, and security review.

## Composition

IOTD composes with:

- `auth` for identity, permissions, and device RBAC.
- `encr` for telemetry and device credential protection.
- `audl` for durable audit events.
- `conf` for tenant device and telemetry policy.
- `edge` for edge runtime deployment.
- `dtwn` for digital-twin synchronization.
- `logt` and `moni` for logs and monitoring.

Batch device mutation and telemetry pipelines must use the `bytewax`
event-stream adapter.

## Verification

Focused verification for this packet:

```bash
./.venv/bin/python -m py_compile capabilities/common/iotd/__init__.py capabilities/common/iotd/capability_contract.py capabilities/common/iotd/models.py capabilities/common/iotd/device_runtime.py capabilities/common/iotd/service.py capabilities/common/iotd/api.py capabilities/common/iotd/views.py capabilities/common/iotd/app.py capabilities/common/iotd/test_capability_contract.py
./.venv/bin/pytest -q capabilities/common/iotd/test_capability_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/iotd --json
./.venv/bin/apg capabilities publish-plan capabilities/common/iotd --json
```

Live device brokers, edge gateways, certificate authorities, durable audit
stores, rendered UI, and Bytewax workers are integration concerns outside the
package proof.
