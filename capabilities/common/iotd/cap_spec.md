# IoT Device Integration Capability Specification

- **Capability Name**: IoT Device Integration
- **Capability ID**: `iotd`
- **Category**: common
- **Version**: 1.0.0

## Purpose

IOTD gives generated APG applications an executable IoT operations control
plane. It manages tenant-scoped device identities, fleets, encrypted telemetry
events, governed command dispatch, signed firmware artifacts, firmware
deployments, audit events, health reports, UI metadata, and theme metadata
without requiring live device brokers or edge infrastructure during local APG
generation.

The package keeps live device brokers, MQTT/OPC gateways, hardware security
modules, certificate stores, command transports, and firmware delivery networks
behind adapter boundaries. The in-process runtime is the deterministic baseline
that APG examples, generated applications, and package publish plans can
execute today.

## Provided Services

- `device_registry`
- `telemetry_ingestion`
- `command_dispatch`
- `firmware_lifecycle`
- `device_security`
- `device_health`
- `device_audit`

## Required Services

- `tenant_context`
- `mqeb`
- `auth`
- `encr`
- Optional: `edge`, `dtwn`, `logt`, `moni`

## Runtime Surfaces

- `models.py` defines device, telemetry, command, firmware, deployment, audit,
  and health-report records.
- `device_runtime.py` provides telemetry schema validation, device freshness,
  and health-summary helpers.
- `service.py` provides tenant-aware device registration, telemetry ingestion,
  command dispatch and acknowledgement, firmware registration and deployment,
  stale-device queues, health reporting, compatibility record creation,
  dashboard summaries, and rule enforcement.
- `api.py` exposes dependency-light helpers that mirror generated APG endpoint
  handlers.
- `views.py` exposes dashboard, device console, telemetry monitor, command
  center, firmware manager, security, and rule view models.

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. The default configuration requires device
identity, owners, encrypted telemetry, event-bus routing, telemetry schema
validation, command approvals for dangerous actions, command audit trails,
signed firmware artifacts, tenant context, RBAC-ready devices, and compact IoT
operations UI surfaces.

## Rules And Guardrails

IOTD evaluates deterministic contract rules and service-level guardrails:

- `tenant_context_required`
- `device_requires_identity`
- `telemetry_requires_encryption`
- `dangerous_command_requires_approval`
- `firmware_requires_signature`
- `stale_device_requires_review`
- `device_owner_required`
- `event_bus_required`
- `telemetry_schema_invalid`
- `device_missing`
- `command_missing`
- `firmware_missing`

Service methods raise explicit errors when a guardrail blocks an operation, so
capability users can test negative cases without external devices or brokers.

## UI And Theme

The package exposes eight APG Python route contracts:

- `/iotd/dashboard`
- `/iotd/devices`
- `/iotd/telemetry`
- `/iotd/commands`
- `/iotd/firmware`
- `/iotd/security`
- `/iotd/rules`
- `/iotd/settings`

The theme contract is `iotd_device_ops` with compact device cards, telemetry
stream tables, approval consoles, and firmware rollout lanes.

## Adapter Boundaries

The current package intentionally does not open network connections, send
commands to live devices, subscribe to production telemetry, store live private
keys, or deploy firmware to real fleets. Those concerns belong behind explicit
adapters that can be verified independently:

- MQTT, OPC-UA, Modbus, and gateway adapters.
- APG event-bus adapters.
- Device certificate and hardware-security adapters.
- Command transport and acknowledgement adapters.
- Firmware artifact store and rollout adapters.
- Monitoring and audit sink adapters.

## Focused Verification

Use focused verification for IOTD changes:

```bash
./.venv/bin/python -m py_compile capabilities/common/iotd/__init__.py capabilities/common/iotd/models.py capabilities/common/iotd/device_runtime.py capabilities/common/iotd/service.py capabilities/common/iotd/api.py capabilities/common/iotd/views.py capabilities/common/iotd/test_capability_contract.py
./.venv/bin/pytest -q capabilities/common/iotd/test_capability_contract.py capabilities/common/iotd/tests
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/iotd --json
./.venv/bin/apg capabilities publish-plan capabilities/common/iotd --json
```
