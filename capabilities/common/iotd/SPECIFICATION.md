# IOTD IoT Device Integration Specification

## Purpose

IOTD is APG's common IoT device integration capability. It lets generated and
composed applications register devices, ingest telemetry, dispatch governed
commands, manage signed firmware, monitor fleet health, and operate device
workflows through APG UI and API surfaces.

The capability is designed for executable applications first. It provides a
dependency-light runtime and explicit adapter boundaries so production systems
can connect real gateways, certificate authorities, edge runtimes, digital
twins, and observability services later.

## Capability Identity

- Capability id: `iotd`
- Display name: `IoT Device Integration`
- Category: `common`
- Runtime target: `python`
- Primary service: `service.IotdService`
- UI prefix: `/iotd`
- API prefix: `/iotd/api/v1`
- Event stream processor: `bytewax`

## Provided Services

- `device_registry`
- `telemetry_ingestion`
- `command_dispatch`
- `firmware_lifecycle`
- `device_security`
- `device_health`
- `iotd_agents`

## Required Capabilities

- `auth` for identity, permissions, and RBAC.
- `encr` for telemetry and credential encryption.
- `audl` for durable audit evidence.
- `conf` for tenant configuration.

Optional adapters include `edge`, `dtwn`, `logt`, and `moni`.

## Domain Model

`DeviceIdentity`

- Tenant-local device id, device key, owner, fleet, certificate id, lifecycle
  status, registration time, last-seen time, and metadata.

`TelemetryEvent`

- Tenant-local telemetry event with device id, schema, payload, encryption
  state, event stream, acceptance state, and receipt time.

`DeviceCommand`

- Tenant-local command with parameters, dangerous-command flag, approval id,
  dispatch state, acknowledgement time, and acknowledgement message.

`FirmwareArtifact`

- Tenant-local firmware artifact with version, artifact URI, signature id,
  signature verification, lifecycle state, and registration time.

`FirmwareDeployment`

- Tenant-local deployment that targets a fleet and explicit device ids.

`DeviceHealthReport`

- Health summary for online/offline devices, stale devices, pending commands,
  and unsigned firmware.

`DeviceAuditEvent`

- Audit record for device lifecycle actions.

`IotdAgent`

- Registered AI device-operation agent with tenant, runtime, role, explicit
  scope, registration status, contribution disclosure, and activity state.

## Rule Engine

The deterministic rule engine must enforce:

- tenant context on every device operation;
- provisioned device identity;
- accountable device owner;
- device certificate identity;
- Bytewax event stream for telemetry ingestion;
- telemetry encryption;
- telemetry schema validation;
- command name;
- approval for dangerous commands;
- signed firmware;
- firmware artifact URI;
- target devices for firmware deployment;
- stale-device review;
- registered AI device-operation agent;
- supported AI-agent runtime;
- supported AI-agent role;
- explicit AI-agent scope;
- AI contribution disclosure;
- audit evidence for lifecycle state changes;
- Bytewax event stream for batch IoT mutation.

## UI Contract

The capability exposes these APG Python UI routes:

- `/iotd/dashboard`
- `/iotd/devices`
- `/iotd/telemetry`
- `/iotd/commands`
- `/iotd/firmware`
- `/iotd/agents`
- `/iotd/health`
- `/iotd/security`
- `/iotd/rules`
- `/iotd/audit`
- `/iotd/settings`

View models must expose device registry, telemetry stream, command center,
firmware manager, agent panel, health dashboard, security, rules, and audit
data.

## Theme

The default theme is `iotd_device_ops`. Theme components cover device cards,
telemetry streams, command centers, firmware rollout lanes, agent panels,
health dashboards, and audit timelines.

## Event Stream

Lifecycle telemetry is described by:

- processor: `bytewax`
- topic: `apg.iotd.lifecycle`
- state: devices, telemetry, commands, firmware, deployments, health reports,
  IOTD agents, audit events
- events: device registered, telemetry ingested, command dispatched, command
  acknowledged, firmware registered, firmware deployed, agent registered
- guardrail: `batch_iot_mutation_requires_bytewax`

The package records stream metadata and guardrails. Live Bytewax topology is
an application deployment concern.

## Acceptance Criteria

- The package has local README, specification, plan, contract, runtime, view,
  API, test, package-manifest, semantic-model, and release-report artifacts.
- The contract exposes provides/requires, configuration schema, rules, routes,
  theme, and Bytewax stream metadata.
- The service supports device registration, telemetry ingestion, command
  dispatch and acknowledgement, firmware registration and deployment, health
  reporting, stale-device queue, AI-agent registration, audit events,
  tenant-local IDs, and Bytewax batch mutation validation.
- Focused tests prove the main lifecycle, guardrails, tenant isolation,
  generated evidence, and docs.
- Compile, focused pytest, implementation-audit, publish-plan, marker scan,
  and diff checks pass before commit.
