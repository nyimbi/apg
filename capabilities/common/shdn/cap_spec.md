# Shutdown and Lifecycle Control Capability Specification

- **Capability Name**: Shutdown and Lifecycle Control
- **Capability ID**: `shdn`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package provides the executable APG runtime for `shdn`.
It gives composed applications a deterministic shutdown and lifecycle-control
surface for service registration, shutdown planning, drain orchestration,
backup and restore-test evidence, guarded shutdown execution, recovery evidence,
audit events, UI route metadata, semantic-model publication, and publish-plan
evidence.

## Provided Services

- `service_lifecycle_registry`
- `shutdown_plan_builder`
- `drain_orchestrator`
- `backup_snapshot_gate`
- `shutdown_execution_gate`
- `recovery_evidence_center`
- `lifecycle_audit_events`

## Required Services

- `tenant_context`
- `monitoring_health_gate`
- `backup_snapshot_store`
- `operator_identity`
- `audit_sink`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `service_requires_owner`
- `shutdown_requires_health_gate`
- `shutdown_requires_backup_snapshot`
- `production_shutdown_requires_approval`
- `force_shutdown_requires_review`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

The view helpers provide dashboard, service console, plan builder, execution
monitor, approval queue, recovery center, audit, and settings models.

## Theme

The package uses the `shdn_lifecycle_control` APG theme contract.

## Runtime Behavior

`ShdnService` is intentionally dependency-light so it can run inside generated
applications, tests, and publish-plan probes without external infrastructure.
It supports:

- `register_service()` for tenant-scoped lifecycle targets with owner,
  environment, dependency, criticality, drain timeout, and health-gate metadata.
- `create_shutdown_plan()` for approved plans with rollback plan, restart
  sequence, maintenance window, and production approval controls.
- `start_drain()` for active-session and queue-depth drain tracking.
- `record_backup_snapshot()` for backup and restore-test evidence.
- `execute_shutdown()` for rule-driven health, snapshot, approval, and
  force-shutdown review gates.
- `record_recovery()` for incident/change evidence and post-shutdown health
  proof.
- `dashboard_summary()` and list helpers for API and UI composition.

## Adapter Boundaries

The in-package runtime stores records in memory by design. Production adapters
are expected to bind tenant context, identity, health monitoring, backup
evidence, change windows, incident/change references, and audit sinks at the APG
composition layer without changing the deterministic package contract.

## Focused Verification

- `./.venv/bin/python -m py_compile capabilities/common/shdn/__init__.py capabilities/common/shdn/models.py capabilities/common/shdn/lifecycle_runtime.py capabilities/common/shdn/service.py capabilities/common/shdn/api.py capabilities/common/shdn/views.py capabilities/common/shdn/capability_contract.py capabilities/common/shdn/app.py capabilities/common/shdn/test_capability_contract.py capabilities/common/shdn/tests/test_package_contract.py`
- `./.venv/bin/pytest -q capabilities/common/shdn/test_capability_contract.py capabilities/common/shdn/tests/test_package_contract.py`
- `./.venv/bin/apg capabilities implementation-audit --root capabilities/common/shdn --json`
- `./.venv/bin/apg capabilities publish-plan capabilities/common/shdn --json`
