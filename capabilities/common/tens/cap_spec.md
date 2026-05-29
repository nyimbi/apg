# Tenants Legacy Capability Specification

- **Capability Name**: Tenants Legacy
- **Capability ID**: `tens`
- **Category**: common
- **Version**: 1.0.0

## Purpose

This package provides the executable APG runtime for `tens`.
It gives composed applications a deterministic legacy tenant migration surface
for legacy tenant registration, tenant mapping, access-boundary validation,
migration approval and completion, deprecation planning, audit events, UI route
metadata, semantic-model publication, and publish-plan evidence.

## Provided Services

- `legacy_tenant_registry`
- `tenant_mapping_workbench`
- `access_boundary_validation`
- `tenant_migration_queue`
- `tenant_deprecation_governance`
- `legacy_tenant_audit_events`

## Required Services

- `tenant_context`
- `multi_tenant_registry`
- `identity_boundary_validation`
- `legacy_role_mapping`
- `audit_sink`

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `legacy_tenant_requires_owner`
- `mapping_requires_validation`
- `migration_requires_approval`
- `access_boundary_required`
- `stale_legacy_tenant_requires_review`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

The view helpers provide dashboard, legacy tenant registry, mapping workbench,
migration queue, boundary review, deprecation, audit, and settings models.

## Theme

The package uses the `tens_legacy_tenant_migration` APG theme contract.

## Runtime Behavior

`TensService` is intentionally dependency-light so it can run inside generated
applications, tests, and publish-plan probes without external infrastructure.
It supports:

- `register_legacy_tenant()` for source-system, owner, compatibility scope, and
  stale-tenant review tracking.
- `map_tenant()` for validated legacy-to-APG tenant mappings.
- `validate_access_boundary()` for auth boundary, legacy role mapping, tenant
  isolation, and privileged-access review evidence.
- `create_migration_plan()` and `complete_migration()` for approved migration,
  rollback, and post-migration validation proof.
- `record_deprecation_plan()` for deprecation governance.
- `dashboard_summary()` and list helpers for API and UI composition.

## Adapter Boundaries

The in-package runtime stores records in memory by design. Production adapters
are expected to bind the APG multi-tenant registry, identity/RBAC boundary
validation, legacy role mapping, migration execution, deprecation notices, and
audit sinks at the APG composition layer without changing the deterministic
package contract.

## Focused Verification

- `./.venv/bin/python -m py_compile capabilities/common/tens/__init__.py capabilities/common/tens/models.py capabilities/common/tens/tenant_runtime.py capabilities/common/tens/service.py capabilities/common/tens/api.py capabilities/common/tens/views.py capabilities/common/tens/capability_contract.py capabilities/common/tens/app.py capabilities/common/tens/test_capability_contract.py capabilities/common/tens/tests/test_package_contract.py`
- `./.venv/bin/pytest -q capabilities/common/tens/test_capability_contract.py capabilities/common/tens/tests/test_package_contract.py`
- `./.venv/bin/apg capabilities implementation-audit --root capabilities/common/tens --json`
- `./.venv/bin/apg capabilities publish-plan capabilities/common/tens --json`
