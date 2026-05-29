# Environment Management Capability Specification

- **Capability Name**: Environment Management
- **Capability ID**: `envm`
- **Category**: common
- **Version**: 1.0.0

## Purpose

ENVM is APG's package-backed environment-management capability. It provides
tenant-scoped environment inventory, stage and region policy enforcement,
production change approval, promotion-path governance, deployment link and
rollback binding, configuration drift review, secret-scope policy checks, audit
events, UI route metadata, semantic-model publication, and publish-plan
evidence.

The package now carries executable runtime behavior instead of generic record
storage: `service.py`, `models.py`, `environment_engine.py`, `api.py`, and
`views.py` manage environment definitions, promotion paths and runs, drift
reports, secret scopes, deterministic environment fingerprints, dashboard
summaries, compatibility helpers, and APG rule enforcement.

## Provided Services

- `envm_operations`

## Required Services

- `tenant_context`
- `depl` for deployment links and rollback execution handoff
- `conf` for declared environment configuration sources
- `auth`/RBAC for environment access and promotion authorization
- `audl` for durable environment-change audit trails
- `keym` or an equivalent credential vault for scoped secret references

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context is required for executable
operations.

## Rules

- `tenant_context_required`
- `environment_requires_owner`
- `production_change_requires_approval`
- `promotion_requires_path`
- `secret_scope_requires_policy`
- `high_drift_requires_review`

## UI

The package exposes 8 APG Python UI route contract(s) through
`views.py` and the package semantic model.

The UI contract covers environment inventory, promotion console, drift
dashboard, secret scopes, environment policy, analytics, settings, and the
overview dashboard.

## Theme

The package uses the `envm_environment_ops` APG theme contract.

## External Runtime Boundary

ENVM keeps live infrastructure, deployment systems, configuration repositories,
secret managers, observability feeds, and access-control providers behind APG
integration boundaries. Capability tests and publish evidence exercise
deterministic package behavior without requiring cloud credentials or mutable
infrastructure. Production deployments can bind ENVM to Kubernetes namespaces,
cloud accounts, deployment controllers, Git configuration repositories, secret
vaults, drift scanners, audit logging, and RBAC providers through APG
configuration and credential-vault services.
