# APG CONF Capability Specification

## Purpose

`conf` is the APG Configuration Management capability. It provides the foundation configuration control plane that generated APG applications and other capabilities use to define, validate, approve, deploy, monitor, and remediate configuration safely across tenants and environments.

The capability must be executable without importing the full production automation stack. Advanced adapters for GitOps, AI optimization, security services, cloud deployment, edge orchestration, and collaboration remain valid, but the package contract must expose a small, deterministic lifecycle that generated applications can compose immediately.

## Capability Outcomes

- Maintain tenant-isolated configuration records with owner, environment, secret, validation, and version evidence.
- Require validation before configuration changes can be promoted.
- Require independent approval before production changes can be deployed.
- Require encrypted handling for secret-bearing configuration.
- Require rollback evidence for production deployments.
- Track drift findings and require a remediation plan plus independent review before drift remediation is executed.
- Expose rule, API, UI, theme, audit, and semantic model evidence for APG composition.

## First-Class Domain Concepts

### Configuration Record

A tenant-scoped configuration item controlled by CONF.

Required evidence:

- `id`
- `tenant_id`
- `key`
- `environment`
- `owner`
- `value`
- `contains_secrets`
- `secrets_encrypted`
- `validation_status`
- `version`
- `status`

### Configuration Change

A requested modification to a configuration record.

Required evidence:

- `id`
- `tenant_id`
- `record_id`
- `target_environment`
- `requested_by`
- `summary`
- `proposed_value`
- `validation_passed`
- `contains_secrets`
- `secrets_encrypted`
- `rollback_plan`
- `status`
- `decision`
- `reviewer`
- `notes`

### Configuration Deployment

The promotion of an approved configuration change into a target environment.

Required evidence:

- `id`
- `tenant_id`
- `change_id`
- `record_id`
- `target_environment`
- `requested_by`
- `strategy`
- `status`
- `rollback_plan`
- `applied_version`

### Drift Remediation

The detection and governed remediation of configuration drift.

Required evidence:

- `id`
- `tenant_id`
- `record_id`
- `detected_by`
- `drift_summary`
- `remediation_plan`
- `status`
- `decision`
- `reviewer`
- `notes`

### Audit Event

An immutable package-local event emitted for lifecycle actions.

Required evidence:

- `id`
- `tenant_id`
- `subject_id`
- `event_type`
- `actor`
- `decision`
- `reasons`
- `metadata`

## Lifecycle Requirements

### Record Creation

- A tenant context is required.
- A non-empty configuration key is required.
- A non-empty owner is required.
- Secret-bearing values must be marked encrypted.
- The initial record version starts at `1`.
- Duplicate record IDs are rejected within the same tenant but allowed across tenants.

### Change Request

- The target record must exist in the same tenant.
- The requester and summary are required.
- Validation evidence must be captured.
- Secret-bearing changes must include encrypted secret evidence.
- Production changes may be requested before approval but cannot deploy until approved.

### Change Decision

- A reviewer is required.
- Reviewer notes are required.
- The reviewer must differ from the requester.
- Decisions are limited to `approved` and `rejected`.
- Already-decided changes cannot be decided again.

### Deployment

- The change must exist in the same tenant.
- Failed validation blocks deployment.
- Secret-bearing unencrypted changes block deployment.
- Production deployment requires matching approved change state; caller-supplied approval booleans are not trusted.
- Production deployment requires a rollback plan.
- Successful deployment increments the configuration record version and updates the record value.

### Drift Remediation

- Drift findings require a target record in the same tenant.
- Drift findings require a remediation plan before approval.
- Drift approval requires independent reviewer and notes.
- Approved remediation changes the record status back to `active`.
- Rejected remediation leaves the finding as rejected and does not mutate the record.

## Rules

The deterministic rule engine must enforce at least:

- `tenant_context_required`
- `configuration_record_requires_owner`
- `validate_before_apply`
- `encrypted_secrets_required`
- `production_changes_require_approval`
- `production_deployment_requires_rollback`
- `change_review_requires_independent_reviewer`
- `drift_requires_remediation_plan`
- `drift_review_requires_independent_reviewer`

## UI Surfaces

CONF must expose routes and theme components for:

- Dashboard
- Configuration resource catalog
- Change request queue
- Change approval queue
- Deployment center
- Drift console
- Drift remediation queue
- Policy workbench
- GitOps center
- Audit console
- Settings

## Adapter Boundaries

The executable package must not require live cloud, GitOps, database, AI, Flask-AppBuilder, or security framework services to satisfy its package contract. Those systems are adapters behind the lifecycle.

Production adapters must preserve the same guardrails:

- Do not deploy production changes without approved CONF change state.
- Do not accept caller booleans as approval evidence.
- Do not store or deploy unencrypted secrets.
- Do not remediate drift without remediation plan and independent review.
- Do not mutate cross-tenant records.

## Focused Proof

Battery-conscious proof for this slice:

```bash
./.venv/bin/python -m py_compile capabilities/common/conf/__init__.py capabilities/common/conf/models.py capabilities/common/conf/service.py capabilities/common/conf/api.py capabilities/common/conf/views.py capabilities/common/conf/capability_contract.py capabilities/common/conf/app.py capabilities/common/conf/tests/test_capability_contract.py capabilities/common/conf/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/conf/tests/test_capability_contract.py capabilities/common/conf/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/conf --json
./.venv/bin/apg capabilities publish-plan capabilities/common/conf --json
```
