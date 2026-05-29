# APG MTEN Capability Specification

## Purpose

`mten` is the APG Multi-Tenant Management capability. It provides the tenant
control plane for generated APG applications: tenant provisioning, tenant
activation, tenant isolation, custom domain validation, capacity governance,
suspension, live migration governance, resource posture, UI composition, and
audit evidence.

The capability must remain executable without starting cloud providers, AI
engines, analytics engines, FastAPI, Flask-AppBuilder, deployment bundles, or
production databases. Those systems are adapters behind the package contract.

## Capability Outcomes

- Register tenant environments with tenant-qualified state.
- Activate tenants only when DNS and isolation guardrails are satisfied.
- Require capacity approval for high projected compute usage.
- Fail closed on custom domains without DNS validation.
- Suspend tenants on isolation breach evidence.
- Block mutations against suspended tenants until explicit reactivation.
- Require runbook and independent review for live migration.
- Expose API helpers, view models, UI routes, theme components, rules, semantic
  evidence, and focused tests for generated APG applications.

## First-Class Domain Concepts

### Tenant Environment

Tenant runtime state controlled by MTEN.

Required evidence:

- `id`
- `tenant_id`
- `name`
- `owner`
- `tier`
- `primary_domain`
- `custom_domain`
- `dns_validated`
- `projected_compute_units`
- `isolation_boundary_encrypted`
- `capacity_approval_id`
- `status`

### Capacity Approval

Human approval for capacity overcommit or high projected usage.

Required evidence:

- `id`
- `tenant_id`
- `target_tenant_id`
- `requested_by`
- `projected_compute_units`
- `justification`
- `status`
- `decision`
- `reviewer`
- `notes`

### Isolation Incident

Evidence that an isolation boundary was breached or threatened.

Required evidence:

- `id`
- `tenant_id`
- `target_tenant_id`
- `detected_by`
- `breach_summary`
- `severity`
- `status`
- `suspended`

### Live Migration Request

Governed movement of a tenant between providers or regions.

Required evidence:

- `id`
- `tenant_id`
- `target_tenant_id`
- `requested_by`
- `source_provider`
- `target_provider`
- `runbook`
- `status`
- `decision`
- `reviewer`
- `notes`

### Governance Event

Tenant-scoped evidence event emitted for MTEN decisions.

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

### Tenant Registration

- Management tenant context is required.
- Tenant ID, name, owner, tier, and primary domain are required.
- Custom domain activation requires DNS validation.
- Isolation boundaries must be encrypted.
- High projected compute usage requires approved capacity approval state.
- Duplicate target tenant IDs are rejected within a management tenant but
  allowed across management tenants.

### Capacity Approval

- Requester, target tenant, projected units, and justification are required.
- Reviewer identity and notes are required.
- Reviewer must be independent from requester.
- Decision is limited to `approved` or `rejected`.
- Approved capacity state can be linked to tenant registration or upgrade.

### Activation

- Tenant must be registered in the same management tenant.
- Suspended tenants cannot be activated without reactivation.
- Custom domain activation still requires DNS validation.
- Activation changes status to `active`.

### Isolation Incident And Suspension

- Isolation incidents require detector, target tenant, and breach summary.
- Isolation incident records suspend the target tenant.
- Mutations against suspended tenants are blocked.
- Reactivation requires actor, evidence, and a closed incident posture.

### Live Migration

- Live migration requires active target tenant state.
- Runbook is required before request creation.
- Reviewer identity and notes are required.
- Reviewer must be independent from requester.
- Approved migration records can be executed as a dependency-light evidence
  transition; production movers remain adapters.

## Rules

The deterministic rule engine must enforce at least:

- `tenant_context_required`
- `cross_tenant_access_requires_membership`
- `suspended_tenants_block_mutations`
- `custom_domain_requires_dns_validation`
- `capacity_overcommit_requires_review`
- `capacity_review_requires_independent_reviewer`
- `isolation_boundary_requires_encryption`
- `isolation_breach_requires_suspension`
- `live_migration_requires_runbook`
- `live_migration_requires_independent_reviewer`

## UI Surfaces

MTEN must expose routes and theme components for:

- Dashboard
- Tenant portfolio
- Provisioning pipeline
- Capacity approvals
- Isolation incidents
- Live migrations
- Template catalog
- Analytics hub
- Optimization workbench
- Audit/governance timeline
- Settings

## Adapter Boundaries

The executable package must not require live cloud, DNS, IAM, billing, analytics,
AI, service mesh, FastAPI, Flask-AppBuilder, deployment package, or production
database systems to satisfy its package contract.

Production adapters must preserve the same guardrails:

- Do not activate custom domains without DNS validation.
- Do not exceed capacity threshold without approved capacity state.
- Do not mutate suspended tenants.
- Do not ignore isolation breaches.
- Do not run live migration without runbook and independent review.
- Do not mutate cross-tenant records.

## Focused Proof

Battery-conscious proof for this slice:

```bash
./.venv/bin/python -m py_compile capabilities/common/mten/__init__.py capabilities/common/mten/models.py capabilities/common/mten/mten_runtime.py capabilities/common/mten/api_helpers.py capabilities/common/mten/view_models.py capabilities/common/mten/capability_contract.py capabilities/common/mten/app.py capabilities/common/mten/tests/test_capability_contract.py capabilities/common/mten/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/mten/tests/test_capability_contract.py capabilities/common/mten/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/mten --json
./.venv/bin/apg capabilities publish-plan capabilities/common/mten --json
```
