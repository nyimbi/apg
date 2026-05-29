# Platform Foundation Capability Specification

- **Capability Name**: Platform Foundation
- **Capability ID**: `plfd`
- **Category**: common
- **Version**: 1.0.0

## Purpose

`plfd` provides executable platform-foundation management for APG. It owns the
local foundation service registry, dependency posture, configuration and tenant
baselines, readiness gates, platform change approvals, governance evidence, UI
view models, and deterministic rule enforcement behind the capability contract.

The package is intentionally dependency-light. It does not operate live cloud
control planes, production identity systems, monitoring stacks, deployment
orchestrators, or audit stores directly. Those systems should be composed
through adapters around `conf`, `mten`, `auth`, `audl`, `moni`, `hlth`, `regy`,
`secu`, and `plgn` when production integrations are verified.

## Provided Services

- `foundation_registry`
- `dependency_posture`
- `configuration_baselines`
- `readiness_gates`
- `platform_governance`
- `plfd_operations`

## Required Services

- `tenant_context`
- `conf`
- `mten`
- `auth`
- `audl`

Optional composition partners include `moni`, `hlth`, `regy`, `secu`, and
`plgn`.

## Runtime Behavior

The service layer exposes an in-memory platform-foundation runtime for package
evidence and generated-application composition:

1. Register tenant-scoped foundation services with owners, tiers, dependency
   declarations, readiness scores, health state, monitoring posture, rollback
   plans, and change-window references.
2. Record service-to-service dependency health and evidence.
3. Attach configuration, tenant, auth, and audit baselines with approval
   evidence.
4. Assess readiness from service score, dependency health, baseline
   completeness, monitoring, rollback, and change-window posture.
5. Propose platform changes with affected capability counts and review posture.
6. Approve platform changes only when tenant context, dependency health,
   platform approval, broad review, security review, change-window, rollback,
   and configuration baseline guardrails pass.
7. Publish dashboard, service registry, dependency map, baseline manager,
   readiness gate, change queue, governance, and audit state.

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context, service ownership, tier
classification, dependency maps, readiness scores, baseline presence, health
gates, monitoring, rollback, change windows, broad review, security review, UI,
and theme metadata are explicit contract concerns.

## Rules

- `tenant_context_required`
- `foundation_service_requires_owner`
- `dependency_health_required`
- `configuration_baseline_required`
- `platform_change_requires_approval`
- `broad_platform_change_requires_review`

The service calls the deterministic rule engine before tenant-sensitive writes,
foundation service registration, and platform change approval. Deny decisions
raise `PermissionError`; review decisions block approval until the required
review evidence is recorded.

## UI

The package exposes 8 APG Python UI route contract(s) through `views.py` and the
package semantic model:

- `/plfd/dashboard`
- `/plfd/services`
- `/plfd/dependencies`
- `/plfd/baselines`
- `/plfd/readiness`
- `/plfd/changes`
- `/plfd/governance`
- `/plfd/settings`

`views.py` provides dashboard, service registry, dependency-map, baseline
manager, readiness-gate, change-queue, and governance view models.

## Theme

The package uses the `plfd_platform_foundation` APG theme contract.

## Runtime Surfaces

| File | Responsibility |
| --- | --- |
| `models.py` | Foundation services, dependencies, baselines, readiness assessments, platform changes, and audit events. |
| `foundation_runtime.py` | Deterministic IDs, tier, health, baseline, readiness, and change-review helpers. |
| `service.py` | Tenant-scoped lifecycle behavior and APG rule enforcement. |
| `api.py` | Dependency-light API helper functions over `PlfdService`. |
| `views.py` | Route metadata and UI view models. |
| `test_capability_contract.py` | Contract, lifecycle, and guardrail tests. |

## Adapter Boundaries

Production integration should remain behind explicit adapters until directly
verified:

- `conf` for configuration source-of-truth synchronization.
- `mten` for tenant lifecycle and isolation policy.
- `auth` for platform permissions and service-owner identity.
- `audl` for durable audit event persistence.
- `moni` and `hlth` for live health, monitoring, and readiness evidence.
- `regy` for service registry publication.
- `secu` for production security-review workflows.
- `plgn` for plugin dependency posture.
- deployment orchestrators for change-window and rollback execution.

## Focused Verification

```bash
./.venv/bin/pytest -q capabilities/common/plfd/test_capability_contract.py capabilities/common/plfd/tests
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/plfd --json
./.venv/bin/apg capabilities publish-plan capabilities/common/plfd --json
```
