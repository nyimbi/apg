# Deployment Management Capability Specification

- **Capability Name**: Deployment Management
- **Capability ID**: `depl`
- **Category**: common
- **Version**: 1.0.0

## Purpose

`depl` provides dependency-light release and deployment management for APG
applications. It owns tenant deployment environments, release manifests,
rollback plans, health gates, deployment plans, deployment runs, rollback
events, audit evidence, UI state, and deterministic deployment-governance rule
evaluation.

The package is intentionally local and deterministic. It does not invoke a live
cloud provider, Kubernetes cluster, registry, scanner, secret manager,
observability system, ticketing system, or notification service. Those
integrations should be composed through APG capabilities such as `cicd`, `envm`,
`logt`, `moni`, `ntfy`, `comp`, `auth`, and `audl`.

## Provided Services

- `release_management`
- `deployment_rollouts`
- `health_gates`
- `rollback_control`
- `deployment_audit`

## Required Services

- `logt`
- `moni`
- `hlth`

Optional composition partners include `cicd`, `envm`, `ntfy`, and `comp`.

## Runtime Behavior

The service layer exposes an in-memory deployment runtime for package evidence
and generated-application composition:

1. Register tenant deployment environments with tier, owner, policy, and
   approvers.
2. Create release manifests with owner, artifact digest, signature, manifest,
   and change-ticket evidence.
3. Attach tested rollback plans to release manifests.
4. Record health gates with checks, report references, and log trace links.
5. Create deployment plans for rolling, blue-green, and canary strategies.
6. Hold large canary plans for review when the deterministic rule engine
   requires it.
7. Execute approved deployment plans with deployment fingerprints and evidence.
8. Execute rollback events against deployed runs.
9. Publish dashboard, release, rollout, health, rollback, and audit state.

## Configuration

Configuration is defined by `capability_contract.py` and exposed through
`get_capability_contract()`. Tenant context, release ownership, manifests,
artifact signatures, health reports, log traces, change tickets, environment
policies, production approvals, and rollback plans are all explicit contract
concerns.

## Rules

- `tenant_context_required`
- `release_requires_owner`
- `deployment_requires_health_gate`
- `production_requires_approval`
- `rollback_requires_plan`
- `large_canary_requires_review`

The service calls the rule engine before creating releases, planning
deployments, and executing deployments. Deny decisions raise `PermissionError`;
review decisions create pending-review deployment plans until explicitly
approved.

## UI

The package exposes 8 APG Python UI route contract(s) through `views.py` and the
package semantic model:

- `/depl/dashboard`
- `/depl/releases`
- `/depl/deployments`
- `/depl/rollouts`
- `/depl/health`
- `/depl/rollback`
- `/depl/evidence`
- `/depl/settings`

`views.py` provides dashboard and release-detail models for release consoles,
rollout monitors, health gates, rollback centers, environment state, and audit
timelines.

## Theme

The package uses the `depl_release_ops` APG theme contract.
