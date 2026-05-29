# AICR Capability Development Plan

## Current State

AICR is a large AI infrastructure package with model, inference, monitoring,
security, and deployment modules. It already has a valid capability contract,
theme, route metadata, legacy runtime tests, and domain-specific code. The next
focused packet is to make high-risk inference approval executable through a
small provider-neutral governance layer.

## Packet 1: High-Risk Inference Approval

Deliver a focused lifecycle packet:

- add package-level service, approval, and governance event records;
- add a lightweight `AicrService` facade for deterministic package behavior and
  view-model use;
- register tenant-owned AI services with model-policy and health guardrails;
- request inference and require approval for high-risk or large-context
  requests;
- approve or reject inference requests with reviewer evidence;
- execute approved requests through a local deterministic envelope;
- expose dependency-light `api_helpers.py` functions and view models;
- replace stale generated-package test naming with package contract tests;
- update package documentation and progress evidence.

## Implementation Steps

1. Extend `models.py` with `AICRServiceRecord`, `AICRInferenceApproval`, and
   `AICRGovernanceEvent`.
2. Extend `service.py` with a lightweight `AicrService` facade that enforces
   contract rules and emits audit evidence.
3. Add dependency-light `api_helpers.py` functions over the lightweight service.
4. Extend `views.py` with service registry, inference console, governance, and
   metrics view models.
5. Extend package contract tests with positive service-request-approval-run
   coverage and negative missing owner, missing policy, unhealthy routing,
   rejected approval, and tenant mismatch coverage.
6. Update `cap_spec.md` with current behavior and proof commands.
7. Run focused package proof, implementation audit, publish-plan, and diff
   checks.

## Review Checklist

- Service registration requires tenant and owner.
- Inference requires a model policy and healthy service.
- High-risk or large-context inference cannot execute without approval.
- Rejected approval cannot execute.
- Tenant mismatches are blocked.
- API helpers expose the same behavior as service methods.
- View models expose service, approval, inference, event, rule, and theme state.
- Provider integrations remain adapter boundaries.
