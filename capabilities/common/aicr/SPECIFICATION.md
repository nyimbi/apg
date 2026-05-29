# AICR Capability Specification

## Identity

- Capability ID: `aicr`
- Display name: AI Core Framework
- Category: `common`
- Owner: APG Platform Team
- Runtime shell: `apg_python`
- Theme: `aicr_ai_control_console`

## Purpose

AICR is the shared AI control plane for APG applications. It registers
tenant-owned AI services, governs inference requests, exposes model and
workflow surfaces, records AI audit evidence, and keeps high-risk AI work behind
explicit policy and human review.

The package must remain usable without live model providers, GPU services,
network brokers, or vendor credentials. Real inference engines, model stores,
agent runtimes, and monitoring systems remain adapter boundaries. Local package
proof focuses on deterministic governance and composition behavior.

## Users And Outcomes

- Platform teams can register AI services with owners, health state, and model
  policy.
- Application builders can request governed inference against registered
  services.
- Reviewers can approve high-risk or large-context inference requests before
  execution.
- Operators can inspect AI service, inference, approval, and audit-event state
  through APG view models.
- Generated APG applications can compose AICR with AGNT, NLPC, RAGN, GRAG,
  MLCM, MONI, AUDL, and AUTH without coupling to one model provider.

## Domain Model

AICR owns these package-level records:

- `AICRServiceRecord`: tenant-scoped AI service registration with owner,
  service type, endpoint, health, and model policy.
- `AICRInferenceApproval`: governed request for high-risk or large-context
  inference.
- `AICRGovernanceEvent`: tenant-scoped evidence event for service,
  inference, approval, and routing lifecycle changes.

The older model, pipeline, and inference models remain available for the
broader AICR runtime and tests.

## Lifecycle

The focused governance lifecycle is:

1. Register a tenant-owned AI service with model policy and health metadata.
2. Request inference with context size, risk, prompt summary, and caller.
3. Allow normal inference when policy and service health pass.
4. Require approval for high-risk or large-context inference.
5. Approve or reject the inference request with reviewer notes.
6. Execute approved inference through a deterministic local envelope.
7. Block routing to unhealthy services.
8. Emit audit events for registration, approval, inference, and blocking.

## Rules And Guardrails

The contract rules are executable guardrails:

- `tenant_context_required`: operations require tenant context.
- `service_registration_requires_owner`: AI services require owners.
- `inference_requires_model_policy`: inference requires model policy.
- `high_risk_workflow_requires_approval`: high-risk AI workflows require
  approval.
- `unhealthy_service_blocks_routing`: unhealthy services cannot receive routed
  work.
- `large_context_requires_review`: large context windows require cost/safety
  review.

Service methods must enforce these rules and expose the same decisions through
view models and the dependency-light `api_helpers.py` surface. The existing
Flask API module remains a web adapter and must not be required for package
contract proof.

## UI And Theme

AICR exposes route and view-model surfaces for:

- dashboard summary;
- service registry;
- inference console;
- model catalog;
- workflow designer;
- governance center;
- metrics;
- settings.

The `aicr_ai_control_console` theme must provide semantic tokens and component
metadata for service health, inference traces, workflow graphs, and governance
rule decisions.

## Adapter Boundaries

These integrations remain replaceable:

- external model providers and local model servers;
- GPU, TPU, edge, neuromorphic, or cloud runtimes;
- model artifact stores and model registries;
- prompt and inference monitoring systems;
- audit/SIEM exporters;
- approval workflow engines.

Local package tests must not require those systems.

## Acceptance Gates

Focused AICR proof:

```bash
./.venv/bin/pytest -q capabilities/common/aicr/test_capability_contract.py capabilities/common/aicr/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/aicr --json
./.venv/bin/apg capabilities publish-plan capabilities/common/aicr --json
git diff --check -- capabilities/common/aicr
```
