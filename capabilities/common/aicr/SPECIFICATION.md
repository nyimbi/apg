# AICR Capability Specification

## Purpose

AICR is the APG AI control plane. It gives generated applications a provider
and model registry, governed inference envelope, workflow and agent-runtime
composition surface, and audit-ready AI governance records.

The current packet is provider-neutral. It does not require live model
providers, GPUs, cloud credentials, broker services, or network access.

## Scope

The executable packet covers:

- AI service registration.
- AI provider registration with credential-vault and egress-policy controls.
- Model registration with policy, provider linkage, modality, evaluation, and
  promotion state.
- Governed inference requests and approval lifecycle.
- Workflow registration over registered services.
- Agent runtime registration for Codex, Claude Code, OpenCode, Pi, Ollama, and
  custom HTTP runtimes.
- Governance event emission.
- Generated-app API helpers and UI view models.
- Dynamic semantic package evidence.

Out of scope for this packet:

- Physical model inference execution.
- Live third-party provider calls.
- Live agent CLI invocation.
- Live Bytewax, MQEB, MONI, AUDL, KEYM, or AUTH adapter calls.
- Browser-rendered UI verification.

## Lifecycle

1. Register an AI provider.
2. Register a tenant-owned AI service.
3. Register a model against the provider.
4. Record model evaluation and promote the model when ready.
5. Register workflows and agent runtimes.
6. Request inference.
7. Require review for high-risk or large-context requests.
8. Approve or reject the request.
9. Run approved inference through a deterministic local envelope.
10. Record governance and audit events.

## Configuration

The contract defines tenant configuration for services, providers, models,
inference, workflows, agent runtimes, governance, observability, adapters, UI,
and theme.

Important adapters:

- Generated-app runtime: `service.AicrService`
- Production runtime: `service.AICoreService`
- Event stream: `bytewax`
- Model lifecycle: `mlcm`
- Agent composition: `agnt`

## Rules

The deterministic rule engine includes guardrails for tenant context, service
owners and endpoints, supported provider types, provider credentials, egress
policy, model ownership, registered providers, model policy, supported
modalities, evaluation before promotion, retirement impact review, model policy
for inference, health-gated routing, large-context review, high-risk approval,
PII redaction, tool allowlists, cross-tenant denial, cost review, workflow
ownership and service binding, supported agent runtimes, agent tool policy,
external agent action approval, completion audit evidence, streaming trace
capture, and model drift review.

## UI Surfaces

AICR exposes 12 generated-app UI routes:

- Dashboard
- Services
- Providers
- Models
- Inference
- Workflows
- Agents
- Governance
- Evaluations
- Metrics
- Audit
- Settings

## Acceptance Criteria

- Contract validates through the APG capability audit.
- Package publish plan reports no warnings.
- Runtime can register providers, services, models, workflows, and agent
  runtimes.
- Runtime blocks unsafe missing-evidence paths.
- High-risk and large-context inference requires approval before execution.
- Package evidence is generated from the live contract.
- Primary docs do not contain stale baseline or marketing claims.
