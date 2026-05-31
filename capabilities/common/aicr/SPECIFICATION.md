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
- First-class AI-agent registration for Codex, Claude Code, OpenCode, and Pi
  with explicit role, scope, owner, purpose, contribution disclosure, and
  privileged-role approval status.
- Bytewax lifecycle batch validation for model, prompt, inference, evaluation,
  safety, routing, and AI-agent mutation batches.
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
6. Register first-class AI agents that can contribute to or review AI-core
   state.
7. Validate lifecycle mutation batches through the Bytewax stream contract.
8. Request inference.
9. Require review for high-risk or large-context requests.
10. Approve or reject the request.
11. Run approved inference through a deterministic local envelope.
12. Record governance and audit events.

## Configuration

The contract defines tenant configuration for services, providers, models,
inference, workflows, agent runtimes, first-class AI agents, Bytewax lifecycle
streaming, governance, observability, adapters, UI, and theme.

Important adapters:

- Generated-app runtime: `service.AicrService`
- Production runtime: `service.AICoreService`
- Event stream: `bytewax`
- Model lifecycle: `mlcm`
- Agent composition: `agnt`

## First-Class AI Agents

An AICR AI agent is governed platform state, not a loose tool invocation. Each
agent must declare tenant, stable ID, name, runtime, AI-core role, bounded
scope, accountable owner, purpose, and machine contribution disclosure.

Supported runtimes are `codex`, `claude_code`, `opencode`, and `pi`.
Supported roles include model, prompt, inference, safety, evaluation, routing,
tool, cost, and model-steward review roles. Privileged roles without human
approval are accepted only as `pending_review`; unsupported runtime, unsupported
role, missing scope, missing owner, missing purpose, and missing contribution
disclosure are blocked.

## Bytewax Lifecycle Batches

AICR lifecycle mutation batches must use Bytewax as the required processor.
The executable packet validates `model_batch`, `prompt_batch`,
`inference_batch`, `evaluation_batch`, `safety_batch`, `routing_batch`, and
`ai_agent_batch` operations. Non-Bytewax streams are denied and audited. Broker-specific queue
is deliberately not a core requirement for this packet.

## Rules

The deterministic rule engine includes guardrails for tenant context, service
owners and endpoints, supported provider types, provider credentials, egress
policy, model ownership, registered providers, model policy, supported
modalities, evaluation before promotion, retirement impact review, model policy
for inference, health-gated routing, large-context review, high-risk approval,
PII redaction, tool allowlists, cross-tenant denial, cost review, workflow
ownership and service binding, supported agent runtimes, agent tool policy,
first-class AI-agent runtime, role, scope, owner, purpose, contribution
disclosure, privileged approval status, Bytewax lifecycle processing, external
agent action approval, completion audit evidence, streaming trace capture, and
model drift review.

## UI Surfaces

AICR exposes 14 generated-app UI routes:

- Dashboard
- Services
- Providers
- Models
- Inference
- Workflows
- Agent Runtimes
- Agents
- Lifecycle
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
- Runtime can register first-class AI agents and validate Bytewax lifecycle
  batches.
- Runtime blocks unsafe missing-evidence paths.
- Runtime records denied lifecycle batches before raising the guardrail error.
- High-risk and large-context inference requires approval before execution.
- Package evidence is generated from the live contract.
- Primary docs do not contain stale generated-template or marketing claims.
