# AICR Capability Package

AICR is the AI Core Framework capability for APG. It provides a governed AI
control plane for generated applications.

## Capability Contract

- Capability: `aicr`
- Display name: `AI Core Framework`
- Generated runtime: `service.AicrService`
- Event stream adapter: `bytewax`
- Primary dependencies: `conf`, `auth`, `mqeb`, `moni`, `audl`, `keym`
- First-class AI agents: `codex`, `claude_code`, `opencode`, `pi`
- Lifecycle processor: `bytewax`

## Executable Surface

- `capability_contract.py` defines configuration, rules, UI, adapters, and
  theme.
- `service.py` provides `AicrService` for dependency-light generated-app
  behavior.
- `api_helpers.py` exposes API-shaped helper functions over `AicrService`.
- `views.py` composes screen models from runtime state and contract
  configuration.
- `app.py` emits semantic package evidence from the current contract.

## Lifecycle

1. Register providers.
2. Register AI services.
3. Register models.
4. Record evaluations and promote models.
5. Record model metrics and drift-review evidence.
6. Register workflows and agent runtimes.
7. Register first-class AI agents with role, scope, owner, purpose, and
   disclosure metadata.
8. Validate lifecycle mutation batches through Bytewax.
9. Request governed inference.
10. Approve high-risk or large-context requests.
11. Execute approved inference and record audit evidence.

## Guardrails

The package currently exposes deterministic guardrails for tenant context,
owners, endpoints, provider types, provider credentials, egress policy, model
policy, modalities, evaluation, model metrics, drift review, retirement review,
service health, large-context review, high-risk approval, PII redaction, tool
allowlists, cross-tenant routing, cost review, workflow composition, agent runtime policy,
first-class AI-agent metadata, Bytewax lifecycle processing, external agent
actions, audit evidence, trace capture, and drift review.
