# APG ENCR Implementation Plan

## Target Slice

Build one coherent lifecycle and guardrail packet for Encryption Services:
tenant key domains, cryptographic operation decisions, crypto exception review,
threat-adaptive key rotation, first-class crypto-agent composition, Bytewax
lifecycle stream enforcement, durable review evidence, UI view models,
rule/contract evidence, tests, and publish proof.

## Steps

1. Refresh `SPECIFICATION.md`, `PLAN.md`, `README.md`, `cap_spec.md`, and
   `todo.md` so ENCR documentation names the current source of truth.
2. Extend `capability_contract.py` with agent runtime/role metadata, Bytewax
   streaming metadata, deterministic agent/stream guardrails, route metadata,
   and theme components.
3. Extend `EncrService` with `CryptoAgentRecord`, agent registration, agent
   listing, durable review evidence, Bytewax lifecycle batch validation,
   pending-review queues, dashboard counts, and audit evidence.
4. Extend `api.py` and `views.py` to expose crypto-agent registration,
   crypto-agent rosters, streaming metadata, pending reviews, lifecycle batch
   evidence, and posture evidence.
5. Extend `app.py` and generated semantic evidence with contract-derived
   `provides`, `requires`, `agents`, `streaming`, dependency graph edges, and
   self-test staleness checks.
6. Update package tests for the agent lifecycle, privileged-role denials,
   non-Bytewax denials, API helpers, and view models.
7. Refresh `semantic_model.json` and `release_report.json`.
8. Run focused py_compile, focused pytest, app self-test, inspect,
   implementation audit, publish plan, stale marker search on touched source,
   and whitespace checks.
9. Record progress in `docs/progress_log.md`, commit with Lore trailers, and
   push.

## Review Risks

- Do not allow caller-supplied booleans to bypass fail-closed rule checks.
- Do not approve legacy algorithm review through self-review or missing notes.
- Do not complete rotations without evidence.
- Do not allow AI agents to operate without owner, purpose, scope, and
  contribution disclosure.
- Preserve privileged crypto-agent roles without explicit human approval as
  pending-review evidence.
- Do not accept crypto lifecycle batch mutations from broker-specific queue or
  any non-Bytewax stream; preserve denied routing evidence before raising.
- Keep live KMS/HSM/KEYM/post-quantum/ZK/homomorphic providers behind adapters.
- Keep generated semantic evidence derived from the live contract.
