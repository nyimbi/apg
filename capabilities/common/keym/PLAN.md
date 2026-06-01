# APG KEYM Implementation Plan

## Target Slice

Build one coherent lifecycle and guardrail packet for Key Management:
tenant-managed key records, key operation decisions, export approval,
rotation exception review, rotation completion evidence, compromise response,
first-class key-agent composition, Bytewax lifecycle stream enforcement, UI view
models, durable review evidence, rule/contract evidence, tests, and publish
proof.

## Steps

1. Refresh `SPECIFICATION.md`, `PLAN.md`, `README.md`, `cap_spec.md`, and
   `todo.md` so KEYM documentation names the current source of truth.
2. Extend `capability_contract.py` with agent runtime/role metadata, Bytewax
   streaming metadata, deterministic agent/stream guardrails, route metadata,
   and theme components.
3. Extend `KeymService` with `KeymAgentRecord`, agent registration, agent
   listing, durable review evidence, Bytewax lifecycle batch validation,
   pending-review queues, dashboard counts, and audit evidence.
4. Extend `api.py` and `view_models.py` to expose key-agent registration,
   key-agent rosters, streaming metadata, pending reviews, lifecycle batch
   evidence, and posture evidence.
5. Extend `app.py` and generated semantic evidence with contract-derived
   `provides`, `requires`, `agents`, `streaming`, dependency graph edges, and
   self-test staleness checks.
6. Update package tests for the agent lifecycle, privileged-role denials,
   non-Bytewax denials, API helpers, view models, and committed JSON evidence.
7. Refresh `semantic_model.json` and `release_report.json`.
8. Run focused py_compile, focused pytest, app self-test, inspect,
   implementation audit, publish plan, stale marker search on touched source,
   and whitespace checks.
9. Record progress in `docs/progress_log.md`, commit with Lore trailers, and
   push.

## Review Risks

- Do not trust caller-supplied booleans for export approval or rotation
  exception state.
- Do not allow self-review or missing reviewer notes.
- Do not complete rotations without evidence.
- Do not reactivate compromised keys without explicit rotation evidence.
- Do not allow AI agents to operate without owner, purpose, scope, and
  contribution disclosure.
- Preserve privileged key-agent roles without explicit human approval as
  pending-review evidence.
- Do not accept key lifecycle batch mutations from broker-specific queue or any
  non-Bytewax stream; preserve denied routing evidence before raising.
- Keep live HSM/KMS/vault/blockchain/AI integrations behind adapters.
