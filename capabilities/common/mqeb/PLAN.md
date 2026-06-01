# APG MQEB Implementation Plan

## Target Slice

Build one coherent lifecycle and guardrail packet for Message Queue Event Bus:
tenant topics, governed publish decisions, subscription lifecycle, delivery
attempt and dead-letter evidence, priority quota exception review, replay
review, first-class event-agent composition, Bytewax lifecycle-batch validation,
durable review evidence, UI view models, contract evidence, tests, and publish
proof.

## Steps

1. Replace broad marketing claims in `README.md`, `cap_spec.md`, and
   `todo.md` with current executable MQEB usage guidance and explicit
   Bytewax-first adapter boundaries.
2. Add a dependency-light MQEB package service layer in `service.py` without
   disturbing the existing async `MQEBService`.
3. Add package API helpers in `api.py` and dependency-light UI models in a new
   `view_models.py`, including pending review queues and lifecycle batch
   evidence.
4. Extend `capability_contract.py` with delivery governance, first-class event
   agents, Bytewax streaming configuration, topic/subscription/dead-letter/
   replay/quota/agent routes, additional deterministic rules, and theme
   components.
5. Replace the embedded semantic model in `app.py` with contract-derived
   evidence and add staleness checks for routes, rules, event agents, Bytewax,
   and lifecycle surfaces.
6. Keep `tests/test_package_contract.py` focused on positive and negative
   topic-publish-subscription-delivery-review coverage plus event-agent and
   Bytewax lifecycle-batch coverage.
7. Refresh `semantic_model.json`, `release_report.json`, and
   `package_manifest.json`.
8. Run focused py_compile, focused pytest, implementation audit, publish plan,
   stale marker search, and whitespace checks.
9. Record progress in `docs/progress_log.md`, commit with Lore trailers, and
   push.

## Review Risks

- Do not introduce broker-specific queue as MQEB's core dependency; Bytewax is the preferred
  event-flow runtime boundary.
- Do not trust caller-supplied booleans for quota exception, replay approval,
  dead-letter state, or delivery completion.
- Do not let restricted or regulated topics publish without encryption/schema
  evidence.
- Do not let exactly-once delivery proceed without idempotency and dead-letter
  configuration.
- Do not allow replay without a bounded range, reason, reviewer, and evidence.
- Do not register event agents without supported runtimes/roles, owner,
  purpose, scope, and contribution disclosure.
- Preserve privileged event-agent roles without explicit human approval as
  pending-review evidence.
- Do not compose MQEB lifecycle batches unless they declare Bytewax as the
  stream processor; preserve denied routing evidence before raising.
- Keep live brokers, cloud queues, SIEM/SOAR, schema registries, and Bytewax
  workers behind adapters that honor MQEB guardrails.
