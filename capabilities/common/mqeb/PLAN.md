# APG MQEB Implementation Plan

## Target Slice

Build one coherent lifecycle and guardrail packet for Message Queue Event Bus:
tenant topics, governed publish decisions, subscription lifecycle, delivery
attempt and dead-letter evidence, priority quota exception review, replay
review, Bytewax adapter boundaries, UI view models, contract evidence, tests,
and publish proof.

## Steps

1. Replace broad marketing claims in `README.md` with current, executable MQEB
   usage guidance and explicit Bytewax-first adapter boundaries.
2. Add a dependency-light MQEB package service layer in `service.py` without
   disturbing the existing async `MQEBService`.
3. Add package API helpers in `api.py` and dependency-light UI models in a new
   `view_models.py`.
4. Extend `capability_contract.py` with delivery governance configuration,
   topic/subscription/dead-letter/replay/quota routes, additional deterministic
   rules, and theme components.
5. Replace the embedded semantic model in `app.py` with contract-derived
   evidence and add staleness checks for routes, rules, and lifecycle surfaces.
6. Rename the stale materialized package test to `tests/test_package_contract.py`
   and add positive/negative topic-publish-subscription-delivery-review
   coverage.
7. Refresh `semantic_model.json`, `release_report.json`, and
   `package_manifest.json`.
8. Run focused py_compile, focused pytest, implementation audit, publish plan,
   stale marker search, and whitespace checks.
9. Record progress in `docs/progress_log.md`, commit with Lore trailers, and
   push.

## Review Risks

- Do not introduce Kafka as MQEB's core dependency; Bytewax is the preferred
  event-flow runtime boundary.
- Do not trust caller-supplied booleans for quota exception, replay approval,
  dead-letter state, or delivery completion.
- Do not let restricted or regulated topics publish without encryption/schema
  evidence.
- Do not let exactly-once delivery proceed without idempotency and dead-letter
  configuration.
- Do not allow replay without a bounded range, reason, reviewer, and evidence.
- Keep live brokers, cloud queues, SIEM/SOAR, schema registries, and Bytewax
  workers behind adapters that honor MQEB guardrails.
