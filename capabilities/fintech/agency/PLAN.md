# APG Agency Banking Capability Build Plan

## Build Sequence

1. Replace the placeholder module with a concrete APG capability contract.
2. Add domain models for programs, outlets, accredited agents, float accounts,
   customers, transactions, cash movements, commissions, disputes, supervision,
   and evidence.
3. Add side-effect-free runtime helpers for normalization, amounts, limits,
   float calculations, transaction direction, and commission estimates.
4. Implement an in-memory service that enforces deterministic rules before
   mutating state.
5. Add dependency-light API helpers and view-model builders for generated apps.
6. Add an app entrypoint with semantic model, component manifest, and self-test.
7. Add package documentation, manifest, release evidence, and tests.
8. Run focused verification and APG package audits.

## Architecture

- `capability_contract.py` defines identity, dependencies, configuration,
  deterministic rules, UI routes, theme tokens, and Bytewax lifecycle metadata.
- `models.py` contains dataclasses with explicit serializers.
- `agency_runtime.py` contains reusable calculation and normalization helpers.
- `service.py` owns tenant-scoped lifecycle state and audit events.
- `api.py` exposes request-dict functions for generated Python apps.
- `views.py` builds operational dashboards and route view models.
- `app.py` publishes compiler/runtime surfaces.

## Review Criteria

The slice is acceptable when:

- the contract validates through the APG registry;
- lifecycle docs and release evidence exist;
- service methods exercise all primary agency-banking workflows;
- tests prove tenant, policy, evidence, limit, float, settlement, supervision,
  Bytewax, and AI-agent guardrails;
- no stale placeholder marker remains;
- no disallowed broker terminology appears in the package;
- focused package audits and global capability audits pass.

## Deferred Adapter Work

Follow-up slices can add live POS/device adapters, field-force mobile apps, cash
vault posting, cash-in-transit settlement, mobile-money operator adapters, live
regulator filing, rendered UI checks, durable Bytewax topology deployment, and
performance testing.
