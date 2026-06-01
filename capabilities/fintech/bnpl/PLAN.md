# APG Buy Now Pay Later Capability Build Plan

## Build Sequence

1. Replace the placeholder package with a concrete APG capability contract.
2. Add dependency-light domain models for BNPL programs, consumers, merchants,
   checkouts, affordability decisions, plans, installments, settlements,
   disputes, and evidence.
3. Add runtime normalization helpers for codes, currencies, countries, amounts,
   scores, rates, and installment calculations.
4. Implement an in-memory service that enforces deterministic guardrails before
   mutating state.
5. Add API helpers and view-model builders that generated Python applications
   can call directly.
6. Add an app entrypoint that publishes a semantic model, component manifest,
   and self-test.
7. Add package docs, manifest, release evidence, and tests.
8. Run focused package verification and APG audits.

## Architecture

The package keeps live providers behind adapters. The executable core is local
and deterministic:

- `capability_contract.py` owns the contract, defaults, rule engine, UI routes,
  theme metadata, and Bytewax lifecycle metadata.
- `models.py` contains dataclasses with explicit `to_dict()` serializers.
- `bnpl_runtime.py` contains side-effect-free normalization and calculation
  helpers.
- `service.py` owns the BNPL lifecycle and audit events.
- `api.py` provides dependency-light request-dict functions.
- `views.py` builds dashboard and console view models.
- `app.py` publishes compiler/runtime surfaces.

## Review Criteria

The implementation is acceptable when:

- the contract validates through the APG registry;
- service methods cover the main BNPL lifecycle;
- rule tests prove tenant, policy, evidence, review, Bytewax, and agent
  guardrails;
- UI route and theme metadata are present;
- `semantic_model.json`, `package_manifest.json`, and `release_report.json`
  exist;
- focused tests and package audits pass;
- no disallowed broker terminology or stale placeholder marker remains in the
  package.

## Deferred Adapter Work

Follow-up slices can add real adapters for checkout gateway capture, acquirer
settlement files, payment-rail posting, credit bureau pulls, collections,
card-network disputes, regulator reporting, rendered UI checks, and durable
Bytewax topology deployment.
