# APG ENCR Implementation Plan

## Target Slice

Build one coherent lifecycle and guardrail packet for Encryption Services:
tenant key domains, cryptographic operation decisions, crypto exception review,
threat-adaptive key rotation, UI view models, rule/contract evidence, tests,
and publish proof.

## Steps

1. Add package documentation: `SPECIFICATION.md`, `PLAN.md`, top-level
   `README.md`, and current lifecycle notes in `cap_spec.md`.
2. Add a dependency-light `EncrService` to `service.py` without disturbing the
   existing async encryption engine.
3. Extend `api.py` and `views.py` to expose key domains, operations, exception
   reviews, rotations, audit events, and shared service state.
4. Extend `capability_contract.py` with operation governance configuration,
   exception/rotation/audit routes, additional deterministic rules, and theme
   components.
5. Replace embedded semantic evidence in `app.py` with contract-derived
   evidence and add self-test staleness checks.
6. Rename the stale package test and add positive/negative lifecycle coverage.
7. Refresh `semantic_model.json`, `release_report.json`, and
   `package_manifest.json`.
8. Run focused py_compile, focused pytest, implementation audit, publish plan,
   stale marker search, and whitespace checks.
9. Record progress in `docs/progress_log.md`, commit with Lore trailers, and
   push.

## Review Risks

- Do not allow caller-supplied booleans to bypass fail-closed rule checks.
- Do not approve legacy algorithm review through self-review or missing notes.
- Do not complete rotations without evidence.
- Keep live KMS/HSM/KEYM/post-quantum/ZK/homomorphic providers behind adapters.
- Keep generated semantic evidence derived from the live contract.
