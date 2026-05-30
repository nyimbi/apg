# APG KEYM Implementation Plan

## Target Slice

Build one coherent lifecycle and guardrail packet for Key Management:
tenant-managed key records, key operation decisions, export approval,
rotation exception review, rotation completion evidence, compromise response,
UI view models, rule/contract evidence, tests, and publish proof.

## Steps

1. Add package documentation: `SPECIFICATION.md`, `PLAN.md`, top-level
   `README.md`, and current lifecycle notes in `cap_spec.md`.
2. Add a dependency-light `KeymService` to `service.py` without disturbing the
   existing async `KeyManagementService`.
3. Add domain API helpers to `api.py` and dependency-light UI models in
   `view_models.py`.
4. Extend `capability_contract.py` with operation governance configuration,
   export approval, rotation exception, compromise, audit routes, additional
   rules, and theme components.
5. Replace embedded semantic evidence in `app.py` with contract-derived
   evidence and add staleness checks.
6. Rename the stale package test and add positive/negative lifecycle coverage.
7. Refresh `semantic_model.json`, `release_report.json`, and
   `package_manifest.json`.
8. Run focused py_compile, focused pytest, implementation audit, publish plan,
   stale marker search, and whitespace checks.
9. Record progress in `docs/progress_log.md`, commit with Lore trailers, and
   push.

## Review Risks

- Do not trust caller-supplied booleans for export approval or rotation
  exception state.
- Do not allow self-review or missing reviewer notes.
- Do not complete rotations without evidence.
- Do not reactivate compromised keys without explicit rotation evidence.
- Keep live HSM/KMS/vault/blockchain/AI integrations behind adapters.
