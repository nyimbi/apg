# FREC Implementation Plan

## Scope

Build one coherent facial-recognition lifecycle and guardrail packet: docs, executable contract, deterministic generated-app runtime, dependency-light API helpers, dependency-light UI models, dynamic package evidence, focused tests, progress-log evidence, review, commit, and push.

## Steps

1. Replace stale root docs.
   - Write a practical `README.md`.
   - Add `SPECIFICATION.md`.
   - Add `PLAN.md`.
   - Convert `cap_spec.md` into a pointer to the active specification.

2. Expand the executable contract.
   - Add consent, enrollment, template, liveness, verification, identification, watchlist, emotion, review, privacy, security, governance, observability, adapter, UI, and theme sections.
   - Add Bytewax as the batch event-stream adapter.
   - Expand deterministic guardrails beyond the existing thin six-rule set.

3. Implement generated-app runtime.
   - Add `face_runtime.py` with tenant-scoped in-memory records and deterministic lifecycle methods.
   - Enforce guardrails through `evaluate_capability_rules`.
   - Store only face-template metadata and decision evidence.

4. Add generated-app helper surfaces.
   - Add `api_helpers.py` for serializable helper functions.
   - Add `view_models.py` for route-ready UI data.
   - Keep production `api.py`, `views.py`, and `service.py` as adapter targets.

5. Refresh package evidence.
   - Replace static `app.py` semantic model with contract-derived output.
   - Regenerate `semantic_model.json`, `release_report.json`, and `package_manifest.json`.

6. Verify focused slice.
   - Compile edited FREC package files.
   - Run focused FREC contract/package tests.
   - Run `app.self_test()`.
   - Run APG implementation audit and publish-plan for FREC.
   - Scan the primary packet for stale scaffold/hype markers.
   - Run `git diff --check`.

## Review Checklist

- Runtime methods are tenant-scoped.
- Identification cannot run without watchlist policy.
- Verification cannot run without liveness and active template evidence.
- Emotion analysis requires an approved purpose.
- Batch mutation declares Bytewax.
- Generated-app helpers do not import production web frameworks.
- Package metadata is synchronized with the live contract.
