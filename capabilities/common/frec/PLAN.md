# FREC Implementation Plan

## Scope

Build one coherent facial-recognition lifecycle and guardrail packet: docs, executable contract, deterministic generated-app runtime, first-class provider-neutral facial-recognition agents, Bytewax lifecycle batch validation, dependency-light API helpers, dependency-light UI models, dynamic package evidence, focused tests, progress-log evidence, review, commit, and push.

## Steps

1. Replace stale root docs.
   - Write a practical `README.md`.
   - Add `SPECIFICATION.md`.
   - Add `PLAN.md`.
   - Convert `cap_spec.md` into a pointer to the active specification.

2. Expand the executable contract.
	- Add consent, enrollment, template, liveness, verification, identification, watchlist, emotion, review, privacy, security, governance, observability, adapter, UI, and theme sections.
	- Add first-class facial-recognition agent configuration for `codex`, `claude_code`, `opencode`, and `pi` through a provider-neutral AICR adapter contract.
	- Add Bytewax as the batch event-stream adapter and lifecycle processor.
	- Expand deterministic guardrails beyond the existing thin six-rule set.

3. Implement generated-app runtime.
	- Add `face_runtime.py` with tenant-scoped in-memory records and deterministic lifecycle methods.
	- Enforce guardrails through `evaluate_capability_rules`.
	- Store only face-template metadata and decision evidence.
	- Add facial-recognition agent records and lifecycle batch records.

4. Add generated-app helper surfaces.
	- Add `api_helpers.py` for serializable helper functions.
	- Add `view_models.py` for route-ready UI data.
	- Keep production `api.py`, `views.py`, and `service.py` as adapter targets.
	- Surface `/frec/agents` and `/frec/lifecycle` route-ready models.
	- Harden production `api.py` image-source normalization so base64 payloads,
	  data URLs, and governed HTTP/HTTPS image URLs are executable instead of
	  returning unimplemented responses.

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
- Facial-recognition agents have supported runtime, supported role, owner, purpose, scope, contribution disclosure, and privileged-role review guardrails.
- Lifecycle batches are non-empty, use supported FREC operations, and declare Bytewax.
- Generated-app helpers do not import production web frameworks.
- Production API image URL handling blocks unsupported schemes, private network
  hosts, oversized payloads, and non-image content before service invocation.
- Package metadata is synchronized with the live contract.
