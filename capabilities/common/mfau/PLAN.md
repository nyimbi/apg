# MFAU Implementation Plan

## Scope

Build MFAU as a coherent lifecycle and guardrail packet: specification, contract, deterministic runtime, dependency-light API helpers, dependency-light UI helpers, semantic model, package metadata, tests, and progress log.

## Steps

1. Document the capability.
   - Add `README.md` for usage and composition.
   - Add `SPECIFICATION.md` for functional requirements and acceptance criteria.
   - Keep `cap_spec.md` as a compatibility pointer to the specification.

2. Complete the executable contract.
   - Ensure configuration covers profiles, methods, enrollment, challenge, risk, devices, recovery, backup codes, policies, biometrics, governance, observability, adapters, UI, and theme.
   - Keep Bytewax as the event-stream adapter.
   - Keep rules deterministic and compatible with generated apps.

3. Implement generated-app runtime.
   - Add `MfauService` for tenant-scoped profile, method, device, risk, challenge, recovery, backup code, policy, and audit operations.
   - Reject unsafe operations through the rule engine.
   - Return serializable dictionaries for compiler output and example apps.

4. Implement dependency-light API and UI surfaces.
   - Replace framework-bound API code with helper functions that can be wrapped by generated apps.
   - Replace framework-bound views with UI model helper functions that expose route-ready data.

5. Build package metadata from the live contract.
   - Update `app.py` to construct semantic model data dynamically.
   - Regenerate `semantic_model.json`, `package_manifest.json`, and `release_report.json`.

6. Verify focused slices.
   - Compile changed files.
   - Run MFAU contract and package tests.
   - Run `app.self_test()`.
   - Run APG capability audit and publish-plan only for MFAU.
   - Scan the primary MFAU packet for stale scaffold or hype markers.

## Review Checklist

- Rule names, route names, and adapter keys are stable and readable.
- Runtime operations are tenant-scoped.
- Denied and review-required guardrails produce clear reasons.
- API helpers do not depend on Flask.
- UI helpers do not depend on Flask-AppBuilder.
- Package metadata is derived from the live contract.
- Tests cover the packet rather than isolated lines.
