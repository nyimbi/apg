# POSE Development Plan

## Objective

Build POSE into a coherent APG pose-estimation lifecycle and guardrail packet
that can be packaged and tested without heavyweight runtime dependencies while
leaving clear adapter boundaries for production inference.

## Implementation Slices

1. **Contract**
   - Expand configuration for models, sessions, tracking, analysis, AI pose
     agents, governance, observability, adapters, UI, theme, and Bytewax.
   - Expand deterministic guardrails for consent, source security, sensitive
     use, keypoint quality, medical review, 3D calibration, agents, state
     changes, tenant isolation, and Bytewax.

2. **Runtime Models**
   - Replace heavyweight ORM-only models with dependency-light dataclasses for
     generated applications.
   - Preserve compatibility aliases for older package callers.

3. **Service Runtime**
   - Add executable model registration, session start, frame capture, pose
     estimate, analysis, reconstruction, agent registration, state change,
     list/dashboard, and audit support.
   - Keep production model inference behind CVSN/MLCM/edge adapters.

4. **API And Views**
   - Add payload helpers and framework-neutral UI models for the complete
     lifecycle.

5. **Documentation**
   - Add README, full specification, and this implementation plan.
   - Replace older overclaiming docs with current adapter-boundary notes.

6. **Verification**
   - Run focused `py_compile`.
   - Run POSE contract/package tests only.
   - Run generated app self-test.
   - Run APG implementation audit and publish plan for POSE.
   - Search POSE for unsupported overclaims, unfinished scaffolding, and banned stream
     choices.

## Review Checklist

- The contract exposes provides, requires, rules, UI routes, theme, adapters,
  and Bytewax streaming.
- Runtime imports are dependency-light.
- Subject consent, secure streams, sensitive use approval, quality review,
  medical review, and calibration guardrails are enforced.
- AI pose agents are first-class records with runtime, role, scope, policy, and
  disclosure.
- Generated package evidence is refreshed after contract changes.
