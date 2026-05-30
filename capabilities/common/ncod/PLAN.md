# NCOD Development Plan

## Objective

Build NCOD into a coherent APG no-code/low-code lifecycle and guardrail packet
that can compose executable applications rapidly while keeping governance,
AI-agent participation, theming, workflows, publishing, deployment, and audit
explicit.

## Implementation Slices

1. **Contract**
   - Expand configuration for apps, builder, extensions, AI builder agents,
     deployments, governance, observability, adapters, UI, and theme.
   - Expand deterministic rules for app creation, screen composition, data
     models, bindings, workflows, scripts, connectors, AI agents, validation,
     publishing, deployments, state changes, tenant isolation, and Bytewax.
   - Add Bytewax streaming contract and richer UI routes.

2. **Runtime Models**
   - Add data model, theme variant, AI builder agent, and deployment records.
   - Extend workflow bindings with policy references.
   - Preserve compatibility aliases for older package callers.

3. **Service Runtime**
   - Enforce tenant, owner, policy, validation, AI-agent, publish, deployment,
     and state-change guardrails.
   - Add data model definition, theme variants, AI builder-agent registration,
     deployment, state-change, list, dashboard, and audit support.
   - Keep operations deterministic and dependency-light.

4. **API And Views**
   - Add payload helpers for data models, theme variants, AI agents,
     deployments, and state changes.
   - Add data modeler, workflow designer, deployment center, AI agent panel,
     audit trail, and analytics view models.

5. **Documentation**
   - Add README, full specification, and this implementation plan.
   - Replace stale `cap_spec.md` with a compatibility pointer.

6. **Verification**
   - Run focused `py_compile`.
   - Run NCOD contract/package tests only.
   - Run generated app self-test.
   - Run APG implementation audit and publish plan for NCOD.
   - Search NCOD for stale generated-package claims or banned stream choices.

## Review Checklist

- The contract exposes `provides`, `requires`, `streaming`, UI routes, theme,
  configuration, and rule engine.
- AI builder agents are first-class runtime records with supported runtimes,
  role, scope, registration, and disclosure guardrails.
- Bytewax is the only batch/runtime stream policy named by NCOD.
- Publish cannot bypass validation and approval.
- Deployment cannot bypass target, approval, and rollback evidence.
- Tenant isolation is enforced through lookup boundaries and explicit rules.
- Tests prove the main executable lifecycle and representative guardrail
  failures.
- Generated package evidence is refreshed after code changes.

