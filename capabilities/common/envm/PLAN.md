# ENVM Environment Management Packet Plan

## Scope

Build `envm` as a coherent lifecycle and guardrail packet for APG applications
that need environment inventory, promotion, configuration drift, secret scopes,
AI-agent review, UI metadata, theme metadata, Bytewax stream governance, and
publishable package evidence.

## Implementation Packets

1. Specification and contract
   - Replace stale narrative in `cap_spec.md` with a pointer to the active
     specification.
   - Add `SPECIFICATION.md` for the normative behavior.
   - Expand `capability_contract.py` with configuration, rules, UI routes,
     theme metadata, provides/requires, agent metadata, and Bytewax streaming.

2. Dependency-light service
   - Preserve environment, promotion, drift, secret-scope, audit, and
     view-model behavior already present in `EnvmService`.
   - Add ENVM-agent data contracts and service methods.
   - Add batch mutation validation tied to the Bytewax stream guardrail.
   - Keep deployment providers, configuration stores, secret vaults,
     monitoring, and stream workers behind adapters.

3. Package entrypoint and helper surfaces
   - Make `__init__.py` export the expanded contract, service, agent model, and
     stream metadata.
   - Extend API helpers and view models with ENVM-agent and batch mutation
     surfaces.

4. Documentation and generated evidence
   - Add root package `README.md` with practical usage and composition notes.
   - Refresh semantic model, package manifest, and release evidence from the
     live contract.
   - Update the progress log with proof commands and review notes.

5. Focused proof and review
   - Extend focused contract/service tests without invoking live deployment,
     secret, monitoring, or stream-worker fixtures.
   - Run compile checks, focused tests, semantic probes, implementation audit,
     publish plan, stale-marker scan, and diff checks.
   - Review tenant isolation, production approval, promotion paths, drift
     review, secret scope policy, AI-agent boundaries, Bytewax guardrails,
     import behavior, and generated evidence consistency.

## Out Of Scope

- Live deployment provider integration.
- Durable configuration stores and secret vault calls.
- Runtime identity enforcement beyond metadata contracts.
- Live Bytewax topology deployment.
- Browser-rendered UI.
- Full repository test suite.

## Review Checklist

- Contract is registry-valid and APG Python route metadata uses practical
  targets.
- Dependency-light package import does not start deployment, secret,
  monitoring, or stream services.
- Environments require owner, supported stage, region, configuration source,
  RBAC policy, and secret-scope policy.
- Production changes require approval evidence.
- Promotion paths require source, target, deployment link, rollback environment,
  and approval.
- Promotion runs require artifact references.
- Drift above threshold requires review.
- Secret scopes require policy, secret references, and access roles.
- AI-agent guardrails include runtime, role, scope, registration, and
  contribution disclosure.
- Batch mutation is rejected unless the event stream is Bytewax.
- Generated semantic evidence matches the executable contract.
