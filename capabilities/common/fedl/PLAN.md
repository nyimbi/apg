# FEDL Capability Plan

## Build Plan

1. Inspect current service, models, contract, package evidence, tests, and
   documentation.
2. Write capability documentation first so implementation has a clear target.
3. Expand the executable contract with full lifecycle configuration, at least
   38 deterministic guardrails, at least 15 UI routes, first-class federation
   agents, Bytewax lifecycle-stream evidence, and privacy-mesh theme
   components.
4. Extend runtime records and service behavior only where the lifecycle packet
   needs executable support.
5. Refresh API helpers and UI view models so generated applications can compose
   federations, AI agents, lifecycle batches, guardrails, and theme metadata.
6. Replace static package evidence with dynamic contract-derived evidence.
7. Update focused tests for contract, rules, service lifecycle, guardrails,
   generated-app views, package evidence, and compatibility behavior.
8. Run battery-conscious verification only on the changed packet.
9. Review the diff, stale markers, package audit, publish plan, and whitespace.
10. Commit and push the verified slice.

## Review Checklist

- Configuration schema includes all lifecycle domains.
- Rule count and route count are high enough to prevent stale narrow packets.
- Bytewax appears in adapters and package streaming metadata.
- Federation agents are first-class citizens with provider-neutral runtime
  codes, bounded scope, owner, purpose, contribution disclosure, privileged-role
  review handling, and UI metadata.
- `FedlService` remains dependency-light and tenant-scoped.
- Federated model release requires MLCM linkage, approval, and privacy review.
- `app.py` builds semantic evidence from the live contract.
- Tests do not require full repository execution.

## Verification Plan

- Compile the FEDL Python files.
- Run only `capabilities/common/fedl/test_capability_contract.py` and
  `capabilities/common/fedl/tests/test_package_contract.py`.
- Run APG implementation audit and publish-plan checks for the FEDL directory.
- Run stale-marker search on the primary FEDL packet files.
- Run `git diff --check` on FEDL and the progress log.

## Current Packet Focus

The current coherent slice is federation-agent composition and Bytewax
lifecycle guardrails. It adds executable state for `FederationAgentRecord` and
`FedlLifecycleBatchRecord`, contract-level `agents` and `streaming` manifests,
deterministic policy rules, API helpers, roster and lifecycle view models,
semantic-model evidence, and focused regression coverage.
