# MLCM Capability Plan

## Build Plan

1. Inspect current service, models, contract, package evidence, tests, and
   documentation.
2. Write capability documentation first so implementation has a clear target.
3. Expand the executable contract with full lifecycle configuration, at least
   30 deterministic guardrails, at least 12 UI routes, Bytewax adapter evidence,
   and theme components.
   - Promote model lifecycle agents and Bytewax lifecycle batches to first-class
     contract citizens.
4. Extend runtime records and service behavior only where the lifecycle packet
   needs executable support.
   - Add model lifecycle agent registration and lifecycle-batch validation.
5. Refresh API helpers and UI view models so generated applications can compose
   the capability.
   - Add API/view coverage for model lifecycle agent rosters and Bytewax batch
     monitors.
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
- Model lifecycle agents have supported-runtime, supported-role, scope, owner,
  purpose, disclosure, and privileged approval guardrails.
- Lifecycle batches require Bytewax and reject Kafka or other broker-first
  streams.
- `MlcmService` remains dependency-light and tenant-scoped.
- Retirement and rollback behavior does not permit cross-model or serving-risk
  violations.
- `app.py` builds semantic evidence from the live contract.
- Tests do not require full repository execution.

## Verification Plan

- Compile the MLCM Python files.
- Run only `capabilities/common/mlcm/test_capability_contract.py` and
  `capabilities/common/mlcm/tests/test_package_contract.py`.
- Run APG implementation audit and publish-plan checks for the MLCM directory.
- Run stale-marker search on the primary MLCM packet files.
- Run `git diff --check` on MLCM and the progress log.
