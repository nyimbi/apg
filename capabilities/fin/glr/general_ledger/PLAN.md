# Financial Management General Ledger Development Plan

## Build Sequence

1. Replace the generated contract factory wrapper with an explicit APG contract.
2. Define the lifecycle scope: accounts, dimensions, periods, batches, journals,
   postings, reversals, allocations, trial balance, and GLR agents.
3. Implement deterministic guardrails for tenant context, write policy,
   accounting controls, Bytewax routing, and agent approvals.
4. Replace provider-heavy public service/API/view/app surfaces with
   dependency-light executable boundaries.
5. Preserve compatibility aliases where existing importers expect them.
6. Add README, SPECIFICATION, PLAN, and refreshed `cap_spec.md` package docs.
7. Rename focused package tests and expand them to cover rules, service, API,
   views, app metadata, agents, and Bytewax behavior.
8. Refresh semantic and release evidence from the executable contract.
9. Run focused battery-conscious verification.
10. Record progress evidence, commit, push, and rescan remaining capability
    packets.

## Implementation Notes

- Do not introduce new dependencies.
- Keep top-level imports free of Flask, AppBuilder, SQLAlchemy sessions, AI
  provider SDKs, and rendering engines.
- Keep Bytewax as the only lifecycle stream processor in the executable packet.
- Keep AI agents as first-class tenant-scoped records with supported runtime and
  role validation.
- Keep the legacy helper modules available but outside the default import path.

## Review Checklist

- Contract shape validates through `validate_contract_shape`.
- Rule names and service error reasons are stable enough for tests and composed
  applications.
- Posting requires approval, an open period, an idempotency key, and a different
  poster from the preparer.
- Journal line validation uses posting account ids from the same tenant.
- Trial balance is generated from posted entries only.
- UI route names match the semantic model.
- Stale generated wording is removed from touched package files.

## Verification

- Python syntax compile for public files and focused tests.
- Syntax compile for edited legacy helper files.
- Focused package pytest.
- `app.py` self-test.
- `apg capabilities inspect`.
- `apg capabilities publish-plan`.
- `apg capabilities implementation-audit`.
- Semantic metadata spot-check for Bytewax, provides, requires, agent route, and
  selected guardrails.
- Service smoke test for account, period, batch, journal, approval, posting,
  trial balance, agent, and dashboard behavior.
- Marker scan and `git diff --check`.
