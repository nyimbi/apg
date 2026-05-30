# Cash Management Development Plan

## Build Sequence

1. Replace the generated contract factory wrapper with an explicit APG contract.
2. Define lifecycle scope: banks, cash accounts, positions, flows, forecasts,
   liquidity, reconciliation, investments, payment runs, and CBM agents.
3. Implement deterministic guardrails for tenant context, write policy,
   liquidity, reconciliation, payment funding, Bytewax routing, and agent
   approval.
4. Replace provider-heavy public service/API/view/app surfaces with
   dependency-light executable boundaries.
5. Preserve compatibility aliases where existing importers expect them.
6. Add README, SPECIFICATION, PLAN, and refreshed `cap_spec.md` package docs.
7. Rename focused package tests and expand them to cover rules, service, API,
   views, app metadata, agents, liquidity, reconciliation, payment funding, and
   Bytewax behavior.
8. Refresh semantic and release evidence from the executable contract.
9. Run focused battery-conscious verification.
10. Record progress evidence, commit, push, and rescan remaining capability
    packets.

## Implementation Notes

- Do not introduce new dependencies.
- Keep top-level imports free of FastAPI, SQLAlchemy, Redis, bank SDKs, AI
  provider SDKs, and visualization engines.
- Keep Bytewax as the only lifecycle stream processor in the executable packet.
- Keep AI agents as first-class tenant-scoped records with supported runtime and
  role validation.
- Keep optional legacy modules available but outside the default import path.

## Review Checklist

- Contract shape validates through `validate_contract_shape`.
- Rule names and service error reasons are stable enough for tests and composed
  applications.
- Position recording enforces liquidity review when below buffer.
- Payment-run validation blocks unapproved cash deficits.
- Reconciliation enforces review for material variance.
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
- Service smoke test for bank, account, position, flow, forecast, reconciliation,
  investment, payment run, agent, and dashboard behavior.
- Marker scan and `git diff --check`.
