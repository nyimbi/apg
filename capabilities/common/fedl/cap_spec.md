# FEDL Capability Summary

Federated Learning governs tenant-scoped federations, participant attestation,
privacy budgets, secure training rounds, model updates, poisoning defense,
secure aggregation, model release to MLCM, first-class federation agents,
Bytewax lifecycle-batch validation, retirement, audit evidence, UI composition,
and theming.

The full packet definition is maintained in `SPECIFICATION.md`; the execution
sequence is maintained in `PLAN.md`.

## Current Packet

- Runtime: `service.FedlService`
- Deterministic helper: `federated_engine.py`
- API helpers: `api.py`
- UI view models: `views.py`
- Contract: `capability_contract.py`
- Package evidence: `app.py`, `semantic_model.json`, `package_manifest.json`,
  and `release_report.json`
- Event stream adapter: Bytewax
- First-class agents: Codex, Claude Code, OpenCode, Pi via AICR adapter
  contracts
- Lifecycle guardrail: `fedl.lifecycle` Bytewax stream with accepted/denied
  batch evidence

## Lifecycle

`federation -> participant attestation -> training round -> model update ->
secure aggregation -> federated model -> MLCM release -> federation agent ->
Bytewax lifecycle batch -> retirement -> audit`
