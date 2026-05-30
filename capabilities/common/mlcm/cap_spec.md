# MLCM Capability Summary

AI Model Lifecycle Management governs tenant-scoped model registration,
versioning, evaluation, promotion, deployment, monitoring, drift response,
rollback, retirement, audit evidence, UI composition, and theming.

The full packet definition is maintained in `SPECIFICATION.md`; the execution
sequence is maintained in `PLAN.md`.

## Current Packet

- Runtime: `service.MlcmService`
- API helpers: `api.py`
- UI view models: `views.py`
- Contract: `capability_contract.py`
- Package evidence: `app.py`, `semantic_model.json`, `package_manifest.json`,
  and `release_report.json`
- Event stream adapter: Bytewax

## Lifecycle

`model -> version -> evaluation -> promotion -> deployment -> monitoring ->
drift review -> rollback -> retirement -> audit`
