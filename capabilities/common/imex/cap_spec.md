# IMEX Capability Package

IMEX is the Import/Export capability for APG. It provides a governed transfer
control plane for compiler-generated applications.

## Capability Contract

- Capability: `imex`
- Display name: `Import/Export`
- Generated runtime: `imex_runtime.ImexService`
- Event stream adapter: `bytewax`
- Primary dependencies: `etlp`, `conn`, `auth`, `audl`, `moni`, `keym`, `encr`

## Executable Surface

- `capability_contract.py` defines configuration, rules, UI, adapters, and
  theme.
- `imex_runtime.py` implements generated-app lifecycle behavior.
- `view_models.py` composes screen models from runtime state and contract
  configuration.
- `app.py` emits semantic package evidence from the current contract.

## Lifecycle

1. Register endpoints.
2. Create a mapping profile.
3. Create a transfer job.
4. Validate a preview.
5. Execute a run.
6. Complete the run.
7. Publish retained artifacts.
8. Replay or purge with guardrails.

## Guardrails

The package currently exposes deterministic guardrails for tenant context,
ownership, transfer direction, endpoints, supported formats, profiling,
checksums, mapping evidence, PII policy, destination approval, preview
validation, production approval, encryption, monitoring, checkpointing, quality,
quarantine, capacity, retry/replay, schedules, artifacts, retention, purge,
owner transfer, ETLP plan linkage, CONN binding, audit, and final quality
evidence.
