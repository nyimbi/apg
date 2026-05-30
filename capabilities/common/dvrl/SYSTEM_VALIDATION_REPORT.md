# DVRL System Validation Notes

This file is intentionally scoped to the current DVRL capability packet.

## Current Validated Surface

- Executable capability contract in `capability_contract.py`.
- Dependency-light lifecycle service in `service.py`.
- Generated UI view models in `view_models.py`.
- Contract-derived package entrypoint in `app.py`.
- Focused package tests in `test_capability_contract.py` and
  `tests/test_package_contract.py`.

## Validation Boundary

The current packet validates source, schema, virtual table, query, cache,
policy, retirement, audit, UI, theme, and package-evidence behavior without
opening live physical data-source connections.

## Deferred Validation

- Physical database, SaaS, file, object-store, Singer, and streaming adapters.
- Live query execution and optimizer performance.
- Cache backend persistence.
- Metadata catalog, credential vault, audit sink, and Bytewax runtime flows.
- Rendered browser UI behavior.
- Full repository test suite.
