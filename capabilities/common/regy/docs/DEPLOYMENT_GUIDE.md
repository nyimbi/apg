# REGY Deployment Guide

REGY can run as a generated-app package or as an adapter-backed production
registry surface.

## Generated-App Package

The generated-app path needs only the Python package files:

- `capability_contract.py`
- `registry_runtime.py`
- `api.py`
- `view_models.py`
- `app.py`
- `semantic_model.json`
- `package_manifest.json`
- `release_report.json`

This path is suitable for local composition, focused tests, and package
publication evidence.

## Production Integration

Production deployments must bind adapters for:

- APG auth and RBAC;
- configuration/discovery metadata;
- monitoring, metrics, traces, and health;
- audit logging;
- gateway synchronization;
- cache storage;
- Bytewax registry lifecycle event streams.

Adapters must honor REGY rule decisions before writing to external systems.

## Verification

Use focused package checks on battery-constrained machines:

```bash
./.venv/bin/pytest -q capabilities/common/regy/test_capability_contract.py capabilities/common/regy/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/regy --json
./.venv/bin/apg capabilities publish-plan capabilities/common/regy --json
```

Run full integration, rendered UI, gateway, cache, monitoring, audit, Bytewax,
and performance verification in a powered CI or staging environment.
