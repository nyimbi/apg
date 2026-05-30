# CONN Deployment Guide

CONN can run as a generated-app package or as an adapter-backed production
connector runtime.

## Generated-App Package

The generated-app path needs only the Python package files:

- `capability_contract.py`
- `conn_runtime.py`
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

- APG auth and permissions;
- credential vault and encryption providers;
- audit logging;
- monitoring, metrics, alerts, and health checks;
- local Singer tap and target execution;
- APG registry and gateway publication;
- lineage and data-quality engines;
- Bytewax connector lifecycle event streams.

Adapters must call CONN guardrails before writing to external systems, reading
secrets, running taps, or publishing side effects.

## Verification

Use focused package checks on battery-constrained machines:

```bash
./.venv/bin/pytest -q capabilities/common/conn/tests/test_capability_contract.py capabilities/common/conn/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/conn --json
./.venv/bin/apg capabilities publish-plan capabilities/common/conn --json
```

Run full Singer tap execution, external SaaS/database calls, frontend rendering,
Bytewax flows, and performance verification in a powered CI or staging
environment.
