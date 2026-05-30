# ETLP Documentation

This directory contains operational documentation for APG ETLP. The root
`../README.md`, `../SPECIFICATION.md`, and `../PLAN.md` define the current
capability packet; this file provides a shorter navigation guide.

## Current Capability

ETLP supports governed pipeline composition for generated APG applications:

- Register pipelines and datasources.
- Define field mappings.
- Evaluate execution guardrails.
- Record quality evidence.
- Review publish requests.
- Control retry, replay, backfill, and retirement requests.
- Render generated application state through view models.

## Important Boundaries

The dependency-light lifecycle service does not run physical data movement. It
records lifecycle state and guardrail decisions. Production execution,
connectors, Bytewax streams, quality engines, lineage emitters, secret stores,
monitoring sinks, and persistence layers are adapters.

## Developer Entry Points

- `../capability_contract.py` - configuration, rules, routes, and theme.
- `../service.py` - production runtime and generated-app lifecycle service.
- `../api.py` - FastAPI controller and generated-app helper functions.
- `../view_models.py` - generated application screen models.
- `../field_mapper.py` - field mapping support.
- `../app.py` - package entrypoint and semantic evidence.

## Verification

Use focused verification while working on battery:

```bash
./.venv/bin/python -m py_compile capabilities/common/etlp/__init__.py capabilities/common/etlp/capability_contract.py capabilities/common/etlp/models.py capabilities/common/etlp/service.py capabilities/common/etlp/api.py capabilities/common/etlp/field_mapper.py capabilities/common/etlp/views.py capabilities/common/etlp/view_models.py capabilities/common/etlp/app.py capabilities/common/etlp/test_capability_contract.py capabilities/common/etlp/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/etlp/test_capability_contract.py capabilities/common/etlp/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/etlp --json
./.venv/bin/apg capabilities publish-plan capabilities/common/etlp --json
```
