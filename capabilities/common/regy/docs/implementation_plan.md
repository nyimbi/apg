# REGY Implementation Plan

This document mirrors the current capability packet and replaces older
speculative roadmap language. The authoritative executable source is
`../capability_contract.py`.

## Implementation Tracks

1. **Contract**
   - Maintain tenant-scoped configuration defaults and schema.
   - Keep deterministic guardrails for service, instance, discovery, version,
     gateway publication, health override, ownership, and retirement workflows.
   - Keep adapter metadata explicit, including Bytewax for lifecycle event
     streaming.

2. **Generated-App Runtime**
   - Use `../registry_runtime.py` for dependency-light lifecycle behavior.
   - Keep this runtime free of external service-mesh, gateway, cache, monitor,
     audit, and stream dependencies.
   - Return dictionaries suitable for generated applications and tests.

3. **Production Runtime**
   - Keep `../service.py`, `../api.py`, and `../views.py` as adapter-backed
     production and legacy surfaces.
   - Treat optional APG dependencies as adapters that must enforce REGY
     decisions before side effects.

4. **UI**
   - Use `../view_models.py` for generated-app composition.
   - Keep Flask-AppBuilder views as a separate runtime surface.

5. **Evidence**
   - Keep `../app.py`, `../semantic_model.json`, `../release_report.json`, and
     `../package_manifest.json` derived from the current contract shape.
   - Add focused tests when the contract or lifecycle surface changes.

## Focused Verification

```bash
./.venv/bin/python -m py_compile capabilities/common/regy/__init__.py capabilities/common/regy/capability_contract.py capabilities/common/regy/models.py capabilities/common/regy/registry_runtime.py capabilities/common/regy/api.py capabilities/common/regy/view_models.py capabilities/common/regy/app.py capabilities/common/regy/test_capability_contract.py capabilities/common/regy/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/regy/test_capability_contract.py capabilities/common/regy/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/regy --json
./.venv/bin/apg capabilities publish-plan capabilities/common/regy --json
```

Full runtime integration, rendered UI, Bytewax execution, gateway side effects,
and performance benchmarks remain production-adapter verification tasks.
