# GEOS Deployment Guide

Deploy GEOS as part of a generated APG Python application by including the
`geos` capability package and its generated `app.py` entrypoint evidence.

Minimum local proof:

```bash
./.venv/bin/python -m py_compile capabilities/common/geos/__init__.py capabilities/common/geos/capability_contract.py capabilities/common/geos/service.py capabilities/common/geos/api.py capabilities/common/geos/views.py capabilities/common/geos/app.py
./.venv/bin/pytest -q capabilities/common/geos/test_capability_contract.py capabilities/common/geos/tests/test_package_contract.py
```

Production deployments should provide external adapters for maps, routing,
indexes, telemetry ingestion, notifications, audit sinks, workflow handoff, and
Bytewax topology.
