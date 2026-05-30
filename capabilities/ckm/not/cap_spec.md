# CKM Notification System Specification Pointer

The active CKM Notification System specification is maintained in
`SPECIFICATION.md`.

Use:

- `README.md` for package usage and generated-application composition notes.
- `SPECIFICATION.md` for the normative capability contract, lifecycle, rules,
  UI, configuration, adapter boundaries, and acceptance gates.
- `PLAN.md` for the current implementation and review plan.

Focused proof commands:

```bash
./.venv/bin/python -m py_compile capabilities/ckm/__init__.py capabilities/ckm/not/__init__.py capabilities/ckm/not/capability_contract.py capabilities/ckm/not/lifecycle.py capabilities/ckm/not/app.py capabilities/ckm/not/test_capability_contract.py
./.venv/bin/pytest -q capabilities/ckm/not/test_capability_contract.py
./.venv/bin/python -c "import importlib; pkg = importlib.import_module('capabilities.ckm.not'); service = pkg.NotificationLifecycleService('tenant-proof'); print(service.dashboard_summary())"
./.venv/bin/apg capabilities implementation-audit --root capabilities/ckm/not --json
./.venv/bin/apg capabilities publish-plan capabilities/ckm/not --json
```
