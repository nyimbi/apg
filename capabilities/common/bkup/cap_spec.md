# BKUP Capability Specification Pointer

The active BKUP specification is maintained in `SPECIFICATION.md`.

Use:

- `README.md` for package usage and generated-application composition notes.
- `SPECIFICATION.md` for the normative capability contract, lifecycle, rules,
  UI, configuration, adapter boundaries, and acceptance gates.
- `PLAN.md` for the current implementation and review plan.

Focused proof commands:

```bash
./.venv/bin/python -m py_compile capabilities/common/bkup/__init__.py capabilities/common/bkup/capability_contract.py capabilities/common/bkup/models.py capabilities/common/bkup/backup_engine.py capabilities/common/bkup/service.py capabilities/common/bkup/api.py capabilities/common/bkup/views.py capabilities/common/bkup/app.py capabilities/common/bkup/test_capability_contract.py capabilities/common/bkup/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/bkup/test_capability_contract.py capabilities/common/bkup/tests/test_package_contract.py
./.venv/bin/python -c "from capabilities.common.bkup import app; r=app.self_test(); print(r); assert r['passed']"
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/bkup --json
./.venv/bin/apg capabilities publish-plan capabilities/common/bkup --json
```
