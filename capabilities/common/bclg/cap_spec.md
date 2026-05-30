# BCLG Capability Specification Pointer

The active BCLG specification is maintained in `SPECIFICATION.md`.

Use:

- `README.md` for package usage and generated-application composition notes.
- `SPECIFICATION.md` for the normative capability contract, lifecycle, rules,
  UI, configuration, adapter boundaries, and acceptance gates.
- `PLAN.md` for the current implementation and review plan.

Focused proof commands:

```bash
./.venv/bin/python -m py_compile capabilities/common/bclg/__init__.py capabilities/common/bclg/capability_contract.py capabilities/common/bclg/models.py capabilities/common/bclg/ledger_engine.py capabilities/common/bclg/service.py capabilities/common/bclg/api.py capabilities/common/bclg/views.py capabilities/common/bclg/app.py capabilities/common/bclg/test_capability_contract.py capabilities/common/bclg/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/bclg/test_capability_contract.py capabilities/common/bclg/tests/test_package_contract.py
./.venv/bin/python -c "from capabilities.common.bclg import app; r=app.self_test(); print(r); assert r['passed']"
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/bclg --json
./.venv/bin/apg capabilities publish-plan capabilities/common/bclg --json
```
