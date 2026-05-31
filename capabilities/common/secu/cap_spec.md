# SECU Capability Specification Pointer

The active SECU specification is maintained in `SPECIFICATION.md`.

Use:

- `README.md` for package usage and generated-application composition notes.
- `SPECIFICATION.md` for the normative functional contract, lifecycle, rules,
  AI security-agent composition, Bytewax streaming, UI, configuration, adapter
  boundaries, and acceptance gates.
- `PLAN.md` for the current implementation and review plan.

Focused proof commands:

```bash
./.venv/bin/python -m py_compile capabilities/common/secu/__init__.py capabilities/common/secu/security_runtime.py capabilities/common/secu/service.py capabilities/common/secu/api.py capabilities/common/secu/views.py capabilities/common/secu/capability_contract.py capabilities/common/secu/app.py capabilities/common/secu/tests/test_capability_contract.py capabilities/common/secu/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/secu/tests/test_capability_contract.py capabilities/common/secu/tests/test_package_contract.py
./.venv/bin/python capabilities/common/secu/app.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/secu --json
./.venv/bin/apg capabilities publish-plan capabilities/common/secu --json
```
