# AUTH Capability Specification Pointer

The active AUTH specification is maintained in `SPECIFICATION.md`.

Use:

- `README.md` for package usage and generated-application composition notes.
- `SPECIFICATION.md` for the normative functional contract, lifecycle,
  rules, AI security-agent composition, UI, configuration, adapter boundaries,
  and acceptance gates.
- `PLAN.md` for the current implementation and review plan.

The current executable packet adds durable AUTH review evidence for role
approvals, privacy approvals, privacy queries, privileged security-agent
review, denied Bytewax batch mutation routing, and audit events.

Focused proof commands:

```bash
./.venv/bin/python -m py_compile capabilities/common/auth/models.py capabilities/common/auth/service.py capabilities/common/auth/api_helpers.py capabilities/common/auth/view_models.py capabilities/common/auth/capability_contract.py capabilities/common/auth/app.py capabilities/common/auth/tests/test_capability_contract.py capabilities/common/auth/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/auth/tests/test_capability_contract.py capabilities/common/auth/tests/test_package_contract.py
./.venv/bin/python -c "from capabilities.common.auth import app; r=app.self_test(); print(r); assert r['passed']"
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/auth --json
./.venv/bin/apg capabilities publish-plan capabilities/common/auth --json
```
