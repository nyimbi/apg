# Notifications and Alerts Capability Spec Pointer

The active specification is `SPECIFICATION.md`.

Use this file as the stable compatibility pointer for older capability tooling
that still expects `cap_spec.md`.

## Current Packet

`ntfy` now owns a coherent notification lifecycle packet:

- tenant-scoped recipient preferences;
- channel providers with health and fallback route metadata;
- template registration, approval, locale, owner, and content state;
- single-message delivery decisions;
- campaign creation, approval, batch review, and send lifecycle;
- idempotent delivery guardrails;
- audit events for notification state changes;
- UI route, view-model, theme, and adapter metadata;
- Bytewax as the required event-stream adapter for batch notification mutation.

## Proof Commands

```bash
./.venv/bin/python -m py_compile capabilities/common/ntfy/__init__.py capabilities/common/ntfy/capability_contract.py capabilities/common/ntfy/notification_runtime.py capabilities/common/ntfy/package_api.py capabilities/common/ntfy/view_models.py capabilities/common/ntfy/app.py capabilities/common/ntfy/test_capability_contract.py capabilities/common/ntfy/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/ntfy/test_capability_contract.py capabilities/common/ntfy/tests/test_package_contract.py
./.venv/bin/python -c "from capabilities.common.ntfy import app; r=app.self_test(); print(r); assert r['passed']"
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/ntfy --json
./.venv/bin/apg capabilities publish-plan capabilities/common/ntfy --json
```
