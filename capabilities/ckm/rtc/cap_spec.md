# CKM Real-Time Collaboration Specification Pointer

The active CKM Real-Time Collaboration specification is maintained in
`SPECIFICATION.md`.

Use:

- `README.md` for package usage and generated-application composition notes.
- `SPECIFICATION.md` for the normative capability contract, lifecycle, rules,
  UI, configuration, adapter boundaries, and acceptance gates.
- `PLAN.md` for the current implementation and review plan.
- `runtime_app.py` for the preserved legacy FastAPI/WebSocket runtime entrypoint.

Focused proof commands:

```bash
./.venv/bin/python -m py_compile capabilities/ckm/rtc/__init__.py capabilities/ckm/rtc/capability_contract.py capabilities/ckm/rtc/lifecycle.py capabilities/ckm/rtc/app.py capabilities/ckm/rtc/test_capability_contract.py
./.venv/bin/pytest -q capabilities/ckm/rtc/test_capability_contract.py
./.venv/bin/python -c "import importlib; pkg = importlib.import_module('capabilities.ckm.rtc'); service = pkg.RtcLifecycleService('tenant-proof'); print(service.dashboard_summary())"
./.venv/bin/apg capabilities implementation-audit --root capabilities/ckm/rtc --json
./.venv/bin/apg capabilities publish-plan capabilities/ckm/rtc --json
```
