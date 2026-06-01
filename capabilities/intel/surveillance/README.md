# APG Digital Surveillance

`intel_surveillance` is an executable APG capability for lawful, defensive
digital-surveillance workflows. It can be composed into generated APG
applications that need facility monitoring, endpoint telemetry review,
authorized public-safety monitoring, fraud monitoring, asset protection,
incident watch, or compliance evidence.

## What It Provides

- Authority, program, monitored asset, sensor, observation, alert, risk
  assessment, referral, dissemination, review, and AI-agent workflows.
- Deterministic rules that enforce tenant context, lawful authority, privacy
  review, calibration, evidence, approvals, Bytewax lifecycle routing, and
  AI-agent guardrails.
- API helpers and view models that generated Python applications can call
  without a web framework dependency.
- UI route metadata and compact theme tokens for generated application screens.
- A publishable `app.py` entrypoint with self-test and semantic-model output.

## Local Usage

```bash
./.venv/bin/python capabilities/intel/surveillance/app.py
./.venv/bin/pytest -q capabilities/intel/surveillance/tests/test_package_contract.py
./.venv/bin/apg capabilities inspect intel_surveillance --json
```

Generated applications can import the service directly:

```python
from capabilities.intel.surveillance import DigitalSurveillanceService

service = DigitalSurveillanceService()
authority = service.record_authority(
    "auth-1",
    "tenant-a",
    "security_monitoring_authority",
    "facility-scope",
    "confidential",
    "approver-1",
    "2026-12-31",
    "evidence-auth",
)
```

## Guardrails

The capability is defensive and compliance-first. It does not implement covert
tracking, stalking, spyware, credential capture, bypass, biometric
identification, exfiltration, live sensor control, or unauthorized monitoring.
AI-agent actions that request those scopes are denied by the rule engine.

## Verification

Use focused verification during battery-constrained development:

```bash
./.venv/bin/python -m py_compile capabilities/intel/surveillance/*.py capabilities/intel/surveillance/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/intel/surveillance/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/intel/surveillance --json
./.venv/bin/apg capabilities lifecycle-audit --root capabilities/intel/surveillance --json
```
