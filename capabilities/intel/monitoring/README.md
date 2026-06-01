# APG Real-Time Monitoring

`intel_monitoring` is an executable APG capability for lawful, defensive
real-time monitoring workflows. It can be composed into generated APG
applications that need security monitoring, fraud monitoring, public-safety
monitoring, compliance monitoring, availability monitoring, or operational
incident triage.

## What It Provides

- Authority, monitoring policy, source, watch, event, signal, incident,
  referral, dissemination, review, and AI-agent workflows.
- Deterministic rules that enforce tenant context, lawful authority, source
  access review, evidence, approvals, Bytewax lifecycle routing, and AI-agent
  guardrails.
- API helpers and view models that generated Python applications can call
  without a web framework dependency.
- UI route metadata and compact theme tokens for generated application screens.
- A publishable `app.py` entrypoint with self-test and semantic-model output.

## Local Usage

```bash
./.venv/bin/python capabilities/intel/monitoring/app.py
./.venv/bin/pytest -q capabilities/intel/monitoring/tests/test_package_contract.py
./.venv/bin/apg capabilities inspect intel_monitoring --json
```

Generated applications can import the service directly:

```python
from capabilities.intel.monitoring import RealTimeMonitoringService

service = RealTimeMonitoringService()
authority = service.record_authority(
    "auth-1",
    "tenant-a",
    "security_monitoring_authority",
    "monitoring-scope",
    "confidential",
    "approver-1",
    "2026-12-31",
    "evidence-auth",
)
```

## Guardrails

The capability is defensive and compliance-first. It does not implement
destructive response, autonomous enforcement, privacy bypass, data exfiltration,
unauthorized monitoring expansion, account actions, or takedowns. AI-agent
actions that request those scopes are denied by the rule engine.

## Verification

Use focused verification during battery-constrained development:

```bash
./.venv/bin/python -m py_compile capabilities/intel/monitoring/*.py capabilities/intel/monitoring/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/intel/monitoring/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/intel/monitoring --json
./.venv/bin/apg capabilities lifecycle-audit --root capabilities/intel/monitoring --json
```
