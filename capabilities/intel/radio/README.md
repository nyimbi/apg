# APG Radio Intelligence Listener

`intel_radio` is an executable APG capability for lawful, passive
radio-monitoring workflows. It can be composed into generated APG applications
that need public-safety monitoring, spectrum management, interference review,
emergency signal triage, asset-signal tracking, or partner-feed analysis.

## What It Provides

- Authority, band plan, receiver, collection session, signal observation,
  transmission classification, event assessment, referral, dissemination,
  review, and AI-agent workflows.
- Deterministic rules that enforce tenant context, lawful authority, frequency
  bounds, receiver calibration, evidence, approvals, Bytewax lifecycle routing,
  and AI-agent guardrails.
- API helpers and view models that generated Python applications can call
  without a web framework dependency.
- UI route metadata and compact theme tokens for generated application screens.
- A publishable `app.py` entrypoint with self-test and semantic-model output.

## Local Usage

```bash
./.venv/bin/python capabilities/intel/radio/app.py
./.venv/bin/pytest -q capabilities/intel/radio/tests/test_package_contract.py
./.venv/bin/apg capabilities inspect intel_radio --json
```

Generated applications can import the service directly:

```python
from capabilities.intel.radio import RadioIntelligenceListenerService

service = RadioIntelligenceListenerService()
authority = service.record_authority(
    "auth-1",
    "tenant-a",
    "spectrum_license",
    "license-scope",
    "confidential",
    "approver-1",
    "2026-12-31",
    "evidence-auth",
)
```

## Guardrails

The capability is defensive, passive, and compliance-first. It does not
implement transmission, jamming, spoofing, interference, decryption, protected
communication interception, or unauthorized collection. AI-agent actions that
request those scopes are denied by the rule engine.

## Verification

Use focused verification during battery-constrained development:

```bash
./.venv/bin/python -m py_compile capabilities/intel/radio/*.py capabilities/intel/radio/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/intel/radio/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/intel/radio --json
./.venv/bin/apg capabilities lifecycle-audit --root capabilities/intel/radio --json
```
