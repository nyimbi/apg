# APG Social Media Intelligence

`intel_socint` is an executable APG capability for lawful public or authorized
social-source intelligence. It can be composed into generated APG applications
that need social monitoring, public-safety alerting, fraud and disinformation
review, crisis tracking, brand-risk analysis, or policy monitoring.

## What It Provides

- Authority, topic, source, post, signal, influence, network, referral,
  dissemination, review, and AI-agent workflows.
- Deterministic rules that enforce tenant context, lawful authority, evidence,
  terms review, approvals, Bytewax lifecycle routing, and AI-agent guardrails.
- API helpers and view models that generated Python applications can call
  without a web framework dependency.
- UI route metadata and compact theme tokens for generated application screens.
- A publishable `app.py` entrypoint with self-test and semantic-model output.

## Local Usage

```bash
./.venv/bin/python capabilities/intel/socint/app.py
./.venv/bin/pytest -q capabilities/intel/socint/tests/test_package_contract.py
./.venv/bin/apg capabilities inspect intel_socint --json
```

Generated applications can import the service directly:

```python
from capabilities.intel.socint import SocialMediaIntelligenceService

service = SocialMediaIntelligenceService()
authority = service.record_authority(
    "auth-1",
    "tenant-a",
    "legal_mandate",
    "case-scope",
    "confidential",
    "approver-1",
    "2026-12-31",
    "evidence-auth",
)
```

## Guardrails

The capability is defensive and compliance-first. It does not implement live
scraping, login/cookie collection, evasion, account automation, direct
messaging, takedown actions, harassment, doxxing, or platform-abuse workflows.
AI-agent actions that request those scopes are denied by the rule engine.

## Verification

Use focused verification during battery-constrained development:

```bash
./.venv/bin/python -m py_compile capabilities/intel/socint/*.py capabilities/intel/socint/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/intel/socint/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/intel/socint --json
./.venv/bin/apg capabilities lifecycle-audit --root capabilities/intel/socint --json
```
