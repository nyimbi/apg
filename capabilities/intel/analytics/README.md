# APG Intelligence Analytics

`intel_analytics` is an executable APG capability for governed,
evidence-backed intelligence analytics. It can be composed into generated APG
applications that need threat analytics, fraud analytics, public-safety
analytics, incident analytics, strategic analytics, operational analytics, or
risk analytics.

## What It Provides

- Authority, workspace, dataset, feature-set, model, run, insight, dashboard,
  narrative, recommendation, review, and AI-agent workflows.
- Deterministic rules that enforce tenant context, lawful authority, dataset
  lineage, model validation, evidence, approvals, Bytewax lifecycle routing,
  and AI-agent guardrails.
- API helpers and view models that generated Python applications can call
  without a web framework dependency.
- UI route metadata and compact theme tokens for generated application screens.
- A publishable `app.py` entrypoint with self-test and semantic-model output.

## Local Usage

```bash
./.venv/bin/python capabilities/intel/analytics/app.py
./.venv/bin/pytest -q capabilities/intel/analytics/tests/test_package_contract.py
./.venv/bin/apg capabilities inspect intel_analytics --json
```

Generated applications can import the service directly:

```python
from capabilities.intel.analytics import IntelligenceAnalyticsService

service = IntelligenceAnalyticsService()
authority = service.record_authority(
    "auth-1",
    "tenant-a",
    "mission_order",
    "analytics-scope",
    "confidential",
    "approver-1",
    "2026-12-31",
    "evidence-auth",
)
```

## Guardrails

The capability is evidence-led and compliance-first. It does not implement
hallucinated insights, training-data leakage, privacy bypass, unsupported
automated decisions, unapproved model deployment, autonomous dissemination, or
cross-tenant analytics. AI-agent actions that request those scopes are denied
by the rule engine.

## Verification

Use focused verification during battery-constrained development:

```bash
./.venv/bin/python -m py_compile capabilities/intel/analytics/*.py capabilities/intel/analytics/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/intel/analytics/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/intel/analytics --json
./.venv/bin/apg capabilities lifecycle-audit --root capabilities/intel/analytics --json
```
