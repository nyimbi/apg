# SCM Vendor Management

Vendor Management is the APG capability packet for supplier master records, qualification, onboarding, performance, risk, compliance, contracts, communications, portal users, scorecards, and vendor-focused AI agents.

The package is dependency-light at import time. Web frameworks, databases, procurement systems, contract repositories, document stores, risk providers, and notification services attach through APG composition adapters.

## What It Provides

- `vendor_profile_lifecycle` for vendor master records and ownership.
- `vendor_qualification_lifecycle` for criteria, reviewer, and score capture.
- `vendor_onboarding_workflow` for checklist-driven onboarding.
- `vendor_performance_lifecycle` for quality, delivery, cost, service, sustainability, and innovation scoring.
- `vendor_risk_lifecycle` for low, medium, high, and critical risk records.
- `vendor_compliance_lifecycle` for framework evidence and status review.
- `vendor_contract_lifecycle` for approved commercial records.
- `vendor_communication_lifecycle` and `vendor_portal_lifecycle` for engagement.
- `vendor_scorecard_service` for performance/risk/compliance summaries.
- `vendor_agents` for Codex, Claude Code, OpenCode, and Pi review agents.
- APG Python UI routes, deterministic guardrails, compact theme tokens, semantic metadata, and Bytewax event metadata.

## Runtime Surface

- `capability_contract.py` defines composition contracts, configuration, rules, UI, theme, and streaming.
- `service.py` implements the executable dependency-light lifecycle.
- `api.py` exposes adapter-neutral function wrappers.
- `views.py` provides dashboard, workbench, rule, settings, and agent models.
- `app.py` exposes `semantic_model()`, `component_manifest()`, and `self_test()`.

## Example

```python
from capabilities.scm.ven.service import VendorManagementLifecycleService

service = VendorManagementLifecycleService()
vendor = service.create_vendor(
	"vendor-1",
	"tenant-a",
	"ACME",
	"Acme Supplies",
	"distributor",
	"industrial",
	"KE",
	"owner-1",
)
service.qualify_vendor("qual-1", "tenant-a", vendor["id"], ["tax", "capacity"], "reviewer-1", 82)
service.onboard_vendor("onboard-1", "tenant-a", vendor["id"], ["profile", "banking", "insurance"], "owner-1")
performance = service.record_performance("perf-1", "tenant-a", vendor["id"], "2026-Q2", {"quality": 88, "delivery": 91})
risk = service.record_risk("risk-1", "tenant-a", vendor["id"], "operational", "medium", "capacity concentration")
service.create_scorecard("score-1", "tenant-a", vendor["id"], "2026-Q2", performance["id"], risk["id"], [], "analyst-1")
```

## Composition Notes

This packet requires auth, audit, notification, composition config, workflow, procurement, sourcing, contract, document, risk-policy, and supplier master-data capabilities. The local service records lifecycle state in memory for executable packaging; production deployments should attach durable stores and integration adapters.

All batch and lifecycle event metadata uses Bytewax. Non-Bytewax batch routing is rejected by the rules.

## Focused Verification

```bash
./.venv/bin/python -m py_compile capabilities/scm/ven/__init__.py capabilities/scm/ven/capability_contract.py capabilities/scm/ven/service.py capabilities/scm/ven/api.py capabilities/scm/ven/views.py capabilities/scm/ven/app.py capabilities/scm/ven/tests/conftest.py capabilities/scm/ven/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/scm/ven/tests/test_package_contract.py
./.venv/bin/apg capabilities publish-plan capabilities/scm/ven --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/scm/ven --json
```
