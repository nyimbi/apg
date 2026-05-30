# Sustainability and ESG Management

Sustainability and ESG Management is the APG capability packet for ESG profiles, reporting frameworks, metrics, measurements, targets, supplier assessments, initiatives, risks, reports, stakeholder engagement, and ESG-focused AI agents.

The packet is dependency-light at import time. Databases, carbon feeds, supplier systems, document repositories, regulatory content, workflow engines, and notification providers attach through APG composition adapters.

## What It Provides

- `esg_profile_lifecycle` for tenant reporting entities.
- `esg_framework_lifecycle` for GRI, SASB, TCFD, ISSB, CSRD, SEC climate, and local regulatory frameworks.
- `esg_metric_lifecycle` and `esg_measurement_lifecycle` for ESG data capture with evidence and review gates.
- `esg_target_lifecycle` for absolute, intensity, reduction, and compliance targets.
- `esg_supplier_assessment_lifecycle` for supply-chain ESG scoring.
- `esg_initiative_lifecycle` and `esg_risk_lifecycle` for action tracking and governance.
- `esg_report_workflow` for approved reporting packages.
- `esg_stakeholder_lifecycle` and `esg_engagement_lifecycle` for consented engagement.
- `esg_agents` for Codex, Claude Code, OpenCode, and Pi review agents.
- APG Python UI routes, deterministic guardrails, compact theme tokens, semantic metadata, and Bytewax event metadata.

## Example

```python
from capabilities.ecd.esg.service import ESGManagementLifecycleService

service = ESGManagementLifecycleService()
profile = service.create_esg_profile("profile-1", "tenant-a", "Acme ESG", "manufacturing", "KE", 2026, "owner-1")
framework = service.add_framework("framework-1", "tenant-a", profile["id"], "gri", "2026", True, "owner-1")
metric = service.define_metric("metric-1", "tenant-a", profile["id"], "environmental", "emissions", "tco2e", "Scope 1 emissions", "owner-1")
measurement = service.record_measurement("measure-1", "tenant-a", metric["id"], "2026-Q1", 125.4, "manual", "doc-1")
service.create_report("report-1", "tenant-a", profile["id"], "quarterly", "2026-Q1", [framework["id"]], [measurement["id"]], "approver-1")
```

## Focused Verification

```bash
./.venv/bin/python -m py_compile capabilities/ecd/esg/__init__.py capabilities/ecd/esg/capability_contract.py capabilities/ecd/esg/service.py capabilities/ecd/esg/api.py capabilities/ecd/esg/views.py capabilities/ecd/esg/app.py capabilities/ecd/esg/tests/conftest.py capabilities/ecd/esg/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/ecd/esg/tests/test_package_contract.py
./.venv/bin/apg capabilities publish-plan capabilities/ecd/esg --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/ecd/esg --json
```
