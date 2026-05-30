# ESGC ESG and Carbon Tracking

`esgc` is the APG common ESG and carbon tracking capability. It lets generated
applications compose tenant-scoped emissions inventories, factor libraries,
activity emissions, sustainability reports, reduction targets, compliance
evidence, Bytewax stream governance, visual theme metadata, and AI-agent
assistance.

The package is dependency-light. It defines the executable service, rule
engine, UI route metadata, theme metadata, Bytewax stream declaration, API
helpers, view models, and semantic evidence. Meter integrations, forecasting
models, compliance filings, durable audit stores, geospatial providers, and
stream-worker deployments are adapter responsibilities.

## What It Provides

- Emissions inventory with organization owner, reporting year, boundary,
  geospatial boundary, and compliance framework.
- Approved emission factor library with source evidence, versioning, scope,
  units, and conversion rates.
- Activity data recording with evidence, anomaly review, scope classification,
  and carbon dioxide equivalent calculation.
- Sustainability report publishing with approval, compliance mapping, and audit
  evidence.
- Reduction target tracking with baseline, target year, target reduction, and
  progress calculation.
- AI ESGC-agent registration for Codex, Claude Code, OpenCode, Pi, and future
  runtimes behind the same contract.
- Bytewax stream guardrail for batch ESG mutation.
- UI routes and visual theme tokens for generated APG applications.

## Quick Use

```python
from capabilities.common.esgc import EsgcService

service = EsgcService()

service.create_inventory(
    inventory_id="inventory-2026",
    tenant_id="tenant-acme",
    organization="Acme Manufacturing",
    owner="sustainability-lead",
    reporting_year=2026,
    boundary_ref="boundary:operations",
    geospatial_boundary="geos:ke-operations",
    compliance_framework="GHG Protocol",
)

service.register_factor(
    factor_id="factor-grid-ke",
    tenant_id="tenant-acme",
    name="Kenya grid electricity",
    scope="scope_2",
    unit="kwh",
    co2e_per_unit=0.00025,
    source="national-grid-factor",
    source_evidence="audl:evidence-grid-2026",
    version="2026.1",
    approved_source=True,
)
```

## AI Agent Registration

AI agents are first-class ESG contributors only after registration:

```python
agent = service.register_esgc_agent(
    tenant_id="tenant-acme",
    name="Report reviewer",
    runtime="codex",
    role="report_reviewer",
    scope="review report evidence, approval, and compliance mapping",
    contribution_disclosed=True,
)
```

Supported runtimes are `codex`, `claude_code`, `opencode`, and `pi`. Supported
roles are `inventory_reviewer`, `factor_reviewer`, `activity_reviewer`,
`report_reviewer`, and `target_reviewer`.

## Guardrails

The deterministic rules deny or require review when:

- tenant context is missing;
- inventory owner or reporting boundary is missing;
- factor source is not approved;
- factor source evidence or version is missing;
- activity evidence is missing;
- activity references a factor with a different unit;
- report approval, compliance mapping, or audit evidence is missing;
- reduction target baseline is missing;
- activity anomaly lacks review;
- an AI ESGC agent is unregistered, unsupported, unscoped, or undisclosed;
- lifecycle state changes lack audit evidence;
- batch ESG mutation does not use Bytewax.

## Bytewax Batch Mutation

Batch ESG mutation must use the Bytewax event stream:

```python
allowed = service.validate_batch_esgc_mutation("bytewax")
blocked = service.validate_batch_esgc_mutation("other-stream")

assert allowed["decision"] == "allow"
assert blocked["decision"] == "deny"
```

The contract declares topic `apg.esgc.lifecycle` and state for inventories,
factors, activities, reports, targets, ESGC agents, and audit events.

## Composition

Generated APG applications should compose `esgc` through:

- capability ID: `esgc`;
- provided services: emissions inventory, factor library, activity emissions,
  sustainability reporting, target tracking, ESG evidence, and ESGC agents;
- required services: `auth`, `conf`, `audl`, `geos`, `pred`, and `comp`;
- API prefix: `/esgc/api/v1`;
- UI routes: dashboard, emissions, factors, data sources, reports, targets,
  agents, rules, audit, and settings;
- theme: `esgc_sustainability_ops`;
- stream processor: `bytewax`.

## Proof Commands

```bash
./.venv/bin/python -m py_compile capabilities/common/esgc/__init__.py capabilities/common/esgc/capability_contract.py capabilities/common/esgc/models.py capabilities/common/esgc/service.py capabilities/common/esgc/api.py capabilities/common/esgc/views.py capabilities/common/esgc/app.py capabilities/common/esgc/test_capability_contract.py
./.venv/bin/pytest -q capabilities/common/esgc/test_capability_contract.py
./.venv/bin/python -c "from capabilities.common.esgc import EsgcService; service = EsgcService(); service.register_esgc_agent('tenant-proof', 'Proof agent', 'codex', 'report_reviewer', 'review reports'); print(service.dashboard_summary('tenant-proof'))"
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/esgc --json
./.venv/bin/apg capabilities publish-plan capabilities/common/esgc --json
```
