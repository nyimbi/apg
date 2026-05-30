# ETLP - ETL/ELT Processing

ETLP is APG's tenant-scoped data pipeline capability. It gives generated APG
applications a composable control plane for pipeline design, datasource
registration, field mapping, execution, quality gates, lineage emission,
publishing, monitoring, and operational guardrails.

The capability currently exposes an executable contract, FastAPI runtime
controller, async pipeline service, Pydantic data models, field-mapping
support, view request/response models, package evidence, and focused regression
tests. The next packet hardens the generated-application lifecycle surface so
applications can invoke ETLP without needing the full production runtime.

## What ETLP Provides

- Pipeline registration, update, deletion, execution, monitoring, and
  cancellation.
- Datasource and transformation management for batch, streaming, and ELT
  workloads.
- Field mapping helpers for source-to-target schema mapping.
- Quality-rule definitions and quality-gate enforcement before output
  publication.
- Lineage and audit integration points for META, AUDL, MONI, MQEB, AUTH, and
  MDM.
- Deterministic policy rules for tenant context, owner assignment, production
  approval, quality gates, lineage emission, and cost review.
- UI routes for dashboard, workbench, designer, field mapper, executions,
  quality, datasources, and settings.
- Theme tokens and component contracts for generated application shells.

## Main Files

- `capability_contract.py` - ETLP configuration, rule engine, UI manifest, and
  theme contract.
- `models.py` - Pipeline, execution, datasource, transformation, quality-rule,
  schedule, and metric models.
- `service.py` - Async production-oriented pipeline orchestration service.
- `api.py` - FastAPI controller and route registration.
- `field_mapper.py` - Field schema, mapping configuration, suggestion, and
  transformation helpers.
- `views.py` - API request and response models used by the runtime controller.
- `app.py` - Publishable package entrypoint and semantic evidence.
- `SPECIFICATION.md` - Functional contract for the coherent ETLP packet.
- `PLAN.md` - Build sequence for the remaining ETLP lifecycle work.

## Using the Capability Contract

```python
from capabilities.common.etlp import register_capability
from capabilities.common.etlp.capability_contract import evaluate_capability_rules

registration = register_capability()

decision = evaluate_capability_rules({
    "tenant_context_present": True,
    "operation": "publish_output",
    "quality_gate_passed": False,
})

assert decision["decision"] == "deny"
```

## Using the Runtime Service

```python
from capabilities.common.etlp.service import ETLPService

service = ETLPService("tenant-data", "pipeline-owner")
```

The service is designed for APG runtime injection. Generated applications should
prefer the dependency-light lifecycle helpers that will be added in the next
implementation pass, while production deployments can continue to use
`ETLPService` with injected auth, audit, metadata, notification, and
collaboration services.

## Guardrails

ETLP guardrails must protect:

- Tenant isolation for every pipeline and execution operation.
- Owner assignment before pipeline execution.
- Explicit production approval before production runs.
- Quality-gate evidence before publishing pipeline output.
- Lineage emission for transformations.
- Cost review for high-estimate executions.
- Datasource approval, secret handling, retry policy, schedule review,
  backfill, replay, and destructive-delete controls as the packet is expanded.

## Adapter Boundaries

ETLP should not hardcode external engines into the generated-application control
plane. Durable execution engines, connector registries, Bytewax stream flows,
metadata stores, lineage emitters, quality profilers, AI optimizers, and
observability sinks should be configured as adapters that receive guardrail
decisions from the capability.

## Focused Verification

Battery-conscious checks for this capability:

```bash
./.venv/bin/python -m py_compile capabilities/common/etlp/__init__.py capabilities/common/etlp/capability_contract.py capabilities/common/etlp/models.py capabilities/common/etlp/service.py capabilities/common/etlp/api.py capabilities/common/etlp/field_mapper.py capabilities/common/etlp/views.py capabilities/common/etlp/app.py capabilities/common/etlp/test_capability_contract.py capabilities/common/etlp/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/etlp/test_capability_contract.py capabilities/common/etlp/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/etlp --json
./.venv/bin/apg capabilities publish-plan capabilities/common/etlp --json
```

Full repository tests, live connector execution, Bytewax runtime flows, rendered
UI checks, and performance benchmarks are intentionally left for later
verification passes.
