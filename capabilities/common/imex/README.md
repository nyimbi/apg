# APG IMEX - Import/Export

IMEX is the APG capability for governed import, export, and migration
workflows. It gives generated applications a dependency-light runtime for
building transfer jobs while preserving integration points for ETLP, CONN,
AUTH, AUDL, MONI, KEYM, ENCR, and Bytewax.

## What It Provides

- Tenant-scoped transfer endpoints bound to CONN-managed connections.
- Schema mapping profiles with source profiling, mapping, and quality gate
  references.
- Import, export, and migration jobs with owner, checksum, format, data
  classification, and environment metadata.
- Preview validation before execution.
- Transfer runs with checkpoint, monitoring, quality, audit, replay, and
  completion state.
- Artifact publication with checksum and retention metadata.
- Review queues for destination approvals, quality reviews, capacity reviews,
  purge reviews, and owner transfer.
- UI model functions for dashboards, job design, mappings, transfer monitor,
  validation, import/export workbenches, approvals, artifacts, audit, and
  settings.

## Main Files

- `SPECIFICATION.md` - functional contract and lifecycle definition.
- `PLAN.md` - implementation plan for this packet.
- `capability_contract.py` - executable configuration, rule, UI, adapter, and
  theme contract.
- `imex_runtime.py` - dependency-light runtime service for generated apps.
- `view_models.py` - screen-ready generated-app UI models.
- `app.py` - dynamic package semantic model and self-test.
- `test_capability_contract.py` and `tests/test_package_contract.py` - focused
  package proof.

## Generated-App Usage

```python
from capabilities.common.imex import ImexService

service = ImexService()
service.register_endpoint(
	"source-crm",
	"tenant-a",
	"CRM Export",
	"connection",
	"conn://crm",
	"data",
)
service.register_endpoint(
	"warehouse",
	"tenant-a",
	"Warehouse",
	"connection",
	"conn://warehouse",
	"data",
)
service.create_mapping_profile(
	"crm-map",
	"tenant-a",
	"CRM Mapping",
	"profiles/crm.json",
	"mappings/crm_to_wh.json",
	"quality/crm",
)
service.create_job(
	"crm-migration",
	"tenant-a",
	"CRM Migration",
	"migration",
	"source-crm",
	"warehouse",
	"parquet",
	"data",
	"production",
	"crm-map",
	"sha256:abc",
	etlp_plan_ref="etlp://crm-migration",
)
service.validate_preview("tenant-a", "crm-migration", quality_score=0.99)
run = service.execute_job(
	"tenant-a",
	"crm-migration",
	"run-001",
	record_count=50000,
	approval_recorded=True,
)
service.complete_run("tenant-a", run["id"], records_processed=50000, quality_score=0.99)
```

## Guardrails

IMEX blocks missing tenant context, unsupported formats, missing endpoints,
missing mappings, missing checksums, missing preview validation, unapproved
production transfers, unencrypted sensitive exports, unmonitored large
transfers, missing checkpoints, invalid records without quarantine, replay
without idempotency, artifact publication without retention, and destructive
purge without review.

## Verification

Use focused checks while developing on battery:

```bash
./.venv/bin/python -m py_compile capabilities/common/imex/__init__.py capabilities/common/imex/capability_contract.py capabilities/common/imex/imex_runtime.py capabilities/common/imex/api.py capabilities/common/imex/view_models.py capabilities/common/imex/app.py capabilities/common/imex/test_capability_contract.py capabilities/common/imex/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/imex/test_capability_contract.py capabilities/common/imex/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/imex --json
./.venv/bin/apg capabilities publish-plan capabilities/common/imex --json
```
