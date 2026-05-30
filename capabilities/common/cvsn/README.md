# APG CVSN - Computer Vision

CVSN is the APG capability for governed visual intelligence. It lets generated
applications ingest tenant-scoped image, document, and video assets; run
configured vision tasks; manage model and pipeline lifecycle; expose visual UI
models; and publish audit evidence through deterministic guardrails.

## What It Provides

- Tenant-scoped asset ingestion with source reference, MIME type, size, hash,
  asset kind, metadata, status, and audit evidence.
- Processing jobs for OCR, object detection, image classification, quality
  inspection, factory safety, video analytics, visual similarity, barcode/QR,
  facial analysis, and content moderation.
- Preflight policy checks before generated-app processing runs.
- Model registration and release with MLCM linkage, model-card evidence,
  evaluation evidence, approval evidence, and audit events.
- Pipeline registration with owner, model reference, version, and enabled tasks.
- UI view models for dashboard, assets, documents, images, video, quality,
  safety, similarity search, review, models, governance, and audit.
- Adapter configuration for AICR, MLCM, CONF, AUTH, AUDL, MONI, SRCH, object
  storage, and Bytewax event streaming.

## Main Files

- `SPECIFICATION.md` - complete functional scope for this packet.
- `PLAN.md` - implementation and review plan.
- `capability_contract.py` - executable configuration, rules, UI, adapters, and
  theme contract.
- `cvsn_runtime.py` - `CvsnService`, the dependency-light generated-app runtime.
- `view_models.py` - semantic UI view models for generated applications.
- `app.py` - dynamic package evidence and self-test.
- `tests/test_capability_contract.py` - focused executable contract coverage.
- `tests/test_package_contract.py` - package evidence and compatibility tests.

## Generated-App Usage

```python
from capabilities.common.cvsn.cvsn_runtime import CvsnService

service = CvsnService()
asset = service.ingest_asset(
	"asset-001",
	"tenant-a",
	"image",
	"image/png",
	2.5,
	"s3://tenant-a/line-camera/frame-001.png",
)
model = service.register_model(
	"model-001",
	"tenant-a",
	"Defect Detector",
	"quality_inspection",
	"mlcm://vision/defect-detector",
	"quality-team",
	"1.0.0",
	"model-card://vision/defect-detector",
)
pipeline = service.register_pipeline(
	"pipe-001",
	"tenant-a",
	"Factory Quality Pipeline",
	"quality-team",
	model["id"],
	"1.0.0",
	["quality_inspection", "object_detection"],
)
job = service.run_job(
	"job-001",
	"tenant-a",
	asset["id"],
	pipeline["tasks"][0],
	"operator-1",
	inspection_plan_attached=True,
	defect_taxonomy_attached=True,
)
```

## Guardrails

CVSN blocks missing tenant context, assets without source references,
unsupported media types, oversized assets, missing asset hashes, disabled tasks,
jobs without accountable operators, OCR against video assets, video analytics
against non-video assets, quality inspection without plan or defect taxonomy,
critical defects without alerting, high-severity factory hazards without
alerting, unacknowledged critical incidents, facial analysis without biometric
consent or anonymization, biometric retention beyond the default limit, content
moderation without policy, low-confidence results without review, large batches
without async queueing, batches over configured limits, long video clips, video
analytics without sampling policy, model registration without MLCM linkage or
model card, model release without evaluation or approval, cross-tenant
processing, state changes without audit evidence, and non-Bytewax vision event
streams.

## Focused Verification

```bash
./.venv/bin/python -m py_compile capabilities/common/cvsn/__init__.py capabilities/common/cvsn/capability_contract.py capabilities/common/cvsn/cvsn_runtime.py capabilities/common/cvsn/view_models.py capabilities/common/cvsn/app.py capabilities/common/cvsn/tests/test_capability_contract.py capabilities/common/cvsn/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/cvsn/tests/test_capability_contract.py capabilities/common/cvsn/tests/test_package_contract.py
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/cvsn --json
./.venv/bin/apg capabilities publish-plan capabilities/common/cvsn --json
```
