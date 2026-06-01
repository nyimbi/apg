# APG Geospatial Intelligence

`intel_geoint` is the APG package-backed capability for governed geospatial
intelligence applications. It composes authorities, areas of interest,
imagery/geospatial sources, collection plans, observations, features, change
detections, assessments, dissemination, reviews, Bytewax lifecycle metadata,
UI/view models, visual theming, and provider-neutral AI-agent automation.

## What It Provides

- Lawful authority workflow with scope, approver, expiry, classification, and
  evidence.
- Area-of-interest registry with geometry references, owner, authority, and
  evidence.
- Imagery/geospatial source registry with source type, sensor type, resolution
  class, owner, authority, and evidence.
- Collection plan workflow with authority/source/area matching, retention,
  approval, and evidence.
- Observation, feature, change, assessment, dissemination, and review
  workflows.
- Deterministic rule guardrails enforced before service state changes.
- AI-agent registration for `codex`, `claude_code`, `opencode`, and `pi`.
- Bytewax lifecycle metadata through `apg.intel.geoint.lifecycle`.

## Use The Service

```python
from capabilities.intel.geoint import GeospatialIntelligenceService

service = GeospatialIntelligenceService()
authority = service.record_authority(
	"auth-1",
	"tenant-a",
	"mission_order",
	"scope://mission",
	"secret",
	"approver-1",
	"2026-12-31",
	"evidence://authority",
)
area = service.record_area(
	"area-1",
	"tenant-a",
	"Port corridor",
	"geojson://area-1",
	"secret",
	"owner-1",
	authority["id"],
	"evidence://area",
)
```

Invalid operations raise `PermissionError` with rule reasons such as
`tenant_context_required`, `lawful_authority_required`,
`area_authority_mismatch`, `source_authority_mismatch`,
`targeting_or_harmful_scope_denied`, or `bytewax_event_stream_required`.

## Compose In Generated Apps

- Contract: `capability_contract.py`
- Service: `service.py`
- API helpers: `api.py`
- Views: `views.py`
- App entrypoint: `app.py`
- Tests: `tests/test_package_contract.py`

## Verify Locally

```bash
./.venv/bin/pytest -q capabilities/intel/geoint/tests/test_package_contract.py
./.venv/bin/python capabilities/intel/geoint/app.py
./.venv/bin/apg capabilities inspect intel_geoint --json
./.venv/bin/apg capabilities publish-plan capabilities/intel/geoint --json
./.venv/bin/apg capabilities lifecycle-audit --root capabilities/intel/geoint --json
```

## Production Boundaries

Live satellite/aerial tasking, sensor control, weapon targeting, harmful
operational planning, map tile rendering, GIS engines, large imagery storage,
computer vision extraction, geocoding, routing, dissemination delivery, GraphRAG
projection, and durable Bytewax topology execution stay behind adapters.
