# GEOS Geo-Spatial Services Capability

GEOS is APG's governed location-intelligence capability. It gives generated
applications a dependency-light way to compose event-source registration,
geofence creation, location-event processing, territory planning, spatial
analytics, privacy review, AI location-agent coordination, tenant isolation,
visual theming, and Bytewax lifecycle events.

The local package is intentionally executable without live map providers,
H3/GEOS native extensions, routing engines, data warehouses, web servers, or
stream processors. Production geo engines attach through adapters declared by
the capability contract.

## What GEOS Provides

- Tenant-scoped event-source registration with consent model and data residency
  policy.
- Tenant-scoped geofences with owner, active rule, geometry validation, and
  large-polygon review.
- Location-event processing with source registration, consent, accuracy,
  sensitive-location review, and geofence matching.
- Territory management with overlap review.
- Spatial analytics with spatial-index and aggregation-privacy guardrails.
- First-class AI location agents for Codex, Claude Code, OpenCode, Pi, and
  future runtimes.
- Geofence state changes with reason and audit evidence.
- Framework-neutral API helpers and UI view models.
- Visual theme metadata and Bytewax lifecycle stream metadata.

## Main Files

- `SPECIFICATION.md`: functional requirements, lifecycle, rules, UI, and
  adapter boundaries.
- `PLAN.md`: implementation sequencing and review checklist.
- `capability_contract.py`: executable configuration, rules, UI routes, theme,
  supported location-agent runtimes, and Bytewax stream metadata.
- `service.py`: dependency-light `GeosService` facade at the package boundary,
  plus provider-oriented adapter classes for future integrations.
- `api.py`: package API helpers plus legacy FastAPI integration surfaces.
- `views.py`: dashboard, map, geofence, event, analytics, agent, audit, and
  settings view models.

## Basic Usage

```python
from capabilities.common.geos.service import GeosService

service = GeosService()
service.register_event_source(
	"src-mobile",
	"tenant-a",
	"Mobile GPS",
	"mobile",
	"explicit",
	"ke-residency",
)
service.create_geofence(
	"yard",
	"tenant-a",
	"Operations Yard",
	"fleet-ops",
	{
		"type": "circle",
		"center": {"latitude": -1.286389, "longitude": 36.817223},
		"radius_meters": 500,
	},
)
event = service.process_location_event(
	"evt-1",
	"tenant-a",
	"src-mobile",
	"vehicle-1",
	"vehicle",
	-1.2865,
	36.8171,
	True,
)
```

The event is accepted only when tenant context, source registration, consent,
accuracy, privacy review, and data-residency guardrails are satisfied.

## Focused Verification

```bash
./.venv/bin/python -m py_compile capabilities/common/geos/__init__.py capabilities/common/geos/capability_contract.py capabilities/common/geos/service.py capabilities/common/geos/api.py capabilities/common/geos/views.py capabilities/common/geos/app.py capabilities/common/geos/test_capability_contract.py capabilities/common/geos/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/common/geos/test_capability_contract.py capabilities/common/geos/tests/test_package_contract.py
./.venv/bin/python -c "from capabilities.common.geos import app; r=app.self_test(); print(r); assert r['passed']"
./.venv/bin/apg capabilities implementation-audit --root capabilities/common/geos --json
./.venv/bin/apg capabilities publish-plan capabilities/common/geos --json
```
