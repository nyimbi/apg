# GEOS Capability Specification Pointer

The current executable GEOS capability packet is defined by:

- `README.md` for usage and composition guidance;
- `SPECIFICATION.md` for functional requirements and guardrails;
- `PLAN.md` for implementation sequencing and review criteria;
- `capability_contract.py` for executable configuration, rules, UI, theme, and
  Bytewax streaming metadata;
- `service.py`, `api.py`, and `views.py` for the dependency-light lifecycle
  used by generated APG applications.

Legacy map-provider, routing-engine, and warehouse-specific ambitions are
adapter work. The local package proves governed event-source registration,
geofencing, location-event processing, territory planning, analytics, privacy,
AI location-agent coordination, tenant isolation, and audit evidence without
requiring live geo providers.
