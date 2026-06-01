# Capability Spec: intel_geoint

## Summary

`intel_geoint` provides executable lawful authority, area, source, collection
plan, observation, feature, change detection, assessment, dissemination, review,
and AI-agent workflows for APG intelligence applications.

## Interfaces

- Contract: `capability_contract.py`
- Service: `service.py`
- API helpers: `api.py`
- Views: `views.py`
- App entrypoint: `app.py`
- Tests: `tests/test_package_contract.py`

## Composition

Requires APG auth, audit, notifications, NLP, Graph Data Management,
Retrieval-Augmented Generation, and geospatial services. It emits Bytewax
lifecycle metadata through `apg.intel.geoint.lifecycle`.

## Review Notes

Runtime methods enforce deterministic rules before mutation and key state by
tenant plus record ID. Live tasking, sensor control, targeting, GIS engines,
imagery storage, computer vision extraction, geocoding, routing, GraphRAG
projection, dissemination delivery, and durable Bytewax workers are adapter
responsibilities.
