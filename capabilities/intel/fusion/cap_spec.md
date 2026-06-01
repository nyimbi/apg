# Capability Spec: intel_fusion

## Summary

`intel_fusion` provides executable lawful authority, fusion workspace, source,
artifact, correlation, hypothesis, assessment, referral, dissemination, review,
and AI-agent workflows for APG intelligence-fusion applications.

## Interfaces

- Contract: `capability_contract.py`
- Service: `service.py`
- API helpers: `api.py`
- Views: `views.py`
- App entrypoint: `app.py`
- Tests: `tests/test_package_contract.py`

## Composition

Requires APG auth, audit, notifications, NLP, Graph Data Management,
Retrieval-Augmented Generation, and geospatial capabilities. It emits Bytewax
lifecycle metadata through `apg.intel.fusion.lifecycle`.

## Review Notes

Runtime methods enforce deterministic rules before mutation and key state by
tenant plus record ID. Live source connectors, cross-domain data movement,
entity-resolution engines, graph writes, RAG indexing, storage backends,
dissemination delivery, and durable Bytewax workers are adapter
responsibilities. Evidence fabrication, source tampering, privacy bypass,
unsupported identity resolution, autonomous dissemination, and unapproved
attribution scopes are denied for AI-agent actions.
