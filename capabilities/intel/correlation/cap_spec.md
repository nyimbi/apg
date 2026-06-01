# Capability Spec: intel_correlation

## Summary

`intel_correlation` provides executable lawful authority, correlation
workspace, source, entity, observation, rule, run, cluster, resolution decision,
referral, review, and AI-agent workflows for APG data-correlation
applications.

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
lifecycle metadata through `apg.intel.correlation.lifecycle`.

## Review Notes

Runtime methods enforce deterministic rules before mutation and key state by
tenant plus record ID. Live entity-resolution engines, graph writes, fuzzy
matching providers, geospatial joins, RAG indexing, storage backends,
notification delivery, and durable Bytewax workers are adapter
responsibilities. Unapproved identity merge, source tampering, privacy bypass,
evidence fabrication, autonomous referral, and unreviewed high-impact match
scopes are denied for AI-agent actions.
