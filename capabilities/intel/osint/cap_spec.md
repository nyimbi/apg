# Capability Spec: intel_osint

## Summary

`intel_osint` provides executable open-source intelligence requirement, source,
collection plan, evidence, triage, assessment, dissemination, review, and
AI-agent workflows for APG intelligence applications.

## Interfaces

- Contract: `capability_contract.py`
- Service: `service.py`
- API helpers: `api.py`
- Views: `views.py`
- App entrypoint: `app.py`
- Tests: `tests/test_package_contract.py`

## Composition

Requires APG auth, audit, notifications, NLP, Intelligence Crawler, Search,
Graph Data Management, and Retrieval-Augmented Generation. It emits Bytewax
lifecycle metadata through `apg.intel.osint.lifecycle`.

## Review Notes

Runtime methods enforce deterministic rules before mutation. Live source APIs,
crawler execution, source-term verification, GraphRAG projection, search-index
queries, dissemination delivery, and durable Bytewax workers are adapter
responsibilities.
