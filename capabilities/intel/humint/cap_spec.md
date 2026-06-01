# Capability Spec: intel_humint

## Summary

`intel_humint` provides executable lawful authority, source-management, contact
planning, contact-report, debriefing, reliability, lead, dissemination, review,
and AI-agent workflows for APG intelligence applications.

## Interfaces

- Contract: `capability_contract.py`
- Service: `service.py`
- API helpers: `api.py`
- Views: `views.py`
- App entrypoint: `app.py`
- Tests: `tests/test_package_contract.py`

## Composition

Requires APG auth, audit, notifications, NLP, Graph Data Management, and
Retrieval-Augmented Generation. It emits Bytewax lifecycle metadata through
`apg.intel.humint.lifecycle`.

## Review Notes

Runtime methods enforce deterministic rules before mutation and key state by
tenant plus record ID. Field operations, source recruitment, covert
communications, payment handling, physical security, identity protection,
partner case systems, storage, GraphRAG projection, dissemination delivery, and
durable Bytewax workers are adapter responsibilities.
