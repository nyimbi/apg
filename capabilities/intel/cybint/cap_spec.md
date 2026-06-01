# Capability Spec: intel_cybint

## Summary

`intel_cybint` provides executable lawful authority, indicator, sighting,
enrichment, threat profile, risk assessment, incident link, dissemination,
review, and AI-agent workflows for defensive APG cyber-intelligence
applications.

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
`apg.intel.cybint.lifecycle`.

## Review Notes

Runtime methods enforce deterministic rules before mutation and key state by
tenant plus record ID. Offensive activity, exploit generation, live SIEM/EDR/
SOAR integrations, malware sandboxes, vulnerability scanners, ticketing,
containment execution, storage, GraphRAG projection, dissemination delivery,
and durable Bytewax workers are adapter responsibilities.
