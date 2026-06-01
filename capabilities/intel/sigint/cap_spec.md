# Capability Spec: intel_sigint

## Summary

`intel_sigint` provides executable lawful authority, source, collection task,
observation, processing, pattern, assessment, review, and AI-agent workflows for
APG intelligence applications.

## Interfaces

- Contract: `capability_contract.py`
- Service: `service.py`
- API helpers: `api.py`
- Views: `views.py`
- App entrypoint: `app.py`
- Tests: `tests/test_package_contract.py`

## Composition

Requires APG auth, audit, notifications, NLP, Radio Intelligence Listener,
Intelligence Crawler, Graph Data Management, and Retrieval-Augmented
Generation. It emits Bytewax lifecycle metadata through
`apg.intel.sigint.lifecycle`.

## Review Notes

Runtime methods enforce deterministic rules before mutation. Live receiver
control, lawful-intercept gateways, telecom systems, satellite feeds,
decryptors, speech processing, direction finding, search indexes, GraphRAG
projection, dissemination delivery, and durable Bytewax workers are adapter
responsibilities.
