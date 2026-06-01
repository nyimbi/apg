# Capability Spec: intel_socint

## Summary

`intel_socint` provides executable lawful authority, topic, social source,
post evidence, signal, influence assessment, network assessment, referral,
dissemination, review, and AI-agent workflows for APG social-media-intelligence
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
Retrieval-Augmented Generation capabilities. It emits Bytewax lifecycle
metadata through `apg.intel.socint.lifecycle`.

## Review Notes

Runtime methods enforce deterministic rules before mutation and key state by
tenant plus record ID. Live social-platform APIs, login/cookie collection,
scraping, evasion, account automation, direct messaging, takedown actions,
identity resolution, large-scale search/storage, GraphRAG projection,
dissemination delivery, and durable Bytewax workers are adapter
responsibilities. Harassment, doxxing, platform-abuse, and evasion scopes are
explicitly denied for AI-agent actions.
