# Capability Spec: intel_darkweb

## Summary

`intel_darkweb` provides executable lawful authority, monitoring program,
hidden-service source, observation, exposure indicator, marketplace risk,
threat actor, referral, dissemination, review, and AI-agent workflows for APG
dark-web-monitoring applications.

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
metadata through `apg.intel.darkweb.lifecycle`.

## Review Notes

Runtime methods enforce deterministic rules before mutation and key state by
tenant plus record ID. Live dark-web network access, crawling, credential
handling, marketplace interaction, contraband transactions, exploit
procurement, account automation, identity resolution, large-scale storage,
GraphRAG projection, dissemination delivery, and durable Bytewax workers are
adapter responsibilities. Credential use, exploit procurement, contraband
transactions, evasion, and doxxing scopes are explicitly denied for AI-agent
actions.
