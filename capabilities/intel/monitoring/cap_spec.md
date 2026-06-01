# Capability Spec: intel_monitoring

## Summary

`intel_monitoring` provides executable lawful authority, monitoring policy,
source, watch, event, signal, incident, referral, dissemination, review, and
AI-agent workflows for APG real-time-monitoring applications.

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
metadata through `apg.intel.monitoring.lifecycle`.

## Review Notes

Runtime methods enforce deterministic rules before mutation and key state by
tenant plus record ID. Live collectors, stream connectors, storage backends,
notification delivery, case-management writes, response automation, GraphRAG
projection, and durable Bytewax workers are adapter responsibilities.
Destructive actions, autonomous enforcement, privacy bypass, data exfiltration,
unauthorized scope expansion, account actions, and takedown scopes are denied
for AI-agent actions.
