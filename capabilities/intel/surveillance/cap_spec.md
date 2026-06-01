# Capability Spec: intel_surveillance

## Summary

`intel_surveillance` provides executable lawful authority, program, monitored
asset, sensor, observation, alert, risk assessment, referral, dissemination,
review, and AI-agent workflows for APG digital-surveillance applications.

## Interfaces

- Contract: `capability_contract.py`
- Service: `service.py`
- API helpers: `api.py`
- Views: `views.py`
- App entrypoint: `app.py`
- Tests: `tests/test_package_contract.py`

## Composition

Requires APG auth, audit, notifications, NLP, Computer Vision, Graph Data
Management, Retrieval-Augmented Generation, and geospatial capabilities. It
emits Bytewax lifecycle metadata through
`apg.intel.surveillance.lifecycle`.

## Review Notes

Runtime methods enforce deterministic rules before mutation and key state by
tenant plus record ID. Live camera, endpoint, network, access-control,
telemetry, geospatial, storage, biometric, notification, and durable Bytewax
integrations are adapter responsibilities. Covert tracking, stalking, spyware,
credential capture, bypass, biometric identification, and exfiltration scopes
are explicitly denied for AI-agent actions.
