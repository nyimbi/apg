# Capability Spec: intel_radio

## Summary

`intel_radio` provides executable lawful authority, band plan, receiver,
collection session, signal observation, transmission classification, event
assessment, referral, dissemination, review, and AI-agent workflows for APG
radio-intelligence-listener applications.

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
lifecycle metadata through `apg.intel.radio.lifecycle`.

## Review Notes

Runtime methods enforce deterministic rules before mutation and key state by
tenant plus record ID. Live receiver control, SDR drivers, demodulation,
recording storage, geolocation, decryption, transmission, jamming, spoofing,
interference, protected-communication interception, dissemination delivery, and
durable Bytewax workers are adapter responsibilities. Transmission,
unauthorized interception, decryption, jamming, spoofing, and interference
scopes are explicitly denied for AI-agent actions.
