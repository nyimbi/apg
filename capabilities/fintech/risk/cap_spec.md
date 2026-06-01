# Capability Spec: fintech_risk

## Summary

`fintech_risk` provides executable risk appetite, profile, exposure, control,
stress testing, breach, event, review, and AI-agent workflows for APG fintech
applications.

## Interfaces

- Contract: `capability_contract.py`
- Service: `service.py`
- API helpers: `api.py`
- Views: `views.py`
- App entrypoint: `app.py`
- Tests: `tests/test_package_contract.py`

## Composition

Requires APG auth, audit, notifications, NLP, key management, payments,
wallets, KYC, AML, fraud, analytics, and reporting. It emits Bytewax lifecycle
metadata through `apg.fintech.risk.lifecycle`.

## Review Notes

Runtime methods enforce deterministic rules before mutation. External ledgers,
market feeds, model engines, regulator filing, and durable worker execution are
adapter responsibilities.
