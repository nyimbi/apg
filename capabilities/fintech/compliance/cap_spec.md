# Capability Spec: fintech_compliance

## Summary

`fintech_compliance` provides executable regulatory obligation, control,
testing, evidence, attestation, issue, remediation, report, review, and AI-agent
workflows for APG fintech applications.

## Interfaces

- Contract: `capability_contract.py`
- Service: `service.py`
- API helpers: `api.py`
- Views: `views.py`
- App entrypoint: `app.py`
- Tests: `tests/test_package_contract.py`

## Composition

Requires APG auth, audit, notifications, NLP, key management, payments,
wallets, KYC, AML, fraud, risk, and financial reporting. It emits Bytewax
lifecycle metadata through `apg.fintech.compliance.lifecycle`.

## Review Notes

Runtime methods enforce deterministic rules before mutation. External regulator
filing, document signing, external GRC suites, live ledgers, and durable worker
execution are adapter responsibilities.
