# Capability Spec: fintech_regtech

## Summary

`fintech_regtech` provides executable regulatory source, change, obligation
mapping, policy mapping, impact assessment, filing, submission, inquiry,
response, review, and AI-agent workflows
for APG fintech applications.

## Interfaces

- Contract: `capability_contract.py`
- Service: `service.py`
- API helpers: `api.py`
- Views: `views.py`
- App entrypoint: `app.py`
- Tests: `tests/test_package_contract.py`

## Composition

Requires APG auth, audit, notifications, NLP, key management, compliance, risk,
AML, KYC, and financial reporting. It emits Bytewax lifecycle metadata through
`apg.fintech.regtech.lifecycle`.

## Review Notes

Runtime methods enforce deterministic rules before mutation. Live regulator
portals, external regulatory feeds, signed documents, GRC suites, and durable
worker execution are adapter responsibilities.
