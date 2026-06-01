# Capability Spec: intel_finint

## Summary

`intel_finint` provides executable lawful authority, source, subject,
transaction, pattern, risk assessment, referral, dissemination, review, and
AI-agent workflows for APG financial-intelligence applications.

## Interfaces

- Contract: `capability_contract.py`
- Service: `service.py`
- API helpers: `api.py`
- Views: `views.py`
- App entrypoint: `app.py`
- Tests: `tests/test_package_contract.py`

## Composition

Requires APG auth, audit, notifications, NLP, Graph Data Management,
Retrieval-Augmented Generation, KYC, and AML capabilities. It emits Bytewax
lifecycle metadata through `apg.intel.finint.lifecycle`.

## Review Notes

Runtime methods enforce deterministic rules before mutation and key state by
tenant plus record ID. Funds movement, account freezing, live bank/crypto
integrations, sanctions-screening engines, regulatory report submission, case
writes, storage, GraphRAG projection, dissemination delivery, and durable
Bytewax workers are adapter responsibilities.
