# Capability Spec: fintech_crypto

## Summary

`fintech_crypto` provides executable digital asset, custody, balance, order,
trade, transfer, screening, price, review, and AI-agent workflows for APG
fintech applications.

## Interfaces

- Contract: `capability_contract.py`
- Service: `service.py`
- API helpers: `api.py`
- Views: `views.py`
- App entrypoint: `app.py`
- Tests: `tests/test_package_contract.py`

## Composition

Requires APG auth, audit, notifications, NLP, key management, Blockchain
Services, wallets, risk, compliance, RegTech, AML, and KYC contracts. It emits
Bytewax lifecycle metadata through `apg.fintech.crypto.lifecycle`.

## Review Notes

Runtime methods enforce deterministic rules before mutation. Live exchange
connectivity, custody-provider APIs, order routing, transaction signing,
private-key custody, chain RPC, market-data feeds, and durable worker execution
are adapter responsibilities.
