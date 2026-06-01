# Capability Spec: fintech_defi

## Summary

`fintech_defi` provides executable DeFi protocol, position, action, yield
strategy, reward, governance, risk, review, and AI-agent workflows for APG
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
Services, Cryptocurrency Services, wallets, risk, compliance, RegTech, AML, and
KYC contracts. It emits Bytewax lifecycle metadata through
`apg.fintech.defi.lifecycle`.

## Review Notes

Runtime methods enforce deterministic rules before mutation. Live protocol RPC,
transaction signing, private-key custody, oracle feeds, bridge execution,
liquidation execution, governance submission, MEV protection, and durable
Bytewax worker execution are adapter responsibilities.
