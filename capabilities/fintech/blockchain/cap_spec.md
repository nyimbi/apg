# Capability Spec: fintech_blockchain

## Summary

`fintech_blockchain` provides executable blockchain network, custody, smart
contract, transaction, evidence anchoring, oracle, node-health, review, and
AI-agent workflows for APG fintech applications.

## Interfaces

- Contract: `capability_contract.py`
- Service: `service.py`
- API helpers: `api.py`
- Views: `views.py`
- App entrypoint: `app.py`
- Tests: `tests/test_package_contract.py`

## Composition

Requires APG auth, audit, notifications, NLP, key management, risk, compliance,
RegTech, and wallet contracts. It emits Bytewax lifecycle metadata through
`apg.fintech.blockchain.lifecycle`.

## Review Notes

Runtime methods enforce deterministic rules before mutation. Live chain RPC
access, signing keys, custody providers, chain indexers, bridge operators,
oracle vendors, and durable worker execution are adapter responsibilities.
