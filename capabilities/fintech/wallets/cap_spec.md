# Digital Wallets Capability Specification Pointer

The active APG Digital Wallets specification is maintained in `SPECIFICATION.md`.

## Runtime Summary

`fintech_wallets` is the dependency-light customer and merchant wallet
capability for generated APG applications. It owns tenant-scoped wallets,
stored-value ledger entries, wallet instruments, wallet credits/debits,
wallet-to-wallet transfers, holds, limits, and provider-neutral wallet-agent
evidence.

It composes with `walt` for common wallet/payment core semantics and with
`fintech_payments` for payment order and money-movement workflows.

## Composition Contract

Provides:

- `wallet_lifecycle`
- `stored_value_ledger`
- `wallet_instrument_registry`
- `wallet_transfer_workflow`
- `wallet_hold_workflow`
- `wallet_limit_governance`
- `wallet_agent_workflow`

Requires:

- `auth`
- `audl`
- `ntfy`
- `walt`
- `fintech_payments`
- `fintech_gateway`
- `keym`

All lifecycle batches and events use Bytewax metadata through
`apg.fintech.wallets.lifecycle`.

## Proof Commands

```bash
./.venv/bin/python -m py_compile capabilities/fintech/wallets/__init__.py capabilities/fintech/wallets/capability_contract.py capabilities/fintech/wallets/models.py capabilities/fintech/wallets/wallets_runtime.py capabilities/fintech/wallets/service.py capabilities/fintech/wallets/api.py capabilities/fintech/wallets/views.py capabilities/fintech/wallets/app.py capabilities/fintech/wallets/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/fintech/wallets/tests/test_package_contract.py
./.venv/bin/python capabilities/fintech/wallets/app.py
./.venv/bin/apg capabilities inspect fintech_wallets --json
./.venv/bin/apg capabilities publish-plan capabilities/fintech/wallets --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/fintech/wallets --json
./.venv/bin/apg capabilities lifecycle-audit --root capabilities/fintech/wallets --json
```
