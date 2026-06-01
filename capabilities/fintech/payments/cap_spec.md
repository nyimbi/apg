# Digital Payments Capability Specification Pointer

The active APG Digital Payments specification is maintained in
`SPECIFICATION.md`.

## Runtime Summary

`fintech_payments` is the dependency-light digital payments capability for
generated APG applications. It owns tenant-scoped payment accounts, payment
instruments, payment orders, risk screening evidence, authorization, capture,
refund, payout, settlement, dispute, and payment-agent lifecycle records.

## Composition Contract

Provides:

- `payment_account_lifecycle`
- `payment_instrument_vault`
- `payment_order_lifecycle`
- `risk_screening_workflow`
- `authorization_capture_refund_workflow`
- `payout_workflow`
- `settlement_reconciliation_workflow`
- `payment_dispute_workflow`
- `payment_agents`

Requires:

- `auth`
- `audl`
- `ntfy`
- `keym`
- `encr`
- `fintech_gateway`
- `cash_management`
- `accounts_receivable`

All lifecycle batches and events use Bytewax metadata through
`apg.fintech.payments.lifecycle`. AI payment agents are first-class and
provider-neutral across Codex, Claude Code, OpenCode, and Pi.

## Proof Commands

```bash
./.venv/bin/python -m py_compile capabilities/fintech/payments/__init__.py capabilities/fintech/payments/capability_contract.py capabilities/fintech/payments/models.py capabilities/fintech/payments/payments_runtime.py capabilities/fintech/payments/service.py capabilities/fintech/payments/api.py capabilities/fintech/payments/views.py capabilities/fintech/payments/app.py capabilities/fintech/payments/tests/test_package_contract.py
./.venv/bin/pytest -q capabilities/fintech/payments/tests/test_package_contract.py
./.venv/bin/python capabilities/fintech/payments/app.py
./.venv/bin/apg capabilities inspect fintech_payments --json
./.venv/bin/apg capabilities publish-plan capabilities/fintech/payments --json
./.venv/bin/apg capabilities implementation-audit --root capabilities/fintech/payments --json
./.venv/bin/apg capabilities lifecycle-audit --root capabilities/fintech/payments --json
```
