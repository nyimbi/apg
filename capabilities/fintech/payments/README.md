# Digital Payments

Digital Payments is the APG capability for composing executable payment
experiences into generated applications without requiring live payment
processors, web frameworks, databases, or provider SDKs at import time.

The package owns the application-facing payment lifecycle: account creation,
instrument registration, payment order creation, risk screening, authorization,
capture, refunds, payouts, settlement reconciliation, disputes, and governed AI
payment agents. Live gateways, ledgers, vaults, cash management, accounts
receivable, notifications, and audit sinks remain adapter integrations behind
the contract.

## Runtime Files

- `capability_contract.py`: configuration, rules, routes, theme, streaming,
  provides, and requires.
- `models.py`: dependency-light dataclasses for accounts, instruments, orders,
  and evidence.
- `service.py`: in-memory executable lifecycle and guardrails.
- `api.py`: process-local helper functions for generated applications.
- `views.py`: framework-neutral screen/view models.
- `app.py`: semantic model, component manifest, and self-test.
- `tests/test_package_contract.py`: focused contract, lifecycle, guardrail,
  API, view, and app tests.

## Public Lifecycle

1. `open_payment_account`
2. `register_instrument`
3. `create_payment_order`
4. `screen_payment_risk`
5. `authorize_payment`
6. `capture_payment`
7. `refund_payment`
8. `schedule_payout`
9. `record_settlement`
10. `open_dispute`
11. `register_payment_agent`
12. `validate_batch`
13. `dashboard_summary`

## Guardrails

The deterministic rule engine denies or requires review for missing tenant
context, missing write policy, unsupported currencies, unsupported instrument
types, missing token references, non-positive payment amounts, missing payment
accounts or instruments, high-risk payments without review, blocked risk
authorization, missing provider references, high-value authorization without
approval, overcapture, overrefund, missing payout destinations, settlement
variance without review, dispute ownership gaps, non-Bytewax lifecycle batches,
unsupported AI-agent runtimes or roles, and privileged agent actions without
human approval.

## AI Payment Agents

Payment agents are first-class lifecycle participants. Supported runtimes are
`codex`, `claude_code`, `opencode`, and `pi`. Supported roles include payment
operations, risk, settlement, dispute, and provider reconciliation reviewers.
Agents can prepare and recommend actions, but privileged actions require
recorded human approval.

## Bytewax Streaming

All batch and lifecycle metadata uses:

- processor: `bytewax`
- stream: `apg.fintech.payments.lifecycle`
- key: `tenant_id`

Kafka is intentionally not part of this package boundary.

## Example

```python
from capabilities.fintech.payments import DigitalPaymentsService

svc = DigitalPaymentsService()
acct = svc.open_payment_account("acct-1", "tenant-a", "customer-1", "KES")
inst = svc.register_instrument("inst-1", "tenant-a", acct["id"], "mobile_money", "vault://mpesa/customer-1")
order = svc.create_payment_order("pay-1", "tenant-a", acct["id"], inst["id"], 1500, "KES", "merchant-1")
svc.screen_payment_risk("risk-1", "tenant-a", order["id"], "low", "0.12")
svc.authorize_payment("auth-1", "tenant-a", order["id"], "provider://mpesa")
svc.capture_payment("cap-1", "tenant-a", order["id"], 1500)
```

## Proof

Run the proof commands listed in `cap_spec.md` for the current focused package
verification set.
