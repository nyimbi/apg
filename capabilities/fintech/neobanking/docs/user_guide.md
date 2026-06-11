# Digital Neobanking — User Guide

**Capability ID**: `fintech_neobanking` | **Domain**: `fintech` | **Version**: `1.2.0`

## Description

Digital Neobanking provides the complete core banking layer for digital-first banks: program governance, instant customer onboarding with KYC/AML/fraud evidence chain, deposit account lifecycle, payment rail linking, transaction posting, savings pots with auto-sweep rules, cross-currency FX transfers, spending budgets, chargeback workflows, structured consent management, customer risk scoring, and signed balance attestations.

It is the account ledger that all other fintech capabilities — mobile, cards, lending, remittance — use as their underlying account infrastructure.

---

## Installation

```bash
pip install apg-fintech-neobanking
```

---

## Quick Start

```python
import asyncio
from capabilities.fintech.neobanking import NeobanksService

svc = NeobanksService(tenant_id="acme-bank")

async def main():
    # 1. Register a banking program
    prog = await svc.register_program(
        "prog-001", "Acme Digital Bank", "owner-001",
        country="KE", base_currency="KES",
        settlement_account="settle-001",
    )

    # 2. Onboard a customer
    customer = await svc.onboard_customer(
        "cust-001", "john-doe-ref", "kyc-001",
        country="KE", consent_reference="consent-001",
        aml_reference="aml-001", fraud_reference="fraud-001",
    )

    # 3. Open a current account
    account = await svc.open_account(
        "cust-001", "current", "KES",
        program_id="prog-001", initial_balance=5000.0,
    )
    print(account["id"])  # e.g. "3e4f..."

asyncio.run(main())
```

---

## Core Concepts

### Banking Programs
A `BankProgram` is the top-level entity representing a bank's operating license or a BaaS sponsor bank relationship. All accounts belong to a program.

### Deposit Accounts
Accounts are typed (`current`, `savings`, `joint`, `business`, `youth`, `merchant`) and currency-specific. Feature bundles (`starter`, `standard`, `premium`, `business`) control daily transfer limits, virtual card quotas, savings pot quotas, cashback rates, and overdraft eligibility.

### Transactions
Every posted transaction requires:
- A `risk_reference` linking it to a fraud/AML screening result
- Currency matching the account
- Human approval for high-value amounts (≥ KES 100,000 or transfers ≥ KES 50,000)

---

## Service Methods Reference

### Account Lifecycle

| Method | Description |
|--------|-------------|
| `open_account(customer_id, account_type, currency, ...)` | Open a deposit account with KYC validation |
| `close_account(account_id, reason)` | Close account (requires zero balance) |
| `account_features_bundle(account_id, bundle)` | Apply starter / standard / premium / business bundle |
| `account_upgrade(account_id, new_bundle)` | Upgrade bundle |
| `account_freeze(account_id, reason, frozen_by)` | Freeze account pending investigation |

### Customer Management

| Method | Description |
|--------|-------------|
| `onboard_customer(customer_id, ...)` | Onboard with KYC, AML, fraud, consent references |
| `bulk_onboard_customers(customers)` | Batch onboard from list of dicts |
| `kyc_refresh(customer_id, new_kyc_reference, reason)` | Refresh KYC on an existing customer |
| `compute_customer_risk_score(customer_id)` | Aggregate 0-100 risk score from behavioural signals |

### Payments & Transfers

| Method | Description |
|--------|-------------|
| `peer_transfer(from_account, to_account, amount, ...)` | Same-currency transfer between accounts |
| `split_bill(from_account, recipients, total_amount, ...)` | Equal-split transfer to multiple recipients |
| `bulk_transfer(from_account, transfers)` | Batch peer transfers from one account |
| `fx_convert_and_transfer(from_account, to_account, amount, from_currency, to_currency, fx_rate, ...)` | Cross-currency transfer with spread and FX fee |
| `post_transaction(transaction_id, account_id, kind, amount, ...)` | Post arbitrary typed transaction |
| `pesa_link_bank_transfer(account_id, destination_account, bank_code, amount, reference)` | PesaLink interbank transfer |
| `direct_debit_mandate(account_id, creditor_id, max_amount, frequency)` | Set up direct debit mandate |
| `standing_order(account_id, beneficiary_account, amount, frequency, start_date)` | Set up standing order |

### Savings Pots

| Method | Description |
|--------|-------------|
| `savings_pot_create(account_id, name, target_amount, ...)` | Create a named savings goal |
| `savings_pot_deposit(pot_id, amount)` | Move funds from account into pot |
| `savings_round_up(account_id, transaction_id, ...)` | Round up last transaction into pot |
| `savings_pot_autosweep_rule(account_id, pot_id, trigger, value)` | Attach auto-sweep trigger to pot |
| `execute_autosweep_rules(trigger, account_id)` | Execute all matching auto-sweep rules |

**Auto-sweep triggers:**
- `end_of_day` — sweep fixed amount daily
- `percentage_of_balance` — sweep `value`% of account balance on each trigger
- `after_credit` — sweep fixed amount after every inbound credit

### Virtual Cards

| Method | Description |
|--------|-------------|
| `virtual_card_issue(account_id, ...)` | Issue a virtual card with masked PAN |
| `virtual_card_freeze(card_id)` | Freeze card (blocks new transactions) |
| `virtual_card_unfreeze(card_id)` | Unfreeze card |

### Spending Analytics & Budgets

| Method | Description |
|--------|-------------|
| `spending_analytics(account_id, period)` | Spending totals, daily average, top merchants |
| `subscription_tracking(account_id)` | Detect recurring subscription charges |
| `cashback_calculation(account_id, period)` | Calculate and credit cashback |
| `set_spending_budget(account_id, category, monthly_limit)` | Set a monthly category budget |
| `spending_budget_check(account_id, category)` | Check utilisation, burn rate, projected over-budget date |

```python
# Set a KES 10,000/month budget for card purchases
budget = await svc.set_spending_budget(account_id, "card_purchase", 10_000.0)

# Check remaining budget — fires 75 % and 100 % notifications automatically
status = await svc.spending_budget_check(account_id, "card_purchase")
# {
#   "monthly_limit": 10000.0,
#   "spent_this_month": 7500.0,
#   "remaining": 2500.0,
#   "utilisation_pct": 75.0,
#   "burn_rate_daily": 375.0,
#   "days_until_over_budget": 6.7,
# }
```

### Overdraft

| Method | Description |
|--------|-------------|
| `overdraft_protection(account_id, limit)` | Configure overdraft limit (0 to disable) |
| `overdraft_interest_accrual(account_id, period)` | Post daily overdraft interest and fee |
| `interest_accrual(account_id, period)` | Post monthly deposit interest |

### Chargebacks

| Method | Description |
|--------|-------------|
| `open_chargeback(case_id, customer_id, account_id, disputed_transaction_id, reason, ...)` | Open dispute with provisional credit |
| `resolve_chargeback(case_id, ruling, ...)` | Resolve with `upheld` or `rejected` |

```python
# Dispute a transaction — provisional credit issued immediately
cb = await svc.open_chargeback(
    "cb-001", "cust-001", account_id,
    disputed_transaction_id="tx-123",
    reason="unauthorised_transaction",
)
# cb["status"] == "provisional_credit_issued"

# Resolve — upheld keeps credit, rejected reverses it
result = await svc.resolve_chargeback("cb-001", ruling="upheld")
# result["status"] == "resolved_upheld"
```

### Consent Management

| Method | Description |
|--------|-------------|
| `record_consent(customer_id, consent_type, channel, ...)` | Record structured consent event |
| `revoke_consent(customer_id, consent_type, reason)` | Revoke all active consents of a type |

**Consent types**: `account_opening`, `data_sharing`, `marketing`, `overdraft`, `biometric`

**Channels**: `sms_otp`, `biometric`, `e_signature`, `agent_assisted`, `in_app`

Satisfies Kenya Data Protection Act Article 30 requirements. Records are append-only; revocation marks status `revoked` without deleting the audit trail.

### Balance Attestation

| Method | Description |
|--------|-------------|
| `generate_balance_attestation(account_id, purpose)` | Produce HMAC-signed proof-of-funds |

```python
att = await svc.generate_balance_attestation(account_id, purpose="mortgage_application")
# {
#   "attestation_id": "...",
#   "balance": 125000.0,
#   "currency": "KES",
#   "signature": "a3f9...",
#   "algorithm": "HMAC-SHA256",
#   "expires_at": "2026-06-11T23:59:59+00:00",
# }
```

### Webhooks

| Method | Description |
|--------|-------------|
| `register_account_webhook(account_id, webhook_url, event_filter, secret)` | Register HMAC-signed event webhook |

```python
hook = await svc.register_account_webhook(
    account_id,
    webhook_url="https://partner.example.com/hooks/neobanking",
    event_filter=["peer_transfer_completed", "savings_goal_reached"],
    secret="s3cr3t",
)
```

### Customer Risk Scoring

```python
score = await svc.compute_customer_risk_score("cust-001")
# {
#   "risk_score": 42,
#   "tier": "medium",
#   "signals": {
#     "velocity_score": 12.0,
#     "overdraft_score": 0.0,
#     "savings_score": 10.0,
#     "freeze_score": 0,
#     "total_debit_count": 4,
#     "savings_ratio_pct": 50.0,
#   }
# }
```

Tiers: `low` (0–29) | `medium` (30–64) | `high` (65–100). Scores are advisory only and do not gate transactions.

### Statements & Reporting

| Method | Description |
|--------|-------------|
| `issue_statement(statement_id, account_id, period_start, period_end)` | Issue single account statement |
| `bulk_issue_statements(period_start, period_end, account_ids)` | Bulk-generate statements for all or selected accounts |
| `cbk_neobanking_return(period, jurisdiction)` | Draft CBK Digital Banking regulatory return |
| `export_account_data(customer_id, fmt)` | Export account data (json/csv/excel) |

### Cross-Currency FX Transfer

```python
result = await svc.fx_convert_and_transfer(
    from_account="acct-ke",
    to_account="acct-usd",
    amount=100_000.0,
    from_currency="KES",
    to_currency="USD",
    fx_rate=0.00775,      # mid-market rate
    fx_spread_pct=0.5,    # 0.5% spread applied
)
# {
#   "original_amount": 100000.0,
#   "original_currency": "KES",
#   "converted_amount": 770.63,
#   "target_currency": "USD",
#   "mid_market_rate": 0.00775,
#   "effective_rate": 0.00771125,
#   "fx_fee": 3.88,
# }
```

---

## Payment Rails

Register a payment rail link before posting transactions via that rail:

```python
await svc.link_payment_rail(
    "rail-001", account_id, "mobile_money",
    provider_reference="254712345678",
)
```

Supported rails: `bank_transfer`, `card`, `wallet`, `mobile_money`, `internal_transfer`, `pesalink`, `rtgs`, `ach`, `swift`.

---

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| accounts.supported_types | list | current, savings, joint, business, youth, merchant | Account types |
| transactions.supported_types | list | deposit, withdrawal, transfer_in/out, card_purchase, fee, refund, interest, overdraft_fee, fx_transfer_in/out | Transaction types |
| transactions.high_value_threshold | number | 100000 | Amount requiring human approval |
| rails.supported_rails | list | bank_transfer, card, wallet, mobile_money, internal_transfer | Payment rails |
| service_cases.supported_reasons | list | account_access, card_issue, payment_dispute, kyc_review, fraud_review, fee_query, statement_query | Case reasons |
| overdraft.default_rate_pa | float | 0.18 | Annual overdraft interest rate |
| overdraft.daily_fee | float | 50.0 | Daily overdraft facility fee (KES) |
| cashback.qualifying_kinds | list | card_purchase, pos_purchase, online_purchase | Transaction types that earn cashback |

---

## Interoperability

```apg
use fintech_neobanking;
```

`fintech_neobanking` integrates with other APG capabilities through the composition engine. Required adapters:

| Adapter | Constructor param | Purpose |
|---------|------------------|---------|
| auth | `auth=` | Authentication / permission checks |
| audit | `audit=` | Durable audit trail |
| notify | `notify=` | Real-time customer and ops notifications |
| store | `store=` | Persistent state backend |

---

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `capability_contract.py` — Policy rules and supported enumerations
- `WORLD_CLASS_IMPROVEMENTS.md` — Roadmap of 15 production-grade enhancements
- `README.md` — Quick reference
