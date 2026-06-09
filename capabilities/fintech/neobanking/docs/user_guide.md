# Digital Neobanking

**Capability ID**: `fintech_neobanking` | **Domain**: `fintech` | **Version**: `1.1.0`

## Description

Digital Neobanking provides the core banking layer for digital-first banks: program governance, customer onboarding with full AML/KYC/fraud evidence chain, deposit account opening (current, savings, joint, business, youth, merchant), payment rail linking, transaction posting with risk reference, savings pot management, account statement generation, and customer service case handling. It is the account ledger that other capabilities — mobile, cards, lending, remittance — use as their underlying account infrastructure.

## Installation

```bash
pip install apg-fintech-neobanking
```

## Provides

- `neobank_program_governance`
- `digital_customer_onboarding`
- `deposit_account_lifecycle`
- `payment_rail_linking`
- `account_transaction_posting`

## Requires

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/fintech-neobanking/dashboard` | `fintech_neobanking:view` | Overview |
| `/fintech-neobanking/programs` | `fintech_neobanking:manage_programs` | Programs |
| `/fintech-neobanking/customers` | `fintech_neobanking:manage_customers` | Customers |
| `/fintech-neobanking/accounts` | `fintech_neobanking:manage_accounts` | Accounts |
| `/fintech-neobanking/rails` | `fintech_neobanking:manage_rails` | Payments |
| `/fintech-neobanking/transactions` | `fintech_neobanking:post_transactions` | Payments |
| `/fintech-neobanking/savings` | `fintech_neobanking:savings` | Accounts |
| `/fintech-neobanking/statements` | `fintech_neobanking:statements` | Servicing |

## Key Service Methods

- `describe()`
- `evaluate()`
- `open_account()`
- `close_account()`
- `account_features_bundle()`
- `virtual_card_issue()`
- `virtual_card_freeze()`
- `peer_transfer()`
- `split_bill()`
- `savings_pot_create()`

_(See `service.py` for complete API.)_

## Interoperability

`fintech_neobanking` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use fintech_neobanking;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `FINTECH_NEOBANKING_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
