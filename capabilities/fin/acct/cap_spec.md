# Bank Account Management — Capability Spec

capability_id: fin_acct
domain: fin
version: 1.0.0

## Description
Bank account lifecycle management. Double-entry, Decimal-precision, tenant-scoped.

## Key Operations
- open_account, close_account, freeze_account, unfreeze_account
- credit_account, debit_account, transfer_internal
- lock_funds, release_lock
- get_balance, get_transactions, generate_statement
