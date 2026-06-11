# Bank Account Management — Specification

## Overview
Bank account lifecycle engine: open/close/freeze, credit/debit with GL posting,
fund locking, overdraft management, and statement generation.

## Core Entities
- BankAccount: account with running balance, status, linked product
- AccountTransaction: immutable ledger entry (debit or credit)
- FundLock: reserved funds pending release

## Compliance
- Double-entry: every debit/credit generates a matching GL journal entry
- All amounts: Decimal (never float)
- Tenant-scoped: every operation requires valid tenant_id
