# Digital Wallets

## Overview
Digital Wallets provides the stored-value ledger layer: wallet lifecycle (consumer, merchant, agent, escrow, treasury), instrument registration with verified token references, double-entry ledger operations (credit, debit, transfer), hold management for reserved funds, and limit governance. It is the balance-holding layer that other capabilities — payments, mobile, agency, neobanking — use to maintain available and held balances for their customers and operational accounts.

Negative balances are architecturally blocked. Transfers between same-currency wallets only — cross-currency transfers are denied to enforce explicit FX routing. Large transfers require review. Hold releases cannot exceed the held balance. Both batch operations and individual events require Bytewax routing. Events stream to `apg.fintech.wallets.lifecycle` via Bytewax.

## Capability ID
`fintech_wallets`  Version: 1.1.0

## Provides
| Service | Description |
|---------|-------------|
| wallet_lifecycle | Open and manage consumer, merchant, agent, escrow, and treasury wallets |
| stored_value_ledger | Credit, debit, and transfer operations with negative-balance protection |
| wallet_instrument_registry | Register verified tokenized instruments (bank account, card, mobile money, wallet, voucher) |
| wallet_transfer_workflow | Transfer between wallets with currency match enforcement and limit review |
| wallet_hold_workflow | Place and release holds with available-balance checks |
| wallet_limit_governance | Manage daily debit limits and single-transfer limits with review gates |
| wallet_agent_workflow | Register AI agents for wallet operations, risk, limits, and settlement review |

## Requires
| Capability | Purpose |
|------------|---------|
| auth | Authentication |
| audl | Audit trail |
| ntfy | Customer notifications |
| walt | Core wallet engine |
| fintech_payments | Payment execution backing |
| fintech_gateway | Provider routing |
| keym | Key management for instrument tokens |

## Configuration Reference
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| wallets.supported_types | list | consumer, merchant, agent, escrow, treasury | Wallet categories |
| wallets.supported_currencies | list | USD, EUR, GBP, KES, ZAR, NGN, GHS, UGX, TZS | Supported currencies |
| wallets.negative_balance_blocked | bool | true | Negative balances blocked |
| limits.daily_debit_limit_minor | number | 500000 | Daily debit ceiling (minor units) |
| limits.single_transfer_limit_minor | number | 250000 | Single transfer ceiling (minor units) |
| instruments.supported_types | list | bank_account, card, mobile_money, wallet, voucher | Instrument types |

## API Routes
| Name | Path | Method | Permission | Group |
|------|------|--------|------------|-------|
| dashboard | /fintech-wallets/dashboard | GET | fintech_wallets:view | Overview |
| wallets | /fintech-wallets/wallets | GET/POST | fintech_wallets:manage_wallets | Wallets |
| instruments | /fintech-wallets/instruments | GET/POST | fintech_wallets:manage_instruments | Wallets |
| ledger | /fintech-wallets/ledger | GET | fintech_wallets:view_ledger | Ledger |
| limits | /fintech-wallets/limits | GET/POST | fintech_wallets:govern_limits | Governance |
| holds | /fintech-wallets/holds | GET/POST | fintech_wallets:operate | Operations |
| agents | /fintech-wallets/agents | GET/POST | fintech_wallets:admin | Automation |
| settings | /fintech-wallets/settings | GET/POST | fintech_wallets:admin | Administration |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| wallet_type_supported | Unsupported wallet type | deny |
| wallet_currency_supported | Unsupported currency | deny |
| instrument_token_required | Instrument without token reference | deny |
| instrument_verification_required | Instrument without verification evidence | deny |
| credit_amount_positive | Credit amount <= 0 | deny |
| debit_amount_positive | Debit amount <= 0 | deny |
| debit_blocks_negative_balance | Debit would create negative available balance | deny |
| transfer_requires_distinct_wallets | Same wallet as source and destination | deny |
| transfer_requires_matching_currency | Cross-currency transfer attempted | deny |
| transfer_limit_requires_review | Transfer exceeds single-transfer limit | require_review |
| hold_amount_positive | Hold amount <= 0 | deny |
| hold_blocks_negative_available | Hold exceeds available balance | deny |
| hold_release_amount_positive | Hold release amount <= 0 | deny |
| hold_release_blocks_overrelease | Hold release exceeds held balance | deny |
| wallet_batch_requires_bytewax | Batch without Bytewax | deny |
| wallet_event_requires_bytewax | Event without Bytewax | deny |

## Data Models
| Model | Key Fields |
|-------|-----------|
| Wallet | id, owner_reference, wallet_type, currency, available_balance, held_balance, status |
| WalletInstrument | id, wallet_id, instrument_type, token_reference, verified, status |
| LedgerEntry | id, wallet_id, entry_type, amount, reference, idempotency_key |
| WalletTransfer | id, source_wallet_id, destination_wallet_id, amount, currency, review_reference, status |
| WalletHold | id, wallet_id, amount, reason, status |
| WalletHoldRelease | id, hold_id, release_amount |
| WalletLimit | id, wallet_id, daily_debit_limit, single_transfer_limit |

## Streaming Events
Events emitted to the fintech event stream via Bytewax.
| Event | Trigger |
|-------|---------|
| wallet_opened | Wallet created |
| wallet_instrument_registered | Instrument tokenized and verified |
| wallet_credited | Credit posted |
| wallet_debited | Debit posted |
| wallet_transfer_posted | Transfer completed |
| wallet_hold_placed | Hold placed on available balance |
| wallet_hold_released | Hold released |
| wallet_limit_updated | Limit updated |
| wallet_agent_registered | AI agent registered |

## Edge Cases Handled
- Cross-currency transfers are a hard deny — transfers between wallets in different currencies require going through an FX conversion step outside the wallet capability; this prevents accidental cross-currency debits at wrong rates
- Same-wallet transfers are denied — source and destination must be distinct wallet IDs; this prevents no-op ledger entries that would inflate transaction counts
- Hold releases cannot exceed the held balance — even if the available balance is high, a release of more than the held amount is rejected to maintain ledger integrity
- Instrument verification is required at registration — an unverified instrument (e.g., a bank account that hasn't been micro-deposit verified) cannot be linked to a wallet
- Ledger entries use idempotency keys to prevent duplicate posting; the rule engine enforces idempotency as a configuration requirement (`idempotency_required: True`)

## Composability
- **Upstream**: `walt` (core wallet engine) provides the underlying balance management; `keym` manages token references for instruments; `fintech_gateway` provides provider connectivity for top-ups and withdrawals
- **Downstream**: `fintech_payments` uses wallets as the instrument backing for payment orders; `fintech_agency` uses agent wallet float accounts; `fintech_mobile` links wallets as account types; `fintech_neobanking` uses wallet rails for payment execution; `fintech_remittance` uses wallets for funded transfers
- **Peer**: Deployed alongside `fintech_payments` (payment orders) and `fintech_gateway` (provider routing) as the core financial infrastructure trio

## Development Notes
- Limits are stored in minor units (e.g., KES cents, USD cents) to avoid floating-point arithmetic; 500000 minor units = KES 5,000 or USD 5,000 depending on currency precision
- `escrow` and `treasury` wallet types are operational accounts, not customer accounts; they follow the same governance rules as consumer/merchant wallets
- The `walt` dependency is a core wallet engine service (abbreviated as `walt` in the adapter map) — it handles atomic balance operations; the Digital Wallets capability wraps `walt` with APG governance and event streaming
- Both batch and individual event operations require Bytewax routing — three separate guardrail rules: `wallet_batch`, `wallet_event`, and privileged agent actions
