# Cross-Border Remittance

## Overview
Cross-Border Remittance manages the full lifecycle of international money transfers: corridor and currency eligibility, FX quoting with rate and fee locking, transfer creation with dual-side KYC and source-of-funds evidence, AML/sanctions screening, fraud decisioning, payout dispatch with provider receipt, and refund handling. Same-country transfers are architecturally blocked.

Transfers require a quote lock, both sender and beneficiary KYC, AML screen, source-of-funds evidence, and a supported fraud decision. Sanctions hits are a hard deny with no override path. High-value transfers and AML/fraud review outcomes require human approval. Events stream to `apg.fintech.remittance.lifecycle` via Bytewax.

## Capability ID
`fintech_remittance`  Version: 2.0.0

## Provides
| Service | Description |
|---------|-------------|
| remittance_corridor_governance | Define and govern supported send/receive country corridors |
| remittance_quote_lifecycle | Create FX quotes with amount, rate, fee, expiry, and quote lock |
| cross_border_transfer_workflow | Create transfers with sender/beneficiary KYC, AML, fraud, funding, purpose, and source-of-funds |
| remittance_payout_workflow | Release payouts with provider receipt and settlement reference |
| remittance_refund_workflow | File refunds with reason and reviewer assignment |
| remittance_agent_workflow | Register AI agents for compliance review, payout, and treasury roles |

## Requires
| Capability | Purpose |
|------------|---------|
| auth | Authentication |
| audl | Audit trail |
| ntfy | Sender and beneficiary notifications |
| nlpc | NLP processing |
| keym | Key management |
| fintech_payments | Funding and payment execution |
| fintech_wallets | Wallet-based funding and payout rails |
| fintech_kyc | Sender and beneficiary identity verification |
| fintech_aml | Sanctions, PEP, and AML screening |
| fintech_fraud | Fraud risk scoring |

## Quick Start

```python
from capabilities.fintech.remittance.service import RemittanceService

svc = RemittanceService(tenant_id="acme", actor_id="ops-user")

# Initiate a transfer
result = await svc.initiate_remittance(
    sender_id="sender-001",
    recipient={"id": "recip-001", "name": "Jane Doe"},
    amount=50_000,
    send_currency="KES",
    receive_currency="UGX",
    corridor="KE-UG",
    purpose_code="family_support",
    payout_method="mobile_money",
)

# Sandbox mode (no real data)
sandbox = RemittanceService(tenant_id="test", sandbox=True)
await sandbox.sandbox_reset()
```

## Configuration Reference
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| corridors.supported_countries | list | KE, UG, TZ, RW, GH, NG, ZA, GB, US, AE, IN | Supported send/receive countries |
| corridors.same_country_blocked | bool | true | Same-country transfers blocked |
| quotes.supported_currencies | list | KES, UGX, TZS, RWF, GHS, NGN, ZAR, GBP, USD, AED, INR, EUR | Supported currencies |
| transfers.high_value_threshold | number | 100000 | Amount requiring approval |
| payouts.supported_methods | list | bank_account, mobile_money, wallet, cash_pickup, card_push | Payout delivery methods |
| transfers.supported_purpose_codes | list | family_support, education, medical, trade, salary, savings, emergency | Transfer purposes |

## API Routes
| Name | Path | Method | Permission | Group |
|------|------|--------|------------|-------|
| dashboard | /fintech-remittance/dashboard | GET | fintech_remittance:view | Overview |
| corridors | /fintech-remittance/corridors | GET/POST | fintech_remittance:govern_corridors | Corridors |
| quotes | /fintech-remittance/quotes | GET/POST | fintech_remittance:quote | Quotes |
| transfers | /fintech-remittance/transfers | GET/POST | fintech_remittance:transfer | Transfers |
| payouts | /fintech-remittance/payouts | GET/POST | fintech_remittance:payout | Payouts |
| refunds | /fintech-remittance/refunds | GET/POST | fintech_remittance:refund | Exceptions |
| agents | /fintech-remittance/agents | GET/POST | fintech_remittance:admin | Automation |
| settings | /fintech-remittance/settings | GET/POST | fintech_remittance:admin | Administration |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| same_country_blocked | Source and destination country are the same | deny |
| corridor_supported | Corridor not in supported list | deny |
| send_amount_positive | Quote send amount <= 0 | deny |
| fx_rate_positive | FX rate <= 0 | deny |
| fee_non_negative | Negative fee | deny |
| quote_expiry_required | Quote without expiry | deny |
| quote_lock_required | Transfer without quote lock | deny |
| sender_kyc_required | Transfer without sender KYC | deny |
| beneficiary_kyc_required | Transfer without beneficiary KYC | deny |
| source_of_funds_required | Transfer without source-of-funds | deny |
| aml_screen_required | Transfer without AML screen | deny |
| sanctions_hit_blocks_transfer | Sanctions hit detected | deny |
| fraud_block_denies_transfer | Fraud decision is `block` | deny |
| aml_review_requires_approval | AML review outcome without human approval | require_review |
| fraud_review_requires_approval | Fraud hold/review without human approval | require_review |
| high_value_requires_approval | Amount > 100,000 without approval | require_review |
| provider_receipt_required | Payout without provider receipt | deny |
| settlement_reference_required | Payout without settlement reference | deny |
| refund_reviewer_required | Refund without reviewer | deny |

## Data Models
| Model | Key Fields |
|-------|-----------|
| RemittanceCorridor | id, send_country, receive_country, currency_pair, policy_reference, status |
| RemittanceQuote | id, send_country, receive_country, source_currency, destination_currency, send_amount, fx_rate, fee, expiry, status |
| RemittanceTransfer | id, quote_id, sender_reference, beneficiary_reference, sender_kyc_reference, beneficiary_kyc_reference, funding_reference, payout_method, purpose_code, source_of_funds_reference, aml_screen_reference, fraud_decision, status |
| RemittancePayout | id, transfer_id, payout_method, provider_receipt_reference, settlement_reference, status |
| RemittanceRefund | id, transfer_id, reason, reviewer_id, status |

## Streaming Events
Events emitted to the fintech event stream via Bytewax.
| Event | Trigger |
|-------|---------|
| remittance_quote_created | FX quote created |
| remittance_transfer_created | Transfer initiated |
| remittance_payout_released | Payout released to beneficiary |
| remittance_refund_filed | Refund filed |
| remittance_agent_registered | AI agent registered |

---

## World-Class Enhancements (v2.0)

1. **Multi-Hop Corridor Routing** — Dijkstra shortest-path over fee+FX-spread edges. Opens ~40 indirect corridors (e.g. KE→AE→IN) with no new partner agreements. Method: `optimal_corridor_path(send_country, receive_country, send_currency, receive_currency)`.

2. **Real-Time FX Rate Feed** — Pluggable provider adapters (ExchangeRate-API, CBK Open Data, Wise sandbox) with 60s TTL `BoundedCache`. Quotes refused if last refresh >5 min. Drift alert at bid/ask spread >2%. Method: `refresh_fx_rates(providers)`.

3. **Velocity-Based AML Risk Scoring** — Rolling-window (1h/24h/7d/30d) per-sender volume, frequency, and corridor diversity score. Auto-files STR pre-report at score ≥80. Aligns with FATF Recommendation 16. Method: `compute_velocity_risk(sender_id, window_hours, tenant_id)`.

4. **Idempotent Transfer Submission** — SHA-256 idempotency key with configurable 24h deduplication window. Returns existing record on duplicate rather than error. Eliminates the #1 support ticket class on remittance platforms.

5. **Structured Webhook Framework** — HMAC-SHA256 signed outbound webhooks (`X-APG-Signature`), exponential backoff retry (3 attempts: 5s/30s/300s), delivery receipts in evidence store. Methods: `register_webhook`, `dispatch_webhook`.

6. **Tiered KYC Limits Enforcement** — Per-corridor daily/monthly limit matrix enforced at quote time, before FX calculation. Returns `allowed`, `limit_remaining`, `tier_upgrade_required`. CBK PSP Guidelines Section 4.3 compliance. Method: `enforce_kyc_tier_limits(sender_id, amount, send_currency, kyc_tier)`.

7. **Bank Account Validation** — Format validation per country: IBAN checksum (EU/GB), NUBAN algorithm (NG), sort-code+8-digit (GB), RTGS routing (KE). Eliminates ~15% of failed payouts from malformed account numbers. Method: `validate_bank_account(country, bank_code, account_number, account_type)`.

8. **FX Forward Contracts** — Rate lock for up to 30 days with forward points (interest rate differential carry). Transfers can reference `forward_id` in lieu of spot quote. Enables payroll hedging for corporate clients. Method: `create_fx_forward(send_currency, receive_currency, amount, settlement_date, tenor_days)`.

9. **Corridor Risk Heat Map** — Per-corridor: open exposure, FX settlement risk, concentration risk (>30% of volume), partner credit risk. RAG status output. Required for PAPSS participation. Method: `corridor_risk_heatmap(tenant_id, period)`.

10. **Regulatory Sandbox Mode** — `RemittanceService(sandbox=True)` activates deterministic FX rates, configurable compliance outcomes, `SANDBOX-` prefixed transfer IDs, and Bytewax event suppression. Method: `sandbox_reset()` purges all sandbox state.

11. **Multi-Currency Wallet Sweep** — Greedy cover algorithm selects optimal wallet(s) across currencies to minimize FX conversion cost. Atomic fund reservation. Integrates with `fintech_wallets`. Method: `wallet_sweep_funding(sender_id, amount, preferred_currency, wallet_ids)`.

12. **AI Purpose Code Classification** — Local Ollama (mistral/llama3) classifies free-text descriptions into purpose codes. Returns `predicted_code`, `confidence`, `alternative_codes`. Mismatch flags as AML signal. Fallback to rule-based keyword matching. Method: `classify_purpose_code(transaction_description, sender_profile, beneficiary_profile)`.

13. **ISO 20022 pacs.008 Generation** — Compliant FI-to-FI Customer Credit Transfer XML: `GrpHdr`, `CdtTrfTxInf`, SWIFT BIC, IBAN/account. Prerequisite for SWIFT GPI, PAPSS, and bilateral RTGS integration. Method: `generate_pacs008(transfer_id, instruction_id)`.

14. **Dynamic Fee Negotiation** — Volume-based rebate tiers for high-volume senders. Returns `negotiated_fee_pct`, `rebate_amount`, `agreement_id`, `valid_until`. Signed agreements override default corridor fee at quote time. Method: `negotiate_fee(sender_id, monthly_volume_kes, corridor, commitment_months)`.

15. **End-to-End Transfer Simulation (Dry Run)** — Executes full quote→compliance→routing→payout logic read-only against live data. Returns `would_succeed`, `blocking_reasons`, `total_cost`, `estimated_delivery`. Surfaces compliance holds and KYC tier limits before user commits. Method: `simulate_transfer(sender_id, recipient, amount, ...)`.

---

## New Methods

### `simulate_transfer` — Pre-flight check before user commits

```python
sim = await svc.simulate_transfer(
    sender_id="sender-001",
    recipient={"id": "recip-001"},
    amount=250_000,
    send_currency="KES",
    receive_currency="USD",
    corridor="KE-US",
    payout_method="bank_account",
)
# sim["would_succeed"] -> bool
# sim["blocking_reasons"] -> list of blocking condition strings
# sim["total_cost"] -> total fee amount
# sim["estimated_delivery"] -> delivery hours estimate
```

### `refresh_fx_rates` — Pull live rates into cache

```python
rates = await svc.refresh_fx_rates(providers=["exchangerate_api", "cbk_open_data"])
# rates["updated_pairs"] -> number of pairs refreshed
# rates["stalest_age_seconds"] -> oldest cached rate age
# Subsequent get_fx_quote() calls use cached live rates
```

### `compute_velocity_risk` — Rolling AML risk score

```python
risk = await svc.compute_velocity_risk(
    sender_id="sender-001",
    window_hours=24,
    tenant_id="acme",
)
# risk["score"] -> 0–100 composite score
# risk["daily_count"] / risk["daily_value"] -> window aggregates
# risk["str_pre_filed"] -> True if score >= 80
```

### `enforce_kyc_tier_limits` — Tier-based limit check at quote time

```python
check = await svc.enforce_kyc_tier_limits(
    sender_id="sender-001",
    amount=500_000,
    send_currency="KES",
    kyc_tier=1,
)
# check["allowed"] -> False if over tier limit
# check["limit_remaining"] -> KES remaining in daily limit
# check["tier_upgrade_required"] -> True if upgrade needed to proceed
```

### `negotiate_fee` — Volume rebate for high-value senders

```python
deal = await svc.negotiate_fee(
    sender_id="sender-001",
    monthly_volume_kes=5_000_000,
    corridor="KE-UG",
    commitment_months=6,
)
# deal["negotiated_fee_pct"] -> reduced fee percentage
# deal["rebate_amount"] -> KES rebate on this month's volume
# deal["agreement_id"] -> referenced by get_fx_quote() automatically
```

---

## Edge Cases Handled
- Sanctions hits are a hard deny with no override path — distinct from AML review (which allows human approval to proceed)
- Both sender AND beneficiary KYC are required — one-sided KYC is not accepted
- FX rate must be strictly positive; zero is rejected
- Quote expiry is required at quote creation — expired quotes cannot be used to create transfers
- Source-of-funds evidence is required for every transfer, not just high-value ones

## Composability
- **Upstream**: `fintech_kyc` (identity), `fintech_aml` (sanctions), `fintech_fraud` (fraud decisions), `fintech_payments` and `fintech_wallets` (funding rails)
- **Downstream**: `fintech_agency` (cross-border payout at agent outlets), `fintech_mobile` (mobile channel), `fintech_lending` (credit behaviour evidence)
- **Peer**: Deployed alongside `fintech_kyc`, `fintech_aml`, and `fintech_fraud` — all three required for every transfer

## Development Notes
- `cash_pickup` payout requires a physical agent network; agent network management is in `fintech_agency`
- `card_push` (Visa Direct, Mastercard Send) requires card token references from `fintech_cards`
- Purpose codes map to SWIFT/ISO 20022 purpose codes; `family_support`, `education`, `medical` are primary Africa remittance use cases
- Batch operations and individual events require Bytewax routing; three guardrail rules cover batches, events, and privileged agent actions
- AI features (`classify_purpose_code`, `ml_remittance_fraud_detect`) require `OLLAMA_BASE_URL` set in the environment

© 2025 Datacraft — www.datacraft.co.ke
