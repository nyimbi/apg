# Digital Payments

## Overview
Digital Payments is the application-facing payment lifecycle capability: account creation, payment instrument registration with vault token references, payment order creation, risk screening, authorization with provider routing, capture, refunds, payouts, settlement reconciliation, and dispute management. It sits between application capabilities (neobanking, mobile, lending) and the gateway layer, owning the structured payment state machine while delegating actual provider communication to `fintech_gateway`.

Blocked risk levels deny authorization. Overcapture and overrefund are blocked deterministically. Settlement variance requires review. All payment events stream to `apg.fintech.payments.lifecycle` via Bytewax.

Version 2.1.0 adds: recurring payment mandates, network fraud scoring, multi-party approval workflows, real-time health monitoring, and cross-border corridor cost estimation.

## Capability ID
`fintech_payments`  Version: 2.1.0

## Provides
| Service | Description |
|---------|-------------|
| payment_account_lifecycle | Open and manage payment accounts with owner and supported currency |
| payment_instrument_vault | Register tokenized instruments (card, bank account, mobile money, wallet, QR, voucher) |
| payment_order_lifecycle | Create payment orders with account, instrument, amount, currency, and risk screening |
| risk_screening_workflow | Screen payment risk and gate high-risk orders before authorization |
| authorization_capture_refund_workflow | Authorize, capture, and refund payments with overcapture/overrefund protection |
| payout_workflow | Schedule payouts with destination reference |
| settlement_reconciliation_workflow | Record settlements with variance review gates |
| payment_dispute_workflow | Open and manage payment disputes with owner assignment |
| payment_agents | Register AI agents for payment operations, risk, settlement, and dispute review |
| recurring_mandate_engine | Create and execute server-side recurring payment mandates with smart retry |
| network_fraud_scoring | Real-time fan-in network graph fraud detection without ML infrastructure |
| approval_workflow_engine | Configurable multi-party payment approval with quorum and timeout |
| health_monitoring | Real-time per-method health snapshots with anomaly alerting |
| corridor_cost_estimation | Cross-border corridor cost comparison: SWIFT vs stablecoin bridge |

## Requires
| Capability | Purpose |
|------------|---------|
| auth | Authentication |
| audl | Audit trail |
| ntfy | Payment notifications |
| keym | Key management |
| encr | Encryption for instrument data |
| fintech_gateway | Provider routing and authorization execution |
| cash_management | Cash management integration |
| accounts_receivable | Accounts receivable integration |

## Configuration Reference
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| instruments.supported_types | list | card, bank_account, mobile_money, wallet, qr, voucher | Instrument types |
| risk.supported_levels | list | low, medium, high, blocked | Risk level values |
| risk.review_required_for_high_risk | bool | true | High risk requires review gate |
| orders.high_value_threshold | number | 100000 | High-value order threshold |

## API Routes
| Name | Path | Method | Permission | Group |
|------|------|--------|------------|-------|
| dashboard | /fintech-payments/dashboard | GET | fintech_payments:view | Overview |
| accounts | /fintech-payments/accounts | GET/POST | fintech_payments:manage_accounts | Accounts |
| instruments | /fintech-payments/instruments | GET/POST | fintech_payments:manage_instruments | Payments |
| orders | /fintech-payments/orders | GET/POST | fintech_payments:operate | Payments |
| risk | /fintech-payments/risk | GET/POST | fintech_payments:risk | Risk |
| settlement | /fintech-payments/settlement | GET/POST | fintech_payments:settle | Finance |
| disputes | /fintech-payments/disputes | GET/POST | fintech_payments:disputes | Risk |
| agents | /fintech-payments/agents | GET/POST | fintech_payments:admin | Automation |
| settings | /fintech-payments/settings | GET/POST | fintech_payments:admin | Administration |
| mandates | /fintech-payments/mandates | GET/POST | fintech_payments:operate | Recurring |
| mandates_execute | /fintech-payments/mandates/{id}/execute | POST | fintech_payments:operate | Recurring |
| fraud_score | /fintech-payments/fraud/network-score | POST | fintech_payments:risk | Risk |
| approvals | /fintech-payments/approvals | GET/POST | fintech_payments:risk | Approvals |
| approval_decide | /fintech-payments/approvals/{id}/decide | POST | fintech_payments:risk | Approvals |
| health | /fintech-payments/health | GET | fintech_payments:view | Monitoring |
| health_alerts | /fintech-payments/health/alerts | GET/POST | fintech_payments:admin | Monitoring |
| corridor_cost | /fintech-payments/fx/corridor-cost | GET | fintech_payments:view | FX |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| account_currency_supported | Unsupported account currency | deny |
| instrument_token_required | Instrument without vault token reference | deny |
| payment_amount_positive | Order amount <= 0 | deny |
| payment_instrument_required | Order without instrument | deny |
| high_risk_payment_requires_review | Risk level `high` without review | require_review |
| blocked_risk_denies_authorization | Risk level `blocked` | deny |
| authorization_provider_required | Authorization without provider reference | deny |
| high_value_authorization_requires_approval | High-value authorization without approval | require_review |
| capture_requires_authorization | Capture without authorized payment | deny |
| capture_blocks_overcapture | Capture exceeds authorized amount | deny |
| refund_requires_capture | Refund without captured payment | deny |
| refund_blocks_overrefund | Refund exceeds captured balance | deny |
| payout_destination_required | Payout without destination reference | deny |
| settlement_variance_requires_review | Variance detected without review | require_review |
| dispute_owner_required | Dispute without owner | deny |
| payment_batch_requires_bytewax | Batch without Bytewax | deny |
| payment_event_requires_bytewax | Event without Bytewax | deny |

## Data Models
| Model | Key Fields |
|-------|-----------|
| PaymentAccount | id, owner_reference, currency, status |
| PaymentInstrument | id, account_id, instrument_type, token_reference, vault_reference, status |
| PaymentOrder | id, account_id, instrument_id, amount, currency, risk_level, status |
| PaymentAuthorization | id, order_id, provider_reference, authorized_amount, status |
| PaymentCapture | id, authorization_id, capture_amount, status |
| PaymentRefund | id, capture_id, refund_amount, status |
| Payout | id, order_id, destination_reference, amount, status |
| PaymentSettlement | id, provider_reference, amount, variance, review_reference, status |
| PaymentDispute | id, order_id, owner_id, reason, status |

## Streaming Events
Events emitted to the fintech event stream via Bytewax.
| Event | Trigger |
|-------|---------|
| payment_account_opened | Account created |
| payment_instrument_registered | Instrument tokenized |
| payment_order_created | Payment order created |
| payment_risk_screened | Risk assessment recorded |
| payment_authorized | Authorization approved |
| payment_captured | Payment captured |
| payment_refunded | Refund processed |
| payout_scheduled | Payout scheduled |
| settlement_recorded | Settlement reconciled |
| payment_dispute_opened | Dispute opened |
| payment_agent_registered | AI agent registered |
| mandate.created | Recurring mandate registered |
| mandate.executed | Mandate billing cycle completed |
| mandate.cancelled | Mandate deactivated |
| fraud.network_alert | Network fraud score exceeds review/block threshold |
| approval.requested | Payment submitted for multi-party approval |
| approval.approved | Approval quorum reached |
| approval.rejected | Approval rejected by any approver |
| health.anomaly_detected | Payment health anomaly detected |
| health.alert_configured | Health alert rule registered |

## Edge Cases Handled
- `blocked` risk level produces a hard deny on authorization — there is no override path; the only resolution is to re-screen the payment at a lower risk level
- Overcapture is blocked deterministically: the `overcapture: True` flag in context (set when capture amount > authorized amount) produces a deny before any provider call is made
- Overrefund is blocked similarly: refunds exceeding the remaining captured balance are denied before any provider call
- Both batch operations (`payment_batch`) and individual events (`payment_event`) require Bytewax routing — two separate guardrail rules
- Payout destination reference is required but the destination format is not validated by the rule engine — format validation (IBAN, phone number, wallet ID) is the service layer's responsibility
- Instrument vault token is mandatory: raw card numbers, bank account numbers, or mobile money identifiers must never be stored unvaulted

## Composability
- **Upstream**: `fintech_gateway` provides the provider routing and authorization execution layer; `encr` handles encryption of instrument data at vault registration
- **Downstream**: `fintech_neobanking`, `fintech_wallets`, `fintech_mobile`, `fintech_agency`, `fintech_lending`, and `fintech_remittance` all use Digital Payments as their payment execution backbone
- **Peer**: Deployed alongside `fintech_gateway` (provider connectivity) and `fintech_wallets` (stored-value ledger)

## Development Notes
- The capability uses `_lte` and `_gt` condition suffixes for numeric comparisons: `amount_lte: 0` fires when amount <= 0; this is different from the `_ne` pattern used for string comparisons
- `encr` (encryption) is a required dependency distinct from `keym` — keym manages key references, encr handles actual cryptographic operations on instrument data
- Payment accounts are lightweight owners of instruments; they do not maintain a running balance (that is `fintech_wallets`' role); they anchor the instrument-to-owner relationship
- Segregation of duties is enforced at the governance level: the same agent cannot initiate and authorize a payment; this is a configuration flag but not currently enforced by the rule engine rules
