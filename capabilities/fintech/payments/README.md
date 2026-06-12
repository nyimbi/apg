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
| semantic_deduplication | Fuzzy duplicate detection by phone+amount+merchant+time window |
| float_forecasting | Predictive agent float exhaustion ETA with auto top-up triggers |
| auto_ctr_filing | Automatic Currency Transaction Report filing for CBK/CBN/BoU thresholds |
| intelligent_routing | Multi-rail cost/speed/reliability optimisation per transaction |
| velocity_adaptive_limits | Dynamic KYC tier limits based on behavioral credit scoring |
| fx_rate_locks | 5-minute guaranteed FX rate locks with micro-hedge for cross-border |
| chargeback_intelligence | Automated triage with win-probability scoring and evidence pre-population |
| batch_failure_recovery | Auto-classify and re-route batch failures; surface only unresolvable items |
| intraday_settlement | Configurable settlement cycles (2h/4h/real-time) with provisional credit |
| payment_widget_spec | Declarative offline-first payment widget JSON contract for any UI framework |

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

## Quick Start

```python
from apg_fintech_payments import DigitalPaymentsService

svc = DigitalPaymentsService(tenant_id="acme", store=store, event_bus=bus)

# Initiate an M-Pesa STK Push
result = await svc.mpesa_stk_push(
    phone="254712345678",
    amount=Decimal("1500.00"),
    reference="INV-2025-001",
    description="School fees deposit",
)

# Check status
status = await svc.get_payment_status(result["transaction_id"])
```

## World-Class Enhancements (v2.0)

All 15 improvements are implemented as pure calculation functions in `domain/calculations.py`, service methods in `service.py`, and blueprint endpoints — no ML infrastructure required.

| # | Enhancement | Method | Monthly ROI (KES) |
|---|-------------|--------|-------------------|
| 1 | **Semantic Deduplication** — fuzzy duplicate detection by phone+amount+reference similarity; catches "soft duplicates" from human retries | `semantic_duplicate_check` | 200,000 |
| 2 | **Predictive Float Management** — forecasts agent float exhaustion 2-6h ahead based on burn rate and pending batch queue; triggers auto top-up | `forecast_float` | Variable |
| 3 | **Auto CTR Filing** — monitors every completed transaction against CBK/CBN/BoU thresholds; auto-populates and queues regulatory reports | `auto_file_ctr` | 500,000+ |
| 4 | **Intelligent Payment Routing** — ranks M-Pesa/Airtel/bank EFT rails by cost, speed, or reliability per transaction; falls back on failure | `get_optimal_route` | 300,000–800,000 |
| 5 | **Velocity-Adaptive Limits** — dynamic KYC tier limits using behavioral credit score (account age, success rate, dispute rate, AML flags) | `get_dynamic_limit` | Retention value |
| 6 | **FX Rate Locks with Micro-Hedging** — 5-minute guaranteed rate locks for cross-border payments; integrates CBK/CBN interbank rate feeds | `lock_fx_rate` | 150,000/importer |
| 7 | **Contextual Chargeback Intelligence** — auto-triage on dispute creation: 3DS/AVS/CVV scoring, win-probability estimate, rebuttal pre-population | `score_chargeback` | 228,000 |
| 8 | **Batch Failure Recovery** — classifies batch failures by error code; auto-fixes phone normalization, splits, rail switches; surfaces only unresolvable items | `recover_batch_failures` | 187,500 |
| 9 | **Intraday Settlement** — configurable 2h/4h/real-time cycles; 90% provisional credit at cycle open, final 10% at close; compresses T+1 to hours | `intraday_settlement` | Retention/NPS |
| 10 | **Offline-First Payment Widget Spec** — declarative JSON state machine contract (idle→pending→offline_queue→completed) any frontend can implement | `payment_widget_spec` | 2,700,000 |
| 11 | **Tokenised Recurring Mandates** — server-side billing schedules (daily/weekly/monthly/custom) with smart retry, channel failover, and pre-debit notifications | `create_recurring_mandate`, `execute_mandate_cycle` | 240,000/50k subs |
| 12 | **Network Graph Fraud Scoring** — bipartite sender→receiver graph over rolling 24h window; fan-in centrality flags coordinated fraud rings invisible to per-account rules | `score_receiver_network_fraud` | 2,000,000+ |
| 13 | **Stablecoin Settlement Bridge** — USDC intermediary for KES→NGN/GHS corridors; sub-60-second settlement vs 2-5 day SWIFT; corridor cost comparison API | `get_corridor_cost_estimate` | 450,000 |
| 14 | **Configurable Approval Workflows** — declarative JSON rules per tenant; threshold-triggered, cryptographically signed approvals with quorum and auto-expiry | `submit_for_approval`, `record_approval_decision` | Compliance/NPS |
| 15 | **Real-Time Health Dashboard** — rolling 5-minute success rate, TPM, and top failures per method; anomaly detection fires alerts via notify adapter | `get_payment_health_snapshot`, `configure_health_alert` | 50,000/incident |

**Recommended first sprint**: #1 (Dedup), #4 (Routing), #7 (Chargeback), #10 spec-only (Widget), #15 (Health) — achievable in 2 weeks, combined ROI > KES 3M/month.

## New Methods

### Intelligent Routing
```python
routes = await svc.get_optimal_route(
    amount=Decimal("50000"),
    recipient_capabilities=["mpesa", "airtel", "bank_eft"],
    currency="KES",
    priority="cost",   # "cost" | "speed" | "reliability"
)
# routes[0] = {"method": "mpesa_stk", "fee": "108.00", "eta_seconds": 30, "success_rate": 0.97}
```

### Network Fraud Scoring
```python
score = await svc.score_receiver_network_fraud(
    receiver_id="0722000999",
    sender_ids=["0711111111", "0722222222", ...],   # last 24h
    amounts=[Decimal("90000")] * 25,
    total_received=Decimal("2250000"),
)
# {"fraud_score": 0.87, "recommended_action": "block"}
```

### Recurring Mandates
```python
mandate = await svc.create_recurring_mandate(
    customer_id="cust-001",
    instrument_id="inst-abc",
    amount=Decimal("5000"),
    currency="KES",
    schedule="monthly",
    description="Microfinance loan repayment",
)
result = await svc.execute_mandate_cycle(mandate["mandate_id"])
```

### Real-Time Health Snapshot
```python
health = await svc.get_payment_health_snapshot(window_minutes=5)
# {"health_status": "degraded", "overall_success_rate": 0.82, "tpm": 43, "anomalies": [...]}

await svc.configure_health_alert(
    alert_name="mpesa_degraded",
    metric="success_rate",
    threshold=0.90,
    comparison="lt",
    notify_channel="sms",
    notify_recipient="254700000001",
)
```

### FX Rate Lock
```python
lock = await svc.lock_fx_rate(
    from_currency="KES",
    to_currency="USD",
    amount=Decimal("500000"),
    lock_duration_seconds=300,
)
# {"lock_id": "fxlock-KES-USD-500000", "locked_rate": "0.00775", "expires_at": "..."}
```

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
- `blocked` risk level produces a hard deny on authorization — no override path; re-screen at lower risk level to resolve
- Overcapture blocked deterministically: `overcapture: True` flag fires before any provider call
- Overrefund blocked similarly: refunds exceeding remaining captured balance denied before provider call
- Both `payment_batch` and `payment_event` require Bytewax routing — two separate guardrail rules
- Payout destination format validation (IBAN, phone, wallet ID) is the service layer's responsibility, not the rule engine
- Instrument vault token is mandatory: raw card/bank/mobile identifiers must never be stored unvaulted
- Semantic duplicate check catches soft duplicates (INV-001 vs INV-001-retry) within a rolling 5-minute window using Levenshtein similarity + phone + amount matching
- Batch failure recovery auto-classifies by error code — only `escalate`-class failures require human review; all others are auto-patched and re-queued

## Composability
- **Upstream**: `fintech_gateway` provides provider routing and authorization; `encr` handles instrument data encryption at vault registration
- **Downstream**: `fintech_neobanking`, `fintech_wallets`, `fintech_mobile`, `fintech_agency`, `fintech_lending`, and `fintech_remittance` all use Digital Payments as their payment execution backbone
- **Peer**: Deployed alongside `fintech_gateway` (provider connectivity) and `fintech_wallets` (stored-value ledger)

## Development Notes
- `_lte` / `_gt` condition suffixes for numeric comparisons: `amount_lte: 0` fires when amount <= 0
- `encr` is a required dependency distinct from `keym` — keym manages key references, encr handles cryptographic operations on instrument data
- Payment accounts are lightweight owners of instruments; they do not maintain a running balance (that is `fintech_wallets`' role)
- Segregation of duties: the same agent cannot initiate and authorize a payment (configuration flag, not currently enforced by rule engine)
- All v2.0 enhancements are pure Python with no external ML infrastructure — `domain/calculations.py` functions are independently testable
