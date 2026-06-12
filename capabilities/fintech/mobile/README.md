# Mobile Banking

## Overview
Mobile Banking (`fintech_mobile`) provides the customer-facing mobile channel layer: banking program governance, customer enrollment, trusted device binding with attestation, authentication factor registration (passcode, biometric, OTP, device binding, hardware key), account and wallet linking, mobile payment initiation, bill payment, airtime purchase, service request intake, notification preference management, mobile fraud event recording, QR payments, transaction velocity enforcement, standing orders, FX conversion, payment dispute management, spend analytics, SIM-swap detection, outbound webhooks, privacy-preserving balance proofs, KYC refresh, and loan disbursement. Surfaces neobanking, payments, cards, lending, BNPL, and agency services through iOS, Android, web, USSD, and SMS interfaces.

Payment currency must match the linked account's currency. High-value payments require human approval. Devices require attestation before they can be used as a trusted payment device. All mobile banking events stream to `apg.fintech.mobile.lifecycle` via Bytewax.

## Capability ID
`fintech_mobile`  Version: 2.0.0

## Provides
| Service | Description |
|---------|-------------|
| mobile_banking_program_governance | Register mobile banking programs with country, currency, and platform controls |
| mobile_customer_enrollment | Enroll customers with KYC, consent, AML, and fraud evidence |
| trusted_device_lifecycle | Bind trusted devices with fingerprint, attestation, and risk tier |
| mobile_authentication_factor_workflow | Register auth factors (passcode, biometric, OTP, device binding, hardware key) |
| mobile_account_linking | Link deposit, wallet, card, loan, savings, BNPL, and agency float accounts |
| mobile_payment_workflow | Initiate peer transfers, merchant payments, loan repayments, and wallet cash-outs |
| mobile_bill_payment_workflow | Record bill payments with biller and payment references |
| mobile_airtime_workflow | Record airtime purchases with operator and phone references |
| mobile_service_request_workflow | Open service requests with reason, evidence, and reviewer assignment |
| mobile_notification_workflow | Manage notification preferences with channel and consent controls |
| mobile_fraud_event_workflow | Record fraud events with severity and approval gates for high-severity cases |
| mobile_banking_agent_workflow | Register AI agents for device risk, payment review, and compliance |
| qr_payment_workflow | Generate HMAC-signed QR payment payloads and execute scanned QR payments |
| transaction_velocity_control | Sliding-window velocity enforcement per customer per time window |
| standing_order_management | Create and auto-execute scheduled recurring payments (daily/weekly/monthly) |
| fx_conversion_workflow | Indicative FX quotes with short-lived rate locks and execution |
| payment_dispute_management | Raise, track, and resolve payment disputes with SLA and credit control |
| spend_analytics | Categorised spend insights with savings rate and top-merchant ranking |
| sim_swap_detection | Carrier-level SIM swap detection with automatic account lock and fraud escalation |
| webhook_subscriptions | HMAC-signed outbound event webhooks for partner and integrator ecosystems |
| balance_threshold_proof | Privacy-preserving signed balance proofs without raw balance disclosure |
| kyc_refresh_workflow | Periodic KYC re-verification with CBK tier-based refresh cycles |
| loan_disbursement | Idempotent loan disbursement to nominated mobile account |
| biometric_step_up_auth | Cryptographic challenge re-authentication at high-value transaction time |
| device_risk_scoring | Runtime device risk re-evaluation using behavioural signals |
| offline_ussd_queue | Serialise USSD intents for replay on connectivity restoration |
| push_delivery_receipts | FCM/APNs delivery confirmation with retry and channel-fallback logic |

## Quick Start

```python
from apg_fintech_mobile.service import MobileBankingService

svc = MobileBankingService(tenant_id="datacraft", db_url="postgresql+asyncpg://...")

# Enroll a customer
customer = await svc.mobile_onboarding(
    msisdn="+254712345678",
    national_id="12345678",
    name="Jane Doe",
    kyc_reference="kyc-001",
)

# Initiate a peer transfer
payment = await svc.funds_transfer(
    from_account="acc-001",
    to_account="acc-002",
    amount=5000.0,
    reference="school-fees",
)

# Generate a QR payment (90 s TTL)
qr = await svc.generate_qr_payment(
    account_id="acc-001",
    amount=1500.0,
    reference="coffee-shop-001",
)

# Check transaction velocity before a high-frequency action
velocity = await svc.check_velocity(
    customer_id="cust-001",
    amount=10000.0,
)
if not velocity["allowed"]:
    raise ValueError("Velocity limit reached")
```

## World-Class Enhancements (v2.0)

1. **QR Code Payments** — HMAC-signed, short-lived (90 s) QR payloads for contactless POS and P2P payments. Replay-proof via expiry + signature gate.
2. **Biometric Step-Up Auth** — `step_up_auth()` issues a cryptographic challenge at transaction time. Supports FIDO2/WebAuthn and local biometric attestations.
3. **Transaction Velocity Checks** — Sliding-window counters (`check_velocity`) enforce per-customer count and volume limits. Plugs into `initiate_payment` and `funds_transfer` as a pre-flight gate.
4. **Scheduled / Standing Orders** — `create_standing_order()` supports daily/weekly/monthly recurrence. `process_due_standing_orders()` runs idempotently via Bytewax with `last_executed_at` guard.
5. **Multi-Currency FX Conversion** — `fx_conversion_quote()` returns a rate-locked quote; `accept_fx_quote()` executes it. Pluggable rate provider via `_fx_rate_provider` adapter.
6. **Device Risk Scoring** — `score_device_risk()` evaluates `{location, typing_cadence, app_version, rooted_flag, vpn_detected}` at runtime and updates the device's `risk_tier` dynamically.
7. **Offline USSD Queue** — `queue_ussd_transaction()` serialises user intent before network-bound ops; `drain_ussd_queue()` replays on reconnect with idempotency keys. Covers rural GPRS gaps.
8. **KYC Refresh Workflow** — `kyc_refresh()` diffs and records changed fields; `kyc_expiry_check()` returns days-to-expiry per CBK tier cycle (Tier 1: annual, Tier 2: biennial).
9. **Push Delivery Receipts** — `record_push_delivery_receipt()` captures FCM/APNs outcomes. `get_push_delivery_stats()` enables retry logic and SMS channel fallback on push failure.
10. **Loan Disbursement** — `disburse_loan()` validates approval, executes `funds_transfer`, and is fully idempotent — re-calling returns the existing disbursement record without double-credit.
11. **Payment Dispute Management** — `raise_payment_dispute()` creates a typed dispute with `sla_deadline` and `escalation_tier`. `resolve_dispute()` closes and issues credit. Fulfils CBK consumer protection requirements.
12. **SIM Swap Detection** — `detect_sim_swap()` queries a carrier adapter and auto-locks the account + records a `critical` fraud event requiring human approval for recovery.
13. **Spend Analytics** — `spend_analytics()` buckets transactions into 8 categories (food, transport, utilities, airtime, transfers, loan repayments, savings, other), returning `top_merchants` and `savings_rate`.
14. **Webhook Event Subscriptions** — `register_webhook()` stores HMAC-SHA256-signed subscriptions; `dispatch_webhook()` fans out to all matching subscribers with 3-retry exponential backoff.
15. **Zero-Knowledge Balance Proof** — `prove_balance_threshold()` returns `{threshold_met: bool, signature}` without exposing the raw balance. Upgradeable to ZK-SNARK via adapter interface.

## New Methods

### QR Payment Generation and Scan

```python
# Generate — 90-second default TTL, configurable up to 600 s
qr = await svc.generate_qr_payment(account_id="acc-merchant-01", amount=250.0, reference="order-9988")
# {"qr_id": "qr-a1b2c3d4", "payload": "...", "signature": "...", "expires_at": "...", ...}

# Scan and execute from payer side — returns declined dict on expiry, never raises
result = await svc.scan_qr_payment(qr_payload=qr["payload"], payer_account_id="acc-cust-07")
# {"status": "completed", "qr_id": "...", "payment_method": "qr", ...}
```

### Transaction Velocity Gate

```python
# Pre-flight check before any high-frequency operation
v = await svc.check_velocity(
    customer_id="cust-001",
    amount=20_000.0,
    window_seconds=3600,
    max_count=10,
    max_volume=500_000.0,
)
# {"allowed": True, "current_count": 3, "current_volume": 45000.0, "remaining_capacity": 455000.0}
```

### FX Conversion Quote and Execution

```python
# Get a rate-locked quote (30-second TTL by default)
quote = await svc.fx_conversion_quote(from_currency="KES", to_currency="USD", amount=10_000.0)
# {"quote_id": "fxq-abc123", "rate": 131.25, "converted_amount": 74.05, "expires_at": "...", ...}

# Accept before TTL expires — idempotent on double-call
conversion = await svc.accept_fx_quote(quote_id=quote["quote_id"], account_id="acc-001")
# {"conversion_id": "fx-fxq-abc123-...", "deducted": 10000.0, "status": "completed", ...}
```

### Standing Order Creation

```python
# Monthly salary standing order
order = await svc.create_standing_order(
    account_id="acc-payroll-01",
    to_account="acc-emp-jane",
    amount=85_000.0,
    frequency="monthly",
    start_date="2026-07-01",
    end_date="2027-06-30",
)
# {"order_id": "so-...", "status": "active", "next_execution": "2026-07-01", ...}

# Process all due orders (called by Bytewax scheduler)
result = await svc.process_due_standing_orders()
# {"processed": 12, "skipped": 3, "failed": 0, "executed_at": "..."}
```

### Privacy-Preserving Balance Proof

```python
# Prove balance >= 50,000 KES to a lender without revealing the actual balance
proof = await svc.prove_balance_threshold(
    account_id="acc-001",
    threshold=50_000.0,
    verifier_id="lender-sacco-ke",
)
# {"account_id_hash": "...", "threshold_met": True, "verifier_id": "lender-sacco-ke",
#  "signed_at": "...", "signature": "...", "threshold": 50000.0}
```

## Requires
| Capability | Purpose |
|------------|---------|
| auth | Authentication |
| audl | Audit trail |
| ntfy | Customer notifications |
| nlpc | NLP for service requests |
| keym | Key management |
| fintech_payments | Payment execution |
| fintech_wallets | Wallet account linking and funding |
| fintech_cards | Card account linking |
| fintech_kyc | Customer identity verification |
| fintech_aml | AML screening |
| fintech_fraud | Mobile fraud signal scoring |
| fintech_neobanking | Neobank account linking |
| fintech_lending | Loan account linking and repayment |
| fintech_bnpl | BNPL account linking |
| fintech_agency | Agency float account linking |

## Configuration Reference
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| platforms.supported | list | ios, android, web, ussd, sms | Supported mobile platforms |
| auth_factors.supported_types | list | passcode, biometric, device_binding, otp, hardware_key | Auth factor types |
| payments.supported_types | list | peer_transfer, merchant_payment, bill_payment, airtime, loan_repayment, savings_transfer, card_payment, wallet_cash_out | Payment types |
| payments.high_value_threshold | number | 100000 | Amount requiring approval |
| fraud_events.supported_severities | list | low, medium, high, critical | Fraud event severity levels |
| qr_payments.default_ttl_seconds | number | 90 | Default QR payload validity window |
| velocity.default_window_seconds | number | 3600 | Velocity check sliding window |
| velocity.default_max_count | number | 10 | Max transactions per window |
| velocity.default_max_volume | number | 500000 | Max volume per window (KES) |
| fx.default_quote_ttl_seconds | number | 30 | FX rate lock duration |
| fx.conversion_fee_rate | number | 0.015 | FX conversion fee (1.5%) |
| sim_swap.detection_window_hours | number | 48 | Hours within which a SIM swap triggers lock |
| dispute.default_sla_hours | number | 48 | CBK dispute resolution SLA |

## API Routes
| Name | Path | Method | Permission | Group |
|------|------|--------|------------|-------|
| dashboard | /fintech-mobile/dashboard | GET | fintech_mobile:view | Overview |
| programs | /fintech-mobile/programs | GET/POST | fintech_mobile:manage_programs | Programs |
| customers | /fintech-mobile/customers | GET/POST | fintech_mobile:customers | Customers |
| devices | /fintech-mobile/devices | GET/POST | fintech_mobile:devices | Security |
| auth_factors | /fintech-mobile/auth-factors | GET/POST | fintech_mobile:auth | Security |
| account_links | /fintech-mobile/account-links | GET/POST | fintech_mobile:accounts | Accounts |
| payments | /fintech-mobile/payments | GET/POST | fintech_mobile:payments | Payments |
| bills | /fintech-mobile/bills | GET/POST | fintech_mobile:bills | Payments |
| airtime | /fintech-mobile/airtime | GET/POST | fintech_mobile:airtime | Payments |
| service_requests | /fintech-mobile/service-requests | GET/POST | fintech_mobile:service | Servicing |
| notifications | /fintech-mobile/notifications | GET/POST | fintech_mobile:notifications | Engagement |
| fraud_events | /fintech-mobile/fraud-events | GET/POST | fintech_mobile:fraud | Risk |
| agents | /fintech-mobile/agents | GET/POST | fintech_mobile:admin | Automation |
| settings | /fintech-mobile/settings | GET/POST | fintech_mobile:admin | Administration |
| qr_payments | /fintech-mobile/qr-payments | GET/POST | fintech_mobile:payments | Payments |
| standing_orders | /fintech-mobile/standing-orders | GET/POST/DELETE | fintech_mobile:payments | Payments |
| fx_quotes | /fintech-mobile/fx-quotes | GET/POST | fintech_mobile:payments | Payments |
| disputes | /fintech-mobile/disputes | GET/POST | fintech_mobile:service | Servicing |
| analytics | /fintech-mobile/analytics | GET | fintech_mobile:view | Analytics |
| webhooks | /fintech-mobile/webhooks | GET/POST/DELETE | fintech_mobile:admin | Integration |
| balance_proof | /fintech-mobile/balance-proof | POST | fintech_mobile:accounts | Accounts |

## Business Rules
| Rule | Condition | Effect |
|------|-----------|--------|
| device_attestation_required | Device binding without attestation | deny |
| device_fingerprint_required | Device binding without fingerprint | deny |
| auth_strength_required | Auth factor without strength reference | deny |
| payment_currency_matches_link | Payment currency differs from account link | deny |
| payment_risk_reference_required | Payment without risk reference | deny |
| high_value_payment_requires_approval | Payment > 100,000 without approval | require_review |
| notification_consent_required | Notification preference without consent | deny |
| high_severity_fraud_requires_approval | High-severity fraud event without approval | require_review |
| mobile_batch_requires_bytewax | Batch without Bytewax | deny |
| privileged_mobile_agent_action_requires_human_approval | AI agent privileged scope without approval | deny |
| qr_payment_ttl_enforced | QR payload older than TTL | deny |
| qr_payload_signature_required | QR scan without valid HMAC signature | deny |
| velocity_limit_per_customer | Exceeds sliding-window count or volume | deny |
| fx_quote_must_be_accepted_before_expiry | FX quote accepted after TTL | deny |
| webhook_url_must_use_https | Webhook registration over HTTP | deny |
| sim_swap_auto_locks_account | SIM swap detected within 48 hours | lock + critical_fraud_event |
| loan_disbursement_only_for_approved_loans | Disbursement on declined or pending loan | deny |
| dispute_cannot_be_reopened | Dispute already resolved | deny |

## Data Models
| Model | Key Fields |
|-------|-----------|
| MobileProgram | id, name, owner_id, country, currency, supported_platforms, status |
| MobileCustomer | id, customer_reference, kyc_profile_id, country, consent_reference, aml_reference, fraud_reference, status |
| TrustedDevice | id, customer_id, platform, fingerprint, attestation_reference, risk_tier, status |
| AuthFactor | id, customer_id, device_id, factor_type, strength_reference, status |
| AccountLink | id, customer_id, link_type, account_reference, currency, provider_reference, status |
| MobilePayment | id, customer_id, device_id, account_link_id, payment_type, amount, currency, recipient_reference, risk_reference, status |
| BillPayment | id, biller_reference, payment_id, payment_type |
| AirtimePurchase | id, operator_reference, phone_reference, payment_id |
| ServiceRequest | id, customer_id, reason, evidence_references, reviewer_id, status |
| NotificationPreference | id, customer_id, channel, consent_reference |
| MobileFraudEvent | id, customer_id, severity, evidence_references, human_approval_reference |
| StandingOrder | order_id, account_id, to_account, amount, frequency, start_date, end_date, last_executed_at |
| FxQuote | quote_id, from_currency, to_currency, amount, rate, fee_rate, converted_amount, expires_at |
| PaymentDispute | dispute_id, customer_id, payment_id, dispute_reason, amount_disputed, status, sla_deadline |
| WebhookSubscription | webhook_id, tenant_id, url, events, status, delivery_count |

## Streaming Events
Events emitted to the fintech event stream via Bytewax.
| Event | Trigger |
|-------|---------|
| mobile_program_registered | Program registered |
| mobile_customer_enrolled | Customer enrolled |
| trusted_device_bound | Device bound |
| auth_factor_registered | Auth factor registered |
| account_linked | Account linked to mobile profile |
| mobile_payment_initiated | Payment initiated |
| bill_payment_recorded | Bill payment recorded |
| airtime_purchased | Airtime purchased |
| service_request_opened | Service request opened |
| notification_preference_set | Notification preference saved |
| fraud_event_recorded | Fraud event recorded |
| mobile_agent_registered | AI agent registered |
| qr_payment_generated | QR payment payload created |
| qr_payment_executed | QR payment scan completed |
| velocity_check_performed | Transaction velocity evaluated |
| standing_order_created | Standing order registered |
| standing_order_executed | Standing order payment processed |
| fx_quote_generated | FX conversion quote issued |
| fx_conversion_executed | FX conversion accepted and deducted |
| payment_dispute_raised | Customer dispute opened |
| payment_dispute_resolved | Dispute closed with optional credit |
| sim_swap_detected | SIM swap signal from carrier |
| webhook_registered | Outbound webhook subscription created |
| webhook_dispatched | Event fanned out to webhook subscribers |
| balance_threshold_proved | Balance proof signed for verifier |
| mobile_kyc_refreshed | Customer KYC re-verified |
| loan_disbursed | Approved loan credited to mobile account |
| spend_analytics_generated | Spend categorisation report produced |

## Edge Cases Handled
- Payment currency must match the linked account currency exactly — cross-currency mobile payments are denied; FX conversion must happen at the account or wallet level before the mobile payment is initiated
- Device fingerprint and attestation are both required for device binding; either one missing is a deny; attestation verifies the device is genuine hardware, fingerprint identifies the specific device instance
- Auth factor strength reference is mandatory — a strength reference ties the factor to a policy document defining its assurance level (e.g., NIST AAL2); factors without this reference cannot be used for high-assurance operations
- Notification preferences require consent even for opt-out — the consent record documents the customer's explicit communication preference choice
- High-severity fraud events (high, critical) require human approval before recording; low/medium fraud events can be recorded without approval
- QR payloads expire after configurable TTL (default 90 s) to prevent replay; `scan_qr_payment` returns a declined status rather than raising on expired payloads
- `check_velocity` uses per-customer sliding windows; if a transaction is allowed, its amount is immediately counted into the window to prevent races
- `process_due_standing_orders` is idempotent — it uses `last_executed_at` as a guard so re-running within the same period skips already-executed orders
- `accept_fx_quote` marks the quote as consumed to prevent a second execution of the same conversion
- SIM swap detection auto-records a `critical` fraud event; the fraud event itself requires human approval per the existing high-severity rule — creating a mandatory human-in-the-loop before any account recovery
- `disburse_loan` is idempotent — calling it twice with the same `application_id` returns the existing disbursement record without a double-credit
- `prove_balance_threshold` returns `threshold_met: bool` without exposing the raw balance; upgrade path to ZK-SNARK is via the `_fx_rate_provider` adapter pattern

## Composability
- **Upstream**: `fintech_kyc` and `fintech_aml` provide enrollment evidence; `fintech_fraud` provides per-payment risk signals; platform-specific biometric SDKs are adapter boundaries behind `auth`
- **Downstream**: All payment, wallet, card, lending, BNPL, and agency capabilities are accessed via mobile through account links; `fintech_neobanking` is the primary account backing
- **Peer**: Deployed alongside `fintech_neobanking` (the underlying account layer) and `fintech_payments` (payment execution)
- **Integrators**: Partner systems subscribe via `register_webhook` to receive real-time HMAC-signed events; no polling required

## Development Notes
- USSD and SMS platforms are first-class supported platforms — this reflects the African market context where feature phones and USSD banking are primary channels
- `SUPPORTED_ACCOUNT_LINK_TYPES` includes `agency_float` — this allows agency float accounts to be linked and managed via mobile, enabling mobile-first agent operations
- Bill payment and airtime records both require a matching payment transaction (`payment_type_matches` flag) — the rule prevents a bill payment record being created for a non-bill payment transaction
- Auth factor types map to the customer authentication methods; `device_binding` is distinct from biometric — it binds the factor cryptographically to the device
- Velocity windows are held in `_velocity_windows` (in-memory); swap for Redis `ZADD`/`ZRANGEBYSCORE` in production for cross-process safety
- FX rate provider is a stub (deterministic MD5-based rate); inject a live rate adapter via `_fx_rate_provider` attribute on `MobileBankingService`
- Webhook delivery is synchronous in the stub; in production enqueue to Celery/Bytewax with HMAC-SHA256 `X-Signature` header and 3-retry exponential backoff
- Standing orders: CBK regulations require that the customer can cancel any standing order at any time; set `status = "cancelled"` on the order record
