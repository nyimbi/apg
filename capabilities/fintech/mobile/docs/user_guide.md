# Mobile Banking

**Capability ID**: `fintech_mobile` | **Domain**: `fintech` | **Version**: `2.0.0`

## Description

Mobile Banking provides the customer-facing mobile channel layer: banking program governance, customer enrollment, trusted device binding with attestation, authentication factor registration, account and wallet linking, mobile payment initiation, bill payment, airtime purchase, QR payments, transaction velocity enforcement, standing orders, FX conversion, payment dispute management, spend analytics, SIM-swap detection, outbound webhooks, privacy-preserving balance proofs, KYC refresh, and loan disbursement. It surfaces neobanking, payments, cards, lending, BNPL, and agency services through iOS, Android, web, USSD, and SMS interfaces.

## Installation

```bash
pip install apg-fintech-mobile
```

## Quick Start

```python
import asyncio
from capabilities.fintech.mobile.service import MobileBankingService

svc = MobileBankingService(tenant_id="acme", actor_id="ops-user")

# Onboard a customer
result = asyncio.run(svc.mobile_onboarding(
    msisdn="0722000001",
    id_number="12345678",
    kyc_data={"first_name": "Jane", "last_name": "Doe", "date_of_birth": "1990-01-01", "country": "KE"},
))
print(result["customer_id"])  # mob-...

# Check balance
bal = asyncio.run(svc.account_balance_inquiry("acc-jane"))
print(bal["balance"], bal["currency"])

# Transfer funds
txn = asyncio.run(svc.funds_transfer("acc-jane", "acc-john", 5000.0, "school-fees"))
print(txn["status"])  # completed
```

## Provides

- `mobile_banking_program_governance`
- `mobile_customer_enrollment`
- `trusted_device_lifecycle`
- `mobile_authentication_factor_workflow`
- `mobile_account_linking`
- `mobile_payment_workflow`
- `mobile_bill_payment_workflow`
- `mobile_airtime_workflow`
- `mobile_service_request_workflow`
- `mobile_notification_workflow`
- `mobile_fraud_event_workflow`
- `mobile_banking_agent_workflow`
- `qr_payment_workflow`
- `transaction_velocity_control`
- `standing_order_management`
- `fx_conversion_workflow`
- `payment_dispute_management`
- `spend_analytics`
- `sim_swap_detection`
- `webhook_subscriptions`
- `balance_threshold_proof`
- `kyc_refresh_workflow`
- `loan_disbursement`

## Requires

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`
- `fintech_kyc`
- `fintech_aml`
- `fintech_fraud`
- `fintech_payments`
- `fintech_lending`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/fintech-mobile/dashboard` | `fintech_mobile:view` | Overview |
| `/fintech-mobile/programs` | `fintech_mobile:manage_programs` | Programs |
| `/fintech-mobile/customers` | `fintech_mobile:customers` | Customers |
| `/fintech-mobile/devices` | `fintech_mobile:devices` | Security |
| `/fintech-mobile/auth-factors` | `fintech_mobile:auth` | Security |
| `/fintech-mobile/account-links` | `fintech_mobile:accounts` | Accounts |
| `/fintech-mobile/payments` | `fintech_mobile:payments` | Payments |
| `/fintech-mobile/bills` | `fintech_mobile:bills` | Payments |
| `/fintech-mobile/airtime` | `fintech_mobile:airtime` | Payments |
| `/fintech-mobile/qr-payments` | `fintech_mobile:payments` | Payments |
| `/fintech-mobile/standing-orders` | `fintech_mobile:payments` | Payments |
| `/fintech-mobile/fx-quotes` | `fintech_mobile:payments` | Payments |
| `/fintech-mobile/disputes` | `fintech_mobile:service` | Servicing |
| `/fintech-mobile/analytics` | `fintech_mobile:view` | Analytics |
| `/fintech-mobile/webhooks` | `fintech_mobile:admin` | Integration |
| `/fintech-mobile/balance-proof` | `fintech_mobile:accounts` | Accounts |

## Key Service Methods

### Core Lifecycle
- `describe()` — return capability contract
- `evaluate(context)` — evaluate policy rules
- `register_program(...)` — register a mobile banking program
- `enroll_customer(...)` — enroll customer with KYC/AML/consent
- `bind_device(...)` — bind trusted device with attestation
- `register_auth_factor(...)` — register passcode/biometric/OTP factor
- `link_account(...)` — link deposit/wallet/card account
- `initiate_payment(...)` — initiate mobile payment

### Async Banking Operations
- `mobile_onboarding(msisdn, id_number, kyc_data)` — full KYC onboarding
- `account_balance_inquiry(account_id, channel)` — cached balance fetch
- `mini_statement(account_id, limit)` — last N transactions
- `funds_transfer(from_account, to_account, amount, reference)` — P2P transfer
- `bill_payment(account_id, biller_code, account_number, amount)` — pay bill
- `airtime_purchase(account_id, phone, amount, provider)` — buy airtime
- `loan_application_mobile(account_id, amount, tenor)` — credit-scored loan application
- `disburse_loan(application_id, disbursement_account)` — disburse approved loan
- `ussd_session(msisdn, session_id, input_text)` — USSD state machine step
- `send_otp(msisdn, purpose)` — dispatch OTP
- `close_account(account_id, reason, approved_by)` — close account

### QR Payments
- `generate_qr_payment(account_id, amount, reference, ttl_seconds)` — HMAC-signed QR payload (default 90 s TTL)
- `scan_qr_payment(qr_payload, payer_account_id)` — validate and execute QR payment

### Velocity & Standing Orders
- `check_velocity(customer_id, amount, window_seconds, max_count, max_volume)` — sliding-window gate
- `create_standing_order(account_id, to_account, amount, frequency, start_date, end_date)` — schedule recurring payment
- `process_due_standing_orders()` — execute all orders due today (idempotent)

### FX Conversion
- `fx_conversion_quote(from_currency, to_currency, amount, ttl_seconds)` — indicative rate quote
- `accept_fx_quote(quote_id, account_id)` — lock rate and execute conversion

### Disputes & Compliance
- `raise_payment_dispute(customer_id, payment_id, dispute_reason, amount_disputed)` — open dispute with SLA
- `resolve_dispute(dispute_id, resolution, credited_amount, resolved_by)` — close dispute, optional credit
- `kyc_refresh(customer_id, updated_kyc_data, verifier_reference)` — CBK-compliant KYC refresh
- `cbk_mobile_banking_return(period)` — regulatory return for CBK

### Analytics & Insights
- `spend_analytics(account_id, period)` — categorised spend with top merchants
- `mobile_analytics(period)` — platform-wide metrics (customers, payments, fraud)
- `account_statement_mobile(account_id, period)` — full statement export

### Security & Risk
- `detect_sim_swap(msisdn, device_id, swap_recency_hours)` — carrier SIM-swap detection
- `prove_balance_threshold(account_id, threshold, verifier_id)` — privacy-preserving balance proof
- `device_revocation(device_id, reason)` — revoke lost/compromised device
- `record_fraud_event(...)` — record fraud with severity gate
- `fraud_report_mobile(customer_id, incident_description, amount)` — customer-initiated fraud report

### Notifications & Webhooks
- `push_notification_settings(customer_id, preferences)` — update push notification prefs
- `register_webhook(url, events, signing_secret)` — register outbound event hook (HTTPS only)
- `dispatch_webhook(event_type, payload)` — fan out event to matching subscriptions

### Bulk & Admin
- `bulk_enroll_customers(customers)` — batch enroll
- `bulk_payment_upload(account_id, payments)` — batch transfers
- `export_mobile_data(fmt)` — export to CSV/JSON/Excel
- `dashboard_summary(tenant_id)` — tenant metrics overview
- `health_check()` — service liveness

## Usage Examples

### QR Payment Flow

```python
# Merchant generates QR
qr = asyncio.run(svc.generate_qr_payment("acc-merchant", 1500.0, "invoice-42", ttl_seconds=90))

# Customer scans QR
result = asyncio.run(svc.scan_qr_payment(qr["payload"], "acc-customer"))
assert result["status"] == "completed"
```

### Standing Order

```python
order = asyncio.run(svc.create_standing_order(
    account_id="acc-alice",
    to_account="acc-landlord",
    amount=25000.0,
    frequency="monthly",
    start_date="2026-07-01",
    end_date="2027-06-30",
))
# Bytewax calls this daily:
summary = asyncio.run(svc.process_due_standing_orders())
```

### FX Conversion

```python
quote = asyncio.run(svc.fx_conversion_quote("KES", "USD", 100000.0, ttl_seconds=30))
result = asyncio.run(svc.accept_fx_quote(quote["quote_id"], "acc-traveller"))
```

### Velocity Check

```python
gate = asyncio.run(svc.check_velocity("cust-001", 50000.0, window_seconds=3600, max_count=5))
if gate["allowed"]:
    asyncio.run(svc.funds_transfer("acc-001", "acc-002", 50000.0, "ref-001"))
```

### Balance Threshold Proof

```python
proof = asyncio.run(svc.prove_balance_threshold("acc-alice", 100000.0, verifier_id="mortgage-bank"))
# Share proof["signature"] and proof["threshold_met"] with verifier — raw balance stays private
```

### Payment Dispute

```python
dispute = asyncio.run(svc.raise_payment_dispute(
    "cust-001", "pay-xyz", "unauthorised_transaction", amount_disputed=3500.0
))
# After investigation:
asyncio.run(svc.resolve_dispute(dispute["dispute_id"], "credited", credited_amount=3500.0))
```

### Webhook Integration

```python
webhook = asyncio.run(svc.register_webhook(
    url="https://partner.example.com/hooks/mobile",
    events=["mobile_payment_initiated", "fraud_event_recorded"],
    signing_secret="s3cr3t-key",
))
# Partner verifies HMAC-SHA256 X-Signature header on each delivery
```

## Interoperability

`fintech_mobile` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use fintech_mobile;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `FINTECH_MOBILE_`.

Key environment variables:
| Variable | Description |
|----------|-------------|
| `FINTECH_MOBILE_QR_TTL_SECONDS` | Default QR payload TTL (default: 90) |
| `FINTECH_MOBILE_VELOCITY_WINDOW` | Default velocity window in seconds (default: 3600) |
| `FINTECH_MOBILE_FX_QUOTE_TTL` | FX quote lock duration in seconds (default: 30) |
| `FINTECH_MOBILE_SIM_SWAP_HOURS` | SIM swap detection window (default: 48) |
| `OLLAMA_BASE_URL` | Enable ML-powered security scoring via local Ollama |

## Further Reading

- `service.py` — Business logic implementation (all async methods)
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 detailed improvement proposals
- `cap_spec.md` — Full capability specification
