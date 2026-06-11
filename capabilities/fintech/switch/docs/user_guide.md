# Payment Switch — User Guide

**Capability ID**: `fintech_switch` | **Domain**: `fintech` | **Version**: `1.2.0`

© 2025 Datacraft | Author: Nyimbi Odero

---

## Overview

`fintech_switch` is a production-grade ISO 8583 / ISO 20022 payment switch for the APG
platform. It covers the full lifecycle of a payment transaction:

1. **Routing** — rule-based with cost/availability scoring
2. **Authorisation** — PAN/phone, CNP, 3DS, EMV chip (ARQC/ARPC)
3. **Security** — PIN block translation, HSM key management, PAN tokenisation
4. **Settlement** — batch close, clearing file generation, reconciliation
5. **Operations** — circuit-breaker, failover, health checks, event bus

---

## Installation

```bash
pip install apg-fintech-switch
```

Requires Python 3.11+. PostgreSQL 15+ recommended for production.

---

## Quick Start

```python
import asyncio
from apg_fintech_switch.service import PaymentSwitchService

async def main():
    svc = PaymentSwitchService(tenant_id="acme_bank")

    # 1. Route a transaction
    routed = await svc.route_transaction(
        transaction_data={
            "amount": 2500.00,
            "currency": "KES",
            "transaction_type": "purchase",
            "channel": "pos",
            "merchant_id": "MID-NAIROBI-001",
            "pan_masked": "4111****1111",
        },
        routing_rules=[
            {
                "name": "visa_kes_pos",
                "network": "visa",
                "priority": 1,
                "conditions": {"currency": "KES", "channel": "pos"},
            },
        ],
    )
    print(f"Routed via {routed['network']} — STAN {routed['stan']}")

    # 2. Authorise
    auth = await svc.switch_authorisation(
        "254700000000", 2500.00, "MID-NAIROBI-001", "KES"
    )
    print(f"Auth result: {auth['response_code']} — {auth['response_message']}")

asyncio.run(main())
```

---

## Configuration

All configuration is tenant-scoped. Set via environment variables prefixed with
`FINTECH_SWITCH_` or pass `db_url` to `PaymentSwitchService`.

| Variable | Default | Description |
|----------|---------|-------------|
| `FINTECH_SWITCH_DB_URL` | in-memory | PostgreSQL async DSN |
| `FINTECH_SWITCH_TENANT_ID` | `default` | Tenant identifier |
| `FINTECH_SWITCH_HSM_HOST` | `localhost` | HSM network address |
| `FINTECH_SWITCH_HSM_PORT` | `1500` | HSM TCP port |

---

## Feature Reference

### 1. Transaction Routing

```python
result = await svc.route_transaction(transaction_data, routing_rules)
```

Rules are evaluated in ascending `priority` order; first match wins. Fallback network
is `interbank`. Duplicate STAN detection runs on every call.

**Dry-run routing table update:**

```python
diff = await svc.routing_table_update(
    rules=new_rules,
    effective_from="2026-07-01",
    updated_by="ops-team",
    dry_run=True,
)
```

### 2. Authorisation

**Standard authorisation** (PAN or phone):

```python
auth = await svc.switch_authorisation(
    pan_or_phone="0700000000",
    amount=1000.0,
    merchant_id="MID-001",
    currency="KES",
    channel="mobile",
)
```

**Idempotent authorisation** (safe to retry):

```python
auth = await svc.idempotent_authorise(
    idempotency_key="550e8400-e29b-41d4-a716-446655440000",
    pan_or_phone="0700000000",
    amount=1000.0,
    merchant_id="MID-001",
    currency="KES",
)
# Retrying with the same key returns the cached response:
# auth["idempotent_replay"] == True
```

**Card-not-present (e-commerce):**

```python
cnp = await svc.card_not_present_auth(
    token="4111****1111",
    amount=500.0,
    cvv_result="M",   # M=match, N=no-match, P=not-processed, U=unavailable
    avs_result="Y",   # Y=full match, A=address only, Z=zip only, N=no match
)
```

**3D Secure:**

```python
auth3ds = await svc.authenticate_3ds(
    pan="4111111111111111",
    amount=500.0,
    eci="05",   # 05/02 = fully authenticated, 06/01 = attempted, 07 = not authenticated
    cavv="AAAA...",
)
```

### 3. EMV Chip — ARQC/ARPC

```python
emv = await svc.emv_cryptogram_verify(
    pan_masked="1111",
    arqc="A1B2C3D4E5F60718",
    atc="0042",
    amount=3000.0,
    currency="KES",
    terminal_id="TID-001",
)
# emv["arqc_verified"] True/False
# emv["arpc"]          Application Cryptogram response to send back to card
# emv["emv_response_data"] — DE55 response data for the terminal
```

### 4. PAN Tokenisation

```python
# Tokenise
tok = await svc.tokenise_pan(
    pan="4111111111111111",
    requestor_id="APPLE-PAY-001",
    scheme="visa",
    expiry_mmyy="1228",
)
# tok["token"] is a 16-digit value that passes Luhn and preserves BIN

# De-tokenise (no clear-text PAN returned)
detail = await svc.detokenise_pan(
    token=tok["token"],
    requestor_id="APPLE-PAY-001",
    reason="fraud_investigation",
)
# detail["pan_masked"] == "411111****1111"
# detail["pan_hash"]   — SHA-256 of original PAN
```

### 5. HSM and PIN Management

```python
# HSM key operation
hsm = await svc.key_management_hsm(
    operation="rotate",   # inject | generate | rotate | verify
    key_type="ZPK",
    zone="VISA_ACQ",
)

# PIN block verification
pin_result = await svc.pin_verification(
    pan_masked="1111",
    pin_block="041259AB3D6E7F80",  # ANSI X9.8 Format 0
    key_id="ZPK-001",
)
# pin_result["pin_verified"] True/False
# pin_result["hsm_response"] "00" | "55"
```

### 6. Scheme Management

```python
# Register scheme
await svc.scheme_registration(
    scheme_name="pesalink",
    credentials={"api_key": "...", "certificate": "..."},
    effective_date="2026-01-01",
)

# Update interchange rates
await svc.scheme_rate_update(
    scheme="visa",
    rate_table={
        "purchase":   {"rate_pct": 0.0165, "flat_fee_kes": 5.0},
        "refund":     {"rate_pct": 0.0,    "flat_fee_kes": 0.0},
        "quasi_cash": {"rate_pct": 0.02,   "flat_fee_kes": 10.0},
    },
    effective_date="2026-07-01",
    updated_by="treasury-admin",
)
```

### 7. Network Operations

```python
# Register endpoint
await svc.network_interface_register(
    network="pesalink",
    host="192.168.10.5",
    port=7979,
    protocol="iso8583_tcp",
)

# Circuit-breaker state
cb = await svc.network_circuit_breaker_status()
for b in cb["circuit_breakers"]:
    print(f"{b['network']}: {b['state']} ({b['error_rate_pct']}% errors)")

# Manual failover
await svc.downtime_failover(primary_route="visa", failover_route="interswitch")
```

### 8. Settlement and Clearing

```python
# Dry-run to preview batch totals
preview = await svc.settlement_batch_close(
    settlement_date="2026-06-11",
    scheme="visa",
    dry_run=True,
)
print(f"Transactions: {preview['transaction_count']}, Total: KES {preview['total_amount']:,.2f}")

# Close the batch and generate clearing file
batch = await svc.settlement_batch_close(
    settlement_date="2026-06-11",
    scheme="visa",
)

# Reconcile
recon = await svc.reconciliation_switch(
    recon_date="2026-06-11",
    scheme="visa",
)
print(f"Variance: KES {recon['variance']:,.2f} — {recon['status']}")

# Export
export = await svc.export_settlement_file(
    settlement_date="2026-06-11",
    scheme="visa",
    fmt="iso20022",
)
```

### 9. Fraud and Compliance

```python
# Velocity check
vel = await svc.fraud_velocity_check(
    pan_or_phone="254700000000",
    window_seconds=300,  # 5-minute window
    max_attempts=3,
)
if vel["velocity_exceeded"]:
    print(f"Breach: {vel['current_count']} attempts in {vel['window_seconds']}s")

# Scheme compliance
comp = await svc.scheme_compliance_check(
    transaction_id="txn-001",
    scheme="pesalink",
)
if not comp["compliant"]:
    print(comp["violations"])
```

### 10. Analytics and Reporting

```python
# Current-month KPI snapshot
dashboard = await svc.switch_analytics_dashboard()

# Detailed period analytics
analytics = await svc.switch_analytics("2026-Q2")

# Network performance (includes uptime and latency P99)
perf = await svc.network_performance_metrics("2026-06")

# Named report
report = await svc.switch_report("2026-06", "transaction_summary")
```

### 11. Scheme Certification

```python
report = await svc.generate_certification_report(
    scheme="visa",
    test_suite=[
        {
            "test_id":    "VISA-ADVT-001",
            "scenario":   "approved",
            "expected_rc": "00",
            "description": "Standard purchase approval",
        },
        {
            "test_id":    "VISA-ADVT-002",
            "scenario":   "velocity_exceeded",
            "expected_rc": "61",
            "description": "Velocity limit breach",
        },
    ],
)
print(f"Certification: {report['verdict']} ({report['passed']}/{report['test_count']})")
```

### 12. Event Bus

```python
event = await svc.switch_event_publish(
    event_type="scheme_degraded",
    payload={"scheme": "visa", "error_rate_pct": 12.5, "action": "failover_triggered"},
    topic="switch.alerts",
)
# event["chain_hash"] — SHA-256 chaining to prior event (tamper evidence)
```

---

## ISO Standards

### ISO 8583

```python
# Parse raw message
parsed = await svc.iso8583_parse("01004000000000000000123456001000")

# Build message
msg = await svc.iso8583_build(
    mti="0100",
    fields={"f2": "411111****1111", "f3": "000000", "f4": "000000002500"},
)
```

### ISO 20022

```python
iso20022 = await svc.iso20022_conversion(
    iso8583_txn_id="txn-001",
    target_format="pacs.008",
)
```

---

## Mobile Money

```python
# PesaLink validation before routing
val = await svc.pesalink_validation(
    account_number="12345678901",
    bank_code="11",
    amount=50_000.0,
)

# M-Pesa STK Push callback
cb = await svc.mpesa_api_callback(
    checkout_request_id="ws_CO_1234567890",
    result_code=0,          # 0 = success
    result_desc="The service request is processed successfully.",
    amount=500.0,
)
```

---

## Exception Handling and Replay

```python
# Log and resolve an exception
exc = await svc.exception_management(
    transaction_id="txn-001",
    exception_type="timeout",         # timeout | duplicate | reversed | missing_response | format_error
    resolution="retried_on_backup",
)

# Replay a failed transaction
replay = await svc.transaction_replay(
    transaction_id="txn-001",
    target_system="visa_net",
)

# Chargeback
cb = await svc.chargebacks_processing(
    authorization_id="auth-001",
    chargeback_reason="goods_not_delivered",
    amount=2500.0,
)
```

---

## Health and Observability

```python
# Component health
health = await svc.switch_health_check()
# {"overall_status": "ok"|"degraded", "networks_up": N, "networks_down": N, ...}

# Load distribution
lb = await svc.load_balancing_status()
# {"load_distribution_pct": {"visa": 68.2, "pesalink": 31.8}, ...}

# Circuit breakers
cb = await svc.network_circuit_breaker_status()
# {"circuit_breakers": [{"network": "visa", "state": "CLOSED", ...}], ...}
```

---

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/fintech-switch/dashboard` | `fintech_switch:view` | Overview |
| `/fintech-switch/routing` | `fintech_switch:manage_routing` | Routing |
| `/fintech-switch/transactions` | `fintech_switch:monitor` | Transactions |
| `/fintech-switch/channels` | `fintech_switch:manage_channels` | Channels |
| `/fintech-switch/security` | `fintech_switch:manage_keys` | Security |
| `/fintech-switch/mobile-money` | `fintech_switch:mobile_money` | Mobile Money |
| `/fintech-switch/settlement` | `fintech_switch:settle` | Settlement |
| `/fintech-switch/networks` | `fintech_switch:manage_networks` | Networks |

---

## Interoperability

```apg
use fintech_switch;
```

Integrates with: `fintech_auth`, `fintech_settlement`, `fintech_fraud`,
`fintech_ledger`, `fintech_notifications`.

---

## Further Reading

- `service.py` — Complete business logic (all 50+ async methods)
- `models.py` — Pydantic v2 data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views
- `WORLD_CLASS_IMPROVEMENTS.md` — 15 architectural improvement proposals
- `SPECIFICATION.md` — Full functional specification
- `cap_spec.md` — Capability specification
