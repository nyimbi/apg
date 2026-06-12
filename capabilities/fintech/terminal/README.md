# Terminal Management System

## Overview

Terminal Management System provides a standalone-deployable implementation of POS terminal lifecycle management, DUKPT/TR-31 key injection, EMV compliance, geo-fenced operations, ISO 8583 message routing, real-time fraud scoring, and agency-banking transaction processing for the APG platform. Installable independently; composes with other APG capabilities via the standard contract interface.

## Capability ID

`fintech_terminal`  Version: 2.0.0

## Provides

| Service | Description |
|---------|-------------|
| `terminal_lifecycle_management` | Register, activate, suspend, relocate, and decommission terminals |
| `terminal_key_injection_workflow` | DUKPT/TR-31 key injection, rotation, and KSN lifecycle |
| `terminal_parameter_deployment` | OTA parameter push with delta compression and rollback |
| `terminal_certificate_management` | TLS client certificate issuance, pinning, and revocation |
| `terminal_health_monitoring` | Heartbeat, diagnostics, geo-fence, and velocity checks |
| `terminal_transaction_processing` | Deposits, withdrawals, transfers, bill payments, FX, ISO 8583 |
| `float_and_credit_management` | Float top-up, thresholds, agent intraday credit facility |
| `reconciliation_and_reporting` | Per-terminal and network-wide EOD batch reconciliation |
| `fraud_and_compliance` | Velocity scoring, biometric liveness, tamper detection, CBK returns |
| `observability` | OpenTelemetry tracing, Prometheus metrics, Kafka event streaming |

## Requires

| Capability | Purpose |
|------------|---------|
| `auth` | Token validation and RBAC (OPA-backed in v2.0) |
| `audl` | Immutable audit event log |
| `ntfy` | SMS/email/push notifications |
| `keym` | HSM key management integration |
| `encr` | Payload encryption helpers |

## Installation

```bash
pip install apg-fintech-terminal
```

## Quick Start

```python
from apg_fintech_terminal import get_capability_contract
from apg_fintech_terminal.service import TerminalBankingService

contract = get_capability_contract(tenant_id="my_org")
print(contract["capability"])  # fintech_terminal

svc = TerminalBankingService(tenant_id="my_org")
terminal = await svc.register_terminal("T001", {"county": "Nairobi"}, "AGT-1", "mpos", "lte")
await svc.inject_terminal_key("T001", bdk_id="BDK-42", ksn="FFFF9876543210E00000", key_type="AES256", injected_by="HSM-OPS-1")
await svc.activate_terminal("T001", activated_by="field_engineer_7")
```

## Running the Standalone Server

```bash
# In-memory (development)
apg-fintech-terminal --port 8080

# PostgreSQL persistence
apg-fintech-terminal --db-url postgresql+asyncpg://user:pass@localhost/terminal --port 8080
```

## API Routes

| Name | Path | Permission |
|------|------|------------|
| dashboard | `/fintech-terminal/dashboard` | `fintech_terminal:view` |
| terminals | `/fintech-terminal/terminals` | `fintech_terminal:manage` |
| key_management | `/fintech-terminal/keys` | `fintech_terminal:manage_keys` |
| parameters | `/fintech-terminal/parameters` | `fintech_terminal:deploy_parameters` |
| certificates | `/fintech-terminal/certificates` | `fintech_terminal:manage_certificates` |
| compliance | `/fintech-terminal/compliance` | `fintech_terminal:compliance` |
| mobile_money | `/fintech-terminal/mobile-money` | `fintech_terminal:mobile_money` |
| health | `/fintech-terminal/health` | `fintech_terminal:monitor` |
| geo_fence | `/fintech-terminal/geo` | `fintech_terminal:manage` |
| credit | `/fintech-terminal/credit` | `fintech_terminal:float` |
| fx | `/fintech-terminal/fx` | `fintech_terminal:transactions` |
| reconciliation | `/fintech-terminal/reconciliation` | `fintech_terminal:reconcile` |
| metrics | `/metrics` | `fintech_terminal:observe` |

## HTTP Endpoints

```
GET  /health               Liveness probe
GET  /contract             Full capability contract JSON
GET  /metrics              Prometheus metrics scrape endpoint
POST /evaluate             Evaluate governance rules
POST /api/v1/keys/inject   Key injection
POST /api/v1/keys/rotate   Key rotation
POST /api/v1/certs/issue   Certificate provisioning
POST /api/v1/certs/revoke  Certificate revocation
POST /api/v1/geo/check     Geo-fence validation
POST /api/v1/recon/batch   Network-wide EOD reconciliation
POST /api/v1/credit/draw   Agent credit drawdown
POST /api/v1/credit/repay  Agent credit repayment
POST /api/v1/fx            Cross-currency transaction
```

---

## World-Class Enhancements (v2.0)

1. **EMV Level-2 Kernel Integration** — Full AID-table negotiation for Kernel 2/3/6 (contact and contactless). Activation is blocked until EMV configuration is validated. Hard prerequisite for Visa/Mastercard scheme certification.

2. **DUKPT / TR-31 Key-Injection Lifecycle** — Explicit key-injection records tracking IK, BDK, KSN, and expiry. TR-31 key-block wrapping enforced before storage. KSN counter triggers re-injection on exhaustion. Required for PCI PTS compliance.

3. **Terminal CA & TLS Pinning** — Per-terminal TLS client certificates issued by a short-lived platform CA. `provision_terminal_certificate()` handles issuance, scheduling, and revocation on tamper. All terminal-to-host traffic is CA-pinned.

4. **ISO 8583 Message Routing Engine** — Proper ISO 8583 message builder/parser supporting MTIs 0100/0110, 0200/0210, 0400/0410. BIN-range routing table directs messages to the correct acquirer host with full response code, auth code, and stand-in flag.

5. **Real-Time Velocity & Fraud Scoring** — Inline sliding-window velocity check (1h and 24h) per customer, per terminal, and per agent. Rule-based fraud score 0–100. Auto-declines above threshold; emits `fraud_alert_terminal()` event. Velocity counters stored in Redis.

6. **Async Event Streaming via Kafka / CloudEvents** — Every lifecycle, transaction, and health event is published as a CloudEvent to a Kafka topic with at-least-once delivery. Downstream capabilities (`intel_alerts`, `fin_reporting`) subscribe without polling the terminal DB.

7. **EOD Automated Batch Reconciliation** — `batch_reconcile_network()` collects terminal day-totals, diffs against acquirer clearing files, and produces a machine-readable CBK ABR-01 variance report. Terminals with float variance > 0.5% raise auto-tickets.

8. **Geo-Fencing & Anomalous Location Detection** — GPS coordinates stored on every heartbeat. Movement beyond a configurable radius (default 500 m) triggers auto-suspension. `relocate_terminal()` requires dual approval (agent + supervisor).

9. **Offline-First Cryptographic Journaling** — Queued offline transactions are HMAC-SHA256 signed by the terminal's derived session key. On sync, the server verifies the HMAC chain and rejects tampered entries before processing.

10. **Dynamic Parameter & Config Push (OTA)** — `push_terminal_parameters()` delivers BIN tables, commission rates, MCCs, and CAF files using delta compression (bsdiff/xdelta). Includes one-click rollback on terminal errors. Eliminates SD-card field visits.

11. **Biometric Liveness Detection** — `customer_enrolment()` now calls an ISO 30107-3 PAD Level 1 liveness check before storing the biometric hash. Enrolments with liveness score < 0.85 are rejected to prevent photo-spoof attacks.

12. **Tiered Agent Intraday Credit Facility** — Credit lines modelled per agent based on historical transaction volume. `agent_credit_drawdown()` auto-draws when float drops below the alert threshold; `agent_credit_repayment()` reconciles at EOD. Full audit trail.

13. **Multi-Currency & FX Rate Integration** — `supported_currencies` table with real-time CBK FX rate feed. `terminal_transaction()` records `exchange_rate` and `settlement_amount_kes` at transaction time. Enables Uganda/Tanzania cross-border corridors.

14. **Prometheus Metrics & OpenTelemetry Tracing** — Every service method is instrumented with OTel spans (trace ID propagated from HTTP requests). Counters for `transactions_total`, `float_operations_total`; histogram for `transaction_latency_ms`. Native integration with `intel_dashboard`.

15. **Zero-Trust mTLS + RBAC (OPA)** — `AuthAdapter` replaced with OPA-backed RBAC policy engine. `validate_service_token()` verifies short-lived terminal-scoped JWTs before any mutating operation. All service-to-service calls use mTLS. Meets ISO 27001 Annex A.9.

---

## New Methods

### DUKPT / TR-31 Key Injection

```python
# Inject AES-256 key (raw key material never leaves HSM)
key = await svc.inject_terminal_key(
    "T001",
    bdk_id="BDK-42",
    ksn="FFFF9876543210E00000",
    key_type="AES256",
    injected_by="HSM-OPS-1",
    expiry_days=365,
)

# Rotate when KSN counter nears exhaustion
rotation = await svc.rotate_terminal_key(
    "T001",
    new_bdk_id="BDK-43",
    new_ksn="FFFF9876543210E00001",
    key_type="AES256",
    rotated_by="HSM-OPS-1",
)
```

### TLS Certificate Lifecycle

```python
cert = await svc.provision_terminal_certificate(
    "T001", csr_pem=my_csr_pem, issued_by="platform-ca", validity_days=90
)
# Revoke immediately on tamper detection
await svc.revoke_terminal_certificate(
    "T001", cert["id"], reason="tamper_detected", revoked_by="security_ops"
)
```

### Transaction Velocity & Fraud Scoring

```python
verdict = await svc.evaluate_transaction_velocity(
    "T001", customer_id="CUST-99", transaction_type="cash_withdrawal", amount=50_000
)
# verdict["recommendation"] -> "allow" | "review" | "deny"
# verdict["fraud_score"]    -> 0–100
# verdict["velocity_1h"]    -> {"count": 3, "volume": 15000}
```

### Geo-Fence Enforcement

```python
# Check current position against registered location (default 500 m radius)
check = await svc.geo_fence_check("T001", latitude=-1.286, longitude=36.817)
# check["status"] -> "within_fence" | "outside_fence"

# Relocate with dual approval
await svc.relocate_terminal(
    "T001",
    new_location={"latitude": -1.290, "longitude": 36.820, "address": "Tom Mboya St"},
    requested_by="agent_7",
    approved_by="supervisor_3",
)
```

### EOD Network Batch Reconciliation

```python
summary = await svc.batch_reconcile_network("2026-06-11", variance_threshold_pct=0.5)
# summary["flagged_terminals"] -> list of terminals with variance > 0.5%
# summary["total_credits_kes"], summary["total_debits_kes"]
```

### Agent Intraday Credit Facility

```python
drawdown = await svc.agent_credit_drawdown("AGT-1", "T001", amount=30_000)
# drawdown["credit_used"], drawdown["credit_remaining"]

repayment = await svc.agent_credit_repayment("AGT-1", amount=30_000, reference="EFT-20260611")
```

### Cross-Currency (FX) Transactions

```python
txn = await svc.foreign_currency_transaction(
    "T001",
    customer_id="CUST-44",
    amount=100,
    source_currency="USD",
    target_currency="KES",
    exchange_rate=130.5,
)
# txn["settlement_amount_kes"], txn["exchange_rate_applied"]
```

### OTA Parameter Push

```python
push = await svc.push_terminal_parameters(
    "T001",
    parameters={"bin_table_version": "2026-06", "commission_cash_dep": 0.005},
    pushed_by="ops_admin",
    version="2026-06-12",
    rollback_version="2026-05-01",
)
# push["status"] -> "deployed" | "rolled_back"
```

---

## Composability

Auto-discovered by the APG capability registry when installed.

```python
from capabilities.capability_contract_registry import load_contract_registry
registry = load_contract_registry()
contract = registry["fintech_terminal"].contract
```

## Development

```bash
# Run tests
pytest tests/ -q

# Build wheel
python -m build --wheel .

# Type check
uv run pyright

# Validate contract
python -c "from capability_contract import get_capability_contract; print('OK')"
```

## License

Proprietary — © 2025 Datacraft  
Author: Nyimbi Odero <nyimbi@gmail.com>
