# Fraud Detection

**Capability ID**: `fintech_fraud` | **Domain**: `fintech` | **Version**: `1.1.0`

## Description

Fraud Detection provides real-time transaction risk scoring, multi-factor decision making (approve, step-up, hold, block, review), account takeover detection, device risk assessment, chargeback evidence management, and fraud case investigation. It acts as the cross-cutting fraud control layer across all payment-generating capabilities — every financial operation that carries a monetary amount requires a fraud signal before authorization can proceed.

## Installation

```bash
pip install apg-fintech-fraud
```

## Provides

- `fraud_signal_scoring`
- `transaction_risk_decisioning`
- `account_takeover_detection`
- `device_risk_detection`
- `chargeback_evidence_workflow`

## Requires

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/fintech-fraud/dashboard` | `fintech_fraud:view` | Overview |
| `/fintech-fraud/signals` | `fintech_fraud:score` | Signals |
| `/fintech-fraud/decisions` | `fintech_fraud:decide` | Decisions |
| `/fintech-fraud/cases` | `fintech_fraud:investigate` | Cases |
| `/fintech-fraud/chargebacks` | `fintech_fraud:chargebacks` | Evidence |
| `/fintech-fraud/devices` | `fintech_fraud:devices` | Signals |
| `/fintech-fraud/agents` | `fintech_fraud:admin` | Automation |
| `/fintech-fraud/settings` | `fintech_fraud:admin` | Administration |

## Key Service Methods

- `describe()`
- `evaluate()`
- `score_signal()`
- `record_decision()`
- `open_case()`
- `resolve_case()`
- `register_fraud_agent()`
- `validate_batch()`
- `dashboard_summary()`
- `list_signals()`

_(See `service.py` for complete API.)_

## Interoperability

`fintech_fraud` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use fintech_fraud;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `FINTECH_FRAUD_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
