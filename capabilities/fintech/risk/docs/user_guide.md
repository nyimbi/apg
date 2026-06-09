# FinTech Risk Management

**Capability ID**: `fintech_risk` | **Domain**: `fintech` | **Version**: `1.1.0`

## Description

FinTech Risk Management provides the enterprise risk framework for the APG platform: risk appetite registration across credit, market, liquidity, operational, fraud, compliance, model, and third-party domains; tenant-scoped risk profiles for customers, merchants, wallets, accounts, portfolios, loans, agents, and counterparties; exposure tracking with limit enforcement and human-approval-gated overrides; control assurance with effectiveness scoring; stress scenario modeling; limit breach recording; risk event management; and governance reviews.

## Installation

```bash
pip install apg-fintech-risk
```

## Provides

- `risk_appetite_workflow`
- `risk_profile_workflow`
- `risk_exposure_workflow`
- `risk_control_workflow`
- `risk_stress_testing_workflow`

## Requires

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/fintech-risk/dashboard` | `fintech_risk:view` | Overview |
| `/fintech-risk/appetite` | `fintech_risk:appetite` | Governance |
| `/fintech-risk/profiles` | `fintech_risk:profiles` | Risk |
| `/fintech-risk/exposures` | `fintech_risk:exposures` | Risk |
| `/fintech-risk/controls` | `fintech_risk:controls` | Controls |
| `/fintech-risk/stress-tests` | `fintech_risk:stress` | Analytics |
| `/fintech-risk/breaches` | `fintech_risk:breaches` | Issues |
| `/fintech-risk/events` | `fintech_risk:events` | Issues |

## Key Service Methods

- `describe()`
- `evaluate()`
- `register_appetite()`
- `create_profile()`
- `record_exposure()`
- `evaluate_control()`
- `run_stress_scenario()`
- `record_limit_breach()`
- `open_risk_event()`
- `record_review()`

_(See `service.py` for complete API.)_

## Interoperability

`fintech_risk` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use fintech_risk;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `FINTECH_RISK_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
