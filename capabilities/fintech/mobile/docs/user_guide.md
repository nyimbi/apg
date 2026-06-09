# Mobile Banking

**Capability ID**: `fintech_mobile` | **Domain**: `fintech` | **Version**: `1.1.0`

## Description

Mobile Banking provides the customer-facing mobile channel layer: banking program governance, customer enrollment, trusted device binding with attestation, authentication factor registration (passcode, biometric, OTP, device binding, hardware key), account and wallet linking, mobile payment initiation, bill payment, airtime purchase, service request intake, notification preference management, and mobile fraud event recording. It is the channel capability that surfaces neobanking, payments, cards, lending, BNPL, and agency services through iOS, Android, web, USSD, and SMS interfaces.

## Installation

```bash
pip install apg-fintech-mobile
```

## Provides

- `mobile_banking_program_governance`
- `mobile_customer_enrollment`
- `trusted_device_lifecycle`
- `mobile_authentication_factor_workflow`
- `mobile_account_linking`

## Requires

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `keym`

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

## Key Service Methods

- `describe()`
- `evaluate()`
- `register_program()`
- `enroll_customer()`
- `bind_device()`
- `register_auth_factor()`
- `link_account()`
- `initiate_payment()`
- `record_bill_payment()`
- `purchase_airtime()`

_(See `service.py` for complete API.)_

## Interoperability

`fintech_mobile` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use fintech_mobile;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `FINTECH_MOBILE_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
