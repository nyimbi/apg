# Multi-Factor Authentication

**Capability ID**: `mfau` | **Domain**: `common` | **Version**: `1.0.0`

## Description

MFAU provides adaptive multi-factor authentication for APG applications. It is a composable security capability for enrolling factors, assessing risk, issuing challenges, binding devices, governing account recovery, managing backup codes, composing first-class MFA security agents, validating Bytewax lifecycle batches, and exposing UI surfaces that generated applications can assemble into complete authentication flows.

## Installation

```bash
pip install apg-common-mfau
```

## Provides

- `multi_factor_authentication`
- `adaptive_authentication`
- `mfa_agent_composition`

## Requires

- `auth`
- `secu`
- `encr`
- `aicr`
- `conf`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/mfau/dashboard` | `mfau:view` | Overview |
| `/mfau/profiles` | `mfau:view` | Users |
| `/mfau/methods` | `mfau:manage_methods` | Methods |
| `/mfau/enrollment` | `mfau:enroll` | Methods |
| `/mfau/challenges` | `mfau:challenge` | Challenges |
| `/mfau/risk` | `mfau:challenge` | Risk |
| `/mfau/devices` | `mfau:challenge` | Risk |
| `/mfau/recovery` | `mfau:recover` | Recovery |

## Key Service Methods

- `authenticate_user()`
- `enroll_mfa_method()`
- `start_biometric_enrollment()`
- `remove_mfa_method()`
- `initiate_account_recovery()`
- `get_user_mfa_status()`
- `generate_backup_codes()`
- `verify_step_up_authentication()`
- `get_service_metrics()`
- `_authentication_successful()`

_(See `service.py` for complete API.)_

## Interoperability

`mfau` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use mfau;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `MFAU_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
