# Financial Intelligence

**Capability ID**: `intel_finint` | **Domain**: `intel` | **Version**: `1.1.0`

## Description

`intel_finint` is the APG package-backed capability for governed financial-intelligence applications. It composes authorities, financial sources, subjects, transactions, patterns, risk assessments, referrals, dissemination,

## Installation

```bash
pip install apg-intel-finint
```

## Provides

- `finint_authority_workflow`
- `finint_source_workflow`
- `finint_subject_workflow`
- `finint_transaction_workflow`
- `finint_pattern_workflow`

## Requires

- `auth`
- `audl`
- `ntfy`
- `nlpc`
- `grph`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/intel-finint/dashboard` | `intel_finint:view` | Overview |
| `/intel-finint/authorities` | `intel_finint:authorities` | Governance |
| `/intel-finint/sources` | `intel_finint:sources` | Data |
| `/intel-finint/subjects` | `intel_finint:subjects` | Data |
| `/intel-finint/transactions` | `intel_finint:transactions` | Intelligence |
| `/intel-finint/patterns` | `intel_finint:patterns` | Analysis |
| `/intel-finint/risk` | `intel_finint:risk` | Analysis |
| `/intel-finint/referrals` | `intel_finint:referrals` | Release |

## Key Service Methods

- `describe()`
- `evaluate()`
- `record_authority()`
- `register_source()`
- `record_subject()`
- `record_transaction()`
- `record_pattern()`
- `record_risk()`
- `record_referral()`
- `record_dissemination()`

_(See `service.py` for complete API.)_

## Interoperability

`intel_finint` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use intel_finint;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `INTEL_FININT_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
