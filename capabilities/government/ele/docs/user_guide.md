# Electoral and Civil Registration

**Capability ID**: `government_ele` | **Domain**: `government` | **Version**: `1.0.0`

## Description

Voter registration with biometric deduplication, polling station management, election results collation, and civil registry for births, deaths, marriages, and other vital events. Enforces integrity rules that prevent duplicate voter registration, underage registration, and result manipulation.

## Installation

```bash
pip install apg-government-ele
```

## Provides

- `voter_registration_workflow`
- `biometric_deduplication_workflow`
- `polling_station_management_workflow`
- `election_management_workflow`
- `results_collation_workflow`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/government-ele/dashboard` | `government_ele:view` | Overview |
| `/government-ele/registrations` | `government_ele:register` | Registration |
| `/government-ele/deduplication` | `government_ele:deduplicate` | Registration |
| `/government-ele/polling-stations` | `government_ele:stations` | Elections |
| `/government-ele/elections` | `government_ele:elections` | Elections |
| `/government-ele/results` | `government_ele:results` | Results |
| `/government-ele/civil-registry` | `government_ele:civil` | Civil Registry |
| `/government-ele/verifications` | `government_ele:verify` | Verification |

## Key Service Methods

- `describe()`
- `evaluate()`
- `register_voter()`
- `voter_registration()`
- `biometric_capture()`
- `polling_station_setup()`
- `voter_list_verification()`
- `ballot_management()`
- `vote_counting()`
- `result_collation()`

_(See `service.py` for complete API.)_

## Interoperability

`government_ele` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use government_ele;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `GOVERNMENT_ELE_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
