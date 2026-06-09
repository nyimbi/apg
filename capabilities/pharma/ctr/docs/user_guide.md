# Clinical Trials Management

**Capability ID**: `pharma_ctr` | **Domain**: `pharma` | **Version**: `1.0.0`

## Description

Manages the complete clinical trial lifecycle from protocol development through site initiation, patient enrolment, randomisation, adverse event reporting, data management, and regulatory submissions. Enforces GCP compliance, informed consent requirements, IRB approvals, and ICH E6 expedited reporting timelines at every boundary.

## Installation

```bash
pip install apg-pharma-ctr
```

## Provides

- `trial_protocol_workflow`
- `site_selection_workflow`
- `patient_randomisation_workflow`
- `adverse_event_workflow`
- `clinical_data_management_workflow`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/pharma-ctr/dashboard` | `pharma_ctr:view` | Overview |
| `/pharma-ctr/trials` | `pharma_ctr:trials` | Trials |
| `/pharma-ctr/trials/<id>` | `pharma_ctr:trials` | Trials |
| `/pharma-ctr/protocols` | `pharma_ctr:protocols` | Protocols |
| `/pharma-ctr/sites` | `pharma_ctr:sites` | Sites |
| `/pharma-ctr/patients` | `pharma_ctr:patients` | Patients |
| `/pharma-ctr/randomisation` | `pharma_ctr:randomisation` | Patients |
| `/pharma-ctr/adverse-events` | `pharma_ctr:ae` | Safety |

## Key Service Methods

- `describe()`
- `evaluate()`
- `create_trial()`
- `register_trial()`
- `activate_trial()`
- `get_trial()`
- `list_trials()`
- `create_protocol()`
- `approve_protocol()`
- `list_protocols()`

_(See `service.py` for complete API.)_

## Interoperability

`pharma_ctr` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use pharma_ctr;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `PHARMA_CTR_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
