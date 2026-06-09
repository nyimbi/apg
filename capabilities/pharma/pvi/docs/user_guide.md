# Pharmacovigilance

**Capability ID**: `pharma_pvi` | **Domain**: `pharma` | **Version**: `1.0.0`

## Description

Manages the complete pharmacovigilance lifecycle from adverse event intake through ICSR submission, signal detection, PSUR/PBRER generation, and regulatory database reporting. Enforces ICH E2B(R3) formatting, 7-day/15-day expedited reporting timelines, MedDRA coding, duplicate detection, and benefit-risk assessment requirements.

## Installation

```bash
pip install apg-pharma-pvi
```

## Provides

- `adverse_event_collection_workflow`
- `case_processing_workflow`
- `signal_detection_workflow`
- `psur_generation_workflow`
- `regulatory_reporting_workflow`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/pharma-pvi/dashboard` | `pharma_pvi:view` | Overview |
| `/pharma-pvi/cases/intake` | `pharma_pvi:cases` | Cases |
| `/pharma-pvi/cases` | `pharma_pvi:cases` | Cases |
| `/pharma-pvi/cases/<id>` | `pharma_pvi:cases` | Cases |
| `/pharma-pvi/cases/follow-up` | `pharma_pvi:follow_up` | Cases |
| `/pharma-pvi/signals` | `pharma_pvi:signals` | Signal Detection |
| `/pharma-pvi/signals/<id>` | `pharma_pvi:signals` | Signal Detection |
| `/pharma-pvi/literature` | `pharma_pvi:literature` | Literature |

## Key Service Methods

- `describe()`
- `evaluate()`
- `create_case()`
- `process_case()`
- `close_case()`
- `mark_duplicate()`
- `get_case()`
- `list_cases()`
- `submit_icsr()`
- `list_icsr_submissions()`

_(See `service.py` for complete API.)_

## Interoperability

`pharma_pvi` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use pharma_pvi;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `PHARMA_PVI_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
