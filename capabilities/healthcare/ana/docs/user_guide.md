# Clinical Analytics

**Capability ID**: `healthcare_ana` | **Domain**: `healthcare` | **Version**: `1.0.0`

## Description

Provides population health analytics, clinical outcomes measurement, readmission prediction, quality indicator tracking, and care gap identification for healthcare tenants. Supports cohort management, predictive model deployment, and structured report generation aligned with CMS Star, Joint Commission, and peer-group benchmarks.

## Installation

```bash
pip install apg-healthcare-ana
```

## Provides

- `population_health_analytics`
- `clinical_outcomes_measurement`
- `readmission_prediction`
- `quality_indicator_tracking`
- `cohort_management`

## Requires

- `auth`
- `audl`
- `mten`
- `conf`
- `ntfy`

## UI Routes

| Path | Permission | Nav Group |
|------|-----------|-----------|
| `/healthcare-ana/dashboard` | `healthcare_ana:view` | Overview |
| `/healthcare-ana/population` | `healthcare_ana:population` | Analysis |
| `/healthcare-ana/cohorts` | `healthcare_ana:cohorts` | Analysis |
| `/healthcare-ana/cohorts/<id>` | `healthcare_ana:cohorts` | Analysis |
| `/healthcare-ana/metrics` | `healthcare_ana:metrics` | Quality |
| `/healthcare-ana/predictions` | `healthcare_ana:predictions` | Predictive |
| `/healthcare-ana/benchmarks` | `healthcare_ana:benchmarks` | Quality |
| `/healthcare-ana/care-gaps` | `healthcare_ana:care_gaps` | Quality |

## Key Service Methods

- `describe()`
- `evaluate()`
- `create_cohort()`
- `get_cohort()`
- `list_cohorts()`
- `update_cohort()`
- `activate_cohort()`
- `delete_cohort()`
- `population_health_report()`
- `readmission_analysis()`

_(See `service.py` for complete API.)_

## Interoperability

`healthcare_ana` integrates with other APG capabilities through the composition engine. Reference this capability in `.apg` source files:

```apg
use healthcare_ana;
```

## Configuration

All configuration keys are tenant-scoped. Set via the `conf` capability or environment variables prefixed with `HEALTHCARE_ANA_`.

## Further Reading

- `service.py` — Business logic implementation
- `models.py` — Data models
- `api.py` — REST API endpoints
- `views.py` — Flask-AppBuilder views and Pydantic schemas
- `README.md` — Quick reference
